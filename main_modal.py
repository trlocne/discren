import os
import shutil
import sys
from pathlib import Path

import modal

app = modal.App("discren-training")

volume = modal.Volume.from_name("discren-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("discren-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands("apt-get update && apt-get install -y git")
    .pip_install("torch-geometric>=2.3.0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("training", remote_path="/root/training")
    .add_local_dir("evaluation", remote_path="/root/evaluation")
    # Shipped for llm_augment.provenance only: MMHCLDataset prints the sidecar of
    # each LLM feature it loads, which is what makes a dry-run stub or a
    # non-train-filtered artifact visible in the training log instead of passing
    # silently for a real one. Without this the log says "provenance
    # unavailable" and that check is lost exactly where it matters most.
    .add_local_dir("llm_augment", remote_path="/root/llm_augment")
    .add_local_file("main_modal.py", remote_path="/root/main_modal.py")
    .add_local_file("seed_utils.py", remote_path="/root/seed_utils.py")
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt")
)

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume, "/data": dataset_volume},
    timeout=3600 * 6,  # 6 hours max
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)
def train(
    config_path: str = "configs/clothing_full.yaml",
    dataset_name: str = "Clothing",
    seed: int = -1,
):
    import numpy as np
    import torch
    import yaml

    project_root = "/root"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data.mmhcl_dataset import MMHCLDataset
    from model.discren import Discren
    from seed_utils import resolve_save_dir
    from training.trainer import DiscrenTrainer

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config.get('data', {})
    hg_cfg = config.get('model', {}).get('hypergraph', {})
    train_cfg = config.get('training', {})

    def set_seed(seed: int):
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _maybe_shuffle_feature_rows(feature, enabled: bool, seed_value: int):
        if feature is None or not enabled:
            return feature
        rng = np.random.default_rng(seed_value)
        perm = rng.permutation(feature.shape[0])
        return feature[perm]

    # A single seed cannot separate a real effect from run-to-run noise, so the
    # seed is overridable from the CLI and every seed writes to its OWN
    # checkpoint dir. run_seeds.sh sweeps it; aggregate_seeds.py reduces the
    # sweep to mean +/- std and a paired test.
    cfg_seed = int(train_cfg.get('seed', 42))
    seed_override = int(seed)
    seed = cfg_seed if seed_override < 0 else seed_override
    set_seed(seed)
    print(f"[Seed] Global random seed set to {seed} (deterministic mode)"
          + (f" [overridden, config had {cfg_seed}]" if seed_override >= 0 else ""))

    llm_cfg = config.get('llm', {})
    llm_on = bool(llm_cfg.get('enabled', False))

    def _llm(key, default):
        """Read an LLM sub-setting, gated by the master switch."""
        if not llm_on:
            return default          # master off → neutral/disabled value
        return llm_cfg.get(key, default)

    data_dir = data_cfg.get('mmhcl_dir', './data/MMHCL')
    cache_dir = Path(data_dir) / dataset_name / 'cache'

    if data_cfg.get('clear_cache', False) and cache_dir.exists():
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Cache] Cache cleared, will rebuild as sparse.")

    fknn_cfg = data_cfg.get('feature_knn', {})
    mod_feat_files = data_cfg.get('modality_feature_files', {'text': 'text_feat.npy'})
    use_user_profile = _llm('use_user_profile', False)
    dataset = MMHCLDataset(
        dataset=dataset_name,
        mmhcl_dir=data_dir,
        cache_hypergraph=data_cfg.get('cache_hypergraph', True),
        knn_topk=fknn_cfg.get('top_k', 20),
        knn_min_sim=fknn_cfg.get('min_sim', 0.20),
        modality_feature_files=mod_feat_files,
        use_user_profile=use_user_profile,
        user_profile_file=_llm('user_profile_file', 'user_profile_feat.npy'),
        use_item_llm_text=_llm('use_item_llm_text', False),
        item_llm_text_file=_llm('item_llm_text_file', 'text_llm_feat.npy'),
    )

    text_feat = getattr(dataset, 'item_text_feat', None)
    image_feat = getattr(dataset, 'item_image_feat', None)
    _text_dim = text_feat.shape[1] if text_feat is not None else 1024
    _image_dim = image_feat.shape[1] if image_feat is not None else 4096

    user_profile_feat = getattr(dataset, 'user_profile_feat', None)
    _has_user_profile = use_user_profile and user_profile_feat is not None
    _user_profile_dim = user_profile_feat.shape[1] if user_profile_feat is not None else 1024

    use_item_llm_text = _llm('use_item_llm_text', False)
    item_llm_text_feat = getattr(dataset, 'item_llm_text_feat', None)
    _has_item_llm = use_item_llm_text and item_llm_text_feat is not None
    _item_llm_dim = item_llm_text_feat.shape[1] if item_llm_text_feat is not None else 1024

    if _has_user_profile:
        shuffle_user = bool(llm_cfg.get('shuffle_user_profile', False))
        if shuffle_user:
            print(f"[LLM] Shuffling user-profile rows with seed {seed}")
            user_profile_feat = _maybe_shuffle_feature_rows(user_profile_feat, True, seed)

    if _has_item_llm:
        shuffle_item = bool(llm_cfg.get('shuffle_item_llm_text', False))
        if shuffle_item:
            print(f"[LLM] Shuffling item-LLM rows with seed {seed}")
            item_llm_text_feat = _maybe_shuffle_feature_rows(item_llm_text_feat, True, seed)

    model = Discren(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embed_dim=hg_cfg.get('embedding_dim', 64),
        ui_layers=hg_cfg.get('ui_layers', 2),
        user_layers=hg_cfg.get('user_layers', 2),
        item_layers=hg_cfg.get('item_layers', 2),
        temperature=hg_cfg.get('temperature', 0.4),
        tau_modal=hg_cfg.get('tau_modal', 0.2),          # sharper τ for modal CL
        use_item_structural=hg_cfg.get('use_item_structural', True),
        use_modal_purifier=hg_cfg.get('use_modal_purifier', False),
        rca_iterations=hg_cfg.get('rca_iterations', 3),
        modal_layers=hg_cfg.get('modal_layers', 1),
        image_feat_dim=_image_dim,
        text_feat_dim=_text_dim,
        dropedge_rate=hg_cfg.get('dropedge_rate', 0.1),
        use_adaptive_gate=hg_cfg.get('use_adaptive_gate', True),
        use_user_profile=_has_user_profile,
        user_profile_feat_dim=_user_profile_dim,
        use_item_llm_text=_has_item_llm,
        item_llm_text_feat_dim=_item_llm_dim,
        llm_feat_scale=_llm('feat_scale', 0.1),
        user_feat_scale=_llm('user_feat_scale', None),
        item_feat_scale=_llm('item_feat_scale', None),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    UI_mat = dataset.build_UI_mat().to(device)
    U2U_mat = dataset.build_U2U_mat().to(device)
    I2I_mat = dataset.build_I2I_mat().to(device)
    model.set_ui_dropedge_probs(UI_mat)

    if getattr(model, 'use_modal_purifier', False):
        modal_kwargs = dict(
            image_feat=torch.from_numpy(dataset.item_image_feat).float().to(device),
            text_feat=torch.from_numpy(dataset.item_text_feat).float().to(device),
            image_adj=dataset.build_image_adj().to(device),
            text_adj=dataset.build_text_adj().to(device),
            R_norm=dataset.build_R_norm().to(device),
            R_row_norm=dataset.build_R_row_norm().to(device),
            H_I_image=dataset.H_I_image_incidence.to(device)
                if getattr(dataset, 'H_I_image_incidence', None) is not None else None,
            H_I_text=dataset.H_I_text_incidence.to(device)
                if getattr(dataset, 'H_I_text_incidence', None) is not None else None,
        )
        model.set_modal_inputs(**modal_kwargs)

    if _has_user_profile:
        model.set_modal_inputs(
            user_profile_feat=torch.from_numpy(user_profile_feat).float().to(device),
        )
        print(f"[Model] Injected LLM user profile features {user_profile_feat.shape}")

    if _has_item_llm:
        model.set_modal_inputs(
            item_llm_text_feat=torch.from_numpy(item_llm_text_feat).float().to(device),
        )
        print(f"[Model] Injected LLM item text features {item_llm_text_feat.shape}")


    lr = train_cfg.get('lr', 0.0001)
    print(f"\n[Step 3] Starting training")
    val_pairs = dataset.get_val_pairs()
    checkpoint_dir = resolve_save_dir(train_cfg.get('save_dir', '/checkpoints'), seed_override)
    if seed_override >= 0:
        print(f"[Seed] checkpoints -> {checkpoint_dir}")

    trainer = DiscrenTrainer(
        model=model,
        device=device,
        lr=lr,
        save_dir=checkpoint_dir,
        train_pairs=dataset.train_pairs,
        val_pairs=val_pairs,
        UI_mat=UI_mat,
        U2U_mat=U2U_mat,
        I2I_mat=I2I_mat,
        item_loss_ratio=train_cfg.get('item_loss_ratio', 0.7),
        user_loss_ratio=train_cfg.get('user_loss_ratio', 0.1),
        mmhcl_reg=train_cfg.get('emb_reg', 1e-3),
        lambda_modal_align=train_cfg.get('lambda_modal_align', 0.0),
        lambda_modal_modal=train_cfg.get('lambda_modal_modal', 0.0),
        lambda_mae=_llm('lambda_mae', 0.0),
        mae_mask_ratio=_llm('mae_mask_ratio', 0.3),
        hard_neg_synth_rate=train_cfg.get('hard_neg_synth_rate', 0.1),
        hard_neg_pool_size=train_cfg.get('hard_neg_pool_size', 32),
        scheduler_type=train_cfg.get('scheduler_type', 'cosine'),
        scheduler_t_max=train_cfg.get('scheduler_t_max', 300),
        scheduler_eta_min=train_cfg.get('scheduler_eta_min', 1e-5),
    )

    start_epoch = 0
    resume = train_cfg.get('resume', True)

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{dataset_name}_epoch")] if os.path.exists(checkpoint_dir) else []
    if checkpoint_files and resume:
        checkpoint_files.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        _resume_best = checkpoint.get('best_recall20', None)
        _resume_patience = checkpoint.get('patience_counter', None)
        _resume_best_epoch = checkpoint.get('best_epoch', None)
    elif checkpoint_files and not resume:
        print(f"\n[Resume] resume=false, ignoring {len(checkpoint_files)} checkpoint(s). Training from scratch.")
    else:
        print(f"\n[Resume] No checkpoint found. Training from scratch.")

    epochs = train_cfg.get('epochs', 100)
    save_every = train_cfg.get('save_every', 10)
    evaluate_every = train_cfg.get('evaluate_every', 5)
    early_stopping_patience = train_cfg.get('early_stopping_patience', 10)
    early_stopping_min_delta = train_cfg.get('early_stopping_min_delta', 0.001)

    best_recall20 = float('-inf')
    patience_counter = 0
    best_epoch = 0
    # Apply restored early-stopping state if resuming.
    if 'checkpoint' in dir() and resume and checkpoint_files:
        if _resume_best is not None:
            best_recall20 = float(_resume_best)
        if _resume_patience is not None:
            patience_counter = int(_resume_patience)
        if _resume_best_epoch is not None:
            best_epoch = int(_resume_best_epoch)
        print(f"[Resume] restored best_recall20={best_recall20:.4f} "
              f"(epoch {best_epoch}), patience={patience_counter}")

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        train_metrics = trainer.train_epoch(epoch=epoch + 1)

        print(f"Train loss: {train_metrics['train_loss']:.4f}")
        for k, v in train_metrics.get('components', {}).items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

        # Validation
        val_metrics = {}
        if val_pairs and (epoch + 1) % evaluate_every == 0:
            val_metrics = trainer.validate()
            print(f"Val metrics: {val_metrics}")

            current_recall20 = val_metrics.get('recall@20', float('-inf'))
            improved_at_all = current_recall20 > best_recall20
            improved_meaningfully = current_recall20 > best_recall20 + early_stopping_min_delta

            if improved_at_all:
                best_recall20 = current_recall20
                best_epoch = epoch + 1
                best_ckpt_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'recall@20': current_recall20,
                    'best_recall20': best_recall20,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                }, best_ckpt_path)
                volume.commit()
                print(f"[Best] Recall@20={current_recall20:.4f} at epoch {epoch+1} — saved best checkpoint")

            if improved_meaningfully:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[EarlyStopping] No improvement for {patience_counter}/{early_stopping_patience} evals (best={best_recall20:.4f} at epoch {best_epoch})")
                if patience_counter >= early_stopping_patience:
                    print(f"[EarlyStopping] Stopping at epoch {epoch+1}. Best Recall@20={best_recall20:.4f} at epoch {best_epoch}")
                    break

        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_epoch{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_recall20': best_recall20,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
            }, checkpoint_path)
            volume.commit() 
            print(f"Checkpoint saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    return {
        'train_loss': train_metrics['train_loss'],
        'final_epoch': epochs,
    }


@app.local_entrypoint()
def main(
    config_path: str = "configs/clothing_full.yaml",
    dataset_name: str = "Clothing",
    seed: int = -1,
):
    result = train.remote(config_path=config_path, dataset_name=dataset_name, seed=seed)
    if result and result.get('train_loss'):
        print(f"\nFinal train loss: {result['train_loss']:.4f}")
    print("Done!")


