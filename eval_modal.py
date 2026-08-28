"""
Discren test-set evaluation on Modal.

Loads a trained checkpoint, rebuilds the exact graphs/model used at training
time, and scores the held-out test split with train ∪ val items masked out
of the ranking. Reports Recall/Precision/NDCG/MRR/Coverage/ColdRecall for
several K.

Usage:
    modal run eval_modal.py
    modal run eval_modal.py --dataset-name Clothing --checkpoint-name Clothing_best.pt
    modal run eval_modal.py --k-values 1,5,10,20,50
"""

import os
import sys
from pathlib import Path

import modal

app = modal.App("discren-eval")

volume = modal.Volume.from_name("discren-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("discren-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands("apt-get update && apt-get install -y git")
    .pip_install("torch-geometric>=2.3.0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("tabulate")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("training", remote_path="/root/training")
    .add_local_dir("evaluation", remote_path="/root/evaluation")
    # See the note in main_modal.py: needed so the eval log records which LLM
    # artifact the reported metrics were actually produced from.
    .add_local_dir("llm_augment", remote_path="/root/llm_augment")
    .add_local_file("main_modal.py", remote_path="/root/main_modal.py")
    .add_local_file("seed_utils.py", remote_path="/root/seed_utils.py")
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume, "/data": dataset_volume},
    timeout=1800,
)
def evaluate(
    config_path: str = "configs/clothing_full.yaml",
    dataset_name: str = "Clothing",
    checkpoint_name: str = "",
    k_values: str = "1,5,10,20",
    cold_threshold: int = 5,
    seed: int = -1,
):
    import json

    import numpy as np
    import torch
    import yaml
    from tabulate import tabulate

    project_root = "/root"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from data.mmhcl_dataset import MMHCLDataset
    from model.discren import Discren
    from seed_utils import resolve_save_dir
    from evaluation.metrics import (
        recall_at_k, ndcg_at_k, precision_at_k, mrr_at_k,
        coverage_at_k, cold_recall_at_k,
    )

    ks = [int(k) for k in k_values.split(",") if k.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    hg_cfg = config.get("model", {}).get("hypergraph", {})
    train_cfg = config.get("training", {})

    data_dir = data_cfg.get("mmhcl_dir", "./data/MMHCL")
    fknn_cfg = data_cfg.get("feature_knn", {})

    # Centralised LLM control — MUST mirror main_modal.py exactly so the model
    # built here matches the trained checkpoint (same branches → same weights).
    llm_cfg = config.get("llm", {})
    llm_on = bool(llm_cfg.get("enabled", False))

    def _llm(key, default):
        if not llm_on:
            return default
        return llm_cfg.get(key, default)

    mod_feat_files = data_cfg.get("modality_feature_files", {"text": "text_feat.npy"})
    use_user_profile = _llm("use_user_profile", False)

    dataset = MMHCLDataset(
        dataset=dataset_name,
        mmhcl_dir=data_dir,
        cache_hypergraph=data_cfg.get("cache_hypergraph", True),
        knn_topk=fknn_cfg.get("top_k", 20),
        knn_min_sim=fknn_cfg.get("min_sim", 0.20),
        modality_feature_files=mod_feat_files,
        use_user_profile=use_user_profile,
        user_profile_file=_llm("user_profile_file", "user_profile_feat.npy"),
        use_item_llm_text=_llm("use_item_llm_text", False),
        item_llm_text_file=_llm("item_llm_text_file", "text_llm_feat.npy"),
    )
    N_u, N_i = dataset.num_users, dataset.num_items

    text_feat = getattr(dataset, "item_text_feat", None)
    image_feat = getattr(dataset, "item_image_feat", None)
    _text_dim = text_feat.shape[1] if text_feat is not None else 1024
    _image_dim = image_feat.shape[1] if image_feat is not None else 4096

    user_profile_feat = getattr(dataset, "user_profile_feat", None)
    _has_user_profile = use_user_profile and user_profile_feat is not None
    _user_profile_dim = user_profile_feat.shape[1] if user_profile_feat is not None else 1024

    use_item_llm_text = _llm("use_item_llm_text", False)
    item_llm_text_feat = getattr(dataset, "item_llm_text_feat", None)
    _has_item_llm = use_item_llm_text and item_llm_text_feat is not None
    _item_llm_dim = item_llm_text_feat.shape[1] if item_llm_text_feat is not None else 1024

    model = Discren(
        num_users=N_u,
        num_items=N_i,
        embed_dim=hg_cfg.get("embedding_dim", 64),
        ui_layers=hg_cfg.get("ui_layers", 2),
        user_layers=hg_cfg.get("user_layers", 2),
        item_layers=hg_cfg.get("item_layers", 2),
        temperature=hg_cfg.get("temperature", 0.4),
        tau_modal=hg_cfg.get("tau_modal", 0.2),
        use_item_structural=hg_cfg.get("use_item_structural", True),
        use_modal_purifier=hg_cfg.get("use_modal_purifier", False),
        rca_iterations=hg_cfg.get("rca_iterations", 3),
        modal_layers=hg_cfg.get("modal_layers", 1),
        image_feat_dim=_image_dim,
        text_feat_dim=_text_dim,
        dropedge_rate=hg_cfg.get("dropedge_rate", 0.1),
        use_adaptive_gate=hg_cfg.get("use_adaptive_gate", True),
        use_user_profile=_has_user_profile,
        user_profile_feat_dim=_user_profile_dim,
        use_item_llm_text=_has_item_llm,
        item_llm_text_feat_dim=_item_llm_dim,
        llm_feat_scale=_llm("feat_scale", 0.1),
        user_feat_scale=_llm("user_feat_scale", None),
        item_feat_scale=_llm("item_feat_scale", None),
    ).to(device)

    UI_mat = dataset.build_UI_mat().to(device)
    U2U_mat = dataset.build_U2U_mat().to(device)
    I2I_mat = dataset.build_I2I_mat().to(device)
    model.set_ui_dropedge_probs(UI_mat)

    if getattr(model, "use_modal_purifier", False):
        modal_kwargs = dict(
            image_feat=torch.from_numpy(dataset.item_image_feat).float().to(device),
            text_feat=torch.from_numpy(dataset.item_text_feat).float().to(device),
            image_adj=dataset.build_image_adj().to(device),
            text_adj=dataset.build_text_adj().to(device),
            R_norm=dataset.build_R_norm().to(device),
            R_row_norm=dataset.build_R_row_norm().to(device),
            H_I_image=dataset.H_I_image_incidence.to(device)
                if getattr(dataset, "H_I_image_incidence", None) is not None else None,
            H_I_text=dataset.H_I_text_incidence.to(device)
                if getattr(dataset, "H_I_text_incidence", None) is not None else None,
        )
        model.set_modal_inputs(**modal_kwargs)

    if _has_user_profile:
        model.set_modal_inputs(
            user_profile_feat=torch.from_numpy(user_profile_feat).float().to(device),
        )

    if _has_item_llm:
        model.set_modal_inputs(
            item_llm_text_feat=torch.from_numpy(item_llm_text_feat).float().to(device),
        )

    # Must mirror main_modal.py: a seeded run wrote to <save_dir>/seed<N>.
    checkpoint_dir = resolve_save_dir(train_cfg.get("save_dir", "/checkpoints"), seed)
    ckpt_name = checkpoint_name or f"{dataset_name}_best.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint: {ckpt_path} (epoch={checkpoint.get('epoch', '?')})")
    print(f"Users: {N_u}  |  Items: {N_i}  |  Train pairs: {len(dataset.train_pairs)}")

    model.eval()
    with torch.no_grad():
        u_ui, i_ui, _ii, _uu = model(UI_mat, I2I_mat, U2U_mat)
    z_u, z_i = u_ui, i_ui

    # Ground truth + seen-item mask (train ∪ val), scored on the test split.
    test_pairs = dataset.get_test_pairs()
    val_pairs = dataset.get_val_pairs()

    user_test_items = {}
    for u, i in test_pairs:
        user_test_items.setdefault(u, set()).add(i)

    user_seen_items = {}
    for u, i in dataset.train_pairs:
        user_seen_items.setdefault(u, set()).add(i)
    for u, i in val_pairs:
        user_seen_items.setdefault(u, set()).add(i)

    test_users = sorted(u for u in user_test_items if u < N_u)
    print(f"Test users: {len(test_users)}  |  Test pairs: {len(test_pairs)}")

    train_item_freq = np.zeros(N_i, dtype=np.int64)
    for _, i in dataset.train_pairs:
        train_item_freq[i] += 1
    interaction_counts = torch.from_numpy(train_item_freq).float().to(device)

    with torch.no_grad():
        u_idx = torch.tensor(test_users, dtype=torch.long, device=device)
        scores = z_u[u_idx] @ z_i.T  # [U_test, N_i]

        gt = torch.zeros(len(test_users), N_i, device=device)
        for row, u in enumerate(test_users):
            seen = user_seen_items.get(u)
            if seen:
                sidx = torch.tensor(list(seen), dtype=torch.long, device=device)
                scores[row, sidx] = float("-inf")
            gt[row, list(user_test_items[u])] = 1.0

        metrics = {}
        for k in ks:
            if k > N_i:
                continue
            metrics[f"recall@{k}"] = recall_at_k(scores, gt, k)
            metrics[f"precision@{k}"] = precision_at_k(scores, gt, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(scores, gt, k)
            metrics[f"mrr@{k}"] = mrr_at_k(scores, gt, k)
            metrics[f"coverage@{k}"] = coverage_at_k(scores, k)
            metrics[f"cold_recall@{k}"] = cold_recall_at_k(
                scores, gt, interaction_counts, k, cold_threshold
            )

    rows = [
        [k,
         float(metrics.get(f'recall@{k}', 0)),
         float(metrics.get(f'precision@{k}', 0)),
         float(metrics.get(f'ndcg@{k}', 0)),
         float(metrics.get(f'mrr@{k}', 0)),
         float(metrics.get(f'coverage@{k}', 0)),
         float(metrics.get(f'cold_recall@{k}', 0))]
        for k in ks if k <= N_i
    ]
    print(tabulate(
        rows,
        headers=["K", "Recall", "Precision", "NDCG", "MRR", "Coverage", "ColdRecall"],
        tablefmt="grid",
        floatfmt=".4f",
    ))

    # Machine-readable line so aggregate_seeds.py can reduce a seed sweep
    # without re-parsing the table.
    print("[metrics-json] " + json.dumps({
        "config": config_path,
        "seed": int(seed),
        "checkpoint": ckpt_path,
        "epoch": checkpoint.get("epoch", None),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }))

    return {"dataset": dataset_name, "checkpoint": ckpt_path, "metrics": metrics}


@app.local_entrypoint()
def main(
    config_path: str = "configs/clothing_full.yaml",
    dataset_name: str = "Clothing",
    checkpoint_name: str = "",
    k_values: str = "1,5,10,20",
    cold_threshold: int = 5,
    seed: int = -1,
):
    results = evaluate.remote(
        config_path=config_path,
        dataset_name=dataset_name,
        checkpoint_name=checkpoint_name,
        k_values=k_values,
        cold_threshold=cold_threshold,
        seed=seed,
    )
    print("\nDone.")
    return results
