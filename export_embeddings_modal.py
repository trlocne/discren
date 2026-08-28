import os
import sys
import modal
app = modal.App('discren-export-embeddings')
volume = modal.Volume.from_name('discren-checkpoints', create_if_missing=True)
dataset_volume = modal.Volume.from_name('discren-dataset', create_if_missing=True)
export_volume = modal.Volume.from_name('discren-embeddings', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.10').run_commands('apt-get update && apt-get install -y git').pip_install('torch-geometric>=2.3.0').pip_install_from_requirements('requirements.txt').add_local_dir('configs', remote_path='/root/configs').add_local_dir('data', remote_path='/root/data').add_local_dir('model', remote_path='/root/model').add_local_dir('training', remote_path='/root/training').add_local_dir('evaluation', remote_path='/root/evaluation').add_local_file('main_modal.py', remote_path='/root/main_modal.py').add_local_file('requirements.txt', remote_path='/root/requirements.txt')

@app.function(image=image, gpu='A100-80GB', volumes={'/checkpoints': volume, '/data': dataset_volume, '/export': export_volume}, timeout=1800)
def export(config_path: str='configs/clothing_full.yaml', dataset_name: str='Clothing', checkpoint_name: str=''):
    import json
    from pathlib import Path
    import numpy as np
    import torch
    import yaml
    project_root = '/root'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from data.mmhcl_dataset import MMHCLDataset
    from model.discren import Discren
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data_cfg = config.get('data', {})
    hg_cfg = config.get('model', {}).get('hypergraph', {})
    train_cfg = config.get('training', {})
    data_dir = data_cfg.get('mmhcl_dir', './data/MMHCL')
    fknn_cfg = data_cfg.get('feature_knn', {})
    llm_cfg = config.get('llm', {})
    llm_on = bool(llm_cfg.get('enabled', False))

    def _llm(key, default):
        if not llm_on:
            return default
        return llm_cfg.get(key, default)
    mod_feat_files = data_cfg.get('modality_feature_files', {'text': 'text_feat.npy'})
    use_user_profile = _llm('use_user_profile', False)
    dataset = MMHCLDataset(dataset=dataset_name, mmhcl_dir=data_dir, cache_hypergraph=data_cfg.get('cache_hypergraph', True), knn_topk=fknn_cfg.get('top_k', 20), knn_min_sim=fknn_cfg.get('min_sim', 0.2), modality_feature_files=mod_feat_files, use_user_profile=use_user_profile, user_profile_file=_llm('user_profile_file', 'user_profile_feat.npy'), use_item_llm_text=_llm('use_item_llm_text', False), item_llm_text_file=_llm('item_llm_text_file', 'text_llm_feat.npy'))
    N_u, N_i = (dataset.num_users, dataset.num_items)
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
    model = Discren(num_users=N_u, num_items=N_i, embed_dim=hg_cfg.get('embedding_dim', 64), ui_layers=hg_cfg.get('ui_layers', 2), user_layers=hg_cfg.get('user_layers', 2), item_layers=hg_cfg.get('item_layers', 2), temperature=hg_cfg.get('temperature', 0.4), tau_modal=hg_cfg.get('tau_modal', 0.2), use_item_structural=hg_cfg.get('use_item_structural', True), use_modal_purifier=hg_cfg.get('use_modal_purifier', False), rca_iterations=hg_cfg.get('rca_iterations', 3), modal_layers=hg_cfg.get('modal_layers', 1), image_feat_dim=_image_dim, text_feat_dim=_text_dim, dropedge_rate=hg_cfg.get('dropedge_rate', 0.1), use_adaptive_gate=hg_cfg.get('use_adaptive_gate', True), use_user_profile=_has_user_profile, user_profile_feat_dim=_user_profile_dim, use_item_llm_text=_has_item_llm, item_llm_text_feat_dim=_item_llm_dim, llm_feat_scale=_llm('feat_scale', 0.1), user_feat_scale=_llm('user_feat_scale', None), item_feat_scale=_llm('item_feat_scale', None)).to(device)
    UI_mat = dataset.build_UI_mat().to(device)
    U2U_mat = dataset.build_U2U_mat().to(device)
    I2I_mat = dataset.build_I2I_mat().to(device)
    model.set_ui_dropedge_probs(UI_mat)
    if getattr(model, 'use_modal_purifier', False):
        modal_kwargs = dict(image_feat=torch.from_numpy(dataset.item_image_feat).float().to(device), text_feat=torch.from_numpy(dataset.item_text_feat).float().to(device), image_adj=dataset.build_image_adj().to(device), text_adj=dataset.build_text_adj().to(device), R_norm=dataset.build_R_norm().to(device), R_row_norm=dataset.build_R_row_norm().to(device), H_I_image=dataset.H_I_image_incidence.to(device) if getattr(dataset, 'H_I_image_incidence', None) is not None else None, H_I_text=dataset.H_I_text_incidence.to(device) if getattr(dataset, 'H_I_text_incidence', None) is not None else None)
        model.set_modal_inputs(**modal_kwargs)
    if _has_user_profile:
        model.set_modal_inputs(user_profile_feat=torch.from_numpy(user_profile_feat).float().to(device))
    if _has_item_llm:
        model.set_modal_inputs(item_llm_text_feat=torch.from_numpy(item_llm_text_feat).float().to(device))
    checkpoint_dir = train_cfg.get('save_dir', '/checkpoints')
    ckpt_name = checkpoint_name or f'{dataset_name}_best.pt'
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint: {ckpt_path} (epoch={checkpoint.get('epoch', '?')})")
    model.eval()
    with torch.no_grad():
        u_ui, i_ui, _ii, _uu = model(UI_mat, I2I_mat, U2U_mat)
    z_u = u_ui.detach().cpu().numpy().astype(np.float32)
    z_i = i_ui.detach().cpu().numpy().astype(np.float32)
    print(f'z_u: {z_u.shape}  |  z_i: {z_i.shape}')
    ds_root = Path(data_dir) / dataset_name

    def _read_id_list(fname):
        mapping = {}
        fpath = ds_root / fname
        if fpath.exists():
            for line in fpath.read_text(encoding='utf-8').splitlines():
                parts = line.split('\t')
                if len(parts) == 2:
                    raw, idx = parts
                    try:
                        mapping[int(idx)] = raw
                    except ValueError:
                        pass
        return mapping
    item_raw = _read_id_list('item_list.txt')
    user_raw = _read_id_list('user_list.txt')

    def _read_profiles(fname):
        prof = {}
        fpath = ds_root / fname
        if fpath.exists():
            for line in fpath.read_text(encoding='utf-8').splitlines():
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    try:
                        prof[int(parts[0])] = parts[1].strip()
                    except ValueError:
                        pass
        return prof
    item_profiles = _read_profiles('item_profiles.txt')
    user_seen = {}
    for u, i in dataset.train_pairs:
        user_seen.setdefault(int(u), []).append(int(i))

    def _title(idx):
        p = item_profiles.get(idx)
        if p:
            return p.split('.')[0].strip()[:120]
        return item_raw.get(idx, f'item_{idx}')
    meta = {'dataset': dataset_name, 'num_users': int(N_u), 'num_items': int(N_i), 'embed_dim': int(z_i.shape[1]), 'item_raw_id': {str(k): v for k, v in item_raw.items()}, 'user_raw_id': {str(k): v for k, v in user_raw.items()}, 'item_title': {str(i): _title(i) for i in range(N_i)}, 'item_profile': {str(k): v for k, v in item_profiles.items()}, 'user_seen': {str(k): v for k, v in user_seen.items()}}
    out_dir = Path('/export') / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'z_u.npy', z_u)
    np.save(out_dir / 'z_i.npy', z_i)
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    export_volume.commit()
    print(f"Exported to Modal volume 'discren-embeddings' at {dataset_name}/")
    print('Download locally with:')
    print(f'    modal volume get discren-embeddings {dataset_name} serve/artifacts/')
    return {'dataset': dataset_name, 'z_u': z_u.shape, 'z_i': z_i.shape}

@app.local_entrypoint()
def main(config_path: str='configs/clothing_full.yaml', dataset_name: str='Clothing', checkpoint_name: str=''):
    export.remote(config_path=config_path, dataset_name=dataset_name, checkpoint_name=checkpoint_name)
