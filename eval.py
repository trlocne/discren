from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import yaml
from data.mmhcl_dataset import MMHCLDataset
from evaluation.metrics import cold_recall_at_k, coverage_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from model.discren import Discren

def evaluate(model: Discren, dataset: MMHCLDataset, device: str='cuda', k_list: list[int]=[10, 20]) -> dict:
    model.eval()
    with torch.no_grad():
        u_embed, i_embed, _, _ = model(ui_mat=dataset.UI_mat.to(device), i2i_mat=dataset.I2I_mat.to(device), u2u_mat=dataset.U2U_mat.to(device), image_feat=dataset.image_feats.to(device), text_feat=dataset.text_feats.to(device), r_norm=dataset.R_sym.to(device) if hasattr(dataset, 'R_sym') and dataset.R_sym is not None else None, user_profile_feat=dataset.user_profile_feats.to(device) if dataset.user_profile_feats is not None else None, item_llm_text_feat=dataset.item_llm_text_feats.to(device) if dataset.item_llm_text_feats is not None else None)
        test_user_pos = {}
        for u, i in dataset.test_pairs:
            test_user_pos.setdefault(u, []).append(i)
        train_user_pos = {}
        for u, i in dataset.train_pairs:
            train_user_pos.setdefault(u, []).append(i)
        test_users = list(test_user_pos.keys())
        num_test_users = len(test_users)
        results = {k: {'recall': 0.0, 'ndcg': 0.0, 'precision': 0.0, 'mrr': 0.0} for k in k_list}
        batch_size = 512
        for start in range(0, num_test_users, batch_size):
            end = min(start + batch_size, num_test_users)
            batch_users = test_users[start:end]
            u_idx = torch.tensor(batch_users, dtype=torch.long, device=device)
            scores = torch.matmul(u_embed[u_idx], i_embed.T)
            for idx, u in enumerate(batch_users):
                if u in train_user_pos:
                    seen = train_user_pos[u]
                    scores[idx, seen] = -1000000000.0
            gt = torch.zeros_like(scores, dtype=torch.float32)
            for idx, u in enumerate(batch_users):
                pos = test_user_pos[u]
                gt[idx, pos] = 1.0
            for k in k_list:
                results[k]['recall'] += recall_at_k(scores, gt, k=k) * len(batch_users)
                results[k]['ndcg'] += ndcg_at_k(scores, gt, k=k) * len(batch_users)
                results[k]['precision'] += precision_at_k(scores, gt, k=k) * len(batch_users)
                results[k]['mrr'] += mrr_at_k(scores, gt, k=k) * len(batch_users)
        for k in k_list:
            for m in results[k]:
                results[k][m] /= num_test_users
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate DISCREN Checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pt checkpoint file')
    parser.add_argument('--dataset', type=str, default='Clothing', help='Dataset name (Clothing, Sports)')
    parser.add_argument('--data-dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    llm_cfg = config.get('llm', {})
    llm_on = bool(llm_cfg.get('enabled', False))
    hg_cfg = config.get('model', {}).get('hypergraph', {})
    data_cfg = config.get('data', {})
    print(f'[DISCREN] Loading dataset {args.dataset}...')
    dataset = MMHCLDataset(data_dir=os.path.join(args.data_dir, args.dataset), dataset_name=args.dataset, use_user_profile=llm_on and bool(llm_cfg.get('use_user_profile', False)), user_profile_feat_name=llm_cfg.get('user_profile_feat_name', 'user_profile_feats.npy'), use_item_llm_text=llm_on and bool(llm_cfg.get('use_item_llm_text', False)), item_llm_text_feat_name=llm_cfg.get('item_llm_text_feat_name', 'item_llm_text_feats.npy'), cluster_k=int(hg_cfg.get('cluster_k', 256)), num_clusters=int(hg_cfg.get('num_clusters', 151)), user_knn_k=int(data_cfg.get('user_knn_k', 10)), item_knn_k=int(data_cfg.get('item_knn_k', 10)), ui_mat_device=args.device, i2i_mat_device=args.device, u2u_mat_device=args.device)
    model = Discren(num_users=dataset.num_users, num_items=dataset.num_items, embed_dim=int(config['model']['embed_dim']), ui_layers=int(config['model'].get('ui_layers', 2)), user_layers=int(config['model'].get('user_layers', 2)), item_layers=int(config['model'].get('item_layers', 2)), temperature=float(config['model'].get('temperature', 0.2)), tau_modal=float(config['model'].get('tau_modal', 0.2)), use_item_structural=bool(config['model'].get('use_item_structural', True)), use_modal_purifier=bool(config['model'].get('use_modal_purifier', True)), rca_iterations=int(config['model'].get('rca_iterations', 3)), modal_layers=int(config['model'].get('modal_layers', 1)), image_feat_dim=dataset.image_feats.shape[1], text_feat_dim=dataset.text_feats.shape[1], use_adaptive_gate=bool(config['model'].get('use_adaptive_gate', True)), use_user_profile=llm_on and bool(llm_cfg.get('use_user_profile', False)), user_profile_feat_dim=dataset.user_profile_feats.shape[1] if dataset.user_profile_feats is not None else 0, use_item_llm_text=llm_on and bool(llm_cfg.get('use_item_llm_text', False)), item_llm_text_feat_dim=dataset.item_llm_text_feats.shape[1] if dataset.item_llm_text_feats is not None else 0).to(args.device)
    print(f'[DISCREN] Loading weights from {args.checkpoint}...')
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    print(f'[DISCREN] Running evaluation on test set...')
    metrics = evaluate(model, dataset, device=args.device, k_list=[10, 20])
    print('\n' + '=' * 55)
    print(f'  Test Evaluation Results ({args.dataset})')
    print('=' * 55)
    for k in [10, 20]:
        print(f"  Top-{k:2d}:  Recall@{k}: {metrics[k]['recall']:.4f}  |  NDCG@{k}: {metrics[k]['ndcg']:.4f}  |  MRR@{k}: {metrics[k]['mrr']:.4f}")
    print('=' * 55 + '\n')
if __name__ == '__main__':
    main()
