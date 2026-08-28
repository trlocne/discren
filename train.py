"""
Local / Single-machine GPU training entrypoint for DISCREN.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from data.mmhcl_dataset import MMHCLDataset
from model.discren import Discren
from seed_utils import resolve_save_dir
from training.trainer import DiscrenTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DISCREN Model")
    parser.add_argument("--config", type=str, default="configs/clothing_full.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="Clothing", help="Dataset name (Clothing, Sports)")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed override (-1 to use config)")
    parser.add_argument("--epochs", type=int, default=-1, help="Max epochs override (-1 to use config)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    hg_cfg = config.get("model", {}).get("hypergraph", {})
    train_cfg = config.get("training", {})
    llm_cfg = config.get("llm", {})

    cfg_seed = int(train_cfg.get("seed", 42))
    seed = cfg_seed if args.seed < 0 else args.seed
    set_seed(seed)
    print(f"[DISCREN] Global random seed set to {seed}")

    llm_on = bool(llm_cfg.get("enabled", False))

    dataset = MMHCLDataset(
        data_dir=os.path.join(args.data_dir, args.dataset),
        dataset_name=args.dataset,
        use_user_profile=llm_on and bool(llm_cfg.get("use_user_profile", False)),
        user_profile_feat_name=llm_cfg.get("user_profile_feat_name", "user_profile_feats.npy"),
        use_item_llm_text=llm_on and bool(llm_cfg.get("use_item_llm_text", False)),
        item_llm_text_feat_name=llm_cfg.get("item_llm_text_feat_name", "item_llm_text_feats.npy"),
        user_filter_train_only=bool(llm_cfg.get("user_filter_train_only", True)),
        cluster_k=int(hg_cfg.get("cluster_k", 256)),
        num_clusters=int(hg_cfg.get("num_clusters", 151)),
        user_knn_k=int(data_cfg.get("user_knn_k", 10)),
        item_knn_k=int(data_cfg.get("item_knn_k", 10)),
        ui_mat_device=args.device,
        i2i_mat_device=args.device,
        u2u_mat_device=args.device,
    )

    model = Discren(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embed_dim=int(config["model"]["embed_dim"]),
        ui_layers=int(config["model"].get("ui_layers", 2)),
        user_layers=int(config["model"].get("user_layers", 2)),
        item_layers=int(config["model"].get("item_layers", 2)),
        temperature=float(config["model"].get("temperature", 0.2)),
        tau_modal=float(config["model"].get("tau_modal", 0.2)),
        use_item_structural=bool(config["model"].get("use_item_structural", True)),
        use_modal_purifier=bool(config["model"].get("use_modal_purifier", True)),
        rca_iterations=int(config["model"].get("rca_iterations", 3)),
        modal_layers=int(config["model"].get("modal_layers", 1)),
        image_feat_dim=dataset.image_feats.shape[1],
        text_feat_dim=dataset.text_feats.shape[1],
        dropedge_rate=float(train_cfg.get("dropedge_rate", 0.5)),
        use_adaptive_gate=bool(config["model"].get("use_adaptive_gate", True)),
        use_user_profile=llm_on and bool(llm_cfg.get("use_user_profile", False)),
        user_profile_feat_dim=dataset.user_profile_feats.shape[1] if dataset.user_profile_feats is not None else 0,
        use_item_llm_text=llm_on and bool(llm_cfg.get("use_item_llm_text", False)),
        item_llm_text_feat_dim=dataset.item_llm_text_feats.shape[1] if dataset.item_llm_text_feats is not None else 0,
        llm_feat_scale=float(llm_cfg.get("llm_feat_scale", 1.0)),
        user_feat_scale=float(llm_cfg.get("user_feat_scale", 1.0)),
        item_feat_scale=float(llm_cfg.get("item_feat_scale", 1.0)),
    )

    save_dir = resolve_save_dir(config.get("save_dir", "checkpoints"), args.config, seed)
    os.makedirs(save_dir, exist_ok=True)

    trainer = DiscrenTrainer(
        model=model,
        lr=float(train_cfg.get("lr", 5e-4)),
        device=args.device,
        save_dir=save_dir,
        train_pairs=dataset.train_pairs,
        val_pairs=dataset.val_pairs,
        UI_mat=dataset.UI_mat,
        U2U_mat=dataset.U2U_mat,
        I2I_mat=dataset.I2I_mat,
        item_loss_ratio=float(config["loss"].get("item_loss_ratio", 0.7)),
        user_loss_ratio=float(config["loss"].get("user_loss_ratio", 0.1)),
        mmhcl_reg=float(config["loss"].get("mmhcl_reg", 1e-3)),
        lambda_modal_align=float(config["loss"].get("lambda_modal_align", 0.0)),
        lambda_modal_modal=float(config["loss"].get("lambda_modal_modal", 0.0)),
        lambda_mae=float(llm_cfg.get("lambda_mae", 0.0)) if llm_on else 0.0,
        mae_mask_ratio=float(llm_cfg.get("mae_mask_ratio", 0.3)),
        hard_neg_synth_rate=float(train_cfg.get("hard_neg_synth_rate", 0.1)),
        hard_neg_pool_size=int(train_cfg.get("hard_neg_pool_size", 32)),
        scheduler_type=train_cfg.get("scheduler_type", "cosine"),
        scheduler_t_max=int(train_cfg.get("epochs", 300)),
        scheduler_eta_min=float(train_cfg.get("scheduler_eta_min", 1e-5)),
    )

    epochs = int(train_cfg.get("epochs", 300)) if args.epochs < 0 else args.epochs
    print(f"[DISCREN] Starting training for {epochs} epochs on {args.device}...")
    trainer.train(
        num_epochs=epochs,
        patience=int(train_cfg.get("patience", 20)),
        eval_every=int(train_cfg.get("eval_every", 5)),
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
