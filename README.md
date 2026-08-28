<div align="center">

# DISCREN: An Empirical Study of Structural and LLM Semantic Signals for Sparse Multimodal Recommendation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![PyG](https://img.shields.io/badge/PyG-2.3%2B-3C2179.svg?style=flat)](https://pyg.org)
[![Modal](https://img.shields.io/badge/Accelerated%20by-Modal-00D084.svg?style=flat)](https://modal.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview

**DISCREN** (*Discriminative Contrastive Representation Learning with LLM-Augmented Semantics over a Graph–Hypergraph Backbone*) is an empirical research framework for sparse multimodal recommender systems. It investigates the load-bearing characteristics of collaborative co-occurrence, raw visual/textual item content, and large language model (LLM) semantic features.

### 🌟 Key Research Insights
- **The Modality-Induced Item–Item Pathway is the Core Driver**: Removing the item–item structural graph ($\tilde{\mathbf{A}}_{ii}$) incurs a **15.8% drop in aggregate Recall@20** and a **64% drop in cold-item Recall**, demonstrating that content-guided structural propagation is the primary retrieval engine for sparse catalogs.
- **Controlled LLM Semantic Injection**: Provides a targeted $+7.3\%$ improvement on the cold-item slice ($<5$ interactions) with zero online inference latency and minimal parameter overhead ($+2.1\%$).
- **Popularity-Adaptive Routing**: Popularity-aware gating ($\alpha_u, \alpha_i$) trades aggregate head precision to systematically boost tail-item discovery and catalog coverage.

---

## 🏛️ Model Architecture

<div align="center">
  <img src="assets/fig_baseline_r20.png" width="48%" alt="Baseline Comparison" />
  <img src="assets/fig_ablation_waterfall.png" width="48%" alt="Ablation Waterfall" />
</div>

The DISCREN architecture integrates three key structural pathways:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 DISCREN Unified Model                   │
                  └─────────────────────────────────────────────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────────┐
│  B1-B3: Collaborative GNN   │   │ M1-M4: Multimodal Encoder   │   │ L1-L5: LLM Feature Injection│
├─────────────────────────────┤   ├─────────────────────────────┤   ├─────────────────────────────┤
│ • B1: Bipartite LightGCN    │   │ • M1: Linear Projections    │   │ • L1: Freeze-Project Embeds │
│ • B2: User-User Homogeneous │   │ • M2: Reciprocal Attention  │   │ • L2: Gated Residual Fusion │
│ • B3: Item-Item Homogeneous │   │ • M3: HyperGCN Convolution  │   │ • L3: Learnable Gate Omega  │
│ • Degree-Scaled DropEdge    │   │ • M4: Behavior-Guided Gates │   │ • L4: Denoising MAE Loss    │
└─────────────────────────────┘   └─────────────────────────────┘   └─────────────────────────────┘
         │                                      │                                      │
         └──────────────────────────────────────┼──────────────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │   Popularity Gate & Fusion      │
                               │   z_u = u_cf + α_u ⊙ (s_u + l_u)│
                               │   z_i = i_cf + α_i ⊙ (s_i + l_i)│
                               └─────────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Optimization Objectives         │
                               │ L = L_BPR + L_CL + L_MAE + L_reg│
                               └─────────────────────────────────┘
```

---

## 📊 Experimental Results

### 1. Main Benchmark Results

Evaluated on held-out test splits across **Amazon Clothing, Shoes & Jewelry** and **Amazon Sports & Outdoors**.

| Method | Backbone Type | Clothing R@20 $\uparrow$ | Clothing N@20 $\uparrow$ | Sports R@20 $\uparrow$ | Sports N@20 $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MF-BPR** (UAI '09) | Matrix Factorization | 0.0191 | 0.0088 | 0.0431 | 0.0203 |
| **NGCF** (SIGIR '19) | Graph Neural Net | 0.0387 | 0.0168 | 0.0696 | 0.0319 |
| **LightGCN** (SIGIR '20) | Simplified GNN | 0.0470 | 0.0215 | 0.0781 | 0.0370 |
| **SGL** (SIGIR '21) | Self-Supervised GNN | 0.0598 | 0.0268 | 0.0779 | 0.0361 |
| **VBPR** (AAAI '16) | Visual CF | 0.0481 | 0.0205 | 0.0582 | 0.0265 |
| **MMGCN** (MM '19) | Multimodal GCN | 0.0501 | 0.0221 | 0.0639 | 0.0278 |
| **GRCN** (MM '20) | Graph Refinement | 0.0631 | 0.0276 | 0.0834 | 0.0378 |
| **LATTICE** (MM '21) | Latent Graph | 0.0710 | 0.0316 | 0.0915 | 0.0424 |
| **FREEDOM** (MM '23) | Denoised Graph | 0.0812 | 0.0359 | 0.0987 | 0.0436 |
| **MMHCL** (TKDE '24) | Hypergraph Contrastive | 0.0881 | 0.0394 | 0.1064 | 0.0501 |
| **DISCREN (w/o LLM)** | Graph + HyperGCN | 0.0938 | 0.0417 | 0.1114 | 0.0513 |
| **DISCREN (Full)** | Graph + HyperGCN + LLM | **0.0949** | **0.0424** | **0.1137** | **0.0519** |

---

### 2. Multi-Seed Robustness & Full Metric Suite (*Clothing*, 5 seeds)

| Metric | DISCREN (w/o LLM) | DISCREN (Full Model) | Relative Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Recall@10** | $0.0633 \pm 0.0005$ | **$0.0650 \pm 0.0004$** | $+2.6\%$ |
| **Recall@20** | $0.0938 \pm 0.0002$ | **$0.0949 \pm 0.0005$** | $+1.1\%$ |
| **NDCG@10** | $0.0348 \pm 0.0003$ | **$0.0357 \pm 0.0003$** | $+2.6\%$ |
| **NDCG@20** | $0.0417 \pm 0.0002$ | **$0.0424 \pm 0.0002$** | $+1.7\%$ |
| **MRR@20** | $0.0275 \pm 0.0003$ | **$0.0284 \pm 0.0003$** | $+3.3\%$ |
| **Cold Recall@20** | $0.0192 \pm 0.0009$ | **$0.0206 \pm 0.0013$** | **$+7.3\%$** |
| **Coverage@20** | $0.8114 \pm 0.0120$ | **$0.8365 \pm 0.0195$** | $+3.1\%$ |

---

### 3. Ablation Study Summary

| Component Ablation | Recall@20 | NDCG@20 | Cold Recall@20 | Coverage@20 | Finding / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Model** | **0.0941** | **0.0422** | **0.0229** | 0.8699 | Complete unified pipeline |
| w/o Item–Item Graph (B3) | 0.0793 | 0.0363 | 0.0083 | 0.7995 | **Primary driver** ($-15.8\%$ R@20, $-64\%$ Cold) |
| w/o LLM Stack | 0.0938 | 0.0418 | 0.0200 | 0.8114 | Modest aggregate change, tail improvement |
| w/o Popularity Gate ($\alpha$) | 0.0955 | 0.0428 | 0.0142 | 0.7681 | Increases head metrics at the expense of tail |
| Shuffle LLM Semantics | 0.0905 | 0.0407 | 0.0154 | 0.7937 | Unaligned semantics hurts performance |
| w/o Multimodal Branch | 0.0917 | 0.0410 | 0.0175 | 0.8021 | Semantic features stabilize representations |

---

## 📁 Repository Structure

```text
discren/
├── model/                     # Core PyTorch neural architectures
│   ├── discren.py             # Main Discren model assembly
│   └── modules/
│       ├── collaborative.py   # B1-B3: Tri-structural graph GNNs
│       ├── multimodal.py      # M1-M4: Semantic projection, RCA, HyperGCN
│       ├── hypergcn.py        # Weighted hypergraph convolution
│       ├── rca.py             # Reciprocal cross-modal attention
│       ├── llm_injection.py   # Gated LLM injection & MAE restoration
│       ├── graph_ops.py       # Degree-scaled DropEdge & popularity gate
│       └── losses.py          # BPR, InfoNCE, In-Batch CL, MAE losses
├── data/                      # Dataset loaders and graph builders
│   ├── mmhcl_dataset.py       # Multi-modal interaction dataset loader
│   └── hypergraph_builder.py  # Hypergraph incidence matrix builder
├── training/                  # Training loop and optimization
│   └── trainer.py             # DiscrenTrainer with early stopping
├── evaluation/                # Evaluation metrics
│   └── metrics.py             # Recall@K, NDCG@K, MRR, Cold-slice Recall
├── llm_augment/               # Offline LLM semantic feature extraction
├── configs/                   # Experiment YAML configurations
│   ├── clothing_full.yaml
│   ├── sports_full.yaml
│   └── ablations/             # Configs for ablation arms
├── logs/                      # Benchmark logs across seeds and runs
├── scripts/                   # Evaluation and reproduction scripts
├── train.py                   # Local single-GPU training entrypoint
├── eval.py                    # Local checkpoint evaluation script
├── main_modal.py              # Modal remote GPU training
└── eval_modal.py              # Modal remote GPU evaluation
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/trlocne/discren.git
cd discren

# Install Python requirements
pip install -r requirements.txt
```

### 2. Local Training

Train on a local CUDA device:

```bash
# Train on Clothing dataset
python train.py --config configs/clothing_full.yaml --dataset Clothing --device cuda

# Train on Sports dataset
python train.py --config configs/sports_full.yaml --dataset Sports --device cuda
```

### 3. Local Evaluation

Evaluate a saved checkpoint against the held-out test set:

```bash
python eval.py \
  --config configs/clothing_full.yaml \
  --checkpoint checkpoints/clothing_full/Clothing_best.pt \
  --dataset Clothing
```

---

## ☁️ Cloud Acceleration (Modal)

This repository includes first-class support for serverless A100 GPU acceleration via [Modal](https://modal.com):

```bash
# 1. Train on Modal A100 GPU
modal run main_modal.py --config-path configs/clothing_full.yaml --dataset-name Clothing

# 2. Evaluate checkpoint on Modal
modal run eval_modal.py --config-path configs/clothing_full.yaml --dataset-name Clothing

# 3. Export final embeddings for serving
modal run export_embeddings_modal.py --dataset-name Clothing --checkpoint-name Clothing_best.pt
```

---

## 🧪 Reproducing Multi-Seed Sweeps & Ablations

```bash
# Run 5-seed sweep
bash scripts/run_seeds.sh

# Aggregate statistics (mean +/- std and paired t-tests)
python scripts/aggregate_seeds.py --log-dir logs/seeds

# Run ablation suite
bash scripts/run_ablations.sh
```

---

## 📜 Citation

If you find this codebase or our findings useful in your research, please cite:

```bibtex
@article{discren2025,
  title     = {DISCREN: An Empirical Study of Structural and LLM Semantic Signals for Sparse Multimodal Recommendation},
  author    = {Anonymous},
  journal   = {Preprint},
  year      = {2025}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
