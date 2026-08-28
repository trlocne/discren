"""
Hypergraph builders for the dual-view recsys.

Both H_U and H_I are constructed PURELY from the interaction matrix as
binary (0/1) hypergraphs. PPR / CLIP / community signals are NOT mixed
into graph topology — they live in separate loss terms (L_PPR for
distillation, etc.) to avoid redundant smoothing.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

class HypergraphBuilder:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.incidence_matrix = None

    def build_incidence_matrix(self) -> torch.Tensor:
        raise NotImplementedError

class U2UHypergraphBuilder(HypergraphBuilder):
    """
    U2U hypergraph: for each item i, the column H_U[:, i] forms one hyperedge
    containing all users who interacted with item i.

    Binary (0/1) only.

    Shape: [N_u, N_i] (one hyperedge per item).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        interaction_matrix: np.ndarray,
    ):
        super().__init__(n_users)
        self.n_users = n_users
        self.n_items = n_items
        self.R = np.asarray(interaction_matrix, dtype=np.float32)

    def build_incidence_matrix(self) -> torch.Tensor:
        R = np.asarray(self.R, dtype=np.float32)
        H = torch.from_numpy((R > 0).astype(np.float32)).to_sparse_coo().coalesce()
        self.incidence_matrix = H
        return H

class I2IHypergraphBuilder(HypergraphBuilder):
    """
    I2I hypergraph: for each user u, the column H_I[:, u] forms one hyperedge
    containing all items u interacted with.

    Binary (0/1) only.

    Shape: [N_i, N_u] (one hyperedge per user). Mirrors H_U structurally.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        interaction_matrix: np.ndarray,
    ):
        super().__init__(n_items)
        self.n_users = n_users
        self.n_items = n_items
        self.R = np.asarray(interaction_matrix, dtype=np.float32)

    def build_incidence_matrix(self) -> torch.Tensor:
        R_T = self.R.T
        H = torch.from_numpy((R_T > 0).astype(np.float32)).to_sparse_coo().coalesce()
        self.incidence_matrix = H
        return H

def build_modality_knn_hypergraph(
    modality_features: np.ndarray,
    top_k: int = 10,
    include_self: bool = False,
    min_sim_threshold: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build an item-item hypergraph from a single modality's cosine top-k similarity.

    Produces a separate [N_i, N_i] hypergraph from one modality (e.g. text-only or
    image-only features) that can be fed into its own HyperGCN branch.

    Args:
        modality_features: [N_i, D_mod] dense feature matrix for one modality.
        top_k: neighbors per item (excluding self if include_self=False).
        include_self: keep self-loop in top-k.
        min_sim_threshold: drop neighbors with cosine similarity below this value.

    Returns:
        H_mod: [N_i, N_i] float32 incidence matrix.
    """
    feats = torch.from_numpy(np.asarray(modality_features, dtype=np.float32))
    feats = torch.nn.functional.normalize(feats, dim=-1, eps=eps)
    N_i = feats.shape[0]

    sim = feats @ feats.T
    if not include_self:
        sim.fill_diagonal_(-float('inf'))

    k = min(top_k, N_i - (0 if include_self else 1))
    topk_val, topk_idx = sim.topk(k, dim=-1)

    if min_sim_threshold > 0.0:
        above = topk_val >= min_sim_threshold
        above[:, 0] = True
        topk_val = topk_val * above.float()
        topk_idx = torch.where(above, topk_idx, topk_idx[:, :1].expand_as(topk_idx))

    rows = torch.arange(N_i, dtype=torch.long).unsqueeze(-1).expand(-1, k)

    weights = topk_val.clamp(min=0.0)
    H = torch.sparse_coo_tensor(
        torch.stack([rows.flatten(), topk_idx.flatten()]),
        weights.flatten(),
        (N_i, N_i)
    ).coalesce()
    return H

def build_knn_cluster_incidence(
    features: np.ndarray,
    top_k: int = 10,
    n_clusters: Optional[int] = None,
    min_sim: float = 0.0,
    weighted: bool = True,
    seed: int = 42,
) -> torch.Tensor:
    """Build a TRUE incidence matrix H [N_items, E_clusters] for hypergraph convolution.

    Unlike build_modality_knn_hypergraph which returns a square [N×N] adjacency,
    this function returns a non-square [N, E] incidence matrix where:
      - N = number of items
      - E = number of hyperedges (clusters, E << N)
      - H[i, e] > 0 if item i belongs to hyperedge e

    Strategy: K-Means clustering on L2-normalized features.
      - Each cluster centroid defines 1 hyperedge
      - Items within top_k nearest centroids are connected to that hyperedge
      - H[i, e] = cosine_similarity(item_i, centroid_e) if sim > min_sim else 0
      - Default n_clusters = sqrt(N) — standard rule of thumb

    This is the correct input for WeightedHypergraphConv, which implements:
      e[e] = Σ_v H[v,e] · x[v]   (node→edge)
      x'[v] = Σ_e H[v,e] · e[e]  (edge→node)

    Args:
        features: [N, feat_dim] numpy array of raw item features
        top_k: each item connects to its top_k nearest cluster centroids
        n_clusters: number of hyperedges (clusters). Default = int(sqrt(N))
        min_sim: minimum cosine similarity to form a connection
        weighted: if True, H[i,e] = cosine sim; if False, binary {0,1}
        seed: random seed for K-Means initialization

    Returns:
        H: sparse COO tensor [N, E], dtype=float32
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        raise ImportError("scikit-learn required for build_knn_cluster_incidence. pip install scikit-learn")

    N, feat_dim = features.shape
    if n_clusters is None:
        n_clusters = max(int(N ** 0.5), 64)
        n_clusters = min(n_clusters, N // 2)

    print(f"[build_knn_cluster_incidence] N={N}, feat_dim={feat_dim}, "
          f"n_clusters={n_clusters}, top_k={top_k}")

    feats = torch.from_numpy(np.asarray(features, dtype=np.float32))
    feats_norm = F.normalize(feats, p=2, dim=1)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(4096, N),
        n_init=3,
        max_iter=100,
    )
    kmeans.fit(feats_norm.numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))
    centroids_norm = F.normalize(centroids, p=2, dim=1)

    CHUNK = 2048
    rows, cols, vals = [], [], []

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        sim = feats_norm[start:end] @ centroids_norm.T

        topk_sim, topk_idx = sim.topk(min(top_k, n_clusters), dim=1)

        mask = topk_sim > min_sim
        for local_i, (sims_row, idxs_row, valid) in enumerate(
            zip(topk_sim, topk_idx, mask)
        ):
            global_i = start + local_i
            valid_idxs = idxs_row[valid]
            valid_sims = sims_row[valid]
            if valid_idxs.numel() == 0:

                best = sim[local_i].argmax()
                valid_idxs = best.unsqueeze(0)
                valid_sims = sim[local_i, best].unsqueeze(0)
            rows.extend([global_i] * valid_idxs.numel())
            cols.extend(valid_idxs.tolist())
            if weighted:
                vals.extend(valid_sims.tolist())
            else:
                vals.extend([1.0] * valid_idxs.numel())

    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    vals_t = torch.tensor(vals, dtype=torch.float32)

    H = torch.sparse_coo_tensor(
        torch.stack([rows_t, cols_t]),
        vals_t,
        (N, n_clusters),
    ).coalesce()

    nnz = H._nnz()
    density = nnz / (N * n_clusters) * 100
    print(f"[build_knn_cluster_incidence] H shape: ({N}, {n_clusters}), "
          f"nnz={nnz:,}, density={density:.3f}%")
    return H
