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

    def __init__(self, n_users: int, n_items: int, interaction_matrix: np.ndarray):
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

    def __init__(self, n_users: int, n_items: int, interaction_matrix: np.ndarray):
        super().__init__(n_items)
        self.n_users = n_users
        self.n_items = n_items
        self.R = np.asarray(interaction_matrix, dtype=np.float32)

    def build_incidence_matrix(self) -> torch.Tensor:
        R_T = self.R.T
        H = torch.from_numpy((R_T > 0).astype(np.float32)).to_sparse_coo().coalesce()
        self.incidence_matrix = H
        return H

def build_modality_knn_hypergraph(modality_features: np.ndarray, top_k: int=10, include_self: bool=False, min_sim_threshold: float=0.0, eps: float=1e-12) -> torch.Tensor:
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
    H = torch.sparse_coo_tensor(torch.stack([rows.flatten(), topk_idx.flatten()]), weights.flatten(), (N_i, N_i)).coalesce()
    return H

def build_knn_cluster_incidence(features: np.ndarray, top_k: int=10, n_clusters: Optional[int]=None, min_sim: float=0.0, weighted: bool=True, seed: int=42) -> torch.Tensor:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        raise ImportError('scikit-learn required for build_knn_cluster_incidence. pip install scikit-learn')
    N, feat_dim = features.shape
    if n_clusters is None:
        n_clusters = max(int(N ** 0.5), 64)
        n_clusters = min(n_clusters, N // 2)
    feats = torch.from_numpy(np.asarray(features, dtype=np.float32))
    feats_norm = F.normalize(feats, p=2, dim=1)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=min(4096, N), n_init=3, max_iter=100)
    kmeans.fit(feats_norm.numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))
    centroids_norm = F.normalize(centroids, p=2, dim=1)
    CHUNK = 2048
    rows, cols, vals = ([], [], [])
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        sim = feats_norm[start:end] @ centroids_norm.T
        topk_sim, topk_idx = sim.topk(min(top_k, n_clusters), dim=1)
        mask = topk_sim > min_sim
        for local_i, (sims_row, idxs_row, valid) in enumerate(zip(topk_sim, topk_idx, mask)):
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
    H = torch.sparse_coo_tensor(torch.stack([rows_t, cols_t]), vals_t, (N, n_clusters)).coalesce()
    return H
