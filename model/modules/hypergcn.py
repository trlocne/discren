import torch
import torch.nn as nn
from typing import Tuple


class WeightedHypergraphConv(nn.Module):
    """Weighted Hypergraph Convolution with per-node-edge incidence weights.

    Unlike ``torch_geometric.nn.HypergraphConv``, which only supports a single
    scalar weight for each hyperedge, this layer preserves the full weighted
    incidence matrix H. Each node-hyperedge connection can therefore have its
    own weight (e.g. TF-IDF), allowing fine-grained message passing.

    Propagation follows the symmetric HGNN formulation:

        X' = D_v^{-1/2} H D_e^{-1/2} H^T D_v^{-1/2} X

    where

        H[v,e] : incidence weight between node v and hyperedge e
                 (binary or TF-IDF)

        D_v[v] = Σ_e H[v,e]     (node degree)

        D_e[e] = Σ_v H[v,e]     (hyperedge degree)

    Message passing consists of:

        1. Normalize node features:
               X_n = D_v^{-1/2} X

        2. Node → hyperedge aggregation:
               E = H^T X_n

        3. Normalize hyperedge features:
               E = D_e^{-1/2} E

        4. Hyperedge → node propagation:
               X_p = H E

        5. Final node normalization:
               X' = D_v^{-1/2} X_p

    For binary incidence matrices (H ∈ {0,1}), this reduces to the standard
    HGNN propagation.

    For weighted incidence matrices (e.g. TF-IDF), the incidence weights are
    preserved throughout both directions of propagation. Consequently, nodes
    with larger TF-IDF values contribute more strongly to hyperedge embeddings
    and receive proportionally stronger messages during propagation.

    After propagation, a learnable linear projection is applied, followed by
    a degree-aware residual self-loop to preserve each node's own features and
    reduce oversmoothing.

    Args:
        in_features (int):
            Input feature dimension.

        out_features (int):
            Output feature dimension.

        use_tfidf_weights (bool, optional):
            If True, use the weighted incidence matrix as provided.
            If False, convert H into a binary incidence matrix before
            propagation. Default is True.

    Note on caching:
        This layer caches per-hypergraph quantities (coalesced H, its
        transpose, and degree vectors) so repeated forward passes on the
        *same* hypergraph structure don't repeat sparse bookkeeping. The
        cache key is derived from the actual tensor content (shape, indices,
        values), not from ``id(H)`` or object identity, since Python object
        ids can be reused once a tensor is garbage-collected — using ``id()``
        as a cache key can silently return another hypergraph's cached
        statistics. If you call this layer with many distinct hypergraphs,
        consider clearing ``self._cache`` periodically to bound memory.
    """

    def __init__(self, in_features: int, out_features: int, use_tfidf_weights: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_tfidf_weights = bool(use_tfidf_weights)
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        self._cache = {}

    @staticmethod
    def _cache_key(H: torch.Tensor) -> Tuple:
        """Build a content-based cache key for a coalesced sparse tensor.

        Using ``id(H)`` is unsafe: once the original tensor is freed, a new
        unrelated tensor can be allocated at the same address, causing this
        layer to silently reuse degree/normalization statistics computed for
        a completely different hypergraph. Hashing shape + indices + values
        avoids that failure mode at the cost of a bit more bookkeeping.
        """
        H_c = H.coalesce()
        idx = H_c.indices()
        val = H_c.values()
        key = (
            tuple(H_c.shape),
            H_c.device.type,
            H_c.device.index,
            idx.detach().cpu().numpy().tobytes(),
            val.detach().cpu().numpy().tobytes(),
        )
        return key

    def _get_cached(self, H: torch.Tensor):
        """Return (H_coalesced, H_transpose, Dv^-1/2, De^-1/2) for H, cached."""
        key = self._cache_key(H)
        if key not in self._cache:
            H_c = H.coalesce()
            indices = H_c.indices()
            values = H_c.values().float()

            N, E = H_c.shape
            Dv = torch.zeros(N, dtype=torch.float32, device=H_c.device)
            De = torch.zeros(E, dtype=torch.float32, device=H_c.device)

            Dv.scatter_add_(0, indices[0], values)
            De.scatter_add_(0, indices[1], values)

            Dv_inv_sqrt = 1.0 / torch.sqrt(Dv.clamp(min=1e-10))
            De_inv_sqrt = 1.0 / torch.sqrt(De.clamp(min=1e-10))

            H_t = torch.sparse_coo_tensor(
                torch.stack([indices[1], indices[0]]),
                values,
                size=(E, N),
                dtype=torch.float32,
                device=H_c.device,
            ).coalesce()

            H_c_f32 = torch.sparse_coo_tensor(
                indices, values, size=H_c.shape, dtype=torch.float32, device=H_c.device
            ).coalesce()

            # Store raw (unclamped-from-zero) degrees too, in case callers
            # want them directly via _row_col_sum.
            self._cache[key] = (H_c_f32, H_t, Dv_inv_sqrt, De_inv_sqrt, Dv, De)

        return self._cache[key]

    def _row_col_sum(self, H: torch.Tensor):
        """Return raw (unnormalized) node and hyperedge degrees."""
        if H.is_sparse:
            _, _, _, _, Dv, De = self._get_cached(H)
            return Dv, De
        return H.sum(dim=1), H.sum(dim=0)

    def _spmm(self, H: torch.Tensor, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            if H.is_sparse:
                H_c, H_t, _, _, _, _ = self._get_cached(H)
                x_fp32 = x.float()
                out = torch.sparse.mm(H_t if transpose else H_c, x_fp32)
                return out.to(x.dtype)
            return (H.t() @ x) if transpose else (H @ x)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        is_sparse = H.is_sparse

        if not self.use_tfidf_weights:
            # Binarize the incidence matrix: keep structure, drop weights.
            if is_sparse:
                H_c = H.coalesce()
                H = torch.sparse_coo_tensor(
                    H_c.indices(),
                    torch.ones_like(H_c.values(), dtype=torch.float32),
                    size=H_c.shape,
                    dtype=torch.float32,
                    device=H_c.device,
                ).coalesce()
            else:
                H = (H > 0).float()
        else:
            # Keep the incidence weights (e.g. TF-IDF) exactly as provided.
            if is_sparse:
                H_c = H.coalesce()
                if not H_c.is_floating_point():
                    H = torch.sparse_coo_tensor(
                        H_c.indices(),
                        H_c.values().float(),
                        size=H_c.shape,
                        dtype=torch.float32,
                        device=H_c.device,
                    ).coalesce()
                else:
                    H = H_c
            else:
                H = H.float()

        Dv, De = self._row_col_sum(H)
        Dv_inv_sqrt = 1.0 / torch.sqrt(Dv.clamp(min=1.0))  # [N]
        De_inv_sqrt = 1.0 / torch.sqrt(De.clamp(min=1.0))  # [E]

        x_norm = x * Dv_inv_sqrt.unsqueeze(-1)
        He = self._spmm(H, x_norm, transpose=True)
        He = He * De_inv_sqrt.unsqueeze(-1)
        x_new = self._spmm(H, He, transpose=False)
        x_new = x_new * Dv_inv_sqrt.unsqueeze(-1)

        x_proj = self.W(x)
        x_new = self.W(x_new)

        self_loop_weight = (1.0 + 0.5 * torch.log(Dv.clamp(min=1.0))).clamp(max=1.5)
        x_new = x_new + x_proj * Dv_inv_sqrt.unsqueeze(-1) * self_loop_weight.unsqueeze(-1)
        return x_new

    def clear_cache(self):
        """Drop all cached per-hypergraph statistics to free memory."""
        self._cache.clear()