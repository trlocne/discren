import torch
import torch.nn as nn
from typing import Tuple


class WeightedHypergraphConv(nn.Module):
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
        H_c = H.coalesce()
        idx = H_c.indices()
        val = H_c.values()
        return (
            tuple(H_c.shape),
            H_c.device.type,
            H_c.device.index,
            idx.detach().cpu().numpy().tobytes(),
            val.detach().cpu().numpy().tobytes(),
        )

    def _get_cached(self, H: torch.Tensor):
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

            self._cache[key] = (H_c_f32, H_t, Dv_inv_sqrt, De_inv_sqrt, Dv, De)

        return self._cache[key]

    def _row_col_sum(self, H: torch.Tensor):
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
        Dv_inv_sqrt = 1.0 / torch.sqrt(Dv.clamp(min=1.0))
        De_inv_sqrt = 1.0 / torch.sqrt(De.clamp(min=1.0))

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
        self._cache.clear()
