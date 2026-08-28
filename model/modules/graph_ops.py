from typing import Dict, Optional
import torch
import torch.nn as nn

def propagate(adj: torch.Tensor, x: torch.Tensor, num_layers: int) -> torch.Tensor:
    out = x
    for _ in range(int(num_layers)):
        out = torch.sparse.mm(adj, out)
    return out

class DegreeScaledDropEdge(nn.Module):

    def __init__(self, rate: float=0.1):
        super().__init__()
        self.rate = float(rate)
        self.register_buffer('drop_probs', None, persistent=False)
        self.register_buffer('keep_scale', None, persistent=False)
        self.register_buffer('degree', None, persistent=False)
        self.debug: Dict[str, float] = {}

    @torch.no_grad()
    def fit(self, ui_mat: torch.Tensor) -> torch.Tensor:
        ui_mat = ui_mat.coalesce()
        row, col = (ui_mat.indices()[0], ui_mat.indices()[1])
        n_nodes = ui_mat.shape[0]
        degree = torch.zeros(n_nodes, device=ui_mat.device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        p_keep = 1.0 / torch.sqrt((degree[row] * degree[col]).clamp(min=1.0))
        drop_probs = ((1.0 - p_keep).clamp(0.0, 1.0) * self.rate).contiguous()
        self.drop_probs = drop_probs
        self.keep_scale = 1.0 / (1.0 - drop_probs).clamp(min=0.001)
        self.degree = degree
        self.debug = {'num_edges': float(row.numel()), 'p_drop_mean': float(drop_probs.mean()), 'p_drop_max': float(drop_probs.max()), 'keep_scale_mean': float(self.keep_scale.mean())}
        return degree

    def forward(self, ui_mat: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_probs is None:
            return ui_mat
        ui_mat = ui_mat.coalesce()
        keep = torch.rand_like(self.drop_probs) > self.drop_probs
        values = ui_mat.values() * keep.float() * self.keep_scale
        self.debug['keep_frac'] = float(keep.float().mean())
        return torch.sparse_coo_tensor(ui_mat.indices(), values, ui_mat.shape).coalesce()

class PopularityGate(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('alpha_item', None, persistent=False)
        self.debug: Dict[str, float] = {}

    @torch.no_grad()
    def fit(self, degree: torch.Tensor, num_users: int) -> torch.Tensor:
        item_degree = degree[num_users:].float()
        pop = torch.log1p(item_degree)
        pop_z = (pop - pop.mean()) / pop.std().clamp(min=1e-06)
        alpha = (1.0 - torch.sigmoid(pop_z)).unsqueeze(1)
        self.alpha_item = alpha
        self.debug = {'alpha_mean': float(alpha.mean()), 'alpha_std': float(alpha.std()), 'alpha_min': float(alpha.min()), 'alpha_max': float(alpha.max())}
        return alpha

    def user_alpha(self, r_row_norm: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(r_row_norm, self.alpha_item)

    def forward(self) -> Optional[torch.Tensor]:
        return self.alpha_item
