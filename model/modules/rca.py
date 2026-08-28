from typing import Dict, Tuple
import torch
import torch.nn as nn

class ReciprocalCrossModalAttention(nn.Module):

    def __init__(self, embed_dim: int, num_iterations: int=3):
        super().__init__()
        d = int(embed_dim)
        self.embed_dim = d
        self.num_iterations = int(num_iterations)
        self.W_tr = nn.Parameter(torch.empty(2 * d, 2 * d))
        self.b_tr = nn.Parameter(torch.zeros(2 * d))
        self.W_gv = nn.Parameter(torch.empty(2 * d, d))
        self.W_gt = nn.Parameter(torch.empty(2 * d, d))
        self.W_av = nn.Parameter(torch.empty(d, d))
        self.W_at = nn.Parameter(torch.empty(d, d))
        self.W_fv = nn.Parameter(torch.empty(d, d))
        self.W_ft = nn.Parameter(torch.empty(d, d))
        for p in (self.W_tr, self.W_gv, self.W_gt, self.W_av, self.W_at, self.W_fv, self.W_ft):
            nn.init.xavier_uniform_(p)
        self.debug: Dict[str, float] = {}

    def forward(self, x_v: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_v_mean = gate_t_mean = 0.0
        for _ in range(self.num_iterations):
            joint = torch.cat([x_v, x_t], dim=1)
            state = torch.tanh(joint @ self.W_tr + self.b_tr)
            g_v = torch.sigmoid(state @ self.W_gv)
            g_t = torch.sigmoid(state @ self.W_gt)
            f_v = torch.tanh(x_v * g_v @ self.W_av)
            f_t = torch.tanh(x_t * g_t @ self.W_at)
            x_v = f_v + x_v @ self.W_fv
            x_t = f_t + x_t @ self.W_ft
            with torch.no_grad():
                gate_v_mean = float(g_v.mean())
                gate_t_mean = float(g_t.mean())
        with torch.no_grad():
            self.debug = {'gate_v_mean': gate_v_mean, 'gate_t_mean': gate_t_mean, 'x_v_norm': float(x_v.norm(dim=1).mean()), 'x_t_norm': float(x_t.norm(dim=1).mean())}
        return (x_v, x_t)
