import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-4)
    return math.log(math.expm1(y)) if y < 20 else y


class ControlledLLMInjection(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        init_scale: float = 0.1,
        name: str = "llm",
    ):
        super().__init__()
        self.name = str(name)

        # L1: Projection
        self.proj = nn.Linear(feat_dim, embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # L3: Gate
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        # L4: Learnable Scale Omega
        self.omega_raw = nn.Parameter(torch.tensor(_inv_softplus(init_scale)))

        # L5: Masked Reconstruction Head
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.projected: Optional[torch.Tensor] = None
        self.debug: Dict[str, float] = {}

    @property
    def omega(self) -> torch.Tensor:
        return F.softplus(self.omega_raw)

    def forward(self, embedding: torch.Tensor, llm_feat: torch.Tensor) -> torch.Tensor:
        projected = self.proj(llm_feat)
        self.projected = projected
        normalized = F.normalize(projected, p=2, dim=1)
        gate = self.gate(normalized)
        omega = self.omega
        delta = omega * (gate * normalized)

        with torch.no_grad():
            self.debug = {
                f"omega_{self.name}": float(omega),
                f"gate_{self.name}_mean": float(gate.mean()),
                f"delta_{self.name}_norm": float(delta.norm(dim=1).mean()),
            }
        return embedding + delta

    def mae_loss(self, mask_ratio: float = 0.3, gamma: float = 2.0) -> Optional[torch.Tensor]:
        if self.projected is None:
            return None
        projected = self.projected
        target = F.normalize(projected.detach(), p=2, dim=1)

        n = projected.shape[0]
        n_mask = max(1, int(mask_ratio * n))
        idx = torch.randperm(n, device=projected.device)[:n_mask]

        masked = projected.clone()
        masked[idx] = self.mask_token
        recon = self.decoder(masked)

        cos = (F.normalize(recon[idx], p=2, dim=1) * target[idx]).sum(dim=1).clamp(-1.0, 1.0)
        loss = ((1.0 - cos) ** gamma).mean()
        with torch.no_grad():
            self.debug[f"mae_{self.name}"] = float(loss)
        return loss
