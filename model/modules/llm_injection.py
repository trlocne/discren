"""Controlled LLM injection — architecture blocks L1-L4.

An LLM-written profile is a *generated* artefact: it is fluent whether or not
it is grounded, so a fraction of it is confabulated. Adding such a feature to a
collaborative embedding at unit strength lets that confabulation compete with
observed behaviour on equal terms, which is why naive concatenation of LLM
features tends to help sparse users and hurt dense ones simultaneously.

This module makes the injection *attenuable* at four levels:

* **L1** projection into the embedding space,
* **L2** L2 normalization, so the magnitude of the LLM encoder cannot set the
  contribution scale,
* **L3** an element-wise learned sigmoid gate, which can silence individual
  dimensions,
* **L4** a single learned scalar :math:`\\omega = \\mathrm{softplus}(\\tilde\\omega)`
  shared by the branch, which can drive the whole channel toward zero.

.. math::
    \\tilde{s} = \\frac{W s}{\\lVert W s \\rVert_2}, \\qquad
    e \\leftarrow e + \\omega \\cdot \\bigl(\\sigma(W_g \\tilde{s}) \\odot \\tilde{s}\\bigr)

Because :math:`\\omega` is free and initialized small, it is also a *diagnostic*:
its trajectory during training reports whether the model wants more LLM signal
or is rejecting it, which is far more informative than the fixed hyperparameter.

Block **L5** (:meth:`mae_loss`) adds masked feature reconstruction. Masking a
fraction of nodes and forcing a decoder to restore them prevents the projection
from memorizing individual generated profiles verbatim.

This is *restoration*, not denoising of the LLM features themselves: the offline
feature matrix is a fixed input and the reconstruction target is detached, so no
gradient reaches it. The loss shapes the projection and decoder only.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    """Invert softplus so ``softplus(raw) == y`` at initialization."""
    y = max(float(y), 1e-4)
    return math.log(math.expm1(y)) if y < 20 else y


class ControlledLLMInjection(nn.Module):
    """Attenuable injection of one offline LLM feature stream.

    Args:
        feat_dim: dimension of the incoming LLM feature.
        embed_dim: embedding width ``d``.
        init_scale: initial value of :math:`\\omega`.
        name: branch label used in the debug dictionary.
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        init_scale: float = 0.1,
        name: str = "llm",
    ):
        super().__init__()
        self.name = str(name)

        # L1 — projection.
        self.proj = nn.Linear(feat_dim, embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # L3 — element-wise gate.
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        # L4 — learned scalar strength, softplus-parameterized to stay positive.
        self.omega_raw = nn.Parameter(torch.tensor(_inv_softplus(init_scale)))

        # L5 — masked reconstruction head.
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
        """Current effective injection strength :math:`\\omega > 0`."""
        return F.softplus(self.omega_raw)

    def forward(self, embedding: torch.Tensor, llm_feat: torch.Tensor) -> torch.Tensor:
        """Add the gated, scaled LLM signal to ``embedding``.

        Args:
            embedding: collaborative embedding ``[N, d]``.
            llm_feat: offline LLM feature ``[N, feat_dim]``.

        Returns:
            The updated embedding ``[N, d]``.
        """
        projected = self.proj(llm_feat)          # L1
        self.projected = projected               # retained for L5
        normalized = F.normalize(projected, p=2, dim=1)   # L2
        gate = self.gate(normalized)             # L3
        omega = self.omega                       # L4
        delta = omega * (gate * normalized)

        with torch.no_grad():
            self.debug = {
                f"omega_{self.name}": float(omega),
                f"gate_{self.name}_mean": float(gate.mean()),
                f"delta_{self.name}_norm": float(delta.norm(dim=1).mean()),
            }
        return embedding + delta

    def mae_loss(self, mask_ratio: float = 0.3, gamma: float = 2.0) -> Optional[torch.Tensor]:
        """Masked feature-restoration loss (block L5).

        Replaces a random ``mask_ratio`` fraction of projected features with the
        learned mask token, reconstructs them, and penalizes the cosine error on
        masked rows only:

        .. math::
            \\mathcal{L}_{\\text{MAE}} =
            \\mathbb{E}_{v \\in \\mathcal{M}}
            \\bigl(1 - \\cos(\\hat{p}_v, p_v)\\bigr)^{\\gamma}

        The target is detached, so the loss shapes the *decoder and projection*
        rather than collapsing the target itself.

        Returns:
            Scalar loss, or ``None`` if no projection was produced this pass.
        """
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
