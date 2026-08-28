"""Reciprocal cross-modal attention — architecture block M2.

Visual and textual descriptions of the same product are neither redundant nor
independent: a garment photograph resolves cut and colour that the title omits,
while the title resolves material and intended use that the photograph cannot
show. A single fusion step forces the model to commit to one such division of
labour before it knows which modality is reliable for a given item.

This module instead refines both streams *jointly and repeatedly*. At every
round, the concatenated state produces one gate per modality; each stream is
gated by evidence from the other, transformed, and added back through a linear
residual path. Applying the update ``T`` times lets a confident modality
progressively suppress an uninformative partner rather than averaging with it.

One round, for visual/textual states :math:`X_v, X_t \\in \\mathbb{R}^{N \\times d}`:

.. math::
    E &= [X_v \\Vert X_t] \\\\
    E' &= \\tanh(E W_{tr} + b_{tr}) \\\\
    g_v &= \\sigma(E' W_{gv}), \\quad g_t = \\sigma(E' W_{gt}) \\\\
    F_v &= \\tanh\\bigl((X_v \\odot g_v) W_{av}\\bigr), \\quad
      F_t = \\tanh\\bigl((X_t \\odot g_t) W_{at}\\bigr) \\\\
    X_v &\\leftarrow F_v + X_v W_{fv}, \\quad
      X_t \\leftarrow F_t + X_t W_{ft}

The gate is produced from the *joint* state, which is what makes the exchange
reciprocal — neither modality is a fixed query and neither a fixed context.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ReciprocalCrossModalAttention(nn.Module):
    """Iterative gated exchange between two modality streams.

    Args:
        embed_dim: per-modality width ``d``.
        num_iterations: number of refinement rounds ``T``.
    """

    def __init__(self, embed_dim: int, num_iterations: int = 3):
        super().__init__()
        d = int(embed_dim)
        self.embed_dim = d
        self.num_iterations = int(num_iterations)

        # Joint transform producing the gating state.
        self.W_tr = nn.Parameter(torch.empty(2 * d, 2 * d))
        self.b_tr = nn.Parameter(torch.zeros(2 * d))
        # Per-modality gate projections.
        self.W_gv = nn.Parameter(torch.empty(2 * d, d))
        self.W_gt = nn.Parameter(torch.empty(2 * d, d))
        # Gated-state transforms.
        self.W_av = nn.Parameter(torch.empty(d, d))
        self.W_at = nn.Parameter(torch.empty(d, d))
        # Linear residual (skip) projections.
        self.W_fv = nn.Parameter(torch.empty(d, d))
        self.W_ft = nn.Parameter(torch.empty(d, d))

        for p in (
            self.W_tr, self.W_gv, self.W_gt,
            self.W_av, self.W_at, self.W_fv, self.W_ft,
        ):
            nn.init.xavier_uniform_(p)

        self.debug: Dict[str, float] = {}

    def forward(
        self, x_v: torch.Tensor, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine both modality streams for ``num_iterations`` rounds.

        Args:
            x_v: visual features ``[N, d]``.
            x_t: textual features ``[N, d]``.

        Returns:
            Refined ``(x_v, x_t)``, each ``[N, d]``.
        """
        gate_v_mean = gate_t_mean = 0.0
        for _ in range(self.num_iterations):
            joint = torch.cat([x_v, x_t], dim=1)
            state = torch.tanh(joint @ self.W_tr + self.b_tr)

            g_v = torch.sigmoid(state @ self.W_gv)
            g_t = torch.sigmoid(state @ self.W_gt)

            f_v = torch.tanh((x_v * g_v) @ self.W_av)
            f_t = torch.tanh((x_t * g_t) @ self.W_at)

            x_v = f_v + x_v @ self.W_fv
            x_t = f_t + x_t @ self.W_ft

            with torch.no_grad():
                gate_v_mean = float(g_v.mean())
                gate_t_mean = float(g_t.mean())

        with torch.no_grad():
            self.debug = {
                "gate_v_mean": gate_v_mean,
                "gate_t_mean": gate_t_mean,
                "x_v_norm": float(x_v.norm(dim=1).mean()),
                "x_t_norm": float(x_t.norm(dim=1).mean()),
            }
        return x_v, x_t
