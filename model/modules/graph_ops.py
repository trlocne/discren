"""Sparse graph operators shared by every DISCREN branch.

Contains the two *parameter-free* pieces of the architecture:

* :class:`DegreeScaledDropEdge` — stochastic edge removal whose drop
  probability grows with edge endpoint degree (paper Eq. "dropedge").
* :class:`PopularityGate` — the fixed, non-learnable popularity coefficient
  ``alpha`` that modulates how much multimodal signal each item receives
  (paper Eq. "popularity gate").

Neither module owns trainable parameters, which is exactly why they live
outside the encoders: keeping them here makes it obvious in the code that the
gate is a *deterministic function of the interaction degree*, not something the
optimizer can drift.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


def propagate(adj: torch.Tensor, x: torch.Tensor, num_layers: int) -> torch.Tensor:
    """Apply ``num_layers`` rounds of sparse propagation ``x <- adj @ x``.

    Args:
        adj: normalized sparse adjacency ``[N, N]``.
        x: node features ``[N, d]``.
        num_layers: number of propagation rounds. ``0`` returns ``x`` unchanged.

    Returns:
        Propagated features ``[N, d]``.
    """
    out = x
    for _ in range(int(num_layers)):
        out = torch.sparse.mm(adj, out)
    return out


class DegreeScaledDropEdge(nn.Module):
    """Degree-scaled DropEdge on the bipartite user-item graph.

    A uniform DropEdge rate removes proportionally as much evidence from a
    cold user with three interactions as from a head user with three hundred,
    which is the wrong trade-off under extreme sparsity. This module therefore
    makes the drop probability an increasing function of edge *endpoint
    degree*: for an edge ``(r, c)`` of the normalized bipartite adjacency,

    .. math::
        p_{\\text{keep}}(r,c) = \\frac{1}{\\sqrt{d_r d_c}}, \\qquad
        p_{\\text{drop}}(r,c) = \\rho \\cdot \\bigl(1 - p_{\\text{keep}}(r,c)\\bigr)

    so high-degree (head) edges are dropped aggressively while low-degree
    (cold) edges are almost always retained. ``rho`` is the base rate.

    Probabilities are computed once via :meth:`fit` and cached, because the
    training graph is static; only the Bernoulli mask is resampled per step.
    Dropping is a no-op in ``eval()`` mode.

    **Inverted scaling.** Surviving edges are divided by their keep probability
    :math:`1 - p_{\\text{drop}}`, so that
    :math:`\\mathbb{E}[\\hat{\\bA}_{\\text{drop}}] = \\hat{\\bA}` and the operator is
    unbiased. Without this correction the propagated signal is systematically
    attenuated during training but not at evaluation, and the damage is subtler
    than a global scale shift: the layer-averaged readout
    :math:`\\frac{1}{L+1}\\sum_l \\mathbf{E}^{(l)}` includes the ego embedding
    :math:`\\mathbf{E}^{(0)}`, which never passes through the graph and is therefore
    *unaffected* by edge dropping. Attenuating only :math:`\\mathbf{E}^{(l\\geq1)}`
    shifts the mixing ratio between per-node and collaborative signal, so the model
    trains in an ego-dominated regime and is evaluated in a balanced one.
    """

    def __init__(self, rate: float = 0.1):
        super().__init__()
        self.rate = float(rate)
        self.register_buffer("drop_probs", None, persistent=False)
        self.register_buffer("keep_scale", None, persistent=False)
        self.register_buffer("degree", None, persistent=False)
        self.debug: Dict[str, float] = {}

    @torch.no_grad()
    def fit(self, ui_mat: torch.Tensor) -> torch.Tensor:
        """Precompute per-edge drop probabilities and node degrees.

        Args:
            ui_mat: normalized bipartite adjacency ``[N_u + N_i, N_u + N_i]``.

        Returns:
            The node degree vector ``[N_u + N_i]``, which callers reuse to
            derive the popularity gate.
        """
        ui_mat = ui_mat.coalesce()
        row, col = ui_mat.indices()[0], ui_mat.indices()[1]
        n_nodes = ui_mat.shape[0]

        degree = torch.zeros(n_nodes, device=ui_mat.device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

        p_keep = 1.0 / torch.sqrt((degree[row] * degree[col]).clamp(min=1.0))
        drop_probs = ((1.0 - p_keep).clamp(0.0, 1.0) * self.rate).contiguous()
        self.drop_probs = drop_probs
        # Inverted-dropout scale. Clamped away from 1.0 so an edge that is always
        # dropped cannot produce an infinite multiplier.
        self.keep_scale = 1.0 / (1.0 - drop_probs).clamp(min=1e-3)
        self.degree = degree

        self.debug = {
            "num_edges": float(row.numel()),
            "p_drop_mean": float(drop_probs.mean()),
            "p_drop_max": float(drop_probs.max()),
            "keep_scale_mean": float(self.keep_scale.mean()),
        }
        return degree

    def forward(self, ui_mat: torch.Tensor) -> torch.Tensor:
        """Return a stochastically sparsified copy of ``ui_mat`` (train only).

        Surviving edge weights are rescaled by :math:`1/(1-p_{\\text{drop}})` so the
        operator is unbiased in expectation; see the class docstring for why an
        uncorrected mask distorts the layer-averaged readout.
        """
        if not self.training or self.drop_probs is None:
            return ui_mat
        ui_mat = ui_mat.coalesce()
        keep = torch.rand_like(self.drop_probs) > self.drop_probs
        values = ui_mat.values() * keep.float() * self.keep_scale
        self.debug["keep_frac"] = float(keep.float().mean())
        return torch.sparse_coo_tensor(
            ui_mat.indices(), values, ui_mat.shape
        ).coalesce()


class PopularityGate(nn.Module):
    """Fixed popularity coefficient controlling multimodal contribution.

    Long-tail items have too few interactions for collaborative filtering to
    place them well, so content should dominate; head items already have a
    reliable collaborative estimate that noisy content can only perturb. The
    gate encodes that prior directly from interaction degree:

    .. math::
        \\alpha_i = 1 - \\sigma\\!\\left(
            \\frac{\\log(1 + d_i) - \\mu}{\\sigma_{\\text{pop}}}\\right)

    where :math:`\\mu, \\sigma_{\\text{pop}}` are the mean and standard
    deviation of :math:`\\log(1+d_i)` over all items. The result lies in
    ``(0, 1)``, is monotonically *decreasing* in popularity, and is
    **deterministic and non-learnable** — it is computed once from the
    training graph and never updated by gradients. The user-side coefficient
    is the row-normalized interaction average
    :math:`\\alpha_u = \\hat{R}_{\\text{row}} \\alpha_i`.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("alpha_item", None, persistent=False)
        self.debug: Dict[str, float] = {}

    @torch.no_grad()
    def fit(self, degree: torch.Tensor, num_users: int) -> torch.Tensor:
        """Compute ``alpha_item`` from the bipartite node degree vector."""
        item_degree = degree[num_users:].float()
        pop = torch.log1p(item_degree)
        pop_z = (pop - pop.mean()) / pop.std().clamp(min=1e-6)
        alpha = (1.0 - torch.sigmoid(pop_z)).unsqueeze(1)
        self.alpha_item = alpha
        self.debug = {
            "alpha_mean": float(alpha.mean()),
            "alpha_std": float(alpha.std()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
        }
        return alpha

    def user_alpha(self, r_row_norm: torch.Tensor) -> torch.Tensor:
        """Aggregate item coefficients into per-user coefficients."""
        return torch.sparse.mm(r_row_norm, self.alpha_item)

    def forward(self) -> Optional[torch.Tensor]:
        return self.alpha_item
