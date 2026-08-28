"""Collaborative backbone — architecture blocks B1-B3.

Three propagation paths over three views of the same interaction record:

* **B1** bipartite user-item propagation (LightGCN-style layer averaging),
* **B2** user-user co-interaction propagation,
* **B3** item-item co-interaction propagation.

B2 and B3 are kept as *separate embedding tables* rather than extra hops on
the bipartite graph, because their outputs are also the positive views of the
structural contrastive objective. Sharing a table would make the contrastive
term degenerate (the two views would be trivially identical).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.graph_ops import propagate


class CollaborativeBackbone(nn.Module):
    """Bipartite + homogeneous graph propagation over interaction data.

    Args:
        num_users: number of users.
        num_items: number of items.
        embed_dim: embedding width ``d``.
        ui_layers: bipartite propagation depth (block B1).
        user_layers: user-user propagation depth (block B2).
        item_layers: item-item propagation depth (block B3).
        use_item_structural: disable to ablate block B3.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        ui_layers: int = 2,
        user_layers: int = 2,
        item_layers: int = 2,
        use_item_structural: bool = True,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embed_dim = int(embed_dim)
        self.ui_layers = int(ui_layers)
        self.user_layers = int(user_layers)
        self.item_layers = int(item_layers)
        self.use_item_structural = bool(use_item_structural)

        # Bipartite view (B1).
        self.user_ui_embedding = nn.Embedding(num_users, embed_dim)
        self.item_ui_embedding = nn.Embedding(num_items, embed_dim)
        # Homogeneous views (B2, B3) — deliberately separate tables so the
        # structural contrastive loss compares genuinely different encoders.
        self.uu_embedding = nn.Embedding(num_users, embed_dim)
        self.ii_embedding = nn.Embedding(num_items, embed_dim)

        for emb in (
            self.user_ui_embedding,
            self.item_ui_embedding,
            self.uu_embedding,
            self.ii_embedding,
        ):
            nn.init.xavier_uniform_(emb.weight)

        self.debug: Dict[str, float] = {}

    def forward(
        self,
        ui_mat: torch.Tensor,
        i2i_mat: torch.Tensor,
        u2u_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Propagate all three collaborative views.

        Args:
            ui_mat: normalized bipartite adjacency (already DropEdge-augmented).
            i2i_mat: normalized item-item adjacency.
            u2u_mat: normalized user-user adjacency.

        Returns:
            ``(u_cf, i_cf, ii, uu)`` where ``u_cf``/``i_cf`` are the fused
            collaborative representations and ``ii``/``uu`` are the raw
            structural-view outputs retained as contrastive targets. ``ii`` is
            ``None`` when block B3 is ablated.
        """
        # B3 — item-item structural view.
        ii = (
            propagate(i2i_mat, self.ii_embedding.weight, self.item_layers)
            if self.use_item_structural
            else None
        )
        # B2 — user-user structural view.
        uu = propagate(u2u_mat, self.uu_embedding.weight, self.user_layers)

        # B1 — bipartite propagation with layer averaging. Averaging (rather
        # than taking only the last layer) keeps the ego embedding in the
        # readout, which is what makes deep propagation stable without any
        # residual weights.
        ego = torch.cat(
            (self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0
        )
        stack = [ego]
        for _ in range(self.ui_layers):
            ego = torch.sparse.mm(ui_mat, ego)
            stack.append(ego)
        fused = torch.stack(stack, dim=1).mean(dim=1)
        u_cf, i_cf = torch.split(fused, [self.num_users, self.num_items], dim=0)

        # Fuse structural views into the bipartite representation. The
        # structural branches are L2-normalized first so their magnitude
        # cannot dominate the bipartite signal early in training.
        if ii is not None:
            i_cf = i_cf + F.normalize(ii, p=2, dim=1)
        u_cf = u_cf + F.normalize(uu, p=2, dim=1)

        with torch.no_grad():
            self.debug = {
                "u_cf_norm": float(u_cf.norm(dim=1).mean()),
                "i_cf_norm": float(i_cf.norm(dim=1).mean()),
                "uu_norm": float(uu.norm(dim=1).mean()),
            }
            if ii is not None:
                self.debug["ii_norm"] = float(ii.norm(dim=1).mean())

        return u_cf, i_cf, ii, uu
