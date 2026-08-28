"""Multimodal semantic encoder — architecture blocks M1-M4.

Pipeline, in the exact order the tensors flow:

* **M1** modality projection: raw visual/textual features into ``R^d``.
* **M2** reciprocal cross-modal attention (:mod:`model.modules.rca`).
* **M3** semantic hypergraph propagation over the cluster incidence matrix
  (:mod:`model.modules.hypergcn`), with a pairwise kNN adjacency fallback.
* **M4** behaviour gating: the *propagated* modality features gate the shared
  item identity embedding, and the two gated streams are averaged.

The order of M3 before M4 matters and is easy to get backwards. Gating the
identity embedding with *raw* projections would let a single noisy product
photo veto an item's content signal outright; gating with *propagated*
features means the veto has to be corroborated by the item's semantic
neighbourhood first. Both the hypergraph path and the kNN fallback therefore
follow the same propagate-then-gate ordering, so the tensors exported for the
modality-modality contrastive term stay comparable across the two paths.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.hypergcn import WeightedHypergraphConv
from model.modules.rca import ReciprocalCrossModalAttention


class MultimodalSemanticEncoder(nn.Module):
    """Encode visual + textual item content into a collaborative side signal.

    Args:
        embed_dim: embedding width ``d``.
        image_feat_dim: raw visual feature dimension.
        text_feat_dim: raw textual feature dimension.
        rca_iterations: refinement rounds ``T`` for block M2.
        modal_layers: propagation depth for block M3.
    """

    def __init__(
        self,
        embed_dim: int,
        image_feat_dim: int = 4096,
        text_feat_dim: int = 1024,
        rca_iterations: int = 3,
        modal_layers: int = 1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.modal_layers = int(modal_layers)

        # M1 — projection into the collaborative embedding space.
        self.image_trs = nn.Linear(image_feat_dim, embed_dim)
        self.text_trs = nn.Linear(text_feat_dim, embed_dim)
        for lin in (self.image_trs, self.text_trs):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        # M2 — reciprocal cross-modal attention.
        self.rca = ReciprocalCrossModalAttention(embed_dim, rca_iterations)

        # M3 — one weighted hypergraph convolution per modality. Separate
        # weights because the two incidence matrices describe different
        # semantic partitions of the catalogue.
        self.hgconv_img = WeightedHypergraphConv(embed_dim, embed_dim, use_tfidf_weights=True)
        self.hgconv_txt = WeightedHypergraphConv(embed_dim, embed_dim, use_tfidf_weights=True)

        # M4 — behaviour gates.
        self.gate_v = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        self.debug: Dict[str, float] = {}
        # Tensors consumed by the auxiliary contrastive objectives.
        self.img_propagated: Optional[torch.Tensor] = None
        self.txt_propagated: Optional[torch.Tensor] = None
        self.side_item: Optional[torch.Tensor] = None

    def forward(
        self,
        item_id_embedding: torch.Tensor,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor,
        r_norm: torch.Tensor,
        h_image: Optional[torch.Tensor] = None,
        h_text: Optional[torch.Tensor] = None,
        image_adj: Optional[torch.Tensor] = None,
        text_adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce the item-side and user-side multimodal signals.

        Args:
            item_id_embedding: shared item identity table ``[N_i, d]``.
            image_feat: raw visual features ``[N_i, D_v]``.
            text_feat: raw textual features ``[N_i, D_t]``.
            r_norm: normalized interaction matrix ``[N_u, N_i]`` used to lift
                the item signal to users.
            h_image, h_text: semantic hypergraph incidence matrices
                ``[N_i, E]``. When both are given, block M3 uses hypergraph
                propagation.
            image_adj, text_adj: pairwise kNN adjacencies ``[N_i, N_i]`` used
                as the fallback when no incidence matrix is available.

        Returns:
            ``(side_item, side_user)`` with shapes ``[N_i, d]`` and ``[N_u, d]``.
        """
        # M1 — project.
        img = self.image_trs(image_feat)
        txt = self.text_trs(text_feat)

        # M2 — reciprocal refinement.
        img, txt = self.rca(img, txt)

        # M3 — propagate over the semantic structure.
        if h_image is not None and h_text is not None:
            for _ in range(self.modal_layers):
                img = self.hgconv_img(img, h_image)
            for _ in range(self.modal_layers):
                txt = self.hgconv_txt(txt, h_text)
            path = "hypergraph"
        else:
            for _ in range(self.modal_layers):
                img = torch.sparse.mm(image_adj, img)
            for _ in range(self.modal_layers):
                txt = torch.sparse.mm(text_adj, txt)
            path = "knn"

        # Exported for the modality-modality contrastive term.
        self.img_propagated = img
        self.txt_propagated = txt

        # M4 — gate the shared identity embedding with propagated content.
        g_v = self.gate_v(img)
        g_t = self.gate_t(txt)
        img_item = item_id_embedding * g_v
        txt_item = item_id_embedding * g_t

        side_item = 0.5 * (img_item + txt_item)
        side_user = torch.sparse.mm(r_norm, side_item)
        self.side_item = side_item

        with torch.no_grad():
            self.debug = {
                "path": path,
                "gate_v_mean": float(g_v.mean()),
                "gate_t_mean": float(g_t.mean()),
                "img_prop_norm": float(img.norm(dim=1).mean()),
                "txt_prop_norm": float(txt.norm(dim=1).mean()),
                "side_item_norm": float(side_item.norm(dim=1).mean()),
            }
            self.debug.update({f"rca_{k}": v for k, v in self.rca.debug.items()})
        return side_item, side_user
