from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.hypergcn import WeightedHypergraphConv
from model.modules.rca import ReciprocalCrossModalAttention

class MultimodalSemanticEncoder(nn.Module):
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

        self.image_trs = nn.Linear(image_feat_dim, embed_dim)
        self.text_trs = nn.Linear(text_feat_dim, embed_dim)
        for lin in (self.image_trs, self.text_trs):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.rca = ReciprocalCrossModalAttention(embed_dim, rca_iterations)

        self.hgconv_img = WeightedHypergraphConv(embed_dim, embed_dim, use_tfidf_weights=True)
        self.hgconv_txt = WeightedHypergraphConv(embed_dim, embed_dim, use_tfidf_weights=True)

        self.gate_v = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        self.debug: Dict[str, float] = {}
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
        img = self.image_trs(image_feat)
        txt = self.text_trs(text_feat)

        img, txt = self.rca(img, txt)

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

        self.img_propagated = img
        self.txt_propagated = txt

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
