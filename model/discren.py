from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.collaborative import CollaborativeBackbone
from model.modules.graph_ops import DegreeScaledDropEdge, PopularityGate
from model.modules.llm_injection import ControlledLLMInjection
from model.modules.losses import infonce_inbatch
from model.modules.multimodal import MultimodalSemanticEncoder


class Discren(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        ui_layers: int = 2,
        user_layers: int = 2,
        item_layers: int = 2,
        temperature: float = 0.4,
        tau_modal: float = 0.2,
        use_item_structural: bool = True,
        use_modal_purifier: bool = False,
        rca_iterations: int = 3,
        modal_layers: int = 1,
        image_feat_dim: int = 4096,
        text_feat_dim: int = 1024,
        dropedge_rate: float = 0.1,
        use_adaptive_gate: bool = True,
        use_user_profile: bool = True,
        user_profile_feat_dim: int = 1024,
        use_item_llm_text: bool = False,
        item_llm_text_feat_dim: int = 1024,
        llm_feat_scale: float = 0.1,
        user_feat_scale: Optional[float] = None,
        item_feat_scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embed_dim = int(embed_dim)
        self.tau = float(temperature)
        self.tau_modal = float(tau_modal)
        self.use_modal_purifier = bool(use_modal_purifier)
        self.use_adaptive_gate = bool(use_adaptive_gate)
        self.use_user_profile = bool(use_user_profile)
        self.use_item_llm_text = bool(use_item_llm_text)

        # Graph operators
        self.dropedge = DegreeScaledDropEdge(rate=dropedge_rate)
        self.pop_gate = PopularityGate()

        # Collaborative backbone (B1-B3)
        self.backbone = CollaborativeBackbone(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            ui_layers=ui_layers,
            user_layers=user_layers,
            item_layers=item_layers,
            use_item_structural=use_item_structural,
        )

        # Multimodal semantic encoder (M1-M4)
        self.multimodal = (
            MultimodalSemanticEncoder(
                embed_dim=embed_dim,
                image_feat_dim=image_feat_dim,
                text_feat_dim=text_feat_dim,
                rca_iterations=rca_iterations,
                modal_layers=modal_layers,
            )
            if self.use_modal_purifier
            else None
        )

        # Controlled LLM injection (L1-L5)
        user_scale = user_feat_scale if user_feat_scale is not None else llm_feat_scale
        item_scale = item_feat_scale if item_feat_scale is not None else llm_feat_scale
        self.user_llm = (
            ControlledLLMInjection(user_profile_feat_dim, embed_dim, user_scale, "user")
            if self.use_user_profile
            else None
        )
        self.item_llm = (
            ControlledLLMInjection(item_llm_text_feat_dim, embed_dim, item_scale, "item")
            if self.use_item_llm_text
            else None
        )

        # Buffers for offline features
        for name in (
            "_image_feat", "_text_feat", "_user_profile_feat", "_item_llm_text_feat",
            "_image_adj", "_text_adj", "_R_norm", "_R_row_norm",
            "_H_I_image", "_H_I_text",
        ):
            self.register_buffer(name, None, persistent=False)

        self._ema_momentum = 0.999
        self.register_buffer("_ema_i_cf", None, persistent=True)

        self._last_u_ui_for_cl: Optional[torch.Tensor] = None
        self._last_i_ui_for_cl: Optional[torch.Tensor] = None
        self._last_alpha_i: Optional[torch.Tensor] = None

    def set_modal_inputs(self, **tensors) -> None:
        device = self.backbone.item_ui_embedding.weight.device
        aliases = {
            "image_feat": "_image_feat",
            "text_feat": "_text_feat",
            "user_profile_feat": "_user_profile_feat",
            "item_llm_text_feat": "_item_llm_text_feat",
            "image_adj": "_image_adj",
            "text_adj": "_text_adj",
            "R_norm": "_R_norm",
            "R_row_norm": "_R_row_norm",
            "H_I_image": "_H_I_image",
            "H_I_text": "_H_I_text",
        }
        for key, value in tensors.items():
            target = aliases.get(key)
            if target is not None and value is not None:
                setattr(self, target, value.to(device))

    def set_ui_dropedge_probs(self, ui_mat: torch.Tensor) -> None:
        degree = self.dropedge.fit(ui_mat)
        if self.use_adaptive_gate and self.use_modal_purifier:
            self.pop_gate.fit(degree, self.num_users)

    def forward(
        self,
        UI_mat: torch.Tensor,
        I2I_mat: torch.Tensor,
        U2U_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        ui_aug = self.dropedge(UI_mat)
        u_emb, i_emb, ii, uu = self.backbone(ui_aug, I2I_mat, U2U_mat)

        self._last_u_ui_for_cl = u_emb
        self._last_i_ui_for_cl = i_emb

        if self.user_llm is not None and self._user_profile_feat is not None:
            u_emb = self.user_llm(u_emb, self._user_profile_feat)
        if self.item_llm is not None and self._item_llm_text_feat is not None:
            i_emb = self.item_llm(i_emb, self._item_llm_text_feat)

        if self.training:
            self._update_ema(i_emb)

        self._last_alpha_i = None
        if self.multimodal is not None and self._image_feat is not None:
            side_item, side_user = self.multimodal(
                item_id_embedding=self.backbone.item_ui_embedding.weight,
                image_feat=self._image_feat,
                text_feat=self._text_feat,
                r_norm=self._R_norm,
                h_image=self._H_I_image,
                h_text=self._H_I_text,
                image_adj=self._image_adj,
                text_adj=self._text_adj,
            )
            side_item = F.normalize(side_item, p=2, dim=1)
            side_user = F.normalize(side_user, p=2, dim=1)

            alpha_i = self.pop_gate() if self.use_adaptive_gate else None
            if alpha_i is not None and self._R_row_norm is not None:
                self._last_alpha_i = alpha_i
                alpha_u = self.pop_gate.user_alpha(self._R_row_norm)
                i_emb = i_emb + alpha_i * side_item
                u_emb = u_emb + alpha_u * side_user
            else:
                i_emb = i_emb + side_item
                u_emb = u_emb + side_user

        return u_emb, i_emb, ii, uu

    @torch.no_grad()
    def _update_ema(self, i_emb: torch.Tensor) -> None:
        if self._ema_i_cf is None:
            self._ema_i_cf = i_emb.detach().clone()
        else:
            m = self._ema_momentum
            self._ema_i_cf = m * self._ema_i_cf + (1.0 - m) * i_emb.detach()

    def mae_feature_loss(self, mask_ratio: float = 0.3, gamma: float = 2.0) -> torch.Tensor:
        device = self.backbone.item_ui_embedding.weight.device
        total = torch.zeros((), device=device)
        found = False
        for branch in (self.user_llm, self.item_llm):
            if branch is None:
                continue
            loss = branch.mae_loss(mask_ratio=mask_ratio, gamma=gamma)
            if loss is not None:
                total = total + loss
                found = True
        return total if found else torch.zeros((), device=device)

    def inbatch_contrastive_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, inbatch_size: int = 4096
    ) -> torch.Tensor:
        return infonce_inbatch(z1, z2, self.tau, inbatch_size)

    def inbatch_contrastive_loss_tau(
        self, z1: torch.Tensor, z2: torch.Tensor, tau: float, inbatch_size: int = 4096
    ) -> torch.Tensor:
        return infonce_inbatch(z1, z2, tau, inbatch_size)

    @property
    def modal_side_item(self) -> Optional[torch.Tensor]:
        return self.multimodal.side_item if self.multimodal is not None else None

    @property
    def modal_teacher(self) -> Optional[torch.Tensor]:
        return self._ema_i_cf

    @property
    def modal_image_view(self) -> Optional[torch.Tensor]:
        return self.multimodal.img_propagated if self.multimodal is not None else None

    @property
    def modal_text_view(self) -> Optional[torch.Tensor]:
        return self.multimodal.txt_propagated if self.multimodal is not None else None

    def debug_state(self) -> Dict[str, float]:
        state: Dict[str, float] = {}
        blocks = {
            "dropedge": self.dropedge,
            "pop_gate": self.pop_gate,
            "backbone": self.backbone,
            "multimodal": self.multimodal,
            "user_llm": self.user_llm,
            "item_llm": self.item_llm,
        }
        for prefix, block in blocks.items():
            if block is None:
                continue
            for key, value in getattr(block, "debug", {}).items():
                state[f"{prefix}/{key}"] = value
        return state
