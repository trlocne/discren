"""DISCREN — the assembled model.

This file is deliberately thin: it wires together the blocks defined in
:mod:`model.modules` and does nothing else. Each attribute below corresponds to
one labelled block group in the architecture figure:

    ``self.dropedge``   degree-scaled DropEdge
    ``self.pop_gate``   popularity gate (alpha)
    ``self.backbone``   collaborative backbone      (B1-B3)
    ``self.multimodal`` multimodal semantic encoder (M1-M4)
    ``self.user_llm``   controlled LLM injection, user side  (L1-L5)
    ``self.item_llm``   controlled LLM injection, item side  (L1-L5)

Reading :meth:`debug_state` after a forward pass returns a flat dictionary of
statistics from every block, which is the intended way to inspect the model
instead of adding print statements inside the encoders.

Checkpoint compatibility: parameter names are remapped on load, so checkpoints
written by the previous single-file implementation still restore correctly.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.collaborative import CollaborativeBackbone
from model.modules.graph_ops import DegreeScaledDropEdge, PopularityGate
from model.modules.llm_injection import ControlledLLMInjection
from model.modules.losses import infonce_inbatch
from model.modules.multimodal import MultimodalSemanticEncoder

# Maps parameter names written by the pre-refactor single-file model to the
# module-structured names used here. Applied on ``load_state_dict``.
_LEGACY_KEY_MAP = {
    "user_profile_trs.": "user_llm.proj.",
    "user_profile_gate.": "user_llm.gate.",
    "_user_profile_omega": "user_llm.omega_raw",
    "user_profile_mask_token": "user_llm.mask_token",
    "user_profile_decoder.": "user_llm.decoder.",
    "item_llm_text_trs.": "item_llm.proj.",
    "item_llm_text_gate.": "item_llm.gate.",
    "_item_llm_omega": "item_llm.omega_raw",
    "item_llm_mask_token": "item_llm.mask_token",
    "item_llm_decoder.": "item_llm.decoder.",
    "image_trs.": "multimodal.image_trs.",
    "text_trs.": "multimodal.text_trs.",
    "gate_v.": "multimodal.gate_v.",
    "gate_t.": "multimodal.gate_t.",
    "hgconv_img.": "multimodal.hgconv_img.",
    "hgconv_txt.": "multimodal.hgconv_txt.",
    "rca_W_tr": "multimodal.rca.W_tr",
    "rca_b_tr": "multimodal.rca.b_tr",
    "rca_W_gv": "multimodal.rca.W_gv",
    "rca_W_gt": "multimodal.rca.W_gt",
    "rca_W_av": "multimodal.rca.W_av",
    "rca_W_at": "multimodal.rca.W_at",
    "rca_W_fv": "multimodal.rca.W_fv",
    "rca_W_ft": "multimodal.rca.W_ft",
    "user_ui_embedding.": "backbone.user_ui_embedding.",
    "item_ui_embedding.": "backbone.item_ui_embedding.",
    "uu_embedding.": "backbone.uu_embedding.",
    "ii_embedding.": "backbone.ii_embedding.",
}


class Discren(nn.Module):
    """Popularity-gated multimodal hypergraph recommender with LLM injection.

    Args:
        num_users: number of users.
        num_items: number of items.
        embed_dim: embedding width ``d``.
        ui_layers: bipartite propagation depth (block B1).
        user_layers: user-user propagation depth (block B2).
        item_layers: item-item propagation depth (block B3).
        temperature: temperature of the structural contrastive terms.
        tau_modal: temperature of the modality contrastive terms.
        use_item_structural: disable to ablate block B3.
        use_modal_purifier: enable the multimodal semantic encoder (M1-M4).
        rca_iterations: refinement rounds ``T`` for block M2.
        modal_layers: propagation depth for block M3.
        image_feat_dim: raw visual feature dimension.
        text_feat_dim: raw textual feature dimension.
        dropedge_rate: base DropEdge rate.
        use_adaptive_gate: enable the popularity gate; ``False`` uses a fixed
            unit coefficient.
        use_user_profile: enable user-side LLM injection.
        user_profile_feat_dim: user-side LLM feature dimension.
        use_item_llm_text: enable item-side LLM injection.
        item_llm_text_feat_dim: item-side LLM feature dimension.
        llm_feat_scale: shared fallback initialization for ``omega``.
        user_feat_scale: user-branch initialization for ``omega``.
        item_feat_scale: item-branch initialization for ``omega``.
    """

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

        # --- Parameter-free graph operators ---------------------------------
        self.dropedge = DegreeScaledDropEdge(rate=dropedge_rate)
        self.pop_gate = PopularityGate()

        # --- B1-B3 collaborative backbone -----------------------------------
        self.backbone = CollaborativeBackbone(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            ui_layers=ui_layers,
            user_layers=user_layers,
            item_layers=item_layers,
            use_item_structural=use_item_structural,
        )

        # --- M1-M4 multimodal semantic encoder ------------------------------
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

        # --- L1-L5 controlled LLM injection ---------------------------------
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

        # --- Offline inputs, injected via set_modal_inputs ------------------
        for name in (
            "_image_feat", "_text_feat", "_user_profile_feat", "_item_llm_text_feat",
            "_image_adj", "_text_adj", "_R_norm", "_R_row_norm",
            "_H_I_image", "_H_I_text",
        ):
            self.register_buffer(name, None, persistent=False)

        # Exponential moving average of the collaborative item embedding, used
        # as a stop-gradient teacher for the modality-alignment term. Aligning
        # against the live embedding would let the two representations drift
        # together toward a degenerate solution instead of pulling content
        # toward behaviour.
        #
        # PERSISTENT, unlike the offline-input buffers above. At momentum 0.999
        # this average carries roughly the last thousand steps of history, so a
        # non-persistent buffer meant every resume silently re-seeded the
        # teacher to the current embedding -- exactly the degenerate
        # align-against-yourself state the EMA exists to avoid -- and the
        # modality-alignment term jumped at each restart. Persisting it makes a
        # resumed run continue the same trajectory. ``load_state_dict`` below
        # tolerates checkpoints written before this change.
        self._ema_momentum = 0.999
        self.register_buffer("_ema_i_cf", None, persistent=True)

        # Snapshots consumed by the trainer's auxiliary losses.
        self._last_u_ui_for_cl: Optional[torch.Tensor] = None
        self._last_i_ui_for_cl: Optional[torch.Tensor] = None
        self._last_alpha_i: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ setup
    def set_modal_inputs(self, **tensors) -> None:
        """Attach offline features and precomputed graphs to the model.

        Accepts any of ``image_feat``, ``text_feat``, ``user_profile_feat``,
        ``item_llm_text_feat``, ``image_adj``, ``text_adj``, ``R_norm``,
        ``R_row_norm``, ``H_I_image``, ``H_I_text``. Unknown keys are ignored so
        callers can pass a superset of what a given configuration needs.
        """
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
        """Fit the DropEdge probabilities and the popularity gate.

        Both quantities are deterministic functions of the *training* graph, so
        this is called once before training rather than every step.
        """
        degree = self.dropedge.fit(ui_mat)
        if self.use_adaptive_gate and self.use_modal_purifier:
            self.pop_gate.fit(degree, self.num_users)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        UI_mat: torch.Tensor,
        I2I_mat: torch.Tensor,
        U2U_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Run the full encoder.

        Args:
            UI_mat: normalized bipartite adjacency.
            I2I_mat: normalized item-item adjacency.
            U2U_mat: normalized user-user adjacency.

        Returns:
            ``(u_emb, i_emb, ii, uu)`` — final user and item embeddings plus the
            two structural-view outputs kept as contrastive targets.
        """
        # Stochastic sparsification, then the three collaborative views.
        ui_aug = self.dropedge(UI_mat)
        u_emb, i_emb, ii, uu = self.backbone(ui_aug, I2I_mat, U2U_mat)

        # Snapshot for the structural contrastive terms, taken on the PURELY
        # COLLABORATIVE representation -- before both the LLM injection below and
        # the multimodal branch further down.
        #
        # This ordering is load-bearing. The structural terms carry roughly 82% of
        # the total objective, so whatever they can reach, they dominate. When the
        # snapshot was taken *after* injection, their gradients flowed into the LLM
        # projection, gate, and the learned scale omega -- which meant omega was
        # optimized partly for the contrastive objective rather than for ranking.
        # The symptom was unambiguous: omega grew monotonically and never
        # saturated, and past the best checkpoint it kept rising (+20%) while
        # validation recall *fell* (-1.4%) and the generalization gap widened
        # (3.87 -> 4.95). A scale that keeps increasing as the metric it supposedly
        # serves gets worse is not measuring ranking utility.
        #
        # Snapshotting here confines the contrastive gradient to the collaborative
        # encoder. The ranking loss becomes the only objective shaping omega, which
        # is what makes reading omega as "how much the model wants the LLM channel"
        # a valid inference rather than an artifact.
        self._last_u_ui_for_cl = u_emb
        self._last_i_ui_for_cl = i_emb

        # Controlled LLM injection. Applied after the snapshot; since fusion is
        # additive the final embedding is unchanged by this reordering -- only the
        # gradient paths differ.
        if self.user_llm is not None and self._user_profile_feat is not None:
            u_emb = self.user_llm(u_emb, self._user_profile_feat)
        if self.item_llm is not None and self._item_llm_text_feat is not None:
            i_emb = self.item_llm(i_emb, self._item_llm_text_feat)

        if self.training:
            self._update_ema(i_emb)

        # Multimodal side signal, modulated by the popularity gate.
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
            if alpha_i is not None:
                self._last_alpha_i = alpha_i
                # The user-side coefficient is defined as the MEAN of alpha over
                # the user's interacted items, which requires the ROW-normalized
                # R (rows sum to 1). Substituting the symmetric-normalized R_norm
                # (weights 1/sqrt(d_u d_i), rows summing to sqrt(d_u)-ish) is not
                # a mean at all: it rescales alpha_u by user degree, so heavy
                # users receive several times more multimodal signal than the
                # gate intends -- the opposite of what a popularity gate is for.
                # That substitution used to happen silently whenever R_row_norm
                # was not supplied, so require it explicitly instead.
                if self._R_row_norm is None:
                    raise RuntimeError(
                        "The adaptive popularity gate needs the row-normalized "
                        "interaction matrix. Pass R_row_norm to set_modal_inputs(), "
                        "or set use_adaptive_gate=False to ablate the gate."
                    )
                alpha_u = self.pop_gate.user_alpha(self._R_row_norm)
                i_emb = i_emb + alpha_i * side_item
                u_emb = u_emb + alpha_u * side_user
            else:
                # Ablation: fixed unit coefficient for every item.
                i_emb = i_emb + side_item
                u_emb = u_emb + side_user

        return u_emb, i_emb, ii, uu

    @torch.no_grad()
    def _update_ema(self, i_emb: torch.Tensor) -> None:
        """Update the stop-gradient teacher for modality alignment."""
        if self._ema_i_cf is None:
            self._ema_i_cf = i_emb.detach().clone()
        else:
            m = self._ema_momentum
            self._ema_i_cf = m * self._ema_i_cf + (1.0 - m) * i_emb.detach()

    # ------------------------------------------------------- loss entry points
    def mae_feature_loss(self, mask_ratio: float = 0.3, gamma: float = 2.0) -> torch.Tensor:
        """Sum the masked-reconstruction losses of all active LLM branches."""
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
        """In-batch InfoNCE at the structural temperature."""
        return infonce_inbatch(z1, z2, self.tau, inbatch_size)

    def inbatch_contrastive_loss_tau(
        self, z1: torch.Tensor, z2: torch.Tensor, tau: float, inbatch_size: int = 4096
    ) -> torch.Tensor:
        """In-batch InfoNCE at an explicit temperature (modality terms)."""
        return infonce_inbatch(z1, z2, tau, inbatch_size)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix (kept for backwards compatibility)."""
        return torch.mm(F.normalize(z1), F.normalize(z2).t())

    # ------------------------------------------------- auxiliary-loss accessors
    @property
    def modal_side_item(self) -> Optional[torch.Tensor]:
        """Item-side multimodal signal, pre-gate (modality-alignment input)."""
        return self.multimodal.side_item if self.multimodal is not None else None

    @property
    def modal_teacher(self) -> Optional[torch.Tensor]:
        """EMA collaborative item embedding (modality-alignment target)."""
        return self._ema_i_cf

    @property
    def modal_image_view(self) -> Optional[torch.Tensor]:
        """Propagated visual features (modality-modality contrastive view)."""
        return self.multimodal.img_propagated if self.multimodal is not None else None

    @property
    def modal_text_view(self) -> Optional[torch.Tensor]:
        """Propagated textual features (modality-modality contrastive view)."""
        return self.multimodal.txt_propagated if self.multimodal is not None else None

    # ------------------------------------------------------------------ debug
    def debug_state(self) -> Dict[str, float]:
        """Flat dictionary of per-block statistics from the last forward pass.

        Intended for logging and for narrowing down which block is responsible
        when a run diverges, without instrumenting the encoders themselves.
        """
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

    # --------------------------------------------------- checkpoint migration
    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Load a state dict, remapping pre-refactor parameter names.

        Also reconciles ``_ema_i_cf``, which only became a persistent buffer
        after some checkpoints had already been written:

        * an older checkpoint has no such key, so ``strict`` loading is relaxed
          for that one entry and the teacher simply starts cold;
        * a newer checkpoint has it, but the buffer is ``None`` on a freshly
          constructed model, so its shape has to be materialized before the
          tensor can be copied in.
        """
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            for legacy, current in _LEGACY_KEY_MAP.items():
                if key == legacy or key.startswith(legacy):
                    new_key = current + key[len(legacy):]
                    break
            remapped[new_key] = value

        ema = remapped.get("_ema_i_cf", None)
        if ema is not None and torch.is_tensor(ema):
            # Give the buffer a concrete shape so the copy below succeeds.
            self._ema_i_cf = torch.zeros_like(ema)
        elif strict and "_ema_i_cf" not in remapped:
            # Pre-persistence checkpoint: load everything else strictly.
            result = super().load_state_dict(remapped, strict=False)
            unexpected = list(result.unexpected_keys)
            missing = [k for k in result.missing_keys if k != "_ema_i_cf"]
            if unexpected or missing:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {type(self).__name__}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            return result

        return super().load_state_dict(remapped, strict=strict)
