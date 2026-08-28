import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from evaluation.metrics import recall_at_k, ndcg_at_k
from model.modules.losses import (
    embedding_regularization,
    sampled_softmax_ranking_loss,
    synthesize_hard_negatives,
    warmup_weight,
)


class DiscrenTrainer:
    def __init__(
        self,
        model,
        lr: float = 5e-4,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        train_pairs: Optional[list] = None,
        val_pairs: Optional[list] = None,
        UI_mat: Optional[torch.Tensor] = None,
        U2U_mat: Optional[torch.Tensor] = None,
        I2I_mat: Optional[torch.Tensor] = None,
        # --- Tunable loss weights ---
        item_loss_ratio: float = 0.7,
        user_loss_ratio: float = 0.1,
        mmhcl_reg: float = 1e-3,
        lambda_modal_align: float = 0.0,
        lambda_modal_modal: float = 0.0,
        lambda_mae: float = 0.0,
        mae_mask_ratio: float = 0.3,
        hard_neg_synth_rate: float = 0.1,
        hard_neg_pool_size: int = 32,
        scheduler_type: str = 'cosine',
        scheduler_t_max: int = 300,
        scheduler_eta_min: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip_norm = 1.0

        self.UI_mat = UI_mat.to(device) if UI_mat is not None else None
        self.U2U_mat = U2U_mat.to(device) if U2U_mat is not None else None
        self.I2I_mat = I2I_mat.to(device) if I2I_mat is not None else None
        self.item_loss_ratio = float(item_loss_ratio)
        self.user_loss_ratio = float(user_loss_ratio)
        self.mmhcl_reg = float(mmhcl_reg)

        self.lambda_modal_align = float(lambda_modal_align)
        self.lambda_modal_modal = float(lambda_modal_modal)
        # LLMRec strategy D — denoised robustification.
        self.lambda_mae = float(lambda_mae)          # weight of MAE feature-restoration loss
        self.mae_mask_ratio = float(mae_mask_ratio)  # fraction of nodes masked
        # Embedding-space hard-negative synthesis (mixup/convex-combination
        # style, deliberately NOT MixGCF's discrete hop-mixing -- DINS/M-Mix
        # style instead, since the corpus-critic review found MixGCF reports
        # instability specifically on the Amazon dataset family as candidate
        # pool size grows). Now a permanent part of the main pipeline: 10% of
        # each batch's negative pool is replaced by mixup-synthesized hard
        # negatives drawn from the top-32 hardest sampled negatives.
        self.hard_neg_synth_rate = float(hard_neg_synth_rate)
        self.hard_neg_pool_size = int(hard_neg_pool_size)
        self.aux_warmup_epochs = 10
        self.num_negatives = 128
        self.contrastive_inbatch_size = 16384

        self.batch_size = 4096
        self.step_counter = 0
        self.log_every_n_steps = 50

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_pairs = train_pairs
        self._train_pairs_tensor = None
        if train_pairs is not None and len(train_pairs) > 0:
            self._train_pairs_tensor = torch.tensor(train_pairs, dtype=torch.long)
        self.val_pairs = val_pairs
        self._val_pairs_tensor = None
        if val_pairs is not None and len(val_pairs) > 0:
            self._val_pairs_tensor = torch.tensor(val_pairs, dtype=torch.long)

        self._exist_users = None
        self._user_pos_list = None
        if train_pairs is not None and len(train_pairs) > 0:
            user_pos = {}
            for u, i in train_pairs:
                user_pos.setdefault(u, []).append(i)
            self._user_pos_list = {u: list(set(items)) for u, items in user_pos.items()}
            self._exist_users = list(self._user_pos_list.keys())
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        if scheduler_type == 'lambda_096':
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda e: 0.96 ** (e / 50)
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=scheduler_t_max, eta_min=scheduler_eta_min
            )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_recall@20': [],
            'val_ndcg@20': [],
            'components': [],
        }

        self.k_values = [20]

    def train_epoch(self, epoch: int = 1, **kwargs) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        components_accum: Dict[str, float] = {}

        _n_items = int(getattr(self.model, 'num_items', 1))
        n_train = self._train_pairs_tensor.shape[0]
        # Steps per epoch (ceil so an exact multiple doesn't add a spurious batch).
        n_batch = max(1, -(-n_train // self.batch_size))
        _eu = self._exist_users
        _n_eu = len(_eu)

        for step in range(n_batch):
            self.step_counter += 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.batch_size <= _n_eu:
                u_sel = [_eu[int(j)] for j in torch.randperm(_n_eu)[:self.batch_size].tolist()]
            else:
                u_sel = [_eu[int(j)] for j in torch.randint(0, _n_eu, (self.batch_size,)).tolist()]
            pos_ids = []
            for u in u_sel:
                plist = self._user_pos_list[u]
                pos_ids.append(plist[int(torch.randint(0, len(plist), (1,)))])
            u_t = torch.tensor(u_sel, dtype=torch.long, device=self.device)
            pos_t = torch.tensor(pos_ids, dtype=torch.long, device=self.device)
            batch_neg = torch.randint(0, _n_items, (len(u_sel), self.num_negatives), device=self.device)  # [B, 128]

            # Forward: bipartite LightGCN + structural branches.
            u_ui, i_ui, ii, uu = self.model(self.UI_mat, self.I2I_mat, self.U2U_mat)

            u_e = u_ui[u_t]              # [B, d]
            pos_e = i_ui[pos_t]          # [B, d]
            neg_e = i_ui[batch_neg]      # [B, 128, d]

            # Sampled softmax: −log(e^pos / (e^pos + Σ e^neg)).
            pos_scores = (u_e * pos_e).sum(dim=1)                          # [B]
            neg_scores = torch.einsum('bd,bnd->bn', u_e, neg_e)           # [B, 128]
            # Exclude the sampled positive from the negative set: uniform negative
            # sampling over all items can draw the batch positive itself, which
            # leaks the positive score into the negative logsumexp and pushes the
            # positive down (self-competition) — biasing the ranking gradient.
            # Mask those collisions to −inf so they drop out of the logsumexp.
            collide = batch_neg == pos_t.unsqueeze(1)                     # [B, 128]
            if collide.any():
                neg_scores = neg_scores.masked_fill(collide, float('-inf'))

            # Embedding-space hard-negative synthesis (always on — baked into the
            # main pipeline). See model/modules/losses.py for the rationale and
            # the collision-protection argument.
            neg_scores, synth_score_mean = synthesize_hard_negatives(
                user_emb=u_e,
                neg_emb=neg_e,
                neg_scores=neg_scores,
                collision_mask=collide,
                synth_rate=self.hard_neg_synth_rate,
                pool_size=self.hard_neg_pool_size,
            )

            mf_loss = sampled_softmax_ranking_loss(u_e, pos_e, neg_scores)
            emb_loss = self.mmhcl_reg * 0.5 * embedding_regularization(u_e, pos_e, neg_e)

            # Contrastive: in-batch InfoNCE (structural τ).
            _cl = lambda a, b: self.model.inbatch_contrastive_loss(
                a, b, inbatch_size=self.contrastive_inbatch_size)
            _cl_modal = lambda a, b: self.model.inbatch_contrastive_loss_tau(
                a, b, tau=self.model.tau_modal, inbatch_size=self.contrastive_inbatch_size)

            # CF-only embeddings for CL (pre-modal stash).
            i_ui_cl = getattr(self.model, '_last_i_ui_for_cl', i_ui)
            u_ui_cl = getattr(self.model, '_last_u_ui_for_cl', u_ui)
            cl_item = (self.item_loss_ratio * _cl(i_ui_cl, ii)
                       if ii is not None else torch.zeros((), device=self.device))
            cl_user = self.user_loss_ratio * _cl(u_ui_cl, uu)

            loss = mf_loss + emb_loss + cl_item + cl_user
            components = {
                'total': 0.0,
                'ranking': float(mf_loss.item()),
                'emb_reg': float(emb_loss.item()),
                'cl_item': float(cl_item.item()),
                'cl_user': float(cl_user.item()),
                'pos_score': float(pos_scores.mean().item()),
                'synth_neg_score_mean': synth_score_mean,
            }

            warm = warmup_weight(epoch, self.aux_warmup_epochs)

            # Modality-alignment: pull the multimodal side signal toward the EMA
            # collaborative item embedding (stop-gradient teacher).
            _side_item = self.model.modal_side_item
            _teacher = self.model.modal_teacher
            if self.lambda_modal_align > 0 and _side_item is not None and _teacher is not None:
                modal_align = warm * self.lambda_modal_align * _cl_modal(_side_item, _teacher)
                loss = loss + modal_align
                components['modal_align'] = float(modal_align.item())

            # Modality-modality: agreement between the propagated visual and
            # textual views of the same item.
            _img_view = self.model.modal_image_view
            _txt_view = self.model.modal_text_view
            if self.lambda_modal_modal > 0 and _img_view is not None and _txt_view is not None:
                modal_modal = warm * self.lambda_modal_modal * _cl_modal(_img_view, _txt_view)
                loss = loss + modal_modal
                components['modal_modal'] = float(modal_modal.item())

            # MAE feature-restoration loss (LLMRec Eq. 9, strategy D). Denoises
            # the LLM side features; warmup-scaled like the other aux losses.
            if self.lambda_mae > 0 and hasattr(self.model, 'mae_feature_loss'):
                mae_loss = self.model.mae_feature_loss(mask_ratio=self.mae_mask_ratio)
                if torch.is_tensor(mae_loss) and mae_loss.requires_grad:
                    mae_loss = warm * self.lambda_mae * mae_loss
                    loss = loss + mae_loss
                    components['mae'] = float(mae_loss.item())

            alpha_i = getattr(self.model, '_last_alpha_i', None)
            if alpha_i is not None:
                components['gate_alpha_mean'] = float(alpha_i.mean().item())
                components['gate_alpha_std']  = float(alpha_i.std().item())

            # Log the LEARNED injection scale ω for both LLM branches. feat_scale
            # is only the init; ω is a free parameter, so its trajectory tells us
            # whether the model is turning the LLM features UP (wants more) or
            # DOWN toward 0 (rejecting them) — the real signal for tuning.
            if getattr(self.model, 'user_llm', None) is not None:
                components['omega_user'] = float(self.model.user_llm.omega.detach())
            if getattr(self.model, 'item_llm', None) is not None:
                components['omega_item'] = float(self.model.item_llm.omega.detach())

            components['total'] = float(loss.item())

            if torch.isfinite(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                total_loss += loss.item()
                for k, v in components.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        components_accum[k] = components_accum.get(k, 0.0) + v
                num_batches += 1

            if self.step_counter % self.log_every_n_steps == 0 and num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"  [Step {self.step_counter}] avg_loss: {avg_loss:.4f}")

        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_components = {k: v / max(num_batches, 1) for k, v in components_accum.items()}

        self.history['train_loss'].append(avg_loss)
        self.history['components'].append(avg_components)

        return {
            'train_loss': avg_loss,
            'components': avg_components,
        }

    @torch.no_grad()
    def validate(self, **kwargs) -> Dict[str, float]:
        """Validate on validation set (val users from self.val_pairs)."""
        self.model.eval()

        u_ui, i_ui, _ii, _uu = self.model(self.UI_mat, self.I2I_mat, self.U2U_mat)
        z_u, z_i = u_ui, i_ui
        
        if self.val_pairs is not None:
            val_users = sorted({u for u, _ in self.val_pairs})
        else:
            val_users = []

        metrics = {}
        if val_users and self.val_pairs is not None:
            user_gt = {}
            for u, i in self.val_pairs:
                user_gt.setdefault(u, set()).add(i)

            user_train_items = {}
            if self.train_pairs is not None:
                for u, i in self.train_pairs:
                    user_train_items.setdefault(u, set()).add(i)

            val_u_list = [u for u in val_users if u < z_u.shape[0]]
            if val_u_list:
                n_items = z_i.shape[0]
                u_idx = torch.tensor(val_u_list, dtype=torch.long, device=self.device)

                scores_tensor = z_u[u_idx] @ z_i.T   # [U_val, N_i] on GPU

                gt_tensor = torch.zeros(len(val_u_list), n_items, device=self.device)
                for row, u in enumerate(val_u_list):
                    train_items = user_train_items.get(u, set())
                    if train_items:
                        tidx = torch.tensor(
                            list(train_items), dtype=torch.long, device=self.device
                        )
                        scores_tensor[row, tidx] = float('-inf')
                    if u in user_gt:
                        gt_tensor[row, list(user_gt[u])] = 1.0

                for k in self.k_values:
                    metrics[f'recall@{k}'] = recall_at_k(scores_tensor, gt_tensor, k)
                    metrics[f'ndcg@{k}'] = ndcg_at_k(scores_tensor, gt_tensor, k)

        val_loss = 0.0
        if self._val_pairs_tensor is not None and len(self._val_pairs_tensor) > 0:
            def _full_corpus_ce(pairs_t):
                cap = min(self.batch_size, pairs_t.shape[0])
                idx = torch.randperm(pairs_t.shape[0])[:cap]
                s = pairs_t[idx].to(self.device)
                u_, pos_ = s[:, 0], s[:, 1]
                logits = z_u[u_] @ z_i.t()                      # [B, n_items] full-corpus
                pos_logit = logits.gather(1, pos_.view(-1, 1)).squeeze(1)
                lse = torch.logsumexp(logits, dim=1)            # includes pos (standard CE)
                return (-pos_logit + lse).mean()

            v_ce = _full_corpus_ce(self._val_pairs_tensor)
            if torch.isfinite(v_ce):
                val_loss = v_ce.item()
                metrics['val_loss'] = val_loss                  # = val full-corpus CE
                if self._train_pairs_tensor is not None and len(self._train_pairs_tensor) > 0:
                    t_ce = _full_corpus_ce(self._train_pairs_tensor)
                    if torch.isfinite(t_ce):
                        metrics['train_ce'] = t_ce.item()
                        metrics['gen_gap'] = val_loss - t_ce.item()

        self.history['val_loss'].append(val_loss)
        for k in ['recall@20', 'ndcg@20']:
            if k in metrics:
                self.history[f'val_{k}'].append(metrics[k])

        return metrics

