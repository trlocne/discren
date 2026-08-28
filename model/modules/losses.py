"""Training objective — every loss term in one place.

The full objective is

.. math::
    \\mathcal{L} = \\mathcal{L}_{\\text{rank}}
      + \\lambda_{\\text{reg}} \\mathcal{L}_{\\text{reg}}
      + \\lambda_I \\mathcal{L}_{\\text{CL}}^{I}
      + \\lambda_U \\mathcal{L}_{\\text{CL}}^{U}
      + w(t)\\bigl(
          \\lambda_{\\text{ma}} \\mathcal{L}_{\\text{align}}
        + \\lambda_{\\text{mm}} \\mathcal{L}_{\\text{modal}}
        + \\lambda_{\\text{mae}} \\mathcal{L}_{\\text{MAE}}\\bigr)

where :math:`w(t) = \\min(1, t / T_{\\text{warm}})` linearly warms up the three
auxiliary terms. The warm-up exists because all three depend on representations
that are meaningless at initialization: aligning a random modality encoder to a
random collaborative encoder injects pure noise into the ranking gradient.

Keeping these functions free of module state means each one can be unit-tested
on synthetic tensors, which is the main reason they are not methods on the model.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Row-normalized similarity matrix ``[N1, N2]``."""
    return torch.mm(F.normalize(z1, dim=1), F.normalize(z2, dim=1).t())


def infonce_inbatch(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float,
    inbatch_size: int = 4096,
) -> torch.Tensor:
    """In-batch InfoNCE with both inter-view and intra-view negatives.

    For a sampled subset of nodes, the positive pair for node ``k`` is
    ``(z1[k], z2[k])``; the denominator contains every other node in *both*
    views (excluding the self-similarity ``z1[k]·z1[k]``):

    .. math::
        \\mathcal{L} = -\\frac{1}{B}\\sum_{k}
        \\log \\frac{\\exp(s(z^1_k, z^2_k)/\\tau)}
        {\\sum_{j \\neq k}\\exp(s(z^1_k, z^1_j)/\\tau)
         + \\sum_{j}\\exp(s(z^1_k, z^2_j)/\\tau)}

    Including intra-view negatives is what prevents the two encoders from
    satisfying the objective by collapsing all nodes to a single direction.

    Args:
        z1, z2: the two views ``[N, d]``.
        tau: temperature.
        inbatch_size: number of nodes sampled per step; ``<= 0`` uses all nodes.

    Returns:
        Scalar loss.
    """
    n = z1.size(0)
    b = min(int(inbatch_size), n) if inbatch_size > 0 else n
    idx = (
        torch.randperm(n, device=z1.device)[:b]
        if b < n
        else torch.arange(n, device=z1.device)
    )
    z1_b, z2_b = z1[idx], z2[idx]

    exp = lambda s: torch.exp(s / tau)
    intra = exp(cosine_similarity_matrix(z1_b, z1_b))
    inter = exp(cosine_similarity_matrix(z1_b, z2_b))

    positive = inter.diag()
    denominator = intra.sum(1) + inter.sum(1) - intra.diag()
    return (-torch.log(positive / denominator.clamp(min=1e-12))).mean()


def sampled_softmax_ranking_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """Sampled-softmax ranking loss over a shared negative pool.

    .. math::
        \\mathcal{L}_{\\text{rank}} = \\mathbb{E}\\Bigl[
        \\mathrm{softplus}\\bigl(
        \\log \\textstyle\\sum_{j} e^{s_{uj}} - s_{ui^+}\\bigr)\\Bigr]

    This is a multi-negative generalization of pairwise BPR: with a single
    negative the two coincide, but contrasting each positive against many
    negatives at once yields a far lower-variance gradient, which matters when
    the catalogue is large and the positive signal is sparse.

    Args:
        user_emb: ``[B, d]``.
        pos_emb: ``[B, d]``.
        neg_scores: precomputed negative logits ``[B, N_neg]`` (already masked
            for positive collisions and hard-negative substitution).

    Returns:
        Scalar loss.
    """
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    return F.softplus(torch.logsumexp(neg_scores, dim=1) - pos_scores).mean()


def embedding_regularization(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """Scale-invariant L2 penalty on the embeddings touched by a batch.

    The negative term is averaged over the negative count so the penalty has the
    same magnitude whether one or 128 negatives are sampled — without this the
    effective regularization strength would silently scale with ``num_negatives``.
    """
    batch = user_emb.shape[0]
    n_neg = neg_emb.shape[1]
    return (
        (user_emb.pow(2).sum() + pos_emb.pow(2).sum()) / batch
        + neg_emb.pow(2).sum() / (batch * n_neg)
    )


def synthesize_hard_negatives(
    user_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    neg_scores: torch.Tensor,
    collision_mask: torch.Tensor,
    synth_rate: float = 0.1,
    pool_size: int = 32,
) -> Tuple[torch.Tensor, float]:
    """Replace the easiest negatives with mixup-synthesized hard negatives.

    Uniformly sampled negatives are overwhelmingly trivial in a catalogue of
    23k items, so most of the pool contributes almost no gradient. This routine
    takes the ``pool_size`` highest-scoring sampled negatives, forms convex
    combinations of random pairs among them, and uses those synthetic points to
    overwrite the *lowest-scoring* slots — keeping the pool size fixed while
    raising its average difficulty.

    Interpolating in embedding space (rather than mixing propagation hops as in
    hop-mixing schemes) keeps the synthesized point inside the convex hull of
    genuine negatives, which avoids the instability that discrete hop mixing
    exhibits when the candidate pool grows.

    Positive-collision slots are protected: they were masked to ``-inf`` and are
    therefore the global minima, so a naive lowest-score selection would target
    them first and a finite synthetic score would silently revive a masked
    positive inside the softmax denominator.

    Args:
        user_emb: ``[B, d]``.
        neg_emb: sampled negative embeddings ``[B, N_neg, d]``.
        neg_scores: negative logits ``[B, N_neg]``, collisions already ``-inf``.
        collision_mask: ``[B, N_neg]`` marking positive collisions.
        synth_rate: fraction of the pool to replace.
        pool_size: number of hardest negatives used as mixup parents.

    Returns:
        ``(updated_neg_scores, mean_synthetic_score)``.
    """
    batch, n_neg = neg_scores.shape
    device = neg_scores.device
    pool = min(int(pool_size), n_neg)
    n_synth = max(1, int(round(float(synth_rate) * n_neg)))
    dim = neg_emb.shape[-1]

    _, hard_idx = torch.topk(neg_scores, pool, dim=1)
    hard_emb = torch.gather(neg_emb, 1, hard_idx.unsqueeze(-1).expand(-1, -1, dim))

    idx_a = torch.randint(0, pool, (batch, n_synth), device=device)
    idx_b = torch.randint(0, pool, (batch, n_synth), device=device)
    emb_a = torch.gather(hard_emb, 1, idx_a.unsqueeze(-1).expand(-1, -1, dim))
    emb_b = torch.gather(hard_emb, 1, idx_b.unsqueeze(-1).expand(-1, -1, dim))

    lam = torch.rand(batch, n_synth, 1, device=device)
    synth_emb = lam * emb_a + (1.0 - lam) * emb_b
    synth_scores = torch.einsum("bd,bnd->bn", user_emb, synth_emb)

    # Protect masked collisions from being selected as "easiest".
    easy_rank = (
        neg_scores.masked_fill(collision_mask, float("inf"))
        if collision_mask.any()
        else neg_scores
    )
    _, easy_idx = torch.topk(easy_rank, n_synth, dim=1, largest=False)
    updated = neg_scores.scatter(1, easy_idx, synth_scores)
    return updated, float(synth_scores.mean())


def warmup_weight(epoch: int, warmup_epochs: int) -> float:
    """Linear auxiliary-loss warm-up ``w(t) = min(1, t / T_warm)``."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))
