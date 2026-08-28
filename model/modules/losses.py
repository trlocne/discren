from typing import Tuple

import torch
import torch.nn.functional as F

def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return torch.mm(F.normalize(z1, dim=1), F.normalize(z2, dim=1).t())

def infonce_inbatch(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float,
    inbatch_size: int = 4096,
) -> torch.Tensor:
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
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    return F.softplus(torch.logsumexp(neg_scores, dim=1) - pos_scores).mean()

def embedding_regularization(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
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
    masked_scores = neg_scores.masked_fill(collision_mask, -1e9)
    batch_size, num_negs = masked_scores.shape

    k = min(int(pool_size), num_negs)
    n_synth = max(1, int(round(num_negs * float(synth_rate))))

    _, top_idx = masked_scores.topk(k, dim=1)

    r1 = torch.randint(0, k, (batch_size, n_synth), device=user_emb.device)
    r2 = torch.randint(0, k, (batch_size, n_synth), device=user_emb.device)
    same = r1 == r2
    r2[same] = (r2[same] + 1) % k

    parent1_idx = top_idx.gather(1, r1)
    parent2_idx = top_idx.gather(1, r2)

    p1 = neg_emb.gather(1, parent1_idx.unsqueeze(-1).expand(-1, -1, neg_emb.size(-1)))
    p2 = neg_emb.gather(1, parent2_idx.unsqueeze(-1).expand(-1, -1, neg_emb.size(-1)))

    alpha = torch.rand((batch_size, n_synth, 1), device=user_emb.device)
    synth_emb = F.normalize(alpha * p1 + (1.0 - alpha) * p2, p=2, dim=-1)
    synth_scores = (user_emb.unsqueeze(1) * synth_emb).sum(dim=-1)

    _, bot_idx = masked_scores.topk(n_synth, dim=1, largest=False)
    updated_scores = neg_scores.scatter(1, bot_idx, synth_scores)

    with torch.no_grad():
        hardest_mean = float(masked_scores.topk(1, dim=1).values.mean())

    return updated_scores, hardest_mean

def warmup_weight(epoch: int, warmup_epochs: int = 10) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch + 1) / float(warmup_epochs)))
