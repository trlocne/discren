import torch
import numpy as np

def recall_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int=20) -> float:
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    batch_size = predictions.shape[0]
    recall_sum = 0.0
    for i in range(batch_size):
        relevant_items = ground_truth[i]
        top_k_items = top_k_indices[i]
        relevant_in_topk = relevant_items[top_k_items].sum().item()
        total_relevant = relevant_items.sum().item()
        if total_relevant > 0:
            recall_sum += relevant_in_topk / total_relevant
    return recall_sum / batch_size

def ndcg_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int=20) -> float:
    batch_size = predictions.shape[0]
    ndcg_sum = 0.0
    for i in range(batch_size):
        scores = predictions[i]
        relevance = ground_truth[i]
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        top_k_relevance = relevance[top_k_indices].cpu().numpy()
        dcg = 0.0
        for j, rel in enumerate(top_k_relevance):
            dcg += (2 ** rel - 1) / np.log2(j + 2)
        ideal_relevance = torch.sort(relevance, descending=True)[0][:k].cpu().numpy()
        idcg = 0.0
        for j, rel in enumerate(ideal_relevance):
            idcg += (2 ** rel - 1) / np.log2(j + 2)
        if idcg > 0:
            ndcg_sum += dcg / idcg
    return ndcg_sum / batch_size

def precision_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int=20) -> float:
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    batch_size = predictions.shape[0]
    precision_sum = 0.0
    for i in range(batch_size):
        relevant_items = ground_truth[i]
        top_k_items = top_k_indices[i]
        relevant_in_topk = relevant_items[top_k_items].sum().item()
        precision_sum += relevant_in_topk / k
    return precision_sum / batch_size if batch_size > 0 else 0.0

def mrr_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int=20) -> float:
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    batch_size = predictions.shape[0]
    mrr_sum = 0.0
    valid_users = 0
    for i in range(batch_size):
        relevance = ground_truth[i]
        top_k_items = top_k_indices[i]
        top_k_rel = relevance[top_k_items]
        first_rel_rank = (top_k_rel > 0).nonzero(as_tuple=True)[0]
        if len(first_rel_rank) > 0:
            rank = first_rel_rank[0].item() + 1
            mrr_sum += 1.0 / rank
            valid_users += 1
    return mrr_sum / batch_size if batch_size > 0 else 0.0

def coverage_at_k(predictions: torch.Tensor, k: int=20) -> float:
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    unique_items = torch.unique(top_k_indices).numel()
    num_items = predictions.shape[1]
    return unique_items / num_items if num_items > 0 else 0.0

def cold_recall_at_k(predictions: torch.Tensor, ground_truth: torch.Tensor, interaction_counts: torch.Tensor, k: int=20, cold_threshold: int=5) -> float:
    cold_items = interaction_counts < cold_threshold
    if cold_items.sum() == 0:
        return 0.0
    _, top_k_indices = torch.topk(predictions, k, dim=-1)
    batch_size = predictions.shape[0]
    recall_sum = 0.0
    cold_count = 0
    for i in range(batch_size):
        relevant_items = ground_truth[i]
        top_k_items = top_k_indices[i]
        cold_in_topk = cold_items[top_k_items]
        relevant_cold_in_topk = ((relevant_items[top_k_items] > 0) & cold_in_topk).sum().item()
        total_relevant_cold = ((relevant_items > 0) & cold_items).sum().item()
        if total_relevant_cold > 0:
            recall_sum += relevant_cold_in_topk / total_relevant_cold
            cold_count += 1
    return recall_sum / cold_count if cold_count > 0 else 0.0
