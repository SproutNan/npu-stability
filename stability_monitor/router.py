import torch
import torch.distributed as dist


@torch.no_grad()
def per_token_expert_entropy(
    logits: torch.Tensor, gates: torch.Tensor, dp_group=None
) -> float:
    logits = logits.to(torch.float32)
    logsumexp = torch.logsumexp(logits, dim=-1)
    entropy = logsumexp - (gates * logits).sum(dim=-1)

    if dp_group is not None:
        n_token = torch.tensor(
            entropy.numel(), dtype=torch.float32, device=gates.device
        )
        entropy = entropy.sum()
        global_sum_entropy = entropy.clone()
        global_sum_n_token = n_token.clone()
        dist.all_reduce(global_sum_entropy, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(global_sum_n_token, op=dist.ReduceOp.SUM, group=dp_group)
        return (global_sum_entropy / global_sum_n_token).item()
    return entropy.mean().item()


@torch.no_grad()
def max_violation(logits: torch.Tensor, topk: int, dp_group=None) -> float:
    _, indices = torch.topk(logits, dim=-1, k=topk)
    unique_experts, load = torch.unique(indices, return_counts=True)
    load_full = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
    load_full[unique_experts] = load.float()

    if dp_group is not None:
        global_sum_load = load_full.clone()
        dist.all_reduce(global_sum_load, op=dist.ReduceOp.SUM, group=dp_group)
        return (
            (global_sum_load.max() - global_sum_load.mean()) / global_sum_load.mean()
        ).item()
    maxvio = (load_full.max() - load_full.mean()) / load_full.mean()
    return maxvio.item()


@torch.no_grad()
def router_diversification(
    gates: torch.Tensor, topk: int, after_topk: bool = True, dp_group=None
) -> float:
    if after_topk:
        values, indices = torch.topk(gates, k=topk, dim=1)
        gates_mask = torch.zeros_like(gates)
        gates_mask.scatter_(dim=1, index=indices, src=values)
    else:
        gates_mask = gates.clone()

    if dp_group is not None:
        global_sum = gates_mask.sum(0, keepdim=True)
        global_count = torch.tensor(
            gates_mask.shape[0], device=gates_mask.device, dtype=global_sum.dtype
        )
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=dp_group)
        global_mean = global_sum / global_count
        global_diff_sq = (gates_mask - global_mean) ** 2
        global_total_var = global_diff_sq.sum()
        dist.all_reduce(global_total_var, op=dist.ReduceOp.SUM, group=dp_group)
        return (global_total_var / global_count).item()
    var = gates_mask - gates_mask.mean(0, keepdim=True)
    return (var**2).mean().item()


@torch.no_grad()
def expert_specialization(
    expert_output: torch.Tensor,
    scatter_index: torch.Tensor,
    start_idx=None,
    end_idx=None,
    dp_group=None,
) -> float:
    if start_idx is not None and end_idx is not None and start_idx == end_idx:
        global_sum = torch.tensor(0.0, device=expert_output.device, dtype=torch.float32)
        global_count = torch.tensor(0.0, device=expert_output.device, dtype=torch.float32)
    else:
        _in_range = (
            scatter_index >= start_idx
            if start_idx is not None
            else torch.ones_like(scatter_index, dtype=torch.bool)
        ) & (
            scatter_index < end_idx
            if end_idx is not None
            else torch.ones_like(scatter_index, dtype=torch.bool)
        )
        _scatter_index = torch.where(
            _in_range, scatter_index, torch.zeros_like(scatter_index)
        )
        reverted_output = expert_output[_scatter_index.flatten()].reshape(
            -1, _scatter_index.shape[1], expert_output.shape[-1]
        )
        masked_reverted_output = reverted_output * _in_range.unsqueeze(-1).to(
            reverted_output.dtype
        )
        masked_reverted_output = masked_reverted_output.to(torch.float32)
        dot_products = (
            masked_reverted_output.unsqueeze(1) * masked_reverted_output.unsqueeze(2)
        ).sum(-1)
        norms = torch.norm(masked_reverted_output, p=2, dim=-1) + 1e-9
        all_projection = (
            dot_products.abs() / norms.unsqueeze(1)
        )

        sum_all = all_projection.sum(dim=[1, 2])
        sum_diag = torch.diagonal(all_projection, dim1=1, dim2=2).sum(-1)
        non_zeros_in_topk = _in_range.to(reverted_output.dtype).sum(1)
        non_diag_count = non_zeros_in_topk * (non_zeros_in_topk - 1)
        sum_non_diag = sum_all - sum_diag
        mean_non_diag = sum_non_diag / non_diag_count
        mean_non_diag = mean_non_diag[torch.isfinite(mean_non_diag)]

        global_sum = mean_non_diag.sum().clone()
        global_count = torch.tensor(
            mean_non_diag.shape[0], device=global_sum.device, dtype=global_sum.dtype
        )

    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=dp_group)
    return (global_sum / global_count).item()


@torch.no_grad()
def router_weight_cosine_similarity(weight: torch.Tensor) -> float:
    """Average pairwise cosine similarity of expert columns.

    weight: (hidden_dim, num_experts)
    """
    weight = weight.to(torch.float32)
    norms = torch.norm(weight, p=2, dim=0, keepdim=True)
    weight_normalized = weight / (norms + 1e-9)
    cosine_similarity = weight_normalized.T @ weight_normalized
    n_expert = weight.shape[1]
    avg_cosine = (
        cosine_similarity.sum() - torch.trace(cosine_similarity)
    ) / (n_expert * (n_expert - 1))
    return avg_cosine.item()


@torch.no_grad()
def router_weight_centered_frob_sq(weight: torch.Tensor) -> float:
    """||W_c||_F^2 = sum_i ||w_i - mean_w||^2.

    The centered Frobenius energy of the router weight matrix.
    Softmax is invariant to a constant row shift, so routing
    only depends on W_c = (I - 1/E 11^T) W, not W. This is
    the isotropic-reference KL proxy (up to factor 1/(2E))
    for tracking router entropy collapse from weights alone.

    weight: (E, D) with E experts, D hidden dim.
    """
    weight = weight.to(torch.float32)
    row_sq = (weight * weight).sum(-1)
    mean_w = weight.mean(-2)
    return (row_sq.sum(-1) - weight.shape[-2] * (mean_w * mean_w).sum(-1)).item()
