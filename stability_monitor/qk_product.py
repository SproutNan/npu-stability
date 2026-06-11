import torch
from .spectral import svdvals_via_eigvalsh


@torch.no_grad()
def _eigh_U_S(A: torch.Tensor):
    """A: (r, d), r <= d. Returns (U, S) via eigh on A A^T."""
    A = A.to(torch.float32)
    eigvals, U = torch.linalg.eigh(A @ A.T)
    S = torch.sqrt(torch.clamp(eigvals.flip(0), min=0))
    return U.flip(1), S


@torch.no_grad()
def _core(P: torch.Tensor, Q: torch.Tensor):
    """Core matrix M = Sigma_P U_P^T U_Q Sigma_Q. sigma(M) = nonzero sigma(P^T Q)."""
    U_P, S_P = _eigh_U_S(P)
    U_Q, S_Q = _eigh_U_S(Q)
    return S_P[:, None] * (U_P.T @ U_Q) * S_Q[None, :]


@torch.no_grad()
def _get_svd_metrics_4tuple(weight: torch.Tensor):
    """Return (erank, spectrum, srank, condition) for a weight matrix."""
    s = svdvals_via_eigvalsh(weight)

    p = s / (s.sum() + 1e-9)
    p_nonzero = p[p > 0]
    entropy = -(p_nonzero * p_nonzero.log()).sum()
    erank = entropy.exp().item()

    p2 = s.pow(2) / (s.pow(2).sum() + 1e-9)
    p2_nonzero = p2[p2 > 0]
    entropy2 = -(p2_nonzero * p2_nonzero.log()).sum()
    spectrum = entropy2.exp().item()

    srank = (s.pow(2).sum() / (s[0].pow(2) + 1e-9)).item()
    condition = (s[0] / (s[-1] + 1e-9)).item()

    return erank, spectrum, srank, condition


@torch.no_grad()
def get_qk_product_metrics(
    dW_q: torch.Tensor,
    dW_k: torch.Tensor,
    W_q: torch.Tensor = None,
    W_k: torch.Tensor = None,
) -> dict:
    """Per-head QK-product delta metrics.

    dW_q, dW_k: (d_k, d) — update over [t, t+delta]
    W_q,  W_k:  (d_k, d), optional — base W at time t for d2, d1
    """
    dW_q = dW_q.to(torch.float32)
    dW_k = dW_k.to(torch.float32)
    if W_q is not None:
        W_q = W_q.to(torch.float32)
    if W_k is not None:
        W_k = W_k.to(torch.float32)

    out = {}

    # d3 = dW_q^T dW_k
    M3 = _core(dW_q, dW_k)
    out["erank_d3"], out["spectrum_d3"], out["srank_d3"], out["cond_d3"] = (
        _get_svd_metrics_4tuple(M3)
    )
    out["fnorm_d3"] = (M3 * M3).sum().sqrt().item()

    # c_qk = ||d3||_F^2 / (||dW_q||_F^2 * ||dW_k||_F^2)
    nq = (dW_q * dW_q).sum()
    nk = (dW_k * dW_k).sum()
    out["c_qk"] = ((M3 * M3).sum() / (nq * nk + 1e-18)).item()

    if W_q is None or W_k is None:
        return out

    # d2 = [dW_q; W_q]^T [W_k; dW_k]
    M2 = _core(torch.cat([dW_q, W_q], 0), torch.cat([W_k, dW_k], 0))
    out["erank_d2"], out["spectrum_d2"], out["srank_d2"], out["cond_d2"] = (
        _get_svd_metrics_4tuple(M2)
    )
    out["fnorm_d2"] = (M2 * M2).sum().sqrt().item()

    # d1 = [W_q+dW_q; -W_q]^T [W_k+dW_k; W_k]
    M1 = _core(
        torch.cat([W_q + dW_q, -W_q], 0), torch.cat([W_k + dW_k, W_k], 0)
    )
    out["erank_d1"], out["spectrum_d1"], out["srank_d1"], out["cond_d1"] = (
        _get_svd_metrics_4tuple(M1)
    )
    out["fnorm_d1"] = (M1 * M1).sum().sqrt().item()

    return out


@torch.no_grad()
def calc_qk_dw_metrics(
    qkv_current: torch.Tensor,
    qkv_prev: torch.Tensor,
    head_dim: int,
    proj_size: int,
    kv_size: int,
) -> dict:
    """Per-head QK delta metrics from concatenated QKV weights.

    Aggregates across heads using max.
    """
    q_rows = proj_size
    k_rows = kv_size

    q = qkv_current[:q_rows].view(-1, head_dim, qkv_current.shape[-1])
    k = qkv_current[q_rows : q_rows + k_rows].view(-1, head_dim, qkv_current.shape[-1])
    q_prev = qkv_prev[:q_rows].view(-1, head_dim, qkv_prev.shape[-1])
    k_prev = qkv_prev[q_rows : q_rows + k_rows].view(-1, head_dim, qkv_prev.shape[-1])

    n_q_heads = q.shape[0]
    n_k_heads = k.shape[0]
    group_size = n_q_heads // n_k_heads if n_k_heads > 0 else 1

    all_metrics = []
    for q_idx in range(n_q_heads):
        k_idx = q_idx // group_size
        dW_q = q[q_idx] - q_prev[q_idx]
        dW_k = k[k_idx] - k_prev[k_idx]
        W_q = q_prev[q_idx]
        W_k = k_prev[k_idx]
        m = get_qk_product_metrics(dW_q, dW_k, W_q, W_k)
        all_metrics.append(m)

    agg = {}
    for key in all_metrics[0].keys():
        vals = torch.tensor([m[key] for m in all_metrics])
        agg[key] = vals.max().item()
    return agg
