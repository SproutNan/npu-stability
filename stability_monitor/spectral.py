import torch


@torch.no_grad()
def svdvals_via_eigvalsh(A: torch.Tensor) -> torch.Tensor:
    """Singular values via eigenvalues of A^T A (or A A^T if m < n).

    Uses eigvalsh which is better supported than general SVD on CANN/NPU.
    """
    A = A.to(torch.float32)
    m, n = A.shape
    if m < n:
        eigvals = torch.linalg.eigvalsh(A @ A.T)
    else:
        eigvals = torch.linalg.eigvalsh(A.T @ A)
    return torch.sqrt(torch.clamp(eigvals, min=0)).flip(0)


@torch.no_grad()
def stable_rank(A: torch.Tensor) -> torch.Tensor:
    """||A||_F^2 / ||A||_2^2."""
    sv = svdvals_via_eigvalsh(A)
    return sv.pow(2).sum() / sv[0].pow(2).clamp_min(1e-30)


@torch.no_grad()
def spectrum_entropy(A: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """exp(H) where H is Shannon entropy of p_i = sigma_i^alpha / sum sigma_j^alpha."""
    sv = svdvals_via_eigvalsh(A)
    p = sv.pow(alpha)
    p = p / p.sum().clamp_min(1e-30)
    H = -(p * p.clamp_min(1e-30).log()).sum()
    return H.exp()


@torch.no_grad()
def calc_all_svd_metrics(mat: torch.Tensor) -> dict:
    """Return dict with effective_rank, singular_spectrum, stable_rank,
    energy_ratio, condition for a matrix."""
    mat = mat.to(torch.float32)
    s = svdvals_via_eigvalsh(mat)

    p = s / (s.sum() + 1e-9)
    p_nonzero = p[p > 0]
    entropy = -(p_nonzero * p_nonzero.log()).sum()
    effective_rank = entropy.exp().item()

    p2 = s.pow(2) / (s.pow(2).sum() + 1e-9)
    p2_nonzero = p2[p2 > 0]
    entropy2 = -(p2_nonzero * p2_nonzero.log()).sum()
    singular_spectrum = entropy2.exp().item()

    s_sq = s.pow(2)
    energy_ratio = (s_sq[:5].sum() / (s_sq.sum() + 1e-9)).item()
    stable_rank_val = (s_sq.sum() / (s_sq[0] + 1e-9)).item()
    condition = (s[0] / (s[-1] + 1e-9)).item()

    return {
        "effective_rank": effective_rank,
        "singular_spectrum": singular_spectrum,
        "stable_rank": stable_rank_val,
        "energy_ratio": energy_ratio,
        "condition": condition,
    }


@torch.no_grad()
def calc_norm(mat: torch.Tensor) -> float:
    """Mean row-wise L2 norm."""
    mat = mat.to(torch.float32)
    return torch.norm(mat, dim=1).mean().item()
