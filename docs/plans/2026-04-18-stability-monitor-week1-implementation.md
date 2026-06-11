# Stability Monitor — Week 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the framework-agnostic `stability_monitor` library, the `fault_injection` module, and the MindSpeed adapter, then run a smoke-test baseline to validate end-to-end integration.

**Architecture:** Three decoupled Python packages: (1) `stability_monitor` — pure PyTorch metrics library with a thin public API (tensors in, JSONL out); (2) `fault_injection` — mantissa-bit mask + reference Flash Attention with controllable $O \odot dO$ masking; (3) `mindspeed_adapter` — thin glue (layer discovery, training-loop hook, FA swap, CLI args) integrating into MindSpeedRun's existing patch-based system. Smoke test exercises all three end-to-end on a 500-step baseline.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, pytest, MindSpeed-LLM (from `MindSpeedRun-llm0121_gitcode/`), Megatron-Core v0.12.1, HuggingFace `datasets` + `transformers` (for Qwen3 tokenizer + FineWeb-Edu), Ascend NPU runtime for on-device tests.

**Source-of-truth references:**
- Design doc: `docs/plans/2026-04-18-mindspeed-stability-monitor-opensource-design.md`
- Theoretical framework: `大模型训练稳定性监控——从第一性原理出发.md`
- QK-product spec: `2026-04-16-qk-product-delta-monitor-design.md`

---

## Phase 1: Project scaffolding

### Task 1: Project structure + pyproject

**Files:**
- Create: `pyproject.toml`
- Create: `stability_monitor/__init__.py` (empty)
- Create: `fault_injection/__init__.py` (empty)
- Create: `mindspeed_adapter/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore`

**Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "stability-monitor-paper"
version = "0.1.0"
description = "Training-stability monitoring framework (paper artifact)"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-xdist>=3.0",
]
data = [
    "datasets>=2.14",
    "transformers>=4.40",
]

[tool.setuptools.packages.find]
include = ["stability_monitor*", "fault_injection*", "mindspeed_adapter*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Write .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
build/
dist/
logs/
data/*.bin
data/*.idx
checkpoints/
.vscode/
.DS_Store
```

**Step 3: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: `Successfully installed stability-monitor-paper-0.1.0`

**Step 4: Verify pytest discovers the empty test tree**

Run: `pytest --collect-only`
Expected: `collected 0 items`

**Step 5: Commit**

```bash
git init
git add pyproject.toml .gitignore stability_monitor/ fault_injection/ mindspeed_adapter/ tests/
git commit -m "chore: initial project scaffold"
```

---

## Phase 2: Core metrics — spectral

### Task 2: Stable rank

**Files:**
- Create: `stability_monitor/spectral.py`
- Test: `tests/test_spectral.py`

**Step 1: Write failing test**

```python
# tests/test_spectral.py
import torch
from stability_monitor.spectral import stable_rank

def test_stable_rank_identity_matrix():
    A = torch.eye(8)
    # σ_i = 1 for all i ⇒ srank = 8
    assert torch.isclose(stable_rank(A), torch.tensor(8.0), atol=1e-5)

def test_stable_rank_rank_one():
    u = torch.randn(8, 1)
    v = torch.randn(1, 4)
    A = u @ v
    # rank-1 ⇒ srank = 1
    assert torch.isclose(stable_rank(A), torch.tensor(1.0), atol=1e-4)

def test_stable_rank_from_svdvals():
    A = torch.randn(8, 5)
    sv = torch.linalg.svdvals(A)
    expected = (sv.pow(2).sum() / sv[0].pow(2))
    assert torch.isclose(stable_rank(A), expected, atol=1e-5)
```

**Step 2: Run, verify fail**

Run: `pytest tests/test_spectral.py -v`
Expected: ImportError `cannot import name 'stable_rank' from 'stability_monitor.spectral'`

**Step 3: Implement**

```python
# stability_monitor/spectral.py
import torch

def stable_rank(A: torch.Tensor) -> torch.Tensor:
    """||A||_F^2 / ||A||_2^2. Matrix A shape [m, n]. Returns scalar."""
    sv = torch.linalg.svdvals(A)
    return sv.pow(2).sum() / sv[0].pow(2).clamp_min(1e-30)

def stable_rank_from_svdvals(sv: torch.Tensor) -> torch.Tensor:
    """Same as stable_rank but caller already has the singular values."""
    return sv.pow(2).sum() / sv[0].pow(2).clamp_min(1e-30)
```

**Step 4: Verify pass**

Run: `pytest tests/test_spectral.py -v`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add stability_monitor/spectral.py tests/test_spectral.py
git commit -m "feat(spectral): stable rank"
```

---

### Task 3: Singular spectrum entropy

**Files:**
- Modify: `stability_monitor/spectral.py`
- Modify: `tests/test_spectral.py`

**Step 1: Write failing test**

Append to `tests/test_spectral.py`:

```python
from stability_monitor.spectral import spectrum_entropy

def test_spec_entropy_identity_max():
    # Uniform distribution over r singular values ⇒ entropy = r (when exp'd)
    A = torch.eye(16)
    se = spectrum_entropy(A, alpha=2)
    # spec_entropy normalization: exp(H) where H is Shannon entropy of p_i = σ_i^α / Σσ_j^α
    # uniform p_i = 1/16 ⇒ H = log(16), exp(H) = 16
    assert torch.isclose(se, torch.tensor(16.0), atol=1e-4)

def test_spec_entropy_rank_one():
    u = torch.randn(8, 1); v = torch.randn(1, 4)
    A = u @ v
    # single nonzero singular value ⇒ H = 0 ⇒ exp(H) = 1
    assert torch.isclose(spectrum_entropy(A, alpha=2), torch.tensor(1.0), atol=1e-4)

def test_spec_entropy_from_svdvals_identity():
    A = torch.randn(8, 5)
    sv = torch.linalg.svdvals(A)
    from stability_monitor.spectral import spectrum_entropy_from_svdvals
    assert torch.isclose(spectrum_entropy(A, alpha=2), spectrum_entropy_from_svdvals(sv, alpha=2), atol=1e-5)
```

**Step 2: Verify fail**

Run: `pytest tests/test_spectral.py -v`
Expected: ImportError on `spectrum_entropy`.

**Step 3: Implement**

Append to `stability_monitor/spectral.py`:

```python
def spectrum_entropy(A: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    exp(H) where H is Shannon entropy of p_i = σ_i^α / Σ σ_j^α.
    Returns scalar in [1, min(m, n)]. Sensitive to spectrum flatness.
    """
    sv = torch.linalg.svdvals(A)
    return spectrum_entropy_from_svdvals(sv, alpha=alpha)

def spectrum_entropy_from_svdvals(sv: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Same as spectrum_entropy but caller already has the singular values."""
    p = sv.pow(alpha)
    p = p / p.sum().clamp_min(1e-30)
    # use log(p + eps) to avoid -inf on zero singular values
    H = -(p * (p.clamp_min(1e-30)).log()).sum()
    return H.exp()
```

**Step 4: Verify pass**

Run: `pytest tests/test_spectral.py -v`
Expected: 6 passed.

**Step 5: Commit**

```bash
git add stability_monitor/spectral.py tests/test_spectral.py
git commit -m "feat(spectral): singular spectrum entropy"
```

---

## Phase 3: QK-product decomposition

### Task 4: Δ₃ core-matrix reduction

**Files:**
- Create: `stability_monitor/qk_product.py`
- Test: `tests/test_qk_product.py`

**Step 1: Write failing test**

```python
# tests/test_qk_product.py
import torch
from stability_monitor.qk_product import delta3_core_matrix, delta3_indicators

torch.manual_seed(0)

def test_delta3_core_matches_explicit():
    # Verify core-matrix reduction gives same nonzero singular values as full product
    d, d_k = 128, 16
    dWq = torch.randn(d, d_k)
    dWk = torch.randn(d, d_k)
    Uq, Sq, Vqt = torch.linalg.svd(dWq, full_matrices=False)
    Uk, Sk, Vkt = torch.linalg.svd(dWk, full_matrices=False)
    # Core matrix M = diag(Sq) @ Vq^T @ Vk @ diag(Sk)
    M = delta3_core_matrix(Sq, Vqt, Sk, Vkt)
    # Full Δ₃ = dWq @ dWk^T, shape [d, d]
    full = dWq @ dWk.T
    sv_full = torch.linalg.svdvals(full)
    sv_M = torch.linalg.svdvals(M)
    # Top-d_k singular values should match (rest of full are zero)
    assert torch.allclose(sv_full[:d_k], sv_M, atol=1e-4)

def test_delta3_indicators_rank_one_signal():
    d, d_k = 64, 8
    # Shared direction: dWq = u_q * v^T, dWk = u_k * v^T
    v = torch.randn(d_k); v = v / v.norm()
    u_q = torch.randn(d); u_q = u_q / u_q.norm()
    u_k = torch.randn(d); u_k = u_k / u_k.norm()
    dWq = 5.0 * u_q.unsqueeze(1) * v.unsqueeze(0)  # [d, d_k]
    dWk = 3.0 * u_k.unsqueeze(1) * v.unsqueeze(0)
    ind = delta3_indicators(dWq, dWk)
    # Δ₃ should be rank-1 ⇒ srank ≈ 1
    assert ind['srank'].item() < 1.01
    # F-norm should be 5 * 3 = 15 (exact in noiseless case)
    assert torch.isclose(ind['f_norm'], torch.tensor(15.0), atol=1e-3)
```

**Step 2: Verify fail**

Run: `pytest tests/test_qk_product.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# stability_monitor/qk_product.py
import torch
from .spectral import stable_rank_from_svdvals, spectrum_entropy_from_svdvals

def delta3_core_matrix(Sq: torch.Tensor, Vqt: torch.Tensor,
                       Sk: torch.Tensor, Vkt: torch.Tensor) -> torch.Tensor:
    """
    Given thin SVDs of ΔW_q, ΔW_k (shape [d, d_k] each):
        ΔW_q = U_q Σ_q V_q^T,  ΔW_k = U_k Σ_k V_k^T
    Return the d_k × d_k core matrix M s.t. nonzero singular values of
    Δ₃ = ΔW_q ΔW_k^T match those of M:
        M = diag(S_q) @ (V_q^T V_k) @ diag(S_k)
    Vqt, Vkt are V_q^T and V_k^T (shape [d_k, d_k]) from torch.linalg.svd.
    """
    # Vq^T @ Vk = Vqt @ Vkt^T
    G = Vqt @ Vkt.transpose(-2, -1)
    # diag(Sq) @ G @ diag(Sk)
    return Sq.unsqueeze(-1) * G * Sk.unsqueeze(-2)

def delta3_indicators(dWq: torch.Tensor, dWk: torch.Tensor) -> dict:
    """Spectral indicators of Δ₃ = ΔW_q ΔW_k^T via core-matrix reduction."""
    Uq, Sq, Vqt = torch.linalg.svd(dWq, full_matrices=False)
    Uk, Sk, Vkt = torch.linalg.svd(dWk, full_matrices=False)
    M = delta3_core_matrix(Sq, Vqt, Sk, Vkt)
    sv_M = torch.linalg.svdvals(M)
    return dict(
        srank=stable_rank_from_svdvals(sv_M),
        spec_entropy=spectrum_entropy_from_svdvals(sv_M, alpha=2.0),
        f_norm=sv_M.pow(2).sum().sqrt(),
        sigma1=sv_M[0],
    )
```

**Step 4: Verify pass**

Run: `pytest tests/test_qk_product.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add stability_monitor/qk_product.py tests/test_qk_product.py
git commit -m "feat(qk_product): delta3 core-matrix reduction"
```

---

### Task 5: c_qk normalized coupling coefficient

**Files:**
- Modify: `stability_monitor/qk_product.py`
- Modify: `tests/test_qk_product.py`

**Step 1: Write failing test**

Append to `tests/test_qk_product.py`:

```python
from stability_monitor.qk_product import c_qk

def test_c_qk_bounds():
    d, d_k = 64, 8
    dWq = torch.randn(d, d_k); dWk = torch.randn(d, d_k)
    c = c_qk(dWq, dWk)
    assert 0 <= c.item() <= 1 + 1e-6  # tolerance for fp

def test_c_qk_shared_rank1_is_one():
    d, d_k = 64, 8
    v = torch.randn(d_k); v = v / v.norm()
    u_q = torch.randn(d); u_k = torch.randn(d)
    dWq = u_q.unsqueeze(1) * v.unsqueeze(0) * 5.0
    dWk = u_k.unsqueeze(1) * v.unsqueeze(0) * 3.0
    # When row-spaces are perfectly aligned (both are rank-1 with the same v), c_qk = 1
    assert torch.isclose(c_qk(dWq, dWk), torch.tensor(1.0), atol=1e-5)

def test_c_qk_orthogonal_row_spaces_is_zero():
    d_k = 4
    dWq = torch.zeros(8, d_k); dWq[:, 0] = 1.0  # row-space = span(e_0)
    dWk = torch.zeros(8, d_k); dWk[:, 1] = 1.0  # row-space = span(e_1)
    assert torch.isclose(c_qk(dWq, dWk), torch.tensor(0.0), atol=1e-6)

def test_c_qk_iid_noise_approx_one_over_dk():
    # Under iid Gaussian, E[c_qk] ≈ 1/d_k
    d, d_k = 512, 32
    trials = 20
    vals = []
    for _ in range(trials):
        dWq = torch.randn(d, d_k); dWk = torch.randn(d, d_k)
        vals.append(c_qk(dWq, dWk).item())
    mean = sum(vals) / len(vals)
    # Loose bound: within factor 2 of 1/d_k
    assert 0.5 / d_k < mean < 2.0 / d_k
```

**Step 2: Verify fail**

Run: `pytest tests/test_qk_product.py::test_c_qk_bounds -v`
Expected: ImportError.

**Step 3: Implement**

Append to `stability_monitor/qk_product.py`:

```python
def c_qk(dWq: torch.Tensor, dWk: torch.Tensor) -> torch.Tensor:
    """
    c_qk = ||dWq dWk^T||_F^2 / (||dWq||_F^2 ||dWk||_F^2)
         = tr(G_q G_k) / (tr(G_q) tr(G_k)),  G_* = dW_*^T dW_*
    ∈ [0, 1]. Full-subspace coherence indicator (not leading-direction-only).
    """
    Gq = dWq.transpose(-2, -1) @ dWq
    Gk = dWk.transpose(-2, -1) @ dWk
    num = (Gq * Gk).sum()              # tr(Gq @ Gk) = sum of elementwise product (both symmetric)
    den = Gq.diagonal().sum() * Gk.diagonal().sum()
    return num / den.clamp_min(1e-30)
```

**Step 4: Verify pass**

Run: `pytest tests/test_qk_product.py -v`
Expected: 6 passed.

**Step 5: Commit**

```bash
git add stability_monitor/qk_product.py tests/test_qk_product.py
git commit -m "feat(qk_product): c_qk normalized coupling coefficient"
```

---

### Task 6: Δ₁, Δ₂ exact-decomposition identity check

**Files:**
- Modify: `stability_monitor/qk_product.py`
- Modify: `tests/test_qk_product.py`

**Step 1: Write failing test**

Append to `tests/test_qk_product.py`:

```python
from stability_monitor.qk_product import delta1_exact, delta2_linearized

def test_delta1_equals_delta2_plus_delta3():
    # Exact identity: Δ₁ = Δ₂ + Δ₃ (bilinear Taylor, no remainder)
    d, d_k = 32, 4
    Wq = torch.randn(d, d_k); Wk = torch.randn(d, d_k)
    dWq = torch.randn(d, d_k) * 0.01
    dWk = torch.randn(d, d_k) * 0.01
    d1 = delta1_exact(Wq, Wk, dWq, dWk)
    d2 = delta2_linearized(Wq, Wk, dWq, dWk)
    d3 = dWq @ dWk.T
    residual = (d1 - d2 - d3).abs().max()
    assert residual < 1e-5
```

**Step 2: Verify fail**

Run: `pytest tests/test_qk_product.py::test_delta1_equals_delta2_plus_delta3 -v`
Expected: ImportError.

**Step 3: Implement**

Append to `stability_monitor/qk_product.py`:

```python
def delta1_exact(Wq: torch.Tensor, Wk: torch.Tensor,
                 dWq: torch.Tensor, dWk: torch.Tensor) -> torch.Tensor:
    """Δ₁ = (Wq+dWq)(Wk+dWk)^T - Wq Wk^T."""
    return (Wq + dWq) @ (Wk + dWk).transpose(-2, -1) - Wq @ Wk.transpose(-2, -1)

def delta2_linearized(Wq: torch.Tensor, Wk: torch.Tensor,
                      dWq: torch.Tensor, dWk: torch.Tensor) -> torch.Tensor:
    """Δ₂ = dWq Wk^T + Wq dWk^T."""
    return dWq @ Wk.transpose(-2, -1) + Wq @ dWk.transpose(-2, -1)
```

**Step 4: Verify pass**

Run: `pytest tests/test_qk_product.py -v`
Expected: 7 passed.

**Step 5: Commit**

```bash
git add stability_monitor/qk_product.py tests/test_qk_product.py
git commit -m "feat(qk_product): delta1 exact and delta2 linearized + identity test"
```

---

## Phase 4: Router metrics

### Task 7: Per-token expert entropy

**Files:**
- Create: `stability_monitor/router.py`
- Test: `tests/test_router.py`

**Step 1: Write failing test**

```python
# tests/test_router.py
import torch
import math
from stability_monitor.router import per_token_expert_entropy

def test_per_token_entropy_uniform_is_log_n():
    num_experts = 128
    # uniform: all probs = 1/N
    probs = torch.full((32, num_experts), 1.0 / num_experts)
    H = per_token_expert_entropy(probs)
    assert torch.isclose(H, torch.tensor(math.log(num_experts)), atol=1e-5)

def test_per_token_entropy_one_hot_is_zero():
    num_experts = 128
    probs = torch.zeros(32, num_experts); probs[:, 0] = 1.0
    H = per_token_expert_entropy(probs)
    assert H.item() < 1e-6
```

**Step 2: Verify fail**

Run: `pytest tests/test_router.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# stability_monitor/router.py
import torch

def per_token_expert_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [num_tokens, num_experts] softmax routing distribution.
    Returns mean over tokens of -Σ_j p_ij log p_ij. Natural log base.
    """
    p = probs.clamp_min(1e-30)
    H_per_token = -(p * p.log()).sum(dim=-1)
    return H_per_token.mean()
```

**Step 4: Verify pass**

Run: `pytest tests/test_router.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add stability_monitor/router.py tests/test_router.py
git commit -m "feat(router): per-token expert entropy"
```

---

### Task 8: Router weight cosine similarity

**Files:**
- Modify: `stability_monitor/router.py`
- Modify: `tests/test_router.py`

**Step 1: Write failing test**

Append to `tests/test_router.py`:

```python
from stability_monitor.router import router_weight_cosine_similarity

def test_router_cos_sim_orthogonal_is_zero():
    W = torch.eye(8)  # columns are orthonormal basis vectors
    sim = router_weight_cosine_similarity(W)
    assert abs(sim.item()) < 1e-6

def test_router_cos_sim_identical_columns_is_one():
    v = torch.randn(8, 1)
    W = v.expand(-1, 4).contiguous()
    sim = router_weight_cosine_similarity(W)
    assert torch.isclose(sim, torch.tensor(1.0), atol=1e-6)
```

**Step 2: Verify fail**

Run: `pytest tests/test_router.py -v`
Expected: ImportError.

**Step 3: Implement**

Append to `stability_monitor/router.py`:

```python
def router_weight_cosine_similarity(W: torch.Tensor) -> torch.Tensor:
    """
    W: [d, n] router weight. Returns mean over i ≠ j of cos(w_i, w_j).
    """
    # normalize columns
    Wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-30)
    # Gram matrix of normalized columns is all pairwise cosines
    G = Wn.transpose(-2, -1) @ Wn  # [n, n]
    n = G.shape[-1]
    off_diag_sum = G.sum() - G.diagonal().sum()
    return off_diag_sum / (n * (n - 1))
```

**Step 4: Verify pass**

Run: `pytest tests/test_router.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add stability_monitor/router.py tests/test_router.py
git commit -m "feat(router): weight cosine similarity"
```

---

### Task 9: Load Gini & expert specialization

**Files:**
- Modify: `stability_monitor/router.py`
- Modify: `tests/test_router.py`

**Step 1: Write failing test**

Append to `tests/test_router.py`:

```python
from stability_monitor.router import load_gini, expert_specialization_variance

def test_load_gini_uniform_is_zero():
    # Perfectly even expert usage: Gini = 0
    counts = torch.full((128,), 100.0)
    assert load_gini(counts).item() < 1e-6

def test_load_gini_one_expert_used_is_max():
    counts = torch.zeros(128); counts[0] = 1000.0
    g = load_gini(counts).item()
    # max Gini for n experts is (n-1)/n
    assert g > 0.9

def test_expert_specialization_variance_dims():
    # [num_experts, num_token_types] histogram
    hist = torch.rand(128, 10)
    v = expert_specialization_variance(hist)
    # per-expert KL vs uniform, averaged
    assert v.dim() == 0
    assert v.item() >= 0
```

**Step 2: Verify fail**

Run: `pytest tests/test_router.py -v`
Expected: ImportError.

**Step 3: Implement**

Append to `stability_monitor/router.py`:

```python
def load_gini(token_counts_per_expert: torch.Tensor) -> torch.Tensor:
    """
    Gini coefficient of per-expert token counts. 0 = uniform, ≈1 = single expert.
    token_counts_per_expert: [num_experts]
    """
    c = token_counts_per_expert.float()
    n = c.numel()
    c_sorted, _ = c.sort()
    idx = torch.arange(1, n + 1, device=c.device, dtype=c.dtype)
    # Gini = (2 Σ i * c_sorted[i]) / (n Σ c) - (n+1)/n
    total = c_sorted.sum().clamp_min(1e-30)
    return (2 * (idx * c_sorted).sum()) / (n * total) - (n + 1) / n

def expert_specialization_variance(expert_by_type_hist: torch.Tensor) -> torch.Tensor:
    """
    Mean over experts of KL(p_expert || uniform_over_types).
    Higher = more specialized (each expert sees a narrower type distribution).
    expert_by_type_hist: [num_experts, num_token_types] (non-negative)
    """
    p = expert_by_type_hist / expert_by_type_hist.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    p = p.clamp_min(1e-30)
    num_types = p.shape[-1]
    kl_per_expert = (p * (p.log() - math.log(1.0 / num_types))).sum(dim=-1)
    return kl_per_expert.mean()
```

Also ensure `import math` is at the top of `stability_monitor/router.py`.

**Step 4: Verify pass**

Run: `pytest tests/test_router.py -v`
Expected: 7 passed.

**Step 5: Commit**

```bash
git add stability_monitor/router.py tests/test_router.py
git commit -m "feat(router): load Gini + expert specialization variance"
```

---

## Phase 5: Orchestration

### Task 10: MonitorConfig dataclass

**Files:**
- Create: `stability_monitor/config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing test**

```python
# tests/test_config.py
from stability_monitor.config import MonitorConfig

def test_default_config():
    cfg = MonitorConfig()
    assert cfg.router_cadence == 10
    assert cfg.delta_w_cadence == 50
    assert cfg.w_cadence == 500
    assert cfg.delta_window == 50
    assert cfg.per_head is True

def test_custom_config():
    cfg = MonitorConfig(router_cadence=5, output_path="/tmp/out.jsonl")
    assert cfg.router_cadence == 5
    assert cfg.output_path == "/tmp/out.jsonl"
```

**Step 2: Verify fail**

Run: `pytest tests/test_config.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# stability_monitor/config.py
from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class MonitorConfig:
    # Cadence (steps)
    router_cadence: int = 10
    delta_w_cadence: int = 50
    w_cadence: int = 500
    delta_window: int = 50  # δ for ΔW = W_t - W_{t-δ}

    # Per-head decomposition for QK
    per_head: bool = True
    head_aggregation: Tuple[str, ...] = ("mean", "max", "p95")

    # I/O
    output_path: Optional[str] = None  # JSONL path; None = no file write

    # SVD settings (escape hatch for performance)
    use_svdvals_only_when_possible: bool = True
```

**Step 4: Verify pass**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add stability_monitor/config.py tests/test_config.py
git commit -m "feat(config): MonitorConfig dataclass"
```

---

### Task 11: WeightTracker — snapshot and ΔW

**Files:**
- Create: `stability_monitor/tracker.py`
- Test: `tests/test_tracker.py`

**Step 1: Write failing test**

```python
# tests/test_tracker.py
import torch
from stability_monitor.tracker import WeightTracker

def test_tracker_delta_after_window():
    tracker = WeightTracker(delta_window=3)
    W = [torch.tensor([1.0]), torch.tensor([1.5]), torch.tensor([2.5]), torch.tensor([5.0])]
    # At step 0, no prior → delta is None
    assert tracker.update("W", W[0], step=0) is None
    assert tracker.update("W", W[1], step=1) is None
    assert tracker.update("W", W[2], step=2) is None
    # At step 3, we have W[0] as 3-step-old snapshot
    delta = tracker.update("W", W[3], step=3)
    assert torch.isclose(delta, torch.tensor([4.0]))  # W[3] - W[0]

def test_tracker_multiple_names_independent():
    tracker = WeightTracker(delta_window=1)
    tracker.update("Wq", torch.tensor([1.0]), step=0)
    tracker.update("Wk", torch.tensor([2.0]), step=0)
    dq = tracker.update("Wq", torch.tensor([3.0]), step=1)
    dk = tracker.update("Wk", torch.tensor([10.0]), step=1)
    assert torch.isclose(dq, torch.tensor([2.0]))
    assert torch.isclose(dk, torch.tensor([8.0]))
```

**Step 2: Verify fail**

Run: `pytest tests/test_tracker.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# stability_monitor/tracker.py
import torch
from typing import Optional, Dict, Deque
from collections import deque

class WeightTracker:
    """
    Maintains a rolling snapshot per named tensor, with window δ.
    update(name, W, step) returns ΔW = W - W_{step-δ} if available, else None.

    Storage: one clone per name per window-boundary sample.
    """
    def __init__(self, delta_window: int):
        assert delta_window >= 1
        self.delta_window = delta_window
        self._history: Dict[str, Deque] = {}  # name → deque of (step, tensor)

    def update(self, name: str, W: torch.Tensor, step: int) -> Optional[torch.Tensor]:
        dq = self._history.setdefault(name, deque())
        # Find snapshot with step <= current - delta_window
        while dq and dq[0][0] < step - self.delta_window:
            dq.popleft()
        delta = None
        if dq and dq[0][0] == step - self.delta_window:
            delta = W.detach() - dq[0][1]
        # Record current snapshot (detached clone to avoid autograd retention)
        dq.append((step, W.detach().clone()))
        return delta
```

**Step 4: Verify pass**

Run: `pytest tests/test_tracker.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add stability_monitor/tracker.py tests/test_tracker.py
git commit -m "feat(tracker): WeightTracker with delta-window snapshots"
```

---

### Task 12: JSONL logging sink

**Files:**
- Create: `stability_monitor/logging.py`
- Test: `tests/test_logging.py`

**Step 1: Write failing test**

```python
# tests/test_logging.py
import json
from pathlib import Path
from stability_monitor.logging import JsonlLogger

def test_jsonl_logger_writes_and_closes(tmp_path):
    path = tmp_path / "out.jsonl"
    logger = JsonlLogger(str(path))
    logger.log({"step": 5, "metric": "srank", "value": 3.14})
    logger.log({"step": 10, "metric": "entropy", "value": 1.23})
    logger.close()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["value"] == 3.14
    assert json.loads(lines[1])["metric"] == "entropy"

def test_jsonl_logger_noop_when_path_none():
    # No path ⇒ no file, no error
    logger = JsonlLogger(None)
    logger.log({"step": 5, "value": 1.0})
    logger.close()
```

**Step 2: Verify fail**

Run: `pytest tests/test_logging.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# stability_monitor/logging.py
import json
from typing import Optional, Dict, Any

class JsonlLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None
        if path:
            self._fh = open(path, "a", buffering=1)  # line-buffered

    def log(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, default=_default) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _default(o):
    # torch.Tensor scalar → Python float
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.tolist()
    except Exception:
        pass
    raise TypeError(f"Type not serializable: {type(o)}")
```

**Step 4: Verify pass**

Run: `pytest tests/test_logging.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add stability_monitor/logging.py tests/test_logging.py
git commit -m "feat(logging): JSONL sink with tensor scalar support"
```

---

### Task 13: Monitor public API — attention registration

**Files:**
- Create: `stability_monitor/monitor.py`
- Modify: `stability_monitor/__init__.py`
- Test: `tests/test_monitor_attention.py`

**Step 1: Write failing test**

```python
# tests/test_monitor_attention.py
import json
import torch
from pathlib import Path
from stability_monitor import Monitor, MonitorConfig

def test_monitor_attention_emits_deltaw_metrics(tmp_path):
    cfg = MonitorConfig(
        router_cadence=10, delta_w_cadence=2, w_cadence=100, delta_window=2,
        per_head=False, output_path=str(tmp_path / "out.jsonl"),
    )
    monitor = Monitor(cfg)

    # Fake attention layer: Wq [d, d_k_q], Wk [d, d_k_k]
    # Using one head, so num_q_heads=1, head_dim = d_k, num_kv_heads=1.
    W_state = {"Wq": torch.randn(32, 4), "Wk": torch.randn(32, 4)}
    monitor.register_attention(
        layer_id="L0.attn",
        wq_accessor=lambda: W_state["Wq"],
        wk_accessor=lambda: W_state["Wk"],
        num_q_heads=1, num_kv_heads=1, head_dim=4,
    )

    # Step 0, 1, 2 — at step 2, a delta window is available
    for step in range(3):
        W_state["Wq"] = W_state["Wq"] + 0.01 * torch.randn_like(W_state["Wq"])
        W_state["Wk"] = W_state["Wk"] + 0.01 * torch.randn_like(W_state["Wk"])
        monitor.step(step)

    monitor.close()
    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().strip().split("\n")]
    # Expect at least one record with Δ₃ indicators at step 2
    d3_records = [r for r in lines if r.get("metric_group") == "delta3" and r.get("step") == 2]
    assert len(d3_records) >= 1
    assert "srank" in d3_records[0]
    assert "c_qk" in d3_records[0]
```

**Step 2: Verify fail**

Run: `pytest tests/test_monitor_attention.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# stability_monitor/monitor.py
import torch
from typing import Callable, List, Optional
from dataclasses import dataclass, field
from .config import MonitorConfig
from .tracker import WeightTracker
from .logging import JsonlLogger
from .spectral import stable_rank, spectrum_entropy
from .qk_product import delta3_indicators, c_qk

@dataclass
class _AttentionLayerReg:
    layer_id: str
    wq_accessor: Callable[[], torch.Tensor]
    wk_accessor: Callable[[], torch.Tensor]
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

@dataclass
class _RouterReg:
    layer_id: str
    router_weight_accessor: Callable[[], torch.Tensor]
    probs_hook: Callable[[], Optional[torch.Tensor]]
    assignments_hook: Callable[[], Optional[torch.Tensor]]
    num_experts: int
    top_k: int

class Monitor:
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.tracker = WeightTracker(delta_window=cfg.delta_window)
        self.logger = JsonlLogger(cfg.output_path)
        self._attn: List[_AttentionLayerReg] = []
        self._router: List[_RouterReg] = []

    def register_attention(self, layer_id, wq_accessor, wk_accessor,
                            num_q_heads, num_kv_heads, head_dim):
        self._attn.append(_AttentionLayerReg(
            layer_id, wq_accessor, wk_accessor,
            num_q_heads, num_kv_heads, head_dim,
        ))

    def register_router(self, layer_id, router_weight_accessor,
                         probs_hook, assignments_hook, num_experts, top_k):
        self._router.append(_RouterReg(
            layer_id, router_weight_accessor, probs_hook, assignments_hook,
            num_experts, top_k,
        ))

    def step(self, step: int) -> None:
        if step % self.cfg.delta_w_cadence == 0:
            self._delta_w_step(step)
        # router + w tiers: added in subsequent tasks

    def _delta_w_step(self, step: int) -> None:
        for reg in self._attn:
            Wq = reg.wq_accessor()
            Wk = reg.wk_accessor()
            dWq = self.tracker.update(f"{reg.layer_id}.Wq", Wq, step)
            dWk = self.tracker.update(f"{reg.layer_id}.Wk", Wk, step)
            if dWq is None or dWk is None:
                continue
            # Per-matrix srank + spec_entropy + f_norm (full matrix, no per-head yet)
            for name, dW in (("Wq", dWq), ("Wk", dWk)):
                self.logger.log({
                    "step": step,
                    "metric_group": "delta_w",
                    "layer_id": reg.layer_id,
                    "matrix": name,
                    "srank": stable_rank(dW).item(),
                    "spec_entropy": spectrum_entropy(dW, alpha=2.0).item(),
                    "f_norm": dW.norm().item(),
                })
            # Δ₃ indicators + c_qk (one per attention layer; per-head added later)
            ind = delta3_indicators(dWq, dWk)
            self.logger.log({
                "step": step,
                "metric_group": "delta3",
                "layer_id": reg.layer_id,
                "srank": ind["srank"].item(),
                "spec_entropy": ind["spec_entropy"].item(),
                "f_norm": ind["f_norm"].item(),
                "c_qk": c_qk(dWq, dWk).item(),
            })

    def close(self) -> None:
        self.logger.close()
```

And update `stability_monitor/__init__.py`:

```python
from .config import MonitorConfig
from .monitor import Monitor
```

**Step 4: Verify pass**

Run: `pytest tests/test_monitor_attention.py -v`
Expected: 1 passed.

**Step 5: Commit**

```bash
git add stability_monitor/monitor.py stability_monitor/__init__.py tests/test_monitor_attention.py
git commit -m "feat(monitor): public API + attention ΔW + Δ₃ tier"
```

---

### Task 14: Monitor — per-head Δ₃ decomposition

**Files:**
- Modify: `stability_monitor/monitor.py`
- Modify: `tests/test_monitor_attention.py`

**Step 1: Write failing test**

Append to `tests/test_monitor_attention.py`:

```python
def test_monitor_per_head_delta3(tmp_path):
    cfg = MonitorConfig(
        router_cadence=10, delta_w_cadence=2, w_cadence=100, delta_window=2,
        per_head=True, output_path=str(tmp_path / "out.jsonl"),
    )
    monitor = Monitor(cfg)

    num_q_heads, num_kv_heads, head_dim = 4, 2, 8
    d = 32
    Wq = torch.randn(num_q_heads * head_dim, d)  # [Hq*dk, d]
    Wk = torch.randn(num_kv_heads * head_dim, d)  # [Hk*dk, d]
    state = {"Wq": Wq, "Wk": Wk}
    monitor.register_attention(
        layer_id="L0.attn",
        wq_accessor=lambda: state["Wq"],
        wk_accessor=lambda: state["Wk"],
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    for step in range(3):
        state["Wq"] = state["Wq"] + 0.01 * torch.randn_like(state["Wq"])
        state["Wk"] = state["Wk"] + 0.01 * torch.randn_like(state["Wk"])
        monitor.step(step)
    monitor.close()

    import json
    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().strip().split("\n")]
    per_head = [r for r in lines if r.get("metric_group") == "delta3_per_head" and r.get("step") == 2]
    # One record per query head (GQA: num_q_heads query heads share num_kv_heads key heads)
    assert len(per_head) == num_q_heads
    heads_seen = {r["head"] for r in per_head}
    assert heads_seen == set(range(num_q_heads))
```

**Step 2: Verify fail**

Run: `pytest tests/test_monitor_attention.py::test_monitor_per_head_delta3 -v`
Expected: FAIL (only 0 records match).

**Step 3: Implement**

Replace `_delta_w_step` in `stability_monitor/monitor.py` with an extended version:

```python
    def _delta_w_step(self, step: int) -> None:
        for reg in self._attn:
            Wq = reg.wq_accessor()
            Wk = reg.wk_accessor()
            dWq = self.tracker.update(f"{reg.layer_id}.Wq", Wq, step)
            dWk = self.tracker.update(f"{reg.layer_id}.Wk", Wk, step)
            if dWq is None or dWk is None:
                continue
            # Per-matrix
            for name, dW in (("Wq", dWq), ("Wk", dWk)):
                self.logger.log({
                    "step": step, "metric_group": "delta_w",
                    "layer_id": reg.layer_id, "matrix": name,
                    "srank": stable_rank(dW).item(),
                    "spec_entropy": spectrum_entropy(dW, alpha=2.0).item(),
                    "f_norm": dW.norm().item(),
                })
            # Aggregate Δ₃ (flat)
            ind = delta3_indicators(dWq, dWk)
            self.logger.log({
                "step": step, "metric_group": "delta3",
                "layer_id": reg.layer_id,
                "srank": ind["srank"].item(),
                "spec_entropy": ind["spec_entropy"].item(),
                "f_norm": ind["f_norm"].item(),
                "c_qk": c_qk(dWq, dWk).item(),
            })
            # Per-head Δ₃ (GQA: each query head uses its group's key head)
            if self.cfg.per_head:
                self._delta3_per_head(step, reg, dWq, dWk)

    def _delta3_per_head(self, step, reg, dWq, dWk):
        # dWq shape: [Hq*dk, d], dWk shape: [Hk*dk, d]
        Hq, Hk, dk = reg.num_q_heads, reg.num_kv_heads, reg.head_dim
        dWq_h = dWq.view(Hq, dk, -1)  # [Hq, dk, d]
        dWk_h = dWk.view(Hk, dk, -1)  # [Hk, dk, d]
        group_size = Hq // Hk
        for h in range(Hq):
            g = h // group_size
            # Per-head ΔW has shape [dk, d]; we need [d, dk] for our API → transpose
            dWq_i = dWq_h[h].transpose(-2, -1)
            dWk_i = dWk_h[g].transpose(-2, -1)
            ind = delta3_indicators(dWq_i, dWk_i)
            self.logger.log({
                "step": step, "metric_group": "delta3_per_head",
                "layer_id": reg.layer_id, "head": h, "kv_head": g,
                "srank": ind["srank"].item(),
                "spec_entropy": ind["spec_entropy"].item(),
                "f_norm": ind["f_norm"].item(),
                "c_qk": c_qk(dWq_i, dWk_i).item(),
            })
```

**Step 4: Verify pass**

Run: `pytest tests/test_monitor_attention.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add stability_monitor/monitor.py tests/test_monitor_attention.py
git commit -m "feat(monitor): per-head delta3 decomposition with GQA grouping"
```

---

### Task 15: Monitor — router tier

**Files:**
- Modify: `stability_monitor/monitor.py`
- Test: `tests/test_monitor_router.py`

**Step 1: Write failing test**

```python
# tests/test_monitor_router.py
import json
import torch
from stability_monitor import Monitor, MonitorConfig

def test_monitor_router_tier(tmp_path):
    cfg = MonitorConfig(
        router_cadence=1, delta_w_cadence=1000, w_cadence=1000, delta_window=50,
        per_head=False, output_path=str(tmp_path / "out.jsonl"),
    )
    monitor = Monitor(cfg)

    num_experts, top_k = 16, 2
    probs = torch.softmax(torch.randn(64, num_experts), dim=-1)
    router_W = torch.randn(32, num_experts)
    top_idx = probs.topk(top_k, dim=-1).indices

    state = {"probs": probs, "W": router_W, "topk": top_idx}
    monitor.register_router(
        layer_id="L0.router",
        router_weight_accessor=lambda: state["W"],
        probs_hook=lambda: state["probs"],
        assignments_hook=lambda: state["topk"],
        num_experts=num_experts, top_k=top_k,
    )

    monitor.step(0)
    monitor.close()

    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().strip().split("\n")]
    router_recs = [r for r in lines if r.get("metric_group") == "router"]
    assert len(router_recs) == 1
    assert "per_token_entropy" in router_recs[0]
    assert "weight_cos_sim" in router_recs[0]
    assert "load_gini" in router_recs[0]
```

**Step 2: Verify fail**

Run: `pytest tests/test_monitor_router.py -v`
Expected: FAIL (router records not emitted).

**Step 3: Implement**

In `stability_monitor/monitor.py`, update the imports and the `step` + add `_router_step`:

```python
from .router import (
    per_token_expert_entropy,
    router_weight_cosine_similarity,
    load_gini,
)
```

```python
    def step(self, step: int) -> None:
        if step % self.cfg.router_cadence == 0:
            self._router_step(step)
        if step % self.cfg.delta_w_cadence == 0:
            self._delta_w_step(step)

    def _router_step(self, step: int) -> None:
        for reg in self._router:
            W = reg.router_weight_accessor()
            probs = reg.probs_hook()
            topk = reg.assignments_hook()
            if probs is None or topk is None:
                continue
            # Load counts: one-hot over experts from top_k assignments
            counts = torch.zeros(reg.num_experts, device=topk.device)
            counts.scatter_add_(0, topk.flatten(), torch.ones_like(topk.flatten(), dtype=counts.dtype))
            self.logger.log({
                "step": step, "metric_group": "router",
                "layer_id": reg.layer_id,
                "per_token_entropy": per_token_expert_entropy(probs).item(),
                "weight_cos_sim": router_weight_cosine_similarity(W).item(),
                "load_gini": load_gini(counts).item(),
            })
```

**Step 4: Verify pass**

Run: `pytest tests/test_monitor_router.py -v`
Expected: 1 passed.

**Step 5: Commit**

```bash
git add stability_monitor/monitor.py tests/test_monitor_router.py
git commit -m "feat(monitor): router metric tier"
```

---

### Task 16: Monitor — W tier + cos(W_t, W_0)

**Files:**
- Modify: `stability_monitor/monitor.py`
- Test: `tests/test_monitor_wtier.py`

**Step 1: Write failing test**

```python
# tests/test_monitor_wtier.py
import json
import torch
from stability_monitor import Monitor, MonitorConfig

def test_monitor_w_tier_emits_at_cadence(tmp_path):
    cfg = MonitorConfig(
        router_cadence=1000, delta_w_cadence=1000, w_cadence=1, delta_window=50,
        per_head=False, output_path=str(tmp_path / "out.jsonl"),
    )
    monitor = Monitor(cfg)
    Wq = torch.randn(32, 16); Wk = torch.randn(16, 16)
    state = {"Wq": Wq, "Wk": Wk}
    monitor.register_attention(
        layer_id="L0.attn",
        wq_accessor=lambda: state["Wq"],
        wk_accessor=lambda: state["Wk"],
        num_q_heads=1, num_kv_heads=1, head_dim=16,
    )
    monitor.step(0)
    monitor.close()
    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().strip().split("\n")]
    w_recs = [r for r in lines if r.get("metric_group") == "w"]
    assert any(r["matrix"] == "Wq" for r in w_recs)
    assert any("srank" in r for r in w_recs)
    assert any("cos_w_w0" in r for r in w_recs)  # must track initial snapshot
```

**Step 2: Verify fail**

Run: `pytest tests/test_monitor_wtier.py -v`
Expected: FAIL.

**Step 3: Implement**

Extend `Monitor` in `stability_monitor/monitor.py`:

```python
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.tracker = WeightTracker(delta_window=cfg.delta_window)
        self.logger = JsonlLogger(cfg.output_path)
        self._attn: List[_AttentionLayerReg] = []
        self._router: List[_RouterReg] = []
        self._w0: dict = {}  # name → initial W clone

    def step(self, step: int) -> None:
        if step % self.cfg.router_cadence == 0:
            self._router_step(step)
        if step % self.cfg.delta_w_cadence == 0:
            self._delta_w_step(step)
        if step % self.cfg.w_cadence == 0:
            self._w_step(step)

    def _w_step(self, step: int) -> None:
        for reg in self._attn:
            for name, W in (("Wq", reg.wq_accessor()), ("Wk", reg.wk_accessor())):
                key = f"{reg.layer_id}.{name}"
                W0 = self._w0.setdefault(key, W.detach().clone())
                cos_w_w0 = (W.flatten() @ W0.flatten() /
                            (W.norm().clamp_min(1e-30) * W0.norm().clamp_min(1e-30)))
                self.logger.log({
                    "step": step, "metric_group": "w",
                    "layer_id": reg.layer_id, "matrix": name,
                    "srank": stable_rank(W).item(),
                    "spec_entropy": spectrum_entropy(W, alpha=2.0).item(),
                    "f_norm": W.norm().item(),
                    "cos_w_w0": cos_w_w0.item(),
                })
```

**Step 4: Verify pass**

Run: `pytest tests/test_monitor_wtier.py -v`
Expected: 1 passed.

**Step 5: Commit**

```bash
git add stability_monitor/monitor.py tests/test_monitor_wtier.py
git commit -m "feat(monitor): W tier with cos(W_t, W_0) tracking"
```

---

## Phase 6: Gauge invariance test (doubles as E5 evidence)

### Task 17: Gauge invariance empirical test

**Files:**
- Create: `tests/test_gauge_invariance.py`

**Step 1: Write failing test**

```python
# tests/test_gauge_invariance.py
import torch
from stability_monitor.spectral import stable_rank
from stability_monitor.qk_product import delta3_indicators, c_qk

torch.manual_seed(42)

def test_gauge_invariance_qk_product():
    """
    For any invertible R ∈ R^{d_k × d_k}: (Wq R, Wk R^{-T}) leaves Wq Wk^T unchanged,
    and likewise for ΔW transforms. Score-space deltas Δ₁/Δ₂/Δ₃ are invariant.
    Empirical verification.
    """
    d, d_k = 64, 8
    dWq = torch.randn(d, d_k); dWk = torch.randn(d, d_k)
    # Baseline indicators
    ind0 = delta3_indicators(dWq, dWk)
    c0 = c_qk(dWq, dWk)

    # Random invertible R
    R = torch.randn(d_k, d_k)
    R = R + d_k * torch.eye(d_k)  # bias toward invertibility
    Rinv_T = torch.linalg.inv(R).transpose(-2, -1)

    dWq_new = dWq @ R
    dWk_new = dWk @ Rinv_T
    ind1 = delta3_indicators(dWq_new, dWk_new)
    c1 = c_qk(dWq_new, dWk_new)

    assert torch.isclose(ind0["srank"], ind1["srank"], atol=1e-3)
    assert torch.isclose(ind0["f_norm"], ind1["f_norm"], atol=1e-3)
    assert torch.isclose(c0, c1, atol=1e-4)


def test_gauge_non_invariance_of_additive_stack():
    """
    In contrast, the additive-stack spectrum depends on R (unless R is orthogonal),
    confirming the §2.5 claim in the design doc.
    """
    d, d_k = 64, 8
    dWq = torch.randn(d, d_k); dWk = torch.randn(d, d_k)
    stack0 = torch.cat([dWq, dWk], dim=1)  # [d, 2*d_k]
    srank0 = stable_rank(stack0)

    R = torch.randn(d_k, d_k)
    R = R + d_k * torch.eye(d_k)
    Rinv_T = torch.linalg.inv(R).transpose(-2, -1)
    stack1 = torch.cat([dWq @ R, dWk @ Rinv_T], dim=1)
    srank1 = stable_rank(stack1)

    # Should differ (with overwhelming probability)
    assert abs(srank0 - srank1) > 0.1
```

**Step 2: Verify fail → pass (no new code needed)**

Run: `pytest tests/test_gauge_invariance.py -v`
Expected: 2 passed (if previous tasks implemented correctly).

**Step 3: Commit**

```bash
git add tests/test_gauge_invariance.py
git commit -m "test(gauge-invariance): empirical verification of QK-product invariance and additive-stack non-invariance"
```

---

## Phase 7: Fault injection — mantissa mask

### Task 18: Mantissa bit mask (bit-reinterpret)

**Files:**
- Create: `fault_injection/mantissa_mask.py`
- Test: `tests/test_mantissa_mask.py`

**Step 1: Write failing test**

```python
# tests/test_mantissa_mask.py
import torch
from fault_injection.mantissa_mask import mantissa_mask

def test_mantissa_mask_zero_bits_is_identity():
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    y = mantissa_mask(x, 0)
    assert torch.equal(x, y)

def test_mantissa_mask_clears_low_bits():
    # Construct a BF16 whose low 4 bits are nonzero; verify they're cleared.
    raw = torch.tensor([0b0_01111111_0101011], dtype=torch.int16)  # 1.0 + small mantissa bits
    x = raw.view(torch.bfloat16)
    y = mantissa_mask(x, 4)
    y_raw = y.view(torch.int16)
    assert (y_raw & 0b1111) == 0  # low 4 mantissa bits zeroed

def test_mantissa_mask_preserves_sign_and_exponent():
    x = torch.tensor([-1.25, 3.5, -7.0, 0.125], dtype=torch.bfloat16)
    y = mantissa_mask(x, 2)
    # Sign preserved
    assert torch.equal(torch.sign(x), torch.sign(y))
    # Exponent (same order of magnitude)
    assert torch.all(torch.abs(y - x) < torch.abs(x) * 0.5)  # small relative change

def test_mantissa_mask_is_bias_downward_in_magnitude():
    # Masking zero mantissa bits rounds toward zero ⇒ |y| ≤ |x| always
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.bfloat16)
    y = mantissa_mask(x, 4)
    assert torch.all(torch.abs(y) <= torch.abs(x) + 1e-3)
```

**Step 2: Verify fail**

Run: `pytest tests/test_mantissa_mask.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# fault_injection/mantissa_mask.py
import torch

def mantissa_mask(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Zero the bottom `num_bits` mantissa bits of a BF16 tensor.
    Produces a systematic truncation-toward-zero bias (matches Qiu & Yao 2025's
    FA low-precision fault characterization).

    Args:
        x: BF16 tensor.
        num_bits: number of low mantissa bits to clear (0..7 for BF16).

    Returns:
        BF16 tensor, same shape.
    """
    if num_bits == 0:
        return x
    assert x.dtype == torch.bfloat16, f"expected bfloat16, got {x.dtype}"
    assert 0 < num_bits <= 7, f"BF16 has 7 mantissa bits; num_bits={num_bits} invalid"
    # int16 view is safe: BF16 has same endianness/width as int16 in PyTorch
    mask_val = ~((1 << num_bits) - 1) & 0xFFFF
    mask = torch.tensor(mask_val, dtype=torch.int16, device=x.device)
    return (x.view(torch.int16) & mask).view(torch.bfloat16)
```

**Step 4: Verify pass**

Run: `pytest tests/test_mantissa_mask.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add fault_injection/mantissa_mask.py tests/test_mantissa_mask.py
git commit -m "feat(fault_injection): mantissa bit-mask with bit-reinterpret"
```

---

## Phase 8: Fault injection — reference FA

### Task 19: Reference FA forward (numerically verified against scaled_dot_product_attention)

**Files:**
- Create: `fault_injection/fa_reference.py`
- Test: `tests/test_fa_reference.py`

**Step 1: Write failing test**

```python
# tests/test_fa_reference.py
import math
import torch
from fault_injection.fa_reference import reference_attention

torch.manual_seed(0)

def test_reference_attention_matches_sdpa_no_mask():
    B, H, S, d_k = 2, 4, 16, 8
    Q = torch.randn(B, H, S, d_k)
    K = torch.randn(B, H, S, d_k)
    V = torch.randn(B, H, S, d_k)
    O_ref = reference_attention(Q, K, V, mask=None, mantissa_mask_bits=0)
    O_sdpa = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None)
    assert torch.allclose(O_ref, O_sdpa, atol=1e-5)

def test_reference_attention_with_causal_mask():
    B, H, S, d_k = 2, 4, 8, 16
    Q = torch.randn(B, H, S, d_k)
    K = torch.randn(B, H, S, d_k)
    V = torch.randn(B, H, S, d_k)
    # additive mask: -inf above diagonal
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    O_ref = reference_attention(Q, K, V, mask=mask, mantissa_mask_bits=0)
    O_sdpa = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert torch.allclose(O_ref, O_sdpa, atol=1e-5)
```

**Step 2: Verify fail**

Run: `pytest tests/test_fa_reference.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# fault_injection/fa_reference.py
import math
import torch
from .mantissa_mask import mantissa_mask


def reference_attention(Q, K, V, mask=None, mantissa_mask_bits: int = 0):
    """Forward-only wrapper for testing; delegates to custom Function for autograd."""
    return FAWithFaultBackward.apply(Q, K, V, mask, mantissa_mask_bits)


class FAWithFaultBackward(torch.autograd.Function):
    """
    Reference FA that exposes the O⊙dO intermediate in backward.
    When mantissa_mask_bits > 0, applies a systematic downward bias at that
    precise point — reproducing the Qiu & Yao 2025 FA low-precision fault.

    Shapes: Q, K, V of shape [B, H, S, d_k]. mask (optional) broadcast to [*, S, S].
    """
    @staticmethod
    def forward(ctx, Q, K, V, mask, mantissa_mask_bits):
        scale = 1.0 / math.sqrt(Q.shape[-1])
        S_ = (Q @ K.transpose(-2, -1)) * scale
        if mask is not None:
            S_ = S_ + mask
        P = torch.softmax(S_, dim=-1)
        O = P @ V
        ctx.save_for_backward(Q, K, V, P)
        ctx.scale = scale
        ctx.mantissa_mask_bits = mantissa_mask_bits
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, P = ctx.saved_tensors
        # CRITICAL fault-injection point: O⊙dO in BF16, mask low mantissa bits
        O = P @ V
        OdO = O * dO
        if ctx.mantissa_mask_bits > 0:
            # Only mask if BF16 input; promote back to input dtype
            in_dtype = OdO.dtype
            OdO_bf = OdO.to(torch.bfloat16)
            OdO_bf = mantissa_mask(OdO_bf, ctx.mantissa_mask_bits)
            OdO = OdO_bf.to(in_dtype)
        D = OdO.sum(dim=-1, keepdim=True)
        dP = dO @ V.transpose(-2, -1)
        dS = P * (dP - D)
        dQ = dS @ K * ctx.scale
        dK = dS.transpose(-2, -1) @ Q * ctx.scale
        dV = P.transpose(-2, -1) @ dO
        return dQ, dK, dV, None, None
```

**Step 4: Verify pass**

Run: `pytest tests/test_fa_reference.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add fault_injection/fa_reference.py tests/test_fa_reference.py
git commit -m "feat(fault_injection): reference FA with bit-mask injection at O⊙dO"
```

---

### Task 20: Reference FA backward gradient-correctness vs autograd-by-default

**Files:**
- Modify: `tests/test_fa_reference.py`

**Step 1: Write failing test**

Append to `tests/test_fa_reference.py`:

```python
def _attention_default_autograd(Q, K, V, mask=None):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S_ = (Q @ K.transpose(-2, -1)) * scale
    if mask is not None:
        S_ = S_ + mask
    P = torch.softmax(S_, dim=-1)
    return P @ V


def test_reference_attention_backward_matches_autograd_no_fault():
    # With mantissa_mask_bits=0, our custom backward should equal PyTorch autograd
    B, H, S, d_k = 1, 2, 8, 4
    Q = torch.randn(B, H, S, d_k, requires_grad=True)
    K = torch.randn(B, H, S, d_k, requires_grad=True)
    V = torch.randn(B, H, S, d_k, requires_grad=True)

    O1 = reference_attention(Q, K, V, mask=None, mantissa_mask_bits=0)
    grad_out = torch.randn_like(O1)
    dQ1, dK1, dV1 = torch.autograd.grad(O1, (Q, K, V), grad_out, retain_graph=False)

    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    O2 = _attention_default_autograd(Q2, K2, V2, mask=None)
    dQ2, dK2, dV2 = torch.autograd.grad(O2, (Q2, K2, V2), grad_out, retain_graph=False)

    assert torch.allclose(dQ1, dQ2, atol=1e-4)
    assert torch.allclose(dK1, dK2, atol=1e-4)
    assert torch.allclose(dV1, dV2, atol=1e-4)


def test_reference_attention_backward_fault_affects_grads():
    # With mantissa_mask_bits=4, the Q/K gradients must diverge from the clean ones.
    B, H, S, d_k = 1, 2, 8, 4
    Q = torch.randn(B, H, S, d_k, requires_grad=True, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, d_k, requires_grad=True, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, d_k, requires_grad=True, dtype=torch.bfloat16)

    grad_out = torch.randn_like(Q)

    O_clean = reference_attention(Q, K, V, mask=None, mantissa_mask_bits=0)
    dQ_clean, dK_clean, _ = torch.autograd.grad(O_clean, (Q, K, V), grad_out)

    O_faulted = reference_attention(Q, K, V, mask=None, mantissa_mask_bits=4)
    dQ_f, dK_f, _ = torch.autograd.grad(O_faulted, (Q, K, V), grad_out)

    # dV is unaffected (doesn't involve O⊙dO), but dQ and dK must differ
    assert not torch.allclose(dQ_clean.float(), dQ_f.float(), atol=1e-4)
    assert not torch.allclose(dK_clean.float(), dK_f.float(), atol=1e-4)
```

**Step 2: Verify pass**

Run: `pytest tests/test_fa_reference.py -v`
Expected: 4 passed.

**Step 3: Commit**

```bash
git add tests/test_fa_reference.py
git commit -m "test(fa_reference): backward parity without fault + divergence with fault"
```

---

## Phase 9: MindSpeed adapter

**Note on Phase 9 TDD:** Adapter code is integration glue; unit tests use mocks. Full end-to-end validation happens in Phase 11 (smoke test).

### Task 21: Layer discovery — QKV slicing

**Files:**
- Create: `mindspeed_adapter/hooks.py`
- Test: `tests/test_adapter_hooks.py`

**Step 1: Write failing test**

```python
# tests/test_adapter_hooks.py
import torch
from types import SimpleNamespace
from mindspeed_adapter.hooks import slice_qkv_weight

def test_slice_qkv_weight_gqa():
    Hq, Hk, dk, d = 16, 4, 64, 1024
    # fused QKV weight: [Hq*dk + 2*Hk*dk, d]
    qkv_total = Hq * dk + 2 * Hk * dk
    W = torch.randn(qkv_total, d)

    layer = SimpleNamespace(self_attention=SimpleNamespace(
        linear_qkv=SimpleNamespace(weight=W)))
    cfg = SimpleNamespace(num_q_heads=Hq, num_kv_heads=Hk, head_dim=dk)

    Wq, Wk = slice_qkv_weight(layer, cfg)
    assert Wq.shape == (Hq * dk, d)
    assert Wk.shape == (Hk * dk, d)
    assert torch.equal(Wq, W[:Hq * dk])
    assert torch.equal(Wk, W[Hq * dk:Hq * dk + Hk * dk])
```

**Step 2: Verify fail**

Run: `pytest tests/test_adapter_hooks.py -v`
Expected: ImportError.

**Step 3: Implement**

```python
# mindspeed_adapter/hooks.py
import torch
from typing import Any

def slice_qkv_weight(layer: Any, cfg: Any):
    """
    Slice Megatron-Core's fused linear_qkv weight into (Wq, Wk).
    Assumes GQA layout: [Hq*dk + Hk*dk + Hk*dk, d] ordered as [Q, K, V].
    cfg must expose num_q_heads, num_kv_heads, head_dim.
    Returns (Wq, Wk), both 2-D views of the fused weight.
    """
    W = layer.self_attention.linear_qkv.weight
    q_end = cfg.num_q_heads * cfg.head_dim
    k_end = q_end + cfg.num_kv_heads * cfg.head_dim
    return W[:q_end], W[q_end:k_end]


def register_all_attention(model, monitor, cfg):
    """Iterate model.decoder.layers and register attention tier with the monitor."""
    for idx, layer in enumerate(model.decoder.layers):
        monitor.register_attention(
            layer_id=f"layer.{idx}.attention",
            wq_accessor=lambda L=layer: slice_qkv_weight(L, cfg)[0],
            wk_accessor=lambda L=layer: slice_qkv_weight(L, cfg)[1],
            num_q_heads=cfg.num_q_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        )


def register_all_routers(model, monitor, cfg):
    """Register MoE router tier on layers >= cfg.first_k_dense_replace."""
    for idx, layer in enumerate(model.decoder.layers):
        if idx < getattr(cfg, "first_k_dense_replace", 0):
            continue
        moe = layer.mlp
        monitor.register_router(
            layer_id=f"layer.{idx}.router",
            router_weight_accessor=lambda M=moe: M.router.weight,
            probs_hook=lambda M=moe: getattr(M.router, "last_probs", None),
            assignments_hook=lambda M=moe: getattr(M.router, "last_topk_indices", None),
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
        )
```

**Step 4: Verify pass**

Run: `pytest tests/test_adapter_hooks.py -v`
Expected: 1 passed.

**Step 5: Commit**

```bash
git add mindspeed_adapter/hooks.py tests/test_adapter_hooks.py
git commit -m "feat(adapter): layer discovery with QKV slicing and router registration"
```

---

### Task 22: MoE router cache patch

**Files:**
- Create: `mindspeed_adapter/router_cache_patch.py`

**Step 1: Implement (no unit test — too framework-specific; covered by smoke test)**

```python
# mindspeed_adapter/router_cache_patch.py
"""
Megatron-Core's MoE TopKRouter computes probs+topk but does not store them.
We monkey-patch its forward to cache `last_probs` and `last_topk_indices`
for the stability monitor to read.
"""
import functools


def apply_router_cache_patch():
    """Call once after importing megatron; safe to call multiple times."""
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        raise RuntimeError("Megatron-Core TopKRouter not importable; check env.")

    if getattr(TopKRouter, "_stability_monitor_patched", False):
        return

    orig_forward = TopKRouter.forward

    @functools.wraps(orig_forward)
    def patched_forward(self, input_):
        out = orig_forward(self, input_)
        # out is typically (scores, indices) or a tuple including them.
        # Megatron-Core v0.12.1 returns (scores, indices).
        if isinstance(out, tuple) and len(out) >= 2:
            scores, indices = out[0], out[1]
            # Flatten to [tokens, ...] for monitor compatibility
            self.last_probs = scores.detach().reshape(-1, scores.shape[-1])
            self.last_topk_indices = indices.detach().reshape(-1, indices.shape[-1])
        return out

    TopKRouter.forward = patched_forward
    TopKRouter._stability_monitor_patched = True
```

**Step 2: Verify import-ability (no Megatron installed in this env; just import-check the module)**

Run: `python -c "from mindspeed_adapter.router_cache_patch import apply_router_cache_patch; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add mindspeed_adapter/router_cache_patch.py
git commit -m "feat(adapter): MoE router cache patch for last_probs/last_topk_indices"
```

---

### Task 23: Training-loop hook patch

**Files:**
- Create: `mindspeed_adapter/training_loop_patch.py`

**Step 1: Implement**

```python
# mindspeed_adapter/training_loop_patch.py
"""
Monkey-patch megatron.training.training.train_step to invoke monitor.step(iteration)
after the optimizer step completes.
"""
import functools
from typing import Any

_monitor_ref: Any = None


def install_training_loop_hook(monitor):
    """Register the monitor and patch train_step."""
    global _monitor_ref
    _monitor_ref = monitor

    from megatron.training import training as mg_training

    if getattr(mg_training.train_step, "_stability_monitor_patched", False):
        return

    orig_train_step = mg_training.train_step

    @functools.wraps(orig_train_step)
    def patched_train_step(*args, **kwargs):
        result = orig_train_step(*args, **kwargs)
        try:
            from megatron.training import get_args
            iteration = get_args().curr_iteration
            _monitor_ref.step(iteration)
        except Exception as e:
            # Never let monitoring break training.
            import sys
            print(f"[stability_monitor] step({iteration}) failed: {e}", file=sys.stderr)
        return result

    patched_train_step._stability_monitor_patched = True
    mg_training.train_step = patched_train_step
```

**Step 2: Verify import-ability**

Run: `python -c "from mindspeed_adapter.training_loop_patch import install_training_loop_hook; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add mindspeed_adapter/training_loop_patch.py
git commit -m "feat(adapter): train_step monkey-patch to call monitor.step"
```

---

### Task 24: FA swap patch for fault injection

**Files:**
- Create: `mindspeed_adapter/fa_patch.py`

**Step 1: Implement**

```python
# mindspeed_adapter/fa_patch.py
"""
When fault injection is enabled, replace MindSpeed's fused FA path with the
reference FA that exposes the O⊙dO masking point. Applied only on E1, A1, E7 runs.

MindSpeed's attention dispatch route varies between versions. This patch targets
torch.nn.functional.scaled_dot_product_attention at module-import time; MindSpeed
currently dispatches through SDPA in its default attention path.
"""
import functools
import math
import torch

from fault_injection.fa_reference import FAWithFaultBackward

_fault_bits: int = 0
_patched = False


def apply_fa_patch(mantissa_mask_bits: int):
    """Enable the reference-FA path with the given mask bits.

    Idempotent; subsequent calls update the bit count.
    """
    global _fault_bits, _patched
    _fault_bits = mantissa_mask_bits

    if _patched:
        return

    orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    @functools.wraps(orig_sdpa)
    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, **kwargs):
        # Construct additive mask if is_causal; otherwise use attn_mask as-is.
        S = query.shape[-2]
        mask = None
        if is_causal:
            mask = torch.triu(
                torch.full((S, S), float("-inf"), dtype=query.dtype, device=query.device),
                diagonal=1,
            )
        elif attn_mask is not None:
            mask = attn_mask
        return FAWithFaultBackward.apply(query, key, value, mask, _fault_bits)

    patched_sdpa._stability_monitor_patched = True
    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    _patched = True


def disable_fa_patch():
    """Restore unpatched SDPA (useful for test isolation)."""
    global _patched
    if _patched:
        # Cannot fully undo a monkey-patch reliably; best-effort: set bits=0
        globals()["_fault_bits"] = 0
```

**Step 2: Verify import-ability**

Run: `python -c "from mindspeed_adapter.fa_patch import apply_fa_patch; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add mindspeed_adapter/fa_patch.py
git commit -m "feat(adapter): FA patch swapping SDPA with reference FA under fault injection"
```

---

### Task 25: MindSpeed CLI arg extension

**Files:**
- Create: `mindspeed_adapter/cli_args.py`

**Step 1: Implement**

```python
# mindspeed_adapter/cli_args.py
"""
Append stability-monitor + fault-injection args to MindSpeed's arg parser.

Usage in the patched pretrain_gpt.py entry point:
    from mindspeed_adapter.cli_args import add_stability_monitor_args
    # After MindSpeed builds its parser:
    add_stability_monitor_args(parser)
"""

def add_stability_monitor_args(parser):
    g = parser.add_argument_group("stability-monitor")
    g.add_argument("--enable-stability-monitor", action="store_true")
    g.add_argument("--monitor-output-path", type=str, default=None)
    g.add_argument("--monitor-router-cadence", type=int, default=10)
    g.add_argument("--monitor-delta-w-cadence", type=int, default=50)
    g.add_argument("--monitor-w-cadence", type=int, default=500)
    g.add_argument("--monitor-delta-window", type=int, default=50)
    g.add_argument("--monitor-per-head", action="store_true", default=True)

    g2 = parser.add_argument_group("fault-injection")
    g2.add_argument("--inject-fa-fault", action="store_true")
    g2.add_argument("--fa-mantissa-mask-bits", type=int, default=0)
    return parser


def build_monitor_from_args(args):
    """Construct a Monitor from parsed MindSpeed CLI args."""
    from stability_monitor import Monitor, MonitorConfig
    cfg = MonitorConfig(
        router_cadence=args.monitor_router_cadence,
        delta_w_cadence=args.monitor_delta_w_cadence,
        w_cadence=args.monitor_w_cadence,
        delta_window=args.monitor_delta_window,
        per_head=args.monitor_per_head,
        output_path=args.monitor_output_path,
    )
    return Monitor(cfg)
```

**Step 2: Verify import-ability + basic invocation**

Run:
```
python -c "
import argparse
from mindspeed_adapter.cli_args import add_stability_monitor_args
p = argparse.ArgumentParser()
add_stability_monitor_args(p)
args = p.parse_args(['--enable-stability-monitor', '--monitor-output-path', '/tmp/x.jsonl'])
assert args.enable_stability_monitor
assert args.monitor_output_path == '/tmp/x.jsonl'
print('ok')
"
```
Expected: `ok`

**Step 3: Commit**

```bash
git add mindspeed_adapter/cli_args.py
git commit -m "feat(adapter): MindSpeed CLI arg extensions for monitor + fault injection"
```

---

## Phase 10: Data preparation + model config

### Task 26: FineWeb-Edu tokenization script

**Files:**
- Create: `experiments/data/tokenize_fineweb_edu.py`
- Create: `experiments/data/README.md`

**Step 1: Implement**

```python
# experiments/data/tokenize_fineweb_edu.py
"""
Download FineWeb-Edu sample-100BT + tokenize with Qwen3 tokenizer into
Megatron-format .bin/.idx shards for MindSpeed-LLM to consume.

Run (from repo root):
    pip install -e ".[data]"
    python experiments/data/tokenize_fineweb_edu.py \
        --output-dir data/fineweb_edu_qwen3 \
        --tokenizer Qwen/Qwen3-30B-A3B \
        --num-proc 32
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--num-proc", type=int, default=16)
    p.add_argument("--split", default="sample-100BT")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"Tokenizer vocab size: {tok.vocab_size}")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.split,
        split="train",
        num_proc=args.num_proc,
    )
    print(f"Loaded {len(ds):,} documents from {args.split}")

    # MindSpeed-LLM reads Megatron-format .bin/.idx. Use MindSpeed's prepocessor
    # by writing JSONL first, then invoking tools/preprocess_data.py.
    jsonl_path = out / "fineweb_edu_100bt.jsonl"
    print(f"Writing JSONL to {jsonl_path}")
    with jsonl_path.open("w") as f:
        for rec in ds:
            f.write('{"text": ' + repr(rec["text"]) + "}\n")
    print(f"Done. Now run MindSpeed-LLM preprocess_data.py to convert to .bin/.idx:")
    print(f"  python MindSpeed-LLM/tools/preprocess_data.py \\")
    print(f"      --input {jsonl_path} \\")
    print(f"      --output-prefix {out}/fineweb_edu \\")
    print(f"      --tokenizer-type PretrainedFromHF \\")
    print(f"      --tokenizer-name-or-path {args.tokenizer} \\")
    print(f"      --json-keys text --workers {args.num_proc}")


if __name__ == "__main__":
    main()
```

```markdown
# experiments/data/README.md

# Data preparation

## FineWeb-Edu 100BT + Qwen3 tokenizer

Two-step process:
1. Download + JSONL: `python tokenize_fineweb_edu.py --output-dir data/fineweb_edu_qwen3`
2. JSONL → .bin/.idx: invoke MindSpeed-LLM's `preprocess_data.py` (command printed at end of step 1).

Expected output: `data/fineweb_edu_qwen3/fineweb_edu_text_document.{bin,idx}`.

## Checksums

Record SHA256 of final .bin shards in `data/fineweb_edu_qwen3/CHECKSUMS` after preprocessing; include in paper reproducibility appendix.
```

**Step 2: Verify script is runnable (dry-run, --help)**

Run: `python experiments/data/tokenize_fineweb_edu.py --help`
Expected: usage message printed without error.

**Step 3: Commit**

```bash
git add experiments/data/tokenize_fineweb_edu.py experiments/data/README.md
git commit -m "feat(data): FineWeb-Edu sample-100BT tokenization driver"
```

---

### Task 27: Shrunk-Qwen3 model config YAML (primary)

**Files:**
- Create: `experiments/configs/model_shrunk_qwen3_small.yaml`
- Create: `experiments/configs/model_shrunk_qwen3_mini.yaml`

**Step 1: Implement primary config**

```yaml
# experiments/configs/model_shrunk_qwen3_small.yaml
# Primary model for E1a, E2-*, E3, E7, A1.
# ~2.8B total params / ~0.4B active.
model:
  architecture: qwen3_moe_shrunk
  hidden_size: 1024
  ffn_hidden_size: 768        # per-expert FFN intermediate
  num_layers: 16
  num_attention_heads: 16     # query heads
  num_key_value_heads: 4      # GQA
  head_dim: 64
  num_experts: 128
  num_shared_experts: 1
  moe_router_topk: 8
  moe_aux_loss_coeff: 1.0e-4
  moe_router_load_balancing_type: aux_loss
  first_k_dense_replace: 1
  activation: swiglu
  normalization: rmsnorm
  rope_base: 10000
  vocab_size: 151936          # Qwen3 tokenizer
  tokenizer: Qwen/Qwen3-30B-A3B

training:
  seq_length: 2048
  micro_batch_size: 4
  global_batch_size: 128
  lr: 3.0e-4
  min_lr: 3.0e-5
  lr_warmup_iters: 2000
  lr_decay_style: cosine
  train_iters: 40000          # baseline; fault runs override to 20000-25000
  optimizer: adamw
  adam_beta1: 0.9
  adam_beta2: 0.95
  weight_decay: 0.1
  init_method_std: 0.02
  clip_grad: 1.0
  precision: bf16
  seed: 1234

parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 8
  context_parallel_size: 1
  sequence_parallel: true

data:
  data_path: data/fineweb_edu_qwen3/fineweb_edu_text_document
  tokenizer_path: Qwen/Qwen3-30B-A3B
  split: "100,0,0"
```

**Step 2: Implement mini config**

```yaml
# experiments/configs/model_shrunk_qwen3_mini.yaml
# Mini model for E1b scale-ladder. ~0.4B total / ~0.08B active.
model:
  architecture: qwen3_moe_shrunk
  hidden_size: 512
  ffn_hidden_size: 384
  num_layers: 8
  num_attention_heads: 8
  num_key_value_heads: 2
  head_dim: 64
  num_experts: 64
  num_shared_experts: 1
  moe_router_topk: 4
  moe_aux_loss_coeff: 1.0e-4
  moe_router_load_balancing_type: aux_loss
  first_k_dense_replace: 1
  activation: swiglu
  normalization: rmsnorm
  rope_base: 10000
  vocab_size: 151936
  tokenizer: Qwen/Qwen3-30B-A3B

training:
  seq_length: 2048
  micro_batch_size: 8
  global_batch_size: 128
  lr: 3.0e-4
  min_lr: 3.0e-5
  lr_warmup_iters: 2000
  lr_decay_style: cosine
  train_iters: 25000
  optimizer: adamw
  adam_beta1: 0.9
  adam_beta2: 0.95
  weight_decay: 0.1
  init_method_std: 0.02
  clip_grad: 1.0
  precision: bf16
  seed: 1234

parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 8
  context_parallel_size: 1
  sequence_parallel: true

data:
  data_path: data/fineweb_edu_qwen3/fineweb_edu_text_document
  tokenizer_path: Qwen/Qwen3-30B-A3B
  split: "100,0,0"
```

**Step 3: Commit**

```bash
git add experiments/configs/model_shrunk_qwen3_small.yaml experiments/configs/model_shrunk_qwen3_mini.yaml
git commit -m "feat(experiments): shrunk-Qwen3 primary + mini model configs"
```

---

### Task 28: Fault config YAMLs (E1a, E2-LR40, E2-AUX, E2-GBS, E7, A1 variants)

**Files:**
- Create: `experiments/configs/faults/e1a_fa_mask4.yaml`
- Create: `experiments/configs/faults/e2_lr40.yaml`
- Create: `experiments/configs/faults/e2_aux_off.yaml`
- Create: `experiments/configs/faults/e2_gbs_small.yaml`
- Create: `experiments/configs/faults/e7_combined.yaml`
- Create: `experiments/configs/faults/a1_fa_mask{0,2,4,6}.yaml`

**Step 1: Implement (all are small override files)**

```yaml
# experiments/configs/faults/e1a_fa_mask4.yaml
extends: model_shrunk_qwen3_small.yaml
fault:
  type: fa_mantissa_mask
  mantissa_bits: 4
training_overrides:
  train_iters: 25000
run_id: e1a_fa_mask4
```

```yaml
# experiments/configs/faults/e2_lr40.yaml
extends: model_shrunk_qwen3_small.yaml
fault:
  type: hyperparameter
  description: lr 40x
training_overrides:
  lr: 1.2e-2
  train_iters: 20000
run_id: e2_lr40
```

```yaml
# experiments/configs/faults/e2_aux_off.yaml
extends: model_shrunk_qwen3_small.yaml
fault:
  type: hyperparameter
  description: aux-loss disabled
model_overrides:
  moe_aux_loss_coeff: 0.0
training_overrides:
  train_iters: 25000
run_id: e2_aux_off
```

```yaml
# experiments/configs/faults/e2_gbs_small.yaml
extends: model_shrunk_qwen3_small.yaml
fault:
  type: hyperparameter
  description: gbs 16
training_overrides:
  global_batch_size: 16
  train_iters: 25000
run_id: e2_gbs_small
```

```yaml
# experiments/configs/faults/e7_combined.yaml
extends: model_shrunk_qwen3_small.yaml
fault:
  type: combined
  fa_mantissa_bits: 4
  lr_multiplier: 40
training_overrides:
  lr: 1.2e-2
  train_iters: 15000
run_id: e7_combined
```

```yaml
# experiments/configs/faults/a1_fa_mask0.yaml
extends: model_shrunk_qwen3_small.yaml
fault:
  type: fa_mantissa_mask
  mantissa_bits: 0    # control run through reference FA path
training_overrides:
  train_iters: 25000
run_id: a1_fa_mask0
```

(Create `a1_fa_mask2.yaml` and `a1_fa_mask6.yaml` analogously; `a1_fa_mask4.yaml` is identical to `e1a_fa_mask4.yaml`, so link via the run_id convention rather than duplicating.)

**Step 2: Lint YAML (syntactic validity)**

Run: `python -c "import yaml, pathlib; [yaml.safe_load(p.read_text()) for p in pathlib.Path('experiments/configs/faults').glob('*.yaml')]; print('all valid')"`
Expected: `all valid`

**Step 3: Commit**

```bash
git add experiments/configs/faults/
git commit -m "feat(experiments): fault-injection config YAMLs for E1, E2, E7, A1"
```

---

## Phase 11: Smoke test — end-to-end validation

### Task 29: Reference-FA vs fused-FA numerical parity on NPU

**Files:**
- Create: `experiments/validation/fa_parity_test.py`

**Step 1: Implement**

```python
# experiments/validation/fa_parity_test.py
"""
Verify reference FA (no fault) produces loss curves within tolerance of MindSpeed's
fused FA over 1000 training steps on the primary model.
This test MUST pass on NPU before E1a / A1 runs are launched.

Procedure:
    1. Run 1000 steps with MindSpeed fused FA, log loss.
    2. Run 1000 steps with --inject-fa-fault --fa-mantissa-mask-bits 0 (reference FA, no fault).
    3. Compare loss curves: cumulative mean loss delta should be < 1%.

Outputs pass/fail to stdout and writes loss curves to experiments/validation/fa_parity.csv.
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-log", required=True, help="JSONL log of fused-FA baseline")
    p.add_argument("--reference-log", required=True, help="JSONL log of reference-FA run (mask_bits=0)")
    p.add_argument("--tolerance", type=float, default=0.01)
    args = p.parse_args()

    import json
    def load(path):
        return [json.loads(l) for l in Path(path).read_text().strip().split("\n")]
    b = load(args.baseline_log)
    r = load(args.reference_log)

    def losses(records):
        return [x["loss"] for x in records if "loss" in x]
    lb, lr = losses(b), losses(r)
    n = min(len(lb), len(lr))
    assert n >= 100, f"too few loss samples ({n})"

    mean_b = sum(lb[:n]) / n
    mean_r = sum(lr[:n]) / n
    rel_delta = abs(mean_r - mean_b) / mean_b
    print(f"Mean loss baseline={mean_b:.4f}, reference={mean_r:.4f}, rel_delta={rel_delta:.2%}")

    if rel_delta < args.tolerance:
        print(f"PASS: within {args.tolerance:.1%} tolerance.")
        return 0
    else:
        print(f"FAIL: exceeded {args.tolerance:.1%} tolerance.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Verify script importable**

Run: `python experiments/validation/fa_parity_test.py --help`
Expected: usage message.

**Step 3: Commit**

```bash
git add experiments/validation/fa_parity_test.py
git commit -m "feat(validation): reference FA vs fused FA parity checker"
```

---

### Task 30: Smoke-test launcher — 500-step baseline run

**Files:**
- Create: `experiments/run_scripts/smoke_test.sh`
- Create: `experiments/run_scripts/run_with_monitor.sh`

**Step 1: Implement run wrapper**

```bash
#!/bin/bash
# experiments/run_scripts/run_with_monitor.sh
# Drop-in wrapper around MindSpeedRun that enables the stability monitor.
#
# Usage:
#     bash run_with_monitor.sh <CONFIG_YAML> <OUTPUT_DIR> [IP_LIST] [INET_PREFIX]
#
# This layers onto the existing MindSpeedRun 128k_bf16_8p.sh pattern.

set -e
CONFIG_YAML="$1"
OUTPUT_DIR="$2"
IP_LIST="${3:-localhost}"
INET_PREFIX="${4:-}"

mkdir -p "$OUTPUT_DIR/logs"

MONITOR_ARGS="\
  --enable-stability-monitor \
  --monitor-output-path $OUTPUT_DIR/logs/monitor.jsonl \
  --monitor-router-cadence 10 \
  --monitor-delta-w-cadence 50 \
  --monitor-w-cadence 500 \
  --monitor-delta-window 50 \
"

# Parse fault section from CONFIG_YAML; if fault.type == fa_mantissa_mask, add fault flags.
FAULT_TYPE=$(python -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('fault',{}).get('type',''))")
if [[ "$FAULT_TYPE" == "fa_mantissa_mask" ]]; then
    BITS=$(python -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d['fault']['mantissa_bits'])")
    FAULT_ARGS="--inject-fa-fault --fa-mantissa-mask-bits $BITS"
elif [[ "$FAULT_TYPE" == "combined" ]]; then
    BITS=$(python -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d['fault']['fa_mantissa_bits'])")
    FAULT_ARGS="--inject-fa-fault --fa-mantissa-mask-bits $BITS"
else
    FAULT_ARGS=""
fi

# Call the existing MindSpeedRun launcher with extra args.
# The launcher is at MindSpeedRun-llm0121_gitcode/run/run.py; we pass MONITOR_ARGS + FAULT_ARGS
# through to pretrain_gpt.py. The actual model config is derived by the existing bash script
# from CONFIG_YAML (a wrapping script to be completed in a later task adapts our YAML → bash env).

echo "Launching training with:"
echo "  CONFIG_YAML=$CONFIG_YAML"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MONITOR_ARGS=$MONITOR_ARGS"
echo "  FAULT_ARGS=$FAULT_ARGS"

# The exact MindSpeedRun entry point depends on the IP_LIST/INET_PREFIX arguments
# and is invoked from the existing 128k_bf16_8p.sh pattern. Hand off to:
bash MindSpeedRun-llm0121_gitcode/scripts/qwen3_235b_no_swap/run.sh \
    "$IP_LIST" "$INET_PREFIX" \
    --config-yaml "$CONFIG_YAML" \
    --output-dir "$OUTPUT_DIR" \
    $MONITOR_ARGS $FAULT_ARGS \
    2>&1 | tee "$OUTPUT_DIR/logs/train.log"
```

```bash
#!/bin/bash
# experiments/run_scripts/smoke_test.sh
# End-to-end 500-step baseline run to validate the full pipeline.
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="$ROOT/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Use primary model config but override train_iters to 500
python -c "
import yaml, sys
cfg = yaml.safe_load(open('experiments/configs/model_shrunk_qwen3_small.yaml'))
cfg['training']['train_iters'] = 500
cfg['run_id'] = 'smoke_test'
yaml.safe_dump(cfg, open('$OUTPUT_DIR/smoke_config.yaml', 'w'))
"

bash experiments/run_scripts/run_with_monitor.sh \
    "$OUTPUT_DIR/smoke_config.yaml" \
    "$OUTPUT_DIR"

# Assertions on the output
python -c "
import json
from pathlib import Path
records = [json.loads(l) for l in Path('$OUTPUT_DIR/logs/monitor.jsonl').read_text().strip().split('\n')]
assert len(records) > 0, 'no monitor records'
groups = {r.get('metric_group') for r in records}
assert 'router' in groups, 'router tier missing'
assert 'delta_w' in groups, 'delta_w tier missing'
assert 'delta3' in groups or 'delta3_per_head' in groups, 'QK-product tier missing'
print(f'Smoke test PASS. {len(records)} records across {groups}.')
"
```

**Step 2: Make executable**

Run: `chmod +x experiments/run_scripts/*.sh`

**Step 3: Verify syntax (shellcheck)**

Run: `bash -n experiments/run_scripts/smoke_test.sh && bash -n experiments/run_scripts/run_with_monitor.sh && echo ok`
Expected: `ok`

**Step 4: Commit**

```bash
git add experiments/run_scripts/run_with_monitor.sh experiments/run_scripts/smoke_test.sh
git commit -m "feat(run): smoke-test launcher + monitor/fault wrapper"
```

**Step 5: Run smoke test on NPU (execution step — not verification)**

When on the 16× 910B node:
```bash
bash experiments/run_scripts/smoke_test.sh
```
Expected: `Smoke test PASS. N records across {router, delta_w, delta3_per_head, w}.` where N is in the thousands (~50-step delta_w × 10 samples × 16 layers × 16 heads = ~25600 records, plus routers and W tier).

If PASS: ready to proceed to Week 2 experimental runs (E3 baseline + E1a).

---

## Out-of-scope (future plans)

**Deferred to post-Week-1 plans:**

- **Analysis pipeline.** `analysis/` scripts for specific figures (detection-lag table, mantissa-sweep curve, etc.) — deferred per design Section 6 until we have real data.
- **Detection-criterion calibration.** BBP threshold estimation, 3σ / K=3 vs. CUSUM — deferred until baseline spread is observed.
- **Scale-ladder mini run execution (E1b).** Config exists (Task 27); launch script reuses this plan's `run_with_monitor.sh`.
- **Experimental runs themselves (E1-E8, A1, A2).** Not coding — these are launches + monitoring + debugging. Will be driven by a separate "experiment execution" plan after smoke test passes.
- **Offline forensics tooling (E8).** Apply the monitor to saved checkpoints — a thin wrapper around the existing `stability_monitor` library; ~half a day of work after we have checkpoints.

---

## Execution handoff

**Plan complete and saved to `docs/plans/2026-04-18-stability-monitor-week1-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session with executing-plans, batch execution with checkpoints.

**Which approach?**
