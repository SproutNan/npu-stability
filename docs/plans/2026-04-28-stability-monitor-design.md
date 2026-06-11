# Stability Monitor — Detailed Design

**Date:** 2026-04-28  
**Status:** Implemented; training-loop integration complete.  
**Scope:** Forward-only training-stability metrics for MoE Transformers on Ascend NPU.

---

## 1. Motivation

Training large MoE models is fragile. Instabilities manifest in weight matrices, router
distributions, and attention patterns — often silent in the loss curve until collapse is
imminent. The stability monitor samples these internal signals at configurable cadences
and writes them to a flat JSONL log for post-hoc analysis.

Two categories of faults are targetted:

| Category | Origin | Primary signal |
|---|---|---|
| **Runtime fault** | Numerical precision (e.g. mantissa mask in FlashAttention backward) | ΔW spectrum collapse → c_qk ramp → srank(Δ₃) → 1 → delayed loss spike |
| **Pre-running fault** | Hyperparameter misconfiguration (LR, aux-loss, batch size) | W norm explosion, router entropy collapse, weight cosine similarity rise |

The framework is designed so these two fault classes project onto **distinguishable
monitor fingerprints** — i.e., different metrics ring first depending on whether the
root cause is numerical or hyperparametric.

## 2. Architecture

### 2.1 Package layout

```
stability_monitor/
  __init__.py          # re-exports Monitor, MonitorConfig
  config.py            # MonitorConfig dataclass
  monitor.py           # Monitor orchestrator — step(), register_attention(), register_router()
  tracker.py           # WeightTracker — CPU-side weight snapshots, ΔW = W_t - W_{t-δ}
  spectral.py          # SVD-free spectral indicators via eigendecomposition
  qk_product.py        # Per-head QK-product delta decomposition (Δ₁, Δ₂, Δ₃)
  router.py            # Router gate metrics: entropy, max violation, diversification, cos-sim
  logging.py           # JsonlLogger — line-buffered JSONL append
  integration.py       # Glue: CLI args, env vars, layer registration, training-loop hooks
```

### 2.2 Data flow

```
Training loop (MindSpeed-LLM/training/training.py)
  │
  ├─ pretrain() ─► init_monitor_from_args(args, model[0])
  │                  │
  │                  ├─ MonitorConfig(cadences, output_path)
  │                  ├─ Monitor(cfg)
  │                  │    ├─ WeightTracker(delta_window)
  │                  │    └─ JsonlLogger(path)
  │                  └─ register_all_layers(model, monitor, args)
  │                       ├─ monitor.register_attention(...)  × num_layers
  │                       └─ monitor.register_router(...)     × num_moe_layers
  │
  └─ train() loop
       │
       ├─ train_step() ─► Router.forward() stashes _sm_logits, _sm_scores, _sm_routing_map
       │                   (3 lines in Megatron-LM/megatron/core/transformer/moe/router.py:439-442)
       │
       └─ monitor_step(iteration)
            │
            └─ Monitor.step(step)
                 ├─ step % router_cadence == 0  ─► _router_step()
                 │   reads stashed logits/scores from each router
                 │   computes: entropy, max_violation, diversification, cos_sim
                 │   logs:    router_weight SVD + norm
                 │
                 ├─ step % delta_w_cadence == 0  ─► _delta_w_step()
                 │   reads QKV/ router weights from model parameters
                 │   WeightTracker.update() → dW (or None if not at delta_window)
                 │   if dW available: SVD on dW_q, dW_k, dW_full, dW_router
                 │                   + QK-product per-head delta metrics
                 │   if dW is None:   carry-forward last values
                 │
                 └─ step % w_cadence == 0  ─► _w_step()
                     reads QKV/ router weights from model parameters
                     computes SVD + norm on current weights
```

### 2.3 Key design decisions

**Why tensor stashing instead of a forward hook.**
`TopKRouter.forward()` writes three tensors to `self._sm_*` attributes
(`logits.detach()`, `scores.detach()`, `routing_map.detach()`). The monitor
reads these at its own cadence — zero overhead on non-sampled steps. This avoids
registering PyTorch hooks and avoids holding the autograd graph.

**Why eigendecomposition instead of general SVD.**
`torch.linalg.svd` is not well-supported on CANN/NPU. `spectral.py` computes
singular values via `torch.linalg.eigvalsh(A^T A)` (or `A A^T` for tall matrices),
which is stable and supported. This is mathematically equivalent: σ_i(A) = √λ_i(A^T A).

**Why a module-level singleton.**
`integration.py` holds `_monitor: Optional[Monitor] = None` at module scope.
The training loop calls `monitor_step(iteration)` — a one-line function with
try/except that silently raises warnings on failure. This keeps the training
loop surface area minimal (1 line added).

## 3. Module Reference

### 3.1 `config.py` — MonitorConfig

```python
@dataclass
class MonitorConfig:
    router_cadence: int = 10      # steps between router-gate metrics
    delta_w_cadence: int = 50     # steps between ΔW / QK-delta metrics
    w_cadence: int = 500          # steps between full-W health metrics
    delta_window: int = 50        # δ for ΔW = W_t - W_{t-δ}
    per_head: bool = True         # enable per-head QK-product decomposition
    head_aggregation: Tuple = ("mean", "max", "p95")  # (reserved, not yet wired)
    output_path: Optional[str] = None
```

Cadence rationale:

| Metric tier | Default cadence | Cost per step (amortized) | Rationale |
|---|---|---|---|
| Router gate | 10 | ~O(E·T) all-reduce, lightweight | Router collapse can onset in O(10) steps |
| ΔW / QK-delta | 50 | O(d_k³) per head eigendecomp, heavy | Weight spectrum changes on O(100)-step timescale |
| W health | 500 | Same as ΔW but less frequent | W itself changes slowly; serves as long-term trend |

### 3.2 `spectral.py` — SVD-Free Spectral Indicators

All functions operate `@torch.no_grad()` in float32.

| Function | Returns | Definition |
|---|---|---|
| `svdvals_via_eigvalsh(A)` | `Tensor[k]` | σ_i = √λ_i(A^T A) via `eigvalsh`. Uses `A A^T` when m < n. |
| `stable_rank(A)` | scalar | ‖A‖_F² / σ_max². Effective dimensionality of the matrix. |
| `spectrum_entropy(A, α=2)` | scalar | exp(H) where H = -Σ p_i log p_i, p_i = σ_i^α / Σ σ_j^α. Number of "active" singular modes. |
| `calc_all_svd_metrics(mat)` | `dict` | Aggregated: `effective_rank` (α=1 entropy), `singular_spectrum` (α=2 entropy), `stable_rank`, `energy_ratio` (top-5 concentration), `condition` (σ_max / σ_min) |
| `calc_norm(mat)` | float | mean(‖row_i‖₂). Average row-wise L2 norm. |

**Why two entropy orders.** α=1 (effective rank) measures diversity of σ_i directly.
α=2 (singular spectrum) measures diversity of σ_i² — sensitivity to large singular
values. A matrix with one dominant σ and many small ones has erank → 1 but
spectrum may still be >1. The pair disambiguates "one-mode collapse" from
"broad spectrum decay."

### 3.3 `router.py` — Router Gate Metrics

All functions `@torch.no_grad()`. When `dp_group` is provided, metrics are
globally reduced across the data-parallel group.

| Function | Definition | Interpretation |
|---|---|---|
| `per_token_expert_entropy(logits, gates, dp_group)` | E[ H(softmax(logits)) ] across tokens. H = logsumexp - Σ(gates · logits). | High → uniform routing. Low → peaked routing, possible collapse. |
| `max_violation(logits, topk, dp_group)` | (max_expert_load - mean_load) / mean_load, based on top-k argmax of logits. | Quantifies load imbalance. 0 = perfectly balanced. >1 = severe skew. |
| `router_diversification(gates, topk, after_topk, dp_group)` | Variance of gate values per expert column, masked to top-k. | High → experts receive diverse gate magnitudes. Low → uniform gating, router not learning. |
| `expert_specialization(expert_output, scatter_index, start_idx, end_idx, dp_group)` | Mean non-diagonal cosine projection between routed expert outputs. | High → experts produce orthogonal outputs (specialized). Low → experts produce similar outputs (redundant). |
| `router_weight_cosine_similarity(weight)` | Average pairwise cosine similarity of expert embedding columns in the router weight matrix (num_experts × hidden). | High → router weight columns are collinear, routing is undifferentiated. Low → columns are orthogonal, routing is discriminative. |
| `router_weight_centered_frob_sq(weight)` | $\|W_c\|_F^2 = \sum_i \|w_i - \bar{w}\|^2$, the centered Frobenius energy of the router weight matrix. Computed efficiently as $\sum_i\|w_i\|^2 - E\|\bar{w}\|^2$. | Uncalibrated logit-energy scale. Tracks router entropy collapse from weights alone (isotropic-reference KL proxy: $\|W_c\|_F^2/(2E)$). Rising → stronger logit spread, possible routing overconfidence. Should be read against a healthy baseline, not as an absolute entropy estimator. |

**Note on `expert_specialization`:** This is defined but **not wired into the Monitor**
because it requires the expert output tensor after token permutation, which is
not accessible from the router alone. It would require a hook in `MoELayer.forward()`
after `token_unpermutation`. This is left for future integration.

### 3.4 `qk_product.py` — Per-Head QK-Product Delta Decomposition

Following the theoretical framework from QK-product delta monitoring, the
difference between two QK-product matrices over a δ-step interval is decomposed
into three terms:

Let W_q(t), W_k(t) be query and key weight matrices at step t, each of shape
(d_k, d) where d_k = n_kv_heads × head_dim and d = hidden_size. The QK-product
at step t is:

```
QK(t) = W_q(t)^T W_k(t)    ∈ R^{d × d}
```

The delta over δ steps:

```
ΔQK = QK(t)^T QK(t) - QK(t-δ)^T QK(t-δ) = Δ₁ + Δ₂ + Δ₃
```

where:

| Term | Expression | Interpretation |
|---|---|---|
| Δ₁ (exact) | [W_q+ΔW_q; -W_q]^T [W_k+ΔW_k; W_k] | Exact difference (2d_k × d matrices) |
| Δ₂ (linearized) | [ΔW_q; W_q]^T [W_k; ΔW_k] | First-order approximation |
| Δ₃ (core) | ΔW_q^T ΔW_k | Pure-update core (d_k × d_k, never materializes d × d) |

**Core-matrix reduction (never materializes d × d).**
Given thin SVDs ΔW_q = U_q Σ_q V_q^T, ΔW_k = U_k Σ_k V_k^T (computed once for ΔW
tier and reused):

```
Δ₃ = U_q · (Σ_q V_q^T V_k Σ_k) · U_k^T
      └──────── M ────────┘  ∈ R^{d_k × d_k}
```

All spectral indicators of Δ₃ reduce to indicators of the small core matrix M.
Cost: O(d_k³) per head instead of O(d³).

**Metric functions:**

| Function | Description |
|---|---|
| `_eigh_U_S(A)` | Thin SVD of (r, d) matrix A via `eigh(A A^T)`. Returns (U, S). |
| `_core(P, Q)` | Core matrix M = Σ_P U_P^T U_Q Σ_Q. σ(M) = nonzero σ(P^T Q). |
| `_get_svd_metrics_4tuple(weight)` | erank, spectrum, srank, condition for a single matrix. |
| `get_qk_product_metrics(dW_q, dW_k, W_q, W_k)` | All d3/d2/d1 metrics + coupling coefficient c_qk. |
| `calc_qk_dw_metrics(qkv_current, qkv_prev, head_dim, proj_size, kv_size)` | Per-head, max-aggregated. Reshapes fused QKV into per-head slices and computes pairwise GQA head matching. |

**Per-matrix metrics computed for each Δ term:**

| Metric key | Definition |
|---|---|
| `erank_d{1,2,3}` | Effective rank (α=1 spectrum entropy) of the core matrix |
| `spectrum_d{1,2,3}` | Singular spectrum (α=2 spectrum entropy) |
| `srank_d{1,2,3}` | Stable rank ‖M‖_F² / σ_max² |
| `cond_d{1,2,3}` | Condition number σ_max / σ_min |
| `fnorm_d{1,2,3}` | Frobenius norm ‖M‖_F |
| `c_qk` | Coupling coefficient: ‖Δ₃‖_F² / (‖ΔW_q‖_F² · ‖ΔW_k‖_F²) in [0,1]. 1 = perfectly aligned updates, 0 = orthogonal. |

**GQA head matching.** For grouped-query attention with n_q query heads and n_kv key heads
(group_size = n_q / n_kv), query head `q_idx` is paired with key head `q_idx // group_size`.
Each pair contributes one set of Δ metrics; the max across all heads is emitted.

### 3.5 `tracker.py` — WeightTracker

Maintains CPU-side weight snapshots for delta-W computation with micro-batch
deduplication.

```
State:
  _snapshots:   Dict[(name, layer_num), Tensor]  — CPU tensor
  _snapshot_step: Dict[(name, layer_num), int]    — global step of snapshot

update(name, layer_num, W, step) → Optional[Tensor]:
  1. If snapshot_step[name, layer_num] == step → return None
     (deduplicate: first micro-batch stores, subsequent ones skip)
  2. prev = _snapshots[name, layer_num]
  3. If prev exists AND last_snapshot_step == step - delta_window:
       delta = W.detach() - prev.to(W.device)   # compute ΔW
     else:
       delta = None
  4. Store W.detach().cpu().clone() as new snapshot
  5. Store step as new snapshot_step
  6. Return delta (or None)
```

**Key invariant:** Each `(name, layer_num)` pair is updated at most once per
global step. The first micro-batch stores; subsequent micro-batches in the same
gradient accumulation cycle are skipped. This prevents the "overwrite-to-zero"
bug where micro-batch 2 stores a snapshot that overwrites micro-batch 1's,
causing dW ≈ 0 on the next stride step.

**get_prev(name, layer_num):** Returns the previous snapshot (used by QK-product
delta to access W_q(t-δ) and W_k(t-δ) for the Δ₁ / Δ₂ decompositions).

### 3.6 `logging.py` — JsonlLogger

```python
class JsonlLogger:
    def __init__(self, path: Optional[str]):
        # Opens path in append mode ("a") with line buffering (buffering=1).
        # If path is None, log() is a no-op.

    def log(self, record: Dict[str, Any]) -> None:
        # json.dumps(record) + "\n", immediate flush via line buffering.

    def close(self) -> None:
        # Flush and close file handle.
```

**Design rationale.**
- **Line-buffered:** Records are visible during training (tail -f works).
- **Append mode:** Safe across restarts from checkpoint; no overwrite.
- **Flat schema:** Each record carries `step`, `metric_group`, and metric-specific keys.
  Pivot in pandas with `pd.read_json(path, lines=True)`.
- **None-safe:** When no output path is configured, logging is a silent no-op.
  No if-checks needed at call sites.

### 3.7 `monitor.py` — Monitor Orchestrator

**Registration dataclasses:**

```python
@dataclass
class _AttentionReg:
    layer_id: str        # "layer.0.attention"
    layer_num: int       # 0
    layer: Any           # TransformerLayer (for weight access)
    num_q_heads: int     # e.g. 16
    num_kv_heads: int    # e.g. 4 (GQA)
    head_dim: int        # e.g. 64

@dataclass
class _RouterReg:
    layer_id: str        # "layer.0.router"
    layer_num: int       # 0
    layer: Any           # MoELayer (for weight access)
    num_experts: int     # e.g. 128
    top_k: int           # e.g. 8
```

**Weight accessors:**

| Accessor | Path | Returns |
|---|---|---|
| `_get_qkv_weight(reg)` | `reg.layer.self_attention.linear_qkv.weight` | `Wq, Wk, q_rows, k_rows` by slicing fused QKV |
| `_get_router_weight(reg)` | `reg.layer.mlp.router.weight` | Router gate weight `(hidden, num_experts)` |

**Fused QKV slicing.** Megatron-Core's fused QKV linear stores weights as
`[Wq; Wk; Wv]` concatenated along dim 0. The monitor slices:
- `Wq = W[:q_rows]` where `q_rows = num_q_heads × head_dim`
- `Wk = W[q_rows : q_rows + k_rows]` where `k_rows = num_kv_heads × head_dim`

Wv is not accessed (attention output is not monitored).

**Step dispatch:**

```python
def step(self, step: int) -> None:
    if step % self.cfg.router_cadence == 0:
        self._router_step(step)
    if step % self.cfg.delta_w_cadence == 0:
        self._delta_w_step(step)
    if step % self.cfg.w_cadence == 0:
        self._w_step(step)
```

Tiers are independent — a step can trigger 0, 1, 2, or all 3 tiers.
Common multiples: step 0 triggers all three; step 500 triggers router + ΔW + W.

**Carry-forward.** On non-stride ΔW steps (dW is None because snapshot hasn't
accumulated enough history, or the step isn't aligned with delta_window), the
monitor carries forward the last computed dW values under `metric_group: "dw_carry_forward"`.
This prevents zeros in downstream visualization and ensures every step has a value
without re-computing expensive eigendecompositions.

### 3.8 `integration.py` — Training-Loop Glue

Module-level singleton `_monitor: Optional[Monitor] = None`.

**Environment variables:**

| Variable | Purpose | Accepted values |
|---|---|---|
| `STABILITY_MONITOR_ENABLED` | Master enable switch | `1`, `true`, `yes`, `on` (case-insensitive) |
| `STABILITY_MONITOR_OUTPUT_PATH` | Default JSONL path | Any filesystem path |

**Resolution order for enable:**
1. `--enable-stability-monitor` CLI flag
2. `STABILITY_MONITOR_ENABLED` env var

**Resolution order for output path:**
1. `--monitor-output-path` CLI arg
2. `STABILITY_MONITOR_OUTPUT_PATH` env var
3. `stability_monitor_metrics.jsonl` (CWD default)

**CLI arguments** (registered via `add_stability_monitor_args(parser)`):

| Argument | Type | Default |
|---|---|---|
| `--enable-stability-monitor` | flag | False |
| `--monitor-output-path` | str | None |
| `--monitor-router-cadence` | int | 10 |
| `--monitor-delta-w-cadence` | int | 50 |
| `--monitor-w-cadence` | int | 500 |
| `--monitor-delta-window` | int | 50 |
| `--monitor-per-head` | flag | True |

**Layer registration (`register_all_layers`):**

```python
decoder = model.decoder  # GPTModel.decoder → TransformerBlock
for idx, layer in enumerate(decoder.layers):
    # All layers get attention registration
    monitor.register_attention(layer_id=f"layer.{idx}.attention", ...)

    # Layers >= first_k_dense_replace get router registration (MoE layers)
    first_k = getattr(cfg, "first_k_dense_replace", 0)
    if idx >= first_k:
        monitor.register_router(layer_id=f"layer.{idx}.router", ...)
```

**Training loop integration points (MindSpeed-LLM):**

| Location | Call | Purpose |
|---|---|---|
| `pretrain_gpt.py:main()` | Pass `add_stability_monitor_args` as `extra_args_provider` | Register CLI args with Megatron's argument parser |
| `training.py:pretrain()` after `build_train_args()` | `init_monitor_from_args(args, model[0])` | Create Monitor, register all layers |
| `training.py:train()` after `train_step()` | `monitor_step(iteration)` | Sample and log metrics |
| `training.py:pretrain()` before `one_logger_utils.finish()` | `monitor_close()` | Flush and close JSONL |

**Megatron-Core integration point:**

| Location | Modification |
|---|---|
| `megatron/core/transformer/moe/router.py:TopKRouter.forward()` | Three lines at L439-442 stash `_sm_logits`, `_sm_scores`, `_sm_routing_map` |

All imports are guarded with `try/except ImportError` — if `stability_monitor`
is not on the Python path, all calls become no-ops with zero overhead.

## 4. Metric Taxonomy

### 4.1 Router tier (every `router_cadence` steps, default 10)

`metric_group: "router"`:

| Metric | Source function | Range | Fault sensitivity |
|---|---|---|---|
| `per_token_entropy` | `per_token_expert_entropy(logits, scores)` | [0, log(E)] | **E2-AUX ↓↓**, E2-LR ↓ |
| `max_violation` | `max_violation(logits, topk)` | [0, ∞) | **E2-AUX ↑↑** |
| `router_diversification` | `router_diversification(scores, topk, after_topk=True)` | [0, ∞) | E2-AUX ↓, E2-LR ↓ |
| `router_weight_cos_sim` | `router_weight_cosine_similarity(W_router)` | [-1, 1] | **E2-LR ↑↑** |
| `router_weight_c_frob_sq` | `router_weight_centered_frob_sq(W_router)` | [0, ∞) | **E2-LR ↑↑** (centered Frobenius energy; isotropic-ref KL proxy via $\|W_c\|_F^2/(2E)$) |

`metric_group: "router_weight_svd"` and `"router_weight_norm"`:

| Metric | Definition |
|---|---|
| `effective_rank` | exp(-Σ p_i log p_i), p_i = σ_i / Σ σ_j |
| `singular_spectrum` | exp(-Σ p_i log p_i), p_i = σ_i² / Σ σ_j² |
| `stable_rank` | Σ σ_i² / σ_max² |
| `energy_ratio` | Σ_{i=1..5} σ_i² / Σ σ_i² |
| `condition` | σ_max / σ_min |
| `value` (norm) | mean(‖row_i‖₂) |

### 4.2 Delta-W tier (every `delta_w_cadence` steps, default 50)

For each attention layer, three matrices are analyzed:

`metric_group: "qkv_weight_dw_svd"` and `"qkv_weight_dw_norm"`:

| Matrix key | Content |
|---|---|
| `qkv_q` | ΔW_q = W_q(t) - W_q(t-δ) |
| `qkv_k` | ΔW_k = W_k(t) - W_k(t-δ) |
| `qkv_full` | [ΔW_q; ΔW_k] concatenated |

Each carries the same 5 SVD indicators + norm as the router tier.

`metric_group: "qk_dw"` (per-head, max-aggregated):

| Metric | Definition |
|---|---|
| `erank_d3` | Effective rank of core matrix M = Σ_q U_q^T U_k Σ_k |
| `spectrum_d3` | Singular spectrum of M |
| `srank_d3` | Stable rank of M |
| `cond_d3` | Condition number of M |
| `fnorm_d3` | ‖M‖_F |
| `erank_d2` | Effective rank of linearized delta |
| `spectrum_d2` | Singular spectrum of linearized delta |
| `srank_d2` | Stable rank of linearized delta |
| `cond_d2` | Condition number of linearized delta |
| `fnorm_d2` | ‖linearized‖_F |
| `erank_d1` | Effective rank of exact delta |
| `spectrum_d1` | Singular spectrum of exact delta |
| `srank_d1` | Stable rank of exact delta |
| `cond_d1` | Condition number of exact delta |
| `fnorm_d1` | ‖exact‖_F |
| `c_qk` | Coupling coefficient ‖Δ₃‖_F² / (‖ΔW_q‖_F² · ‖ΔW_k‖_F²) |

`metric_group: "router_weight_dw_svd"` and `"router_weight_dw_norm"`:
Same SVD indicators + norm on ΔW_router.

`metric_group: "dw_carry_forward"`:
On non-stride steps, previously computed dW values are re-emitted with their
original `key` field to prevent visualization gaps.

### 4.3 W tier (every `w_cadence` steps, default 500)

`metric_group: "w"`, `"w_norm"`, `"w_router"`, `"w_router_norm"`:
Same 5 SVD indicators + norm, computed on the full current weight matrices
(rather than deltas). This serves as a slow-moving baseline — W itself changes
gradually in healthy training.

## 5. Distributed Reduction

When `torch.distributed` is initialized and a data-parallel group is available
(from Megatron's `parallel_state.get_data_parallel_group()`), router gate metrics
are globally reduced via `all_reduce(SUM)` over the DP group:

| Metric | What is reduced | Final value |
|---|---|---|
| `per_token_entropy` | Sum of per-token entropy + count of tokens | global_sum / global_count |
| `max_violation` | Per-expert token count vector | (max - mean) / mean of global loads |
| `router_diversification` | Sum of squared diffs + token count | global_sum / global_count |

Weight metrics (SVD, norm) are **not** reduced — they are computed on identical
weights across DP ranks (replicated parameters) and would produce identical values.

When `dp_group` is None (single-GPU or dist not initialized), metrics are
computed locally without reduction.

## 6. Output Format

### 6.1 JSONL schema

Each line is a flat JSON object. Common fields:

| Field | Always present | Description |
|---|---|---|
| `step` | Yes | Global training step |
| `metric_group` | Yes | One of: `router`, `router_weight_svd`, `router_weight_norm`, `qkv_weight_dw_svd`, `qkv_weight_dw_norm`, `qk_dw`, `dw_carry_forward`, `router_weight_dw_svd`, `router_weight_dw_norm`, `w`, `w_norm`, `w_router`, `w_router_norm` |
| `layer_id` | Yes | e.g. `"layer.0.attention"`, `"layer.8.router"` |
| `metric` | SVD/norm groups | e.g. `"effective_rank"`, `"stable_rank"`, `"condition"` |
| `matrix` | QKV-dW/W groups | `"qkv_q"`, `"qkv_k"`, `"qkv_full"` |
| `value` | SVD/norm groups | float |
| `key` | carry_forward | Internal key identifying which metric is being forwarded |

### 6.2 Example records

```json
{"step": 10, "metric_group": "router", "layer_id": "layer.8.router",
 "per_token_entropy": 3.45, "max_violation": 0.12,
 "router_diversification": 0.023, "router_weight_cos_sim": 0.08,
 "router_weight_c_frob_sq": 28.5}

{"step": 10, "metric_group": "router_weight_svd", "layer_id": "layer.8.router",
 "metric": "effective_rank", "value": 45.2}

{"step": 50, "metric_group": "qk_dw", "layer_id": "layer.0.attention",
 "metric": "erank_d3", "value": 12.7}

{"step": 51, "metric_group": "dw_carry_forward", "layer_id": "layer.0.attention",
 "key": "qk_dw_erank_d3_layer.0.attention", "value": 12.7}
```

### 6.3 Analysis pattern

```python
import pandas as pd

df = pd.read_json("metrics/monitor_20260428_120000.jsonl", lines=True)

# Pivot a specific metric across layers
pivot = df[df.metric_group == "router"].pivot(
    index="step", columns="layer_id", values="per_token_entropy"
)
pivot.plot(title="Per-token expert entropy by layer")

# Compare ΔW spectrum across layers
dw = df[df.metric_group == "qkv_weight_dw_norm"]
dw_pivot = dw.pivot(index="step", columns="layer_id", values="value")
dw_pivot.plot(title="‖ΔW_qkv‖ by layer")
```

## 7. Integration with MindSpeed-LLM

### 7.1 Files modified

| File | Change | Lines |
|---|---|---|
| `MindSpeed-LLM/pretrain_gpt.py` | Guarded import of `add_stability_monitor_args`; pass as `extra_args_provider` | +5, +1 |
| `MindSpeed-LLM/mindspeed_llm/training/training.py` | Guarded imports; `init_monitor_from_args()` after model init; `monitor_step()` after train_step; `monitor_close()` before finish | +8, +2, +2, +3 |
| `Megatron-LM/megatron/core/transformer/moe/router.py` | Stash `_sm_logits`, `_sm_scores`, `_sm_routing_map` in `TopKRouter.forward()` | +3 |

### 7.2 Launch

```bash
# Via env var (no CLI flags needed)
STABILITY_MONITOR_ENABLED=1 \
STABILITY_MONITOR_OUTPUT_PATH=metrics/monitor.jsonl \
bash train_shrunk_qwen3_moe.sh

# Via CLI flags
bash train_shrunk_qwen3_moe.sh \
  --enable-stability-monitor \
  --monitor-output-path metrics/monitor.jsonl \
  --monitor-router-cadence 5 \
  --monitor-delta-w-cadence 25

# Dedicated script
bash train_shrunk_qwen3_moe_monitor.sh
```

## 8. Performance

### 8.1 Overhead model

For a model with L layers and D = hidden_size, the dominant cost per sampled step is:

| Operation | Complexity | Frequency |
|---|---|---|
| QK dW eigendecomp (2× per layer) | O(d_k³) per head | every `delta_w_cadence` |
| QK-product core matrix (per head) | O(d_k³) per head | every `delta_w_cadence` |
| Router weight SVD | O(min(E,h)³) | every `router_cadence` |
| Router gate metrics | O(T·E) all-reduce | every `router_cadence` |
| W SVD (QK + router) | same as dW | every `w_cadence` |

With the default config (shrunk Qwen3: d_k = 64, 16 heads, L=16, E=128):

| Tier | Stride | Cost per sampled step | Amortized cost per step |
|---|---|---|---|
| Router | 10 | ~2ms | ~0.2ms |
| ΔW + QK-delta | 50 | ~15ms | ~0.3ms |
| W health | 500 | ~15ms | ~0.03ms |
| **Total amortized** | | | **~0.5ms** |

At ~500ms/step training throughput, this is **~0.1% overhead**.

### 8.2 Memory

- **WeightTracker snapshots:** stored on CPU. 2 × L × (q_rows + k_rows + E) × 2 bytes ≈
  2 × 16 × (1024 + 256 + 128 × 1024) × 2 ≈ 8.4 MB for the shrunk model. Negligible.
- **Stashed tensors:** 3 tensors × L_moe × (seq × bs × E) ≈ 3 × 16 × (4096 × 8 × 128) × 2 ≈ 400 MB
  if all stashed simultaneously. In practice only the most recent micro-batch's values
  are retained (same attributes overwritten each micro-batch), so actual is ~25 MB.
- **GPU scratch for eigendecomp:** O(d_k²) per head, freed immediately. < 1 MB.

## 9. Limitations and Future Work

### 9.1 Not implemented (out of scope for v1)

| Feature | Reason |
|---|---|
| Backward metrics (grad_logits, TEV, TEV_grad) | Excluded per user request. Requires gradient hooks in router backward. |
| Online detectors (threshold alarms, trend rules) | Detection logic is post-hoc for research; training loop is monitoring-only. |
| Expert specialization (forward) | Requires hook in `MoELayer.forward()` after `token_unpermutation`. Router-level stashing is insufficient. |
| Gauge invariance tests (Δ₁ = Δ₂ + Δ₃ identity) | Post-hoc analysis on collected data. |
| QKV weight saving to disk | Too large; W metrics (SVD + norm) are sufficient proxies. |
| Checksum / xorsum verification | Requires patching distributed communication ops; framework-specific. |

### 9.2 Known constraints

- **PP > 1:** `model[0]` is only the first pipeline stage's chunk. Layers on other
  stages are not monitored. Pipeline-parallel training needs per-stage monitor
  instances with separate output files.
- **Activation checkpointing:** Router tensors are stashed during the forward pass
  WITHIN the checkpointed region. If recompute is enabled, the stashed tensors
  represent the recomputed forward (grad-enabled), not the checkpoint forward (no-grad).
  This is acceptable because the values should be identical up to BF16 precision.
- **Expert specialization** depends on having the post-permutation expert output
  tensor, which requires an additional modification to `MoELayer.forward()`.
  The function exists in `router.py` but is not wired in.

### 9.3 Extension points

1. **TensorBoard integration:** Pass `SummaryWriter` to `Monitor.__init__` and
   call `writer.add_scalar(f"stability/{metric}", value, step)` in each log call.
2. **Additional weight matrices:** Register arbitrary named weights via
   `monitor.register_weight(name, layer_num, accessor)`.
3. **Cadence adaptation:** Auto-adjust cadences based on metric volatility
   (e.g., increase router cadence when entropy is stable).
4. **Checkpointing monitor state:** Save/restore WeightTracker snapshots so
   ΔW metrics are correct immediately after training resume.

## 10. References

- Megatron-Core MoE router: `Megatron-LM/megatron/core/transformer/moe/router.py`
- Original gate monitor (xpu_speed): `xpu_speed/xpu_speed/vigil/metrics/gate_monitor.py`
- Original Vigil integration: `xpu_speed/xpu_speed/vigil/__init__.py`
- v1.py reference pattern: `xpu_speed/xpu_speed/core/bigop/m8/v1.py`
- Training architecture overview: `docs/qwen3-moe-training-architecture.md`
- Original design doc: `docs/plans/2026-04-18-mindspeed-stability-monitor-opensource-design.md`
