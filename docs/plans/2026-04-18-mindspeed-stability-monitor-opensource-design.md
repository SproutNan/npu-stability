# MindSpeed Open-Source Reproduction of Training-Stability Monitoring

**Date:** 2026-04-18
**Status:** Design approved; ready for implementation planning.
**Context:** Open-source reproduction of the training-stability monitoring framework described in `大模型训练稳定性监控——从第一性原理出发.md` and its QK-product extension `2026-04-16-qk-product-delta-monitor-design.md`. The original experiments were conducted on a proprietary environment and frameworks; this plan ports them to MindSpeed with public models and public data, in support of a top-tier LLM-venue submission.

## 0. Mission

Produce an open-source reproduction of the paper's central claims, in a form that:

1. **Validates the theoretical framework empirically** on public hardware + software + data, so external reviewers can re-run.
2. **Delivers a standalone, framework-agnostic monitor library** as a companion artifact — the community can plug it into their own training stacks.
3. **Supports the two-category story** the paper will tell: **(A) runtime faults** (numerical precision) and **(B) pre-running faults** (hyperparameter misconfiguration), with distinct monitor fingerprints per category.

The source documents provide the theoretical framework; this document defines *how* we empirically validate it under open-source constraints.

## 1. Experimental matrix

| ID | Experiment | Runs | Purpose |
|---|---|---|---|
| **E1a** | FA mantissa-mask fault, default (4-bit mask) | Small model × 2 seeds | Headline: ΔW srank / spec-entropy collapse precedes loss spike |
| **E1b** | FA fault at mini-scale | Mini model × 1 seed | Scale-ladder sanity check |
| **E2-LR10** | Large LR, 10× | Small model × 1 seed | Moderate hyperparam fault |
| **E2-LR40** | Large LR, 40× | Small model × 2 seeds | Severe hyperparam fault; paper claims layer-wise cascade |
| **E2-AUX** | Aux-loss disabled | Small model × 2 seeds | MoE-specific hyperparam fault with router-only fingerprint |
| **E2-GBS** | Global batch size 16 (8× smaller) | Small model × 1 seed | Gradient-noise regime fault |
| **E3** | Baseline long-stable run | Small model × 2 seeds | Null distribution for ΔW spectrum |
| **E4/E5** | QK decomposition + gauge check | Post-hoc on E1/E2 outputs | Δ₁ = Δ₂ + Δ₃ identity; gauge invariance of score-space monitors |
| **E7** | Combined fault (E1a + E2-LR40) | Small model × 1 seed | Fingerprint superposition |
| **E8** | Offline forensics | Post-hoc on E1/E2/E3 checkpoints | Per-layer health diagnosis from frozen weights |
| **A1** | Mantissa-bit sweep (0, 2, 6; bits=4 reuses E1a) | Small model × 3 additional variants × 1 seed | Dose-response curve |
| **A2** | Detection-lag comparison table | Post-hoc on E1/E2/E3 | **Main paper table** — first-alarm step across monitors |

**Totals:** 15 long runs on shrunk-Qwen3 MoE, ~20-40k steps each, ~2.5-3 weeks of 16× 910B single-node compute.

**Explicit non-goals:**
- No claim about scaling laws of monitors at 100B+.
- No online intervention on alarm (monitoring only).
- No real-silicon FP8/FP6 experiments — only synthetic mantissa-mask (Qiu & Yao 2025 already cover real FP8).

## 2. Model architecture

### 2.1 Primary model (for E1a, E2-*, E3, E7, A1)

Shrunk Qwen3-family MoE, reusing Qwen3 modules and tokenizer, with dimensions reduced for compute fit while preserving the *number of experts* that governs router-collapse dynamics.

| Dimension | Value |
|---|---|
| Hidden size | 1024 |
| FFN intermediate (per expert) | 768 |
| Num layers | 16 |
| Query heads | 16 |
| KV heads (GQA) | 4 |
| Head dim | 64 |
| Num routed experts | 128 |
| Top-k | 8 |
| Shared experts | 1 |
| Expert bias (aux-loss-free compatible) | Yes |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Rotary base | 10000 |
| Vocab | 151,936 (Qwen3 tokenizer) |
| Total params | ~2.8B |
| Active params | ~0.4B |

### 2.2 Mini model (for E1b scale-ladder)

Same architecture, further shrunk: hidden 512, 8 layers, 64 experts, top-4. **~0.4B total / ~0.08B active.** Trains ~4× faster than primary.

### 2.3 Baseline training hyperparameters

| Parameter | Value |
|---|---|
| Sequence length | 2048 |
| Micro batch size | 4 |
| Global batch size | 128 |
| Tokens / step | ~262k |
| LR (baseline) | 3e-4 |
| LR schedule | cosine, 2000-step warmup |
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.1 |
| Init std | 0.02 |
| Parallelism | TP=1, PP=1, EP=8, DP=2 |
| Precision | BF16 mixed |
| Train iters (E3 baseline) | 40,000 |
| Train iters (fault runs) | 20,000-25,000 |
| Data | FineWeb-Edu `sample-100BT` + Qwen3 tokenizer, pre-tokenized as Megatron .bin shards, fixed order |

## 3. Fault injection

Two categories, each specified as pure math on tensors (framework-agnostic in spec; MindSpeed adapter mechanics in §5).

### 3.1 Category 1 — Runtime fault: FA mantissa-mask on $O \odot dO$

**Mechanism.** Inside FA backward, immediately after computing $M_{ij} := O_{ij} \odot dO_{ij}$, zero the bottom $k$ mantissa bits of the BF16 representation:

$$
\text{masked}_{bf16}(x, k) = \text{bf16\_view}(\text{uint16\_view}(x) \wedge \lnot((1 \ll k) - 1))
$$

This produces a systematic (not random) downward bias — matching Qiu & Yao 2025's characterization. Applied every backward step, every layer, every attention head.

**Bit-count variants (A1 sweep):**

| Variant | Masked bits | Retained mantissa | Effective precision | Use |
|---|---|---|---|---|
| Control | 0 | 7 | Full BF16 | A1 control |
| Mild | 2 | 5 | ~FP16-ish | A1 mild |
| Default (E1a) | 4 | 3 | ~FP8 E4M3-ish | E1a + A1 default |
| Severe | 6 | 1 | Aggressive | A1 severe |

**Important:** Even the "0 bits masked" control runs the reference FA code path (not MindSpeed fused FA), so that fused-vs-unfused FA is not a confound for A1 comparisons. Baseline E3 and hyperparameter faults E2-* run with MindSpeed fused FA at full speed.

**Expected fingerprint:** ΔW spectrum collapse → c_qk ramp → srank(Δ₃) → 1 → loss spike (lagging). W itself stays near-healthy until late.

### 3.2 Category 2 — Pre-running faults: hyperparameter misconfiguration

Four MoE-relevant knob faults, each flipping exactly one dimension from baseline:

| Fault | Knob | Baseline | Fault value |
|---|---|---|---|
| E2-LR10 | Peak LR | 3e-4 | 3e-3 |
| E2-LR40 | Peak LR | 3e-4 | 1.2e-2 |
| E2-AUX | Aux-loss coefficient | 1e-4 | 0 (disabled) |
| E2-GBS | Global batch size | 128 | 16 |

**Expected fingerprints (to be empirically verified):**
- **E2-LR10 / E2-LR40:** $W$ Frobenius explodes; router weight cos-sim rises; per-token entropy drops; $\cos(W_t, W_0) \to 0$. **$W$ is what's sick, not $\Delta W$.**
- **E2-AUX:** Extreme load imbalance; per-token entropy drops selectively; c_qk near baseline. **Router-only signature.**
- **E2-GBS:** ΔW Frobenius high-frequency variance inflates; spectrum shape more preserved than E1; entropy may oscillate. **Noise-driven signature.**

### 3.3 Combined fault E7

E7 = E1a (FA 4-bit mask) ∪ E2-LR40 (40× LR). Tests whether fingerprints superimpose cleanly as QK doc §2.4 predicts.

### 3.4 Data & seed controls

All fault runs share identical seed, identical data order, identical initialization with matched baseline. Divergence between baseline and fault is attributable solely to the injected perturbation.

## 4. Monitor library (`stability_monitor`)

### 4.1 Invariants

1. **Framework-agnostic.** Tensors in (via user-provided accessors) → metrics out.
2. **Computation-bounded.** Total overhead < 2% of a training step at recommended cadence.
3. **Stateless from caller's perspective.** Library owns its history buffers; training loop just calls `monitor.step(step_num)`.

### 4.2 Module layout

```
stability_monitor/
├── __init__.py
├── config.py                   # MonitorConfig dataclass
├── monitor.py                  # Public entry point; orchestrates everything
├── tracker.py                  # WeightTracker — snapshots + ΔW computation
├── spectral.py                 # srank, spec_entropy, SVD core
├── qk_product.py               # Δ₁/Δ₂/Δ₃ decomposition, c_qk, core-matrix reduction
├── router.py                   # per-token entropy, weight cos-sim, diversification, specialization
├── detectors.py                # threshold rules, trend rules, layer-consistency
├── logging.py                  # JSONL sink
└── testing/
    ├── test_spectral.py
    ├── test_qk_product.py
    ├── test_gauge_invariance.py
    └── fixtures.py
```

### 4.3 Public API

```python
from stability_monitor import Monitor, MonitorConfig

cfg = MonitorConfig(
    router_cadence=10,
    delta_w_cadence=50,
    w_cadence=500,
    delta_window=50,
    per_head=True,
    head_aggregation=("mean", "max", "p95"),
    output_path="logs/monitor.jsonl",
)
monitor = Monitor(cfg)

for idx, layer in enumerate(model.decoder.layers):
    monitor.register_attention(
        layer_id=f"layer.{idx}.attention",
        wq_accessor=..., wk_accessor=...,
        num_q_heads=16, num_kv_heads=4, head_dim=64,
    )
    if idx >= first_k_dense_replace:
        monitor.register_router(
            layer_id=f"layer.{idx}.router",
            router_weight_accessor=..., probs_hook=..., assignments_hook=...,
            num_experts=128, top_k=8,
        )

for step in range(num_iters):
    # ... training step ...
    monitor.step(step)
```

### 4.4 Metrics

**Router tier** (every `router_cadence` steps): per-token entropy, router weight cos-sim, load Gini, expert specialization.

**ΔW tier** (every `delta_w_cadence` steps, SVD-based): ΔW Frobenius norm, srank, spectrum entropy (α=2) — for Wq, Wk, Wrouter each; plus **QK-product additions** — Δ₁/Δ₂/Δ₃ srank + spec-entropy, c_qk normalized coupling, σ₁(Δ₃) vs BBP noise edge — per attention layer and per head.

**W tier** (every `w_cadence` steps): W Frobenius, srank, condition number, cos(W_t, W_0).

### 4.5 Δ₃ via core-matrix reduction (never materialize $d \times d$)

Given thin SVDs $\Delta W_q = U_q \Sigma_q V_q^T, \Delta W_k = U_k \Sigma_k V_k^T$ (computed once for ΔW tier and reused):

$$\Delta_3 = U_q \underbrace{\Sigma_q V_q^T V_k \Sigma_k}_{M \in \mathbb{R}^{d_k \times d_k}} U_k^T$$

All spectral indicators of Δ₃ reduce to indicators of M. Cost: O(d_k³) per head.

### 4.6 Output

One JSONL record per metric-sample event (flat schema; analysis pivots). Rich, dumb, cheap.

### 4.7 Cadence & overhead target

Default cadence {router: 10, ΔW: 50, W: 500}. Estimated overhead ~1.3% — within 2% budget. Escape hatch if over budget: raise ΔW cadence to 100, or use `svdvals` only when U/V not needed.

## 5. MindSpeed adapter

Thin glue. Three responsibilities: layer discovery, training-loop hook, FA swap for fault runs.

### 5.1 Layer discovery

MindSpeed-LLM uses Megatron-Core fused QKV linear. Slice `linear_qkv.weight` into Wq and Wk by head dimensions. MoE router accessors require a minimal patch (~10 LOC) to cache `last_probs` and `last_topk_indices`.

### 5.2 Training-loop hook

Monkey-patch `megatron/training/training.train_step` to call `monitor.step(iteration)` after the optimizer step. Fits the existing `run/training.patch` pattern.

### 5.3 FA patch (fault-injection runs only)

Replace MindSpeed fused FA with `fault_injection.fa_reference.FAWithFaultBackward` — a `torch.autograd.Function` that exposes the $O \odot dO$ intermediate. Applied only on E1, A1, E7. Slowdown ~30-40% on those runs only.

### 5.4 New CLI args

Monitor: `--enable-stability-monitor`, `--monitor-output-path`, `--monitor-{router,delta-w,w}-cadence`, `--monitor-delta-window`.

FA fault: `--inject-fa-fault`, `--fa-mantissa-mask-bits`.

Hyperparameter faults: reuse existing MindSpeed args (`--lr`, `--moe-aux-loss-coeff`, `--global-batch-size`).

### 5.5 Mantissa mask (bit-reinterpret)

```python
def mantissa_mask(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    mask = torch.tensor(~((1 << num_bits) - 1), dtype=torch.int16, device=x.device)
    return (x.view(torch.int16) & mask).view(torch.bfloat16)
```

Must be bit-exact-verified on NPU before E1a launches.

## 6. Analysis

Deferred to empirical: concrete figure inventory and detection-lag criteria are chosen after seeing data. Only structural commitments are fixed here:

1. **JSONL-out, notebook-in.** Rich flat logs from runtime; analysis is post-hoc.
2. **One-script-per-figure.** Each figure/table has a dedicated reproducible script taking run IDs as input.
3. **Reproducibility artifacts.** Configs, seeds, tokenizer version, MindSpeed + patch SHAs, data checksums, final JSONL logs — all released in a companion repo.
4. **≥2 seeds on headline runs.** Specific bands / confidence intervals chosen after seeing across-seed spread.

## 7. Open risks & milestones

### Known unknowns (resolved Week 1)

| Risk | Mitigation |
|---|---|
| Reference FA on NPU may diverge from fused FA beyond BF16 tolerance | 1000-step paired comparison test day 1; tolerable <1% loss delta |
| `torch.view(int16)` bit-reinterpret may behave differently on Ascend | Unit test + bit-compare on NPU before E1a |
| MoE router accessor patch may perturb throughput or correctness | Compare throughput + router load-balance pre/post patch |
| Fault onset timing at shrunk scale uncertain | E1a first run is a probe; extend to 40k steps or escalate bits if no collapse by 25k |
| E2-AUX fingerprint distinguishability from E2-LR is predicted, not verified | First E2-AUX run either confirms or refines the fingerprint table |

### Phased timeline

- **Week 1 — infrastructure & validation.** Stand up `stability_monitor` lib with tests, MindSpeed adapter + patches, reference FA on NPU, mantissa-mask bit test, tokenize data, smoke-test.
- **Week 2 — first long runs.** E3 seed 1 + E1a seed 1 in parallel; first eyeball analysis.
- **Week 3-4 — full sweep.** E3 seed 2, E1a seed 2, E2 variants, A1 variants, E1b mini-scale, E7.
- **Week 5 — analysis, figures, writeup.**

### Escape hatches

- **E1a no collapse by 25k steps:** extend to 40k or promote bits=6 to default.
- **E1b mini-scale shows no signature:** drop from paper or reframe.
- **Monitor overhead > 3%:** raise ΔW cadence to 100 or disable full SVD.
- **E2-AUX indistinguishable from E2-LR:** collapse to one category in the paper.

### Descope priorities (if compute is tight)

1. E2-GBS.
2. Second seed on E2-AUX.
3. A1 bits=2 variant (keep 0, 4, 6 endpoints).

## 8. Next step

Transition to the `writing-plans` skill to produce a concrete, ordered implementation plan with validation checkpoints. The plan should deliver Week 1's deliverables first (`stability_monitor` library + MindSpeed adapter + reference FA + bit-exact mantissa mask + tokenized data + smoke test) before any long experimental run commences.
