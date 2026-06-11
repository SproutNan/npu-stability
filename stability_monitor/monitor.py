import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .config import MonitorConfig
from .tracker import MultiWindowWeightTracker
from .jsonl_logger import JsonlLogger
from .spectral import calc_all_svd_metrics, calc_norm
from .router import (
    per_token_expert_entropy,
    max_violation,
    router_diversification,
    router_weight_cosine_similarity,
    router_weight_centered_frob_sq,
)
from .qk_product import calc_qk_dw_metrics


@dataclass
class _AttentionReg:
    layer_id: str
    layer_num: int
    layer: Any
    num_q_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass
class _RouterReg:
    layer_id: str
    layer_num: int
    layer: Any  # MoELayer
    num_experts: int
    top_k: int


class Monitor:
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.tracker = MultiWindowWeightTracker(list(cfg.delta_windows))
        self.logger = JsonlLogger(cfg.output_path)
        self._attn: List[_AttentionReg] = []
        self._router: List[_RouterReg] = []
        self._dw_last_values: Dict[str, float] = {}

    def register_attention(
        self, layer_id: str, layer_num: int, layer: Any,
        num_q_heads: int, num_kv_heads: int, head_dim: int,
    ):
        reg = _AttentionReg(
            layer_id, layer_num, layer, num_q_heads, num_kv_heads, head_dim
        )
        self._attn.append(reg)
        Wq, Wk, _, _ = self._get_qkv_weight(reg)
        self.tracker.init_step_zero(f"{layer_id}.qkv_q", Wq)
        self.tracker.init_step_zero(f"{layer_id}.qkv_k", Wk)

    def register_router(
        self, layer_id: str, layer_num: int, layer: Any,
        num_experts: int, top_k: int,
    ):
        reg = _RouterReg(
            layer_id, layer_num, layer, num_experts, top_k
        )
        self._router.append(reg)
        Wr = self._get_router_weight(reg)
        self.tracker.init_step_zero(f"{layer_id}.router", Wr)

    # ── Weight accessors ────────────────────────────────────────────────────

    @staticmethod
    def _get_qkv_weight(reg: _AttentionReg):
        W = reg.layer.self_attention.linear_qkv.weight
        q_rows = reg.num_q_heads * reg.head_dim
        k_rows = reg.num_kv_heads * reg.head_dim
        return W[:q_rows], W[q_rows : q_rows + k_rows], q_rows, k_rows

    @staticmethod
    def _get_router_weight(reg: _RouterReg):
        return reg.layer.mlp.router.weight

    # ── Main step ───────────────────────────────────────────────────────────

    def step(self, step: int) -> None:
        if step % self.cfg.router_cadence == 0:
            self._safe_call(self._router_step, step, "router")
        # delta_w is called at every step; the tracker gates per-window internally
        self._safe_call(self._delta_w_step, step, "delta_w")
        if step % self.cfg.w_cadence == 0:
            self._safe_call(self._w_step, step, "w")

    @staticmethod
    def _safe_call(fn, step: int, tag: str) -> None:
        try:
            fn(step)
        except Exception as e:
            import sys
            import traceback
            print(f"[stability_monitor] {tag}({step}) failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # ── Router tier ─────────────────────────────────────────────────────────

    def _router_step(self, step: int) -> None:
        dp_group = _get_dp_group()
        for reg in self._router:
            router = reg.layer.mlp.router
            logits = getattr(router, "_monitor_logits", None)
            scores = getattr(router, "_monitor_scores", None)
            if logits is None or scores is None:
                continue

            # Take ownership of the cached tensors so the next forward pass
            # can store fresh values without leaking GPU memory.
            router._monitor_logits = None
            router._monitor_scores = None

            # Full softmax scores needed for per_token_expert_entropy (the
            # routing scores returned by topk_softmax_with_capacity are
            # top-k masked, which would yield incorrect entropy).
            full_scores = torch.softmax(logits.to(torch.float32), dim=-1)

            W = self._get_router_weight(reg)

            self.logger.log({
                "step": step,
                "metric_group": "router",
                "layer_id": reg.layer_id,
                "per_token_entropy": per_token_expert_entropy(logits, full_scores, dp_group),
                "max_violation": max_violation(logits, reg.top_k, dp_group),
                "router_diversification": router_diversification(
                    scores, reg.top_k, True, dp_group
                ),
                "router_weight_cos_sim": router_weight_cosine_similarity(W),
                "router_weight_c_frob_sq": router_weight_centered_frob_sq(W),
            })

            svd = calc_all_svd_metrics(W)
            for k, v in svd.items():
                self.logger.log({
                    "step": step,
                    "metric_group": "router_weight_svd",
                    "layer_id": reg.layer_id,
                    "metric": k,
                    "value": v,
                })

            self.logger.log({
                "step": step,
                "metric_group": "router_weight_norm",
                "layer_id": reg.layer_id,
                "value": calc_norm(W),
            })

    # ── Delta-W tier ────────────────────────────────────────────────────────

    def _delta_w_step(self, step: int) -> None:
        if step == 0:
            return  # W_0 was captured at registration time

        for reg in self._attn:
            q_key = f"{reg.layer_id}.qkv_q"
            k_key = f"{reg.layer_id}.qkv_k"
            Wq, Wk, q_rows, k_rows = self._get_qkv_weight(reg)

            dWq_by_w = self.tracker.update(q_key, Wq, step)
            dWk_by_w = self.tracker.update(k_key, Wk, step)

            common = set(dWq_by_w.keys()) & set(dWk_by_w.keys())
            for w in sorted(common):
                self._log_dw_metrics(
                    step, reg, Wq, Wk, dWq_by_w[w], dWk_by_w[w],
                    q_rows, k_rows, w,
                )

            # Carry-forward for windows where only one of Q/K produced a delta
            for w in sorted(set(dWq_by_w.keys()) - common):
                self._carry_forward_dw(step, reg, w)
            for w in sorted(set(dWk_by_w.keys()) - common):
                self._carry_forward_dw(step, reg, w)

        for r_reg in self._router:
            r_key = f"{r_reg.layer_id}.router"
            Wr = self._get_router_weight(r_reg)

            dWr_by_w = self.tracker.update(r_key, Wr, step)
            for w, dWr in sorted(dWr_by_w.items()):
                self._log_dw_router_metrics(step, r_reg, dWr, w)

        # Advance window pointers and prune old snapshots — once per step
        self.tracker.finalize_step(step)

    def _log_dw_router_metrics(self, step, r_reg, dWr, window):
        w = window
        svd = calc_all_svd_metrics(dWr)
        for k, v in svd.items():
            key = f"router_dw_{k}_{r_reg.layer_id}_w{w}"
            self._dw_last_values[key] = v
            self.logger.log({
                "step": step,
                "metric_group": "router_weight_dw_svd",
                "layer_id": r_reg.layer_id,
                "metric": k,
                "value": v,
                "window": w,
            })
        norm_v = calc_norm(dWr)
        key = f"router_dw_norm_{r_reg.layer_id}_w{w}"
        self._dw_last_values[key] = norm_v
        self.logger.log({
            "step": step,
            "metric_group": "router_weight_dw_norm",
            "layer_id": r_reg.layer_id,
            "value": norm_v,
            "window": w,
        })

    def _log_dw_metrics(self, step, reg, Wq, Wk, dWq, dWk, q_rows, k_rows, window):
        w = window
        for name, dW in (("qkv_q", dWq), ("qkv_k", dWk)):
            svd = calc_all_svd_metrics(dW)
            for k, v in svd.items():
                key = f"qkv_dw_{name}_{k}_{reg.layer_id}_w{w}"
                self._dw_last_values[key] = v
                self.logger.log({
                    "step": step,
                    "metric_group": "qkv_weight_dw_svd",
                    "layer_id": reg.layer_id,
                    "matrix": name,
                    "metric": k,
                    "value": v,
                    "window": w,
                })
            norm_v = calc_norm(dW)
            key = f"qkv_dw_{name}_norm_{reg.layer_id}_w{w}"
            self._dw_last_values[key] = norm_v
            self.logger.log({
                "step": step,
                "metric_group": "qkv_weight_dw_norm",
                "layer_id": reg.layer_id,
                "matrix": name,
                "value": norm_v,
                "window": w,
            })

        # Full QK dW SVD
        dW_full = torch.cat([dWq, dWk], dim=0)
        svd = calc_all_svd_metrics(dW_full)
        for k, v in svd.items():
            key = f"qkv_dw_full_{k}_{reg.layer_id}_w{w}"
            self._dw_last_values[key] = v
            self.logger.log({
                "step": step,
                "metric_group": "qkv_weight_dw_svd",
                "layer_id": reg.layer_id,
                "matrix": "qkv_full",
                "metric": k,
                "value": v,
                "window": w,
            })
        key = f"qkv_dw_full_norm_{reg.layer_id}_w{w}"
        norm_v = calc_norm(dW_full)
        self._dw_last_values[key] = norm_v
        self.logger.log({
            "step": step,
            "metric_group": "qkv_weight_dw_norm",
            "layer_id": reg.layer_id,
            "matrix": "qkv_full",
            "value": norm_v,
            "window": w,
        })

        # QK-product delta metrics (per-head, max-aggregated)
        if self.cfg.per_head:
            lookback = self.tracker.get_lookback_step(w)
            Wq_prev = self.tracker.get_snapshot(f"{reg.layer_id}.qkv_q", lookback)
            Wk_prev = self.tracker.get_snapshot(f"{reg.layer_id}.qkv_k", lookback)
            if Wq_prev is not None and Wk_prev is not None:
                qk_current = torch.cat([Wq, Wk], dim=0)
                qk_prev = torch.cat(
                    [Wq_prev.to(Wq.device), Wk_prev.to(Wk.device)], dim=0
                )
                qk_metrics = calc_qk_dw_metrics(
                    qk_current, qk_prev, reg.head_dim, q_rows, k_rows
                )
                for k, v in qk_metrics.items():
                    key = f"qk_dw_{k}_{reg.layer_id}_w{w}"
                    self._dw_last_values[key] = v
                    self.logger.log({
                        "step": step,
                        "metric_group": "qk_dw",
                        "layer_id": reg.layer_id,
                        "metric": k,
                        "value": v,
                        "window": w,
                    })

    def _carry_forward_dw(self, step, reg, window):
        suffix = f"_{reg.layer_id}_w{window}"
        for key, value in self._dw_last_values.items():
            if key.endswith(suffix):
                self.logger.log({
                    "step": step,
                    "metric_group": "dw_carry_forward",
                    "layer_id": reg.layer_id,
                    "key": key,
                    "value": value,
                    "window": window,
                })

    # ── W tier ──────────────────────────────────────────────────────────────

    def _w_step(self, step: int) -> None:
        for reg in self._attn:
            Wq, Wk, _, _ = self._get_qkv_weight(reg)
            for name, W in (("qkv_q", Wq), ("qkv_k", Wk)):
                svd = calc_all_svd_metrics(W)
                for k, v in svd.items():
                    self.logger.log({
                        "step": step,
                        "metric_group": "w",
                        "layer_id": reg.layer_id,
                        "matrix": name,
                        "metric": k,
                        "value": v,
                    })
                self.logger.log({
                    "step": step,
                    "metric_group": "w_norm",
                    "layer_id": reg.layer_id,
                    "matrix": name,
                    "value": calc_norm(W),
                })

        for r_reg in self._router:
            W = self._get_router_weight(r_reg)
            svd = calc_all_svd_metrics(W)
            for k, v in svd.items():
                self.logger.log({
                    "step": step,
                    "metric_group": "w_router",
                    "layer_id": r_reg.layer_id,
                    "metric": k,
                    "value": v,
                })
            self.logger.log({
                "step": step,
                "metric_group": "w_router_norm",
                "layer_id": r_reg.layer_id,
                "value": calc_norm(W),
            })

    def close(self) -> None:
        self.logger.close()


def _get_dp_group():
    if not dist.is_available() or not dist.is_initialized():
        return None
    try:
        from megatron.core import parallel_state as mpu
        return mpu.get_data_parallel_group()
    except (ImportError, AssertionError):
        return None
