"""Integration glue between stability_monitor and MindSpeed-LLM/Megatron-Core.

Entry points:
1. add_stability_monitor_args(parser) — CLI argument registration
2. init_monitor_from_args(args, model) — called after model init
3. monitor_step(iteration) — called after each training step

Environment variables:
  STABILITY_MONITOR_ENABLED=1       master enable (alternative to --enable-stability-monitor)
  STABILITY_MONITOR_OUTPUT_PATH     default JSONL path (overridden by --monitor-output-path)
"""

import os
from typing import Any, Optional

_monitor: Optional[Any] = None

DEFAULT_OUTPUT = "stability_monitor_metrics.jsonl"


def _is_enabled(args) -> bool:
    if args.enable_stability_monitor:
        return True
    env_val = os.environ.get("STABILITY_MONITOR_ENABLED", "").strip().lower()
    return env_val in ("1", "true", "yes", "on")


def _resolve_output_path(args) -> str:
    path = args.monitor_output_path
    if path:
        return path
    return os.environ.get("STABILITY_MONITOR_OUTPUT_PATH", DEFAULT_OUTPUT)


def add_stability_monitor_args(parser):
    """Register CLI arguments for the stability monitor."""
    group = parser.add_argument_group("stability-monitor")
    group.add_argument("--enable-stability-monitor", action="store_true",
                       help="Enable training stability monitoring "
                            "(or set STABILITY_MONITOR_ENABLED=1)")
    group.add_argument("--monitor-output-path", type=str, default=None,
                       help="JSONL output path for monitor records "
                            "(or set STABILITY_MONITOR_OUTPUT_PATH)")
    group.add_argument("--monitor-router-cadence", type=int, default=10,
                       help="Steps between router metric computation")
    group.add_argument("--monitor-delta-windows", type=int, nargs="+",
                       default=[5, 10, 20, 50, 100, 500, 1000],
                       help="Window sizes for dW = W_t - W_{t-w}. "
                            "Each size acts as its own cadence "
                            "(computed when step %% window == 0)")
    group.add_argument("--monitor-w-cadence", type=int, default=500,
                       help="Steps between W health metric computation")
    group.add_argument("--monitor-per-head", action="store_true", default=True,
                       help="Enable per-head QK-product decomposition")
    return parser


def init_monitor_from_args(args, model) -> None:
    """Initialize the global monitor if enabled (via CLI flag or env var)."""
    global _monitor
    if not _is_enabled(args):
        return

    from .config import MonitorConfig
    from .monitor import Monitor

    cfg = MonitorConfig(
        router_cadence=args.monitor_router_cadence,
        delta_windows=tuple(args.monitor_delta_windows),
        w_cadence=args.monitor_w_cadence,
        per_head=args.monitor_per_head,
        output_path=_resolve_output_path(args),
    )
    _monitor = Monitor(cfg)
    register_all_layers(model, _monitor, args)
    _monitor.tracker.finalize_step_zero()


def monitor_step(iteration: int) -> None:
    """Call after each training step. Safe to call when monitor is disabled."""
    if _monitor is not None:
        try:
            _monitor.step(iteration)
        except Exception as e:
            import sys
            print(
                f"[stability_monitor] step({iteration}) failed: {e}",
                file=sys.stderr,
            )


def monitor_close() -> None:
    global _monitor
    if _monitor is not None:
        _monitor.close()
        _monitor = None


def _wrap_router_forward(router):
    """Capture router logits and scores via gating-patch + forward hook.

    Logits are captured by wrapping the ``gating`` method.  Scores are
    captured via a PyTorch forward hook on the router module.  Both
    intermediates are stored as ``_monitor_logits`` / ``_monitor_scores``
    on the router instance and consumed by ``_router_step``.
    """

    # ── Capture logits by wrapping gating ───────────────────────────────
    _orig_gating = router.gating

    def _patched_gating(input_tensor):
        logits = _orig_gating(input_tensor)
        router._monitor_logits = logits.detach()
        return logits

    router.gating = _patched_gating

    # ── Capture scores via forward hook ─────────────────────────────────
    def _capture_scores(module, _input, output):
        scores, _routing_map = output
        module._monitor_scores = scores.detach()

    router.register_forward_hook(_capture_scores)


def register_all_layers(model: Any, monitor: Any, cfg: Any) -> None:
    """Discover TransformerLayers in the model and register with the monitor."""

    # Unwrap Megatron wrappers: DDP(Float16Module(GPTModel)) -> GPTModel
    while hasattr(model, "module"):
        model = model.module

    decoder = model.decoder

    for idx, layer in enumerate(decoder.layers):
        layer_id = f"layer.{idx}"

        monitor.register_attention(
            layer_id=f"{layer_id}.attention",
            layer_num=idx,
            layer=layer,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_query_groups,
            head_dim=cfg.kv_channels,
        )

        first_k = getattr(cfg, "first_k_dense_replace", 0)
        if idx >= first_k:
            monitor.register_router(
                layer_id=f"{layer_id}.router",
                layer_num=idx,
                layer=layer,
                num_experts=cfg.num_experts,
                top_k=cfg.moe_router_topk,
            )
            _wrap_router_forward(layer.mlp.router)
