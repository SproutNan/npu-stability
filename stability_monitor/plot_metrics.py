#!/usr/bin/env python3
"""Visualize stability_monitor JSONL output.  No dependencies beyond matplotlib + stdlib.

Usage:
    python stability_monitor/plot_metrics.py metrics/monitor_20260428_120000.jsonl
    python stability_monitor/plot_metrics.py metrics/monitor.jsonl --out figs/ --metrics "router/*" "qk_dw/*"
    python stability_monitor/plot_metrics.py metrics/monitor.jsonl --step-min 100 --step-max 1000

Each metric that appears in multiple layers is plotted on one figure with per-layer
colours from viridis.
"""
import argparse
import json
import fnmatch
import os
import sys
from collections import defaultdict
from pathlib import Path

# ── colour helpers ───────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _layer_colours(n: int):
    """Return n distinct (r,g,b,a) tuples from viridis."""
    return [cm.viridis(i / max(n - 1, 1)) for i in range(n)]


# ── row normalisation ────────────────────────────────────────────────────────

# Groups where one JSON line carries multiple metric columns.
_MULTI_COLUMN_GROUPS = {
    "router": ("per_token_entropy", "max_violation",
               "router_diversification", "router_weight_cos_sim"),
}


def _normalise(records: list) -> list:
    """Convert every JSON record into a uniform {step, metric, value, layer_id} dict."""
    out = []

    for rec in records:
        step = rec.get("step")
        group = rec.get("metric_group", "")
        layer = rec.get("layer_id", "")

        if step is None:
            continue

        # ── multi-column groups: explode into one row per metric ──
        if group in _MULTI_COLUMN_GROUPS:
            for col in _MULTI_COLUMN_GROUPS[group]:
                val = rec.get(col)
                if val is not None:
                    out.append({"step": int(step),
                                "metric": f"{group}/{col}",
                                "value": float(val),
                                "layer_id": str(layer)})
            continue

        # ── dw_carry_forward: skip, these duplicate earlier dW rows ──
        if group == "dw_carry_forward":
            continue

        # ── single-value rows ──
        raw = rec.get("value")
        if raw is None:
            continue

        metric_name = group

        # Append sub-metric if present
        if "metric" in rec and isinstance(rec["metric"], str):
            metric_name = f"{group}/{rec['metric']}"

        # Prepend matrix if present (qkv_q / qkv_k / qkv_full)
        if "matrix" in rec and isinstance(rec["matrix"], str):
            if "metric" in rec and isinstance(rec["metric"], str):
                metric_name = f"{group}/{rec['matrix']}/{rec['metric']}"
            else:
                metric_name = f"{group}/{rec['matrix']}"

        # Append window size suffix when present
        if "window" in rec and isinstance(rec["window"], (int, float)):
            metric_name = f"{metric_name}_w{int(rec['window'])}"

        out.append({"step": int(step),
                    "metric": metric_name,
                    "value": float(raw),
                    "layer_id": str(layer)})

    return out


# ── figure generation ────────────────────────────────────────────────────────

def _plot_one_metric(series_by_layer: dict, metric_name: str, out_dir: str) -> str:
    """Plot a single metric across all layers.  Returns output path."""
    layers = sorted(series_by_layer.keys())
    n_layers = len(layers)
    colours = _layer_colours(max(n_layers, 1))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for idx, layer in enumerate(layers):
        points = series_by_layer[layer]
        points.sort(key=lambda x: x[0])  # sort by step
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=colours[idx], linewidth=0.7, alpha=0.85, label=layer)

    ax.set_title(metric_name, fontsize=10, family="monospace")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    if n_layers <= 32:
        ax.legend(fontsize=6, ncol=max(1, n_layers // 20), loc="best", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    safe_name = metric_name.replace("/", "_")
    path = os.path.join(out_dir, f"{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot stability_monitor JSONL metrics")
    parser.add_argument("jsonl", help="Path to JSONL file")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <jsonl>_plots/)")
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Only plot matching metric names (glob, e.g. 'router/*' 'qk_dw/*')")
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib is required.  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or (Path(args.jsonl).stem + "_plots")
    os.makedirs(out_dir, exist_ok=True)

    # ── load ──
    records = []
    with open(args.jsonl, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print(f"No records found in {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    # ── normalise ──
    normed = _normalise(records)

    # ── filter steps ──
    if args.step_min is not None:
        normed = [r for r in normed if r["step"] >= args.step_min]
    if args.step_max is not None:
        normed = [r for r in normed if r["step"] <= args.step_max]

    # ── group by metric → layer → [(step, value)] ──
    metric_map = defaultdict(lambda: defaultdict(list))
    for r in normed:
        metric_map[r["metric"]][r["layer_id"]].append((r["step"], r["value"]))

    available = sorted(metric_map.keys())

    # ── filter metrics ──
    if args.metrics:
        keep = set()
        for pat in args.metrics:
            keep.update([m for m in available if fnmatch.fnmatch(m, pat)])
        metrics = sorted(keep)
    else:
        metrics = available

    if not metrics:
        print("No metrics matched. Available metrics:")
        for m in available:
            print(f"  {m}")
        sys.exit(1)

    n_total = len(metrics)
    for i, metric in enumerate(metrics):
        path = _plot_one_metric(metric_map[metric], metric, out_dir)
        print(f"[{i + 1:3d}/{n_total}] {path}")

    print(f"\nDone — {n_total} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
