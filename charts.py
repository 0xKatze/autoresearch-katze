#!/usr/bin/env python3
"""
charts.py -- Auto-generated experiment charts with error bars and confidence intervals.

Inspired by AutoResearchClaw's visualize.py. Generates:
1. Condition comparison bar chart (config variants with 95% CI)
2. ASR trajectory over experiment history
3. Fold variance heatmap
4. Parameter sensitivity analysis
5. Quick-vs-full comparison

All charts use colorblind-safe palette and 300 DPI.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR = Path(__file__).parent / "charts"

# Paul Tol colorblind-safe palette
COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
]


def _ensure_matplotlib():
    """Import matplotlib with Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_results() -> List[Dict]:
    """Load all results sorted by timestamp."""
    results = []
    for p in sorted(RESULTS_DIR.glob("exp_*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda r: r.get("timestamp", ""))
    return results


def _ci95(values: List[float]) -> Tuple[float, float, float]:
    """Compute mean and 95% confidence interval."""
    n = len(values)
    if n == 0:
        return 0, 0, 0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = std / math.sqrt(n)
    t_val = 2.776 if n <= 5 else 2.262 if n <= 10 else 1.96
    ci_low = mean - t_val * se
    ci_high = mean + t_val * se
    return mean, ci_low, ci_high


def plot_asr_trajectory(output_dir: Optional[Path] = None) -> Path:
    """Plot ASR trajectory over experiment history with 5-fold and 1-fold markers."""
    plt = _ensure_matplotlib()
    results = _load_results()
    out = output_dir or CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Separate quick and full runs
    quick_x, quick_y = [], []
    full_x, full_y, full_err_lo, full_err_hi = [], [], [], []

    for i, r in enumerate(results):
        asr = r.get("mean_asr", 0)
        nf = r.get("n_folds", 1)
        if nf == 1:
            quick_x.append(i)
            quick_y.append(asr * 100)
        else:
            full_x.append(i)
            full_y.append(asr * 100)
            std = r.get("std_asr", 0)
            full_err_lo.append(std * 100)
            full_err_hi.append(std * 100)

    if quick_x:
        ax.scatter(quick_x, quick_y, c=COLORS[4], s=30, alpha=0.6,
                   label="1-fold (quick)", zorder=3, marker="o")
    if full_x:
        ax.errorbar(full_x, full_y, yerr=[full_err_lo, full_err_hi],
                     fmt="s", c=COLORS[0], markersize=6, capsize=3,
                     label="5-fold (full)", zorder=4)

    # Best 5-fold line
    if full_y:
        best_full = max(full_y)
        ax.axhline(y=best_full, color=COLORS[2], linestyle="--", alpha=0.5,
                    label=f"Best 5-fold: {best_full:.1f}%")

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("Attack Success Rate Trajectory", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    path = out / "asr_trajectory.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_config_comparison(output_dir: Optional[Path] = None) -> Path:
    """Bar chart comparing ASR across different config variants with 95% CI error bars."""
    plt = _ensure_matplotlib()
    results = _load_results()
    out = output_dir or CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Group 5-fold results by key config dimensions
    groups: Dict[str, List[float]] = {}
    for r in results:
        if r.get("n_folds", 1) < 5:
            continue
        cfg = r.get("config", {})
        # Create a label from distinctive config values
        edge = cfg.get("edge_strategy", "?")
        loss = cfg.get("loss_type", "?")
        fs = cfg.get("feat_scale", "?")
        label = f"{edge}/{loss}/{fs}"
        groups.setdefault(label, [])
        groups[label].append(r.get("mean_asr", 0) * 100)

    if not groups:
        # Create empty chart
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 5-fold results", ha="center", va="center")
        path = out / "config_comparison.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    # Sort by mean ASR
    sorted_groups = sorted(groups.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    sorted_groups = sorted_groups[:12]  # Top 12

    labels = [g[0] for g in sorted_groups]
    means = []
    ci_lows = []
    ci_highs = []

    for _, values in sorted_groups:
        m, lo, hi = _ci95(values)
        means.append(m)
        ci_lows.append(m - lo)
        ci_highs.append(hi - m)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=[ci_lows, ci_highs], capsize=4,
                  color=COLORS[0], edgecolor="white", alpha=0.85)

    # Value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("Config Variant Comparison (5-fold, 95% CI)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    path = out / "config_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fold_variance(output_dir: Optional[Path] = None) -> Path:
    """Heatmap of per-fold ASR across experiments."""
    plt = _ensure_matplotlib()
    results = _load_results()
    out = output_dir or CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Collect 5-fold results with per_fold_asr
    data_rows = []
    labels = []
    for r in results:
        if r.get("n_folds", 1) < 5 and "per_fold_asr" not in r:
            continue
        per_fold = r.get("per_fold_asr", [])
        if len(per_fold) < 2:
            continue
        ts = r.get("timestamp", "?")[-6:]
        h = r.get("attack_hash", "?")[:4]
        labels.append(f"{ts}_{h}")
        data_rows.append([v * 100 for v in per_fold])

    if not data_rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No multi-fold results", ha="center", va="center")
        path = out / "fold_variance.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    data = np.array(data_rows[-15:])  # Last 15 experiments
    labels = labels[-15:]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4)))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=50, vmax=100)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"Fold {i}" for i in range(data.shape[1])], fontsize=9)
    ax.set_title("Per-Fold ASR Heatmap (%)", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val > 85 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="ASR %")
    path = out / "fold_variance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_param_sensitivity(param: str = "sigma",
                           output_dir: Optional[Path] = None) -> Path:
    """Scatter plot showing how a parameter affects ASR."""
    plt = _ensure_matplotlib()
    results = _load_results()
    out = output_dir or CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    x_vals, y_vals, sizes = [], [], []
    for r in results:
        cfg = r.get("config", {})
        val = cfg.get(param)
        if val is None or isinstance(val, str):
            continue
        asr = r.get("mean_asr", 0) * 100
        nf = r.get("n_folds", 1)
        x_vals.append(float(val))
        y_vals.append(asr)
        sizes.append(80 if nf >= 5 else 30)

    fig, ax = plt.subplots(figsize=(8, 5))
    if x_vals:
        ax.scatter(x_vals, y_vals, s=sizes, c=COLORS[0], alpha=0.6, edgecolors="white")
        # Add trend line if enough points
        if len(x_vals) >= 3:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(x_vals), max(x_vals), 50)
            ax.plot(x_range, p(x_range), "--", c=COLORS[1], alpha=0.5, label="trend")

    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title(f"Parameter Sensitivity: {param}", fontsize=13, fontweight="bold")
    if x_vals and max(x_vals) / max(min(x_vals), 1e-10) > 100:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = out / f"param_sensitivity_{param}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_quick_vs_full(output_dir: Optional[Path] = None) -> Path:
    """Scatter plot comparing 1-fold vs 5-fold ASR for same configs."""
    plt = _ensure_matplotlib()
    results = _load_results()
    out = output_dir or CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Group by attack hash
    by_hash: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        h = r.get("attack_hash", "")
        nf = r.get("n_folds", 1)
        asr = r.get("mean_asr", 0) * 100
        by_hash.setdefault(h, {"quick": [], "full": []})
        if nf == 1:
            by_hash[h]["quick"].append(asr)
        elif nf >= 5:
            by_hash[h]["full"].append(asr)

    quick_vals, full_vals = [], []
    for h, d in by_hash.items():
        if d["quick"] and d["full"]:
            quick_vals.append(sum(d["quick"]) / len(d["quick"]))
            full_vals.append(sum(d["full"]) / len(d["full"]))

    fig, ax = plt.subplots(figsize=(6, 6))
    if quick_vals:
        ax.scatter(quick_vals, full_vals, c=COLORS[0], s=60, alpha=0.7, edgecolors="white")
        # Diagonal line
        lims = [min(min(quick_vals), min(full_vals)) - 2,
                max(max(quick_vals), max(full_vals)) + 2]
        ax.plot(lims, lims, "--", c=COLORS[6], alpha=0.5, label="y=x")
        # Bias line
        ax.plot(lims, [l - 4.86 for l in lims], ":", c=COLORS[1], alpha=0.5,
                label="bias (-4.86pp)")

    ax.set_xlabel("1-fold ASR (%)", fontsize=11)
    ax.set_ylabel("5-fold ASR (%)", fontsize=11)
    ax.set_title("Quick vs Full Evaluation", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = out / "quick_vs_full.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_charts(output_dir: Optional[Path] = None) -> List[Path]:
    """Generate all available charts. Returns list of created file paths."""
    out = output_dir or CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    charts = [
        ("ASR trajectory", plot_asr_trajectory),
        ("Config comparison", plot_config_comparison),
        ("Fold variance heatmap", plot_fold_variance),
        ("Quick vs full", plot_quick_vs_full),
    ]

    # Parameter sensitivity for key numeric params
    for param in ["sigma", "gen_lr", "kappa"]:
        charts.append((f"Sensitivity: {param}",
                        lambda o=out, p=param: plot_param_sensitivity(p, o)))

    for name, fn in charts:
        try:
            import inspect
            sig = inspect.signature(fn)
            # Lambdas have no params; named functions take output_dir
            if sig.parameters:
                path = fn(out)
            else:
                path = fn()
            paths.append(path)
            print(f"  [OK] {name}: {path.name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    return paths


if __name__ == "__main__":
    print("Generating all charts...")
    paths = generate_all_charts()
    print(f"\nGenerated {len(paths)} charts in {CHARTS_DIR}/")
