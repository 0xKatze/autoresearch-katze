#!/usr/bin/env python3
"""
report_charts.py — Generate presentation-quality figures for the N24
diffusion node-injection attack session report.

Outputs to report/figures/*.png at 200 DPI, colorblind-safe palette.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Palette (Ocean + crimson attack accent) ----
NAVY = "#21295C"
TEAL = "#1C7293"
BLUE = "#065A82"
CRIMSON = "#E63946"
GREEN = "#2A9D8F"
GOLD = "#E9C46A"
LIGHT = "#F4F1DE"
GREY = "#8D99AE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 200,
})


def chart_asr_progression():
    """The tuning stack: master baseline → 98.43% across 5 steps."""
    steps = [
        "Master\nbaseline",
        "+ Joint\nX+A diff",
        "+ Budget\n≤ k",
        "+ k_sweep",
        "+ Per-node\nescalation",
    ]
    asr = [94.85, 97.71, 97.84, 98.07, 98.43]
    std = [7.44, 3.46, 3.22, 3.01, 2.29]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(steps))
    ax.fill_between(x, np.array(asr) - np.array(std), np.array(asr) + np.array(std),
                    color=TEAL, alpha=0.15, label="±1 std (5-fold)")
    ax.plot(x, asr, "-o", color=NAVY, linewidth=2.8, markersize=10,
            markerfacecolor=CRIMSON, markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)

    for xi, a in zip(x, asr):
        ax.annotate(f"{a:.2f}%", (xi, a), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontweight="bold",
                    fontsize=12, color=NAVY)
    # Delta callout
    ax.annotate("", xy=(4, 98.43), xytext=(0, 94.85),
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.6,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(2.0, 96.0, "+3.58 pp\nstd −69%", color=CRIMSON, fontsize=13,
            fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=10.5)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(92, 100)
    ax.set_title("Tuning Stack: PROTEINS GCN  (94.85% → 98.43%)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart_asr_progression.png", bbox_inches="tight")
    plt.close(fig)


def chart_mechanism_evolution():
    """Diffusion went from inert (67%) to effective (97.7%) only when
    topology became part of the state."""
    variants = [
        "v1\ncosine DDIM\n(x0=0)",
        "v2\nbounded DDIM\n(x0=0)",
        "v3\nfeature-only\nhybrid",
        "N24\njoint X+A\ndiffusion",
    ]
    asr = [67.31, 67.31, 94.85, 97.71]
    colors = [GREY, GREY, GOLD, GREEN]
    labels = ["INERT", "INERT", "0 diff saves", "WORKS"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(variants))
    bars = ax.bar(x, asr, color=colors, width=0.62,
                  edgecolor="white", linewidth=1.5)
    # trivial baseline line + annotation parked in empty space (x=1.5 gap)
    ax.axhline(67.31, color=CRIMSON, linestyle=":", linewidth=1.8)
    ax.annotate("trivial zero-feature\nbaseline (67.31%)",
                xy=(1.5, 67.31), xytext=(1.5, 80),
                ha="center", color=CRIMSON, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.3))

    for xi, a, lab in zip(x, asr, labels):
        # value label: lift v1/v2 clear of the dotted line
        yoff = 3.0 if a < 70 else 1.5
        ax.text(xi, a + yoff, f"{a:.1f}%", ha="center", fontweight="bold",
                fontsize=12, color=NAVY)
        ax.text(xi, a / 2, lab, ha="center", va="center", fontweight="bold",
                fontsize=11, color="white", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Why Diffusion Failed Three Times, Then Worked")
    fig.text(0.5, -0.02,
             "Key: pure feature-space diffusion is inert — only adding the "
             "EDGE topology to the diffusion state breaks the ceiling.",
             ha="center", fontsize=10, style="italic", color=NAVY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart_mechanism_evolution.png", bbox_inches="tight")
    plt.close(fig)


def chart_diffusion_contribution():
    """Per-fold stacked: baseline solo vs each diffusion tier's saves."""
    folds = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]
    baseline = [156, 133, 165, 159, 156]
    joint = [0, 18, 7, 0, 0]
    ksweep = [0, 1, 1, 0, 0]
    pernode = [0, 3, 0, 0, 0]
    totals = [156, 165, 175, 160, 156]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(folds))
    w = 0.6
    b1 = ax.bar(x, baseline, w, label="Baseline (5-restart Adam)",
                color=NAVY, edgecolor="white")
    b2 = ax.bar(x, joint, w, bottom=baseline, label="Joint X+A diffusion",
                color=TEAL, edgecolor="white")
    bot2 = np.array(baseline) + np.array(joint)
    b3 = ax.bar(x, ksweep, w, bottom=bot2, label="k_sweep",
                color=GOLD, edgecolor="white")
    bot3 = bot2 + np.array(ksweep)
    b4 = ax.bar(x, pernode, w, bottom=bot3, label="Per-node escalation",
                color=CRIMSON, edgecolor="white")

    # unflipped marker
    for xi, tot, got in zip(x, totals,
                            np.array(baseline)+np.array(joint)+np.array(ksweep)+np.array(pernode)):
        miss = tot - got
        if miss > 0:
            ax.text(xi, tot + 2, f"{miss} stuck", ha="center", fontsize=9,
                    color=CRIMSON, fontweight="bold")
        ax.text(xi, got - 10, f"{got}/{tot}", ha="center", fontsize=9.5,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Graphs flipped")
    ax.set_title("Where Each Attack Tier Contributes (GCN, per fold)")
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9.5,
              bbox_to_anchor=(0.5, -0.30))
    ax.set_ylim(0, 195)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart_diffusion_contribution.png", bbox_inches="tight")
    plt.close(fig)


def chart_robustness():
    """ASR per victim architecture — readout is the attack surface."""
    archs = ["GIN", "SAGE", "GAT", "GCN\n(mean)", "GCN\n(median)"]
    asr = [100.0, 100.0, 99.88, 98.43, 92.46]
    std = [0.0, 0.0, 0.24, 2.29, 6.79]
    # color: vulnerable (crimson) → robust (green)
    colors = [CRIMSON, CRIMSON, CRIMSON, "#C1666B", GREEN]
    readout = ["mean", "mean", "mean", "mean", "median"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(archs))[::-1]
    bars = ax.barh(y, asr, xerr=std, color=colors, edgecolor="white",
                   height=0.62, error_kw=dict(ecolor=NAVY, capsize=4, lw=1.2))
    for yi, a, r in zip(y, asr, readout):
        ax.text(a - 2.5, yi, f"{a:.1f}%", va="center", ha="right",
                color="white", fontweight="bold", fontsize=12)
        ax.text(2, yi, f"readout: {r}", va="center", ha="left",
                color="white", fontsize=9.5, style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(archs, fontsize=11)
    ax.set_xlabel("Attack Success Rate (%)")
    ax.set_xlim(0, 113)
    ax.set_title("Transferability: Mean-Pool is the Attack Surface")
    legend = [Patch(facecolor=CRIMSON, label="Vulnerable (mean-pool)"),
              Patch(facecolor=GREEN, label="Resistant (median-pool)")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart_robustness.png", bbox_inches="tight")
    plt.close(fig)


def chart_defense_breakdown():
    """GCN-median: baseline collapses, diffusion recovers ASR."""
    folds = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]
    baseline = [138, 81, 99, 130, 121]
    diff_total = [0, 23, 7, 1, 1]   # joint+ksweep+pernode saves
    totals = [138, 128, 119, 140, 124]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(folds))
    w = 0.6
    base_pct = 100 * np.array(baseline) / np.array(totals)
    final_pct = 100 * (np.array(baseline) + np.array(diff_total)) / np.array(totals)

    ax.bar(x, base_pct, w, label="Baseline solo ASR", color=GREY,
           edgecolor="white")
    ax.bar(x, final_pct - base_pct, w, bottom=base_pct,
           label="Recovered by diffusion", color=GREEN, edgecolor="white")

    for xi, b, f in zip(x, base_pct, final_pct):
        ax.text(xi, b - 5, f"{b:.0f}%", ha="center", color="white",
                fontsize=9.5, fontweight="bold")
        ax.text(xi, f + 1.5, f"{f:.0f}%", ha="center", color=NAVY,
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Defeating the Median-Pool Defense (GCN-median)")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.text(0.5, -0.02,
             "Median readout collapses the baseline (~63% on fold 1); "
             "joint diffusion recovers overall ASR to 92.46%.",
             ha="center", fontsize=10, style="italic", color=NAVY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chart_defense_breakdown.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    chart_asr_progression()
    chart_mechanism_evolution()
    chart_diffusion_contribution()
    chart_robustness()
    chart_defense_breakdown()
    print("Charts written to", FIG_DIR)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" ", p.name)
