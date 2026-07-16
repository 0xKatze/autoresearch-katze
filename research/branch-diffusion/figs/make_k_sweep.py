"""K-sweep curve from multinode_20260524_061442.json (K ∈ {1,2,3} only — honest about gap)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
data = json.load(open(ROOT / "results/multinode_20260524_061442.json"))["results"]

# Index: (m, alloc, tau, feat) -> (asr_mean, asr_std, q_per_graph)
idx = {(r["m"], r["alloc"], r["tau"], r["feat_mode"]):
       (r["asr_mean"], r["asr_std"], r["q_per_graph"]) for r in data}

# K=1 uses alloc="single" — use it as the K=1 point for partition/duplicate/interconnect
def lookup(K, alloc, tau, feat):
    if K == 1:
        return idx[(1, "single", tau, feat)]
    return idx.get((K, alloc, tau, feat), (None, None, None))

Ks = [1, 2, 3]
allocs = ["partition", "duplicate", "interconnect"]
alloc_styles = {
    "partition":     dict(color="#36d399", marker="o", linestyle="-",  lw=2.2, ms=8),
    "duplicate":     dict(color="#6b7280", marker="s", linestyle="--", lw=1.6, ms=7),
    "interconnect":  dict(color="#4f8cff", marker="^", linestyle=":",  lw=1.6, ms=8),
}

# 2x2 panel
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
panels = [
    (0, 0, "zo_free",  0.5, "ASR @ τ=0.5 (zo_free — optimized features)"),
    (0, 1, "zo_free",  1.0, "ASR @ τ=1.0 (zo_free — optimized features)"),
    (1, 0, "manifold", 0.5, "ASR @ τ=0.5 (manifold — fixed clean-mean features)"),
    (1, 1, "manifold", 1.0, "ASR @ τ=1.0 (manifold — fixed clean-mean features)"),
]

for r, c, feat, tau, title in panels:
    ax = axes[r][c]
    for alloc in allocs:
        means = []
        stds = []
        for K in Ks:
            asr, std, _ = lookup(K, alloc, tau, feat)
            means.append(asr)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)
        sty = alloc_styles[alloc]
        ax.errorbar(Ks, means, yerr=stds, label=alloc, capsize=4, **sty)
        # Highlight K=1 in red
        ax.scatter([1], [means[0]], s=140, color="#ef4444",
                   zorder=10, edgecolor="black", linewidth=1.0)

    # Compact "future work" hint outside data region
    ax.axvspan(3.4, 4.0, color="#fde68a", alpha=0.30, zorder=-3)
    ax.text(3.7, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.50,
            "K ≥ 5\nfuture\nwork", ha="center", va="center",
            fontsize=8, color="#92400e", style="italic")
    ax.set_xlim(0.5, 4.0)
    ax.set_xticks([1, 2, 3])

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("ASR")
    if r == 1:
        ax.set_xlabel("K (number of injected nodes)")
    ax.grid(linestyle=":", alpha=0.4)
    if r == 0 and c == 0:
        ax.legend(loc="lower right", framealpha=0.95)

    # Annotate K=1 self-dilution direction on zo_free panels
    if feat == "zo_free":
        asr_k1 = lookup(1, "partition", tau, feat)[0]
        ax.annotate(f"K=1\n{asr_k1*100:.1f}%",
                    xy=(1, asr_k1), xytext=(1.15, asr_k1 - 0.05),
                    fontsize=10, color="#ef4444", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2))

fig.suptitle("K-sweep curve — PROTEINS GCN, 5-fold (multinode_20260524)",
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(ROOT / "figs/fig6_k_sweep.png")
plt.savefig(ROOT / "figs/fig6_k_sweep.pdf")
plt.close()
print("✓ fig6_k_sweep.{png,pdf}")


# ============================================================
# Companion: zo_free standalone (headline single-panel for slide use)
# ============================================================
fig, ax = plt.subplots(figsize=(7.4, 4.5))
for tau, line_style, marker, label_suffix in [
    (0.5, "-",  "o", "τ=0.5"),
    (1.0, "--", "s", "τ=1.0"),
]:
    means_part = []
    stds_part = []
    for K in Ks:
        asr, std, _ = lookup(K, "partition", tau, "zo_free")
        means_part.append(asr)
        stds_part.append(std)
    color = "#36d399" if tau == 0.5 else "#0a8a4f"
    ax.errorbar(Ks, means_part, yerr=stds_part,
                color=color, marker=marker, markersize=9,
                linestyle=line_style, linewidth=2.2,
                capsize=4, label=f"partition, {label_suffix}")

# Annotate K=1 self-dilution arrow
asr_k1_05 = lookup(1, "single", 0.5, "zo_free")[0]
asr_k1_10 = lookup(1, "single", 1.0, "zo_free")[0]
ax.annotate("", xy=(1, asr_k1_10), xytext=(1, asr_k1_05),
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.0))
ax.text(1.07, (asr_k1_05 + asr_k1_10) / 2,
        f"self-dilution\n−{(asr_k1_05 - asr_k1_10)*100:.1f} pp",
        fontsize=11, color="#ef4444", fontweight="bold", va="center")

# K=3 partition lift
asr_k3_05 = lookup(3, "partition", 0.5, "zo_free")[0]
ax.annotate("", xy=(3, asr_k3_05), xytext=(2.85, asr_k1_05),
            arrowprops=dict(arrowstyle="->", color="#0a8a4f", lw=2.0,
                            connectionstyle="arc3,rad=-0.18"))
ax.text(2.0, 0.98, f"K=3 partition: +{(asr_k3_05 - asr_k1_05)*100:.1f} pp",
        fontsize=11, color="#0a8a4f", fontweight="bold")

# K=1 red marker
ax.scatter([1, 1], [asr_k1_05, asr_k1_10], s=180, color="#ef4444",
           zorder=10, edgecolor="black", linewidth=1.0, label="K=1 (KATZE/SGDI default)")

# Compact "future work" hint at right edge
ax.axvspan(3.4, 4.0, color="#fde68a", alpha=0.35, zorder=-3)
ax.text(3.7, 0.93, "K ≥ 5\nfuture\nwork",
        ha="center", va="center", fontsize=10, color="#92400e",
        style="italic", fontweight="bold")

ax.set_xlim(0.5, 4.0)
ax.set_ylim(0.84, 1.02)
ax.set_xticks([1, 2, 3])
ax.set_xlabel("K (number of injected nodes)")
ax.set_ylabel("ASR")
ax.set_title("K-sweep — PROTEINS GCN zo_free (5-fold)")
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(linestyle=":", alpha=0.4)
ax.text(0.99, 0.01, "Source: multinode_20260524_061442.json",
        transform=ax.transAxes, ha="right", fontsize=8, color="#6b7280")

plt.savefig(ROOT / "figs/fig7_k_sweep_zo_free.png")
plt.savefig(ROOT / "figs/fig7_k_sweep_zo_free.pdf")
plt.close()
print("✓ fig7_k_sweep_zo_free.{png,pdf}")

# Print data table for the slide caption
print()
print("=== K-sweep data (zo_free, partition) ===")
for tau in [0.5, 1.0]:
    print(f"  τ={tau}:")
    for K in Ks:
        asr, std, q = lookup(K, "partition", tau, "zo_free")
        print(f"    K={K}: ASR={asr:.3f}±{std:.3f}, q/g={q:.0f}")
