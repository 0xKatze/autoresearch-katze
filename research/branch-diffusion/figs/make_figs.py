"""Generate paper/slide-ready figures for "Single-node injection is not enough"."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# ---- consistent style (paper-friendly) ----
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

OUT = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion/figs")
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#36d399"   # lime — multi-node
RED = "#ef4444"      # red — K=1 / problem
GREY = "#6b7280"
DARK = "#0b1020"


# ============================================================
# FIG 1 — ASR ceiling: K=1 self-dilution + K=3 wins at matched budget
# ============================================================
fig, ax = plt.subplots(figsize=(6.8, 4.2))
configs = ["K=1, τ=0.5", "K=1, τ=1.0", "K=3 part., τ=0.5", "K=3 part., τ=1.0"]
asr     = [0.933,        0.886,        0.981,             0.969]
colors  = [RED,           RED,          ACCENT,            ACCENT]
x = np.arange(len(configs))
bars = ax.bar(x, asr, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
for bar, v in zip(bars, asr):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
            f"{v*100:.1f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
# self-dilution arrow K=1 0.5 -> 1.0
ax.annotate("", xy=(1, 0.886), xytext=(0, 0.933),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
ax.text(0.5, 0.965, "self-dilution: −4.7 pp",
        ha="center", color=RED, fontsize=10, fontweight="bold")
# multi-node lift
ax.annotate("", xy=(3, 0.969), xytext=(1, 0.886),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(2, 1.005, "K=3 partition: +8.3 pp",
        ha="center", color="#0a8a4f", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=0)
ax.set_ylabel("ASR")
ax.set_title("Matched-budget ASR — PROTEINS GCN, 5-fold (zo_free)")
ax.set_ylim(0.84, 1.04)
ax.set_yticks(np.arange(0.85, 1.05, 0.05))
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.text(0.99, 0.01, "Source: multinode_20260524_061442.json",
        transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
plt.savefig(OUT / "fig1_asr_ceiling.png")
plt.savefig(OUT / "fig1_asr_ceiling.pdf")
plt.close()
print("✓ fig1_asr_ceiling.{png,pdf}")


# ============================================================
# FIG 2 — Full multinode grid (6 configurations, PROTEINS τ=1.0 zo_free)
# ============================================================
fig, ax = plt.subplots(figsize=(7.6, 4.2))
configs = ["K=1\nsingle", "K=2\nduplicate", "K=2\npartition", "K=2\ninterconn.",
           "K=3\nduplicate", "K=3\npartition", "K=3\ninterconn."]
asr     = [0.886,           0.943,          0.943,          0.948,
           0.972,            0.969,            0.970]
q_per_g = [98,               143,            143,            137,
           299,              188,            195]
colors  = [RED] + [GREY]*3 + [GREY, ACCENT, GREY]
x = np.arange(len(configs))
bars = ax.bar(x, asr, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
for bar, v in zip(bars, asr):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10)
# q/g labels below
for i, q in enumerate(q_per_g):
    ax.text(i, 0.83, f"q/g={q}", ha="center", va="bottom",
            fontsize=9, color=GREY, style="italic")
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("ASR")
ax.set_title("PROTEINS GCN — full multinode grid (τ=1.0, zo_free, 5-fold)")
ax.set_ylim(0.82, 1.00)
ax.set_yticks(np.arange(0.85, 1.01, 0.05))
ax.axhline(0.886, color=RED, linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(6.4, 0.886, "K=1 baseline", ha="right", va="bottom",
        fontsize=9, color=RED, style="italic")
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.text(0.99, 0.01, "Source: multinode_20260524_061442.json",
        transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
plt.savefig(OUT / "fig2_multinode_grid.png")
plt.savefig(OUT / "fig2_multinode_grid.pdf")
plt.close()
print("✓ fig2_multinode_grid.{png,pdf}")


# ============================================================
# FIG 3 — Defense vulnerability: single-node universal floor under GDDM-A
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.2))
attacks = ["zo_free", "zo_free\n+gddm-A", "SGDI", "SGDI\n+gddm-A",
           "SpecGDI v1", "SpecGDI v1\n+gddm-A"]
asr     = [1.000,    0.975,           0.975,  0.175,
           0.867,    0.067]
colors  = [GREY, GREY, GREY, RED, GREY, RED]
x = np.arange(len(attacks))
bars = ax.bar(x, asr, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
for bar, v in zip(bars, asr):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.018,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
# the catastrophe
ax.annotate("", xy=(3, 0.175), xytext=(2, 0.975),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
ax.text(2.5, 0.62, "−80 pp", ha="center", color=RED,
        fontsize=11, fontweight="bold")
# the deeper one
ax.annotate("", xy=(5, 0.067), xytext=(4, 0.867),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
ax.text(4.5, 0.52, "−80 pp", ha="center", color=RED,
        fontsize=11, fontweight="bold")
# floor band
ax.axhspan(0, 0.20, color=RED, alpha=0.08, zorder=-2)
ax.text(0.02, 0.18, "K=1 universal floor (≤ 0.18)",
        transform=ax.transAxes, fontsize=10, color=RED,
        style="italic", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(attacks, fontsize=10)
ax.set_ylabel("ASR")
ax.set_title("PROTEINS — single-node attacks under GDDM-A adjacency denoiser")
ax.set_ylim(0, 1.10)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.text(0.99, 0.01, "Source: smoke_gddm_a.py PROTEINS fold 0, n=40",
        transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
plt.savefig(OUT / "fig3_defense_vulnerability.png")
plt.savefig(OUT / "fig3_defense_vulnerability.pdf")
plt.close()
print("✓ fig3_defense_vulnerability.{png,pdf}")


# ============================================================
# FIG 4 — Literature K consensus (KATZE/SGDI K=1 is the outlier)
# ============================================================
fig, ax = plt.subplots(figsize=(7.4, 4.0))
papers = ["KATZE/SGDI\n(this lineage)", "HAO\n'22", "GANI\n'24",
          "JANUS\n'25", "QUGIA\n'25", "TDGIA\n'21"]
K_vals  = [1,                              5,         8,
           5,                              7,         30]
colors  = [RED] + [ACCENT]*5
x = np.arange(len(papers))
bars = ax.bar(x, K_vals, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
for bar, v in zip(bars, K_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.7,
            f"K={v}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(papers, fontsize=10)
ax.set_ylabel("Typical K (number of injected nodes)")
ax.set_title("Multi-node consensus across recent GIA stealth literature")
ax.set_ylim(0, 35)
ax.axhline(1, color=RED, linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(5.4, 1.5, "K=1", ha="right", color=RED, fontsize=10, style="italic")
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.text(0.99, 0.01, "Source: lit-search 2026-05-26",
        transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
plt.savefig(OUT / "fig4_literature_K.png")
plt.savefig(OUT / "fig4_literature_K.pdf")
plt.close()
print("✓ fig4_literature_K.{png,pdf}")


# ============================================================
# FIG 5 — Self-dilution theoretical curve (per-edge influence vs degree)
# ============================================================
fig, ax = plt.subplots(figsize=(6.6, 4.0))
deg_v = np.arange(1, 51)
mean_d_u = 4  # PROTEINS clean mean degree
influence = 1 / np.sqrt(deg_v * mean_d_u)
ax.plot(deg_v, influence, color=DARK, linewidth=2, label=r"per-edge weight $\frac{1}{\sqrt{\deg(v^*)\deg(u)}}$")
# K=1 typical degree marker
ax.axvline(40, color=RED, linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(40.5, 0.45, "K=1, deg=40\n(full edge)", color=RED, fontsize=10)
# K=3 marker
ax.axvline(13, color=ACCENT, linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(13.5, 0.45, "K=3 partition\nper-node deg ≈ 13", color="#0a8a4f", fontsize=10)
# Annotate influence ratio
inf_k1 = 1/np.sqrt(40*mean_d_u)
inf_k3 = 1/np.sqrt(13*mean_d_u)
ratio = inf_k3 / inf_k1
ax.annotate(f"K=3 per-edge: {ratio:.2f}× stronger",
            xy=(20, 0.10), fontsize=11, color="#0a8a4f", fontweight="bold")
ax.set_xlabel(r"injected-node degree $\deg(v^*)$")
ax.set_ylabel("per-edge message weight (clean $\\bar{d}_u = 4$)")
ax.set_title("GCN self-dilution: per-edge influence vs injected-node degree")
ax.grid(linestyle=":", alpha=0.4)
ax.set_xlim(0, 50)
ax.set_ylim(0, 0.55)
ax.legend(loc="upper right")
plt.savefig(OUT / "fig5_self_dilution_curve.png")
plt.savefig(OUT / "fig5_self_dilution_curve.pdf")
plt.close()
print("✓ fig5_self_dilution_curve.{png,pdf}")

print(f"\nAll figures written to {OUT}")
print(f"Files: {sorted(p.name for p in OUT.glob('*.p*g')) + sorted(p.name for p in OUT.glob('*.pdf'))}")
