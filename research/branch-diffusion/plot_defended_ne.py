#!/usr/bin/env python3
"""Defended (Mahalanobis) ASR vs feat_scale, K=1 vs K=3 — node-budget boost figure."""
import json, glob
from statistics import mean
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
SCALES = ["1", "auto", "auto_x2"]
XLAB = ["1", r"$\sqrt{N}$", r"$2\sqrt{N}$"]
METHODS = [("none", "#1f77b4", "o"), ("n4", "#ff7f0e", "s"), ("sgdi", "#d62728", "^")]
LABELS = {"none": "none (KATZE-style)", "n4": "N4 (norm-cap)", "sgdi": "SGDI (Mahalanobis)"}


def load(dsn):
    return json.load(open(sorted(glob.glob(str(HERE / f"results/defended_nodeedge_{dsn}_*.json")))[-1]))


def mo(rows, dn, m, K, s, eb="full"):
    key = f"{dn}|{m}|K{K}|s{s}"
    v = [r["res"][key][eb] for r in rows if key in r["res"]]
    return mean(v) if v else 0.0


# Fig A: Mahalanobis-defended ASR vs feat_scale, K1 solid / K3 dashed
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ci, (dsn, vacc) in enumerate([("PROTEINS", 0.746), ("ENZYMES", 0.380)]):
    rows = load(dsn)["rows"]; ax = axes[ci]
    for m, color, mk in METHODS:
        ax.plot(range(len(SCALES)), [mo(rows, "mahalanobis", m, 1, s) for s in SCALES],
                color=color, marker=mk, lw=2.2, ms=8, label=f"{LABELS[m]}, K=1")
        ax.plot(range(len(SCALES)), [mo(rows, "mahalanobis", m, 3, s) for s in SCALES],
                color=color, marker=mk, lw=1.8, ms=6, ls="--", alpha=0.65, label=f"{LABELS[m]}, K=3")
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels(XLAB)
    ax.set_ylim(-0.04, 1.0); ax.grid(alpha=0.3)
    ax.set_title(f"{dsn} (victim acc {vacc:.2f})", fontsize=11)
    ax.set_xlabel("feat_scale (injected magnitude)")
    if ci == 0:
        ax.set_ylabel("ASR under Mahalanobis defense (full-edge)")
axes[1].legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.95)
fig.suptitle("Under a distribution-aware (Mahalanobis) defense: only SGDI@low-scale survives, "
             "and node budget (K=3, dashed) boosts it\n(none/N4 ≈ 0 at every scale; 2 seeds × 5 folds)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = HERE / "figs/fig_defended_nodebudget.png"
fig.savefig(out, dpi=160); fig.savefig(HERE / "figs/fig_defended_nodebudget.pdf")
print("saved", out)
