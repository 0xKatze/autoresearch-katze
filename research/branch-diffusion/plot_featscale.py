#!/usr/bin/env python3
"""Plot the controlled feat_scale sweep: ASR vs feat_scale, paneled by dataset x edge budget."""
import json, glob
from statistics import mean
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
SCALES = ["1", "auto", "auto_x2", "auto_x4"]
XLAB = ["1", r"$\sqrt{N}$", r"$2\sqrt{N}$", r"$4\sqrt{N}$"]
METHODS = [("none", "#1f77b4", "o", "none (KATZE-style)"),
           ("n4", "#ff7f0e", "s", "N4 (norm-cap)"),
           ("sgdi", "#d62728", "^", "SGDI (Mahalanobis)")]


def load(dsn):
    f = sorted(glob.glob(str(HERE / f"results/featscale_sweep_{dsn}_*.json")))[-1]
    return json.load(open(f))


def cell_mean(rows, m, K, s, eb):
    key = f"{m}|K{K}|s{s}"
    v = [r["res"][key][eb] for r in rows if key in r["res"]]
    return mean(v) if v else None


datasets = [("PROTEINS", 0.734), ("ENZYMES", 0.343)]
ebs = [("1", "edge budget = 1 (sparse)"), ("full", "full-edge")]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
for ri, (dsn, vacc) in enumerate(datasets):
    d = load(dsn); rows = d["rows"]
    for ci, (eb, ebtitle) in enumerate(ebs):
        ax = axes[ri][ci]
        for m, color, mk, lab in METHODS:
            y1 = [cell_mean(rows, m, 1, s, eb) for s in SCALES]
            y3 = [cell_mean(rows, m, 3, s, eb) for s in SCALES]
            x = range(len(SCALES))
            ax.plot(x, y1, color=color, marker=mk, lw=2, ms=7,
                    label=f"{lab}, K=1")
            ax.plot(x, y3, color=color, marker=mk, lw=1.5, ms=5, ls="--", alpha=0.6,
                    label=f"{lab}, K=3")
        ax.set_xticks(list(range(len(SCALES)))); ax.set_xticklabels(XLAB)
        ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
        ax.set_title(f"{dsn} (victim acc {vacc:.2f}) — {ebtitle}", fontsize=11)
        if ci == 0:
            ax.set_ylabel("Attack Success Rate (undefended)")
        if ri == 1:
            ax.set_xlabel("feat_scale (injected-feature magnitude)")
axes[0][1].legend(fontsize=8, loc="lower right", ncol=1, framealpha=0.9)
fig.suptitle("Controlled undefended comparison: ASR vs feat_scale, by stealth projection\n"
             "(same victim & ZO optimizer; projection = direction/shape, feat_scale = magnitude; "
             "3 seeds × 3 folds)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = HERE / "figs/fig_featscale_sweep.png"
out_pdf = HERE / "figs/fig_featscale_sweep.pdf"
fig.savefig(out_png, dpi=160); fig.savefig(out_pdf)
print("saved", out_png)
