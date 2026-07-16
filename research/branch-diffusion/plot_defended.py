#!/usr/bin/env python3
"""Cost-vs-benefit figure: ASR vs feat_scale per defense (none/normfilter/mahalanobis)."""
import json, glob
from statistics import mean
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
SCALES = ["1", "auto", "auto_x2"]
XLAB = ["1", r"$\sqrt{N}$", r"$2\sqrt{N}$"]
METHODS = [("none", "#1f77b4", "o", "none (KATZE-style)"),
           ("n4", "#ff7f0e", "s", "N4 (norm-cap)"),
           ("sgdi", "#d62728", "^", "SGDI (Mahalanobis)")]
DEFS = [("none", "Undefended"), ("normfilter", "NormFilter (magnitude)"),
        ("mahalanobis", "Mahalanobis (distribution-aware)")]


def load(dsn):
    return json.load(open(sorted(glob.glob(str(HERE / f"results/defended_sweep_{dsn}_*.json")))[-1]))


def cm(rows, dn, m, s):
    key = f"{dn}|{m}|s{s}"
    v = [r["res"][key]["full"] for r in rows if key in r["res"]]
    return mean(v) if v else 0.0


datasets = [("PROTEINS", 0.736), ("ENZYMES", 0.346)]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
for ri, (dsn, vacc) in enumerate(datasets):
    rows = load(dsn)["rows"]
    for ci, (dn, dtitle) in enumerate(DEFS):
        ax = axes[ri][ci]
        for m, color, mk, lab in METHODS:
            y = [cm(rows, dn, m, s) for s in SCALES]
            ax.plot(range(len(SCALES)), y, color=color, marker=mk, lw=2.2, ms=8, label=lab)
        ax.set_xticks(range(len(SCALES))); ax.set_xticklabels(XLAB)
        ax.set_ylim(-0.04, 1.05); ax.grid(alpha=0.3)
        if ri == 0:
            ax.set_title(dtitle, fontsize=11)
        if ci == 0:
            ax.set_ylabel(f"{dsn} (acc {vacc:.2f})\nASR (full-edge, K=1)")
        if ri == 1:
            ax.set_xlabel("feat_scale (injected magnitude)")
        if dn == "mahalanobis":
            ax.axhspan(-0.04, 1.05, color="#d62728", alpha=0.04)
axes[0][2].legend(fontsize=9, loc="upper right", framealpha=0.95)
fig.suptitle("Cost vs benefit: ASR across feat_scale, by defense (PROTEINS top / ENZYMES bottom; "
             "3 seeds × 3 folds)\nUndefended: scaling wins for all.  Mahalanobis: only SGDI at "
             "low scale survives (in-distribution) — others → 0 at every scale.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "figs/fig_cost_vs_benefit.png"
fig.savefig(out, dpi=160); fig.savefig(HERE / "figs/fig_cost_vs_benefit.pdf")
print("saved", out)
