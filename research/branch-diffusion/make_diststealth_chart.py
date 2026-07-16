#!/usr/bin/env python3
"""make_diststealth_chart.py — grouped-bar chart of ASR by feat_mode x defense per dataset
from the latest eval_paper_grade JSON. Highlights SGDI's invariance vs norm-cap collapse
under the Mahalanobis (distribution-aware) defense. New script; reads results only."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

jp = sorted(glob.glob("results/eval_paper_grade_*.json"))[-1]
d = json.loads(Path(jp).read_text())
cells = d["cells"]
feats = d["feat_modes"]; defs = d["defenses"]
by = {(c["dataset"], c["defense"], c["feat_mode"]): c for c in cells}
datasets = sorted({c["dataset"] for c in cells}, reverse=True)  # PROTEINS, ENZYMES
colors = {"manifold":"#999999","zo_normcap":"#d77757","zo_free":"#e8b007","sgdi":"#827dbd"}
CHARTS = Path("charts"); CHARTS.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5), sharey=True)
for ax, ds in zip(np.atleast_1d(axes), datasets):
    x = np.arange(len(defs)); w = 0.2
    for k, fm in enumerate(feats):
        vals = [by[(ds, de, fm)]["asr_mean"] if (ds,de,fm) in by else 0 for de in defs]
        err  = [by[(ds, de, fm)]["asr_std"]  if (ds,de,fm) in by else 0 for de in defs]
        ax.bar(x + (k-1.5)*w, vals, w, yerr=err, capsize=2,
               label=("sgdi (ours)" if fm=="sgdi" else fm.replace("zo_normcap","zo_normcap (N4)")),
               color=colors.get(fm,"#444"), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(defs, rotation=15)
    ax.set_title(f"{ds}  ({'discrete / topology-driven' if ds=='PROTEINS' else 'continuous / feature-driven'})")
    ax.set_ylim(0,1.0); ax.grid(axis="y", alpha=0.3)
    ax.axvspan(x[defs.index('mahalanobis')]-0.5, x[defs.index('mahalanobis')]+0.5, color="#827dbd", alpha=0.06)
axes[0].set_ylabel("ASR @ full-edge budget (2 seeds x 5 folds)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Distributional stealth: SGDI is defense-invariant; norm-cap/free collapse under Mahalanobis", fontsize=11)
fig.tight_layout()
out = CHARTS / "diststealth_asr_by_defense.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved {out} from {jp}")
