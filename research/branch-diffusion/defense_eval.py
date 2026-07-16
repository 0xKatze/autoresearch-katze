#!/usr/bin/env python3
"""
defense_eval.py — Motivation grid: does a feature-stealth gap exist under defense?

Measures ADAPTIVE robust ASR. We wrap the PROTEINS GCN victim in each defense, then run
the black-box injection attack THROUGH the defense (the attacker queries the defended
model, so its CGE feature optimization adapts to whatever the filter does). For each
(feat_mode x defense) cell we record, 5-fold:
  - asr_best : fraction of graphs flippable at any edge budget rho<=1 (per-graph best)
  - asr_full : fraction flippable at the rho=1 full-edge anchor (KATZE-comparable)

Why this is the gate for the diffusion line:
  H0 (null): zo_free ASR is invariant to normfilter/jaccard -> there is NO stealth problem,
             so a diffusion feature generator is a solution looking for a problem (the N23 trap).
  H1       : zo_free collapses under normfilter (its high-norm injected node is pruned) while
             manifold survives -> an ASR<->stealth gap exists -> guided diffusion has a target:
             produce on-manifold (un-prunable) features that still carry the adversarial signal.

Defenses: none, jaccard(thr .01/.05 cosine), normfilter(95/90 pct). SVD is skipped here:
batched low-rank SVD mixes graphs in the block-diagonal adjacency and is topology-only,
irrelevant to the feature-stealth question this grid is built to answer.

Output: results/defense_eval_<stamp>.json + printed grid. New script; core harness untouched.

Usage: micromamba run -n graph_adversarial python defense_eval.py [--quick]
"""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from prepare import MODELS_DIR, RESULTS_DIR, DEVICE
from sweep_stealth import load_fold, asr_at_rho
from defenses import wrap_with_defense
import attack

FEAT_MODES = ["manifold", "zo_box", "zo_free"]
DEFENSES = ["none", "jaccard", "jaccard_strong", "normfilter", "normfilter_strong"]


def run_cell(feat_mode, defense_key, folds, meta):
    cfg = {"node_budget": 1, "feat_mode": feat_mode}
    best, full, qpg = [], [], []
    for fi in folds:
        model, graphs = load_fold(fi, meta)
        if not graphs:
            continue
        dmodel = wrap_with_defense(model, defense_key)
        out = attack.run_attack_sweep(dmodel, graphs, DEVICE, cfg)
        best.append(asr_at_rho(out["rho_to_flip"], 1.0))   # flippable at any rho<=1
        full.append(out["asr_full"])
        qpg.append(out["queries"] / max(out["n_graphs"], 1))
    return {
        "feat_mode": feat_mode, "defense": defense_key,
        "asr_best_mean": float(np.mean(best)) if best else 0.0,
        "asr_best_std": float(np.std(best)) if best else 0.0,
        "asr_full_mean": float(np.mean(full)) if full else 0.0,
        "asr_best_folds": [round(x, 4) for x in best],
        "queries_per_graph_mean": float(np.mean(qpg)) if qpg else 0.0,
        "n_folds": len(best),
    }


def main():
    quick = "--quick" in sys.argv
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    folds = meta["folds"]
    feat_modes, defenses = FEAT_MODES, DEFENSES
    if quick:
        folds = folds[:1]
        defenses = ["none", "normfilter", "jaccard"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== defense_eval {stamp} | {len(feat_modes)}x{len(defenses)} cells "
          f"x {len(folds)} folds (device={DEVICE}) ===")

    cells, t0 = [], time.time()
    for fm in feat_modes:
        for dk in defenses:
            tc = time.time()
            rec = run_cell(fm, dk, folds, meta)
            cells.append(rec)
            print(f"  {fm:<9} {dk:<18} ASR_best={rec['asr_best_mean']:.2%} "
                  f"(+/-{rec['asr_best_std']:.2%})  ASR@full={rec['asr_full_mean']:.2%} "
                  f"q/g={rec['queries_per_graph_mean']:.0f} [{time.time()-tc:.1f}s]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"defense_eval_{stamp}.json"
    jp.write_text(json.dumps(
        {"stamp": stamp, "quick": quick, "node_budget": 1,
         "feat_modes": feat_modes, "defenses": defenses, "cells": cells}, indent=2))
    print(f"Saved {jp} [{time.time()-t0:.1f}s total]")

    by = {(c["feat_mode"], c["defense"]): c for c in cells}
    print("\nASR_best grid (rows=feat_mode, cols=defense):")
    print("feat_mode   " + "".join(f"{d:>18}" for d in defenses))
    for fm in feat_modes:
        row = f"{fm:<11} "
        for d in defenses:
            c = by.get((fm, d))
            row += f"{(c['asr_best_mean'] if c else float('nan')):>17.1%} "
        print(row)


if __name__ == "__main__":
    main()
