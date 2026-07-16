#!/usr/bin/env python3
"""
eval_kappa_frontier.py — fairness analysis: why SGDI < N4 on UNDEFENDED ASR (NEW).

The operational budget is matched (node_budget, full-edge rho=1.0, query budget SGDI>=N4).
The undefended-ASR gap comes from the STEALTH CONSTRAINT SET: N4 (zo_normcap) projects to an
L2 ball of radius kappa*median_norm — a large, direction-agnostic set that admits strong but
DETECTABLE (off-manifold) features; SGDI projects to the tight Mahalanobis ellipsoid. So N4's
undefended edge is bought with detectability.

This traces N4's stealth-ASR tradeoff by sweeping kappa and recording BOTH undefended ASR and
ASR-under-Mahalanobis (the stealth-matched metric). SGDI and zo_free are single reference points.
If SGDI sits above N4's (undefended, defended) frontier, it Pareto-dominates the tradeoff.

Grid: dataset {PROTEINS, ENZYMES} x {zo_normcap@kappa in [0.5,1.0,2.0,4.0], sgdi, zo_free}
      x defense {none, mahalanobis} x 5 folds (seed 0), nb=1, ASR@full-edge.

Output: results/eval_kappa_frontier_<stamp>.json + printed frontier. Core harness untouched.
Usage: micromamba run -n graph_adversarial python eval_kappa_frontier.py
"""
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from prepare import MODELS_DIR, RESULTS_DIR, DEVICE
from prepare_cont import MODELS_CONT_DIR
from eval_paper_grade import (load_discrete_folds, load_continuous_folds, build_defended)
import attack

KAPPAS = [0.5, 1.0, 2.0, 4.0]
DEFS = ["none", "mahalanobis"]
BASE = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 15, "zo_restarts": 2, "seed_offset": 0}


def configs():
    cfgs = [("zo_normcap", k, {**BASE, "feat_mode": "zo_normcap", "normcap_kappa": k}) for k in KAPPAS]
    cfgs.append(("sgdi", None, {**BASE, "feat_mode": "sgdi"}))
    cfgs.append(("zo_free", None, {**BASE, "feat_mode": "zo_free"}))
    return cfgs


def run_dataset(ds_key, folds_data, acc, save_cb):
    defended = {de: [build_defended(m, de, g) for (m, g) in folds_data] for de in DEFS}
    for (fm, kappa, cfg) in configs():
        row = {"dataset": ds_key, "feat_mode": fm, "kappa": kappa}
        for de in DEFS:
            t0 = time.time()
            per_fold = []
            for dm, (m, g) in zip(defended[de], folds_data):
                o = attack.run_attack_sweep(dm, g, DEVICE, cfg)
                per_fold.append(o["asr_full"])
            row[f"asr_{de}_mean"] = float(np.mean(per_fold))
            row[f"asr_{de}_std"] = float(np.std(per_fold))
            row[f"asr_{de}_folds"] = [round(x, 4) for x in per_fold]
            print(f"  [{ds_key}] {fm}{'' if kappa is None else f'@k={kappa}':<8} {de:<11} "
                  f"ASR={row[f'asr_{de}_mean']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        acc.append(row)
        save_cb(acc)


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"eval_kappa_frontier_{stamp}.json"
    acc = []
    save_cb = lambda a: jp.write_text(json.dumps(
        {"stamp": stamp, "kappas": KAPPAS, "defenses": DEFS, "base": BASE, "cells": a}, indent=2))

    print(f"=== eval_kappa_frontier {stamp} (device={DEVICE}) ===")
    t0 = time.time()
    pmeta = json.loads((MODELS_DIR / "meta.json").read_text())
    run_dataset("PROTEINS", load_discrete_folds(pmeta), acc, save_cb)
    emeta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    run_dataset("ENZYMES", load_continuous_folds("ENZYMES", emeta), acc, save_cb)
    save_cb(acc)
    print(f"\nSaved {jp} [{time.time()-t0:.0f}s]")

    for ds in ("PROTEINS", "ENZYMES"):
        print(f"\n{ds}: feat (kappa) -> undefended ASR | ASR under Mahalanobis (stealth-matched)")
        for r in acc:
            if r["dataset"] != ds:
                continue
            tag = r["feat_mode"] + ("" if r["kappa"] is None else f"@k={r['kappa']}")
            print(f"  {tag:<16} none={r['asr_none_mean']:.3f}  maha={r['asr_mahalanobis_mean']:.3f}")


if __name__ == "__main__":
    main()
