#!/usr/bin/env python3
"""
eval_nb_crossover.py — node-budget crossover for the SGDI-vs-normcap stealth tradeoff (NEW).

Headline cells only: feat {zo_normcap (=N4 magnitude stealth), sgdi (ours, distribution
stealth)} x defense {none, mahalanobis} x node_budget {1,3,5,10} x {PROTEINS discrete,
ENZYMES continuous}, 5-fold (seed 0), ASR @ fixed full-edge budget.

Purpose: show honestly WHERE distribution-stealth (SGDI, defense-invariant) beats and where
magnitude-stealth (norm-cap) overtakes via brute-force multi-node aggregation. Complements
the nb=1 grid in eval_paper_grade.py. Every number -> results/eval_nb_crossover_<stamp>.json.

Usage: micromamba run -n graph_adversarial python eval_nb_crossover.py
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

FEATS = ["zo_normcap", "sgdi"]
DEFS = ["none", "mahalanobis"]
NBS = [1, 5, 10]
CFG = {"skip_greedy": True, "zo_epochs": 12, "zo_restarts": 1, "seed_offset": 0}


def run_dataset(ds_key, folds_data, save_cb, acc):
    for de in DEFS:
        defended = [build_defended(m, de, g) for (m, g) in folds_data]
        for fm in FEATS:
            for nb in NBS:
                t0 = time.time()
                per_fold = []
                for dm, (m, g) in zip(defended, folds_data):
                    o = attack.run_attack_sweep(dm, g, DEVICE,
                                                {**CFG, "feat_mode": fm, "node_budget": nb})
                    per_fold.append(o["asr_full"])
                rec = {"dataset": ds_key, "defense": de, "feat_mode": fm, "node_budget": nb,
                       "asr_mean": float(np.mean(per_fold)), "asr_std": float(np.std(per_fold)),
                       "asr_folds": [round(x, 4) for x in per_fold], "n_folds": len(per_fold)}
                acc.append(rec)
                print(f"  [{ds_key}] {de:<11} {fm:<10} nb={nb:<2} "
                      f"ASR={rec['asr_mean']:.3f}±{rec['asr_std']:.3f} [{time.time()-t0:.0f}s]",
                      flush=True)
                save_cb(acc)


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"eval_nb_crossover_{stamp}.json"
    acc = []
    save_cb = lambda a: jp.write_text(json.dumps(
        {"stamp": stamp, "feats": FEATS, "defenses": DEFS, "nbs": NBS,
         "cfg": CFG, "cells": a}, indent=2))

    print(f"=== eval_nb_crossover {stamp} (device={DEVICE}) ===")
    t0 = time.time()

    pmeta = json.loads((MODELS_DIR / "meta.json").read_text())
    run_dataset("PROTEINS", load_discrete_folds(pmeta), save_cb, acc)

    emeta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    run_dataset("ENZYMES", load_continuous_folds("ENZYMES", emeta), save_cb, acc)

    save_cb(acc)
    print(f"\nSaved {jp} [{time.time()-t0:.0f}s]")
    # crossover table per dataset/defense
    by = {(c["dataset"], c["defense"], c["feat_mode"], c["node_budget"]): c for c in acc}
    for ds in ("PROTEINS", "ENZYMES"):
        for de in DEFS:
            print(f"\n{ds} / {de}:  nb -> ASR (normcap | sgdi)")
            for nb in NBS:
                a = by.get((ds, de, "zo_normcap", nb))
                b = by.get((ds, de, "sgdi", nb))
                if a and b:
                    win = "sgdi" if b["asr_mean"] > a["asr_mean"] else "normcap"
                    print(f"  nb={nb:<2} {a['asr_mean']:.3f} | {b['asr_mean']:.3f}  -> {win}")


if __name__ == "__main__":
    main()
