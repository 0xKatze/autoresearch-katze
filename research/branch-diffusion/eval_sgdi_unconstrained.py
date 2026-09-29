#!/usr/bin/env python3
"""SGDI constraint ablation: what does SGDI do once its stealth constraints are removed?

Variants (all node_budget=1, ASR@full-edge, same ATTACK_CFG as eval_paper_grade.py):
  sgdi                  original (Langevin + prior score + hard Mahalanobis projection)
  sgdi_noproj           Mahalanobis projection removed, prior pull kept (eta=0.3)
  sgdi_noproj_noprior   projection removed and prior pull off (eta=0)
  sgdi_free_x2          as above + KATZE feature amplification feat_scale=auto_x2
  zo_free / zo_free_x2  plain ZO references (no diffusion terms)
Defenses: none, mahalanobis (calibrated exactly as eval_paper_grade.build_defended).
attack.py is NOT modified: the projection is switched off by swapping attack._project_maha
for the identity inside a context manager.

usage: python eval_sgdi_unconstrained.py [PROTEINS|ENZYMES|both] [seeds, e.g. 0,1]
"""
import json, sys, time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attack  # noqa: E402
import eval_paper_grade as epg  # noqa: E402

VARIANTS = [
    ("sgdi", "sgdi", {}, False),
    ("sgdi_noproj", "sgdi", {}, True),
    ("sgdi_noproj_noprior", "sgdi", {"sgdi_eta": 0.0}, True),
    ("sgdi_free_x2", "sgdi", {"sgdi_eta": 0.0, "feat_scale": "auto_x2"}, True),
    ("zo_free", "zo_free", {}, False),
    ("zo_free_x2", "zo_free", {"feat_scale": "auto_x2"}, False),
]
DEFENSES = ["none", "mahalanobis"]


@contextmanager
def projection_off(active):
    orig = attack._project_maha
    if active:
        attack._project_maha = lambda x, mu, cov_inv, thr: x
    try:
        yield
    finally:
        attack._project_maha = orig


def load(ds):
    if ds == "PROTEINS":
        meta = json.loads((epg.MODELS_DIR / "meta.json").read_text())
        return epg.load_discrete_folds(meta)
    meta = json.loads((epg.MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    return epg.load_continuous_folds("ENZYMES", meta)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1").split(",")]
    datasets = ["PROTEINS", "ENZYMES"] if which == "both" else [which]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = HERE / "results/cdma" / f"sgdi_unconstrained_{which}_{stamp}.json"
    cells = []
    for ds in datasets:
        folds = load(ds)
        for defense in DEFENSES:
            dms = [epg.build_defended(m, defense, g, ds_key=ds) for m, g in folds]
            for name, fm, extra, noproj in VARIANTS:
                obs, qs, t0 = [], [], time.time()
                with projection_off(noproj):
                    for seed in seeds:
                        for dm, (_m, graphs) in zip(dms, folds):
                            cfg = {**epg.ATTACK_CFG, "feat_mode": fm, "seed_offset": seed, **extra}
                            o = attack.run_attack_sweep(dm, graphs, epg.DEVICE, cfg)
                            obs.append(o["asr_full"])
                            qs.append(o["queries"] / max(o["n_graphs"], 1))
                cells.append({"dataset": ds, "defense": defense, "variant": name,
                              "feat_mode": fm, "extra_cfg": extra, "projection_off": noproj,
                              "asr_mean": float(np.mean(obs)), "asr_std": float(np.std(obs)),
                              "asr_obs": [round(x, 4) for x in obs],
                              "q_per_graph": float(np.mean(qs)), "n_obs": len(obs)})
                c = cells[-1]
                print(f"[{ds}] {defense:<11} {name:<20} ASR={c['asr_mean']:.3f}±{c['asr_std']:.3f} "
                      f"q/g={c['q_per_graph']:.0f} [{time.time()-t0:.0f}s]", flush=True)
                json.dump({"script": "eval_sgdi_unconstrained.py", "timestamp": stamp,
                           "attack_cfg": epg.ATTACK_CFG, "seeds": seeds, "device": str(epg.DEVICE),
                           "cells": cells}, open(out_path, "w"), indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
