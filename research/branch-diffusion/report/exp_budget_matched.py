#!/usr/bin/env python3
"""
exp_budget_matched.py — STRICT budget-matched comparison: GZOO-faithful vs
KATZE(no-diffusion) vs N24, under BOTH the GZOO edge budget (1 edge) and the
N24 edge budget (full), all with node_budget = 1 (exactly one injected node).

Does NOT modify attack.py or best.json. Reassigns attack.CONFIG in-memory per
condition and replicates run.py's per-fold eval faithfully.

Usage:
  micromamba run -n graph_adversarial python exp_budget_matched.py [--quick]
"""
import sys, json, time, copy
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch_geometric.datasets import TUDataset

from prepare import (
    GCN_GraphClassification, DATA_DIR, MODELS_DIR, RESULTS_DIR, DEVICE,
    collect_test_graphs,
)
import attack

BASE = copy.deepcopy(attack.CONFIG)

METHODS = {
    # label: (feat_scale, diffusion_on)
    "GZOO-faithful (fs=1.0)":   dict(feat_scale=1.0,        diff=False),
    "KATZE-nodiff (fs=auto_x2)": dict(feat_scale="auto_x2", diff=False),
    "N24 (full stack)":          dict(feat_scale="auto_x2", diff=True),
}
BUDGETS = {
    # label: (edge_strategy, joint_edge_budget)
    "GZOO-budget (1 edge)": dict(edge_strategy="topk", joint_edge_budget=1),
    "N24-budget (full)":    dict(edge_strategy="full", joint_edge_budget="avg_deg"),
}


def make_cfg(m, b):
    c = copy.deepcopy(BASE)
    # STRICT: exactly one injected node for every condition
    c["node_budget"] = 1
    c["node_budget_pct"] = None
    c["node_budget_floor"] = 1
    # method knobs
    c["feat_scale"] = m["feat_scale"]
    c["use_diffusion_restart"] = m["diff"]
    c["use_joint_diffusion"] = m["diff"]
    c["use_k_sweep"] = m["diff"]
    # budget knobs
    c["edge_strategy"] = b["edge_strategy"]
    c["joint_edge_budget"] = b["joint_edge_budget"]
    return c


def eval_cfg(cfg, folds, meta, quick):
    attack.CONFIG = cfg  # run_attack reads module-global CONFIG at call time
    asrs, succ, tot = [], 0, 0
    fl = folds[:1] if quick else folds
    for fi in fl:
        model = GCN_GraphClassification(
            meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(
            torch.load(fi["model_path"], map_location=DEVICE, weights_only=True))
        model.eval()
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        tg = collect_test_graphs(model, ds, fi["test_idx"])
        if not tg:
            asrs.append(0.0)
            continue
        asr = attack.run_attack(model, tg, DEVICE)
        asrs.append(asr)
        succ += int(round(asr * len(tg)))
        tot += len(tg)
    return float(np.mean(asrs)), float(np.std(asrs)), succ, tot


def main():
    quick = "--quick" in sys.argv
    with open(MODELS_DIR / "meta.json") as f:
        meta = json.load(f)
    folds = meta["folds"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'='*70}\nBudget-matched comparison {stamp}"
          f" ({'QUICK 1-fold' if quick else 'FULL 5-fold'})\n{'='*70}")

    rows = {}
    for mlabel, m in METHODS.items():
        rows[mlabel] = {}
        for blabel, b in BUDGETS.items():
            cfg = make_cfg(m, b)
            t0 = time.time()
            mean, std, succ, tot = eval_cfg(cfg, folds, meta, quick)
            dt = time.time() - t0
            rows[mlabel][blabel] = dict(mean=mean, std=std, succ=succ, tot=tot, sec=dt)
            print(f"  {mlabel:28s} | {blabel:22s} "
                  f"ASR={mean*100:6.2f}% +/- {std*100:5.2f}% [{dt:5.1f}s]")

    print(f"\n{'='*70}\nSUMMARY TABLE (ASR%)\n{'='*70}")
    blabels = list(BUDGETS.keys())
    hdr = f"{'Method':30s} | " + " | ".join(f"{b:22s}" for b in blabels)
    print(hdr)
    print("-" * len(hdr))
    print(f"{'GZOO-paper (cited, fs=1.0)':30s} | "
          f"{'54.26 (Table V)':22s} | {'n/a (cannot rerun)':22s}")
    for mlabel in METHODS:
        cells = []
        for blabel in blabels:
            r = rows[mlabel][blabel]
            cells.append(f"{r['mean']*100:.2f} +/- {r['std']*100:.2f}")
        print(f"{mlabel:30s} | " + " | ".join(f"{c:22s}" for c in cells))

    out = RESULTS_DIR / f"budget_matched_{stamp}{'_quick' if quick else ''}.json"
    with open(out, "w") as f:
        json.dump({"timestamp": stamp, "quick": quick,
                   "gzoo_paper_cited": {"asr": 0.5426, "budget": "nb=1,eb=1,fs=1.0"},
                   "rows": rows}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
