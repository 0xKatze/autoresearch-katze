#!/usr/bin/env python3
"""
eval_decisive.py — decisive grid for SpecGDI (B1/B2) vs SGDI vs N4-style normcap,
under distribution-aware defenses (none, mahalanobis, gnnsafe, gddm).

Mirrors eval_paper_grade.py's JSON schema so existing analysis scripts work,
but runs a focused 4x4x2 = 32-cell grid (or probe / partial via --probe / --partial).

Usage:
  micromamba run -n graph_adversarial python eval_decisive.py --probe
  micromamba run -n graph_adversarial python eval_decisive.py
  micromamba run -n graph_adversarial python eval_decisive.py --partial '<cells_json>'
"""
import json
import sys
import time
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from prepare import MODELS_DIR, RESULTS_DIR, DEVICE, GCN_GraphClassification
from sweep_stealth import load_fold as load_fold_discrete
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from defenses import gddm_ckpt_path  # noqa: F401 (used by build_defended)
import attack
from eval_paper_grade import (
    build_defended,
    clean_acc_defended,
    ATTACK_CFG,
    GDDM_T_STAR,  # noqa: F401
    GNNSAFE_T, GNNSAFE_K, GNNSAFE_ALPHA, GNNSAFE_PCT_LOW,  # noqa: F401
)

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


FEAT_MODES_DECISIVE = ["zo_normcap", "sgdi", "specgdi", "specgdi_signal"]
DEFENSES_DECISIVE = ["none", "mahalanobis", "gnnsafe", "gddm"]
SEEDS_DECISIVE = [0, 1]
# PROTEINS-only override for SpecGDI: reduce swaps_per_step from default 8 -> 2
# to keep large-graph wall-clock manageable (see SPECGDI_IMPL.md open questions).
SPECGDI_PROTEINS_OVERRIDE = {"specgdi_swaps_per_step": 2}


def load_discrete_folds(meta):
    out = []
    for fi in meta["folds"]:
        out.append(load_fold_discrete(fi, meta))
    return out


def load_continuous_folds(name, meta):
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs(name, mean, std)
    out = []
    for fi in meta["folds"]:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE,
                                         weights_only=True))
        model.eval()
        graphs = []
        for i in fi["test_idx"]:
            g = deepcopy(graphs_all[i]).to(DEVICE)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                if model(g).argmax(-1).item() == g.y.item():
                    graphs.append(g)
            out_g = graphs  # noqa
        out.append((model, graphs))
    return out


def run_grid(ds_key, folds_data, feat_modes, defenses, seeds, save_cb,
             extra_cfg_for_specgdi=None):
    """Run grid for one dataset; calls save_cb after every cell completes."""
    cells = []
    for defense in defenses:
        defended, cleanaccs = [], []
        for (model, graphs) in folds_data:
            dm = build_defended(model, defense, graphs, ds_key=ds_key)
            defended.append(dm)
            cleanaccs.append(clean_acc_defended(dm, graphs))
        for fm in feat_modes:
            obs, qs, t0 = [], [], time.time()
            extra = {}
            if fm in ("specgdi", "specgdi_signal") and extra_cfg_for_specgdi:
                extra = dict(extra_cfg_for_specgdi)
            for seed in seeds:
                for dm, (model, graphs) in zip(defended, folds_data):
                    cfg = {**ATTACK_CFG, "feat_mode": fm, "seed_offset": seed, **extra}
                    o = attack.run_attack_sweep(dm, graphs, DEVICE, cfg)
                    obs.append(o["asr_full"])
                    qs.append(o["queries"] / max(o["n_graphs"], 1))
            rec = {"dataset": ds_key, "defense": defense, "feat_mode": fm,
                   "asr_mean": float(np.mean(obs)), "asr_std": float(np.std(obs)),
                   "asr_obs": [round(x, 4) for x in obs],
                   "q_per_graph": float(np.mean(qs)),
                   "clean_acc_def_mean": float(np.mean(cleanaccs)),
                   "n_obs": len(obs),
                   "wall_s": round(time.time() - t0, 1),
                   "cfg_extra": extra}
            cells.append(rec)
            print(f"  [{ds_key}] {defense:<11} {fm:<15} "
                  f"ASR={rec['asr_mean']:.3f}±{rec['asr_std']:.3f} "
                  f"cleanAcc={rec['clean_acc_def_mean']:.3f} "
                  f"q/g={rec['q_per_graph']:.0f} [{rec['wall_s']:.0f}s]",
                  flush=True)
            save_cb(cells)
    return cells


def paired_tests_extended(cells):
    """For each (dataset, defense): pairwise Wilcoxon among specgdi, sgdi, zo_normcap."""
    by = {(c["dataset"], c["defense"], c["feat_mode"]): c for c in cells}
    rows = []
    pairs = [("specgdi", "sgdi"), ("specgdi", "zo_normcap"),
             ("sgdi", "zo_normcap"), ("specgdi_signal", "sgdi"),
             ("specgdi_signal", "specgdi")]
    seen = set()
    for (ds, defe, _) in by:
        if (ds, defe) in seen:
            continue
        seen.add((ds, defe))
        for a_fm, b_fm in pairs:
            ca = by.get((ds, defe, a_fm))
            cb = by.get((ds, defe, b_fm))
            if not ca or not cb:
                continue
            a = np.array(ca["asr_obs"])
            b = np.array(cb["asr_obs"])
            if len(a) != len(b):
                continue
            delta = float(a.mean() - b.mean())
            p = None
            if wilcoxon is not None and np.any(a - b != 0):
                try:
                    p = float(wilcoxon(a, b).pvalue)
                except Exception:
                    p = None
            rows.append({
                "dataset": ds, "defense": defe,
                "pair": f"{a_fm}_vs_{b_fm}",
                "a_mean": round(float(a.mean()), 4),
                "b_mean": round(float(b.mean()), 4),
                "delta": round(delta, 4),
                "wilcoxon_p": p,
                "n_pairs": int(len(a)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="Run only 1 cell (specgdi/none/ENZYMES, seed 0) as a cost probe.")
    ap.add_argument("--datasets", default="PROTEINS,ENZYMES")
    ap.add_argument("--feat_modes", default=",".join(FEAT_MODES_DECISIVE))
    ap.add_argument("--defenses", default=",".join(DEFENSES_DECISIVE))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = "eval_decisive_probe" if args.probe else "eval_decisive"
    jp = RESULTS_DIR / f"{fname}_{stamp}.json"

    datasets = args.datasets.split(",")
    feat_modes = args.feat_modes.split(",")
    defenses = args.defenses.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.probe:
        # Probe: specgdi + sgdi on none, both datasets, 1 seed * 5 folds.
        # Extrapolate full-grid cost from this slice.
        datasets = ["PROTEINS", "ENZYMES"]
        feat_modes = ["specgdi", "sgdi"]
        defenses = ["none"]
        seeds = [0]

    all_cells = []
    meta_info = {
        "stamp": stamp, "probe": args.probe, "attack_cfg": ATTACK_CFG,
        "feat_modes": feat_modes, "defenses": defenses, "seeds": seeds,
        "datasets": datasets, "note": args.note,
        "specgdi_proteins_override": SPECGDI_PROTEINS_OVERRIDE,
    }

    def save_cb(partial_cells):
        merged_cells = all_cells + partial_cells
        # paired tests computed only on the cells gathered so far
        try:
            pt = paired_tests_extended(merged_cells)
        except Exception:
            pt = []
        out = {**meta_info, "cells": merged_cells, "paired_tests": pt}
        jp.write_text(json.dumps(out, indent=2))

    print(f"=== eval_decisive {stamp} | datasets={datasets} "
          f"feat={feat_modes} def={defenses} seeds={seeds} (device={DEVICE}) ===",
          flush=True)
    t0 = time.time()

    if "PROTEINS" in datasets:
        meta = json.loads((MODELS_DIR / "meta.json").read_text())
        folds = load_discrete_folds(meta)
        cells_p = run_grid("PROTEINS", folds, feat_modes, defenses, seeds, save_cb,
                           extra_cfg_for_specgdi=SPECGDI_PROTEINS_OVERRIDE)
        all_cells += cells_p

    if "ENZYMES" in datasets:
        meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
        folds = load_continuous_folds("ENZYMES", meta)
        cells_e = run_grid("ENZYMES", folds, feat_modes, defenses, seeds, save_cb,
                           extra_cfg_for_specgdi=None)
        all_cells += cells_e

    save_cb([])
    print(f"\nSaved {jp} [{time.time()-t0:.0f}s total]", flush=True)


if __name__ == "__main__":
    main()
