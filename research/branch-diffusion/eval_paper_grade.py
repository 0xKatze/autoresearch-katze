#!/usr/bin/env python3
"""
eval_paper_grade.py — §4.4 distributional-stealth evaluation (NEW; core harness untouched).

Tests the branch-diffusion thesis: a distribution-matched injection (SGDI, guided diffusion
inside the data covariance ellipsoid) survives DISTRIBUTION-AWARE defenses that magnitude-only
norm-cap (N4) does not — and that this gap only opens on CONTINUOUS features, being degenerate
on PROTEINS binary one-hot (norm=1.0). See n4-stealth-precedent.

Grid: dataset {PROTEINS discrete, ENZYMES continuous} x feat_mode {manifold, zo_normcap (=N4
magnitude-only baseline), zo_free (detectable upper ref), sgdi (ours)} x defense {none,
normfilter, jaccard, mahalanobis (the dist-aware linchpin)} x seeds {0,1} x 5 folds.

Per cell (pooled 2 seeds x 5 folds = 10 obs): ASR at FIXED full-edge budget (not best-rho),
defended clean acc, queries/graph. Paired Wilcoxon SGDI vs zo_normcap per (dataset,defense).
Defense stats (featclip mean/std; mahalanobis mu/cov/threshold) calibrated on each fold's clean
test nodes at the 99.5th Mahalanobis percentile so clean acc is preserved. Every number -> JSON.

Usage: micromamba run -n graph_adversarial python eval_paper_grade.py [--quick]
"""
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from prepare import MODELS_DIR, RESULTS_DIR, DEVICE, GCN_GraphClassification
from sweep_stealth import load_fold as load_fold_discrete
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from defenses import (DefendedModel, jaccard_defense, norm_filter_defense,
                      featclip_defense, mahalanobis_defense, gnnsafe_defense,
                      _gnnsafe_node_logits, _energy_propagation,
                      gddm_defense, gddm_ckpt_path,
                      gddm_a_defense, gddm_a_ckpt_path)
import attack

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

FEAT_MODES = ["manifold", "zo_normcap", "zo_free", "sgdi", "specgdi", "specgdi_signal", "specgdi_v2"]
DEFENSES = ["none", "normfilter", "jaccard", "mahalanobis", "gnnsafe", "gddm", "gddm_a"]
GDDM_T_STAR = 10  # partial-noise level for DiffPure-style purification (see GDDM_NOTES.md)
GDDM_A_T_STAR = 30  # A-channel partial-noise level (see GDDM_A_NOTES.md)
GNNSAFE_T, GNNSAFE_K, GNNSAFE_ALPHA = 1.0, 2, 0.5
GNNSAFE_PCT_LOW = 5.0  # 5th-pct of neg_energy on clean ref nodes
SEEDS = [0, 1]
ATTACK_CFG = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 20, "zo_restarts": 2}


# ---- dataset loaders: each returns list[(model, graphs)] per fold (graphs on DEVICE) ----

def load_discrete_folds(meta):
    out = []
    for fi in meta["folds"]:
        out.append(load_fold_discrete(fi, meta))   # (model, correctly-classified test graphs)
    return out


def load_continuous_folds(name, meta):
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs(name, mean, std)  # load once
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
        out.append((model, graphs))
    return out


# ---- defense wrapping with per-fold clean-node calibration ----

def build_defended(model, defense, ref_graphs, ds_key=None):
    if defense == "none":
        return model
    if defense == "normfilter":
        return DefendedModel(model, norm_filter_defense, "normfilter", percentile=95)
    if defense == "jaccard":
        return DefendedModel(model, jaccard_defense, "jaccard",
                             threshold=0.01, similarity="cosine")
    X = torch.cat([g.x for g in ref_graphs], dim=0).to(DEVICE).float()
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)
    if defense == "featclip":
        return DefendedModel(model, featclip_defense, "featclip", mean=mean, std=std, k=2.0)
    if defense == "mahalanobis":
        d = X.size(1)
        cov = torch.cov(X.t()) + 1e-3 * torch.eye(d, device=DEVICE)
        cov_inv = torch.linalg.inv(cov)
        diff = X - mean
        sqm = (diff @ cov_inv * diff).sum(dim=1)
        thr = float(torch.quantile(sqm, 0.995).item())  # 99.5% clean nodes pass
        return DefendedModel(model, mahalanobis_defense, "mahalanobis",
                             mean=mean, cov_inv=cov_inv, threshold=thr)
    if defense == "gnnsafe":
        # Calibrate threshold on clean ref graphs: 5th-pct of propagated neg_energy
        # (mirrors calibrate_gnnsafe.py; symmetric to a 95th-pct cut on +E).
        all_e = []
        for g in ref_graphs:
            with torch.no_grad():
                logits = _gnnsafe_node_logits(model, g)
                ne = GNNSAFE_T * torch.logsumexp(logits / GNNSAFE_T, dim=-1)
                ne = _energy_propagation(ne, g.edge_index, g.num_nodes,
                                         K=GNNSAFE_K, alpha=GNNSAFE_ALPHA)
            all_e.append(ne)
        ne_all = torch.cat(all_e) if all_e else torch.zeros(0, device=DEVICE)
        thr = float(torch.quantile(ne_all, GNNSAFE_PCT_LOW / 100.0).item())
        return DefendedModel(model, gnnsafe_defense, "gnnsafe",
                             victim_model=model, threshold=thr,
                             T=GNNSAFE_T, K=GNNSAFE_K, alpha=GNNSAFE_ALPHA)
    if defense == "gddm":
        # Per-dataset trained denoiser (see train_gddm.py + GDDM_NOTES.md).
        # If the checkpoint is missing on disk, gddm_defense returns the input unchanged.
        ckpt = gddm_ckpt_path(ds_key) if ds_key else None
        return DefendedModel(model, gddm_defense, "gddm",
                             ckpt_path=ckpt, t_star=GDDM_T_STAR)
    if defense == "gddm_a":
        # A-channel (adjacency) GDDM (see train_gddm_a.py + GDDM_A_NOTES.md).
        ckpt = gddm_a_ckpt_path(ds_key) if ds_key else None
        return DefendedModel(model, gddm_a_defense, "gddm_a",
                             ckpt_path=ckpt, t_star=GDDM_A_T_STAR)
    raise ValueError(defense)


def clean_acc_defended(dmodel, graphs):
    if not graphs:
        return 0.0
    c = 0
    for g in graphs:
        with torch.no_grad():
            if dmodel(g).argmax(-1).item() == g.y.item():
                c += 1
    return c / len(graphs)


def run_dataset(ds_key, folds_data, feat_modes, defenses, seeds, save_cb):
    cells = []
    for defense in defenses:
        defended, cleanaccs = [], []
        for (model, graphs) in folds_data:
            dm = build_defended(model, defense, graphs, ds_key=ds_key)
            defended.append(dm)
            cleanaccs.append(clean_acc_defended(dm, graphs))
        for fm in feat_modes:
            obs, qs, t0 = [], [], time.time()
            for seed in seeds:
                for dm, (model, graphs) in zip(defended, folds_data):
                    cfg = {**ATTACK_CFG, "feat_mode": fm, "seed_offset": seed}
                    o = attack.run_attack_sweep(dm, graphs, DEVICE, cfg)
                    obs.append(o["asr_full"])
                    qs.append(o["queries"] / max(o["n_graphs"], 1))
            rec = {"dataset": ds_key, "defense": defense, "feat_mode": fm,
                   "asr_mean": float(np.mean(obs)), "asr_std": float(np.std(obs)),
                   "asr_obs": [round(x, 4) for x in obs],
                   "q_per_graph": float(np.mean(qs)),
                   "clean_acc_def_mean": float(np.mean(cleanaccs)),
                   "n_obs": len(obs)}
            cells.append(rec)
            print(f"  [{ds_key}] {defense:<11} {fm:<10} ASR={rec['asr_mean']:.3f}"
                  f"±{rec['asr_std']:.3f} cleanAcc={rec['clean_acc_def_mean']:.3f} "
                  f"q/g={rec['q_per_graph']:.0f} [{time.time()-t0:.0f}s]")
            save_cb(cells)
    return cells


def paired_tests(cells):
    by = {(c["dataset"], c["defense"], c["feat_mode"]): c for c in cells}
    rows = []
    for (ds, defe, fm) in list(by):
        if fm != "sgdi":
            continue
        base = by.get((ds, defe, "zo_normcap"))
        if not base:
            continue
        a = np.array(by[(ds, defe, "sgdi")]["asr_obs"])
        b = np.array(base["asr_obs"])
        delta = float(a.mean() - b.mean())
        p = None
        if wilcoxon is not None and len(a) == len(b) and np.any(a - b != 0):
            try:
                p = float(wilcoxon(a, b).pvalue)
            except Exception:
                p = None
        rows.append({"dataset": ds, "defense": defe, "delta_sgdi_minus_normcap": round(delta, 4),
                     "wilcoxon_p": p})
    return rows


def main():
    quick = "--quick" in sys.argv
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"eval_paper_grade_{stamp}.json"

    feat_modes, defenses, seeds = FEAT_MODES, DEFENSES, SEEDS
    datasets = ["PROTEINS", "ENZYMES"]
    if quick:
        feat_modes = ["manifold", "zo_normcap", "sgdi"]
        defenses = ["none", "mahalanobis"]
        seeds = [0]
        datasets = ["ENZYMES"]

    all_cells = []

    def save_cb(partial_cells):
        merged = all_cells + partial_cells
        jp.write_text(json.dumps(
            {"stamp": stamp, "quick": quick, "attack_cfg": ATTACK_CFG,
             "feat_modes": feat_modes, "defenses": defenses, "seeds": seeds,
             "cells": merged, "paired_tests": paired_tests(merged)}, indent=2))

    print(f"=== eval_paper_grade {stamp} | datasets={datasets} feat={feat_modes} "
          f"def={defenses} seeds={seeds} (device={DEVICE}) ===")
    t0 = time.time()

    if "PROTEINS" in datasets:
        meta = json.loads((MODELS_DIR / "meta.json").read_text())
        folds = load_discrete_folds(meta)
        if quick:
            folds = folds[:1]
        all_cells += run_dataset("PROTEINS", folds, feat_modes, defenses, seeds, save_cb)

    if "ENZYMES" in datasets:
        meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
        folds = load_continuous_folds("ENZYMES", meta)
        if quick:
            folds = folds[:1]
        all_cells += run_dataset("ENZYMES", folds, feat_modes, defenses, seeds, save_cb)

    save_cb([])
    print(f"\nSaved {jp} [{time.time()-t0:.0f}s total]")

    # grid + paired-test summary
    by = {(c["dataset"], c["defense"], c["feat_mode"]): c for c in all_cells}
    for ds in datasets:
        print(f"\nASR @ full-edge budget — {ds} (rows=feat_mode, cols=defense):")
        print("feat_mode   " + "".join(f"{d:>13}" for d in defenses))
        for fm in feat_modes:
            row = f"{fm:<11} "
            for d in defenses:
                c = by.get((ds, d, fm))
                row += f"{(c['asr_mean'] if c else float('nan')):>12.1%} "
            print(row)
    print("\nPaired SGDI - zo_normcap (Δ ASR, Wilcoxon p):")
    for r in paired_tests(all_cells):
        print(f"  {r['dataset']:<9} {r['defense']:<11} Δ={r['delta_sgdi_minus_normcap']:+.3f} "
              f"p={r['wilcoxon_p']}")


if __name__ == "__main__":
    main()
