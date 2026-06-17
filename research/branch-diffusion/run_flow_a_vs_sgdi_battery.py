#!/usr/bin/env python3
"""
run_flow_a_vs_sgdi_battery.py — DIAGNOSTIC head-to-head driver (NOT a modification
of any attack/model/defense module). Mirrors run_flow_b_defense_battery.py /
defense_eval.py / run_defended_sweep.py: imports the existing attack entry points
and runs them THROUGH each defense on the SAME victim object.

Attack under test : attack_flow.run_attack   (N28 FM variant A, analytic rectified
                    flow + black-box ZO-CW drift + per-step Mahalanobis projection).
Baseline          : attack.run_attack with CONFIG.feat_mode="sgdi" (the existing
                    branch-diffusion SGDI Langevin+Mahalanobis injection).

Both attacks honor the SAME contract  run_attack(model, graphs, device) -> float
and attack the SAME defended victim per (fold, defense) cell — no cross-harness
confound. Defenses: {none, normfilter, mahalanobis}. feat_scale axis {auto_x2, 1.0}
(auto_x2 = the shipped FM/diffusion default; 1.0 = the in-norm stealth regime where
FM's pre-scale in-ball stat literally equals what is injected).

Query metric: a forward-hook QueryCounter tallies victim graphs-per-forward at the
DEFENDED interface (what the attacker actually queries) — identical instrument for
both attacks => apples-to-apples.

DIAGNOSTIC scale only (single seed, few folds, small cap). NOT gate-passed.

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python run_flow_a_vs_sgdi_battery.py [--quick]
"""
import os
# Force CPU before importing prepare (its module-global DEVICE is read at import).
# The two sibling battery scripts do the same: documented CUDA segfault on this GPU,
# and the card is currently busy with an external process. Diagnostic -> CPU is fine.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
import time
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.datasets import TUDataset
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs  # noqa: E402
DEVICE = "cpu"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the LOCAL SGDI attack.py under a unique name FIRST (it is self-contained, no
# `import attack`), so importing attack_flow next (which registers the canonical
# katze attack.py as sys.modules['attack']) cannot clobber our SGDI reference.
sgdi_attack = _load_module("branch_sgdi_attack", HERE / "attack.py")
import attack_flow  # heals env: registers canonical 'attack', imports attack_diffusion
from defenses import (DefendedModel, norm_filter_defense, mahalanobis_defense,  # noqa: E402
                      degree_filter_defense, node_degrees)


# ---- diagnostic config -------------------------------------------------------
N_FOLDS = 3
GRAPH_CAP = 10                       # ~10/fold -> ~30 total
# degfilter = STRUCTURE/degree-aware defense (EVOLVE binding-bug fix): the full-edge
# single-node injection rides node degree ~= N, which feature filters (normfilter/
# mahalanobis) never engage on. degfilter prunes nodes whose degree exceeds the clean
# train mean + k*std, so it genuinely BINDS against this attack.
DEFENSES = ["none", "normfilter", "mahalanobis", "degfilter"]
DEGFILTER_K = 3.0                    # threshold = clean_mean_deg + k*clean_std_deg
FEAT_SCALES = ["auto_x2", "1.0"]     # shipped default ; in-norm stealth regime


class QueryCounter:
    """Tally victim graphs-per-forward (same wrapper as the sibling battery scripts)."""
    def __init__(self, model):
        self.model = model
        self.n = 0
        self._orig = model.forward

    def _graphs(self, inp):
        g = getattr(inp, "num_graphs", None)
        if g is not None:
            return int(g)
        b = getattr(inp, "batch", None)
        if b is not None and b.numel() > 0:
            return int(b.max().item()) + 1
        return 1

    def __enter__(self):
        def counting_forward(*args, **kw):
            if args:
                self.n += self._graphs(args[0])
            self.model.forward = self._orig
            try:
                out = self._orig(*args, **kw)
            finally:
                self.model.forward = counting_forward
            return out
        self.model.forward = counting_forward
        return self

    def __exit__(self, *exc):
        self.model.forward = self._orig


def clean_test_acc(model, ds, test_idx):
    c = t = 0
    for data in Subset(ds, test_idx):
        data = data.to(DEVICE)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def build_defended(model, defense, ref_graphs):
    if defense == "none":
        return model
    if defense == "normfilter":
        return DefendedModel(model, norm_filter_defense, "normfilter", percentile=95)
    if defense == "mahalanobis":
        X = torch.cat([g.x for g in ref_graphs], dim=0).to(DEVICE).float()
        mean_ = X.mean(dim=0); d = X.size(1)
        cov = torch.cov(X.t()) + 1e-3 * torch.eye(d, device=DEVICE)
        cov_inv = torch.linalg.inv(cov)
        diff = X - mean_
        sqm = (diff @ cov_inv * diff).sum(dim=1)
        thr = float(torch.quantile(sqm, 0.995).item())
        return DefendedModel(model, mahalanobis_defense, "mahalanobis",
                             mean=mean_, cov_inv=cov_inv, threshold=thr)
    if defense == "degfilter":
        # Clean degree distribution from the reference clean graphs (same counting
        # convention as the defense via node_degrees). Threshold = mean + k*std is a
        # FIXED clean-distribution value -> the injected full-edge node (degree ~= N)
        # cannot evade it by inflating in-graph stats.
        degs = torch.cat([node_degrees(g) for g in ref_graphs]).float()
        d_mean = float(degs.mean().item())
        d_std = float(degs.std().item())
        return DefendedModel(model, degree_filter_defense, "degfilter",
                             mean=d_mean, std=d_std, k=DEGFILTER_K)
    raise ValueError(defense)


def run_fm(victim, graphs, fs):
    """attack_flow (variant A). Returns (asr, queries, in_ball, total)."""
    old = attack_flow.CONFIG["feat_scale"]
    attack_flow.CONFIG["feat_scale"] = fs
    try:
        with QueryCounter(victim) as qc:
            asr = attack_flow.run_attack(victim, graphs, DEVICE)
        st = attack_flow.FLOW_STATS
        return asr, qc.n, int(st["in_ball"]), int(st["total"])
    finally:
        attack_flow.CONFIG["feat_scale"] = old


def run_sgdi(victim, graphs, fs):
    """attack.py feat_mode=sgdi. Returns (asr, queries)."""
    old_fm = sgdi_attack.CONFIG.get("feat_mode")
    old_fs = sgdi_attack.CONFIG.get("feat_scale")
    old_nb = sgdi_attack.CONFIG.get("node_budget")
    sgdi_attack.CONFIG["feat_mode"] = "sgdi"
    sgdi_attack.CONFIG["feat_scale"] = fs
    sgdi_attack.CONFIG["node_budget"] = 1
    try:
        with QueryCounter(victim) as qc:
            asr = sgdi_attack.run_attack(victim, graphs, DEVICE)
        return asr, qc.n
    finally:
        sgdi_attack.CONFIG["feat_mode"] = old_fm
        sgdi_attack.CONFIG["feat_scale"] = old_fs
        sgdi_attack.CONFIG["node_budget"] = old_nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="1 fold x 4 graphs, feat_scale=auto_x2 only (timing smoke)")
    args = ap.parse_args()

    n_folds = 1 if args.quick else N_FOLDS
    cap = 4 if args.quick else GRAPH_CAP
    feat_scales = ["auto_x2"] if args.quick else FEAT_SCALES

    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== FM(attack_flow, variant A) vs SGDI defense battery | {meta['dataset']} "
          f"{meta.get('model_type','GCN')} | {n_folds} folds x<= {cap} graphs | "
          f"defenses={DEFENSES} | feat_scales={feat_scales} | device={DEVICE} | "
          f"{stamp} ===", flush=True)
    print(f"FM CONFIG: T_flow={attack_flow.CONFIG['T_flow']} "
          f"integrator={attack_flow.CONFIG['integrator']} "
          f"grad={attack_flow.CONFIG['grad_method']} sigma={attack_flow.CONFIG['sigma']} "
          f"n_restarts={attack_flow.CONFIG['n_restarts']} "
          f"edge_strategy={attack_flow.CONFIG['edge_strategy']}", flush=True)

    # accumulators keyed by (feat_scale, defense)
    fm_asr = {(s, d): [] for s in feat_scales for d in DEFENSES}
    sgdi_asr = {(s, d): [] for s in feat_scales for d in DEFENSES}
    fm_q = {(s, d): 0 for s in feat_scales for d in DEFENSES}
    sgdi_q = {(s, d): 0 for s in feat_scales for d in DEFENSES}
    fm_ball = {s: [0, 0] for s in feat_scales}   # [in_ball, total] aggregated per fs
    victim_accs = []
    # clean test accuracy THROUGH each defense (depends on defense only, not feat_scale)
    # — proves a binding defense doesn't tank clean acc. Undefended == DEFENSES[0]=="none".
    clean_acc_def = {dn: [] for dn in DEFENSES}
    fold_records = []
    t0 = time.time()

    for fi in range(n_folds):
        fold_info = meta["folds"][fi]
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
        model.eval()

        vacc = clean_test_acc(model, ds, fold_info["test_idx"])
        victim_accs.append(vacc)
        graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
        rec = {"fold": fi, "victim_acc": round(vacc, 4), "n_graphs": len(graphs), "cells": {}}

        for fs in feat_scales:
            fs_arg = float(fs) if fs.replace(".", "", 1).isdigit() else fs
            for dn in DEFENSES:
                # rebuild defended victim per cell (same object for FM and SGDI)
                victim = build_defended(model, dn, graphs)
                # clean acc through the defense — compute once per (fold, dn)
                if fs == feat_scales[0]:
                    cacc = clean_test_acc(victim, ds, fold_info["test_idx"])
                    clean_acc_def[dn].append(cacc)
                    rec.setdefault("clean_acc_defended", {})[dn] = round(cacc, 4)
                a_fm, q_fm, nb, nt = run_fm(victim, graphs, fs_arg)
                a_sg, q_sg = run_sgdi(victim, graphs, fs_arg)

                fm_asr[(fs, dn)].append(a_fm); sgdi_asr[(fs, dn)].append(a_sg)
                fm_q[(fs, dn)] += q_fm;        sgdi_q[(fs, dn)] += q_sg
                fm_ball[fs][0] += nb;          fm_ball[fs][1] += nt
                rec["cells"][f"{fs}|{dn}"] = {
                    "fm_asr": round(a_fm, 4), "sgdi_asr": round(a_sg, 4),
                    "fm_queries": q_fm, "sgdi_queries": q_sg,
                    "fm_in_ball": nb, "fm_in_ball_total": nt,
                }
                print(f"  [fold {fi}] fs={fs:<7} {dn:<12} "
                      f"FM={a_fm*100:6.2f}% (q={q_fm:6d})  "
                      f"SGDI={a_sg*100:6.2f}% (q={q_sg:6d})  "
                      f"FM_in_ball={nb}/{nt}  vacc={vacc:.3f} n={len(graphs)}",
                      flush=True)
        fold_records.append(rec)

    def agg(d):
        return {f"{s}|{dn}": {"mean_pct": round(mean(d[(s, dn)]) * 100, 2),
                              "fold_std_pct": round(pstdev(d[(s, dn)]) * 100, 2)
                              if len(d[(s, dn)]) > 1 else 0.0,
                              "per_fold": [round(x, 4) for x in d[(s, dn)]]}
                for s in feat_scales for dn in DEFENSES}

    out = {
        "stamp": stamp,
        "experiment": "FM(attack_flow variant A) vs SGDI — defended head-to-head",
        "attack_under_test": "attack_flow.run_attack (N28 FM variant A, analytic rectified flow)",
        "baseline": "attack.run_attack feat_mode=sgdi (branch-diffusion SGDI Langevin+Mahalanobis)",
        "dataset": meta["dataset"],
        "victim_model": meta.get("model_type", "GCN"),
        "mode": "DIAGNOSTIC single-seed (NOT gate-passed)",
        "device": DEVICE,
        "n_folds": n_folds,
        "graph_cap_per_fold": cap,
        "defenses": DEFENSES,
        "feat_scales": feat_scales,
        "wandb": False,
        "fm_config": {k: attack_flow.CONFIG[k] for k in
                      ("T_flow", "integrator", "grad_method", "sigma", "kappa",
                       "loss_type", "edge_strategy", "node_budget", "n_restarts")},
        "sgdi_config": {"feat_mode": "sgdi", "node_budget": 1,
                        "sgdi_eta": sgdi_attack.CONFIG.get("sgdi_eta"),
                        "sgdi_temp": sgdi_attack.CONFIG.get("sgdi_temp"),
                        "zo_epochs": sgdi_attack.CONFIG.get("zo_epochs"),
                        "zo_restarts": sgdi_attack.CONFIG.get("zo_restarts")},
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_per_fold": [round(v, 4) for v in victim_accs],
        "degfilter_k": DEGFILTER_K,
        "clean_acc_by_defense": {dn: {"mean": round(mean(clean_acc_def[dn]), 4),
                                      "per_fold": [round(x, 4) for x in clean_acc_def[dn]]}
                                 for dn in DEFENSES if clean_acc_def[dn]},
        "fm_asr_by_cell": agg(fm_asr),
        "sgdi_asr_by_cell": agg(sgdi_asr),
        "fm_total_queries_by_cell": {f"{s}|{dn}": fm_q[(s, dn)]
                                     for s in feat_scales for dn in DEFENSES},
        "sgdi_total_queries_by_cell": {f"{s}|{dn}": sgdi_q[(s, dn)]
                                       for s in feat_scales for dn in DEFENSES},
        "fm_total_queries_overall": sum(fm_q.values()),
        "sgdi_total_queries_overall": sum(sgdi_q.values()),
        "fm_in_ball_fraction_by_fs": {
            s: {"in_ball": fm_ball[s][0], "total": fm_ball[s][1],
                "fraction": round(fm_ball[s][0] / fm_ball[s][1], 4) if fm_ball[s][1] else None}
            for s in feat_scales},
        "wall_s": round(time.time() - t0, 1),
        "fold_records": fold_records,
    }
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flow_a_vs_sgdi_battery_{meta['dataset']}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 72, flush=True)
    print(f"victim_acc(clean) mean={out['victim_acc_mean']}  "
          f"per-fold={out['victim_acc_per_fold']}", flush=True)
    print(f"\n--- clean test acc THROUGH each defense (mean over folds) ---", flush=True)
    for dn in DEFENSES:
        c = out["clean_acc_by_defense"].get(dn)
        if c:
            print(f"  {dn:<12} {c['mean']*100:6.2f}%   per-fold={c['per_fold']}", flush=True)
    for fs in feat_scales:
        print(f"\n--- feat_scale={fs} : ASR by defense (mean% +/- fold_std%) ---", flush=True)
        print(f"  {'defense':<12} {'FM(attack_flow)':<22} {'SGDI':<22}", flush=True)
        for dn in DEFENSES:
            f = out["fm_asr_by_cell"][f"{fs}|{dn}"]
            s = out["sgdi_asr_by_cell"][f"{fs}|{dn}"]
            print(f"  {dn:<12} {f['mean_pct']:6.2f} +/- {f['fold_std_pct']:<11.2f} "
                  f"{s['mean_pct']:6.2f} +/- {s['fold_std_pct']:<11.2f}", flush=True)
        b = out["fm_in_ball_fraction_by_fs"][fs]
        print(f"  FM injected-feat in clean Mahalanobis ball (pre-scale): "
              f"{b['in_ball']}/{b['total']} = "
              f"{(b['fraction']*100 if b['fraction'] is not None else float('nan')):.1f}%", flush=True)
    print(f"\nqueries overall: FM={out['fm_total_queries_overall']}  "
          f"SGDI={out['sgdi_total_queries_overall']}  "
          f"(ratio FM/SGDI={out['fm_total_queries_overall']/max(out['sgdi_total_queries_overall'],1):.3f})",
          flush=True)
    print(f"Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
