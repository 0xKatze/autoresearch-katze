#!/usr/bin/env python3
"""eval_pg_floor_enzymes.py — SECONDARY (best-effort) continuous-feature floor study
on ENZYMES (21-dim z-scored node attrs), GCN victim. Variant-B's proper home.

REDUCED RIGOR vs the PROTEINS primary: the existing continuous victims are 5-fold,
seed 42 only (models_cont/meta_cont_ENZYMES.json) -> 5 (fold) units, single seed. No
new victims trained (continuous training infra = prepare_cont.py, seed-42 only). This
is documented honestly as reduced-rigor secondary evidence, NOT gate-grade.

Reuses the PRIMARY runner's helpers (ATTACKS, build_defended, run_attack_cell,
om2c_greedy_flip, QueryCounter) verbatim by import; only the data layer differs
(continuous z-scored ENZYMES loaded with the meta's stored feat_mean/feat_std).

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python eval_pg_floor_enzymes.py [--cap 10]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, argparse
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"
DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR  # noqa: E402

# reuse the primary runner's frozen logic
import eval_paper_grade_floor as PG  # noqa: E402
from eval_paper_grade_floor import (ATTACKS, DEFENSES, EFFECTIVE, CANDIDATE,  # noqa: E402
                                    build_defended, run_attack_cell, clean_test_acc)
import attack_om2c as O  # noqa: E402

META_PATH = HERE / "models_cont" / "meta_cont_ENZYMES.json"


def load_zscored_enzymes(meta):
    ds = TUDataset(root=str(DATA_DIR / f"{meta['dataset']}_attr"),
                   name=meta["dataset"], use_node_attr=True)
    mean_ = torch.tensor(meta["feat_mean"]).float().unsqueeze(0)
    std_ = torch.tensor(meta["feat_std"]).float().unsqueeze(0).clamp_min(1e-6)
    graphs = []
    for i in range(len(ds)):
        g = ds[i]
        xz = (g.x.float() - mean_) / std_
        graphs.append(Data(x=xz, edge_index=g.edge_index, y=g.y))
    return graphs


def collect_correct(model, graphs, test_idx):
    out = []
    for i in test_idx:
        d = graphs[i].to(DEVICE)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if int(model(d).argmax(-1).item()) == int(d.y.item()):
                out.append(d)
    return out


def clean_acc(model, graphs, test_idx):
    c = t = 0
    for i in test_idx:
        d = graphs[i].to(DEVICE)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            c += int(int(model(d).argmax(-1).item()) == int(d.y.item())); t += 1
    return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=10)
    args = ap.parse_args()
    meta = json.loads(META_PATH.read_text())
    graphs_all = load_zscored_enzymes(meta)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== ENZYMES-cont FLOOR (SECONDARY, reduced rigor: {len(meta['folds'])}-fold seed{meta['seed']}) | "
          f"GCN | feat={meta['num_features']} classes={meta['num_classes']} | cap={args.cap} | "
          f"attacks={list(ATTACKS)} candidate={CANDIDATE} | {stamp} ===", flush=True)

    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    in_ball = {a: [0, 0] for a in ATTACKS}
    floor_per_unit = {a: [] for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; units = []
    t0 = time.time()

    for fold_info in meta["folds"]:
        fi = fold_info["fold_idx"]
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
        model.eval()
        vacc = clean_acc(model, graphs_all, fold_info["test_idx"]); victim_accs.append(vacc)
        graphs = collect_correct(model, graphs_all, fold_info["test_idx"])[:args.cap]
        if not graphs:
            print(f"  fold{fi}: 0 attackable graphs, skip", flush=True); continue
        om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                  O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
        cacc_def = {}
        for dn in DEFENSES:
            v = build_defended(model, dn, graphs)
            ca = clean_acc(v, graphs_all, fold_info["test_idx"])
            clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
        unit = {"fold": fi, "victim_acc": round(vacc, 4), "n_graphs": len(graphs), "cells": {}}
        for akey, (rk, ov) in ATTACKS.items():
            asrs = {}
            for dn in DEFENSES:
                victim = build_defended(model, dn, graphs)
                r = run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                cell_asr[akey][dn].append(r["asr"]); cell_q[akey][dn] += r["q"]
                if dn == "none":
                    in_ball[akey][0] += r["in_ball"]; in_ball[akey][1] += r["in_ball_total"]
                asrs[dn] = round(r["asr"], 4)
            floor = min(asrs[d] for d in EFFECTIVE)
            floor_per_unit[akey].append(floor)
            unit["cells"][akey] = {**{d: asrs[d] for d in DEFENSES}, "FLOOR": round(floor, 4)}
        units.append(unit)
        print(f"  fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
              "  ".join(f"{a}:fl={floor_per_unit[a][-1]*100:.0f}" for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst)*100, 2),
                "std_pct": round(pstdev(lst)*100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}
    floor_summary = {a: agg(floor_per_unit[a]) for a in ATTACKS}
    asr_summary = {a: {d: agg(cell_asr[a][d]) for d in DEFENSES} for a in ATTACKS}
    others = [a for a in ATTACKS if a != CANDIDATE]
    next_best = max(others, key=lambda a: floor_summary[a]["mean_pct"])
    paired = {"candidate": CANDIDATE, "next_best": next_best}
    try:
        from scipy.stats import wilcoxon, ttest_rel
        cand = floor_per_unit[CANDIDATE]; nb = floor_per_unit[next_best]
        diffs = [c - n for c, n in zip(cand, nb)]
        if any(d != 0 for d in diffs):
            try: w_stat, w_p = wilcoxon(cand, nb)
            except Exception as e: w_stat, w_p = None, f"wilcoxon_error:{e}"
        else: w_stat, w_p = None, "all_diffs_zero"
        t_stat, t_p = ttest_rel(cand, nb)
        paired.update({"candidate_floor_mean_pct": floor_summary[CANDIDATE]["mean_pct"],
                       "next_best_floor_mean_pct": floor_summary[next_best]["mean_pct"],
                       "mean_diff_pct": round(mean(diffs)*100, 3),
                       "wilcoxon_p": (w_p if isinstance(w_p, str) else float(w_p)),
                       "paired_t_p": float(t_p), "n_pairs": len(diffs)})
    except ImportError:
        paired["error"] = "scipy unavailable"

    out = {"stamp": stamp, "experiment": "SECONDARY ENZYMES-cont floor (reduced rigor)",
           "rigor_note": f"{len(meta['folds'])}-fold seed{meta['seed']} only (single seed) — NOT gate-grade",
           "dataset": "ENZYMES", "victim_model": "GCN", "continuous": True,
           "candidate": CANDIDATE, "device": DEVICE, "cap_per_fold": args.cap,
           "n_units": len(units), "defenses": DEFENSES, "effective_defenses": EFFECTIVE,
           "degfilter_k": PG.DEGFILTER_K,
           "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
           "victim_acc_mean": round(mean(victim_accs), 4),
           "victim_acc_std": round(pstdev(victim_accs), 4),
           "victim_acc_min": round(min(victim_accs), 4),
           "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4)} for d in DEFENSES if clean_acc_def[d]},
           "floor_summary": floor_summary, "asr_summary": asr_summary,
           "total_queries_by_attack": {a: sum(cell_q[a].values()) for a in ATTACKS},
           "in_ball_fraction_by_attack": {a: {"in_ball": in_ball[a][0], "total": in_ball[a][1],
               "fraction": round(in_ball[a][0]/in_ball[a][1], 4) if in_ball[a][1] else None} for a in ATTACKS},
           "paired_test_H1": paired,
           "floor_per_unit": {a: [round(x, 4) for x in floor_per_unit[a]] for a in ATTACKS},
           "wall_s": round(time.time()-t0, 1), "units": units}
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"paper_grade_floor_ENZYMES_cont_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 84, flush=True)
    print(f"victim_acc mean={out['victim_acc_mean']} std={out['victim_acc_std']} min={out['victim_acc_min']}", flush=True)
    print(f"\n{'attack':<12} " + " ".join(f"{d:>13}" for d in DEFENSES) + f" {'FLOOR':>13}", flush=True)
    for a in ATTACKS:
        cells = " ".join(f"{asr_summary[a][d]['mean_pct']:6.1f}±{asr_summary[a][d]['std_pct']:<5.1f}" for d in DEFENSES)
        fl = floor_summary[a]
        print(f"{a:<12} {cells}  {fl['mean_pct']:6.1f}±{fl['std_pct']:<5.1f}", flush=True)
    print(f"\nH1 paired ({CANDIDATE} vs {next_best}): cand={paired.get('candidate_floor_mean_pct')}% "
          f"nextbest={paired.get('next_best_floor_mean_pct')}% wilcoxon_p={paired.get('wilcoxon_p')} "
          f"t_p={paired.get('paired_t_p')} n={paired.get('n_pairs')}", flush=True)
    print(f"Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
