#!/usr/bin/env python3
"""eval_pg_floor_nb10ctrl.py — SAME-BUDGET (node_budget=10) CONTROL for the paper-grade
worst-case-defense FLOOR study. Closes the node-budget CONFOUND in the primary study.

------------------------------------------------------------------------------
Confound being closed
------------------------------------------------------------------------------
The primary study (eval_paper_grade_floor.py) showed OM2C-nb10 (10 low-degree nodes +
on-manifold fs=1.0) has a HIGH worst-case-defense floor while the nb=1 single-axis
baselines collapse. But OM2C-nb10 spends 10x the NODE budget vs those nb=1 baselines,
so the floor advantage could be a budget artifact rather than the ORTHOGONALITY of its
two channels (degree-1 mass channel ⟂ on-manifold feature channel).

This control fixes node_budget=10 for EVERY attack and tests whether each SAME-BUDGET
single-axis baseline STILL collapses on its complementary defense. If so, OM2C-nb10's
floor advantage is the orthogonality, NOT the node budget.

H1 (control): each nb=10 single-axis baseline collapses (floor ≈ 0) on >= 1 defense
axis; OM2C-nb10's floor >> next-best nb=10 baseline floor, paired-significant.

------------------------------------------------------------------------------
Attacks (ALL node_budget=10; same victim per (seed,fold) cell — no confound)
------------------------------------------------------------------------------
  SGDI-nb10-full       sgdi, node_budget=10, fs=auto_x2, full-edge anchor (rho=1.0).
                       10 injected nodes each wired to ALL N victim nodes => degree≈N.
                       EXPECT: dies on degfilter (extreme injected-node degree).
  FM-nb10-full         flow, edge_strategy=full, node_budget=10, fs=auto_x2.
                       10 high-norm nodes each wired to ALL N nodes (degree≈N).
                       EXPECT: dies on degfilter AND normfilter (high degree + high norm).
  FM-nb10-eb1-highnorm flow, edge_strategy=topk, node_budget=10, fs=auto_x2.
                       Edge-sparser high-norm control. NOTE on realized degree:
                       attack_flow.construct_perturbed_graph wires EACH injected node to
                       ALL `targets`, and select_targets_topk returns node_budget targets,
                       so the realized per-node degree == node_budget == 10 (NOT 1). This
                       is still far sparser than the full-edge degree≈N regime, so it
                       isolates the FEATURE axis: high norm with bounded (non-full) degree.
                       EXPECT: dies on normfilter / mahalanobis (high norm); the degfilter
                       outcome at degree-10 is reported empirically.
  OM2C-nb10            om2c, node_budget=10, fs=1.0  — the candidate, re-included as the
                       anchor (degree-1 sparse-greedy mass channel + on-manifold features).

floor(attack) := min ASR over EFFECTIVE defenses {normfilter, mahalanobis, degfilter}.

DRIVER ONLY — no attack/model/defense code modified. Reuses the PRIMARY runner's frozen
helpers (build_defended, run_attack_cell, om2c_greedy_flip, QueryCounter, clean_test_acc,
load_meta_for_seed) verbatim by import; only the ATTACKS dict + output naming differ.

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python eval_pg_floor_nb10ctrl.py [--seeds 42,7,1027] [--cap 10] [--quick]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, argparse
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.datasets import TUDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"
DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs  # noqa: E402

# Reuse the PRIMARY runner's frozen harness logic verbatim (driver glue, not attack code).
from eval_paper_grade_floor import (DEFENSES, EFFECTIVE, DEGFILTER_K,  # noqa: E402
                                    build_defended, run_attack_cell,
                                    clean_test_acc, load_meta_for_seed)
import attack_om2c as O  # noqa: E402

# Same-budget (node_budget=10) attack battery. OM2C-nb10 is the candidate anchor.
ATTACKS = {
    "SGDI-nb10-full":       ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-full":         ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-highnorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": "auto_x2"}),
    "OM2C-nb10":            ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
}
CANDIDATE = "OM2C-nb10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = None
    if args.quick:
        seeds = seeds[:1]; cap = 4; fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    in_ball = {a: [0, 0] for a in ATTACKS}
    extra_acc = {a: {"max_inj_degree": 0, "edges": []} for a in ATTACKS}
    floor_per_unit = {a: [] for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; units = []
    meta0 = load_meta_for_seed(seeds[0])

    print(f"=== NB10 SAME-BUDGET CONTROL (floor study) | {meta0['dataset']} "
          f"{meta0.get('model_type','GCN')} | seeds={seeds} | cap={cap}/fold | "
          f"defenses={DEFENSES} | attacks={list(ATTACKS)} | candidate={CANDIDATE} | "
          f"device={DEVICE} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = load_meta_for_seed(seed)
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            model.eval()
            vacc = clean_test_acc(model, ds, fold_info["test_idx"]); victim_accs.append(vacc)
            graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = build_defended(model, dn, graphs)
                ca = clean_test_acc(v, ds, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                asrs = {}
                for dn in DEFENSES:
                    victim = build_defended(model, dn, graphs)
                    pruned_crash = False
                    try:
                        r = run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                    except IndexError:
                        # Documented full-prune neutralization (primary study docstring):
                        # a pruning defense (e.g. degfilter on full-edge nb=10) removes ALL
                        # injected nodes, collapsing the anchor batch so attack.py's
                        # _eval_graphs returns fewer margins than anchors -> IndexError.
                        # The graph is fully pruned => the perturbation is erased => no flip
                        # => ASR = 0. Recorded honestly as a neutralized cell (driver-level
                        # guard; NO attack code modified). Flagged for transparency.
                        r = {"asr": 0.0, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                        pruned_crash = True
                    cell_asr[akey][dn].append(r["asr"]); cell_q[akey][dn] += r["q"]
                    if dn == "none":
                        in_ball[akey][0] += r["in_ball"]; in_ball[akey][1] += r["in_ball_total"]
                        ex = r.get("extra", {})
                        if ex.get("max_inj_degree") is not None:
                            extra_acc[akey]["max_inj_degree"] = max(extra_acc[akey]["max_inj_degree"], ex["max_inj_degree"])
                        if ex.get("median_edges_at_flip") is not None:
                            extra_acc[akey]["edges"].append(ex["median_edges_at_flip"])
                    asrs[dn] = round(r["asr"], 4)
                    unit["cells"][f"{akey}|{dn}"] = {"asr": round(r["asr"], 4), "q": r["q"],
                                                     **({"defense_pruned_crash": True} if pruned_crash else {})}
                floor = min(asrs[d] for d in EFFECTIVE)
                floor_per_unit[akey].append(floor)
                unit["cells"][f"{akey}|FLOOR"] = round(floor, 4)
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}:fl={floor_per_unit[a][-1]*100:.0f}" for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

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
        else:
            w_stat, w_p = None, "all_diffs_zero"
        t_stat, t_p = ttest_rel(cand, nb)
        paired.update({
            "candidate_floor_mean_pct": floor_summary[CANDIDATE]["mean_pct"],
            "next_best_floor_mean_pct": floor_summary[next_best]["mean_pct"],
            "mean_diff_pct": round(mean(diffs) * 100, 3),
            "wilcoxon_stat": (None if w_stat is None else float(w_stat)),
            "wilcoxon_p": (w_p if isinstance(w_p, str) else float(w_p)),
            "paired_t_stat": float(t_stat), "paired_t_p": float(t_p), "n_pairs": len(diffs),
        })
    except ImportError:
        paired["error"] = "scipy unavailable"

    out = {
        "stamp": stamp,
        "experiment": "NB10 SAME-BUDGET CONTROL for paper-grade floor study (closes node-budget confound)",
        "hypothesis_H1": "Each nb=10 single-axis baseline still collapses on its complementary "
                         "defense; OM2C-nb10 floor >> next-best nb=10 baseline floor (p<0.01). "
                         "=> floor advantage is ORTHOGONALITY, not node budget.",
        "candidate": CANDIDATE,
        "controls_for": "node_budget (all attacks node_budget=10; same victim per cell)",
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE, "degfilter_k": DEGFILTER_K,
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "fm_realized_degree_note": "FM-nb10-eb1-highnorm realized per-node degree == node_budget "
                                   "(10), not 1, due to attack_flow bipartite-complete wiring; "
                                   "still far below full-edge degree≈N. Reported honestly.",
        "om2c_metric": "SPARSE greedy regime (degree-1 mass channel); dense frac_anchors NOT used",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "floor_summary": floor_summary, "asr_summary": asr_summary,
        "total_queries_by_attack": {a: sum(cell_q[a].values()) for a in ATTACKS},
        "queries_by_attack_defense": {a: {d: cell_q[a][d] for d in DEFENSES} for a in ATTACKS},
        "in_ball_fraction_by_attack": {
            a: {"in_ball": in_ball[a][0], "total": in_ball[a][1],
                "fraction": round(in_ball[a][0] / in_ball[a][1], 4) if in_ball[a][1] else None}
            for a in ATTACKS},
        "om2c_degree_info": {a: {"max_inj_degree_observed": extra_acc[a]["max_inj_degree"],
                                 "median_edges_at_flip": (sorted(extra_acc[a]["edges"])[len(extra_acc[a]["edges"])//2]
                                                          if extra_acc[a]["edges"] else None)}
                             for a in ATTACKS if ATTACKS[a][0] == "om2c"},
        "paired_test_H1": paired,
        "floor_per_unit": {a: [round(x, 4) for x in floor_per_unit[a]] for a in ATTACKS},
        "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"paper_grade_floor_nb10ctrl_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 84, flush=True)
    print(f"victim_acc: mean={out['victim_acc_mean']} std={out['victim_acc_std']} min={out['victim_acc_min']} (floor 0.45)", flush=True)
    print("clean acc thru defenses:", {d: out['clean_acc_by_defense'][d]['mean'] for d in DEFENSES}, flush=True)
    print(f"\n{'attack':<22} " + " ".join(f"{d:>13}" for d in DEFENSES) + f" {'FLOOR':>13}", flush=True)
    for a in ATTACKS:
        cells = " ".join(f"{asr_summary[a][d]['mean_pct']:6.1f}±{asr_summary[a][d]['std_pct']:<5.1f}" for d in DEFENSES)
        fl = floor_summary[a]
        print(f"{a:<22} {cells}  {fl['mean_pct']:6.1f}±{fl['std_pct']:<5.1f}", flush=True)
    print(f"\nH1 paired ({CANDIDATE} floor vs next-best={next_best}): "
          f"cand={paired.get('candidate_floor_mean_pct')}% nextbest={paired.get('next_best_floor_mean_pct')}% "
          f"diff={paired.get('mean_diff_pct')}%  wilcoxon_p={paired.get('wilcoxon_p')} "
          f"t_p={paired.get('paired_t_p')} n={paired.get('n_pairs')}", flush=True)
    print(f"om2c degree info: {out['om2c_degree_info']}", flush=True)
    print(f"queries: " + " ".join(f"{a}={sum(cell_q[a].values())}" for a in ATTACKS), flush=True)
    print(f"\nSaved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
