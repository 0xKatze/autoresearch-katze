#!/usr/bin/env python3
"""eval_pg_floor_om2c_fm.py — FLOOR study for the MISSING 2x2 CELL: OM2C+FM.

Fills feature {ZO-tangent, FM} x edge {sparse-greedy-deg1, topk}:
  OM2C-nb10            ZO-tangent + sparse-greedy  (anchor, query-EXPENSIVE, floor5~22.7%)
  FM-nb10-eb1-lownorm  FM         + topk           (control, query-cheap,    floor5~8.7%)
  OM2C+FM              FM         + sparse-greedy   (the MISSING cell, this study)

H_mech: OM2C+FM floor5 ~= OM2C's 22.7% (>> the 8.7% FM+topk control) => the EDGE channel
        (degree-1 sparse-greedy mass), not the feature-gen method, sources the floor.
H_cost: OM2C+FM total queries << OM2C (FM feature gen cheaper than ZO feature opt).

DRIVER ONLY — no attack/model/defense code modified. Reuses eval_paper_grade_floor's
frozen harness (build_defended, run_attack_cell, training_ref_graphs, clean_test_acc,
load_meta_for_seed, QueryCounter), defense_battery6's BiSmoothingModel, attack_om2c (O),
and the new additive composition attack_om2c_fm (FM features + OM2C stealth + OM2C greedy).

Battery (== task request): none, normfilter, mahalanobis, degfilter, bismooth.
The rspool architectural test-time readout swap is DROPPED (invalid: swaps the readout on a
mean-trained GCN; and it never binds — rspool ASR was HIGH for both OM2C and the control in
floor_battery6_*_20260622_220259.json, so the floor is set by bismooth/degfilter anyway).
floor5(attack) := min ASR over EFFECTIVE {normfilter, mahalanobis, degfilter, bismooth}.

CPU-pinned (FM CGE latency-bound + device-FP ASR divergence per memory; GPU busy w/ gap2).
Seed-level floor aggregation (n=3 genuine replicates; 10 folds within a seed share data).

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python eval_pg_floor_om2c_fm.py [--seeds 42,7,1027] [--cap 10] \
         [--bismooth-T 20] [--bismooth-p 0.2] [--quick] [--max-folds N]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU ONLY (do not contend with gap2 GPU)
import sys, json, time, argparse, traceback
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

import eval_paper_grade_floor as F          # frozen harness (build_defended/run_attack_cell/...)
import attack_om2c as O                      # OM2C primitives
import attack_om2c_fm as AFM                 # the OM2C+FM composition (additive)
from defense_battery6 import BiSmoothingModel  # node-deletion randomized smoothing

# Battery: 3 degfilter-era axes + bi-smoothing. rspool DROPPED (see docstring).
DEFENSES = ["none", "normfilter", "mahalanobis", "degfilter", "bismooth"]
EFFECTIVE = ["normfilter", "mahalanobis", "degfilter", "bismooth"]

ATTACKS = {
    "OM2C+FM":             ("om2c_fm", {"node_budget": 10, "feat_scale": 1.0}),  # MISSING cell
    "OM2C-nb10":           ("om2c",    {"node_budget": 10, "feat_scale": 1.0}),  # anchor
    "FM-nb10-eb1-lownorm": ("flow",    {"edge_strategy": "topk", "node_budget": 10,
                                        "feat_scale": 1.0}),                      # control
}
CANDIDATE = "OM2C+FM"
ANCHOR = "OM2C-nb10"
CONTROL = "FM-nb10-eb1-lownorm"

_BT, _BP = 20, 0.2   # bi-smoothing T / p (set from CLI)


def build_defended_ext(model, dn, ref_graphs):
    if dn == "bismooth":
        return BiSmoothingModel(model, T=_BT, p=_BP)
    return F.build_defended(model, dn, ref_graphs)


def run_attack_cell_ext(run_key, overrides, victim, graphs, om2c_prior_cache):
    """Delegate sgdi/flow/om2c to the frozen harness; handle the new om2c_fm cell here."""
    if run_key != "om2c_fm":
        return F.run_attack_cell(run_key, overrides, victim, graphs, om2c_prior_cache)
    # --- OM2C+FM: FM features + OM2C stealth + OM2C sparse-greedy edges ---
    cfg = {**O.CONFIG, **overrides}
    O._SGDI.QUERY_COUNT = 0
    O._reset_stats()
    fm_prior = om2c_prior_cache[:4]          # (mu, cov_inv, L, thr) — FM's data_prior
    max_deg = 0; flips = 0; edges = []
    with F.QueryCounter(victim) as qc:
        for g in graphs:
            fl, ne, md = AFM.om2c_fm_greedy_flip(victim, g.to(DEVICE),
                                                 om2c_prior_cache, fm_prior, cfg, DEVICE)
            flips += int(fl); max_deg = max(max_deg, md)
            if ne is not None:
                edges.append(ne)
    asr = flips / max(len(graphs), 1)
    return {"asr": asr, "q": qc.n,
            "in_ball": O.OM2C_STATS["in_ball"], "in_ball_total": O.OM2C_STATS["total"],
            "extra": {"max_inj_degree": max_deg,
                      "median_edges_at_flip": (sorted(edges)[len(edges) // 2] if edges else None)}}


def main():
    global _BT, _BP
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--bismooth-T", type=int, default=20)
    ap.add_argument("--bismooth-p", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    _BT, _BP = args.bismooth_T, args.bismooth_p
    F.DEVICE = DEVICE; prepare.DEVICE = DEVICE

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = args.max_folds
    if args.quick:
        seeds = seeds[:1]; cap = 4; fold_limit = 1 if fold_limit is None else fold_limit

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    in_ball = {a: [0, 0] for a in ATTACKS}
    extra_acc = {a: {"max_inj_degree": 0, "edges": []} for a in ATTACKS}
    floor_by_seed = {a: {s: [] for s in seeds} for a in ATTACKS}
    floor_per_unit = {a: [] for a in ATTACKS}
    binding_count = {a: {d: 0 for d in EFFECTIVE} for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; victim_acc_by_seed = {s: [] for s in seeds}
    errored_cells = []; units = []
    meta0 = F.load_meta_for_seed(seeds[0])

    print(f"=== OM2C+FM FLOOR (missing 2x2 cell) | {meta0['dataset']} "
          f"{meta0.get('model_type','GCN')} | seeds={seeds} | cap={cap}/fold | "
          f"defenses={DEFENSES} | bismooth(T={_BT},p={_BP}) | attacks={list(ATTACKS)} | "
          f"candidate={CANDIDATE} | device={DEVICE} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = F.load_meta_for_seed(seed)
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            model.eval()
            vacc = F.clean_test_acc(model, ds, fold_info["test_idx"])
            victim_accs.append(vacc); victim_acc_by_seed[seed].append(vacc)
            graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
            ref_graphs = F.training_ref_graphs(ds, meta, fold_info)
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = build_defended_ext(model, dn, ref_graphs)
                ca = F.clean_test_acc(v, ds, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                asrs = {}
                for dn in DEFENSES:
                    victim = build_defended_ext(model, dn, ref_graphs)
                    pruned_crash = False; hard_err = None
                    try:
                        r = run_attack_cell_ext(rk, ov, victim, graphs, om2c_prior)
                    except IndexError:
                        r = {"asr": 0.0, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                        pruned_crash = True
                    except Exception as e:
                        tb = traceback.format_exc()
                        hard_err = f"{type(e).__name__}: {e}"
                        errored_cells.append({"attack": akey, "defense": dn, "seed": seed,
                                              "fold": fi, "error": hard_err,
                                              "traceback": tb.splitlines()[-8:]})
                        print(f"  !! ERROR {akey}|{dn} seed{seed} fold{fi}: {hard_err}", flush=True)
                        print(tb, flush=True)
                        r = {"asr": None, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                    if r["asr"] is not None:
                        cell_asr[akey][dn].append(r["asr"])
                    cell_q[akey][dn] += r["q"]
                    if dn == "none" and r["asr"] is not None:
                        in_ball[akey][0] += r["in_ball"]; in_ball[akey][1] += r["in_ball_total"]
                        ex = r.get("extra", {})
                        if ex.get("max_inj_degree") is not None:
                            extra_acc[akey]["max_inj_degree"] = max(extra_acc[akey]["max_inj_degree"], ex["max_inj_degree"])
                        if ex.get("median_edges_at_flip") is not None:
                            extra_acc[akey]["edges"].append(ex["median_edges_at_flip"])
                    asrs[dn] = (None if r["asr"] is None else round(r["asr"], 4))
                    unit["cells"][f"{akey}|{dn}"] = {
                        "asr": asrs[dn], "q": r["q"],
                        **({"defense_pruned_crash": True} if pruned_crash else {}),
                        **({"error": hard_err} if hard_err else {})}
                eff = [asrs[d] for d in EFFECTIVE if asrs[d] is not None]
                if len(eff) == len(EFFECTIVE):
                    f5 = min(eff)
                    floor_by_seed[akey][seed].append(f5)
                    floor_per_unit[akey].append(f5)
                    bind = min(EFFECTIVE, key=lambda d: asrs[d])
                    binding_count[akey][bind] += 1
                    unit["cells"][f"{akey}|FLOOR5"] = round(f5, 4)
                    unit["cells"][f"{akey}|BIND"] = bind
                else:
                    unit["cells"][f"{akey}|FLOOR5"] = None
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}:f5="
                           f"{(floor_by_seed[a][seed][-1]*100 if floor_by_seed[a][seed] else float('nan')):.0f}"
                           for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

    def seed_floor_summary(fbs):
        out = {}
        for a in ATTACKS:
            per = {str(s): (round(mean(fbs[a][s]) * 100, 2) if fbs[a][s] else None) for s in seeds}
            vals = [v for v in per.values() if v is not None]
            out[a] = {"per_seed_pct": per,
                      "mean_pct": round(mean(vals), 2) if vals else None,
                      "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
                      "n_seeds": len(vals)}
        return out

    sf5 = seed_floor_summary(floor_by_seed)
    asr_summary = {a: {d: (agg(cell_asr[a][d]) if cell_asr[a][d] else {"mean_pct": None, "n": 0})
                       for d in DEFENSES} for a in ATTACKS}
    binding_summary = {a: {"counts": binding_count[a],
                           "mode": (max(binding_count[a], key=lambda d: binding_count[a][d])
                                    if any(binding_count[a].values()) else None)}
                       for a in ATTACKS}

    # H_mech paired test (OM2C+FM vs control), descriptive at seed level + fold-unit p-value.
    paired = {"H_mech_candidate": CANDIDATE, "vs_control": CONTROL, "vs_anchor": ANCHOR}
    try:
        from scipy.stats import wilcoxon, ttest_rel
        for label, other in (("vs_control", CONTROL), ("vs_anchor", ANCHOR)):
            cand = floor_per_unit[CANDIDATE]; oth = floor_per_unit[other]
            nmin = min(len(cand), len(oth))
            cand, oth = cand[:nmin], oth[:nmin]
            diffs = [c - o for c, o in zip(cand, oth)]
            entry = {"mean_diff_pct": round(mean(diffs) * 100, 3) if diffs else None,
                     "n_pairs": len(diffs)}
            if diffs and any(d != 0 for d in diffs):
                try:
                    w_stat, w_p = wilcoxon(cand, oth)
                    entry["wilcoxon_p"] = float(w_p)
                except Exception as e:
                    entry["wilcoxon_p"] = f"err:{e}"
                t_stat, t_p = ttest_rel(cand, oth)
                entry["paired_t_p"] = float(t_p)
            else:
                entry["wilcoxon_p"] = "all_diffs_zero"
            paired[f"floor_{label}"] = entry
    except ImportError:
        paired["error"] = "scipy unavailable"

    out = {
        "stamp": stamp,
        "experiment": "OM2C+FM — FLOOR study for the MISSING 2x2 cell (FM feature gen + OM2C "
                      "sparse-greedy degree-1 edges). Battery {normfilter,mahalanobis,degfilter,"
                      "bismooth}; rspool dropped. Seed-level floor (n=3).",
        "hypotheses": {
            "H_mech": "OM2C+FM floor5 ~= OM2C's ~22.7% (>> FM+topk control ~8.7%) => edge "
                      "channel sources the floor; feature-gen interchangeable.",
            "H_cost": "OM2C+FM total queries << OM2C (FM feature gen cheaper than ZO feat opt)."},
        "candidate": CANDIDATE, "anchor": ANCHOR, "control": CONTROL,
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE,
        "degfilter_k": F.DEGFILTER_K, "normfilter_k": F.NORMFILTER_K,
        "bismooth_T": _BT, "bismooth_p": _BP,
        "rspool_dropped_reason": "invalid test-time readout swap on a mean-trained GCN; also "
                                 "never binds (rspool ASR high for OM2C+control in battery6).",
        "om2c_fm_build": "attack_om2c_fm.om2c_fm_greedy_flip: attack_flow velocity-ODE features "
                         "(fs=1.0) -> O._project_stealth (tangent ∩ 2σ ∩ Maha) -> OM2C sparse-"
                         "greedy degree-1 edge loop (O._candidate_pool/_build/_eval_graphs).",
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "victim_acc_by_seed": {str(s): {"mean": round(mean(victim_acc_by_seed[s]), 4),
                                        "n_folds": len(victim_acc_by_seed[s])} for s in seeds},
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "asr_summary": asr_summary,
        "seed_floor5": sf5,
        "fold_floor5_descriptive": {a: agg(floor_per_unit[a]) if floor_per_unit[a]
                                    else {"mean_pct": None, "n": 0} for a in ATTACKS},
        "binding_defense": binding_summary,
        "paired_tests": paired,
        "total_queries_by_attack": {a: sum(cell_q[a].values()) for a in ATTACKS},
        "queries_by_attack_defense": {a: {d: cell_q[a][d] for d in DEFENSES} for a in ATTACKS},
        "in_ball_fraction_by_attack": {
            a: {"in_ball": in_ball[a][0], "total": in_ball[a][1],
                "fraction": round(in_ball[a][0] / in_ball[a][1], 4) if in_ball[a][1] else None}
            for a in ATTACKS},
        "degree_info": {a: {"max_inj_degree_observed": extra_acc[a]["max_inj_degree"],
                            "median_edges_at_flip": (sorted(extra_acc[a]["edges"])[len(extra_acc[a]["edges"])//2]
                                                     if extra_acc[a]["edges"] else None)}
                        for a in ATTACKS},
        "floor_by_seed": {a: {str(s): [round(x, 4) for x in floor_by_seed[a][s]] for s in seeds}
                          for a in ATTACKS},
        "errored_cells": errored_cells,
        "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"floor_om2c_fm_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 104, flush=True)
    print(f"victim_acc: mean={out['victim_acc_mean']} min={out['victim_acc_min']}", flush=True)
    print("clean acc thru defenses:", {d: out['clean_acc_by_defense'][d]['mean'] for d in DEFENSES}, flush=True)
    print(f"\n{'attack':<22} " + " ".join(f"{d:>12}" for d in DEFENSES) +
          f" {'FLOOR5':>8} {'BIND':>11} {'totQ':>10}", flush=True)
    for a in ATTACKS:
        cells = " ".join(
            (f"{asr_summary[a][d]['mean_pct']:5.1f}±{asr_summary[a][d].get('std_pct',0):<4.1f}"
             if asr_summary[a][d]['mean_pct'] is not None else f"{'ERR':>12}") for d in DEFENSES)
        f5 = sf5[a]['mean_pct']
        f5s = f"{f5:6.1f}" if f5 is not None else "   ERR"
        bind = binding_summary[a]['mode'] or "-"
        print(f"{a:<22} {cells}  {f5s} {bind:>11} {sum(cell_q[a].values()):>10}", flush=True)
    print(f"\n--- H_mech: does OM2C+FM behave like OM2C (edge channel sources floor)? ---", flush=True)
    print(f"  OM2C+FM   floor5={sf5[CANDIDATE]['mean_pct']}%  bind={binding_summary[CANDIDATE]['mode']}  "
          f"degfilter_ASR={asr_summary[CANDIDATE]['degfilter']['mean_pct']}", flush=True)
    print(f"  OM2C-nb10 floor5={sf5[ANCHOR]['mean_pct']}%  bind={binding_summary[ANCHOR]['mode']}  "
          f"degfilter_ASR={asr_summary[ANCHOR]['degfilter']['mean_pct']}", flush=True)
    print(f"  control   floor5={sf5[CONTROL]['mean_pct']}%  bind={binding_summary[CONTROL]['mode']}  "
          f"degfilter_ASR={asr_summary[CONTROL]['degfilter']['mean_pct']}", flush=True)
    print(f"  paired (fold-unit): {paired.get('floor_vs_control')}  |  {paired.get('floor_vs_anchor')}", flush=True)
    print(f"\n--- H_cost: total queries ---", flush=True)
    for a in ATTACKS:
        print(f"  {a:<22} {sum(cell_q[a].values()):>12}", flush=True)
    print(f"\nerrored_cells: {len(errored_cells)}", flush=True)
    print(f"Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
