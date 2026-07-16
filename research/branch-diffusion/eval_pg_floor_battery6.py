#!/usr/bin/env python3
"""eval_pg_floor_battery6.py — FLOOR study with the TWO REAL defenses added.

Re-runs the COMPLETE worst-case-defense floor comparison on PROTEINS/GCN with the
defense battery EXPANDED from the degfilter-era 3 axes to FIVE effective axes:
    normfilter, mahalanobis, degfilter, RS-Pool (median readout), bi-smoothing.
The make-or-break question: does the architectural RS-Pool readout BIND the OM2C
multi-node orthogonal attack (whose mechanism is mean-pool mass dilution)?

DRIVER ONLY — imports the FIXED harness (eval_paper_grade_floor: reference normfilter,
train-fold mahalanobis/degree stats, C3 train-fold ref graphs, run_attack_cell) verbatim.
The two new defenses come from defense_battery6 (RSPoolModel reused from smoke_om2c.py;
BiSmoothingModel = node-deletion randomized smoothing, arXiv:2312.03979). No attack /
model / existing-defense code modified.

CPU-pinned (matches the FM/flow-on-CPU rule: device-FP ASR divergence + CGE latency, and
sidesteps the graph_adversarial GPU CUDA segfault). Seed-level floor aggregation (n=3).

Floors reported per attack:
  floor_new3   = min over {normfilter, mahalanobis, degfilter}   (degfilter-era, for compare)
  floor_full5  = min over {normfilter, mahalanobis, degfilter, rspool, bismooth}
  binding_defense = argmin effective defense (= the defense that sets floor_full5)

Usage: micromamba run -n graph_adversarial python eval_pg_floor_battery6.py \
         [--seeds 42,7,1027] [--cap 10] [--bismooth-T 20] [--bismooth-p 0.2] [--quick]
"""
import os, sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- GPU-enable hack (verbatim approach from eval_pg_floor_battery5_gpu.py) ---------
# The CPU-pinned harness (eval_paper_grade_floor) blanks CUDA_VISIBLE_DEVICES at import,
# which caches device_count=0. We set the GPU env var, force CUDA enumeration NOW (so the
# count caches WITH the GPU visible), import prepare/harness, then restore the env var and
# flip DEVICE globals at runtime. User directive (2026-06-22): use GPU when it is free.
_GPU = os.environ.get("FLOOR_GPU", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
import torch  # noqa: E402
_CUDA_OK = torch.cuda.is_available()
if _CUDA_OK:
    _ = (torch.zeros(1, device="cuda") + 1).item()   # force context init on device 0
import prepare  # noqa: E402
import eval_paper_grade_floor as F  # noqa: E402  (re-blanks CUDA_VISIBLE_DEVICES at import)
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU            # restore after harness import
from torch_geometric.datasets import TUDataset  # noqa: E402
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs  # noqa: E402
import attack_om2c as O  # noqa: E402
from defense_battery6 import RSPoolModel, BiSmoothingModel  # noqa: E402

DEVICE = "cpu"   # set from --device in main()

# Expanded battery: 3 degfilter-era axes + RS-Pool + bi-smoothing
DEFENSES = ["none", "normfilter", "mahalanobis", "degfilter", "rspool", "bismooth"]
EFFECTIVE = ["normfilter", "mahalanobis", "degfilter", "rspool", "bismooth"]
OLD_EFFECTIVE = ["normfilter", "mahalanobis", "degfilter"]   # degfilter-era 3-axis floor

ATTACKS = {
    "OM2C-nb10":           ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "OM2C-nb1":            ("om2c", {"node_budget": 1,  "feat_scale": 1.0}),
    "FM-nb10-full":        ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-lownorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "SGDI-nb10-full":      ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
}
CANDIDATE = "OM2C-nb10"

_BT, _BP = 20, 0.2   # bi-smoothing T / p (set from CLI in main)


def build_defended_ext(model, dn, ref_graphs):
    """none/normfilter/mahalanobis/degfilter -> FIXED harness build_defended.
    rspool -> median-readout architectural wrapper; bismooth -> node-deletion smoothing."""
    if dn == "rspool":
        return RSPoolModel(model)
    if dn == "bismooth":
        return BiSmoothingModel(model, T=_BT, p=_BP)
    return F.build_defended(model, dn, ref_graphs)


def main():
    global _BT, _BP, DEVICE, DEFENSES, EFFECTIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--bismooth-T", type=int, default=20)
    ap.add_argument("--bismooth-p", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--no-rspool", action="store_true",
                    help="drop the INVALID rspool test-time architectural swap from the "
                         "battery (canonical 8-seed run: {none,normfilter,mahalanobis,"
                         "degfilter,bismooth}).")
    args = ap.parse_args()
    _BT, _BP = args.bismooth_T, args.bismooth_p
    if args.no_rspool:
        DEFENSES = [d for d in DEFENSES if d != "rspool"]
        EFFECTIVE = [d for d in EFFECTIVE if d != "rspool"]

    DEVICE = args.device
    if DEVICE.startswith("cuda") and not _CUDA_OK:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available()==False "
                           "(NO silent CPU fallback). Check CUDA_VISIBLE_DEVICES.")
    F.DEVICE = DEVICE
    prepare.DEVICE = DEVICE

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = args.max_folds
    if args.quick:
        seeds = seeds[:1]; cap = 4; fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    extra_acc = {a: {"max_inj_degree": 0, "edges": []} for a in ATTACKS}
    floor5_by_seed = {a: {s: [] for s in seeds} for a in ATTACKS}
    floor3_by_seed = {a: {s: [] for s in seeds} for a in ATTACKS}
    floor5_per_unit = {a: [] for a in ATTACKS}
    binding_count = {a: {d: 0 for d in EFFECTIVE} for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; victim_acc_by_seed = {s: [] for s in seeds}
    errored_cells = []; units = []
    meta0 = F.load_meta_for_seed(seeds[0])

    print(f"=== FLOOR BATTERY-6 (RS-Pool + bi-smoothing) | {meta0['dataset']} "
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
            # FIX2 (BS-009): bi-smoothing's per-model CPU torch.Generator silently no-ops
            # on CUDA, so node-deletion sampling falls back to the GLOBAL torch RNG. Seed
            # the global RNG deterministically per (seed,fold) BEFORE model creation /
            # bismooth sampling so the binding-defense runs are reproducible on GPU. (CPU
            # and CUDA RNG streams differ by device -> the floor min must stay single-
            # device; see the CPU/GPU bismooth divergence validation.)
            fold_seed = (int(seed) * 100003 + int(fi)) % (2**31 - 1)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)
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
                        r = F.run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                    except IndexError:
                        r = {"asr": 0.0, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                        pruned_crash = True
                    except Exception as e:
                        tb = traceback.format_exc()
                        hard_err = f"{type(e).__name__}: {e}"
                        errored_cells.append({"attack": akey, "defense": dn, "seed": seed,
                                              "fold": fi, "error": hard_err,
                                              "traceback": tb.splitlines()[-6:]})
                        print(f"  !! ERROR {akey}|{dn} seed{seed} fold{fi}: {hard_err}", flush=True)
                        print(tb, flush=True)
                        r = {"asr": None, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                    if r["asr"] is not None:
                        cell_asr[akey][dn].append(r["asr"])
                    cell_q[akey][dn] += r["q"]
                    if dn == "none" and r["asr"] is not None:
                        ex = r.get("extra", {})
                        if ex.get("max_inj_degree") is not None:
                            extra_acc[akey]["max_inj_degree"] = max(extra_acc[akey]["max_inj_degree"], ex["max_inj_degree"])
                        if ex.get("median_edges_at_flip") is not None:
                            extra_acc[akey]["edges"].append(ex["median_edges_at_flip"])
                    asrs[dn] = (None if r["asr"] is None else round(r["asr"], 4))
                    unit["cells"][f"{akey}|{dn}"] = {
                        "asr": asrs[dn], "q": r["q"],
                        "n_eligible": r.get("n_eligible"), "n_total": r.get("n_total"),
                        **({"defense_pruned_crash": True} if pruned_crash else {}),
                        **({"error": hard_err} if hard_err else {})}
                eff5 = [asrs[d] for d in EFFECTIVE if asrs[d] is not None]
                eff3 = [asrs[d] for d in OLD_EFFECTIVE if asrs[d] is not None]
                if len(eff5) == len(EFFECTIVE):
                    f5 = min(eff5)
                    floor5_by_seed[akey][seed].append(f5)
                    floor5_per_unit[akey].append(f5)
                    bind = min(EFFECTIVE, key=lambda d: asrs[d])
                    binding_count[akey][bind] += 1
                    unit["cells"][f"{akey}|FLOOR5"] = round(f5, 4)
                    unit["cells"][f"{akey}|BIND"] = bind
                else:
                    unit["cells"][f"{akey}|FLOOR5"] = None
                if len(eff3) == len(OLD_EFFECTIVE):
                    f3 = min(eff3)
                    floor3_by_seed[akey][seed].append(f3)
                    unit["cells"][f"{akey}|FLOOR3"] = round(f3, 4)
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}:f5="
                           f"{(floor5_by_seed[a][seed][-1]*100 if floor5_by_seed[a][seed] else float('nan')):.0f}"
                           for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

    def seed_floor_summary(floor_by_seed):
        out = {}
        for a in ATTACKS:
            per = {str(s): (round(mean(floor_by_seed[a][s]) * 100, 2) if floor_by_seed[a][s] else None)
                   for s in seeds}
            vals = [v for v in per.values() if v is not None]
            out[a] = {"per_seed_pct": per,
                      "mean_pct": round(mean(vals), 2) if vals else None,
                      "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
                      "n_seeds": len(vals)}
        return out

    sf5 = seed_floor_summary(floor5_by_seed)
    sf3 = seed_floor_summary(floor3_by_seed)
    asr_summary = {a: {d: (agg(cell_asr[a][d]) if cell_asr[a][d] else {"mean_pct": None, "n": 0})
                       for d in DEFENSES} for a in ATTACKS}
    binding_summary = {a: {"counts": binding_count[a],
                           "mode": (max(binding_count[a], key=lambda d: binding_count[a][d])
                                    if any(binding_count[a].values()) else None)}
                       for a in ATTACKS}

    out = {
        "stamp": stamp,
        "experiment": "FLOOR BATTERY-6: degfilter-era 3 axes + RS-Pool(median readout) + "
                      "bi-smoothing(node-deletion, arXiv:2312.03979). Tests whether the REAL "
                      "defenses bind OM2C's orthogonal multi-node attack. Seed-level floor (n=3).",
        "candidate": CANDIDATE,
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses_full5": EFFECTIVE,
        "effective_defenses_old3": OLD_EFFECTIVE,
        "degfilter_k": F.DEGFILTER_K, "normfilter_k": F.NORMFILTER_K,
        "bismooth_T": _BT, "bismooth_p": _BP,
        "no_rspool": bool(args.no_rspool),
        "asr_metric": "FIX1 defended-correct denominator (per (attack,defense) cell, "
                      "eligible = graphs the DEFENDED model classifies correctly pre-attack; "
                      "graphs the defense itself misclassifies excluded from num+denom).",
        "global_seed_scheme": "FIX2 BS-009: torch.manual_seed((seed*100003+fold)%2**31-1) "
                              "+ cuda.manual_seed_all per fold before model/bismooth sampling.",
        "rspool_impl": "median readout (smoke_om2c.RSPoolModel), test-time architectural swap "
                       "on the mean-trained GCN classifier",
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "stat_note_H1": "30 (seed,fold) units NOT independent; genuine replicate = per-seed mean "
                        "floor => n=3. Reported descriptively, no 30-sample p-value claimed.",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "victim_acc_by_seed": {str(s): {"mean": round(mean(victim_acc_by_seed[s]), 4),
                                        "n_folds": len(victim_acc_by_seed[s])} for s in seeds},
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "asr_summary": asr_summary,
        "seed_floor_full5": sf5,
        "seed_floor_old3": sf3,
        "binding_defense_full5": binding_summary,
        "fold_floor_full5_descriptive": {a: agg(floor5_per_unit[a]) if floor5_per_unit[a]
                                         else {"mean_pct": None, "n": 0} for a in ATTACKS},
        "total_queries_by_attack": {a: sum(cell_q[a].values()) for a in ATTACKS},
        "queries_by_attack_defense": {a: {d: cell_q[a][d] for d in DEFENSES} for a in ATTACKS},
        "om2c_degree_info": {a: {"max_inj_degree_observed": extra_acc[a]["max_inj_degree"],
                                 "median_edges_at_flip": (sorted(extra_acc[a]["edges"])[len(extra_acc[a]["edges"])//2]
                                                          if extra_acc[a]["edges"] else None)}
                             for a in ATTACKS if ATTACKS[a][0] == "om2c"},
        "errored_cells": errored_cells,
        "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"floor_battery6_rspool_bismooth_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 110, flush=True)
    print(f"victim_acc: mean={out['victim_acc_mean']} min={out['victim_acc_min']}", flush=True)
    print("clean acc thru defenses:", {d: out['clean_acc_by_defense'][d]['mean'] for d in DEFENSES}, flush=True)
    print(f"\n{'attack':<22} " + " ".join(f"{d:>12}" for d in DEFENSES) +
          f" {'FLOOR3':>8} {'FLOOR5':>8} {'BIND':>11}", flush=True)
    for a in ATTACKS:
        cells = " ".join(
            (f"{asr_summary[a][d]['mean_pct']:5.1f}±{asr_summary[a][d].get('std_pct',0):<4.1f}"
             if asr_summary[a][d]['mean_pct'] is not None else f"{'ERR':>12}") for d in DEFENSES)
        f3 = sf3[a]['mean_pct']; f5 = sf5[a]['mean_pct']
        f3s = f"{f3:6.1f}" if f3 is not None else "   ERR"
        f5s = f"{f5:6.1f}" if f5 is not None else "   ERR"
        bind = binding_summary[a]['mode'] or "-"
        print(f"{a:<22} {cells}  {f3s} {f5s} {bind:>11}", flush=True)
    print(f"\n--- OM2C-nb10 DECISIVE ---", flush=True)
    print(f"  RS-Pool ASR        : {asr_summary[CANDIDATE]['rspool']}", flush=True)
    print(f"  bismooth ASR       : {asr_summary[CANDIDATE]['bismooth']}", flush=True)
    print(f"  OLD floor (3-axis) : {sf3[CANDIDATE]['mean_pct']}%  (degfilter-era, cf 52.7)", flush=True)
    print(f"  NEW floor (5-axis) : {sf5[CANDIDATE]['mean_pct']}%", flush=True)
    print(f"  binding defense    : {binding_summary[CANDIDATE]}", flush=True)
    print(f"\nerrored_cells: {len(errored_cells)}", flush=True)
    print(f"Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
