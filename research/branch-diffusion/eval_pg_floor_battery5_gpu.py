#!/usr/bin/env python3
"""eval_pg_floor_battery5_gpu.py — GPU re-run of the worst-case-defense FLOOR study
WITH the just-applied fixes (reference normfilter + train-stat mahalanobis + the new
C1 control FM-nb10-eb1-lownorm), over the EXACT requested 5-attack battery.

DRIVER ONLY — imports the FIXED harness functions verbatim from eval_paper_grade_floor
(build_defended = reference-norm + train-fold mahalanobis/degree stats; run_attack_cell;
training_ref_graphs = held-out train-fold C3 fix). No attack / model / defense code is
modified here; this file only (a) assembles the requested attack set, (b) enables CUDA,
and (c) aggregates the floor at the SEED level (H1 stats fix: the 30 fold-units are NOT
independent — 10 folds within a seed share the dataset, so the genuine replicate is the
per-seed mean floor => n=3).

Battery (== user request):
  OM2C-nb10            om2c,  node_budget=10, fs=1.0  (candidate: 2-channel tangent proj)
  FM-nb10-eb1-lownorm  flow,  node_budget=10, topk edges, fs=1.0 (C1 CONTROL: plain
                       on-manifold-norm flow WITHOUT OM2C's tangent-subspace projection)
  SGDI-nb10-full       sgdi,  node_budget=10, full-edge, fs=auto_x2
  FM-nb10-full         flow,  node_budget=10, full-edge, fs=auto_x2
  FM-nb10-eb1-highnorm flow,  node_budget=10, topk edges, fs=auto_x2
Defenses: none, normfilter(fixed ref), mahalanobis(fixed train-stat), degfilter.
floor(attack) := min ASR over EFFECTIVE {normfilter, mahalanobis, degfilter}.

Usage: micromamba run -n graph_adversarial python eval_pg_floor_battery5_gpu.py \
         [--seeds 42,7,1027] [--cap 10] [--device cuda] [--quick]
"""
import os, sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- pin CUDA BEFORE importing the CPU-pinned harness ---------------------------------
# eval_paper_grade_floor (and prepare) call torch.cuda.is_available() at IMPORT time while
# eval_paper_grade_floor has just set CUDA_VISIBLE_DEVICES="" — that caches device_count=0
# and permanently disables CUDA for the process. We therefore (1) set the GPU env var, (2)
# import torch and force CUDA enumeration NOW so the count is cached WITH the GPU visible,
# (3) import prepare ourselves so its is_available() check sees the GPU, and only THEN
# import the harness. The harness re-blanks the env var, but CUDA is already enumerated, so
# we restore the env var afterwards and flip the module DEVICE globals at runtime.
_GPU = os.environ.get("FLOOR_GPU", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
import torch  # noqa: E402
_CUDA_OK = torch.cuda.is_available()          # caches device_count WITH gpu visible
if _CUDA_OK:
    _ = (torch.zeros(1, device="cuda") + 1).item()  # force context init on device 0
import prepare  # noqa: E402  (its module-level is_available() now sees the GPU)
import eval_paper_grade_floor as F  # noqa: E402  (re-blanks CUDA_VISIBLE_DEVICES)
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU    # restore after harness import
from torch_geometric.datasets import TUDataset  # noqa: E402
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs  # noqa: E402
import attack_om2c as O  # noqa: E402
import numpy as np  # noqa: E402
from defenses import (gnnsafe_defense, DefendedModel,  # noqa: E402
                      _gnnsafe_node_logits, _energy_propagation)

# H3: GNNSafe (energy-based OOD) added as a 5th defense and counted as EFFECTIVE
# (it is a genuine worst-case-defense axis, like normfilter/mahalanobis/degfilter).
# Its energy threshold is calibrated from the TRAINING fold's clean-node energy
# (5th-percentile of propagated neg_energy), independent of the attacked test
# graphs — the SAME train-fold principle as the C3 mahalanobis/norm/degree fix.
GNNSAFE_T, GNNSAFE_K, GNNSAFE_ALPHA = 1.0, 2, 0.5   # == calibrate_gnnsafe.py / official repo
GNNSAFE_PCT_CUT_LOW = 5.0                           # keep 95% of clean nodes
DEFENSES = F.DEFENSES + ["gnnsafe"]
EFFECTIVE = F.EFFECTIVE + ["gnnsafe"]


def calibrate_gnnsafe_threshold(model, ref_graphs):
    """5th-percentile of propagated clean-node neg_energy over the TRAIN-fold graphs.

    Mirrors calibrate_gnnsafe.py verbatim (T=1, K=2, alpha=0.5; node kept iff
    neg_energy >= thr). ref_graphs are the held-out training-fold graphs, so the
    threshold never peeks at the attacked test graphs (C3 train-fold principle)."""
    chunks = []
    for g in ref_graphs:
        with torch.no_grad():
            logits = _gnnsafe_node_logits(model, g)
            e = GNNSAFE_T * torch.logsumexp(logits / GNNSAFE_T, dim=-1)
            e = _energy_propagation(e, g.edge_index, g.num_nodes,
                                    K=GNNSAFE_K, alpha=GNNSAFE_ALPHA)
        chunks.append(e.detach().cpu().numpy())
    energies = np.concatenate(chunks) if chunks else np.array([])
    if energies.size == 0:
        raise RuntimeError("gnnsafe calibration: empty train-fold ref set")
    return float(np.percentile(energies, GNNSAFE_PCT_CUT_LOW))


def build_defended_ext(model, dn, ref_graphs, gnnsafe_thr):
    """Wrap `model` with defense `dn`. Delegates to the FIXED harness build_defended
    for the original 4 defenses; handles the new 'gnnsafe' axis locally with a
    train-fold-calibrated energy threshold (no harness/attack code modified)."""
    if dn == "gnnsafe":
        return DefendedModel(model, gnnsafe_defense, "gnnsafe",
                             victim_model=model, threshold=gnnsafe_thr,
                             T=GNNSAFE_T, K=GNNSAFE_K, alpha=GNNSAFE_ALPHA)
    return F.build_defended(model, dn, ref_graphs)

ATTACKS = {
    "OM2C-nb10":            ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "FM-nb10-eb1-lownorm":  ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "SGDI-nb10-full":       ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-full":         ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-highnorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": "auto_x2"}),
}
CANDIDATE = "OM2C-nb10"
CONTROL = "FM-nb10-eb1-lownorm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None,
                    help="cap folds per seed (smoke/engagement runs); default = all folds")
    args = ap.parse_args()

    # --- enable GPU on the CPU-pinned harness (undo import-time blanking) ---
    dev = args.device
    if dev.startswith("cuda"):
        if not _CUDA_OK:
            raise RuntimeError("--device cuda requested but torch.cuda.is_available()==False "
                               "(NO silent CPU fallback per instruction). Check CUDA_VISIBLE_DEVICES.")
    F.DEVICE = dev
    prepare.DEVICE = dev
    DEVICE = dev

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = args.max_folds
    if args.quick:
        seeds = seeds[:1]; cap = 4; fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    in_ball = {a: [0, 0] for a in ATTACKS}
    extra_acc = {a: {"max_inj_degree": 0, "edges": []} for a in ATTACKS}
    floor_per_unit = {a: [] for a in ATTACKS}                       # 30 (seed,fold) floors
    floor_by_seed = {a: {s: [] for s in seeds} for a in ATTACKS}    # H1: group by seed
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; victim_acc_by_seed = {s: [] for s in seeds}
    errored_cells = []
    units = []
    meta0 = F.load_meta_for_seed(seeds[0])

    print(f"=== FLOOR BATTERY-5 (GPU, fixed normfilter+mahalanobis +C1) | {meta0['dataset']} "
          f"{meta0.get('model_type','GCN')} | seeds={seeds} | cap={cap}/fold | "
          f"defenses={DEFENSES} | attacks={list(ATTACKS)} | candidate={CANDIDATE} | "
          f"control={CONTROL} | device={DEVICE} "
          f"({torch.cuda.get_device_name(0) if DEVICE.startswith('cuda') else 'cpu'}) | "
          f"{stamp} ===", flush=True)
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
            # C3 fix: defense stats fit on held-out TRAIN fold, not the attacked test graphs.
            ref_graphs = F.training_ref_graphs(ds, meta, fold_info)
            # H3: calibrate GNNSafe energy threshold ONCE per fold on the same train fold.
            gnnsafe_thr = calibrate_gnnsafe_threshold(model, ref_graphs)
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = build_defended_ext(model, dn, ref_graphs, gnnsafe_thr)
                ca = F.clean_test_acc(v, ds, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                asrs = {}
                for dn in DEFENSES:
                    victim = build_defended_ext(model, dn, ref_graphs, gnnsafe_thr)
                    pruned_crash = False; hard_err = None
                    try:
                        r = F.run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                    except IndexError:
                        # Documented full-prune neutralization: a pruning defense removes
                        # ALL injected nodes (full-edge nb=10 on degfilter), collapsing the
                        # anchor batch so _eval_graphs returns fewer margins than anchors ->
                        # IndexError. Graph fully pruned => perturbation erased => ASR=0.
                        r = {"asr": 0.0, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                        pruned_crash = True
                    except Exception as e:  # genuine error: record honestly, do NOT fake a number
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
                    floor = min(eff)
                    floor_per_unit[akey].append(floor)
                    floor_by_seed[akey][seed].append(floor)
                    unit["cells"][f"{akey}|FLOOR"] = round(floor, 4)
                else:
                    unit["cells"][f"{akey}|FLOOR"] = None  # an effective-defense cell errored
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}:fl="
                           f"{(floor_by_seed[a][seed][-1]*100 if floor_by_seed[a][seed] else float('nan')):.0f}"
                           for a in ATTACKS), flush=True)

    # ---- aggregation ----
    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

    # SEED-LEVEL floor: mean floor over each seed's folds => n=3 genuine replicates.
    seed_floor = {a: {s: (round(mean(floor_by_seed[a][s]) * 100, 2) if floor_by_seed[a][s] else None)
                      for s in seeds} for a in ATTACKS}
    seed_floor_summary = {}
    for a in ATTACKS:
        vals = [seed_floor[a][s] for s in seeds if seed_floor[a][s] is not None]
        seed_floor_summary[a] = {
            "per_seed_pct": {str(s): seed_floor[a][s] for s in seeds},
            "mean_pct": round(mean(vals), 2) if vals else None,
            "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            # H4: per-attack query budget reported ALONGSIDE the floor (total across all
            # defense cells, and the per-defense breakdown so 'none' vs defended is visible).
            "total_queries": sum(cell_q[a].values()),
            "queries_by_defense": {d: cell_q[a][d] for d in DEFENSES}}

    fold_floor_summary = {a: agg(floor_per_unit[a]) if floor_per_unit[a] else {"mean_pct": None, "n": 0}
                          for a in ATTACKS}
    asr_summary = {a: {d: (agg(cell_asr[a][d]) if cell_asr[a][d] else {"mean_pct": None, "n": 0})
                       for d in DEFENSES} for a in ATTACKS}

    out = {
        "stamp": stamp,
        "experiment": "FLOOR BATTERY-5 GPU re-run: fixed normfilter(reference)+mahalanobis(train-stat) "
                      "+ C1 control FM-nb10-eb1-lownorm. SEED-LEVEL floor aggregation (n=3).",
        "candidate": CANDIDATE, "control_C1": CONTROL,
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE,
        "gpu_name": (torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else None),
        "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE, "degfilter_k": F.DEGFILTER_K,
        "normfilter_k": F.NORMFILTER_K,
        "fixes_applied": {
            "normfilter": "REFERENCE threshold = train-fold node-norm mean + k*std (engages multi-node)",
            "mahalanobis": "ellipsoid fit on held-out TRAIN fold (C3), not attacked test graphs",
            "C1_control": "FM-nb10-eb1-lownorm = plain on-manifold flow WITHOUT OM2C tangent projection"},
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "stat_note_H1": "30 (seed,fold) units are NOT independent (10 folds within a seed share "
                        "the dataset). Genuine replicate = per-seed mean floor => n=3. We report "
                        "the 3 seed-floors per attack DESCRIPTIVELY and do NOT claim a 30-sample p-value.",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "victim_acc_by_seed": {str(s): {"mean": round(mean(victim_acc_by_seed[s]), 4),
                                        "std": round(pstdev(victim_acc_by_seed[s]), 4),
                                        "n_folds": len(victim_acc_by_seed[s])} for s in seeds},
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "seed_floor_summary": seed_floor_summary,
        "fold_floor_summary_descriptive_only": fold_floor_summary,
        "asr_summary": asr_summary,
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
        "floor_per_unit": {a: [round(x, 4) for x in floor_per_unit[a]] for a in ATTACKS},
        "floor_by_seed": {a: {str(s): [round(x, 4) for x in floor_by_seed[a][s]] for s in seeds}
                          for a in ATTACKS},
        "errored_cells": errored_cells,
        "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"floor_battery5_gpu_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 96, flush=True)
    print(f"victim_acc: mean={out['victim_acc_mean']} std={out['victim_acc_std']} "
          f"min={out['victim_acc_min']}  per-seed={ {s: out['victim_acc_by_seed'][str(s)]['mean'] for s in seeds} }",
          flush=True)
    print("clean acc thru defenses:", {d: out['clean_acc_by_defense'][d]['mean'] for d in DEFENSES}, flush=True)
    print(f"\n{'attack':<22} " + " ".join(f"{d:>13}" for d in DEFENSES) + f" {'FLOOR(fold)':>12}", flush=True)
    for a in ATTACKS:
        cells = " ".join(
            (f"{asr_summary[a][d]['mean_pct']:6.1f}±{asr_summary[a][d].get('std_pct',0):<5.1f}"
             if asr_summary[a][d]['mean_pct'] is not None else f"{'ERR':>12}") for d in DEFENSES)
        fl = fold_floor_summary[a]
        fls = f"{fl['mean_pct']:6.1f}" if fl['mean_pct'] is not None else "  ERR"
        print(f"{a:<22} {cells}  {fls}", flush=True)
    print(f"\n--- SEED-LEVEL floor (n=3 genuine replicates, descriptive) ---", flush=True)
    print(f"{'attack':<22} {'seed42':>8} {'seed7':>8} {'seed1027':>9}   {'mean±std':>12} {'totQ':>10}", flush=True)
    for a in ATTACKS:
        sf = seed_floor_summary[a]; ps = sf['per_seed_pct']
        ms = f"{sf['mean_pct']}±{sf['std_pct']}" if sf['mean_pct'] is not None else "ERR"
        print(f"{a:<22} {str(ps.get('42')):>8} {str(ps.get('7')):>8} {str(ps.get('1027')):>9}   "
              f"{ms:>12} {sf['total_queries']:>10}", flush=True)
    print(f"\nerrored_cells: {len(errored_cells)}", flush=True)
    print(f"queries: " + " ".join(f"{a}={sum(cell_q[a].values())}" for a in ATTACKS), flush=True)
    print(f"\nSaved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
