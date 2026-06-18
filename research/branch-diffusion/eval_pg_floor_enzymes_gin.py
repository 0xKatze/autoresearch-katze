#!/usr/bin/env python3
"""eval_pg_floor_enzymes_gin.py — SECOND-DATASET floor study: ENZYMES-continuous, GIN victim.

Mirrors the VALIDATED PROTEINS round-1 FIXED floor harness (eval_paper_grade_floor:
reference-threshold normfilter + train-fold mahalanobis + degfilter; C3 train-fold ref graphs),
re-targeted to a CONTINUOUS-feature dataset (z-scored ENZYMES node attrs, use_node_attr=True)
with the project-standard GIN victim (NOT GCN). This is the generalizability check for the
worst-case-defense FLOOR result first established on PROTEINS/GCN.

DELIBERATELY EXCLUDES GNNSafe. Round-2 critique: GNNSafe inflates ASR by pruning legitimate
nodes / degrades clean acc — flawed, excluded pending a separate diagnosis. So the defense
battery is the validated FOUR: {none, normfilter, mahalanobis, degfilter}; floor := min ASR
over the THREE EFFECTIVE ones.

Battery (== validated 5-attack set; node_budget=10; OM2C/FM attack code UNCHANGED):
  OM2C-nb10            om2c,  node_budget=10, fs=1.0  (candidate: 2-channel tangent proj)
  FM-nb10-eb1-lownorm  flow,  node_budget=10, topk edges, fs=1.0  (C1 CONTROL: plain
                       on-manifold flow WITHOUT OM2C's tangent-subspace projection)
  SGDI-nb10-full       sgdi,  node_budget=10, full-edge, fs=auto_x2
  FM-nb10-full         flow,  node_budget=10, full-edge, fs=auto_x2
  FM-nb10-eb1-highnorm flow,  node_budget=10, topk edges, fs=auto_x2

DRIVER ONLY — no attack / model / defense code modified. Reuses the FIXED harness functions
verbatim by import (F.build_defended = reference-norm + train-fold mahalanobis/degree; the
GIN victim is wrapped through the SAME DefendedModel path as GCN). Differences vs the harness:
(a) data layer = z-scored continuous ENZYMES, (b) victim = GIN, (c) train-fold ref graphs are
rebuilt from the meta's STORED train_idx (deterministic, no fold re-derivation), (d) seed-level
floor aggregation (n=3 genuine replicates: 5 folds within a seed share the dataset).

CPU-pinned (matches the round-1 validated harness, which sets CUDA_VISIBLE_DEVICES="" at import).
Victim TRAINING is a separate GPU job (prepare_cont_gin.py); the floor sweep itself runs on CPU.

Usage: micromamba run -n graph_adversarial python eval_pg_floor_enzymes_gin.py \
         [--seeds 42,7,1027] [--cap 10] [--quick] [--max-folds N]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"
DEVICE = "cpu"
from prepare import DATA_DIR  # noqa: E402
from torch_geometric.datasets import TUDataset  # noqa: E402

# Reuse the FIXED round-1 harness verbatim (CPU-pinned at import; no attack/defense code touched).
import eval_paper_grade_floor as F  # noqa: E402
import attack_om2c as O  # noqa: E402
from gin_model import GIN_GraphClassification  # noqa: E402

DEFENSES = F.DEFENSES        # {none, normfilter, mahalanobis, degfilter} — NO gnnsafe
EFFECTIVE = F.EFFECTIVE      # {normfilter, mahalanobis, degfilter}

ATTACKS = {
    "OM2C-nb10":            ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "FM-nb10-eb1-lownorm":  ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "SGDI-nb10-full":       ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-full":         ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-highnorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": "auto_x2"}),
}
CANDIDATE = "OM2C-nb10"
CONTROL = "FM-nb10-eb1-lownorm"


def load_zscored(meta):
    """Z-scored continuous ENZYMES graphs using the meta's stored per-dim feat_mean/std."""
    ds = TUDataset(root=str(DATA_DIR / f"{meta['dataset']}_attr"),
                   name=meta["dataset"], use_node_attr=True)
    m = torch.tensor(meta["feat_mean"]).float().unsqueeze(0)
    s = torch.tensor(meta["feat_std"]).float().unsqueeze(0).clamp_min(1e-6)
    return [Data(x=(ds[i].x.float() - m) / s, edge_index=ds[i].edge_index, y=ds[i].y)
            for i in range(len(ds))]


def collect_correct(model, graphs, test_idx, cap):
    out = []
    for i in test_idx:
        d = Data(x=graphs[i].x.clone(), edge_index=graphs[i].edge_index, y=graphs[i].y).to(DEVICE)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if int(model(d).argmax(-1).item()) == int(d.y.item()):
                out.append(d)
        if len(out) >= cap:
            break
    return out


def clean_acc(model, graphs, test_idx):
    c = t = 0
    for i in test_idx:
        d = Data(x=graphs[i].x.clone(), edge_index=graphs[i].edge_index, y=graphs[i].y).to(DEVICE)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            c += int(int(model(d).argmax(-1).item()) == int(d.y.item())); t += 1
    return c / max(t, 1)


def ref_graphs_from_train_idx(graphs, train_idx):
    """Held-out TRAIN-fold z-scored graphs (C3): defense stats fit off the attacked test set."""
    out = []
    for i in train_idx:
        d = Data(x=graphs[i].x.clone(), edge_index=graphs[i].edge_index, y=graphs[i].y).to(DEVICE)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
        out.append(d)
    return out


def load_meta(seed, dataset):
    p = HERE / "models_cont" / f"meta_cont_gin_{dataset}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing GIN victim meta for seed {seed}: {p} "
                                f"(run prepare_cont_gin.py first)")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--dataset", type=str, default="ENZYMES")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = args.max_folds
    if args.quick:
        seeds = seeds[:1]; cap = 4; fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    in_ball = {a: [0, 0] for a in ATTACKS}
    extra_acc = {a: {"max_inj_degree": 0, "edges": []} for a in ATTACKS}
    floor_per_unit = {a: [] for a in ATTACKS}
    floor_by_seed = {a: {s: [] for s in seeds} for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; victim_acc_by_seed = {s: [] for s in seeds}
    errored_cells = []; units = []

    meta0 = load_meta(seeds[0], args.dataset)
    graphs_all = load_zscored(meta0)   # z-score identical across seeds
    print(f"=== ENZYMES-cont FLOOR (GIN victim, 4-defense battery, NO GNNSafe) | "
          f"feat={meta0['num_features']} classes={meta0['num_classes']} | seeds={seeds} "
          f"cap={cap}/fold | defenses={DEFENSES} | attacks={list(ATTACKS)} | "
          f"candidate={CANDIDATE} control={CONTROL} | device={DEVICE} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = load_meta(seed, args.dataset)
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            model = GIN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            model.eval()
            vacc = clean_acc(model, graphs_all, fold_info["test_idx"])
            victim_accs.append(vacc); victim_acc_by_seed[seed].append(vacc)
            graphs = collect_correct(model, graphs_all, fold_info["test_idx"], cap)
            if not graphs:
                print(f"  seed{seed} fold{fi}: 0 attackable graphs, skip", flush=True); continue
            ref_graphs = ref_graphs_from_train_idx(graphs_all, fold_info["train_idx"])
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = F.build_defended(model, dn, ref_graphs)
                ca = clean_acc(v, graphs_all, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                asrs = {}
                for dn in DEFENSES:
                    victim = F.build_defended(model, dn, ref_graphs)
                    pruned_crash = False; hard_err = None
                    try:
                        r = F.run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                    except IndexError:
                        # Full-prune neutralization: pruning defense removes ALL injected nodes
                        # (full-edge nb=10 on degfilter) -> _eval_graphs returns fewer margins than
                        # anchors -> IndexError. Graph fully pruned => perturbation erased => ASR=0.
                        r = {"asr": 0.0, "q": 0, "in_ball": 0, "in_ball_total": 0, "extra": {}}
                        pruned_crash = True
                    except Exception as e:  # genuine error: record honestly, do NOT fabricate a number
                        tb = traceback.format_exc()
                        hard_err = f"{type(e).__name__}: {e}"
                        errored_cells.append({"attack": akey, "defense": dn, "seed": seed,
                                              "fold": fi, "error": hard_err,
                                              "traceback": tb.splitlines()[-6:]})
                        print(f"  !! ERROR {akey}|{dn} seed{seed} fold{fi}: {hard_err}", flush=True)
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
                    unit["cells"][f"{akey}|FLOOR"] = None
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}:fl="
                           f"{(floor_by_seed[a][seed][-1]*100 if floor_by_seed[a][seed] else float('nan')):.0f}"
                           for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

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
            "total_queries": sum(cell_q[a].values()),
            "queries_by_defense": {d: cell_q[a][d] for d in DEFENSES}}

    fold_floor_summary = {a: (agg(floor_per_unit[a]) if floor_per_unit[a] else {"mean_pct": None, "n": 0})
                          for a in ATTACKS}
    asr_summary = {a: {d: (agg(cell_asr[a][d]) if cell_asr[a][d] else {"mean_pct": None, "n": 0})
                       for d in DEFENSES} for a in ATTACKS}

    out = {
        "stamp": stamp,
        "experiment": "SECOND-DATASET floor: ENZYMES-continuous, GIN victim, 4-defense battery "
                      "(NO GNNSafe). Mirrors validated PROTEINS round-1 FIXED harness.",
        "candidate": CANDIDATE, "control_C1": CONTROL,
        "dataset": args.dataset, "victim_model": "GIN", "continuous": True,
        "gnnsafe_excluded": "round-2 critic: GNNSafe inflates ASR by pruning legit nodes / "
                            "degrades clean acc — excluded pending separate diagnosis",
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE,
        "degfilter_k": F.DEGFILTER_K, "normfilter_k": F.NORMFILTER_K,
        "fixes_applied": {
            "normfilter": "REFERENCE threshold = train-fold node-norm mean + k*std (engages multi-node)",
            "mahalanobis": "ellipsoid fit on held-out TRAIN fold (C3), not attacked test graphs",
            "ref_graphs": "rebuilt from meta STORED train_idx (deterministic)"},
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "victim_acc_caveat": "GIN ENZYMES is a known-weak victim (expect ~0.40-0.45 clean acc); "
                             "reported honestly. Floor magnitudes interpret relative to this.",
        "stat_note": "5 folds within a seed share the dataset (not independent). Genuine replicate "
                     "= per-seed mean floor => n=3 (seeds), reported descriptively.",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "victim_acc_by_seed": {str(s): {"mean": round(mean(victim_acc_by_seed[s]), 4),
                                        "std": round(pstdev(victim_acc_by_seed[s]), 4),
                                        "n_folds": len(victim_acc_by_seed[s])}
                               for s in seeds if victim_acc_by_seed[s]},
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)}
                                 for d in DEFENSES if clean_acc_def[d]},
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
    out_path = out_dir / f"floor_enzymes_gin_{args.dataset}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 96, flush=True)
    print(f"victim_acc: mean={out['victim_acc_mean']} std={out['victim_acc_std']} min={out['victim_acc_min']} "
          f"(GIN known-weak; per-seed={ {s: out['victim_acc_by_seed'].get(str(s), {}).get('mean') for s in seeds} })",
          flush=True)
    print("clean acc thru defenses:", {d: out['clean_acc_by_defense'].get(d, {}).get('mean') for d in DEFENSES}, flush=True)
    print(f"\n{'attack':<22} " + " ".join(f"{d:>13}" for d in DEFENSES) + f" {'FLOOR(fold)':>12}", flush=True)
    for a in ATTACKS:
        cells = " ".join(
            (f"{asr_summary[a][d]['mean_pct']:6.1f}±{asr_summary[a][d].get('std_pct', 0):<5.1f}"
             if asr_summary[a][d]['mean_pct'] is not None else f"{'ERR':>12}") for d in DEFENSES)
        fl = fold_floor_summary[a]
        fls = f"{fl['mean_pct']:6.1f}" if fl['mean_pct'] is not None else "  ERR"
        print(f"{a:<22} {cells}  {fls}", flush=True)
    print(f"\n--- SEED-LEVEL floor (n=3 descriptive) ---", flush=True)
    print(f"{'attack':<22} " + " ".join(f"{'seed'+str(s):>9}" for s in seeds) + f"   {'mean±std':>12} {'totQ':>10}", flush=True)
    for a in ATTACKS:
        sf = seed_floor_summary[a]; ps = sf['per_seed_pct']
        ms = f"{sf['mean_pct']}±{sf['std_pct']}" if sf['mean_pct'] is not None else "ERR"
        print(f"{a:<22} " + " ".join(f"{str(ps.get(str(s))):>9}" for s in seeds) +
              f"   {ms:>12} {sf['total_queries']:>10}", flush=True)
    print(f"\nerrored_cells: {len(errored_cells)}", flush=True)
    print(f"Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
