#!/usr/bin/env python3
"""gap1_attack_median.py — re-attack the PROPER retrained RS-Pool (median) victim.

Make-or-break: does the median-TRAINED victim BIND OM2C-nb10 (drop its ASR far below
baselines)? Loads the median victims trained by gap1_train_median.py (meta_rspool_seed{S}),
and runs OM2C-nb10 + FM-nb10-full + FM-nb10-eb1-lownorm(control) DIRECTLY against the
median classifier (the robust readout IS the defense — no extra DefendedModel wrapper).

CPU-pinned (imports eval_paper_grade_floor F, which forces CPU): matches the FM/flow
device-FP + CGE-latency rule. Reuses F.run_attack_cell / O.om2c_prior verbatim — no
attack/model/defense code modified.

Comparison reference (battery-6 mean-victim, 20260622_220259):
  OM2C-nb10 floor5=22.7%  OM2C-nb1 floor=6.3%  FM-full=0  FM-eb1-lownorm(control)=8.7%
  rspool-SWAP (invalid) inflated OM2C to 81.7%.
The proper-RS-Pool verdict: OM2C >= baselines => binding fails, novelty holds;
OM2C << baselines => median readout binds OM2C, novelty BREAKS (flag for evolution).

Usage: micromamba run -n graph_adversarial python gap1_attack_median.py \
         [--seeds 42,7,1027] [--cap 10]
"""
import os, sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import eval_paper_grade_floor as F   # forces CPU at import
import prepare
prepare.DEVICE = "cpu"
from prepare import collect_test_graphs
import attack_om2c as O
from median_gcn import MedianReadoutGCN

DEVICE = "cpu"
ATTACKS = {
    "OM2C-nb10":           ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "FM-nb10-full":        ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-lownorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
}
CANDIDATE = "OM2C-nb10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap

    from torch_geometric.datasets import TUDataset
    from prepare import DATA_DIR
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    asr_by_seed = {a: {s: [] for s in seeds} for a in ATTACKS}
    cell_q = {a: 0 for a in ATTACKS}
    victim_accs = []; victim_acc_by_seed = {s: [] for s in seeds}
    units = []; errored = []
    readout, trim = "trimmed", 0.2   # overwritten from each seed's meta

    print(f"=== GAP-1 ATTACK proper retrained RS-Pool (median) | PROTEINS GCN-median | "
          f"seeds={seeds} cap={cap} | attacks={list(ATTACKS)} | device={DEVICE} | {stamp} ===",
          flush=True)
    t0 = time.time()
    for seed in seeds:
        mp = HERE / "models_saved" / f"meta_rspool_seed{seed}.json"
        if not mp.exists():
            print(f"  !! missing {mp} — train it first (gap1_train_median.py)", flush=True)
            continue
        meta = json.loads(mp.read_text())
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if args.max_folds is None else meta["folds"][:args.max_folds]
        readout = meta.get("readout", "trimmed"); trim = meta.get("trim", 0.2)
        for fi in folds:
            model = MedianReadoutGCN(meta["num_features"], meta["num_classes"],
                                     readout=readout, trim=trim).to(DEVICE)
            model.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE,
                                             weights_only=True))
            model.eval()
            vacc = F.clean_test_acc(model, ds, fi["test_idx"])
            victim_accs.append(vacc); victim_acc_by_seed[seed].append(vacc)
            graphs = collect_test_graphs(model, ds, fi["test_idx"])[:cap]
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            unit = {"seed": seed, "fold": fi["fold_idx"], "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                try:
                    r = F.run_attack_cell(rk, ov, model, graphs, om2c_prior)
                    asr = r["asr"]; q = r["q"]
                except Exception as e:
                    errored.append({"attack": akey, "seed": seed, "fold": fi["fold_idx"],
                                    "error": f"{type(e).__name__}: {e}",
                                    "tb": traceback.format_exc().splitlines()[-5:]})
                    print(f"  !! ERROR {akey} seed{seed} fold{fi['fold_idx']}: {e}", flush=True)
                    asr = None; q = 0
                if asr is not None:
                    asr_by_seed[akey][seed].append(asr)
                cell_q[akey] += q
                unit["cells"][akey] = {"asr": (None if asr is None else round(asr, 4)), "q": q}
            units.append(unit)
            print(f"  seed{seed} fold{fi['fold_idx']} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}={ (unit['cells'][a]['asr'] if unit['cells'][a]['asr'] is not None else float('nan')) }"
                           for a in ATTACKS), flush=True)

    def seed_summary(a):
        per = {str(s): (round(mean(asr_by_seed[a][s]) * 100, 2) if asr_by_seed[a][s] else None)
               for s in seeds}
        vals = [v for v in per.values() if v is not None]
        return {"per_seed_pct": per,
                "mean_pct": round(mean(vals), 2) if vals else None,
                "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals)}

    summ = {a: seed_summary(a) for a in ATTACKS}
    out = {
        "stamp": stamp,
        "experiment": "GAP-1: attacks vs PROPER retrained median-readout (RS-Pool) victim. "
                      "Make-or-break: does training the robust readout bind OM2C-nb10?",
        "candidate": CANDIDATE, "dataset": "PROTEINS",
        "victim_model": f"GCN-robustpool (trained, readout={readout}, trim={trim})",
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "victim_acc_mean": round(mean(victim_accs), 4) if victim_accs else None,
        "victim_acc_std": round(pstdev(victim_accs), 4) if len(victim_accs) > 1 else 0.0,
        "victim_acc_by_seed": {str(s): (round(mean(victim_acc_by_seed[s]), 4)
                                        if victim_acc_by_seed[s] else None) for s in seeds},
        "asr_summary_seed_level": summ,
        "battery6_mean_victim_ref": {"OM2C-nb10_floor5": 22.7, "OM2C-nb1_floor": 6.3,
                                     "FM-full": 0.0, "FM-eb1-lownorm_control": 8.7,
                                     "rspool_SWAP_invalid_OM2C": 81.7},
        "queries_by_attack": cell_q, "errored": errored,
        "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"gap1_rspool_retrained_attack_PROTEINS_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 90, flush=True)
    print(f"median-victim clean acc: mean={out['victim_acc_mean']} by_seed={out['victim_acc_by_seed']}",
          flush=True)
    print(f"\n{'attack':<22} {'per_seed_pct':<40} {'mean_pct':>9}", flush=True)
    for a in ATTACKS:
        print(f"{a:<22} {str(summ[a]['per_seed_pct']):<40} {str(summ[a]['mean_pct']):>9}", flush=True)
    om = summ[CANDIDATE]["mean_pct"]
    ctrl = summ["FM-nb10-eb1-lownorm"]["mean_pct"]
    print(f"\n--- VERDICT (Gap-1) ---", flush=True)
    print(f"  OM2C-nb10 ASR on proper RS-Pool : {om}%", flush=True)
    print(f"  FM-eb1-lownorm control          : {ctrl}%", flush=True)
    print(f"  FM-full                         : {summ['FM-nb10-full']['mean_pct']}%", flush=True)
    if om is not None and ctrl is not None:
        if om >= ctrl:
            print(f"  => OM2C >= control: proper RS-Pool does NOT bind OM2C -> NOVELTY HOLDS (Gap-1 pass)", flush=True)
        else:
            print(f"  => OM2C < control: proper RS-Pool BINDS OM2C -> NOVELTY BREAKS (flag for evolution)", flush=True)
    print(f"\nerrored: {len(errored)}  Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
