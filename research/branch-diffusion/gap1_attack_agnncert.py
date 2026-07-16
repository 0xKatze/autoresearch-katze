#!/usr/bin/env python3
"""gap1_attack_agnncert.py — attack the OFFICIAL AGNNCert-E (vote) victim.

Loads subgraph-aug victims (meta_agnncert_official_seed{S}) trained by
train_agnncert_official.py, wraps each in AGNNCertVoteModel (T-subgraph division + majority
vote), and runs the 5 battery attacks DIRECTLY against the voting classifier (the vote IS
the defense). Produces the CORRECT AGNNCert column (replaces our earlier T=300 from-PDF
no-retrain version — see synthesis/impl-audit-20260624).

CPU-pinned (eval_paper_grade_floor forces CPU). Reuses F.run_attack_cell verbatim.

Usage: micromamba run -n graph_adversarial python gap1_attack_agnncert.py \
         [--seeds 42] [--cap 5] [--max-folds 3] [--T 30]
"""
import sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import eval_paper_grade_floor as F   # forces CPU
import prepare
prepare.DEVICE = "cpu"
from prepare import collect_test_graphs, GCN_GraphClassification
import attack_om2c as O
from agnncert_official import AGNNCertVoteModel

DEVICE = "cpu"
ATTACKS = {
    "OM2C-nb10":           ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "OM2C-nb1":            ("om2c", {"node_budget": 1, "feat_scale": 1.0}),
    "FM-nb10-eb1-lownorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "FM-nb10-full":        ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "SGDI-nb10-full":      ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=3)
    ap.add_argument("--T", type=int, default=30)
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

    print(f"=== ATTACK official AGNNCert-E (vote, T={args.T}) | PROTEINS | seeds={seeds} "
          f"cap={cap} max_folds={args.max_folds} | {stamp} ===", flush=True)
    t0 = time.time()
    for seed in seeds:
        mp = HERE / "models_saved" / f"meta_agnncert_official_seed{seed}.json"
        if not mp.exists():
            print(f"  !! missing {mp}", flush=True); continue
        meta = json.loads(mp.read_text())
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"][:args.max_folds]
        T = meta.get("T", args.T)
        for fi in folds:
            base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            base.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE, weights_only=True))
            model = AGNNCertVoteModel(base, meta["num_classes"], T=T).to(DEVICE).eval()
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
                    print(f"  !! ERROR {akey} s{seed} f{fi['fold_idx']}: {e}", flush=True)
                    asr = None; q = 0
                if asr is not None:
                    asr_by_seed[akey][seed].append(asr)
                cell_q[akey] += q
                unit["cells"][akey] = {"asr": (None if asr is None else round(asr, 4)), "q": q}
            units.append(unit)
            print(f"  seed{seed} fold{fi['fold_idx']} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}={unit['cells'][a]['asr']}" for a in ATTACKS), flush=True)

    def summ(a):
        per = {str(s): (round(mean(asr_by_seed[a][s]) * 100, 2) if asr_by_seed[a][s] else None) for s in seeds}
        vals = [v for v in per.values() if v is not None]
        return {"per_seed_pct": per, "mean_pct": round(mean(vals), 2) if vals else None,
                "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}
    S = {a: summ(a) for a in ATTACKS}
    out = {"stamp": stamp, "experiment": "OFFICIAL AGNNCert-E (vote, subgraph-aug) victim — correct AGNNCert column",
           "dataset": "PROTEINS", "victim_model": f"GCN-agnncert-official (T={args.T})",
           "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "max_folds": args.max_folds,
           "T": args.T, "n_units": len(units),
           "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
           "victim_acc_mean": round(mean(victim_accs), 4) if victim_accs else None,
           "victim_acc_by_seed": {str(s): (round(mean(victim_acc_by_seed[s]), 4) if victim_acc_by_seed[s] else None) for s in seeds},
           "asr_summary_seed_level": S, "queries_by_attack": cell_q, "errored": errored,
           "wall_s": round(time.time() - t0, 1), "units": units}
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"agnncert_official_attack_PROTEINS_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("=" * 90, flush=True)
    print(f"AGNNCert-official clean acc: {out['victim_acc_mean']}", flush=True)
    print(f"\n{'attack':<22} {'per_seed_pct':<32} {'mean_pct':>9}", flush=True)
    for a in ATTACKS:
        print(f"{a:<22} {str(S[a]['per_seed_pct']):<32} {str(S[a]['mean_pct']):>9}", flush=True)
    print(f"\n--- DIRECTION: does official AGNNCert (T={args.T}) kill SPARSE, spare DENSE? ---", flush=True)
    print(f"  OM2C-nb10 (sparse): {S['OM2C-nb10']['mean_pct']}%  control(sparse): {S['FM-nb10-eb1-lownorm']['mean_pct']}%  "
          f"FM-full(dense): {S['FM-nb10-full']['mean_pct']}%  SGDI(dense): {S['SGDI-nb10-full']['mean_pct']}%", flush=True)
    print(f"\nerrored: {len(errored)}  Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
