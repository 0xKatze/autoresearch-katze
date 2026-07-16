#!/usr/bin/env python3
"""validate_bismooth_cpu_gpu.py — CPU/GPU divergence gate for the binding defense.

Runs the FIXED floor metric (FIX1 defended-correct denominator) + FIX2 global per-fold
seeding for OM2C-nb10 through the bismooth (binding) defense on one seed / K folds, on a
chosen device. Run once with --device cuda and once with --device cpu, then compare the
per-fold ASR to quantify the device divergence (bi-smoothing's node-deletion masks are
drawn from device-specific RNG streams, so CPU != GPU even at the same seed). Also smoke-
tests the fixes on real GCN victims.

Run:
  FLOOR_GPU=0 micromamba run -n graph_adversarial python validate_bismooth_cpu_gpu.py --device cuda --kfolds 5
  micromamba run -n graph_adversarial python validate_bismooth_cpu_gpu.py --device cpu  --kfolds 5
"""
import os, sys, json, time, argparse
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_pg_floor_battery6 as B   # GPU-enable hack runs at import
import torch
import prepare
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs
from torch_geometric.datasets import TUDataset
import attack_om2c as O

F = B.F


def run_device(device, seed, kfolds, cap, T, p):
    B._BT, B._BP = T, p
    F.DEVICE = device
    prepare.DEVICE = device
    meta = F.load_meta_for_seed(seed)
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    rows = []
    for fold_info in meta["folds"][:kfolds]:
        fi = fold_info["fold_idx"]
        fold_seed = (int(seed) * 100003 + int(fi)) % (2**31 - 1)   # FIX2 scheme
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(device)
        model.load_state_dict(torch.load(fold_info["model_path"], map_location=device))
        model.eval()
        graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
        ref = F.training_ref_graphs(ds, meta, fold_info)
        prior = O.om2c_prior(graphs, device, O.CONFIG["tangent_dim"],
                             O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
        ov = {"node_budget": 10, "feat_scale": 1.0}
        cell = {}
        for dn in ("none", "bismooth"):
            victim = B.build_defended_ext(model, dn, ref)
            r = F.run_attack_cell("om2c", ov, victim, graphs, prior)
            cell[dn] = {"asr": round(r["asr"], 4), "n_eligible": r["n_eligible"],
                        "n_total": r["n_total"], "q": r["q"]}
        rows.append({"fold": fi, **cell})
        print(f"  [{device}] fold{fi}: none asr={cell['none']['asr']} "
              f"(elig {cell['none']['n_eligible']}/{cell['none']['n_total']})  "
              f"bismooth asr={cell['bismooth']['asr']} "
              f"(elig {cell['bismooth']['n_eligible']}/{cell['bismooth']['n_total']})",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--bismooth-T", type=int, default=20)
    ap.add_argument("--bismooth-p", type=float, default=0.2)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable (no silent CPU fallback)")
    t0 = time.time()
    print(f"=== bismooth CPU/GPU divergence validation | device={args.device} | seed="
          f"{args.seed} | kfolds={args.kfolds} | cap={args.cap} | "
          f"bismooth(T={args.bismooth_T},p={args.bismooth_p}) ===", flush=True)
    rows = run_device(args.device, args.seed, args.kfolds, args.cap,
                      args.bismooth_T, args.bismooth_p)
    bism = [r["bismooth"]["asr"] for r in rows]
    none = [r["none"]["asr"] for r in rows]
    out = {"device": args.device, "seed": args.seed, "kfolds": args.kfolds, "cap": args.cap,
           "bismooth_T": args.bismooth_T, "bismooth_p": args.bismooth_p,
           "attack": "OM2C-nb10", "rows": rows,
           "mean_bismooth_asr": round(sum(bism) / len(bism), 4) if bism else None,
           "mean_none_asr": round(sum(none) / len(none), 4) if none else None,
           "wall_s": round(time.time() - t0, 1)}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    op = HERE / "results" / f"bismooth_divergence_{args.device}_{stamp}.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"mean bismooth ASR={out['mean_bismooth_asr']}  mean none ASR={out['mean_none_asr']}  "
          f"[{out['wall_s']}s]", flush=True)
    print(f"Saved {op}", flush=True)


if __name__ == "__main__":
    main()
