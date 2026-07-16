#!/usr/bin/env python3
"""
run_feat_scale_sweep.py — Controlled 3-axis sweep: stealth-projection x node-budget x
edge-budget, ALL AT THE SAME feature-scaling range (feat_scale as the 3rd axis).

Fixes the feat_scale confound: every method gets the SAME feat_scale knob (applied
uniformly in attack._build). The projection sets the DIRECTION/SHAPE, feat_scale sets
the MAGNITUDE -> the three stealth strategies are compared at matched scaling.

Methods (same ZO optimizer, IV = stealth projection):
  none = zo_free      (unconstrained; KATZE-style: no stealth projection)
  n4   = zo_normcap   (L2-ball, SA-ZOO magnitude-only)
  sgdi = sgdi         (Mahalanobis ellipsoid; Langevin + hard projection)

Axes: feat_scale in {1, auto(=sqrt N), auto_x2(=2 sqrt N), auto_x4} x node_budget K in
{1,3} x edge budget eb in {1,5,full} (eb free from one greedy sweep). Undefended.
Same victim per (seed, fold). 3 seeds x 3 folds.

Usage: python run_feat_scale_sweep.py --dataset PROTEINS [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import GCN_GraphClassification, DEVICE, DATA_DIR  # noqa: E402
import attack as sgdi_attack  # noqa: E402  (branch harness; has the feat_scale knob)

N_FOLDS = 3
SEEDS = [0, 1, 2]
TRAIN_EPOCHS = 100
HIDDEN = 64
N_TEST_CAP = 25
METHODS = {"none": "zo_free", "n4": "zo_normcap", "sgdi": "sgdi"}
SCALES = ["1", "auto", "auto_x2", "auto_x4"]      # feat_scale axis
KS = [1, 3]                                        # node-budget axis
EB_GRID = [1, 5]                                   # edge-budget axis (+ 'full')
EB_KEYS = [str(e) for e in EB_GRID] + ["full"]


def _scale_arg(s):
    return 1.0 if s == "1" else s


def stratified_folds(labels, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(range(len(labels)), labels))
    out = []
    for fi in range(N_FOLDS):
        te = list(folds[fi][1]); va = list(folds[(fi + 1) % N_FOLDS][1])
        tr = [i for i in range(len(labels)) if i not in set(te) | set(va)]
        out.append((fi, tr, va, te))
    return out


def train_victim(ds, tr_idx, va_idx, seed, fold):
    torch.manual_seed(seed * 100 + fold)
    model = GCN_GraphClassification(ds.num_features, ds.num_classes, HIDDEN).to(DEVICE)
    tr = DataLoader(Subset(ds, tr_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(ds, va_idx), batch_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, bacc = None, -1.0
    for _ in range(TRAIN_EPOCHS):
        model.train()
        for b in tr:
            b = b.to(DEVICE)
            loss = F.cross_entropy(model(b), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); c = t = 0
        with torch.no_grad():
            for b in va:
                b = b.to(DEVICE)
                c += (model(b).argmax(-1) == b.y).sum().item(); t += b.y.size(0)
        if c / max(t, 1) > bacc:
            bacc = c / max(t, 1); best = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    return model, bacc


def get_test_graphs(model, ds, te_idx, cap):
    out = []
    for i in te_idx:
        g = ds[i].clone().to(DEVICE)
        if g.num_nodes > N_TEST_CAP:
            continue
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if model(g).argmax(-1).item() != g.y.item():
                continue
        out.append(g)
        if len(out) >= cap:
            break
    return out


def asr_at_eb(model, tg, feat_mode, K, scale):
    cfg = {"feat_mode": feat_mode, "node_budget": K, "feat_scale": scale}
    o = sgdi_attack.run_attack_sweep(model, tg, DEVICE, cfg)
    ge = o.get("greedy_edges", [])
    n = len(ge) or 1
    d = {str(eb): sum(1 for e in ge if e is not None and e <= eb) / n for eb in EB_GRID}
    d["full"] = float(o["asr_full"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PROTEINS")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [0] if args.quick else SEEDS
    n_folds = 1 if args.quick else N_FOLDS
    cap = 8 if args.quick else N_TEST_CAP
    scales = ["1", "auto_x2"] if args.quick else SCALES
    ks = [1] if args.quick else KS

    ds = TUDataset(root=str(DATA_DIR / args.dataset), name=args.dataset)
    labels = [ds[i].y.item() for i in range(len(ds))]
    print(f"[load] {args.dataset} n={len(ds)} feat={ds.num_features} "
          f"scales={scales} K={ks} EB={EB_KEYS} quick={args.quick}", flush=True)

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"featscale_sweep_{args.dataset}_{ts}.json"

    rows, t0 = [], time.time()
    for seed in seeds:
        for (fi, tr, va, te) in stratified_folds(labels, seed)[:n_folds]:
            model, vacc = train_victim(ds, tr, va, seed, fi)
            tg = get_test_graphs(model, ds, te, cap)
            if not tg:
                continue
            cell = {"seed": seed, "fold": fi, "val_acc": round(vacc, 4), "n_test": len(tg),
                    "res": {}}
            for mname, fmode in METHODS.items():
                for K in ks:
                    for S in scales:
                        key = f"{mname}|K{K}|s{S}"
                        cell["res"][key] = asr_at_eb(model, tg, fmode, K, _scale_arg(S))
            rows.append(cell)
            f_full = cell["res"].get(f"none|K1|s{scales[-1]}", {}).get("full")
            print(f"[s{seed} f{fi}] n={len(tg)} vacc={vacc:.2f} cells={len(cell['res'])} "
                  f"({round(time.time()-t0)}s elapsed)", flush=True)
            json.dump({"dataset": args.dataset, "n_cells": len(rows),
                       "axes": {"methods": list(METHODS), "scales": scales, "K": ks,
                                "eb": EB_KEYS},
                       "constraints": {"defense": "none", "n_test_cap": cap,
                                       "seeds": seeds, "n_folds": n_folds},
                       "rows": rows, "wall_s": round(time.time() - t0)},
                      open(out_path, "w"), indent=2)

    # aggregate table
    def mean_over(key, eb):
        vals = [c["res"][key][eb] for c in rows if key in c["res"]]
        return round(sum(vals) / len(vals), 3) if vals else None
    print(f"\n=== {args.dataset} ASR vs feat_scale (full-edge, K=1) ===", flush=True)
    print("  method " + " ".join(f"{s:>8s}" for s in scales), flush=True)
    for m in METHODS:
        print(f"  {m:6s} " + " ".join(
            f"{mean_over(f'{m}|K1|s{s}','full') or 0:8.3f}" for s in scales), flush=True)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
