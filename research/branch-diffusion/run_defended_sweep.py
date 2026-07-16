#!/usr/bin/env python3
"""
run_defended_sweep.py — DEFENDED controlled sweep with node + edge budget axes.
method{none,n4,sgdi} x feat_scale{1,auto,auto_x2} x defense{none,normfilter,mahalanobis}
x node-budget K{1,3} x edge-budget eb{1,5,full from greedy}. 2 seeds x 5 folds.

Same victim per (seed,fold); all methods/defenses/scales attack the same object.

Usage: python run_defended_sweep.py --dataset PROTEINS [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean as _mn

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import GCN_GraphClassification, DEVICE, DATA_DIR  # noqa: E402
import attack as sgdi_attack  # noqa: E402
from defenses import (DefendedModel, norm_filter_defense,  # noqa: E402
                      mahalanobis_defense)

N_FOLDS = 5
SEEDS = [0, 1]
TRAIN_EPOCHS = 100
HIDDEN = 64
N_TEST_CAP = 25
METHODS = {"none": "zo_free", "n4": "zo_normcap", "sgdi": "sgdi"}
SCALES = ["1", "auto", "auto_x2"]
DEFENSES = ["none", "normfilter", "mahalanobis"]
KS = [1, 3]
EB_GRID = [1, 5]
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


def build_defended(model, defense, ref_graphs):
    if defense == "none":
        return model
    if defense == "normfilter":
        return DefendedModel(model, norm_filter_defense, "normfilter", percentile=95)
    X = torch.cat([g.x for g in ref_graphs], dim=0).to(DEVICE).float()
    mean = X.mean(dim=0); d = X.size(1)
    cov = torch.cov(X.t()) + 1e-3 * torch.eye(d, device=DEVICE)
    cov_inv = torch.linalg.inv(cov)
    diff = X - mean
    sqm = (diff @ cov_inv * diff).sum(dim=1)
    thr = float(torch.quantile(sqm, 0.995).item())
    return DefendedModel(model, mahalanobis_defense, "mahalanobis",
                         mean=mean, cov_inv=cov_inv, threshold=thr)


def asr_at_eb(victim, tg, feat_mode, K, scale):
    cfg = {"feat_mode": feat_mode, "node_budget": K, "feat_scale": scale}
    o = sgdi_attack.run_attack_sweep(victim, tg, DEVICE, cfg)
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
    defenses = ["none", "mahalanobis"] if args.quick else DEFENSES
    ks = KS

    ds = TUDataset(root=str(DATA_DIR / args.dataset), name=args.dataset)
    labels = [ds[i].y.item() for i in range(len(ds))]
    print(f"[load] {args.dataset} feat={ds.num_features} scales={scales} "
          f"defenses={defenses} K={ks} seeds={seeds} folds={n_folds} eb={EB_KEYS}", flush=True)

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"defended_nodeedge_{args.dataset}_{ts}.json"

    rows, t0 = [], time.time()
    for seed in seeds:
        for (fi, tr, va, te) in stratified_folds(labels, seed)[:n_folds]:
            model, vacc = train_victim(ds, tr, va, seed, fi)
            tg = get_test_graphs(model, ds, te, cap)
            if not tg:
                continue
            cell = {"seed": seed, "fold": fi, "val_acc": round(vacc, 4),
                    "n_test": len(tg), "res": {}}
            for dname in defenses:
                victim = build_defended(model, dname, tg)
                for mname, fmode in METHODS.items():
                    for K in ks:
                        for S in scales:
                            cell["res"][f"{dname}|{mname}|K{K}|s{S}"] = \
                                asr_at_eb(victim, tg, fmode, K, _scale_arg(S))
            rows.append(cell)
            print(f"[s{seed} f{fi}] n={len(tg)} vacc={vacc:.2f} "
                  f"cells={len(cell['res'])} ({round(time.time()-t0)}s)", flush=True)
            json.dump({"dataset": args.dataset, "n_cells": len(rows),
                       "axes": {"defenses": defenses, "methods": list(METHODS),
                                "scales": scales, "K": ks, "eb": EB_KEYS},
                       "constraints": {"n_test_cap": cap, "seeds": seeds, "n_folds": n_folds},
                       "rows": rows, "wall_s": round(time.time() - t0)},
                      open(out_path, "w"), indent=2)

    def mo(dn, m, K, s, eb):
        v = [c["res"][f"{dn}|{m}|K{K}|s{s}"][eb] for c in rows
             if f"{dn}|{m}|K{K}|s{s}" in c["res"]]
        return _mn(v) if v else 0.0
    print(f"\n=== {args.dataset} DEFENDED ASR (full-edge) — mahalanobis ===", flush=True)
    for K in ks:
        print(f"  K={K}  method " + " ".join(f"{s:>8s}" for s in scales), flush=True)
        for m in METHODS:
            print(f"        {m:6s} " + " ".join(
                f"{mo('mahalanobis',m,K,s,'full'):8.3f}" for s in scales), flush=True)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
