#!/usr/bin/env python3
"""
run_katze_vs_sgdi.py — Fair undefended comparison: GZOO-KATZE vs SGDI (+N4) across an
EDGE-BUDGET SWEEP, under IDENTICAL constraints (same victim per seed/fold, K=1 node).

Resolves the harness-fairness problem: KATZE (auto_x2) from the canonical main attack
(`attack.gzoo_graph_classification.GZOO_GraphClassification`); SGDI/N4 from the branch
harness (`attack.run_attack_sweep`) — BOTH attack the SAME victim per (seed, fold).

Edge-budget sweep (the comparison axis):
  - branch SGDI/N4: one greedy sweep per cell -> min-edges-to-flip per graph ->
    ASR@eb = frac(greedy_edges <= eb) for eb in EB_GRID; plus full-edge (asr_full).
  - KATZE: run per eb with edge_strategy='topk', edge_top_k=eb; plus edge_strategy='full'.

Methods: katze (feat_scale=auto_x2, cge) | sgdi (Langevin+Mahalanobis) | n4 (norm-cap).

Usage:
  python run_katze_vs_sgdi.py --quick                 # 1 seed,1 fold,8 graphs smoke
  python run_katze_vs_sgdi.py --dataset PROTEINS      # 5 seed x 5 fold sweep
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
# `attack` collision: /workspace/attack/ is a NAMESPACE package (no __init__); a regular
# attack.py on sys.path shadows it. Python auto-adds the script dir -> scrub it, keep only
# /workspace, and load the (self-contained) branch modules by explicit importlib path.
import importlib.util as _ilu  # noqa: E402


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The branch dir (where this script lives when copied there) must not shadow main attack.
_BRANCH = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
sys.path[:] = [p for p in sys.path
               if p not in ("", ".") and Path(p or ".").resolve() not in (HERE, _BRANCH)]
sys.path.insert(0, "/workspace")
from attack.gzoo_graph_classification import GZOO_GraphClassification  # noqa: E402

_prepare = _load("branch_prepare", _BRANCH / "prepare.py")
GCN_GraphClassification = _prepare.GCN_GraphClassification
DEVICE = _prepare.DEVICE
DATA_DIR = _prepare.DATA_DIR
sgdi_attack = _load("branch_attack", _BRANCH / "attack.py")

KATZE_DEFAULTS = {
    'gen_lr': 5e-3, 'attack_epochs': 50, 'sigma': 1e-3, 'eval_num': 500,
    'patience': 4, 'kappa': -0.1, 'feat_scale': 'auto_x2', 'grad_method': 'cge',
    'gen_hid_dim': 128,
}

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
TRAIN_EPOCHS = 100
HIDDEN = 64
N_TEST_CAP = 30
EB_GRID = [1, 3, 5, 10]            # absolute attachment-edge budgets; plus 'full'
EB_KEYS = [str(e) for e in EB_GRID] + ["full"]


def stratified_folds(labels, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(range(len(labels)), labels))
    out = []
    for fi in range(N_FOLDS):
        test_idx = list(folds[fi][1])
        val_idx = list(folds[(fi + 1) % N_FOLDS][1])
        tr = [i for i in range(len(labels)) if i not in set(test_idx) | set(val_idx)]
        out.append((fi, tr, val_idx, test_idx))
    return out


def train_victim(ds, train_idx, val_idx, seed, fold):
    torch.manual_seed(seed * 100 + fold)
    model = GCN_GraphClassification(ds.num_features, ds.num_classes, HIDDEN).to(DEVICE)
    tr = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(ds, val_idx), batch_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_acc = None, -1.0
    for _ in range(TRAIN_EPOCHS):
        model.train()
        for b in tr:
            b = b.to(DEVICE)
            loss = F.cross_entropy(model(b), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for b in va:
                b = b.to(DEVICE)
                c += (model(b).argmax(-1) == b.y).sum().item(); t += b.y.size(0)
        acc = c / max(t, 1)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, best_acc


def get_test_graphs(model, ds, test_idx, cap):
    graphs = []
    for i in test_idx:
        g = ds[i].clone().to(DEVICE)
        if g.num_nodes > N_TEST_CAP:
            continue
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if model(g).argmax(-1).item() != g.y.item():
                continue
        graphs.append(g)
        if len(graphs) >= cap:
            break
    return graphs


def run_katze_eb(model, test_graphs, n_feat, eb):
    es = 'full' if eb == 'full' else 'topk'
    etk = 64 if eb == 'full' else int(eb)
    att = GZOO_GraphClassification(
        node_budget=1, feat_dim=n_feat, discrete_feat=False,
        kappa=KATZE_DEFAULTS['kappa'], sigma=KATZE_DEFAULTS['sigma'],
        eval_num=KATZE_DEFAULTS['eval_num'], patience=KATZE_DEFAULTS['patience'],
        gen_hid_dim=KATZE_DEFAULTS['gen_hid_dim'], gen_lr=KATZE_DEFAULTS['gen_lr'],
        attack_epochs=KATZE_DEFAULTS['attack_epochs'], device=DEVICE,
        edge_strategy=es, edge_top_k=etk, dual_gradient=False,
        feat_scale=KATZE_DEFAULTS['feat_scale'], grad_method=KATZE_DEFAULTS['grad_method'],
        inject_topology='none', loss_type='cw', shared_perturbation=False)
    r = att.run_attack(model, test_graphs, num_samples=len(test_graphs))
    return float(r.get('success_rate', 0.0))


def run_branch_sweep(model, test_graphs, feat_mode):
    """One greedy sweep -> ASR@eb for each eb in EB_GRID + full-edge."""
    cfg = {"feat_mode": feat_mode, "node_budget": 1}   # greedy runs (no skip_greedy)
    o = sgdi_attack.run_attack_sweep(model, test_graphs, DEVICE, cfg)
    ge = o.get("greedy_edges", [])
    n = len(ge) or 1
    asr = {str(eb): sum(1 for e in ge if e is not None and e <= eb) / n for eb in EB_GRID}
    asr["full"] = float(o["asr_full"])
    return asr, o.get("queries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PROTEINS")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = [0] if args.quick else SEEDS
    n_folds = 1 if args.quick else N_FOLDS
    cap = 8 if args.quick else N_TEST_CAP

    ds = TUDataset(root=str(DATA_DIR / args.dataset), name=args.dataset)
    labels = [ds[i].y.item() for i in range(len(ds))]
    n_feat = ds.num_features
    print(f"[load] {args.dataset} n={len(ds)} feat={n_feat} cls={ds.num_classes} "
          f"quick={args.quick} EB={EB_KEYS}", flush=True)

    out_dir = _BRANCH / "results"; out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"katze_vs_sgdi_ebsweep_{args.dataset}_{ts}.json"

    rows, t_start = [], time.time()
    for seed in seeds:
        for (fi, tr, va, te) in stratified_folds(labels, seed)[:n_folds]:
            t0 = time.time()
            model, vacc = train_victim(ds, tr, va, seed, fi)
            tg = get_test_graphs(model, ds, te, cap)
            if not tg:
                print(f"[s{seed} f{fi}] no test graphs", flush=True); continue
            katze = {k: run_katze_eb(model, tg, n_feat, (k if k == "full" else int(k)))
                     for k in EB_KEYS}
            n4, _ = run_branch_sweep(model, tg, "zo_normcap")
            sgdi, _ = run_branch_sweep(model, tg, "sgdi")
            rows.append({"dataset": args.dataset, "seed": seed, "fold": fi,
                         "val_acc": round(vacc, 4), "n_test": len(tg),
                         "katze": katze, "n4": n4, "sgdi": sgdi,
                         "wall_s": round(time.time() - t0, 1)})
            kf = katze["full"]
            print(f"[s{seed} f{fi}] n={len(tg)} vacc={vacc:.2f} | full-edge "
                  f"K={kf:.2f} N4={n4['full']:.2f} SGDI={sgdi['full']:.2f} "
                  f"({rows[-1]['wall_s']}s)", flush=True)

            def _mean(method, k):
                return round(sum(r[method][k] for r in rows) / len(rows), 4)
            agg = {"dataset": args.dataset, "n_cells": len(rows), "eb_keys": EB_KEYS,
                   "constraints": {"K": 1, "defense": "none", "n_test_cap": cap,
                                   "seeds": seeds, "n_folds": n_folds},
                   "mean": {m: {k: _mean(m, k) for k in EB_KEYS}
                            for m in ("katze", "n4", "sgdi")},
                   "rows": rows, "wall_s": round(time.time() - t_start, 1)}
            out_path.write_text(json.dumps(agg, indent=2))

    print(f"\n=== {args.dataset} undefended — ASR vs edge budget (K=1) ===", flush=True)
    print(f"  {'method':6s} " + " ".join(f"{k:>6s}" for k in EB_KEYS), flush=True)
    for m in ("katze", "n4", "sgdi"):
        print(f"  {m:6s} " + " ".join(f"{agg['mean'][m][k]:6.3f}" for k in EB_KEYS), flush=True)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
