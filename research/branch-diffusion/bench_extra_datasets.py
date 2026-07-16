#!/usr/bin/env python3
"""bench_extra_datasets.py — End-to-end pipeline to add new TUDatasets to the
GDDM clean-acc comparison: 5-fold GCN baseline → gddm_x denoiser → gddm_a
denoiser → clean-acc benchmark. Resumable: skips any step whose checkpoint
already exists.

Adaptation notes vs the PROTEINS/ENZYMES setup that the report cites:
  - Same SEED=42, N_FOLDS=5, HIDDEN_DIM=64, TRAIN_EPOCHS=100 for GCN
  - Same T=200, β=[1e-4, 0.02], hidden=64, epochs=100 for both denoisers
  - Same t*=20 (gddm_x) / t*=30 (gddm_a) at inference (no per-dataset tuning here —
    matches the comparison sheet; ENZYMES-style contamination may recur)
  - DD has large-n tail (max ~5748); we CAP test graphs to n ≤ 500 to keep
    gddm_a inference bounded (dense A is O(n²)). This is documented and applied
    consistently to no_def baseline so the comparison stays fair.

CLI:  python bench_extra_datasets.py NCI1 COX2 DD
"""
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from prepare import (DATA_DIR, DEVICE, GCN_GraphClassification,
                     HIDDEN_DIM, SEED, N_FOLDS, TRAIN_EPOCHS)
from defenses import (DefendedModel, gddm_defense, gddm_a_defense)

import train_gddm as TGX
import train_gddm_a as TGA


HERE = Path(__file__).parent
MODELS_EXTRA = HERE / "models_extra"
RESULTS = HERE / "results"
MODELS_EXTRA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

T_STAR_X = 20
T_STAR_A = 30
DD_MAX_N = 500   # see adaptation note above


# ------------------------------------------------------------
# Dataset loaders (per-dataset quirks isolated here)
# ------------------------------------------------------------

def load_tu(name):
    """Load TUDataset with discrete one-hot node features (default)."""
    root = DATA_DIR / name
    root.mkdir(parents=True, exist_ok=True)
    ds = TUDataset(root=str(root), name=name)
    return ds


def graph_size_filter(ds, idxs, max_n=None):
    if max_n is None:
        return list(idxs)
    return [i for i in idxs if ds[i].num_nodes <= max_n]


# ------------------------------------------------------------
# 5-fold GCN training
# ------------------------------------------------------------

def stratified_folds(ds):
    from sklearn.model_selection import StratifiedKFold
    labels = [ds[i].y.item() for i in range(len(ds))]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(range(len(ds)), labels))
    out = []
    for fi in range(N_FOLDS):
        test_idx = list(folds[fi][1])
        val_idx = list(folds[(fi + 1) % N_FOLDS][1])
        train_idx = [i for i in range(len(ds)) if i not in set(test_idx) | set(val_idx)]
        out.append((fi, train_idx, val_idx, test_idx))
    return out


def train_or_load_gcn(name, ds, fold_idx, train_idx, val_idx):
    import torch.nn.functional as F
    save_path = MODELS_EXTRA / f"{name}_gcn_fold{fold_idx}.pt"
    model = GCN_GraphClassification(ds.num_features, ds.num_classes, HIDDEN_DIM).to(DEVICE)
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        model.eval()
        return model, save_path
    tr_loader = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True)
    va_loader = DataLoader(Subset(ds, val_idx),   batch_size=32)
    torch.manual_seed(SEED + fold_idx)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = 0.0
    for ep in range(TRAIN_EPOCHS):
        model.train()
        for b in tr_loader:
            b = b.to(DEVICE)
            out = model(b)
            loss = F.cross_entropy(out, b.y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for b in va_loader:
                b = b.to(DEVICE)
                pred = model(b).argmax(-1)
                c += (pred == b.y).sum().item(); t += b.y.size(0)
        cur = c / max(t, 1)
        if cur > best:
            best = cur
            torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  fold {fold_idx}: best_val_acc={best:.4f}  → {save_path.name}")
    return model, save_path


# ------------------------------------------------------------
# Denoisers — re-use train_one from train_gddm{,_a}.py
# ------------------------------------------------------------

def train_or_load_gddm_x(name, ds):
    ckpt = TGX.CKPT_DIR / f"{name}.pt"
    if ckpt.exists():
        print(f"  gddm_x ckpt exists ({ckpt})")
        return ckpt
    graphs = [Data(x=ds[i].x.float(), edge_index=ds[i].edge_index, y=ds[i].y)
              for i in range(len(ds))]
    print(f"  training gddm_x denoiser on {len(graphs)} graphs (T={TGX.T_TRAIN}, ep={TGX.EPOCHS})")
    TGX.train_one(graphs, ds.num_features, ckpt)
    return ckpt


def train_or_load_gddm_a(name, ds):
    ckpt = TGA.CKPT_DIR / f"{name}.pt"
    if ckpt.exists():
        print(f"  gddm_a ckpt exists ({ckpt})")
        return ckpt
    graphs = [Data(x=ds[i].x.float(), edge_index=ds[i].edge_index, y=ds[i].y)
              for i in range(len(ds))]
    print(f"  training gddm_a denoiser on {len(graphs)} graphs (T={TGA.T_TRAIN}, ep={TGA.EPOCHS})")
    TGA.train_one(graphs, ds.num_features, ckpt)
    return ckpt


# ------------------------------------------------------------
# Clean acc eval (5-fold mean ± std, full test split unless capped)
# ------------------------------------------------------------

def _eval_full(model, graphs):
    correct = total = 0
    for g in graphs:
        g = g.to(DEVICE)
        if not hasattr(g, "batch") or g.batch is None:
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred = model(g).argmax(-1).item()
        correct += int(pred == g.y.item()); total += 1
    return (correct / max(total, 1)), total


def bench(name, ds, folds, ckpt_x, ckpt_a, max_n=None):
    rows = []
    for fi, _tr, _va, te in folds:
        te_filtered = graph_size_filter(ds, te, max_n=max_n)
        graphs = [ds[i].clone() for i in te_filtered]
        model_path = MODELS_EXTRA / f"{name}_gcn_fold{fi}.pt"
        model = GCN_GraphClassification(ds.num_features, ds.num_classes, HIDDEN_DIM).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        n_skip = len(te) - len(te_filtered)
        skip_note = f" (skipped {n_skip} graphs with n>{max_n})" if max_n else ""
        print(f"\n  --- {name} fold {fi} (n_test={len(graphs)}{skip_note}) ---", flush=True)
        t0 = time.time()
        acc_no, n = _eval_full(model, graphs)
        print(f"    no_def  acc={acc_no:.4f}  wall={time.time()-t0:.1f}s  n={n}", flush=True)

        t0 = time.time()
        dm_x = DefendedModel(model, gddm_defense, "gddm_x", ckpt_path=str(ckpt_x), t_star=T_STAR_X)
        acc_x, _ = _eval_full(dm_x, graphs)
        print(f"    gddm_x  acc={acc_x:.4f}  wall={time.time()-t0:.1f}s  Δ={(acc_x-acc_no)*100:+.2f}pp", flush=True)

        t0 = time.time()
        dm_a = DefendedModel(model, gddm_a_defense, "gddm_a", ckpt_path=str(ckpt_a), t_star=T_STAR_A)
        acc_a, _ = _eval_full(dm_a, graphs)
        print(f"    gddm_a  acc={acc_a:.4f}  wall={time.time()-t0:.1f}s  Δ={(acc_a-acc_no)*100:+.2f}pp", flush=True)

        rows.append({"dataset": name, "fold": fi, "n_test": n, "n_skipped": n_skip,
                     "acc_no": acc_no, "acc_gddm_x": acc_x, "acc_gddm_a": acc_a,
                     "drop_x_pp": (acc_x - acc_no) * 100.0,
                     "drop_a_pp": (acc_a - acc_no) * 100.0})
    return rows


def summarize(rows):
    out = {}
    for ds in sorted(set(r["dataset"] for r in rows)):
        rs = [r for r in rows if r["dataset"] == ds]
        def ms(xs):
            mu = sum(xs) / len(xs); sd = math.sqrt(sum((x - mu)**2 for x in xs) / max(len(xs) - 1, 1))
            return mu, sd
        mn, sn = ms([r["acc_no"] for r in rs])
        mx, sx = ms([r["acc_gddm_x"] for r in rs])
        ma, sa = ms([r["acc_gddm_a"] for r in rs])
        mdx, sdx = ms([r["drop_x_pp"] for r in rs])
        mda, sda = ms([r["drop_a_pp"] for r in rs])
        out[ds] = {"acc_no": (mn, sn), "acc_gddm_x": (mx, sx), "acc_gddm_a": (ma, sa),
                   "drop_x_pp": (mdx, sdx), "drop_a_pp": (mda, sda)}
    return out


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def pipeline_one(name):
    print(f"\n{'='*60}\nDATASET: {name}\n{'='*60}", flush=True)
    ds = load_tu(name)
    print(f"  loaded {name}: n_graphs={len(ds)}, feat={ds.num_features}, classes={ds.num_classes}")

    folds = stratified_folds(ds)

    # 1) GCN baseline 5-fold
    print(f"\n[1/4] Training/loading 5-fold GCN baseline")
    for fi, tr, va, te in folds:
        train_or_load_gcn(name, ds, fi, tr, va)

    # 2) gddm_x denoiser
    print(f"\n[2/4] Training/loading gddm_x denoiser")
    ckpt_x = train_or_load_gddm_x(name, ds)

    # 3) gddm_a denoiser
    print(f"\n[3/4] Training/loading gddm_a denoiser")
    ckpt_a = train_or_load_gddm_a(name, ds)

    # 4) Clean-acc benchmark
    max_n = DD_MAX_N if name == "DD" else None
    print(f"\n[4/4] Clean-acc benchmark"
          + (f" (capping test graphs to n<={max_n})" if max_n else ""))
    rows = bench(name, ds, folds, ckpt_x, ckpt_a, max_n=max_n)
    return rows


def main():
    names = sys.argv[1:] or ["NCI1", "COX2", "DD"]
    print(f"=== bench_extra_datasets — {names}  (t*={T_STAR_X}/{T_STAR_A}) ===")

    all_rows = []
    for name in names:
        rows = pipeline_one(name)
        all_rows.extend(rows)

    summary = summarize(all_rows)
    print("\n\n=== 5-FOLD MEAN ± STD ===")
    print(f"  {'dataset':<10}  {'no_def':>14}  {'gddm_x':>22}  {'gddm_a':>22}")
    for ds, s in summary.items():
        mn, sn = s["acc_no"]; mx, sx = s["acc_gddm_x"]; ma, sa = s["acc_gddm_a"]
        mdx, sdx = s["drop_x_pp"]; mda, sda = s["drop_a_pp"]
        print(f"  {ds:<10}  {mn:.4f}±{sn:.4f}  "
              f"{mx:.4f}±{sx:.4f} (Δ{mdx:+.2f}±{sdx:.2f}pp)  "
              f"{ma:.4f}±{sa:.4f} (Δ{mda:+.2f}±{sda:.2f}pp)")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"clean_acc_gddm_extra_{stamp}.json"
    out.write_text(json.dumps({"rows": all_rows,
                               "summary": {k: {kk: list(vv) for kk, vv in v.items()}
                                            for k, v in summary.items()},
                               "t_star_x": T_STAR_X, "t_star_a": T_STAR_A,
                               "dd_max_n": DD_MAX_N}, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
