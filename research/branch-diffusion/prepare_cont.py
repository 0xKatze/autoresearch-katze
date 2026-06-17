#!/usr/bin/env python3
"""
prepare_cont.py — Continuous-feature victim pipeline (NEW; prepare.py is DO NOT MODIFY).

Mirrors prepare.py but loads a TU graph-classification dataset WITH continuous node
attributes (use_node_attr=True) and globally z-score-normalizes node features, so the
feature manifold has genuine spread + direction + inter-dim correlation — unlike PROTEINS
binary one-hot (norm = 1.0 exactly), where norm-based stealth is degenerate (see
n4-stealth-precedent). This continuous testbed is where DISTRIBUTIONAL stealth (a diffusion
feature prior) can beat MAGNITUDE-only norm-cap (N4): N4 controls ||f|| but not direction;
distribution-aware defenses (jaccard cosine / featclip / Mahalanobis) catch off-manifold
direction even at correct norm.

Trains 5-fold GCN victims (same architecture as prepare.py), saves models_cont/ +
meta_cont_<DS>.json, and reports victim CLEAN test accuracy per fold — the GATE that must
pass (>~50%) before we build the attack on this dataset.

Usage: micromamba run -n graph_adversarial python prepare_cont.py [DATASET]   # default ENZYMES
"""
import sys
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from prepare import (
    GCN_GraphClassification, DATA_DIR, DEVICE, HIDDEN_DIM, SEED, N_FOLDS, TRAIN_EPOCHS,
)

MODELS_CONT_DIR = Path(__file__).parent / "models_cont"


def load_normalized_graphs(dataset_name, mean=None, std=None):
    """Load a TU dataset with continuous node attrs, return (graphs, labels, mean, std).

    Global per-dim z-score. If mean/std given (eval time), reuse them for identical
    normalization; else compute from the full dataset (prepare time) and return them.
    """
    ds = TUDataset(root=str(DATA_DIR / f"{dataset_name}_attr"),
                   name=dataset_name, use_node_attr=True)
    X = torch.cat([ds[i].x for i in range(len(ds))], dim=0).float()
    if mean is None or std is None:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    graphs = []
    for i in range(len(ds)):
        g = ds[i]
        xz = (g.x.float() - mean) / std
        graphs.append(Data(x=xz, edge_index=g.edge_index, y=g.y))
    labels = [int(ds[i].y.item()) for i in range(len(ds))]
    return graphs, labels, mean, std, ds.num_classes


def get_folds(labels):
    """Stratified k-fold over indices; test fold f, val fold f+1, rest train (mirror prepare)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    idx = list(range(len(labels)))
    folds = list(skf.split(idx, labels))
    for f in range(N_FOLDS):
        test_idx = list(folds[f][1])
        val_idx = list(folds[(f + 1) % N_FOLDS][1])
        tv = set(test_idx) | set(val_idx)
        train_idx = [i for i in idx if i not in tv]
        yield f, train_idx, val_idx, test_idx


def train_victim(graphs, num_features, num_classes, train_idx, val_idx, save_path):
    if save_path.exists():
        model = GCN_GraphClassification(num_features, num_classes, HIDDEN_DIM).to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        return model.eval(), None
    tr = DataLoader(Subset(graphs, train_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(graphs, val_idx), batch_size=32)
    torch.manual_seed(SEED)
    model = GCN_GraphClassification(num_features, num_classes, HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_va = 0.0
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
                c += (model(b).argmax(1) == b.y).sum().item(); t += b.y.size(0)
        if t and c / t > best_va:
            best_va = c / t
            torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    return model.eval(), best_va


def clean_test_acc(model, graphs, test_idx):
    """Fraction of test graphs correctly classified (= the attackable-set size / |test|)."""
    c = 0
    for i in test_idx:
        g = deepcopy(graphs[i]).to(DEVICE)   # deepcopy: PyG Data.to() is in-place; keep list on CPU
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if model(g).argmax(-1).item() == g.y.item():
                c += 1
    return c / max(len(test_idx), 1)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ENZYMES"
    MODELS_CONT_DIR.mkdir(parents=True, exist_ok=True)
    graphs, labels, mean, std, n_classes = load_normalized_graphs(name)
    n_feat = graphs[0].x.size(1)
    print(f"=== prepare_cont {name} | n={len(graphs)} feat={n_feat} (z-scored, continuous) "
          f"classes={n_classes} device={DEVICE} ===")

    fold_meta, accs = [], []
    for f, tr, va, te in get_folds(labels):
        sp = MODELS_CONT_DIR / f"{name}_gcn_fold{f}.pt"
        model, best_va = train_victim(graphs, n_feat, n_classes, tr, va, sp)
        acc = clean_test_acc(model, graphs, te)
        accs.append(acc)
        n_corr = int(round(acc * len(te)))
        print(f"  fold{f}: val_acc={best_va if best_va is not None else 'cached'} "
              f"clean_test_acc={acc:.4f} ({n_corr}/{len(te)} attackable)")
        fold_meta.append({"fold_idx": f, "model_path": str(sp),
                          "test_idx": [int(i) for i in te],
                          "clean_test_acc": acc, "n_attackable": n_corr})

    meta = {
        "dataset": name, "model": "gcn", "continuous": True, "use_node_attr": True,
        "zscore": True, "feat_mean": mean.squeeze(0).tolist(), "feat_std": std.squeeze(0).tolist(),
        "num_features": n_feat, "num_classes": n_classes, "seed": SEED, "n_folds": N_FOLDS,
        "device": DEVICE, "folds": fold_meta,
        "clean_test_acc_mean": float(np.mean(accs)), "clean_test_acc_std": float(np.std(accs)),
    }
    mp = MODELS_CONT_DIR / f"meta_cont_{name}.json"
    mp.write_text(json.dumps(meta, indent=2))
    print(f"\nGATE: clean_test_acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}  "
          f"({'PASS >=0.50' if np.mean(accs) >= 0.50 else 'LOW <0.50 -> consider switching dataset'})")
    print(f"Saved {mp}")


if __name__ == "__main__":
    main()
