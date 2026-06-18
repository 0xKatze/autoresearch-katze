#!/usr/bin/env python3
"""prepare_cont_gin.py — GIN victim trainer for the ENZYMES-continuous floor study.

SECOND-DATASET generalizability test: mirror the validated PROTEINS floor harness on a
CONTINUOUS-feature dataset with the project-standard GIN backbone (NOT GCN — GCN ENZYMES
clean acc ~37% is too weak; see CLAUDE.md). Trains 5-fold GIN victims for seeds 42/7/1027
on z-scored ENZYMES node attributes (use_node_attr=True), 200 epochs each, save-best-val
checkpoint. The z-score (per-dim mean/std over the full dataset) is identical to the GCN-cont
victims (models_cont/meta_cont_ENZYMES.json) so ONLY the victim backbone differs.

For each fold the meta stores BOTH train_idx and test_idx, so the floor runner can rebuild the
held-out TRAIN-fold reference graphs (C3 fix: defense stats fit off-attack-set) deterministically
from stored indices, without re-deriving the StratifiedKFold split.

EXPECTED clean test acc ~0.40-0.45 — a KNOWN-WEAK victim, reported honestly as a caveat.

Usage: micromamba run -n graph_adversarial python prepare_cont_gin.py \
         [--seeds 42,7,1027] [--epochs 200] [--device cuda]
"""
import sys, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from prepare import HIDDEN_DIM, N_FOLDS
from prepare_cont import load_normalized_graphs, clean_test_acc
from gin_model import GIN_GraphClassification

HERE = Path(__file__).resolve().parent
MODELS_CONT_DIR = HERE / "models_cont"


def get_folds(labels, seed, n_folds):
    """Stratified k-fold (test fold f, val fold f+1, rest train), seeded per victim seed."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = list(range(len(labels)))
    folds = list(skf.split(idx, labels))
    for f in range(n_folds):
        test_idx = list(folds[f][1])
        val_idx = list(folds[(f + 1) % n_folds][1])
        tv = set(test_idx) | set(val_idx)
        train_idx = [i for i in idx if i not in tv]
        yield f, train_idx, val_idx, test_idx


def train_victim(graphs, num_features, num_classes, train_idx, val_idx, save_path,
                 seed, epochs, device):
    tr = DataLoader(Subset(graphs, train_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(graphs, val_idx), batch_size=32)
    torch.manual_seed(seed)
    model = GIN_GraphClassification(num_features, num_classes, HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_va = 0.0
    for _ in range(epochs):
        model.train()
        for b in tr:
            b = b.to(device)
            loss = F.cross_entropy(model(b), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for b in va:
                b = b.to(device)
                c += (model(b).argmax(1) == b.y).sum().item(); t += b.y.size(0)
        if t and c / t > best_va:
            best_va = c / t
            torch.save(model.state_dict(), save_path)
    if not save_path.exists():  # never improved past 0 -> still save final state
        torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model.eval(), best_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dataset", type=str, default="ENZYMES")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"!! requested device={args.device} but CUDA unavailable -> using CPU", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    MODELS_CONT_DIR.mkdir(parents=True, exist_ok=True)

    # z-score computed over the FULL dataset (seed-independent), identical to GCN-cont victims.
    graphs, labels, mean, std, n_classes = load_normalized_graphs(args.dataset)
    n_feat = graphs[0].x.size(1)
    print(f"=== prepare_cont_gin {args.dataset} | n={len(graphs)} feat={n_feat} (z-scored) "
          f"classes={n_classes} | GIN | seeds={seeds} epochs={args.epochs} device={device} ===",
          flush=True)

    for seed in seeds:
        fold_meta, accs = [], []
        for f, tr, va, te in get_folds(labels, seed, N_FOLDS):
            sp = MODELS_CONT_DIR / f"{args.dataset}_gin_seed{seed}_fold{f}.pt"
            model, best_va = train_victim(graphs, n_feat, n_classes, tr, va, sp,
                                          seed, args.epochs, device)
            acc = clean_test_acc(model, graphs, te)
            accs.append(acc)
            n_corr = int(round(acc * len(te)))
            print(f"  seed{seed} fold{f}: val_acc={best_va:.4f} clean_test_acc={acc:.4f} "
                  f"({n_corr}/{len(te)} attackable)", flush=True)
            fold_meta.append({"fold_idx": f, "model_path": str(sp),
                              "train_idx": [int(i) for i in tr],
                              "test_idx": [int(i) for i in te],
                              "clean_test_acc": acc, "n_attackable": n_corr})

        meta = {
            "dataset": args.dataset, "model": "gin", "continuous": True, "use_node_attr": True,
            "zscore": True, "feat_mean": mean.squeeze(0).tolist(), "feat_std": std.squeeze(0).tolist(),
            "num_features": n_feat, "num_classes": n_classes, "seed": seed, "n_folds": N_FOLDS,
            "epochs": args.epochs, "device": device, "folds": fold_meta,
            "clean_test_acc_mean": float(np.mean(accs)), "clean_test_acc_std": float(np.std(accs)),
        }
        mp = MODELS_CONT_DIR / f"meta_cont_gin_{args.dataset}_seed{seed}.json"
        mp.write_text(json.dumps(meta, indent=2))
        gate = "PASS >=0.40" if np.mean(accs) >= 0.40 else "WEAK <0.40 (report as caveat)"
        print(f"  -> seed{seed} clean_test_acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}  ({gate})",
              flush=True)
        print(f"  -> Saved {mp}", flush=True)


if __name__ == "__main__":
    main()
