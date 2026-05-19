#!/usr/bin/env python3
"""
prepare_robust.py — Train multi-architecture victim zoo on PROTEINS.

Reuses prepare.py's dataset + fold split logic so test sets stay
identical to the canonical GCN evaluation. Trains one model per
(architecture, fold), saves to models_saved/PROTEINS_<arch>_fold<i>.pt,
writes meta_robust.json with paths + clean accuracies.

Run once:
  micromamba run -n graph_adversarial python prepare_robust.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from prepare import (
    DATA_DIR, MODELS_DIR, DEVICE,
    DATASET_NAME, SEED, N_FOLDS, TRAIN_EPOCHS, HIDDEN_DIM,
    prepare_dataset, get_folds, collect_test_graphs,
)
from victims import get_model_class, ARCHITECTURES


META_ROBUST_PATH = MODELS_DIR / "meta_robust.json"


def train_one(dataset, arch, train_idx, val_idx, fold_idx):
    save_path = MODELS_DIR / f"{DATASET_NAME}_{arch}_fold{fold_idx}.pt"
    cls = get_model_class(arch)

    if save_path.exists():
        model = cls(dataset.num_features, dataset.num_classes,
                    HIDDEN_DIM).to(DEVICE)
        model.load_state_dict(
            torch.load(save_path, map_location=DEVICE, weights_only=True))
        model.eval()
        return model, save_path, None  # accuracy unknown, model exists

    tr_loader = DataLoader(Subset(dataset, train_idx),
                           batch_size=32, shuffle=True)
    va_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)

    torch.manual_seed(SEED + fold_idx)
    model = cls(dataset.num_features, dataset.num_classes,
                HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_va = 0.0
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = F.cross_entropy(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(DEVICE)
                pred = model(batch).argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        va_acc = correct / total
        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(
        torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model, save_path, best_va


def prepare_all_robust():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ds = prepare_dataset()
    archs = list(ARCHITECTURES.keys())  # gin, gat, sage, gcn_median
    print(f"Architectures to train: {archs}")
    print(f"Folds: {N_FOLDS}, dataset: {DATASET_NAME}, "
          f"n_graphs: {len(ds)}, device: {DEVICE}")

    folds_list = list(get_folds(ds))
    arch_records = {a: [] for a in archs}
    t0 = time.time()

    for arch in archs:
        print(f"\n=== {arch} ===")
        for fold_idx, tr_idx, va_idx, te_idx in folds_list:
            ts = time.time()
            model, path, va_acc = train_one(ds, arch, tr_idx, va_idx, fold_idx)
            test_graphs = collect_test_graphs(model, ds, te_idx)
            # Compute clean accuracy on full test set
            test_loader = DataLoader(Subset(ds, te_idx), batch_size=32)
            correct = total = 0
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(DEVICE)
                    pred = model(batch).argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            te_acc = correct / total if total else 0.0
            elapsed = time.time() - ts
            va_str = f"va={va_acc:.3f}" if va_acc is not None else "va=cached"
            print(f"  Fold {fold_idx}: {va_str}, te_acc={te_acc:.3f}, "
                  f"n_correct={len(test_graphs)}/{total}  [{elapsed:.1f}s]")
            arch_records[arch].append({
                "fold_idx": fold_idx,
                "model_path": str(path),
                "test_acc": round(te_acc, 4),
                "n_test_correct": len(test_graphs),
                "n_test_total": total,
                "test_idx": [int(i) for i in te_idx],
            })

    meta = {
        "dataset": DATASET_NAME,
        "n_folds": N_FOLDS,
        "num_features": ds.num_features,
        "num_classes": ds.num_classes,
        "seed": SEED,
        "device": DEVICE,
        "architectures": arch_records,
    }
    with open(META_ROBUST_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Meta saved to {META_ROBUST_PATH}")
    return meta


if __name__ == "__main__":
    prepare_all_robust()
