#!/usr/bin/env python3
"""
prepare.py — Fixed setup (DO NOT MODIFY during experiments)

Downloads datasets, trains victim models, and saves them for attack experiments.
This file is run once and provides the fixed evaluation environment.
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

# ============================================================
# Constants
# ============================================================

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models_saved"
RESULTS_DIR = Path(__file__).parent / "results"
BEST_FILE = Path(__file__).parent / "best.json"

SEED = 42
N_FOLDS = 5
TRAIN_EPOCHS = 100
HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Primary benchmark: PROTEINS + GCN (fast, clear signal)
DATASET_NAME = "PROTEINS"
MODEL_TYPE = "gcn"

# ============================================================
# Model definition (frozen — same as main project)
# ============================================================

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn


class GCN_GraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


# ============================================================
# Dataset + Model preparation
# ============================================================

def prepare_dataset():
    """Download and prepare dataset with stratified folds."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = TUDataset(root=str(DATA_DIR / DATASET_NAME), name=DATASET_NAME)
    print(f"Dataset: {DATASET_NAME}, n={len(ds)}, "
          f"feat={ds.num_features}, classes={ds.num_classes}")
    return ds


def get_folds(dataset):
    """Generate stratified k-fold splits."""
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    indices = list(range(len(dataset)))
    folds = list(skf.split(indices, labels))

    for fold_idx in range(N_FOLDS):
        test_idx = list(folds[fold_idx][1])
        val_idx = list(folds[(fold_idx + 1) % N_FOLDS][1])
        test_val_set = set(test_idx) | set(val_idx)
        train_idx = [i for i in indices if i not in test_val_set]
        yield fold_idx, train_idx, val_idx, test_idx


def train_victim(dataset, train_idx, val_idx, fold_idx):
    """Train and save a victim model for one fold."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{DATASET_NAME}_{MODEL_TYPE}_fold{fold_idx}.pt"

    if save_path.exists():
        print(f"  Fold {fold_idx}: model exists, loading...")
        model = GCN_GraphClassification(
            dataset.num_features, dataset.num_classes, HIDDEN_DIM).to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        model.eval()
        return model

    from torch.utils.data import Subset
    tr_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    va_loader = DataLoader(Subset(dataset, val_idx), batch_size=32)

    torch.manual_seed(SEED + fold_idx)
    model = GCN_GraphClassification(
        dataset.num_features, dataset.num_classes, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_va = 0
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

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  Fold {fold_idx}: val_acc={best_va:.4f}, saved to {save_path}")
    return model


def collect_test_graphs(model, dataset, test_idx):
    """Return correctly classified test graphs."""
    from torch.utils.data import Subset
    graphs = []
    for data in Subset(dataset, test_idx):
        data = data.to(DEVICE)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        if pred == data.y.item():
            graphs.append(data)
    return graphs


def prepare_all():
    """Run full preparation: dataset + victim models + test graphs for all folds."""
    ds = prepare_dataset()
    fold_data = []

    for fold_idx, tr_idx, va_idx, te_idx in get_folds(ds):
        print(f"\nFold {fold_idx}/{N_FOLDS-1}")
        model = train_victim(ds, tr_idx, va_idx, fold_idx)
        test_graphs = collect_test_graphs(model, ds, te_idx)
        print(f"  Test graphs (correctly classified): {len(test_graphs)}")
        fold_data.append({
            "fold_idx": fold_idx,
            "model_path": str(MODELS_DIR / f"{DATASET_NAME}_{MODEL_TYPE}_fold{fold_idx}.pt"),
            "n_test_graphs": len(test_graphs),
            "test_idx": te_idx,
        })

    # Save fold metadata
    meta = {
        "dataset": DATASET_NAME,
        "model": MODEL_TYPE,
        "n_folds": N_FOLDS,
        "num_features": ds.num_features,
        "num_classes": ds.num_classes,
        "seed": SEED,
        "device": DEVICE,
        "folds": fold_data,
    }
    meta_path = MODELS_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved to {meta_path}")

    # Initialize best.json
    if not BEST_FILE.exists():
        with open(BEST_FILE, "w") as f:
            json.dump({"best_asr": 0.0, "best_config": {}, "history": []}, f, indent=2)
        print(f"Initialized {BEST_FILE}")

    return meta


if __name__ == "__main__":
    prepare_all()
