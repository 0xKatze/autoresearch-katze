#!/usr/bin/env python3
"""
victims.py — Multi-architecture victim model zoo.

Parallel to prepare.py — does NOT modify it. Adds alternative GNN
architectures to test the attack pipeline's transferability across
victim inductive biases.

Architectures (all 3-layer, hidden=64, dropout=0.5, global_mean_pool
unless noted):
  - gin   : GINConv aggregator (sum + MLP) — strong topology expressiveness
  - gat   : GATConv aggregator (attention) — may attend away from injection
  - sage  : SAGEConv aggregator (mean + transform) — robust to single
            high-degree neighbor
  - gcn_median : same GCN body but global_median_pool — directly defends
                 against mean-pool injection by ignoring outliers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, SAGEConv,
    global_mean_pool,
)


# ============================================================
# Architectures
# ============================================================

def _classifier_head(hidden_dim, num_classes, dropout):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
    )


class GIN_GraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64,
                 num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        in_dim = num_features
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_layers.append(GINConv(mlp))
            in_dim = hidden_dim
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = _classifier_head(hidden_dim, num_classes, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GAT_GraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64,
                 num_layers=3, dropout=0.5, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        # First layer: multi-head, output concatenated → hidden_dim total
        self.conv_layers.append(
            GATConv(num_features, hidden_dim // heads, heads=heads,
                    dropout=dropout))
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                        dropout=dropout))
        # Last layer: single head for stability
        self.conv_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = _classifier_head(hidden_dim, num_classes, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class SAGE_GraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64,
                 num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = _classifier_head(hidden_dim, num_classes, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def _global_median_pool(x, batch):
    """Per-graph median across nodes. Robust to outlier injection."""
    if batch is None:
        return x.median(dim=0, keepdim=True).values
    num_graphs = int(batch.max().item()) + 1
    out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
    for g in range(num_graphs):
        mask = batch == g
        if mask.any():
            out[g] = x[mask].median(dim=0).values
    return out


class GCN_Median_GraphClassification(nn.Module):
    """GCN with median-pool readout — defends against mean-pool injection.

    A single injected node only contributes 1/(N+1) to the mean — but
    contributes ~0 to the median when its embedding sits in the tails.
    This is a simple algorithmic defense to test if the attack still works
    when the readout is not mean.
    """
    def __init__(self, num_features, num_classes, hidden_dim=64,
                 num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = _classifier_head(hidden_dim, num_classes, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = _global_median_pool(x, batch)
        return self.classifier(x)


# ============================================================
# Factory
# ============================================================

ARCHITECTURES = {
    "gin": GIN_GraphClassification,
    "gat": GAT_GraphClassification,
    "sage": SAGE_GraphClassification,
    "gcn_median": GCN_Median_GraphClassification,
}


def get_model_class(arch):
    if arch == "gcn":
        # Re-export the canonical GCN from prepare for parity
        from prepare import GCN_GraphClassification
        return GCN_GraphClassification
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Choices: {list(ARCHITECTURES.keys()) + ['gcn']}")
    return ARCHITECTURES[arch]


if __name__ == "__main__":
    print("victims.py — available architectures:")
    for name in ["gcn"] + list(ARCHITECTURES.keys()):
        cls = get_model_class(name)
        model = cls(num_features=3, num_classes=2)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:12s}: {cls.__name__}  ({n_params:,} params)")
