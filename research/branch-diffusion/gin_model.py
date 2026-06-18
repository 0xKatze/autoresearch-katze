#!/usr/bin/env python3
"""gin_model.py — GIN victim for the ENZYMES-continuous floor study (SECOND-DATASET test).

Project convention (CLAUDE.md): ENZYMES victims use GIN, NOT GCN — GCN on ENZYMES sits at
~37% clean acc (too weak); GIN is the standard backbone for this dataset. This module adds a
GIN graph-classification victim with the SAME forward(data) interface as
prepare.GCN_GraphClassification, so the frozen floor harness (build_defended / run_attack_cell
/ training_ref_graphs in eval_paper_grade_floor) wraps and queries it unchanged.

Architecture (mirrors the GCN victim's capacity: 3 layers, hidden 64, BatchNorm, dropout 0.5):
  3x [ GINConv(MLP: lin->BN->ReLU->lin) -> BN -> ReLU -> dropout ] -> global_mean_pool -> MLP head.

prepare.py is DO-NOT-MODIFY, so the GIN class lives here and is imported by both the victim
trainer (prepare_cont_gin.py) and the floor runner (eval_pg_floor_enzymes_gin.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GIN_GraphClassification(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        for li in range(num_layers):
            in_dim = num_features if li == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_layers.append(GINConv(mlp, train_eps=True))
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
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
