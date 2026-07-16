#!/usr/bin/env python3
"""singular_gcn.py — TRAINABLE OFFICIAL RS-Pool (Singular Pooling) GCN victim.

Ports the OFFICIAL RS-Pool readout from Ennadir et al., "Enhancing Graph Classification
Robustness with Singular Pooling", NeurIPS 2025 (github.com/king/rs-pool,
src/models/gin.py::_svd_pool) into our frozen GCN harness. This REPLACES our earlier
median/trimmed `median_gcn.MedianReadoutGCN`, which was NOT the official method (a
different mathematical object — see synthesis/impl-audit-20260624).

Official readout (per graph, node-embedding matrix X in R^{N x d}):
    v  = colsum(X) / ||colsum(X)||
    repeat num_iterations:  v <- (X^T X) v ; v <- v / ||v||      # power iteration -> v1
    sigma1 = ||X v||                                              # dominant singular value
    z = v * (sigma1 / value_alpha)                               # graph representation in R^d

Like MedianReadoutGCN this subclasses the FROZEN prepare.GCN_GraphClassification and
overrides ONLY the readout; state_dict stays structurally identical to the base GCN so the
same attacks query it unchanged. The victim MUST be trained end-to-end with this readout
(a mean-trained head cannot read a singular-vector input).
"""
import torch
import torch.nn.functional as F
from prepare import GCN_GraphClassification

DEFAULT_NUM_ITERATIONS = 5   # official default (power-iteration count)
DEFAULT_VALUE_ALPHA = 1.0    # official default; alpha is the robustness knob (paper Lemma 5.3)


def _svd_pool_one(x_graph, num_iterations, value_alpha):
    """Official _svd_pool for one graph's node-embedding matrix x_graph (N x d) -> (d,)."""
    if x_graph.numel() == 0:
        return torch.zeros(x_graph.size(1), device=x_graph.device)
    v = torch.sum(x_graph, dim=0)
    v = v / (torch.norm(v) + 1e-8)
    for _ in range(num_iterations):
        v = torch.matmul(x_graph.t(), torch.matmul(x_graph, v))
        v = v / (torch.norm(v) + 1e-8)
    singular_value = torch.norm(torch.matmul(x_graph, v))
    return v * (singular_value / value_alpha)


def _singular_pool(x, batch, n_graphs, num_iterations, value_alpha):
    out = []
    for g in range(n_graphs):
        out.append(_svd_pool_one(x[batch == g], num_iterations, value_alpha))
    return torch.stack(out, dim=0)


class SingularReadoutGCN(GCN_GraphClassification):
    """Same architecture/params as GCN_GraphClassification; OFFICIAL singular-pooling
    readout (`_svd_pool`) instead of global_mean_pool. Must be trained end-to-end."""

    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3,
                 dropout=0.5, num_iterations=DEFAULT_NUM_ITERATIONS,
                 value_alpha=DEFAULT_VALUE_ALPHA):
        super().__init__(num_features, num_classes, hidden_dim, num_layers, dropout)
        self.num_iterations = int(num_iterations)
        self.value_alpha = float(value_alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        n_graphs = int(batch.max().item()) + 1
        pooled = _singular_pool(x, batch, n_graphs, self.num_iterations, self.value_alpha)
        return self.classifier(pooled)
