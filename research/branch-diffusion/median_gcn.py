#!/usr/bin/env python3
"""median_gcn.py — TRAINABLE median-readout GCN victim (proper Ennadir-et-al RS-Pool).

The battery-6 RSPoolModel was a TEST-TIME median SWAP on a mean-trained classifier
(invalid: clean acc 0.72->0.65, ASR inflates). The PROPER RS-Pool is a victim TRAINED
end-to-end with the robust (median) readout, so the conv stack + classifier learn to be
discriminative UNDER median pooling. This class subclasses the FROZEN
prepare.GCN_GraphClassification and overrides ONLY the readout (global_mean_pool ->
per-graph median over node embeddings). No attack/defense/base-model code is modified.

At eval (training=False, dropout off) the forward is byte-identical to
defense_battery6.RSPoolModel.forward, so the same attacks query it unchanged.
"""
import torch
import torch.nn.functional as F
from prepare import GCN_GraphClassification


def _robust_pool(x, batch, n_graphs, readout, trim):
    """Per-graph robust readout. readout in {'median','trimmed'}.

    'median'  : per-dim median over a graph's node embeddings (RSPoolModel proxy).
    'trimmed' : per-dim symmetric trimmed mean — drop the `trim` fraction of the
                lowest AND highest node values per dim, average the rest. A minority of
                injected outlier nodes is trimmed away (the RS-Pool defense mechanism)
                while the dense average retains the count/mass signal pure median discards.
    """
    out = []
    D = x.size(1)
    for g in range(n_graphs):
        xs = x[batch == g]
        if xs.numel() == 0:
            out.append(torch.zeros(D, device=x.device)); continue
        if readout == "median":
            out.append(xs.median(dim=0).values)
        else:
            n = xs.size(0)
            k = int(n * trim)
            if 2 * k >= n:
                out.append(xs.mean(dim=0))
            else:
                xss, _ = torch.sort(xs, dim=0)
                out.append(xss[k:n - k].mean(dim=0))
    return torch.stack(out, dim=0)


class MedianReadoutGCN(GCN_GraphClassification):
    """Same architecture/params as GCN_GraphClassification; ROBUST (median/trimmed)
    readout instead of global_mean_pool.

    state_dict is structurally identical to the base GCN, so weights are
    interchangeable for loading. Dropout is kept DURING TRAINING (matches the base
    regularization). readout/trim are buffers so they round-trip through save/load.

    `readout='median'` reproduces defense_battery6.RSPoolModel exactly at eval.
    `readout='trimmed'` (default trim=0.2) is the trainable robust readout that stays
    usable on PROTEINS (pure post-ReLU median collapses the count-based class signal)."""

    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3,
                 dropout=0.5, readout="trimmed", trim=0.2):
        super().__init__(num_features, num_classes, hidden_dim, num_layers, dropout)
        self.readout = readout
        self.trim = float(trim)

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
        pooled = _robust_pool(x, batch, n_graphs, self.readout, self.trim)
        return self.classifier(pooled)
