#!/usr/bin/env python3
"""defense_battery6.py — two REAL defenses for the floor battery:

  RSPoolModel    : ARCHITECTURAL robust readout. Median over node embeddings instead
                   of global_mean_pool. REUSED VERBATIM from smoke_om2c.py (the already
                   smoke-verified RS-Pool proxy). Directly targets OM2C's mechanism:
                   OM2C dilutes 10 injected nodes' mass into a MEAN readout; a median
                   ignores a minority of injected outlier nodes.

  BiSmoothingModel : node-aware randomized smoothing for graph injection
                   (arXiv:2312.03979). At inference, delete each node (and its incident
                   edges) i.i.d. with prob p, repeat T times, majority-vote the class.
                   Count-aware: injected nodes are randomly removed across the T samples,
                   so a small set of injected nodes rarely survives in the voted class.

Both are model WRAPPERS (architectural / inference-time), not DefendedModel preprocessors.
They expose .forward(data) -> logits so the existing attacks query them unchanged.

No existing attack/model/defense file is modified; this is a NEW additive module.
"""
import torch
import torch.nn.functional as F
from types import SimpleNamespace


# ---- RS-Pool: robust (median) readout over the GCN's node embeddings -----------------
# Copied VERBATIM from smoke_om2c.py:RSPoolModel (the smoke-verified implementation).
# Reuses the trained conv stack + classifier; only swaps global_mean_pool -> median.
class RSPoolModel(torch.nn.Module):
    def __init__(self, base_gcn):
        super().__init__()
        self.base = base_gcn
        self.eval()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        b = self.base
        for i in range(b.num_layers):
            x = b.conv_layers[i](x, edge_index)
            x = b.batch_norms[i](x)
            x = F.relu(x)
        # robust readout: per-graph median over nodes (vs mean_pool in the base)
        n_graphs = int(batch.max().item()) + 1
        pooled = torch.stack([
            x[batch == g].median(dim=0).values if (batch == g).any()
            else torch.zeros(x.size(1), device=x.device)
            for g in range(n_graphs)
        ], dim=0)
        return b.classifier(pooled)


# ---- Node-aware Bi-smoothing (arXiv:2312.03979) --------------------------------------
class BiSmoothingModel(torch.nn.Module):
    """Randomized node-deletion smoothing for graph classification.

    forward(data) returns per-graph VOTE-FRACTION "logits" over classes (argmax = the
    randomized-smoothing majority class). T node-dropout samples are evaluated in a
    SINGLE batched base-model forward (the T*n_graphs sub-graphs are concatenated), so
    the wrapper costs ~T x the node-compute of one clean forward but only one kernel
    launch. Each sub-graph is guaranteed >=1 node (force-keep one node if a sample drops
    them all) so the pooled batch indices stay contiguous [0, T*n_graphs).
    """

    def __init__(self, base_model, T=20, p=0.2, num_classes=None, seed=None):
        super().__init__()
        self.base = base_model
        self.T = int(T)
        self.p = float(p)
        self.num_classes = num_classes
        self._gen = None
        if seed is not None:
            self._gen = torch.Generator()
            self._gen.manual_seed(int(seed))
        self.eval()

    def _rand(self, shape, device):
        g = self._gen
        if g is not None and device.type != "cpu":
            # CPU generator can't drive a CUDA sample; fall back to global RNG on GPU
            g = None
        return torch.rand(shape, device=device, generator=g)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        device = x.device
        N = x.size(0)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        n_graphs = int(batch.max().item()) + 1 if N > 0 else 1
        T = self.T

        xs, eis, bs = [], [], []
        node_offset = 0
        ei_src, ei_dst = edge_index[0], edge_index[1]
        for t in range(T):
            keep = self._rand((N,), device) >= self.p          # True = keep node
            # force-keep >=1 node per graph so every (t,g) sub-graph is non-empty
            for g in range(n_graphs):
                gmask = batch == g
                if gmask.any() and not (keep & gmask).any():
                    idx = torch.nonzero(gmask, as_tuple=False).flatten()
                    keep[idx[torch.randint(len(idx), (1,), device=device)]] = True
            node_map = torch.full((N,), -1, dtype=torch.long, device=device)
            kept_idx = torch.nonzero(keep, as_tuple=False).flatten()
            node_map[kept_idx] = torch.arange(kept_idx.numel(), device=device)
            em = keep[ei_src] & keep[ei_dst]
            new_ei = node_map[edge_index[:, em]] + node_offset
            xs.append(x[keep])
            eis.append(new_ei)
            bs.append(batch[keep] + t * n_graphs)
            node_offset += kept_idx.numel()

        big = SimpleNamespace(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(eis, dim=1) if eis else edge_index.new_zeros((2, 0)),
            batch=torch.cat(bs, dim=0),
        )
        with torch.no_grad():
            logits_all = self.base(big)                        # [T*n_graphs, C]
        C = logits_all.size(1)
        pred = logits_all.argmax(dim=1).view(T, n_graphs)      # [T, n_graphs]
        votes = torch.zeros(n_graphs, C, device=device)
        for c in range(C):
            votes[:, c] = (pred == c).sum(dim=0).float()
        return votes / float(T)                                # vote-fraction logits
