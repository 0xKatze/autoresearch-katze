#!/usr/bin/env python3
"""agnncert_official.py — OFFICIAL AGNNCert-E (edge-division) for graph classification.

Faithful port of the official repo (JetRichardLee/AGNNCert, edge_hash.py: HashAgent +
RobustGraphClassifier), REPLACING our earlier defense_agnncert.py which used T=300, kept
ALL nodes per subgraph, and did NOT retrain (see synthesis/impl-audit-20260624).

Official AGNNCert-E (Li & Wang, USENIX Sec 2025), graph classification:
  divide:  hash each edge (u,v) -> i = md5(hex(V*min+max)) % T  -> T subgraphs, each keeping
           ONLY the non-isolated nodes touched by its edges (index-remapped), PLUS a single
           zero-feature padding node at index 0 (official quirk). T=30 for graph-cls.
  train:   subgraph augmentation — train the base GNN on {T subgraphs + original} per train
           graph (RobustGraphClassifier.enlarge_dataset).
  vote:    classify each of a test graph's T subgraphs, majority-vote the class.

This module provides HashAgentOfficial (division) + AGNNCertVoteModel (a query-able wrapper
exposing .forward(data)->vote-fraction logits, argmax = AGNNCert majority class) so the
existing black-box attacks query it UNCHANGED. The wrapper uses OUR GCN arch as the base
(fair vs the battery), trained with the official subgraph-augmentation protocol.
"""
import hashlib
import numpy as np
import torch
from torch_geometric.data import Data


class HashAgentOfficial:
    def __init__(self, h="md5", T=30):
        self.h = h
        self.T = int(T)

    def hash_edge(self, V, u, v):
        hexstring = hex(int(V) * int(u) + int(v)).encode()
        hd = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256}[self.h]()
        hd.update(hexstring)
        return int(hd.hexdigest(), 16) % self.T

    def generate_graph_subgraphs(self, edge_index, x, y):
        """Faithful port: T subgraphs, each = zero pad-node(idx0) + remapped non-isolated
        nodes touched by its hashed edges. Returns list of T PyG Data (edge_index None if empty)."""
        T = self.T
        V = x.shape[0]
        zerox = torch.zeros(x[0].size(), device=x.device).reshape(1, -1)
        subgraphs = [Data(x=zerox.clone(), y=y, edge_index=[]) for _ in range(T)]
        mappings = -np.ones((T, V))
        ei = edge_index
        for k in range(ei.shape[1]):
            u = int(ei[0, k]); v = int(ei[1, k])
            I = self.hash_edge(V, v, u) if u > v else self.hash_edge(V, u, v)
            if mappings[I, u] == -1:
                mappings[I, u] = subgraphs[I].x.shape[0]
                subgraphs[I].x = torch.cat((subgraphs[I].x, x[u].reshape(1, -1)), dim=0)
            if mappings[I, v] == -1:
                mappings[I, v] = subgraphs[I].x.shape[0]
                subgraphs[I].x = torch.cat((subgraphs[I].x, x[v].reshape(1, -1)), dim=0)
            subgraphs[I].edge_index.append([int(mappings[I, u]), int(mappings[I, v])])
        for i in range(T):
            if len(subgraphs[i].edge_index) == 0:
                subgraphs[i].edge_index = None
            else:
                subgraphs[i].edge_index = torch.tensor(
                    subgraphs[i].edge_index, dtype=torch.int64, device=x.device).t().contiguous()
        return subgraphs


def _empty_ei(device):
    return torch.zeros((2, 0), dtype=torch.int64, device=device)


class AGNNCertVoteModel(torch.nn.Module):
    """Wraps a trained base GCN. forward(data) -> per-graph vote-fraction logits over classes
    (argmax = AGNNCert-E majority vote). DETERMINISTIC (hash, no RNG). Splits a batched `data`
    into graphs, divides each into T subgraphs, classifies each subgraph with the base model,
    majority-votes. Attacks query .forward unchanged."""

    def __init__(self, base_model, num_classes, T=30, h="md5"):
        super().__init__()
        self.base = base_model
        self.hasher = HashAgentOfficial(h=h, T=T)
        self.num_classes = int(num_classes)
        self.T = int(T)
        self.eval()

    def _classify_subgraph(self, sx, sei, device):
        d = Data(x=sx, edge_index=(sei if sei is not None else _empty_ei(device)))
        d.batch = torch.zeros(sx.size(0), dtype=torch.long, device=device)
        return self.base(d)  # (1, C)

    def _vote_one(self, x, edge_index, y, device):
        subs = self.hasher.generate_graph_subgraphs(edge_index, x, y)
        votes = torch.zeros(self.num_classes, device=device)
        for s in subs:
            logits = self._classify_subgraph(s.x.to(device), s.edge_index, device)
            votes[int(logits[0].argmax())] += 1.0
        return votes / float(self.T)  # vote-fraction "logits"

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        device = x.device
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        n_graphs = int(batch.max().item()) + 1
        y = getattr(data, "y", None)
        out = []
        for g in range(n_graphs):
            nmask = (batch == g)
            nidx = torch.nonzero(nmask, as_tuple=False).flatten()
            remap = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            remap[nidx] = torch.arange(nidx.numel(), device=device)
            emask = nmask[edge_index[0]] & nmask[edge_index[1]]
            sub_ei = remap[edge_index[:, emask]]
            out.append(self._vote_one(x[nidx], sub_ei, y, device))
        return torch.stack(out, dim=0)
