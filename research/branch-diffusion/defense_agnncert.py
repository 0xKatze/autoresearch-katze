#!/usr/bin/env python3
"""defense_agnncert.py — AGNNCert (Li & Wang, USENIX Security 2025) as a model WRAPPER.

A 6th defense for the floor battery, re-implemented FROM THE PAPER (no external repo
cloned/run). AGNNCert is the only PRIOR certified defense covering GRAPH-classification
node injection (deterministic subgraph-division + majority vote), so reviewers will demand
it alongside the empirical battery {normfilter, mahalanobis, degfilter, bismooth, rspool}.

Mechanism implemented here — AGNNCert-E (edge-centric graph division), the paper's default
graph-classification strategy (Sec. 3.2; same lineage as GNNCert [64]):

  Step I  (divide):  use a hash function h (MD5) to map every edge e=(u,v) of the graph
                     into a subgraph index  i_e = h[str(u)+str(v)] mod T  (Eq. 3). For an
                     UNDIRECTED edge the smaller node index is placed first so (u,v) and
                     (v,u) hash to the SAME subgraph (paper footnote 3). This yields T
                     subgraphs G_i = (V, E_i, X): every subgraph keeps ALL nodes and ALL
                     features but a DISJOINT partition of the edges (E_i ∩ E_j = ∅).
  Step II (vote):    run the trained graph classifier f on each of the T subgraphs and
                     count votes c_y = Σ_i 1[f(G_i)=y]; the voting classifier returns the
                     majority class argmax_y c_y  (Eq. 8, 10).
  Step III (certify): the deterministic certified perturbation size (Thms. 1-6/11) is NOT
                     needed for an EMPIRICAL ASR battery — we only need the voting forward.
                     We therefore implement Steps I-II as a query-able model wrapper; the
                     certification guarantee is what motivates including it, the voting
                     classifier is what the attacks query.

Paper hyperparameters (Sec. 4.1 "Parameter setting"): hash = MD5; number of subgraphs
T = 300 by default for GRAPH classification (T = 30 for node classification). PROTEINS/GCN
clean (m=0) graph accuracy with AGNNCert-E in the paper = 0.75 (Table 2), vs 0.75 undefended
— but the paper's GNN is RE-TRAINED on the generated subgraphs. Here, like every other
inference-time wrapper in the battery (bismooth, rspool-swap), we apply AGNNCert to the
EXISTING mean-trained GCN victim WITHOUT retraining, so a clean-accuracy drop is expected
and is the honest operating point reported by the add-on eval.

Like RSPoolModel / BiSmoothingModel this is a model WRAPPER exposing .forward(data)->logits
(here vote-fraction logits; argmax = the AGNNCert majority class) so the existing black-box
attacks query it UNCHANGED. AGNNCert is DETERMINISTIC (hash, no RNG) — unlike bi-smoothing
there is no CPU/GPU RNG-stream divergence, so the GPU floor is exact.

No existing attack/model/defense file is modified; this is a NEW additive module.
"""
import hashlib
import torch
from types import SimpleNamespace

_HASH_WIDTH = 7   # zero-pad node-index strings to a fixed width (paper: "adds 0 prefix to
                  # align it into a fixed length") so str(u)+str(v) is unambiguous. 7 digits
                  # >> any TU-graph node count (PROTEINS max ~620 nodes).


def _edge_bucket(a, b, T):
    """AGNNCert-E Eq.(3): subgraph index of an undirected edge {a,b} (a,b = LOCAL node
    indices within one graph). Smaller index first (footnote 3) so (u,v)==(v,u); MD5 hash
    of the zero-padded concatenated index string, taken mod T."""
    lo, hi = (a, b) if a <= b else (b, a)
    s = f"{lo:0{_HASH_WIDTH}d}{hi:0{_HASH_WIDTH}d}"
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % T


class AGNNCertModel(torch.nn.Module):
    """AGNNCert-E voting graph classifier wrapping a trained GCN victim.

    forward(data) returns per-graph VOTE-FRACTION "logits" over classes (argmax = the
    AGNNCert majority class). The T subgraphs of every graph share the SAME node set and
    features and differ only in their disjoint edge partition; all T*n_graphs subgraphs are
    concatenated into ONE batched base forward (like BiSmoothingModel), so the wrapper costs
    ~T x the node-compute of one clean forward but a single kernel launch. Every subgraph
    keeps all original nodes, so each pooled block is non-empty and batch indices stay
    contiguous [0, T*n_graphs).

    Edge buckets use per-graph LOCAL node indices, so the partition (and hence the vote) of
    a given graph is INDEPENDENT of where it sits in a batch — matching the paper's
    single-graph semantics exactly.
    """

    def __init__(self, base_model, T=300, num_classes=None):
        super().__init__()
        self.base = base_model
        self.T = int(T)
        self.num_classes = num_classes
        self.eval()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        device = x.device
        N = x.size(0)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        n_graphs = int(batch.max().item()) + 1 if N > 0 else 1
        T = self.T

        # per-graph node counts and the global index of each graph's first node (PyG batches
        # are node-contiguous, so local index = global index - graph_start[graph_of_node]).
        counts = torch.bincount(batch, minlength=n_graphs)                 # [n_graphs]
        starts = torch.zeros(n_graphs, dtype=torch.long, device=device)
        if n_graphs > 1:
            starts[1:] = torch.cumsum(counts, 0)[:-1]
        local = torch.arange(N, device=device) - starts[batch]            # [N] local node idx

        # block (g, i) holds graph g's subgraph i; blocks ordered g-major then i so the
        # base output row for (g,i) is at index g*T+i. block_start[g] = nodes before g's T
        # blocks; node offset of block (g,i) = block_start[g] + i*counts[g].
        block_start = torch.zeros(n_graphs, dtype=torch.long, device=device)
        if n_graphs > 1:
            block_start[1:] = torch.cumsum(counts * T, 0)[:-1]

        # --- Step I: assign every edge to a subgraph bucket (CPU md5 loop) -----------------
        E = edge_index.size(1)
        if E > 0:
            ge = batch[edge_index[0]]                                      # graph id per edge
            lu = local[edge_index[0]].tolist()
            lv = local[edge_index[1]].tolist()
            ie = torch.tensor([_edge_bucket(a, b, T) for a, b in zip(lu, lv)],
                              dtype=torch.long, device=device)             # [E] bucket index
            node_off_e = block_start[ge] + ie * counts[ge]
            new_src = local[edge_index[0]] + node_off_e
            new_dst = local[edge_index[1]] + node_off_e
            big_ei = torch.stack([new_src, new_dst], dim=0)
        else:
            big_ei = edge_index.new_zeros((2, 0))

        # --- replicate nodes/features T times per graph; build the big batch vector --------
        xs, bs = [], []
        for g in range(n_graphs):
            gmask = batch == g
            xg = x[gmask]                                                  # [N_g, F]
            Ng = xg.size(0)
            xs.append(xg.repeat(T, 1))                                     # T*N_g nodes
            bl = torch.arange(T, device=device).repeat_interleave(Ng) + g * T
            bs.append(bl)
        big = SimpleNamespace(
            x=torch.cat(xs, dim=0),
            edge_index=big_ei,
            batch=torch.cat(bs, dim=0),
        )

        # --- Step II: base classifier on all subgraphs, then majority vote -----------------
        with torch.no_grad():
            logits_all = self.base(big)                                    # [T*n_graphs, C]
        C = logits_all.size(1)
        pred = logits_all.argmax(dim=1).view(n_graphs, T)                  # [n_graphs, T]
        votes = torch.zeros(n_graphs, C, device=device)
        for c in range(C):
            votes[:, c] = (pred == c).sum(dim=1).float()
        return votes / float(T)                                           # vote-fraction logits
