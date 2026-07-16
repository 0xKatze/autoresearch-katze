#!/usr/bin/env python3
"""agnncert_fast.py — fast, structure-cached evaluator for AGNNCert-E under a FIXED injected-edge
set (the N33/N34 cover attack). The subgraph DIVISION depends only on graph topology (original +
injected edges), NOT on injected node features, which is all that changes during ZO. So we
precompute the T-subgraph structure ONCE and BATCH all T subgraphs into a single base-model forward.

Reproduces agnncert_official.HashAgentOfficial.generate_graph_subgraphs EXACTLY (pad node idx0 +
first-appearance node ordering + directed local edges), validated in __main__ against the slow path.
"""
import sys
from pathlib import Path
BD = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
sys.path.insert(0, str(BD))
import torch
from torch_geometric.data import Data
from agnncert_official import HashAgentOfficial


class FastCoverEval:
    """Precompute the batched T-subgraph structure for (original graph + injected cover edges).
    Then eval(x_injected, fs) -> per-subgraph logits (T, C) in ONE base forward."""

    def __init__(self, base, data, edges, nb, hasher, T, device):
        self.base = base
        self.device = device
        self.T = T
        self.N = int(data.num_nodes)
        self.nb = nb
        self.nfeat = int(data.x.size(1))
        self.orig_x = data.x.to(device)                     # (N, F) fixed
        V = self.N + nb
        # combined directed edge list in SAME column order as build_cover_graph:
        #   original edge_index columns, then per (i,v): (j->v),(v->j)
        cols = [data.edge_index.to(device)]
        if edges:
            src, dst = [], []
            for (i, v) in edges:
                j = self.N + i
                src += [j, v]; dst += [v, j]
            cols.append(torch.tensor([src, dst], dtype=torch.long, device=device))
        comb = torch.cat(cols, dim=1)
        # replicate generate_graph_subgraphs structure per subgraph
        # per subgraph i: ordered list of combined node-ids present, and local directed edges
        maps = [dict() for _ in range(T)]      # comb_id -> local pos (pad=0, reals from 1)
        order = [[] for _ in range(T)]         # ordered comb ids (local pos = index+1)
        ledges = [[] for _ in range(T)]        # local directed [lu, lv]
        cs = comb[0].tolist(); cd = comb[1].tolist()
        for u, v in zip(cs, cd):
            I = hasher.hash_edge(V, v, u) if u > v else hasher.hash_edge(V, u, v)
            m = maps[I]
            if u not in m:
                m[u] = len(order[I]) + 1; order[I].append(u)
            if v not in m:
                m[v] = len(order[I]) + 1; order[I].append(v)
            ledges[I].append([m[u], m[v]])
        # assemble ONE batched graph: each subgraph = pad(0) + its ordered nodes
        feat_src = []        # per global node: combined-id, or -1 for pad (zero feature)
        batch = []
        glob_ei_s, glob_ei_d = [], []
        off = 0
        for i in range(T):
            feat_src.append(-1); batch.append(i)            # pad node
            for cid in order[i]:
                feat_src.append(cid); batch.append(i)
            n_i = 1 + len(order[i])
            for (lu, lv) in ledges[i]:
                glob_ei_s.append(off + lu); glob_ei_d.append(off + lv)
            off += n_i
        self.feat_src = torch.tensor(feat_src, dtype=torch.long, device=device)
        self.real_mask = self.feat_src >= 0
        self.feat_src_clamped = self.feat_src.clamp(min=0)
        self.batch = torch.tensor(batch, dtype=torch.long, device=device)
        self.edge_index = (torch.tensor([glob_ei_s, glob_ei_d], dtype=torch.long, device=device)
                           if glob_ei_s else torch.zeros((2, 0), dtype=torch.long, device=device))
        self.total_nodes = off

    def _build_X(self, x_injected, fs):
        # combined feature matrix (N+nb, F): original rows + injected*fs
        Xc = torch.cat([self.orig_x, x_injected * fs], dim=0)        # (N+nb, F)
        gathered = Xc[self.feat_src_clamped]                          # (total, F)
        gathered = gathered * self.real_mask.unsqueeze(1)            # pad rows -> zero
        return gathered

    def logits(self, x_injected, fs):
        X = self._build_X(x_injected, fs)
        d = Data(x=X, edge_index=self.edge_index)
        d.batch = self.batch
        with torch.no_grad():
            return self.base(d)        # (T, C)

    def vote_counts(self, x_injected, fs):
        lg = self.logits(x_injected, fs)
        votes = torch.bincount(lg.argmax(dim=1), minlength=self.base_nc())
        return votes.float()

    def base_nc(self):
        # infer #classes from a tiny forward cache
        if not hasattr(self, "_nc"):
            self._nc = int(self.logits(torch.zeros(self.nb, self.nfeat, device=self.device), 1.0).size(1))
        return self._nc


if __name__ == "__main__":
    # VALIDATION: fast vote counts must equal the slow official path for random injected features.
    import json, time
    import prepare; prepare.DEVICE = "cpu"
    from prepare import GCN_GraphClassification, collect_test_graphs, DATA_DIR
    from agnncert_official import AGNNCertVoteModel
    from torch_geometric.datasets import TUDataset
    import attack_agnncert_adaptive as A
    torch.set_num_threads(2)
    dev = "cpu"
    meta = json.loads((BD / "models_saved" / "meta_agnncert_official_seed42.json").read_text())
    T = meta.get("T", 30)
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    fi = meta["folds"][0]
    base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(dev)
    base.load_state_dict(torch.load(fi["model_path"], map_location=dev, weights_only=True)); base.eval()
    vm = AGNNCertVoteModel(base, meta["num_classes"], T=T).to(dev).eval()
    g = collect_test_graphs(vm, ds, fi["test_idx"])[0].to(dev)
    N = g.num_nodes; nb = 10; fs = A.resolve_fs("auto_x2", N)
    edges, ncov = A.cover_edges(N, nb, vm.hasher, k=1)
    fce = FastCoverEval(base, g, edges, nb, vm.hasher, T, dev)
    ok = True
    t_slow = t_fast = 0.0
    for r in range(4):
        x = torch.randn(nb, meta["num_features"], generator=torch.Generator().manual_seed(r), device=dev)
        t0 = time.time()
        gg = A.build_cover_graph(g, x, edges, fs)
        d = Data(x=gg.x, edge_index=gg.edge_index); d.batch = torch.zeros(gg.x.size(0), dtype=torch.long)
        with torch.no_grad():
            slow_counts = torch.round(vm(d)[0] * T)
        t_slow += time.time() - t0
        t0 = time.time()
        fast_counts = fce.vote_counts(x, fs)
        t_fast += time.time() - t0
        match = torch.equal(slow_counts, fast_counts)
        ok = ok and match
        print(f"r{r}: slow={slow_counts.tolist()} fast={fast_counts.tolist()} match={match}")
    print(f"ALL MATCH={ok}  slow={t_slow/4*1000:.1f}ms/eval  fast={t_fast/4*1000:.1f}ms/eval  "
          f"speedup={t_slow/max(t_fast,1e-9):.1f}x")
