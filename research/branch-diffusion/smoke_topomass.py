#!/usr/bin/env python3
"""smoke_topomass.py — fast sanity check for the TopoMass feat_mode.

Verifies (does NOT measure publishable ASR):
  1. CONTRACT: run_attack(model, test_graphs, device) -> float in [0,1] on the
     prepared GCN/PROTEINS victim with feat_mode="topomass".
  2. EDGE-BUDGET sweep returns one ASR per eb in {1,4,8,16} (the KILL-test shape).
  3. MULTI-DIRECTION capped features: k injected rows are NOT collinear and every
     row sits inside the L2 cap (norm <= kappa * median_node_norm + eps).
  4. GAP path against an actual RS-Pool readout: the forward hook captures H and
     the singular-gap selector runs end-to-end, returning a float.

Run:  micromamba run -n graph_adversarial python3 smoke_topomass.py
"""
import sys
import math
import torch

import attack
from prepare import (
    GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs,
)
from torch_geometric.datasets import TUDataset
import json

# Smoke runs on CPU: it validates contract + logic only, and the CI sandbox has no GPU.
DEVICE = torch.device("cpu")


def _load_gcn_fold0(n_graphs=12):
    meta = json.load(open(MODELS_DIR / "meta.json"))
    fold = meta["folds"][0]
    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(fold["model_path"], map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    graphs = collect_test_graphs(model, ds, fold["test_idx"])[:n_graphs]
    return model, graphs


class _MiniRSPool(torch.nn.Module):
    """Smoke-only RS-Pool readout (tau * v1(H)) so the gap path hits a real victim."""

    def __init__(self, d_in, n_cls, hidden=16, alpha=20.0):
        super().__init__()
        self.l1 = torch.nn.Linear(d_in, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, n_cls)
        self.alpha = alpha

    def _readout(self, H):
        v = H.sum(dim=0)
        v = v / (v.norm() + 1e-8)
        for _ in range(5):
            v = H.t() @ (H @ v)
            v = v / (v.norm() + 1e-8)
        sigma1 = (H @ v).norm()
        return v * (sigma1 / self.alpha)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        # one shared message-passing step (adj@x) to couple topology into H
        outs = []
        for b in range(int(batch.max().item()) + 1):
            mask = batch == b
            xb = x[mask]
            H = torch.relu(self.l2(torch.relu(self.l1(xb))))  # node embeddings [n,hidden]
            outs.append(self.out(self._readout(H).unsqueeze(0)))
        return torch.cat(outs, dim=0)


def main():
    ok = True

    # ---- 1+2+3: contract + eb sweep + multi-direction cap on GCN/PROTEINS ----
    model, graphs = _load_gcn_fold0()
    cfg = {"feat_mode": "topomass", "tm_edge_select": "margin",
           "tm_edge_budgets": [1, 4, 8, 16], "tm_headline_budget": 8,
           "zo_epochs": 6, "zo_restarts": 1, "tm_pool_mult": 2}
    rec = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    print(f"[GCN] asr_full={rec['asr_full']:.3f} asr_by_eb={rec['asr_by_eb']} "
          f"queries={rec['queries']} k={rec['n_inj_nodes']} r={rec['tm_r']}")
    assert set(rec["asr_by_eb"]) == {1, 4, 8, 16}, "eb sweep keys wrong"
    assert all(0.0 <= v <= 1.0 for v in rec["asr_by_eb"].values()), "ASR out of range"
    # monotone in budget (early-stop greedy guarantees this)
    vals = [rec["asr_by_eb"][b] for b in (1, 4, 8, 16)]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])), "ASR not monotone in eb"

    # multi-direction + cap check on the raw initialiser
    stats = attack.data_stats(graphs, DEVICE)
    radius = 1.95 * stats[2]
    mu, dirs = attack._topomass_principal_dirs(graphs, DEVICE, r=2)
    feats = attack._topomass_init_features(mu, dirs, m=4, radius=radius)
    norms = feats.norm(dim=1)
    assert (norms <= radius + 1e-4).all(), f"norm cap violated: {norms} > {radius}"
    # not all rows collinear -> 2nd singular value of (feats-mu) is non-trivial
    # (true multi-direction, unlike single-direction N4-S where sigma2 == 0)
    svals, _ = attack._topk_right_singular((feats - mu).float(), 2)
    s1, s2 = float(svals[0]), float(svals[1])
    print(f"[feat] radius={radius:.3f} row_norms={[round(x,3) for x in norms.tolist()]} "
          f"sigma1={s1:.3f} sigma2={s2:.3f}")
    assert s2 > 1e-3 * max(s1, 1e-9), "injected rows are single-direction (collinear)"

    # ---- contract: run_attack returns a float in [0,1] ----
    orig_cfg = dict(attack.CONFIG)
    try:
        attack.CONFIG.update({**attack.TOPOMASS_DEFAULTS, "feat_mode": "topomass",
                              "tm_edge_select": "margin", "zo_epochs": 4, "zo_restarts": 1,
                              "tm_pool_mult": 2})
        asr = attack.run_attack(model, graphs, DEVICE)
    finally:
        attack.CONFIG.clear(); attack.CONFIG.update(orig_cfg)
    print(f"[contract] run_attack -> {asr!r}")
    assert isinstance(asr, float) and 0.0 <= asr <= 1.0, "run_attack must return float in [0,1]"

    # ---- 4: gap path against an RS-Pool readout (hook captures H) ----
    rs = _MiniRSPool(graphs[0].x.size(1), 2).to(DEVICE).eval()
    # relabel graphs to the RS-Pool victim's own predictions so flips are well-defined
    rs_graphs = []
    for g in graphs[:6]:
        gg = g.clone().to(DEVICE)
        with torch.no_grad():
            pred = rs(gg).argmax(dim=-1).item()
        gg.y = torch.tensor([pred], device=DEVICE)
        rs_graphs.append(gg)
    rec_gap = attack.run_attack_sweep(rs, rs_graphs, DEVICE, {
        "feat_mode": "topomass", "tm_edge_select": "gap",
        "tm_edge_budgets": [1, 4, 8], "tm_headline_budget": 8,
        "zo_epochs": 4, "zo_restarts": 1, "tm_pool_mult": 2})
    print(f"[RS-Pool/gap] asr_full={rec_gap['asr_full']:.3f} "
          f"asr_by_eb={rec_gap['asr_by_eb']} queries={rec_gap['queries']}")
    assert isinstance(rec_gap["asr_full"], float) and 0.0 <= rec_gap["asr_full"] <= 1.0

    print("\nSMOKE PASS" if ok else "\nSMOKE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
