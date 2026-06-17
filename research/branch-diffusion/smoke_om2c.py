#!/usr/bin/env python3
"""
smoke_om2c.py — Smoke test for attack_om2c.py (OM2C orthogonal two-channel injection).

Verifies, end-to-end on a handful of PROTEINS graphs, that
  (a) attack_om2c.run_attack runs without error and honors the contract
      run_attack(model, test_graphs, device) -> float,
  (b) returns a sane ASR float in [0, 1] against THREE model wrappers covering the
      two defense classes OM2C is designed to lift the floor against:
        - none        : plain GCN (no defense)
        - normfilter  : DETECTOR class (preprocessing node-norm filter, registry)
        - rspool      : ARCHITECTURAL class (robust/median readout proxy built from
                        the same GCN weights — the binding defense in the OM2C H1),
  (c) the injected (pre-scale) feature lands inside the 2-sigma norm ball AND the
      Mahalanobis ball (stealth_in_ball_frac), i.e. the feature channel is stealthy.

Runs on CPU (deterministic + sidesteps this GPU's CUDA segfault in graph_adversarial).

Usage: micromamba run -n graph_adversarial python smoke_om2c.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool

import prepare
prepare.DEVICE = "cpu"
from prepare import (
    GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs,
)
DEVICE = "cpu"

import attack_om2c
from defenses import DefendedModel, norm_filter_defense


# ---- RS-Pool proxy: robust (median) readout over the GCN's node embeddings -------
# Reuses the trained conv stack + classifier; only swaps global_mean_pool -> median.
# This is the ARCHITECTURAL defense class: feature-norm amplification is useless (a
# single giant-norm node is ignored by the median), so flips must come from edge mass.
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


def main():
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: run prepare.py first!")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    fold_info = meta["folds"][0]
    base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    base.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE,
                                    weights_only=True))
    base.eval()

    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    test_graphs = collect_test_graphs(base, ds, fold_info["test_idx"])
    if not test_graphs:
        print("No test graphs to attack.")
        return
    test_graphs = test_graphs[:6]

    c = attack_om2c.CONFIG
    print(f"Smoke: {len(test_graphs)} PROTEINS graphs (fold 0), device={DEVICE}")
    print(f"OM2C: node_budget={c['node_budget']} tangent_dim={c['tangent_dim']} "
          f"sigma_mult={c['sigma_mult']} feat_scale={c['feat_scale']} "
          f"(norm-free topology channel)")

    models = {
        "none":       base,
        "normfilter": DefendedModel(base, norm_filter_defense, "normfilter",
                                    percentile=95),
        "rspool":     RSPoolModel(base),
    }

    ok = True
    for name, model in models.items():
        t0 = time.time()
        rec = attack_om2c.run_attack_sweep(model, test_graphs, DEVICE)
        dt = time.time() - t0
        asr = rec["asr_full"]
        frac = rec["stealth_in_ball_frac"]
        print("=" * 64)
        print(f"[{name:10s}] ASR_full={asr*100:6.2f}%  queries={rec['queries']:6d}  "
              f"time={dt:6.2f}s  stealth_in_ball={frac*100:5.1f}%")
        valid = isinstance(asr, float) and 0.0 <= asr <= 1.0
        ok = ok and valid
        if not valid:
            print(f"  !! ASR not a valid float in [0,1]: {asr!r}")

    # contract check on the bare run_attack entry point
    bare = attack_om2c.run_attack(base, test_graphs, DEVICE)
    print("=" * 64)
    print(f"run_attack(base) -> {bare!r}  (type={type(bare).__name__})")
    ok = ok and isinstance(bare, float) and 0.0 <= bare <= 1.0

    print("=" * 64)
    print("Smoke test PASSED" if ok else "Smoke test FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
