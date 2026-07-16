#!/usr/bin/env python3
"""
test_specgdi.py — Smoke test for SpecGDI on PROTEINS fold 0, seed 0.

Runs feat_mode="specgdi" on a small slice of the PROTEINS test set and logs:
  - per-graph S_spec achieved (mean / max)
  - τ_spec from specgdi_thresholds.json
  - ASR at the spec-stealth budget (full_flip after edge selection)
  - wall-clock

Sanity expectation (from program spec):
  mean S_spec << diagnostic baseline (~0.5) — target < 0.1.

Usage:
  micromamba run -n graph_adversarial python test_specgdi.py [--full]
"""
import json
import sys
import time
from pathlib import Path

import torch

import attack
from prepare import (
    MODELS_DIR, DEVICE, GCN_GraphClassification, DATA_DIR, collect_test_graphs,
)
from torch_geometric.datasets import TUDataset


N_GRAPHS_SMOKE = 10  # small slice to stay fast


def main():
    full = "--full" in sys.argv
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]

    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    graphs = collect_test_graphs(model, ds, fi["test_idx"])
    print(f"loaded {len(graphs)} correctly-classified PROTEINS fold-0 test graphs")
    if not full:
        # pick the 10 median-sized graphs — small enough for fast smoke, large
        # enough to be representative (the head/tail of PROTEINS is degenerate).
        graphs_sorted = sorted(graphs, key=lambda g: g.num_nodes)
        mid = len(graphs_sorted) // 2
        graphs = graphs_sorted[max(0, mid - N_GRAPHS_SMOKE // 2):
                               mid + N_GRAPHS_SMOKE - N_GRAPHS_SMOKE // 2]
        print(f"smoke slice: {len(graphs)} median-sized graphs "
              f"(sizes {[g.num_nodes for g in graphs]})")

    cfg = {
        "node_budget": 1, "skip_greedy": True,
        "feat_mode": "specgdi",
        "seed_offset": 0,
        # keep diffusion cheap for the smoke; bump for paper runs.
        "sgdi_T_diff": 8,
        "zo_epochs": 8, "zo_restarts": 1,
        "specgdi_swaps_per_step": 4,
    }
    t0 = time.time()
    o = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    dt = time.time() - t0

    s_specs = o["s_spec_per_graph"]
    s_mean = sum(s_specs) / max(len(s_specs), 1)
    s_max = max(s_specs) if s_specs else 0.0

    print("\n=== SpecGDI smoke ===")
    print(f"  n_graphs              : {o['n_graphs']}")
    print(f"  tau_spec              : {o['tau_spec']:.4f}  (dataset_key={o['dataset_key']})")
    print(f"  S_spec  mean / max    : {s_mean:.4f} / {s_max:.4f}")
    print(f"  baseline diag (~full) : 0.5262 (PROTEINS, full-edge)")
    print(f"  S_spec  target        : < 0.1  ->  {'PASS' if s_mean < 0.1 else 'FAIL'}")
    print(f"  S_spec ≤ τ_spec       : {'PASS' if s_max <= o['tau_spec'] else 'FAIL'}")
    print(f"  ASR (full_flip)       : {o['asr_full']:.3f}")
    print(f"  queries / graph       : {o['queries'] / max(o['n_graphs'],1):.1f}")
    print(f"  wall-clock            : {dt:.1f}s ({dt/max(o['n_graphs'],1):.2f}s/graph)")

    # also a quick specgdi_signal sanity (1 graph) so we exercise that path too
    if graphs:
        cfg2 = {**cfg, "feat_mode": "specgdi_signal"}
        o2 = attack.run_attack_sweep(model, graphs[:3], DEVICE, cfg2)
        print(f"\n  signal-variant (n=3)  : S_spec mean={sum(o2['s_spec_per_graph'])/3:.4f}  "
              f"ASR={o2['asr_full']:.3f}")


if __name__ == "__main__":
    main()
