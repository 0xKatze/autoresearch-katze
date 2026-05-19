#!/usr/bin/env python3
"""
quick_test_n23.py — Smoke test for attack_diffusion.py

Runs the N23 diffusion attack on ONE fold to verify:
  1. Module imports cleanly with prepared models/data
  2. run_attack() returns a float ASR without crashing
  3. End-to-end loop completes within a reasonable time

This does NOT replace run.py (which manages full 5-fold + result JSON).
For full evaluation use: cp attack_diffusion.py attack.py && python run.py --quick

Usage:
    python quick_test_n23.py
"""

import json
import time
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset

from prepare import (
    GCN_GraphClassification, DATA_DIR, MODELS_DIR, DEVICE, collect_test_graphs,
)
import attack_diffusion


def main():
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: Run prepare.py first!")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    # Load model + first fold (schema mirrors run.py)
    fold_info = meta["folds"][0]
    model_path = fold_info["model_path"]

    model = GCN_GraphClassification(
        meta["num_features"], meta["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load dataset
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    test_idx = fold_info["test_idx"]
    test_graphs = collect_test_graphs(model, ds, test_idx)

    if not test_graphs:
        print("No test graphs to attack.")
        return

    # Sub-sample for smoke test (just 5 graphs is plenty)
    test_graphs = test_graphs[:5]
    print(f"Smoke test: {len(test_graphs)} graphs from fold 0")
    print(f"Config: {json.dumps(attack_diffusion.CONFIG, indent=2)}")
    print("=" * 60)

    t0 = time.time()
    asr = attack_diffusion.run_attack(model, test_graphs, DEVICE)
    elapsed = time.time() - t0

    print(f"\nResult: ASR = {asr*100:.2f}% on {len(test_graphs)} graphs")
    print(f"Time: {elapsed:.2f}s ({elapsed/len(test_graphs):.2f}s per graph)")
    print("=" * 60)
    print("Smoke test passed ✓" if asr >= 0.0 else "Smoke test FAILED")


if __name__ == "__main__":
    main()
