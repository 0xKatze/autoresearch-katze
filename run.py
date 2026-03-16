#!/usr/bin/env python3
"""
run.py — Experiment runner (DO NOT MODIFY)

Loads prepared models and data, runs attack.py, records results.
Designed to complete in < 5 minutes for quick iteration.
"""

import json
import time
import sys
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset

from prepare import (
    GCN_GraphClassification, DATA_DIR, MODELS_DIR, RESULTS_DIR, BEST_FILE,
    DEVICE, collect_test_graphs,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_attack_hash():
    """Hash attack.py content for tracking changes."""
    with open(Path(__file__).parent / "attack.py", "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def get_git_diff():
    """Get current git diff of attack.py."""
    try:
        result = subprocess.run(
            ["git", "diff", "attack.py"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return result.stdout[:2000] if result.stdout else "(no changes)"
    except Exception:
        return "(git not available)"


def run_single_fold(fold_info, meta):
    """Run attack on a single fold, return ASR."""
    fold_idx = fold_info["fold_idx"]
    model_path = fold_info["model_path"]
    test_idx = fold_info["test_idx"]

    # Load model
    model = GCN_GraphClassification(
        meta["num_features"], meta["num_classes"]).to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Load dataset and get test graphs
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    test_graphs = collect_test_graphs(model, ds, test_idx)

    if not test_graphs:
        return 0.0, 0, 0

    # Import attack (reimport to pick up changes)
    import importlib
    import attack
    importlib.reload(attack)

    # Run attack
    asr = attack.run_attack(model, test_graphs, DEVICE)
    n_success = int(asr * len(test_graphs))
    return asr, n_success, len(test_graphs)


def run_experiment(quick=False):
    """Run full experiment across all folds.

    Args:
        quick: if True, only run 1 fold (for fast iteration)
    """
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: Run prepare.py first!")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    import attack
    config = attack.CONFIG.copy()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attack_hash = get_attack_hash()
    print(f"{'='*60}")
    print(f"Experiment: {stamp} (attack.py hash: {attack_hash})")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}")

    folds = meta["folds"]
    if quick:
        folds = folds[:1]

    asrs = []
    total_success = 0
    total_attacked = 0
    t0 = time.time()

    for fold_info in folds:
        fold_idx = fold_info["fold_idx"]
        ft = time.time()
        asr, ns, na = run_single_fold(fold_info, meta)
        elapsed = time.time() - ft
        asrs.append(asr)
        total_success += ns
        total_attacked += na
        print(f"  Fold {fold_idx}: ASR={asr*100:.2f}% ({ns}/{na}) [{elapsed:.1f}s]")

    total_time = time.time() - t0
    mean_asr = float(np.mean(asrs))
    std_asr = float(np.std(asrs))

    print(f"\n{'='*60}")
    print(f"RESULT: ASR = {mean_asr*100:.2f}% +/- {std_asr*100:.2f}%")
    print(f"Total: {total_success}/{total_attacked}, Time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Save result
    result = {
        "timestamp": stamp,
        "attack_hash": attack_hash,
        "config": config,
        "mean_asr": round(mean_asr, 4),
        "std_asr": round(std_asr, 4),
        "per_fold_asr": [round(a, 4) for a in asrs],
        "total_success": total_success,
        "total_attacked": total_attacked,
        "wall_time_s": round(total_time, 2),
        "n_folds": len(folds),
        "git_diff": get_git_diff(),
    }

    result_path = RESULTS_DIR / f"exp_{stamp}_{attack_hash}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {result_path}")

    # Update best.json
    with open(BEST_FILE) as f:
        best = json.load(f)

    if mean_asr > best["best_asr"]:
        print(f"\n*** NEW BEST: {mean_asr*100:.2f}% (prev: {best['best_asr']*100:.2f}%) ***")
        best["best_asr"] = round(mean_asr, 4)
        best["best_config"] = config
        best["best_timestamp"] = stamp
        best["best_hash"] = attack_hash

    best["history"].append({
        "timestamp": stamp,
        "hash": attack_hash,
        "mean_asr": round(mean_asr, 4),
        "std_asr": round(std_asr, 4),
    })

    with open(BEST_FILE, "w") as f:
        json.dump(best, f, indent=2)

    return result


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_experiment(quick=quick)
