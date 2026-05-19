#!/usr/bin/env python3
"""
run_robust.py — Run attack.py against each victim architecture.

Loads meta_robust.json (from prepare_robust.py), runs the current
attack pipeline against each (arch, fold), tabulates per-fold and
mean ASR. Output goes to results/robust_eval_<timestamp>.json plus
a printed markdown table.

Does NOT modify attack.py, prepare.py, or run.py.
"""

import json
import time
import sys
import importlib
import hashlib
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch_geometric.datasets import TUDataset

from prepare import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, DEVICE,
    collect_test_graphs,
)
from victims import get_model_class


META_ROBUST_PATH = MODELS_DIR / "meta_robust.json"


def attack_hash():
    with open(Path(__file__).parent / "attack.py", "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def evaluate_arch(arch, fold_records, num_features, num_classes,
                  dataset_name, attack_module):
    cls = get_model_class(arch)
    fold_results = []
    for rec in fold_records:
        model = cls(num_features, num_classes).to(DEVICE)
        model.load_state_dict(
            torch.load(rec["model_path"], map_location=DEVICE,
                       weights_only=True))
        model.eval()

        ds = TUDataset(root=str(DATA_DIR / dataset_name), name=dataset_name)
        test_graphs = collect_test_graphs(model, ds, rec["test_idx"])
        if not test_graphs:
            asr = 0.0
            t = 0.0
            ns = 0
            nt = 0
        else:
            t0 = time.time()
            asr = attack_module.run_attack(model, test_graphs, DEVICE)
            t = time.time() - t0
            nt = len(test_graphs)
            ns = int(round(asr * nt))
        fold_results.append({
            "fold_idx": rec["fold_idx"],
            "asr": round(asr, 4),
            "n_success": ns,
            "n_test": nt,
            "test_acc": rec.get("test_acc"),
            "wall_time_s": round(t, 2),
        })
        print(f"  [{arch}] Fold {rec['fold_idx']}: ASR={asr*100:.2f}% "
              f"({ns}/{nt}), clean_acc={rec.get('test_acc')}, t={t:.1f}s")
    return fold_results


def main():
    if not META_ROBUST_PATH.exists():
        print(f"ERROR: {META_ROBUST_PATH} not found. Run prepare_robust.py first.")
        sys.exit(1)

    with open(META_ROBUST_PATH) as f:
        meta = json.load(f)

    archs = list(meta["architectures"].keys())
    print(f"Architectures: {archs}")
    print(f"Dataset: {meta['dataset']}, folds: {meta['n_folds']}")
    print(f"Attack hash: {attack_hash()}")
    print()

    # Reimport attack to pick up any changes
    import attack
    importlib.reload(attack)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    overall_t0 = time.time()

    for arch in archs:
        print(f"=== Attacking {arch} ===")
        records = meta["architectures"][arch]
        fold_results = evaluate_arch(
            arch, records, meta["num_features"], meta["num_classes"],
            meta["dataset"], attack)
        asrs = [r["asr"] for r in fold_results]
        mean_asr = float(np.mean(asrs))
        std_asr = float(np.std(asrs))
        all_results[arch] = {
            "folds": fold_results,
            "mean_asr": round(mean_asr, 4),
            "std_asr": round(std_asr, 4),
            "total_success": sum(r["n_success"] for r in fold_results),
            "total_test": sum(r["n_test"] for r in fold_results),
        }
        print(f"  → {arch}: mean ASR = {mean_asr*100:.2f}% ± {std_asr*100:.2f}%")
        print()

    total_time = time.time() - overall_t0
    print("=" * 64)
    print("# Robustness sweep — ASR per architecture")
    print()
    print("| Arch | Mean ASR | Std | Total | Wall time |")
    print("|------|----------|-----|-------|-----------|")
    for arch, r in all_results.items():
        tt = sum(f["wall_time_s"] for f in r["folds"])
        print(f"| {arch:11s} | {r['mean_asr']*100:6.2f}% | "
              f"{r['std_asr']*100:5.2f}% | "
              f"{r['total_success']}/{r['total_test']} | {tt:6.1f}s |")
    print()
    print(f"Total elapsed: {total_time:.1f}s")

    # Save
    out_path = RESULTS_DIR / f"robust_eval_{stamp}_{attack_hash()}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": stamp,
            "attack_hash": attack_hash(),
            "dataset": meta["dataset"],
            "results": all_results,
            "wall_time_s": round(total_time, 2),
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
