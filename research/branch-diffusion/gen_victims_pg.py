#!/usr/bin/env python3
"""gen_victims_pg.py — generate paper-grade victim populations for the floor study.
Reuses prepare.py's FROZEN GCN architecture + training loop verbatim (import, no
reimplementation). For each victim-training seed in {42,7,1027} builds a 10-fold
StratifiedKFold split (random_state=seed) and trains 10 GCN victims, namespaced by
seed so filenames don't collide. Writes one meta_pg_seed{S}.json per seed.

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python gen_victims_pg.py [--folds N] [--seeds 42,7,1027] [--one-fold-probe]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, argparse
from pathlib import Path
import torch
from torch.utils.data import Subset
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"

def gen_for_seed(seed, n_folds, base_models_dir, probe_one=False):
    prepare.SEED = seed
    prepare.N_FOLDS = n_folds
    seed_dir = base_models_dir / f"pg_floor_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prepare.MODELS_DIR = seed_dir       # train_victim saves here
    ds = prepare.prepare_dataset()
    fold_data = []
    folds = list(prepare.get_folds(ds))
    if probe_one:
        folds = folds[:1]
    for fold_idx, tr, va, te in folds:
        t0 = time.time()
        model = prepare.train_victim(ds, tr, va, fold_idx)
        tg = prepare.collect_test_graphs(model, ds, te)
        dt = time.time() - t0
        print(f"  seed{seed} fold{fold_idx}: n_test_correct={len(tg)} ({dt:.1f}s)", flush=True)
        fold_data.append({
            "fold_idx": fold_idx,
            "model_path": str(seed_dir / f"{prepare.DATASET_NAME}_{prepare.MODEL_TYPE}_fold{fold_idx}.pt"),
            "n_test_graphs": len(tg),
            "test_idx": [int(i) for i in te],
        })
    meta = {
        "dataset": prepare.DATASET_NAME, "model": prepare.MODEL_TYPE,
        "model_type": "GCN", "n_folds": len(fold_data),
        "num_features": ds.num_features, "num_classes": ds.num_classes,
        "seed": seed, "device": "cpu", "train_epochs": prepare.TRAIN_EPOCHS,
        "folds": fold_data,
    }
    out = base_models_dir / f"meta_pg_seed{seed}.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"  -> wrote {out}", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--one-fold-probe", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    base = Path(prepare.MODELS_DIR)  # original models_saved (symlink target)
    print(f"gen victims: seeds={seeds} folds={args.folds} base={base}", flush=True)
    for s in seeds:
        print(f"=== seed {s} ===", flush=True)
        gen_for_seed(s, args.folds, base, probe_one=args.one_fold_probe)

if __name__ == "__main__":
    main()
