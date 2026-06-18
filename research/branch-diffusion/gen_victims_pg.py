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
from torch.utils.data import Subset as _Subset


def _clean_test_acc(model, ds, test_idx):
    """Population-level clean test accuracy (all test graphs)."""
    model.eval()
    c = t = 0
    for data in _Subset(ds, test_idx):
        data = data.to(prepare.DEVICE)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=prepare.DEVICE)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def gen_for_seed(seed, n_folds, base_models_dir, probe_one=False, force=False):
    prepare.SEED = seed
    prepare.N_FOLDS = n_folds
    seed_dir = base_models_dir / f"pg_floor_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prepare.MODELS_DIR = seed_dir       # train_victim saves here
    # train_victim() short-circuits when a checkpoint already exists (it loads
    # instead of retraining). To regenerate at a DIFFERENT epoch budget (M3:
    # 100 -> 200), the stale checkpoints must be removed first, otherwise the old
    # 100-epoch weights would be silently reused. --force backs them up then clears.
    if force:
        stale = sorted(seed_dir.glob(f"{prepare.DATASET_NAME}_{prepare.MODEL_TYPE}_fold*.pt"))
        if stale:
            bak = seed_dir / "_bak_epoch100"
            bak.mkdir(exist_ok=True)
            for p in stale:
                p.replace(bak / p.name)
            print(f"  seed{seed}: moved {len(stale)} stale checkpoints -> {bak}", flush=True)
    ds = prepare.prepare_dataset()
    fold_data = []
    folds = list(prepare.get_folds(ds))
    if probe_one:
        folds = folds[:1]
    for fold_idx, tr, va, te in folds:
        t0 = time.time()
        model = prepare.train_victim(ds, tr, va, fold_idx)
        tg = prepare.collect_test_graphs(model, ds, te)
        # clean test accuracy over ALL test graphs (not just the correctly-classified
        # subset) so M3 can report the population-level clean acc vs the old 100-epoch run.
        clean_acc = _clean_test_acc(model, ds, te)
        dt = time.time() - t0
        print(f"  seed{seed} fold{fold_idx}: clean_test_acc={clean_acc:.4f} "
              f"n_test_correct={len(tg)} ({dt:.1f}s)", flush=True)
        fold_data.append({
            "fold_idx": fold_idx,
            "model_path": str(seed_dir / f"{prepare.DATASET_NAME}_{prepare.MODEL_TYPE}_fold{fold_idx}.pt"),
            "n_test_graphs": len(tg),
            "clean_test_acc": round(clean_acc, 4),
            "test_idx": [int(i) for i in te],
        })
    mean_acc = sum(f["clean_test_acc"] for f in fold_data) / max(len(fold_data), 1)
    meta = {
        "dataset": prepare.DATASET_NAME, "model": prepare.MODEL_TYPE,
        "model_type": "GCN", "n_folds": len(fold_data),
        "num_features": ds.num_features, "num_classes": ds.num_classes,
        "seed": seed, "device": "cpu", "train_epochs": prepare.TRAIN_EPOCHS,
        "clean_test_acc_mean": round(mean_acc, 4),
        "folds": fold_data,
    }
    out = base_models_dir / f"meta_pg_seed{seed}.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"  -> wrote {out}  (mean clean_test_acc={mean_acc:.4f}, "
          f"train_epochs={prepare.TRAIN_EPOCHS})", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--train-epochs", type=int, default=None,
                    help="override prepare.TRAIN_EPOCHS (M3: use 200, the project standard; "
                         "default keeps prepare.py's value)")
    ap.add_argument("--force", action="store_true",
                    help="back up + remove existing fold checkpoints before retraining "
                         "(required when changing --train-epochs, else train_victim reuses them)")
    ap.add_argument("--one-fold-probe", action="store_true")
    args = ap.parse_args()
    if args.train_epochs is not None:
        prepare.TRAIN_EPOCHS = args.train_epochs
    seeds = [int(s) for s in args.seeds.split(",")]
    base = Path(prepare.MODELS_DIR)  # original models_saved (symlink target)
    print(f"gen victims: seeds={seeds} folds={args.folds} "
          f"train_epochs={prepare.TRAIN_EPOCHS} force={args.force} base={base}", flush=True)
    for s in seeds:
        print(f"=== seed {s} ===", flush=True)
        gen_for_seed(s, args.folds, base, probe_one=args.one_fold_probe, force=args.force)

if __name__ == "__main__":
    main()
