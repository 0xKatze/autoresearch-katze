#!/usr/bin/env python3
"""gap1_train_singular.py — TRAIN OFFICIAL RS-Pool (singular-pooling) victims.

Mirror of gap1_train_median.py but uses singular_gcn.SingularReadoutGCN (official
Ennadir et al. _svd_pool) instead of the median readout. Same protocol as the mean-trained
pg victims (prepare.get_folds(seed,10), 200ep, Adam lr=1e-3, save-best-val) so folds
reconstruct identically and the comparison is fair. Clean acc target: usable (>~0.65).

GPU training (per feedback_experiment_gpu_errorcheck). No attack/model code modified.

Usage: micromamba run -n graph_adversarial python gap1_train_singular.py \
         [--seeds 42,7,1027] [--epochs 200] [--n-folds 10] [--device cuda] \
         [--num-iterations 5] [--value-alpha 1.0] [--one-fold-probe]
"""
import sys, json, time, argparse
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare
from prepare import HIDDEN_DIM, collect_test_graphs
from singular_gcn import SingularReadoutGCN


def clean_test_acc(model, ds, test_idx, device):
    model.eval()
    c = t = 0
    for data in Subset(ds, test_idx):
        data = data.to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def train_one(ds, tr_idx, va_idx, seed, fold_idx, epochs, device, save_path,
              num_iterations, value_alpha):
    tr = DataLoader(Subset(ds, tr_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(ds, va_idx), batch_size=32)
    torch.manual_seed(seed + fold_idx)
    model = SingularReadoutGCN(ds.num_features, ds.num_classes, HIDDEN_DIM,
                               num_iterations=num_iterations, value_alpha=value_alpha).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_va = 0.0
    for _ in range(epochs):
        model.train()
        for b in tr:
            b = b.to(device)
            loss = F.cross_entropy(model(b), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        c = t = 0
        with torch.no_grad():
            for b in va:
                b = b.to(device)
                c += (model(b).argmax(1) == b.y).sum().item(); t += b.y.size(0)
        if t and c / t > best_va:
            best_va = c / t
            torch.save(model.state_dict(), save_path)
    if not save_path.exists():
        torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model.eval(), best_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-iterations", type=int, default=5)
    ap.add_argument("--value-alpha", type=float, default=1.0)
    ap.add_argument("--one-fold-probe", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA unavailable (NO silent CPU fallback)")
    prepare.DEVICE = device
    seeds = [int(s) for s in args.seeds.split(",")]
    base = Path(prepare.MODELS_DIR)

    print(f"=== TRAIN singular-pooling (OFFICIAL RS-Pool) victims | PROTEINS GCN | seeds={seeds} "
          f"folds={args.n_folds} epochs={args.epochs} iters={args.num_iterations} "
          f"alpha={args.value_alpha} device={device} ===", flush=True)
    t0 = time.time()
    summary = {}
    for seed in seeds:
        prepare.SEED = seed; prepare.N_FOLDS = args.n_folds
        ds = prepare.prepare_dataset()
        seed_dir = base / f"singular_retrained_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        folds = list(prepare.get_folds(ds))
        if args.one_fold_probe:
            folds = folds[:1]
        fold_meta, accs = [], []
        for fold_idx, tr, va, te in folds:
            sp = seed_dir / f"PROTEINS_gcn_fold{fold_idx}.pt"
            tf = time.time()
            model, best_va = train_one(ds, tr, va, seed, fold_idx, args.epochs, device, sp,
                                       args.num_iterations, args.value_alpha)
            acc = clean_test_acc(model, ds, te, device)
            tg = collect_test_graphs(model, ds, te)
            accs.append(acc)
            print(f"  seed{seed} fold{fold_idx}: val_acc={best_va:.4f} "
                  f"clean_test_acc={acc:.4f} n_attackable={len(tg)} ({time.time()-tf:.1f}s)",
                  flush=True)
            fold_meta.append({"fold_idx": fold_idx, "model_path": str(sp),
                              "n_test_graphs": len(tg), "clean_test_acc": round(acc, 4),
                              "test_idx": [int(i) for i in te]})
        meta = {"dataset": "PROTEINS", "model": "gcn", "model_type": "GCN-singularpool",
                "readout": "singular", "num_iterations": args.num_iterations,
                "value_alpha": args.value_alpha,
                "readout_desc": f"official RS-Pool singular pooling (iters={args.num_iterations}, alpha={args.value_alpha})",
                "n_folds": len(fold_meta), "num_features": ds.num_features,
                "num_classes": ds.num_classes, "seed": seed, "device": device,
                "train_epochs": args.epochs,
                "clean_test_acc_mean": round(mean(accs), 4),
                "clean_test_acc_std": round(pstdev(accs), 4) if len(accs) > 1 else 0.0,
                "folds": fold_meta}
        mp = base / f"meta_singular_seed{seed}.json"
        mp.write_text(json.dumps(meta, indent=2))
        summary[seed] = (round(mean(accs), 4), round(pstdev(accs), 4) if len(accs) > 1 else 0.0)
        print(f"  -> seed{seed} clean_test_acc = {mean(accs):.4f} +/- "
              f"{(pstdev(accs) if len(accs)>1 else 0):.4f}  (mean-victim ref ~0.72)  "
              f"wrote {mp.name}", flush=True)

    print("=" * 80, flush=True)
    allm = mean([v[0] for v in summary.values()])
    print(f"singular-victim clean acc by seed: {summary}", flush=True)
    print(f"OVERALL singular-victim clean acc = {allm:.4f} "
          f"(usable? {'YES' if allm >= 0.65 else 'WEAK'})  [{time.time()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
