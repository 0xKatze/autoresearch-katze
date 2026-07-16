#!/usr/bin/env python3
"""gap1_train_median.py — TRAIN proper retrained RS-Pool (median-readout) victims.

Gap-1 make-or-break: the floor study's rspool was a test-time median SWAP on a
mean-trained victim (clean acc drops, ASR inflates -> invalid). Here we TRAIN GCN
victims end-to-end with the median readout (median_gcn.MedianReadoutGCN), matching the
EXISTING mean-trained pg victims' protocol EXACTLY (prepare.get_folds(seed,10) splits,
200 epochs, Adam lr=1e-3, save-best-val), so test/train folds reconstruct identically
to meta_pg_seed{S}.json and the comparison is fair. Clean acc must recover to ~0.72.

GPU training (free GPU; per feedback_experiment_gpu_errorcheck use GPU for victim
training). Attacks against these victims run separately on CPU (gap1_attack_median.py,
device-FP rule). No attack/model/defense code modified; reuses prepare data+folds.

Usage: micromamba run -n graph_adversarial python gap1_train_median.py \
         [--seeds 42,7,1027] [--epochs 200] [--n-folds 10] [--device cuda]
"""
import os, sys, json, time, argparse
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
from median_gcn import MedianReadoutGCN


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
              readout="trimmed", trim=0.2):
    tr = DataLoader(Subset(ds, tr_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(ds, va_idx), batch_size=32)
    torch.manual_seed(seed + fold_idx)            # matches prepare.train_victim seeding
    model = MedianReadoutGCN(ds.num_features, ds.num_classes, HIDDEN_DIM,
                             readout=readout, trim=trim).to(device)
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
    ap.add_argument("--readout", type=str, default="trimmed", choices=["trimmed", "median"])
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--one-fold-probe", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA unavailable (NO silent CPU fallback)")
    prepare.DEVICE = device
    seeds = [int(s) for s in args.seeds.split(",")]
    base = Path(prepare.MODELS_DIR)

    print(f"=== GAP-1 TRAIN median-readout victims | PROTEINS GCN | seeds={seeds} "
          f"folds={args.n_folds} epochs={args.epochs} device={device} ===", flush=True)
    t0 = time.time()
    summary = {}
    for seed in seeds:
        prepare.SEED = seed; prepare.N_FOLDS = args.n_folds
        ds = prepare.prepare_dataset()
        seed_dir = base / f"rspool_retrained_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        folds = list(prepare.get_folds(ds))
        if args.one_fold_probe:
            folds = folds[:1]
        fold_meta, accs = [], []
        for fold_idx, tr, va, te in folds:
            sp = seed_dir / f"PROTEINS_gcn_fold{fold_idx}.pt"
            tf = time.time()
            model, best_va = train_one(ds, tr, va, seed, fold_idx, args.epochs, device, sp,
                                       readout=args.readout, trim=args.trim)
            acc = clean_test_acc(model, ds, te, device)
            tg = collect_test_graphs(model, ds, te)
            accs.append(acc)
            print(f"  seed{seed} fold{fold_idx}: val_acc={best_va:.4f} "
                  f"clean_test_acc={acc:.4f} n_attackable={len(tg)} ({time.time()-tf:.1f}s)",
                  flush=True)
            fold_meta.append({"fold_idx": fold_idx, "model_path": str(sp),
                              "n_test_graphs": len(tg), "clean_test_acc": round(acc, 4),
                              "test_idx": [int(i) for i in te]})
        meta = {"dataset": "PROTEINS", "model": "gcn", "model_type": "GCN-robustpool",
                "readout": args.readout, "trim": args.trim,
                "readout_desc": f"{args.readout} (trained, proper RS-Pool, trim={args.trim})",
                "n_folds": len(fold_meta), "num_features": ds.num_features,
                "num_classes": ds.num_classes, "seed": seed, "device": device,
                "train_epochs": args.epochs,
                "clean_test_acc_mean": round(mean(accs), 4),
                "clean_test_acc_std": round(pstdev(accs), 4) if len(accs) > 1 else 0.0,
                "folds": fold_meta}
        mp = base / f"meta_rspool_seed{seed}.json"
        mp.write_text(json.dumps(meta, indent=2))
        summary[seed] = (round(mean(accs), 4), round(pstdev(accs), 4) if len(accs) > 1 else 0.0)
        print(f"  -> seed{seed} clean_test_acc = {mean(accs):.4f} +/- "
              f"{(pstdev(accs) if len(accs)>1 else 0):.4f}  (mean-victim ref ~0.72)  "
              f"wrote {mp.name}", flush=True)

    print("=" * 80, flush=True)
    allm = mean([v[0] for v in summary.values()])
    print(f"GAP-1 median-victim clean acc by seed: {summary}", flush=True)
    print(f"GAP-1 OVERALL median-victim clean acc = {allm:.4f} "
          f"(recovers ~0.72? {'YES' if allm >= 0.69 else 'NO'})  [{time.time()-t0:.1f}s]",
          flush=True)


if __name__ == "__main__":
    main()
