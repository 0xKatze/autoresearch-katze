#!/usr/bin/env python3
"""gen_victims_gcn10_driver.py — THIN DRIVER (no model/attack/defense code modified).

Round-4 disentangle: trains ENZYMES-continuous GCN victims at the SAME rigor as the GIN
10-fold victims (gen_victims_gin10_driver.py) so dataset-vs-backbone can be separated with a
matched protocol. The stock GCN continuous trainer prepare_cont.py is single-seed (global SEED)
AND its meta stores only test_idx (no train_idx, which the C3 floor harness needs). This driver
reuses prepare_cont's data layer + the GCN model class VERBATIM, and mirrors the VALIDATED
prepare_cont_gin.py per-seed training loop (stratified 10-fold, seeded per victim seed, save-best
val checkpoint, train_idx stored). Output schema is byte-for-byte compatible with the GIN floor
runner's load_meta (meta_cont_gcn_{DS}_seed{seed}.json in models_cont_10fold/).

NON-DESTRUCTIVE: writes only models_cont_10fold/ (the validated single-seed 5-fold GCN victims in
models_cont/ stay untouched). Device = cuda (matches the cont/GIN victims; CPU fallback if no GPU).

Usage: micromamba run -n graph_adversarial python gen_victims_gcn10_driver.py \
         [--seeds 42,7,1027] [--epochs 200] [--n-folds 10] [--device cuda] \
         [--models-dir models_cont_10fold] [--dataset ENZYMES]
"""
import sys, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from prepare import HIDDEN_DIM, GCN_GraphClassification
from prepare_cont import load_normalized_graphs, clean_test_acc

HERE = Path(__file__).resolve().parent


def get_folds(labels, seed, n_folds):
    """Stratified k-fold (test fold f, val fold f+1, rest train), seeded per victim seed.
    Identical to prepare_cont_gin.get_folds (matched split protocol across backbones)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = list(range(len(labels)))
    folds = list(skf.split(idx, labels))
    for f in range(n_folds):
        test_idx = list(folds[f][1])
        val_idx = list(folds[(f + 1) % n_folds][1])
        tv = set(test_idx) | set(val_idx)
        train_idx = [i for i in idx if i not in tv]
        yield f, train_idx, val_idx, test_idx


def train_victim(graphs, num_features, num_classes, train_idx, val_idx, save_path,
                 seed, epochs, device):
    """Mirror prepare_cont_gin.train_victim exactly; only the backbone is GCN."""
    tr = DataLoader(Subset(graphs, train_idx), batch_size=32, shuffle=True)
    va = DataLoader(Subset(graphs, val_idx), batch_size=32)
    torch.manual_seed(seed)
    model = GCN_GraphClassification(num_features, num_classes, HIDDEN_DIM).to(device)
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
    ap.add_argument("--dataset", type=str, default="ENZYMES")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--models-dir", type=str, default="models_cont_10fold")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"!! requested device={args.device} but CUDA unavailable -> using CPU", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = (HERE / args.models_dir) if not Path(args.models_dir).is_absolute() else Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # z-score over the FULL dataset (seed-independent) — identical normalization to GIN-cont victims
    graphs, labels, mean, std, n_classes = load_normalized_graphs(args.dataset)
    n_feat = graphs[0].x.size(1)
    print(f"=== gen_victims_gcn10 {args.dataset} | n={len(graphs)} feat={n_feat} (z-scored) "
          f"classes={n_classes} | GCN | seeds={seeds} folds={args.n_folds} epochs={args.epochs} "
          f"device={device} | out={out_dir} ===", flush=True)

    for seed in seeds:
        fold_meta, accs = [], []
        for f, tr, va, te in get_folds(labels, seed, args.n_folds):
            sp = out_dir / f"{args.dataset}_gcn_seed{seed}_fold{f}.pt"
            model, best_va = train_victim(graphs, n_feat, n_classes, tr, va, sp,
                                          seed, args.epochs, device)
            acc = clean_test_acc(model, graphs, te)
            accs.append(acc)
            n_corr = int(round(acc * len(te)))
            print(f"  seed{seed} fold{f}: val_acc={best_va:.4f} clean_test_acc={acc:.4f} "
                  f"({n_corr}/{len(te)} attackable)", flush=True)
            fold_meta.append({"fold_idx": f, "model_path": str(sp),
                              "train_idx": [int(i) for i in tr],
                              "test_idx": [int(i) for i in te],
                              "clean_test_acc": acc, "n_attackable": n_corr})

        meta = {
            "dataset": args.dataset, "model": "gcn", "continuous": True, "use_node_attr": True,
            "zscore": True, "feat_mean": mean.squeeze(0).tolist(), "feat_std": std.squeeze(0).tolist(),
            "num_features": n_feat, "num_classes": n_classes, "seed": seed, "n_folds": args.n_folds,
            "epochs": args.epochs, "device": device, "folds": fold_meta,
            "clean_test_acc_mean": float(np.mean(accs)), "clean_test_acc_std": float(np.std(accs)),
        }
        mp = out_dir / f"meta_cont_gcn_{args.dataset}_seed{seed}.json"
        mp.write_text(json.dumps(meta, indent=2))
        gate = "PASS >=0.40" if np.mean(accs) >= 0.40 else "WEAK <0.40 (report as caveat)"
        print(f"  -> seed{seed} clean_test_acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}  ({gate})",
              flush=True)
        print(f"  -> Saved {mp}", flush=True)


if __name__ == "__main__":
    main()
