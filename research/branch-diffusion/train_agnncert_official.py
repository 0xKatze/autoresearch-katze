#!/usr/bin/env python3
"""train_agnncert_official.py — TRAIN official AGNNCert-E (subgraph-augmented) victims.

Official protocol (RobustGraphClassifier): subgraph augmentation — for each train graph,
add its T edge-hash subgraphs + the original; train the base GCN to classify all of them;
validate via the VOTE wrapper (AGNNCertVoteModel). Uses OUR GCN arch + our folds
(prepare.get_folds) for fair comparison with the battery. Mini-batch (faster than the
official full-batch loop; the METHOD = subgraph augmentation is what matters).

Usage: micromamba run -n graph_adversarial python train_agnncert_official.py \
         [--seeds 42] [--n-folds 10] [--epochs 200] [--T 30] [--val-every 25]
         [--device cuda] [--one-fold-probe] [--max-folds N]
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
from prepare import HIDDEN_DIM, GCN_GraphClassification, collect_test_graphs
from agnncert_official import HashAgentOfficial, AGNNCertVoteModel


def build_augmented(graphs, hasher):
    """{T subgraphs + original} per graph, each labelled with the graph's y."""
    aug = []
    for g in graphs:
        for s in hasher.generate_graph_subgraphs(g.edge_index, g.x, g.y):
            ei = s.edge_index if s.edge_index is not None else torch.zeros((2, 0), dtype=torch.int64)
            aug.append(type(g)(x=s.x, edge_index=ei, y=g.y))
        aug.append(type(g)(x=g.x, edge_index=g.edge_index, y=g.y))
    return aug


def vote_acc(vote_model, ds, idx, device):
    vote_model.eval(); c = t = 0
    for data in Subset(ds, idx):
        data = data.to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = int(vote_model(data)[0].argmax())
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def train_one(ds, tr_idx, va_idx, seed, fold_idx, epochs, T, val_every, device, save_path):
    hasher = HashAgentOfficial(T=T)
    t_aug = time.time()
    aug = build_augmented([ds[i] for i in tr_idx], hasher)
    aug_s = time.time() - t_aug
    loader = DataLoader(aug, batch_size=64, shuffle=True)
    torch.manual_seed(seed + fold_idx)
    base = GCN_GraphClassification(ds.num_features, ds.num_classes, HIDDEN_DIM).to(device)
    vote_model = AGNNCertVoteModel(base, ds.num_classes, T=T).to(device)
    opt = torch.optim.Adam(base.parameters(), lr=1e-3)
    best_va = 0.0
    for ep in range(epochs):
        base.train()
        for b in loader:
            b = b.to(device)
            loss = F.cross_entropy(base(b), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            va = vote_acc(vote_model, ds, va_idx, device)
            if va >= best_va:
                best_va = va
                torch.save(base.state_dict(), save_path)
    if not save_path.exists():
        torch.save(base.state_dict(), save_path)
    base.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return vote_model.eval(), best_va, len(aug), aug_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--one-fold-probe", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA unavailable")
    prepare.DEVICE = device
    seeds = [int(s) for s in args.seeds.split(",")]
    base_dir = Path(prepare.MODELS_DIR)

    print(f"=== TRAIN official AGNNCert-E (subgraph-aug) | PROTEINS GCN | seeds={seeds} "
          f"folds={args.n_folds} epochs={args.epochs} T={args.T} device={device} ===", flush=True)
    t0 = time.time(); summary = {}
    for seed in seeds:
        prepare.SEED = seed; prepare.N_FOLDS = args.n_folds
        ds = prepare.prepare_dataset()
        seed_dir = base_dir / f"agnncert_official_seed{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        folds = list(prepare.get_folds(ds))
        if args.one_fold_probe: folds = folds[:1]
        elif args.max_folds: folds = folds[:args.max_folds]
        fold_meta, accs = [], []
        for fold_idx, tr, va, te in folds:
            sp = seed_dir / f"PROTEINS_gcn_fold{fold_idx}.pt"
            tf = time.time()
            vm, best_va, n_aug, aug_s = train_one(ds, tr, va, seed, fold_idx, args.epochs,
                                                  args.T, args.val_every, device, sp)
            acc = vote_acc(vm, ds, te, device)
            tg = collect_test_graphs(vm, ds, te)
            accs.append(acc)
            print(f"  seed{seed} fold{fold_idx}: vote_val={best_va:.4f} vote_test={acc:.4f} "
                  f"n_aug={n_aug} n_attackable={len(tg)} (aug {aug_s:.0f}s, total {time.time()-tf:.0f}s)",
                  flush=True)
            fold_meta.append({"fold_idx": fold_idx, "model_path": str(sp),
                              "n_test_graphs": len(tg), "clean_test_acc": round(acc, 4),
                              "test_idx": [int(i) for i in te]})
        meta = {"dataset": "PROTEINS", "model": "gcn", "model_type": "GCN-agnncert-official",
                "defense": "agnncert_official", "T": args.T, "train_epochs": args.epochs,
                "readout_desc": f"official AGNNCert-E subgraph-aug + vote (T={args.T})",
                "n_folds": len(fold_meta), "num_features": ds.num_features,
                "num_classes": ds.num_classes, "seed": seed, "device": device,
                "clean_test_acc_mean": round(mean(accs), 4),
                "clean_test_acc_std": round(pstdev(accs), 4) if len(accs) > 1 else 0.0,
                "folds": fold_meta}
        mp = base_dir / f"meta_agnncert_official_seed{seed}.json"
        mp.write_text(json.dumps(meta, indent=2))
        summary[seed] = round(mean(accs), 4)
        print(f"  -> seed{seed} vote clean acc = {mean(accs):.4f} (mean-victim ~0.72) wrote {mp.name}", flush=True)
    print("=" * 80, flush=True)
    print(f"AGNNCert-official clean acc by seed: {summary}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
