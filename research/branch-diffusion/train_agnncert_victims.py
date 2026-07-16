#!/usr/bin/env python3
"""train_agnncert_victims.py — FAIR AGNNCert-E base classifier (subgraph-RETRAINED).

The make-or-break AGNNCert smoke wrapped the MEAN-trained GCN in the AGNNCert-E voting
classifier WITHOUT retraining -> clean acc collapses to 0.5625 (< PROTEINS majority 0.596),
a DEGENERATE victim. Per AGNNCert (Li & Wang, USENIX Security 2025, Sec.4.1 "GNN classifiers
and AGNNCert training", p.9):

    "AGNNCert trains the GNN classifier using BOTH the training graphs AND their generated
     subgraphs, whose labels are same as the training graphs'."

i.e. the base classifier must be trained on the SUBGRAPH-DIVIDED distribution so it classifies
fragmented (mostly edge-sparse) subgraphs well; the voting classifier then recovers the paper's
PROTEINS/GCN clean acc = 0.75 (Table 2, AGNNCert-E, p.11). This script implements that retrain.

This is a NEW additive script: it does NOT modify prepare.py / defense_agnncert.py / any attack.
It REUSES verbatim:
  * prepare.GCN_GraphClassification  (frozen 3-layer GCN, hidden 64, BatchNorm, dropout 0.5),
  * prepare.get_folds                (the FROZEN StratifiedKFold split per victim seed),
  * defense_agnncert._edge_bucket    (the EXACT MD5 edge->subgraph hash used at test time),
  * defense_agnncert.AGNNCertModel   (voting classifier, for best-checkpoint val selection).

Augmentation is STOCHASTIC and faithful: each epoch, for every training graph we add (a) the
original graph and (b) ONE uniformly-random edge-divided subgraph (random bucket in [0,T)). A
uniform random bucket reproduces the natural subgraph frequency the vote sees at test time (at
T=300 on PROTEINS, ~227/300 buckets are empty, so the base mostly sees edge-less subgraphs —
exactly the test-time vote distribution). Over the training epochs the base sees the full
subgraph distribution with correct frequencies, at the SAME per-epoch cost as normal training.

Best checkpoint = highest VOTING (AGNNCert) validation accuracy (the deployed metric), checked
every --val-every epochs. Output: models_saved/pg_floor_seed{S}/PROTEINS_gcn_agnncertT{T}_fold{F}.pt
+ meta_agnncertT{T}_seed{S}.json (mirrors meta_pg_seed{S}.json shape so the fair eval can load it).

Usage:
  micromamba run -n graph_adversarial python3 train_agnncert_victims.py \
      --seeds 42,7,1027 --folds 10 --T 300 --epochs 200 --val-every 20 --device cuda
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_DEV = os.environ.get("AGC_DEVICE", "")
if _DEV.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("AGC_GPU", "0")
elif _DEV == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    # CPU mode: torch.cuda.is_available() intermittently SEGFAULTS (exit 139) at the CUDA
    # driver probe on this shared RTX-5080 while pid 64005 (the degenerate-wrapper full
    # eval) saturates the GPU. prepare.py calls it at import (line 31). Short-circuit the
    # probe so the CPU retrain imports deterministically (verified intermittent: 2/5 crash).
    torch.cuda.is_available = lambda: False
    # The autograd C++ engine's thread pool SEGFAULTS intermittently under the heavy CPU
    # contention from pid 64005; single-threaded execution removes the thread-pool race.
    torch.set_num_threads(1)
import torch.nn.functional as Fn  # noqa: E402
from torch_geometric.datasets import TUDataset  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

import prepare  # noqa: E402
from prepare import GCN_GraphClassification, DATA_DIR  # noqa: E402
from defense_agnncert import _edge_bucket, AGNNCertModel  # noqa: E402


def precompute_buckets(data, T):
    """Per-edge subgraph bucket for one graph, using the EXACT test-time hash
    (defense_agnncert._edge_bucket on LOCAL node indices). Returns LongTensor[E]."""
    ei = data.edge_index
    E = ei.size(1)
    if E == 0:
        return torch.empty(0, dtype=torch.long)
    src = ei[0].tolist(); dst = ei[1].tolist()
    return torch.tensor([_edge_bucket(u, v, T) for u, v in zip(src, dst)], dtype=torch.long)


def random_subgraph(data, buckets, T):
    """A uniformly-random edge-divided subgraph G_b: all nodes + features, only the
    edges in a random bucket b (most buckets empty at large T -> edge-less subgraph)."""
    b = random.randrange(T)
    if buckets.numel() > 0:
        keep = (buckets == b)
        ei = data.edge_index[:, keep]
    else:
        ei = data.edge_index[:, :0]
    return Data(x=data.x, edge_index=ei.contiguous(), y=data.y)


def voting_val_acc(base, val_graphs, T, device):
    """AGNNCert voting accuracy on the validation graphs (the deployed metric)."""
    v = AGNNCertModel(base, T=T)
    c = 0
    for d in val_graphs:
        dd = d.clone()
        dd.batch = torch.zeros(dd.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = int(v(dd).argmax(dim=-1).item())
        c += int(pred == int(d.y.item()))
    return c / max(len(val_graphs), 1)


def train_one_fold(ds, train_idx, val_idx, fold_idx, T, epochs, val_every, device, seed, save_path):
    g_train = []
    for i in train_idx:
        d = ds[i].to(device)
        g_train.append((d, precompute_buckets(d, T)))
    g_val = [ds[i].to(device) for i in val_idx]

    torch.manual_seed(seed + fold_idx)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed + fold_idx)
    random.seed(seed * 1000 + fold_idx)

    model = GCN_GraphClassification(ds.num_features, ds.num_classes, prepare.HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_va = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train()
        aug = []
        for d, bk in g_train:
            aug.append(Data(x=d.x, edge_index=d.edge_index, y=d.y))   # original graph
            aug.append(random_subgraph(d, bk, T))                     # one random subgraph
        loader = DataLoader(aug, batch_size=32, shuffle=True)
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = Fn.cross_entropy(out, batch.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            model.eval()
            va = voting_val_acc(model, g_val, T, device)
            if va > best_va:
                best_va = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model, best_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-folds", type=int, default=None, help="cap folds (smoke/CPU verify)")
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but torch.cuda.is_available()==False (no silent CPU fallback)")
    prepare.DEVICE = device
    seeds = [int(s) for s in args.seeds.split(",")]
    T = args.T
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"=== FAIR AGNNCert RETRAIN | PROTEINS GCN | seeds={seeds} | folds={args.folds} "
          f"| T={T} | epochs={args.epochs} | device={device} | {stamp} ===", flush=True)

    for seed in seeds:
        prepare.SEED = seed; prepare.N_FOLDS = args.folds
        ds = TUDataset(root=str(DATA_DIR / prepare.DATASET_NAME), name=prepare.DATASET_NAME)
        seed_dir = Path(prepare.MODELS_DIR) / f"pg_floor_seed{seed}"
        folds_meta = []
        folds = list(prepare.get_folds(ds))
        if args.max_folds is not None:
            folds = folds[:args.max_folds]
        for fold_idx, tr, va, te in folds:
            t0 = time.time()
            save_path = seed_dir / f"{prepare.DATASET_NAME}_{prepare.MODEL_TYPE}_agnncertT{T}_fold{fold_idx}.pt"
            model, best_va = train_one_fold(ds, tr, va, fold_idx, T, args.epochs,
                                            args.val_every, device, seed, save_path)
            # voting clean TEST acc (the deployed AGNNCert clean accuracy)
            g_test = [ds[i].to(device) for i in te]
            test_va = voting_val_acc(model, g_test, T, device)
            dt = time.time() - t0
            print(f"  seed{seed} fold{fold_idx}: voting val_acc={best_va:.4f} "
                  f"voting TEST clean acc={test_va:.4f}  ({dt:.1f}s) -> {save_path.name}", flush=True)
            folds_meta.append({
                "fold_idx": fold_idx,
                "model_path": str(save_path),
                "agnncert_voting_clean_test_acc": round(test_va, 4),
                "agnncert_voting_best_val_acc": round(best_va, 4),
                "test_idx": [int(i) for i in te],
            })
        mean_clean = mean(f["agnncert_voting_clean_test_acc"] for f in folds_meta) if folds_meta else 0.0
        meta = {
            "dataset": prepare.DATASET_NAME, "model_type": "GCN",
            "defense": "AGNNCert-E (retrained, subgraph-augmented)", "T": T,
            "recipe": "AGNNCert Sec.4.1 (p.9): train base GNN on training graphs + their "
                      "edge-divided subgraphs (same labels). Stochastic per-epoch augmentation: "
                      "original + 1 uniform-random subgraph per graph.",
            "n_folds": len(folds_meta), "num_features": ds.num_features,
            "num_classes": ds.num_classes, "seed": seed, "device": device,
            "train_epochs": args.epochs, "val_every": args.val_every,
            "agnncert_voting_clean_test_acc_mean": round(mean_clean, 4),
            "paper_target_clean_acc_PROTEINS_GCN_retrained": 0.75,
            "folds": folds_meta,
        }
        out = Path(prepare.MODELS_DIR) / f"meta_agnncertT{T}_seed{seed}.json"
        out.write_text(json.dumps(meta, indent=2))
        print(f"  -> wrote {out}  (voting clean TEST acc mean={mean_clean:.4f}; paper target 0.75)",
              flush=True)


if __name__ == "__main__":
    main()
