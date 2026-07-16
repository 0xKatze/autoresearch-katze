#!/usr/bin/env python3
"""attack_agnncert_degrade.py — N34: CERTIFICATE-DEGRADATION attack on official AGNNCert-E.

Goal (NOT a label flip): find a budget-b, LABEL-PRESERVING injection that minimises the vote
margin (c_ya - c_yb), driving the certified radius M_G = floor((c_ya-c_yb-I)/2) toward 0 while the
AGNNCert majority vote still predicts the (correct) class y. Exposes that the published certified
accuracy (measured on clean inputs) is optimistic under an adaptive adversary.

Reuses the N33 ZO machinery from attack_agnncert_adaptive (cover_edges / build_cover_graph /
soft_objective / zo_grad): minimising the mean per-subgraph CW margin for y pushes subgraphs away
from y, lowering c_ya — we just STOP-track the lowest M reached while the label is preserved instead
of pushing all the way to a flip.

Usage: python attack_agnncert_degrade.py [--seed 42] [--max-folds 1] [--cap 5] [--nb 10] [--k 1]
        [--fs auto_x2] [--steps 40] [--restarts 3]
"""
import sys, json, time, argparse
from pathlib import Path
from statistics import mean
from collections import Counter

BD = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
sys.path.insert(0, str(BD))
import torch
import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, collect_test_graphs, DATA_DIR
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from agnncert_official import AGNNCertVoteModel
import attack_agnncert_adaptive as A   # cover_edges, build_cover_graph, zo_grad, resolve_fs

DEVICE = "cpu"


def M_from_counts(counts):
    vals, idx = torch.sort(counts, descending=True)
    c_ya = float(vals[0]); c_yb = float(vals[1])
    ya = int(idx[0]); yb = int(idx[1])
    I = 1.0 if ya > yb else 0.0
    return ya, c_ya, c_yb, int((c_ya - c_yb - I) // 2)


def vote_counts(vm, data, x, edges, fs, device):
    if x is None or len(edges) == 0:
        g = data
    else:
        g = A.build_cover_graph(data, x, edges, fs)
    gg = Data(x=g.x, edge_index=g.edge_index)
    gg.batch = torch.zeros(gg.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        frac = vm(gg)[0]
    return torch.round(frac * vm.T)


def degrade_one(base, vm, data, nb, fs_spec, steps, restarts, sigma, lr, device, k=1):
    N = data.num_nodes; nc = vm.num_classes; fs = A.resolve_fs(fs_spec, N)
    hasher = vm.hasher; y = int(data.y.item()); nfeat = data.x.size(1)
    clean_counts = vote_counts(vm, data, None, [], fs, device)
    _, _, _, clean_M = M_from_counts(clean_counts)
    edges, ncov = A.cover_edges(N, nb, hasher, k=k)
    best_M = clean_M; flipped_ever = False
    for r in range(restarts):
        gen = torch.Generator().manual_seed(r * 17 + N)
        x = torch.randn(nb, nfeat, generator=gen, device=device)
        for _ in range(steps):
            counts = vote_counts(vm, data, x, edges, fs, device)
            pred, _, _, M = M_from_counts(counts)
            if pred == y:
                best_M = min(best_M, M)
            else:
                flipped_ever = True
            grad = A.zo_grad(base, hasher, data, x, edges, fs, y, nc, sigma, device)
            x = x - lr * grad
        counts = vote_counts(vm, data, x, edges, fs, device)
        pred, _, _, M = M_from_counts(counts)
        if pred == y:
            best_M = min(best_M, M)
    return clean_M, best_M, len(edges), flipped_ever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--fs", type=str, default="auto_x2")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()

    meta = json.loads((BD / "models_saved" / f"meta_agnncert_official_seed{args.seed}.json").read_text())
    T = int(meta.get("T", 30))
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    print(f"=== N34 DEGRADE vs AGNNCert-E | seed{args.seed} folds={args.max_folds} cap{args.cap} "
          f"nb={args.nb} k={args.k} fs={args.fs} steps={args.steps}x{args.restarts} T={T} ===", flush=True)
    t0 = time.time(); rows = []
    for fi in meta["folds"][:args.max_folds]:
        base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        base.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE, weights_only=True))
        base.eval()
        vm = AGNNCertVoteModel(base, meta["num_classes"], T=T).to(DEVICE).eval()
        graphs = collect_test_graphs(vm, ds, fi["test_idx"])[:args.cap]
        for g in graphs:
            g = g.to(DEVICE)
            cM, dM, b, fl = degrade_one(base, vm, g, args.nb, args.fs, args.steps,
                                        args.restarts, args.sigma, args.lr, DEVICE, k=args.k)
            rows.append({"fold": int(fi["fold_idx"]), "clean_M": cM, "degraded_M": dM,
                         "budget_edges": b, "flipped_ever": fl})
        print(f"  fold{fi['fold_idx']}: {len(rows)} graphs [{time.time()-t0:.0f}s]", flush=True)

    cleanMs = [r["clean_M"] for r in rows]; degMs = [r["degraded_M"] for r in rows]
    to_zero = sum(1 for r in rows if r["degraded_M"] == 0)
    out = {"seed": args.seed, "dataset": meta["dataset"], "T": T, "nb": args.nb, "k": args.k,
           "fs": args.fs, "n_graphs": len(rows), "budget_edges": rows[0]["budget_edges"] if rows else None,
           "clean_M_mean": mean(cleanMs) if rows else None,
           "degraded_M_mean": mean(degMs) if rows else None,
           "n_driven_to_M0": to_zero, "rows": rows}
    op = BD / "results" / f"agnncert_degrade_seed{args.seed}_nb{args.nb}_k{args.k}_smoke.json"
    op.write_text(json.dumps(out, indent=2))
    print("=" * 70, flush=True)
    print(f"N34 DEGRADE: clean M mean={mean(cleanMs):.2f} -> degraded M mean={mean(degMs):.2f}  "
          f"({to_zero}/{len(rows)} driven to M=0)  budget={rows[0]['budget_edges']} edges  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"clean_M dist:    {dict(sorted(Counter(cleanMs).items()))}", flush=True)
    print(f"degraded_M dist: {dict(sorted(Counter(degMs).items()))}", flush=True)
    print(f"-> {op}", flush=True)


if __name__ == "__main__":
    main()
