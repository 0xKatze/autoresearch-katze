#!/usr/bin/env python3
"""degrade_fast.py — N34 certificate-degradation attack, FAST (uses agnncert_fast.FastCoverEval).
Same objective as attack_agnncert_degrade.py (minimise vote margin, label-preserving, track min M_G)
but ~7x faster via structure-cached batched subgraph evaluation. Validated evaluator (agnncert_fast).
"""
import sys, json, time, argparse
from pathlib import Path
from statistics import mean
from collections import Counter
BD = Path("/workspace/autoresearch-katze-n23/research/branch-diffusion")
sys.path.insert(0, str(BD))
sys.path.insert(0, "/home/devuser/.claude/jobs/f6cffe61/tmp")
import torch
import prepare; prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, collect_test_graphs, DATA_DIR
from torch_geometric.datasets import TUDataset
import attack_agnncert_adaptive as A
from agnncert_fast import FastCoverEval
DEVICE = "cpu"


def M_from_counts(counts):
    vals, idx = torch.sort(counts, descending=True)
    ya, yb = int(idx[0]), int(idx[1])
    I = 1.0 if ya > yb else 0.0
    return ya, int((float(vals[0]) - float(vals[1]) - I) // 2)


def soft_margin(fce, x, fs, y):
    """mean per-subgraph CW margin for class y (lower => subgraphs lean away from y)."""
    lg = fce.logits(x, fs)               # (T, C)
    other = lg.clone(); other[:, y] = -1e9
    return (lg[:, y] - other.max(dim=1).values).mean().item()


def zo_grad(fce, x, fs, y, sigma):
    g = torch.zeros_like(x)
    for i in range(x.size(0)):
        for k in range(x.size(1)):
            e = torch.zeros_like(x); e[i, k] = sigma
            g[i, k] = (soft_margin(fce, x + e, fs, y) - soft_margin(fce, x - e, fs, y)) / (2 * sigma)
    return g


def degrade_one(base, vm, data, nb, fs_spec, steps, restarts, sigma, lr, k, device):
    N = data.num_nodes; fs = A.resolve_fs(fs_spec, N); y = int(data.y.item())
    nfeat = data.x.size(1)
    edges, ncov = A.cover_edges(N, nb, vm.hasher, k=k)
    fce = FastCoverEval(base, data, edges, nb, vm.hasher, vm.T, device)
    fce0 = FastCoverEval(base, data, [], 0, vm.hasher, vm.T, device)  # clean (no injection)
    _, clean_M = M_from_counts(fce0.vote_counts(torch.zeros(0, nfeat), fs))
    best_M = clean_M; flipped = False
    for r in range(restarts):
        x = torch.randn(nb, nfeat, generator=torch.Generator().manual_seed(r * 17 + N), device=device)
        for _ in range(steps):
            pred, M = M_from_counts(fce.vote_counts(x, fs))
            if pred == y:
                best_M = min(best_M, M)
            else:
                flipped = True
            x = x - lr * zo_grad(fce, x, fs, y, sigma)
        pred, M = M_from_counts(fce.vote_counts(x, fs))
        if pred == y:
            best_M = min(best_M, M)
    return clean_M, best_M, len(edges), flipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--cap", type=int, default=5); ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--fs", type=str, default="auto_x2")
    ap.add_argument("--steps", type=int, default=40); ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.1); ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    meta = json.loads((BD / "models_saved" / f"meta_agnncert_official_seed{args.seed}.json").read_text())
    T = meta.get("T", 30)
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    from agnncert_official import AGNNCertVoteModel
    print(f"=== N34 DEGRADE-FAST vs AGNNCert-E | seed{args.seed} folds={args.max_folds} cap{args.cap} "
          f"nb={args.nb} k={args.k} fs={args.fs} steps={args.steps}x{args.restarts} T={T} thr={args.threads} ===",
          flush=True)
    t0 = time.time(); rows = []
    for fi in meta["folds"][:args.max_folds]:
        base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        base.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE, weights_only=True)); base.eval()
        vm = AGNNCertVoteModel(base, meta["num_classes"], T=T).to(DEVICE).eval()
        graphs = collect_test_graphs(vm, ds, fi["test_idx"])[:args.cap]
        for g in graphs:
            g = g.to(DEVICE)
            cM, dM, b, fl = degrade_one(base, vm, g, args.nb, args.fs, args.steps,
                                        args.restarts, args.sigma, args.lr, args.k, DEVICE)
            rows.append({"fold": int(fi["fold_idx"]), "clean_M": cM, "degraded_M": dM,
                         "budget_edges": b, "flipped": fl})
        print(f"  fold{fi['fold_idx']}: {len(rows)} graphs [{time.time()-t0:.0f}s]", flush=True)
    cleanMs = [r["clean_M"] for r in rows]; degMs = [r["degraded_M"] for r in rows]
    to_zero = sum(1 for r in rows if r["degraded_M"] == 0)
    out = {"seed": args.seed, "nb": args.nb, "k": args.k, "fs": args.fs, "n_graphs": len(rows),
           "budget_edges": rows[0]["budget_edges"] if rows else None,
           "clean_M_mean": mean(cleanMs), "degraded_M_mean": mean(degMs), "n_to_M0": to_zero, "rows": rows}
    op = BD / "results" / f"agnncert_degrade_fast_seed{args.seed}_nb{args.nb}_k{args.k}.json"
    op.write_text(json.dumps(out, indent=2))
    print("=" * 70, flush=True)
    print(f"N34 DEGRADE: clean M mean={mean(cleanMs):.2f} -> degraded M mean={mean(degMs):.2f}  "
          f"({to_zero}/{len(rows)} to M=0)  budget={rows[0]['budget_edges']}e  [{time.time()-t0:.0f}s]", flush=True)
    print(f"clean_M dist:    {dict(sorted(Counter(cleanMs).items()))}", flush=True)
    print(f"degraded_M dist: {dict(sorted(Counter(degMs).items()))}", flush=True)
    print(f"-> {op}", flush=True)


if __name__ == "__main__":
    main()
