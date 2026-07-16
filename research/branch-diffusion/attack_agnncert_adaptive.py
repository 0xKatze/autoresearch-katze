#!/usr/bin/env python3
"""attack_agnncert_adaptive.py — N33 adaptive attack vs official AGNNCert-E (edge-division vote).

Naive injection is defeated by AGNNCert (OM2C 6.7%, dense 0%) because injected edges scatter
across T subgraphs and most subgraphs stay clean. This ADAPTIVE attack exploits the KNOWN
deterministic MD5 edge-hash (white-box-on-defense, black-box-on-model):

  1. COVER-ROUTE: route injected edges to cover a MAJORITY (>T/2) of the T subgraphs, one
     targeted (injected_node, victim) edge per subgraph (cover_edges).
  2. SOFT objective: the attacker knows the subgraph division, so it queries the base model on
     each subgraph and minimises the SUM of per-subgraph CW margins (smooth) via ZO-CGE — the
     hard majority vote is a step function (flat ZO gradient), so we optimise its soft surrogate.
  3. Flip = hard AGNNCert majority vote flips (checked on the real vote model).

Standalone eval on the official AGNNCert victim. PROTEINS has 3 feats -> ZO is cheap.
Usage: micromamba run -n graph_adversarial python attack_agnncert_adaptive.py \
         [--seed 42] [--max-folds 1] [--cap 10] [--nb 10] [--fs auto_x2] [--steps 40] [--restarts 3]
"""
import sys, json, math, time, argparse
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, collect_test_graphs
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from prepare import DATA_DIR
from agnncert_official import AGNNCertVoteModel, HashAgentOfficial

DEVICE = "cpu"


def resolve_fs(fs, n):
    if fs == "auto":    return math.sqrt(n)
    if fs == "auto_x2": return 2.0 * math.sqrt(n)
    return float(fs)


def cover_edges(N, nb, hasher, k=1):
    """Route up to k (injected_offset i, victim v) edges into EACH subgraph, covering as many
    of T as possible. k=1 reproduces the original one-edge-per-subgraph cover. For k>1 we prefer
    k DISTINCT injected nodes per subgraph so each subgraph gets more optimisable injected mass
    (extra feature vectors), the per-subgraph flip-strength lever (plateau finding)."""
    V = N + nb
    covered = {}     # s -> list of (i, v)
    used_i = {}      # s -> set of injected nodes already routed into this subgraph
    for i in range(nb):
        j = N + i
        for v in range(N):
            lo, hi = (v, j) if v < j else (j, v)
            s = hasher.hash_edge(V, lo, hi)
            lst = covered.setdefault(s, [])
            seen = used_i.setdefault(s, set())
            if len(lst) < k and i not in seen:
                lst.append((i, v)); seen.add(i)
    edges = [e for lst in covered.values() for e in lst]
    return edges, len(covered)


def build_cover_graph(data, x, edges, fs):
    device = data.x.device
    N = data.num_nodes
    X = torch.cat([data.x, x * fs], dim=0)
    src, dst = [], []
    for (i, v) in edges:
        j = N + i
        src += [j, v]; dst += [v, j]
    if src:
        ne = torch.tensor([src, dst], dtype=torch.long, device=device)
        ei = torch.cat([data.edge_index, ne], dim=1)
    else:
        ei = data.edge_index
    return Data(x=X, edge_index=ei, y=getattr(data, "y", None))


def soft_objective(base, hasher, data, x, edges, fs, y_true, nc, device):
    """Sum of per-subgraph CW margins (f_ytrue - max_other) over subgraphs that CONTAIN an
    injected node. Lower = more subgraphs lean wrong. The attacker knows the division."""
    g = build_cover_graph(data, x, edges, fs)
    subs = hasher.generate_graph_subgraphs(g.edge_index, g.x, g.y)
    N = data.num_nodes
    total = torch.tensor(0.0, device=device); cnt = 0
    for s in subs:
        if s.edge_index is None:
            continue
        d = Data(x=s.x.to(device), edge_index=s.edge_index.to(device))
        d.batch = torch.zeros(s.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = base(d)[0]
        other = logits.clone(); other[y_true] = -1e9
        total = total + (logits[y_true] - other.max())   # CW margin
        cnt += 1
    return (total / max(cnt, 1)).item()


def zo_grad(base, hasher, data, x, edges, fs, y, nc, sigma, device):
    g = torch.zeros_like(x)
    for i in range(x.size(0)):
        for k in range(x.size(1)):
            e = torch.zeros_like(x); e[i, k] = sigma
            fp = soft_objective(base, hasher, data, x + e, edges, fs, y, nc, device)
            fm = soft_objective(base, hasher, data, x - e, edges, fs, y, nc, device)
            g[i, k] = (fp - fm) / (2 * sigma)
    return g


def hard_flip(vote_model, data, x, edges, fs):
    g = build_cover_graph(data, x, edges, fs)
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=g.x.device)
    with torch.no_grad():
        pred = int(vote_model(g)[0].argmax())
    return pred != int(data.y.item())


def adaptive_attack(base, vote_model, data, nb, fs_spec, steps, restarts, sigma, lr, device, k=1):
    N = data.num_nodes
    nc = vote_model.num_classes
    fs = resolve_fs(fs_spec, N)
    edges, n_cov = cover_edges(N, nb, vote_model.hasher, k=k)
    y = int(data.y.item()); nfeat = data.x.size(1)
    for r in range(restarts):
        gen = torch.Generator().manual_seed(r * 17 + N)
        x = torch.randn(nb, nfeat, generator=gen, device=device)
        for _ in range(steps):
            if hard_flip(vote_model, data, x, edges, fs):
                return True, n_cov
            grad = zo_grad(base, vote_model.hasher, data, x, edges, fs, y, nc, sigma, device)
            x = x - lr * grad
        if hard_flip(vote_model, data, x, edges, fs):
            return True, n_cov
    return False, n_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=1)
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--fs", type=str, default="auto_x2")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--k", type=int, default=1, help="edges (distinct injected nodes) per subgraph")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()

    mp = HERE / "models_saved" / f"meta_agnncert_official_seed{args.seed}.json"
    meta = json.loads(mp.read_text())
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    folds = meta["folds"][:args.max_folds]
    print(f"=== N33 ADAPTIVE attack vs official AGNNCert | seed{args.seed} folds={args.max_folds} "
          f"nb={args.nb} k={args.k} fs={args.fs} steps={args.steps}x{args.restarts} T={meta.get('T',30)} ===", flush=True)
    t0 = time.time(); flips = 0; n = 0; covs = []
    for fi in folds:
        base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        base.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE, weights_only=True))
        base.eval()
        vote_model = AGNNCertVoteModel(base, meta["num_classes"], T=meta.get("T", 30)).to(DEVICE).eval()
        graphs = collect_test_graphs(vote_model, ds, fi["test_idx"])[:args.cap]
        for g in graphs:
            g = g.to(DEVICE)
            fl, ncov = adaptive_attack(base, vote_model, g, args.nb, args.fs,
                                       args.steps, args.restarts, args.sigma, args.lr, DEVICE,
                                       k=args.k)
            flips += int(fl); n += 1; covs.append(ncov)
        print(f"  fold{fi['fold_idx']}: ASR={flips/max(n,1)*100:.1f}% ({flips}/{n}) "
              f"avg_subgraphs_covered={mean(covs):.1f}/{meta.get('T',30)}", flush=True)
    asr = flips / max(n, 1) * 100
    print("=" * 70, flush=True)
    print(f"N33 ADAPTIVE OM2C-cover vs AGNNCert: ASR = {asr:.1f}%  ({flips}/{n})  "
          f"cover~{mean(covs):.0f}/{meta.get('T',30)}  [vs naive OM2C 6.7%]  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
