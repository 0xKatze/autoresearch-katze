#!/usr/bin/env python3
"""
calibrate_specgdi_v2.py — Calibrate τ_spec_composite and clean_community_density
for the v2 (K-node joint trojan) attack.

Procedure:
  For each dataset (PROTEINS, ENZYMES):
    1. clean_community_density = mean intra-community edge density over N_cal=200
       clean training/test graphs, using networkx greedy modularity communities.
       (Used to set the discretization density on A_VV.)
    2. τ_spec_composite = 99th-pct of S_spec_composite when randomly inserting a
       K-node trojan (K=3) into N_cal=200 clean graphs. Each trojan has random
       inter-injected edges (sampled at the community-density rate) and random
       cross-edges (uniformly sampled, ~min(N, max_cross)).

Writes: specgdi_v2_thresholds.json with shape
   { "PROTEINS": {"tau_spec_composite":..., "clean_community_density":..., ...},
     "ENZYMES":  {...},
     "default":  {...} }

Usage:
  micromamba run -n graph_adversarial python calibrate_specgdi_v2.py
"""
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

import attack as _A
from prepare import MODELS_DIR, DEVICE, GCN_GraphClassification, DATA_DIR, collect_test_graphs
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from torch_geometric.datasets import TUDataset

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
except Exception:
    nx = None

N_CAL = 200
K_TROJAN = 3
QUANTILE = 0.99
K_TOP = 20
BETA = 1.0
SEED = 23
MAX_CROSS_PER_NODE_FRAC = 1.0  # full-edge cross at calibration (worst case)

HERE = Path(__file__).parent
OUT = HERE / "specgdi_v2_thresholds.json"


def _to_nx(g):
    """Convert a PyG graph to NetworkX undirected (no features)."""
    G = nx.Graph()
    n = g.num_nodes
    G.add_nodes_from(range(n))
    ei = g.edge_index.cpu().numpy()
    for s, d in zip(ei[0].tolist(), ei[1].tolist()):
        if s != d:
            G.add_edge(int(s), int(d))
    return G


def _community_density(g):
    """Mean intra-community edge density across communities of g (greedy modularity).

    For each community C with |C| ≥ 2:
        density = (#intra edges) / (|C|*(|C|-1)/2)
    Returns nan if no community has size >= 2. Singletons skipped.
    """
    if nx is None:
        return float("nan")
    G = _to_nx(g)
    if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
        return float("nan")
    try:
        comms = list(greedy_modularity_communities(G))
    except Exception:
        return float("nan")
    dens = []
    for C in comms:
        C = set(C)
        sz = len(C)
        if sz < 2:
            continue
        edges_in = 0
        for u in C:
            for v in G[u]:
                if v in C and v > u:
                    edges_in += 1
        max_e = sz * (sz - 1) / 2
        dens.append(edges_in / max_e)
    if not dens:
        return float("nan")
    return float(sum(dens) / len(dens))


def _sample_trojan(g, K, density_VV, rng, max_cross_per_node_frac):
    """Sample a random K-node trojan: A_VV ~ Bernoulli(density_VV) (symmetric, zero diag);
    A_VU ~ each row keeps top-r random victims with r ~ Uniform(1, max_cross)."""
    n = g.num_nodes
    A_VV = torch.zeros(K, K, device=g.x.device)
    for i in range(K):
        for j in range(i + 1, K):
            if rng.random() < density_VV:
                A_VV[i, j] = 1.0
                A_VV[j, i] = 1.0
    A_VU = torch.zeros(K, n, device=g.x.device)
    max_cross = max(1, int(max_cross_per_node_frac * n))
    for i in range(K):
        r = rng.randint(1, max_cross)
        targets = rng.sample(range(n), min(r, n))
        for v in targets:
            A_VU[i, v] = 1.0
    return A_VV, A_VU


def _s_spec_composite(g, A_VV, A_VU, K, k_top, beta):
    """Compute S_spec_composite via exact eigh on the (n+K) composite graph."""
    n = g.num_nodes
    device = g.x.device
    A_VV0 = torch.zeros(K, K, device=device)
    A_VU0 = torch.zeros(K, n, device=device)
    lam_c, _ = _A.composite_laplacian_spectrum(g.edge_index, n, A_VV0, A_VU0, k_top, device)
    lam_p, _ = _A.composite_laplacian_spectrum(g.edge_index, n, A_VV, A_VU, k_top, device)
    L = min(lam_p.numel(), lam_c.numel())
    if L == 0:
        return 0.0
    w = torch.exp(-beta * lam_c[:L])
    return float((w * (lam_p[:L] - lam_c[:L]).pow(2)).sum().item())


def _load_proteins_graphs():
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    graphs = []
    for fi in meta["folds"]:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE,
                                         weights_only=True))
        model.eval()
        graphs.extend(collect_test_graphs(model, ds, fi["test_idx"]))
    return graphs


def _load_enzymes_graphs():
    meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    out = []
    for fi in meta["folds"]:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        mp = fi["model_path"]
        if not Path(mp).exists():
            mp = str(MODELS_CONT_DIR / Path(mp).name)
        model.load_state_dict(torch.load(mp, map_location=DEVICE,
                                         weights_only=True))
        model.eval()
        for i in fi["test_idx"]:
            g = deepcopy(graphs_all[i]).to(DEVICE)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                if model(g).argmax(-1).item() == g.y.item():
                    out.append(g)
    return out


def calibrate(dataset_name, graphs):
    if not graphs:
        return None
    rng = random.Random(SEED + hash(dataset_name) % 9973)
    t0 = time.time()

    # ---- pass 1: community density ----
    dens_samples = []
    for s in range(min(N_CAL, len(graphs) * 3)):
        g = graphs[rng.randrange(len(graphs))]
        d = _community_density(g)
        if d == d:  # not nan
            dens_samples.append(d)
        if len(dens_samples) >= N_CAL:
            break
    if not dens_samples:
        mean_density = 0.5
    else:
        mean_density = float(sum(dens_samples) / len(dens_samples))

    # ---- pass 2: S_spec_composite over random K-trojans ----
    s_samples = []
    for s in range(N_CAL):
        g = graphs[rng.randrange(len(graphs))]
        try:
            A_VV, A_VU = _sample_trojan(g, K_TROJAN, mean_density, rng,
                                        MAX_CROSS_PER_NODE_FRAC)
            sv = _s_spec_composite(g, A_VV, A_VU, K_TROJAN, K_TOP, BETA)
            s_samples.append(sv)
        except Exception as e:
            print(f"  skip s={s}: {e}")
        if (s + 1) % 50 == 0 and s_samples:
            med = sorted(s_samples)[len(s_samples) // 2]
            print(f"  [{dataset_name}] {s+1}/{N_CAL}  median_S={med:.4f}")
    if not s_samples:
        return None
    tens = torch.tensor(s_samples)
    q99 = float(torch.quantile(tens, QUANTILE).item())
    median = float(tens.median().item())
    print(f"  [{dataset_name}] N={len(s_samples)} median_S={median:.4f} q99={q99:.4f} "
          f"density={mean_density:.4f}  [{time.time()-t0:.1f}s]")
    return {
        "tau_spec_composite": q99,
        "clean_community_density": mean_density,
        "n_samples": len(s_samples),
        "n_density_samples": len(dens_samples),
        "median_S": median,
        "K_trojan": K_TROJAN,
        "k_top": K_TOP,
        "beta": BETA,
    }


def main():
    out = {}
    if OUT.exists():
        try:
            out = json.loads(OUT.read_text())
        except Exception:
            out = {}

    print("=== Calibrating PROTEINS (v2) ===")
    gp = _load_proteins_graphs()
    print(f"  loaded {len(gp)} clean PROTEINS test graphs")
    rec = calibrate("PROTEINS", gp)
    if rec:
        out["PROTEINS"] = rec

    print("=== Calibrating ENZYMES (v2) ===")
    ge = _load_enzymes_graphs()
    print(f"  loaded {len(ge)} clean ENZYMES test graphs")
    rec = calibrate("ENZYMES", ge)
    if rec:
        out["ENZYMES"] = rec

    if "PROTEINS" in out:
        out["default"] = out["PROTEINS"]
    elif "ENZYMES" in out:
        out["default"] = out["ENZYMES"]
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk in
                           ("tau_spec_composite", "clean_community_density")})
                      for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
