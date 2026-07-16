#!/usr/bin/env python3
"""attack_v3_mimicry.py — SpecGDI v3-B: community-mimicry trojan.

Idea: instead of synthesizing K-node trojan from scratch (v2's strategy that
gddm_a's denoiser flags as anomalous neighborhood), **sample a real K-node
subgraph from same-class PROTEINS graphs** and use it as the trojan template.
Then σ-bounded perturb X within Mahalanobis ball.

Hypothesis: gddm_a's per-edge denoiser learned "natural neighborhood" priors
from clean graphs in training. A real subgraph (motif) lifted from the train
set IS such a neighborhood — should not trigger anomaly score, while still
acting as a feature-poisoning injection.

Adaptation note vs KATZE / v2:
  - Trojan content sourced from train-set subgraphs, not synthesized
  - Threat model unchanged from v2 (pure black-box at defended pipeline) —
    just changes the *initialization* of (X*, A_VV)
  - A_VV (inter-injected edges) is FIXED to sampled subgraph topology, only
    A_VU (cross-edges to victim) and X (per-row σ-perturbed) are optimised
"""
from __future__ import annotations
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from prepare import MODELS_DIR, DEVICE
from torch_geometric.datasets import TUDataset
from prepare import DATA_DIR

import attack as A
from attack import (run_attack_sweep, CONFIG, SPECGDI_DEFAULTS,
                     SPECGDI_V2_DEFAULTS, data_prior, data_stats)


def sample_class_motifs(dataset, target_class: int, K: int, n_motifs: int = 100,
                         seed: int = 42):
    """Sample n_motifs K-node connected subgraphs from graphs labeled target_class.

    For each chosen graph, do a BFS from a random node up to K nodes.
    Returns list of (X_sub, edge_index_sub) tuples.
    """
    rng = torch.Generator().manual_seed(seed)
    candidates = [i for i in range(len(dataset)) if int(dataset[i].y.item()) == target_class]
    motifs = []
    attempts = 0
    while len(motifs) < n_motifs and attempts < 10 * n_motifs:
        attempts += 1
        idx = candidates[torch.randint(0, len(candidates), (1,), generator=rng).item()]
        g = dataset[idx]
        n = g.num_nodes
        if n < K:
            continue
        # BFS from random start
        start = torch.randint(0, n, (1,), generator=rng).item()
        ei = g.edge_index.tolist()
        adj = {v: [] for v in range(n)}
        for s, d in zip(ei[0], ei[1]):
            adj[s].append(d)
        # BFS
        visited = {start}
        frontier = [start]
        order = [start]
        while frontier and len(order) < K:
            new_frontier = []
            for u in frontier:
                for v in adj.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        order.append(v)
                        new_frontier.append(v)
                        if len(order) >= K:
                            break
                if len(order) >= K:
                    break
            frontier = new_frontier
        if len(order) < K:
            continue
        # Build sub edge_index with local node indices
        local = {nid: i for i, nid in enumerate(order)}
        ei_sub = []
        for s, d in zip(ei[0], ei[1]):
            if s in local and d in local:
                ei_sub.append((local[s], local[d]))
        if not ei_sub:
            continue
        ei_t = torch.tensor(ei_sub, dtype=torch.long).t().contiguous()
        X_sub = g.x[order].float().clone()
        motifs.append((X_sub, ei_t))
    return motifs


def motif_to_A_VV(K: int, ei_sub: torch.Tensor) -> torch.Tensor:
    A_VV = torch.zeros(K, K)
    if ei_sub.numel() > 0:
        A_VV[ei_sub[0], ei_sub[1]] = 1.0
        A_VV = ((A_VV + A_VV.t()) > 0).float()
    A_VV.fill_diagonal_(0.0)
    return A_VV


# This is a v2-init override — we mutate the existing _run_specgdi_v2_sweep's
# init to use sampled motifs instead of synthetic random init.
# Implementation: monkey-patch the K-node trojan init at the v2 dispatch.

def run_v3_mimicry_sweep(model, test_graphs, device, motifs, cfg=None):
    """Like _run_specgdi_v2_sweep but trojans init from sampled motifs (per-graph,
    one motif sampled per attack). A_VV stays FIXED; X gets σ-perturbed; A_VU
    free. Reports ASR on the defended forward.

    Args:
        motifs: list of (X_sub, ei_sub) tuples from sample_class_motifs.
    """
    import attack
    cfg = {**CONFIG, **SPECGDI_DEFAULTS, **SPECGDI_V2_DEFAULTS, **(cfg or {})}
    K = cfg["v2_K"]
    prior = data_prior(test_graphs, device)
    mu = prior[0]

    # Patch v2 init by monkey-patching `specgdi_v2_optimize_trojan` to use
    # motif-derived (X_init, A_VV_init). Save original.
    orig = attack.specgdi_v2_optimize_trojan
    motif_iter = iter(motifs * (len(test_graphs) // len(motifs) + 1))

    def patched(model, data, X_star_init, A_VV_init, A_VU_init, cfg_arg, device, prior):
        # Override init with next motif
        X_sub, ei_sub = next(motif_iter)
        X_sub = X_sub.to(device)
        # Mahalanobis-project the motif features into the in-distribution ball
        X_init = X_sub
        A_VV_init_motif = motif_to_A_VV(K, ei_sub).to(device)
        # A_VU random init same as before
        return orig(model, data, X_init, A_VV_init_motif, A_VU_init, cfg_arg, device, prior)

    attack.specgdi_v2_optimize_trojan = patched
    try:
        cfg["feat_mode"] = "specgdi_v2"
        result = run_attack_sweep(model, test_graphs, device, cfg)
    finally:
        attack.specgdi_v2_optimize_trojan = orig
    return result


if __name__ == "__main__":
    print("=== v3-B mimicry smoke — PROTEINS fold 0 ===")
    from sweep_stealth import load_fold as load_fold_discrete
    from defenses import DefendedModel, gddm_a_defense, gddm_a_ckpt_path

    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    graphs = [g for g in graphs if 30 <= g.num_nodes <= 80][:30]
    print(f"  loaded {len(graphs)} graphs")

    # Sample motifs from class-1 PROTEINS graphs (the non-target class — flip *toward* class 1
    # by injecting class-1-like content; symmetric assumption)
    ds = TUDataset(root=str(DATA_DIR / "PROTEINS"), name="PROTEINS")
    motifs = sample_class_motifs(ds, target_class=0, K=3, n_motifs=100, seed=42)
    print(f"  sampled {len(motifs)} motifs from class 0")

    ckpt = gddm_a_ckpt_path("PROTEINS")
    dmodel = DefendedModel(model, gddm_a_defense, "gddm_a",
                            ckpt_path=ckpt, t_star=30)
    cfg = {"v2_K": 3, "v2_T_diff": 10, "v2_budget_VU_per_node": 1.0,
            "v2_tau_composite_fallback": 1e9, "v2_density_fallback": 0.6253,
            "specgdi_dataset_key": "PROTEINS_MIMIC",
            "skip_greedy": True, "zo_epochs": 10, "zo_restarts": 1,
            "seed_offset": 0, "node_budget": 3}

    t0 = time.time()
    o = run_v3_mimicry_sweep(dmodel, graphs, DEVICE, motifs, cfg=cfg)
    print(f"  v3-B (mimicry) × gddm_a: ASR={o['asr_full']:.3f}  wall={time.time()-t0:.1f}s")

    out = Path(__file__).parent / "results" / f"v3_mimicry_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"asr": o["asr_full"], "n_graphs": len(graphs),
                                "n_motifs": len(motifs)}, indent=2, default=str))
    print(f"\nSaved {out}")
