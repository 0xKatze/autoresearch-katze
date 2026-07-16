#!/usr/bin/env python3
"""
calibrate_specgdi.py — Compute τ_spec for SpecGDI per dataset.

Procedure (from §2.3 of specgdi-spectral-guided-diffusion.md, Hard projection):
  For each of N=200 random single-node insertions into a clean test graph, where
  the inserted node gets a random victim-edge set sampled uniformly from V,
  compute  S_spec = sum_i w_i (lam_i(L') - lam_i(L))^2  using the same Hellmann–
  Feynman closed-form the attacker will use. Save the 99th-percentile of S_spec
  per dataset to `specgdi_thresholds.json`.

This is the same predictor the attacker uses inside `_specgdi_optimize_one`, so
the threshold is internally consistent (no train/test mismatch between τ_spec
and the per-step S_spec proxy).

Usage:
  micromamba run -n graph_adversarial python calibrate_specgdi.py
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

import attack as _A
from prepare import MODELS_DIR, DEVICE, GCN_GraphClassification, DATA_DIR, collect_test_graphs
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from torch_geometric.datasets import TUDataset
from copy import deepcopy

N_CAL = 200
QUANTILE = 0.99
K_TOP = 20
BETA = 1.0
SEED = 17

HERE = Path(__file__).parent
OUT = HERE / "specgdi_thresholds.json"


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
        # Stale meta path fallback (cf. SPECTRUM_DIAGNOSTIC.md "Stale meta path"):
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


def _s_spec_random_insertion(g, k, beta, rng):
    """One sample: pick a random victim-edge set, compute Hellmann–Feynman S_spec."""
    n = g.num_nodes
    # random edge count uniform in [1, n] (matches the SpecGDI attack envelope)
    n_e = rng.randint(1, n)
    victims = rng.sample(range(n), n_e)
    v_edges = [(0, v) for v in victims]
    A_clean = _A._build_adj(n, g.edge_index, [], 0, [], g.x.device)
    lam_c, U_c = _A.compute_laplacian_spectrum(A_clean, k)
    lam_c = lam_c.to(g.x.device); U_c = U_c.to(g.x.device)
    A_p = _A._build_adj(n, g.edge_index, v_edges, 1, [], g.x.device)
    L_p = _A._normalized_laplacian(A_p)
    L_c_pad = torch.zeros_like(L_p)
    L_c_pad[:n, :n] = _A._normalized_laplacian(A_clean)
    dL = L_p - L_c_pad
    dlam = _A.closed_form_delta_lambda(U_c, lam_c, dL)[:lam_c.numel()]
    lam_pert_est = lam_c + dlam
    return _A.spec_disc(lam_pert_est, lam_c, beta).item()


def calibrate(dataset_name, graphs):
    if not graphs:
        return None
    rng = random.Random(SEED + hash(dataset_name) % 9973)
    samples = []
    t0 = time.time()
    for s in range(N_CAL):
        g = graphs[rng.randrange(len(graphs))]
        try:
            val = _s_spec_random_insertion(g, K_TOP, BETA, rng)
            samples.append(val)
        except Exception as e:
            print(f"  skip sample {s}: {e}")
        if (s + 1) % 50 == 0:
            print(f"  [{dataset_name}] {s+1}/{N_CAL}  median={sorted(samples)[len(samples)//2]:.4f}")
    tens = torch.tensor(samples)
    q99 = float(torch.quantile(tens, QUANTILE).item())
    median = float(tens.median().item())
    print(f"  [{dataset_name}] N={len(samples)} median={median:.4f}  q99={q99:.4f}  "
          f"[{time.time()-t0:.1f}s]")
    return {
        "n_samples": len(samples),
        "median": median,
        "p99": q99,
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

    print("=== Calibrating PROTEINS ===")
    g_p = _load_proteins_graphs()
    print(f"  loaded {len(g_p)} clean PROTEINS test graphs")
    rec = calibrate("PROTEINS", g_p)
    if rec:
        out["PROTEINS"] = rec["p99"]
        out["PROTEINS_meta"] = rec

    print("=== Calibrating ENZYMES ===")
    g_e = _load_enzymes_graphs()
    print(f"  loaded {len(g_e)} clean ENZYMES test graphs")
    rec = calibrate("ENZYMES", g_e)
    if rec:
        out["ENZYMES"] = rec["p99"]
        out["ENZYMES_meta"] = rec

    out.setdefault("default", float(out.get("PROTEINS", out.get("ENZYMES", 0.10))))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_meta")}, indent=2))


if __name__ == "__main__":
    main()
