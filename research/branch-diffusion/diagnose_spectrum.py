#!/usr/bin/env python3
"""diagnose_spectrum.py — cheap diagnostic: clean vs injected normalized-Laplacian
spectral discrepancy per (dataset, feat_mode).

For each test graph at the rho=1.0 full-edge anchor (the regime where eval_paper_grade
reports asr_full), we run the SAME feature optimization the attack uses, then build the
perturbed Data(x=..., edge_index=...) by appending features+edges (the attack never
mutates the original). We DO NOT call any greedy / multi-rho probing — just one
optimization round per (graph, feat_mode), single seed=0, all 5 folds.

We compute top-k eigenvalues of the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
on clean (n nodes) and perturbed (n+1 nodes), with k = min(20, n). Spectral discrepancy
delta = ||sort(lam_clean)[:k] - sort(lam_pert)[:k]||_2 (matched-rank smallest eigenvalues).

Output: results/spectrum_diagnostic_<stamp>.json
"""
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# import the attack's *primitives* read-only (no behaviour change)
import attack as A
from prepare import MODELS_DIR, RESULTS_DIR, DEVICE
from sweep_stealth import load_fold as load_fold_discrete
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from prepare import GCN_GraphClassification

FEAT_MODES = ["zo_normcap", "zo_free", "sgdi"]
ATTACK_CFG = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 20, "zo_restarts": 2,
              "use_inter_edges": True}
SEED = 0
TOPK = 20


# ---- dataset loaders (mirror eval_paper_grade.py, one-seed, all folds) -------------

def load_discrete_folds(meta):
    return [load_fold_discrete(fi, meta) for fi in meta["folds"]]


def load_continuous_folds(name, meta):
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs(name, mean, std)
    out = []
    for fi in meta["folds"]:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        mp = Path(fi["model_path"])
        if not mp.is_file():  # meta has stale absolute path -> fallback to local copy
            mp = MODELS_CONT_DIR / mp.name
        model.load_state_dict(torch.load(str(mp), map_location=DEVICE,
                                         weights_only=True))
        model.eval()
        graphs = []
        for i in fi["test_idx"]:
            g = deepcopy(graphs_all[i]).to(DEVICE)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                if model(g).argmax(-1).item() == g.y.item():
                    graphs.append(g)
        out.append((model, graphs))
    return out


# ---- spectrum --------------------------------------------------------------------

def normalized_laplacian_spectrum(data):
    """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
    Returns sorted eigenvalues (ascending) as a numpy array; flags isolated nodes."""
    n = data.num_nodes
    ei = data.edge_index.detach().cpu().numpy()
    A_mat = np.zeros((n, n), dtype=np.float64)
    A_mat[ei[0], ei[1]] = 1.0
    A_mat = np.maximum(A_mat, A_mat.T)  # symmetrize (TUD/PyG usually already symmetric)
    deg = A_mat.sum(axis=1)
    isolated = int((deg == 0).sum())
    d_inv_sqrt = np.zeros(n)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    L = np.eye(n) - (d_inv_sqrt[:, None] * A_mat) * d_inv_sqrt[None, :]
    # symmetrize to clobber float noise
    L = 0.5 * (L + L.T)
    eigvals = np.linalg.eigvalsh(L)
    eigvals.sort()
    return eigvals, isolated


def build_perturbed_full_anchor(model, data, feat_mode, stats, prior):
    """Replicate the attack's full-budget perturbation (rho=1.0) on ONE graph.

    feat_mode in {zo_normcap, zo_free, sgdi}: run the feature optimizer, then attach
    edges from the single injected node to ALL victim nodes (m=1 so no inter-edges).
    """
    cfg = {**A.CONFIG, **ATTACK_CFG, "feat_mode": feat_mode, "seed_offset": SEED}
    # radius like attack.run_attack_sweep
    cfg["_norm_radius"] = cfg.get("normcap_kappa", 1.95) * stats[2]
    if feat_mode == "sgdi":
        cfg["_prior"] = prior

    n = data.num_nodes
    m = 1
    feats = A.make_features(stats, m, feat_mode, n, DEVICE)
    all_v = [(0, v) for v in range(n)]   # full-edge anchor (rho=1.0)
    inter_full = []  # m=1, no inter edges

    if feat_mode == "sgdi":
        feats = A.sgdi_optimize_features(model, data, feats, all_v, inter_full,
                                         cfg, DEVICE, prior)
    else:
        feats = A.zo_optimize_features(model, data, feats, all_v, inter_full, cfg, DEVICE)

    pert = A._build(data, feats, all_v, inter_full, DEVICE)
    return pert


# ---- driver ----------------------------------------------------------------------

def run_dataset(ds_key, folds_data, prior_per_fold, stats_per_fold):
    by_fm = {fm: [] for fm in FEAT_MODES}
    fold_iso = []
    for fi, (model, graphs) in enumerate(folds_data):
        if not graphs:
            continue
        stats = stats_per_fold[fi]
        prior = prior_per_fold[fi]
        for gi, g in enumerate(graphs):
            data = g.to(DEVICE)
            lam_clean, iso_c = normalized_laplacian_spectrum(data)
            k = min(TOPK, lam_clean.size)
            base = lam_clean[:k]
            for fm in FEAT_MODES:
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                try:
                    pert = build_perturbed_full_anchor(model, data, fm, stats, prior)
                    lam_pert, iso_p = normalized_laplacian_spectrum(pert)
                except Exception as e:
                    by_fm[fm].append({"error": repr(e), "fold": fi, "gi": gi})
                    continue
                kp = min(k, lam_pert.size)
                delta = float(np.linalg.norm(lam_pert[:kp] - base[:kp]))
                by_fm[fm].append({
                    "fold": fi, "gi": gi, "n_clean": int(data.num_nodes), "k": int(kp),
                    "delta": delta,
                    "iso_clean": int(iso_c), "iso_pert": int(iso_p),
                })
        fold_iso.append({"fold": fi, "n_graphs": len(graphs)})
        print(f"  [{ds_key}] fold {fi}: {len(graphs)} graphs done")

    summary = {}
    for fm in FEAT_MODES:
        deltas = [r["delta"] for r in by_fm[fm] if "delta" in r]
        n_err = sum(1 for r in by_fm[fm] if "error" in r)
        if deltas:
            summary[fm] = {
                "mean_spec_delta": float(np.mean(deltas)),
                "median_spec_delta": float(np.median(deltas)),
                "std_spec_delta": float(np.std(deltas)),
                "min": float(np.min(deltas)), "max": float(np.max(deltas)),
                "n_graphs": len(deltas), "n_errors": n_err,
            }
        else:
            summary[fm] = {"mean_spec_delta": None, "n_graphs": 0, "n_errors": n_err}
    return summary, by_fm, fold_iso


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"spectrum_diagnostic_{stamp}.json"

    datasets = ["PROTEINS", "ENZYMES"]
    if "--enzymes-only" in sys.argv:
        datasets = ["ENZYMES"]
    if "--proteins-only" in sys.argv:
        datasets = ["PROTEINS"]

    output = {"stamp": stamp, "seed": SEED, "topk": TOPK,
              "attack_cfg": ATTACK_CFG, "feat_modes": FEAT_MODES,
              "datasets": {}, "per_graph": {}}

    print(f"=== diagnose_spectrum {stamp} | datasets={datasets} (device={DEVICE}) ===")
    t0 = time.time()

    for ds_key in datasets:
        td = time.time()
        if ds_key == "PROTEINS":
            meta = json.loads((MODELS_DIR / "meta.json").read_text())
            folds = load_discrete_folds(meta)
        else:
            meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
            folds = load_continuous_folds("ENZYMES", meta)

        # precompute per-fold stats & prior using fold's test graphs
        stats_per_fold = []
        prior_per_fold = []
        for (model, graphs) in folds:
            if not graphs:
                stats_per_fold.append(None); prior_per_fold.append(None); continue
            stats_per_fold.append(A.data_stats(graphs, DEVICE))
            prior_per_fold.append(A.data_prior(graphs, DEVICE))

        summary, by_fm, fold_iso = run_dataset(ds_key, folds, prior_per_fold, stats_per_fold)
        output["datasets"][ds_key] = {"summary": summary, "fold_meta": fold_iso}
        output["per_graph"][ds_key] = by_fm
        # save incrementally
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  -> {ds_key} done in {time.time()-td:.0f}s")
        for fm in FEAT_MODES:
            s = summary[fm]
            if s.get("mean_spec_delta") is None:
                print(f"     {fm:<11} no data (errors={s.get('n_errors',0)})")
            else:
                print(f"     {fm:<11} mean Δ={s['mean_spec_delta']:.4f} "
                      f"median={s['median_spec_delta']:.4f} std={s['std_spec_delta']:.4f} "
                      f"n={s['n_graphs']} err={s['n_errors']}")

    print(f"\nSaved {out_path} [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
