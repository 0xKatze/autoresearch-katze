#!/usr/bin/env python3
"""
calibrate_gnnsafe.py — Per-dataset, per-fold GNNSafe energy-threshold calibration.

For each (dataset, fold) loads the trained victim GCN, computes the
post-propagation neg_energy (= T*logsumexp(logits/T) then K=2, alpha=0.5
energy belief propagation) per node on the CLEAN train/val graphs (i.e. all
graphs except this fold's test_idx, so we never peek at test), and dumps the
5th-percentile cut so that 95% of clean nodes pass. (Symmetric to the
paper-convention 95th-pct cut on positive energy E = -neg_energy.)

Writes:   research/branch-diffusion/gnnsafe_thresholds.json
Run:      micromamba run -n graph_adversarial python calibrate_gnnsafe.py

Cross-checked against official `qitianwu/GraphOOD-GNNSafe/GNNSafe/gnnsafe.py`
(`neg_energy = T * logsumexp(logits/T, dim=-1)`, `propagation(..., prop_layers=1
or CLI --K=2, alpha=0.5)` with `d_norm = 1/deg(col)`). See vault page
`references/GNNSafe-ICLR2023.md` (confidence: high).
"""
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from prepare import (GCN_GraphClassification, DATA_DIR, MODELS_DIR, DEVICE)
from prepare_cont import MODELS_CONT_DIR, load_normalized_graphs
from defenses import _gnnsafe_node_logits, _energy_propagation


OUT_PATH = Path(__file__).parent / "gnnsafe_thresholds.json"
PCT_CUT_LOW = 5.0           # 5th-pct of neg_energy == 95th-pct of positive E
T = 1.0
K = 2
ALPHA = 0.5
SMOKE_TEST_PCT = 5.0        # clean-acc smoke test on test split


# -------------- model + graph loaders --------------

def _resolve_model_path(p):
    """meta files reference absolute paths that may not exist on this checkout.
    Fall back to the same filename inside branch-diffusion/models_cont/ or models_saved/.
    """
    pp = Path(p)
    if pp.exists():
        return pp
    here = Path(__file__).parent
    for sub in ("models_cont", "models_saved"):
        cand = here / sub / pp.name
        if cand.exists():
            return cand
    raise FileNotFoundError(p)


def _load_model(meta, fi):
    m = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    m.load_state_dict(torch.load(str(_resolve_model_path(fi["model_path"])),
                                 map_location=DEVICE, weights_only=True))
    return m.eval()


def load_proteins_graphs(meta):
    """Return per-fold (model, ref_graphs (NON-test), test_graphs)."""
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    out = []
    for fi in meta["folds"]:
        test_set = set(fi["test_idx"])
        ref = []
        test = []
        for i in range(len(ds)):
            g = ds[i].clone()
            g = g.to(DEVICE)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
            (test if i in test_set else ref).append(g)
        out.append((_load_model(meta, fi), ref, test))
    return out


def load_enzymes_graphs(meta):
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    out = []
    for fi in meta["folds"]:
        test_set = set(fi["test_idx"])
        ref, test = [], []
        for i in range(len(graphs_all)):
            g = deepcopy(graphs_all[i]).to(DEVICE)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
            (test if i in test_set else ref).append(g)
        out.append((_load_model(meta, fi), ref, test))
    return out


# -------------- core calibration --------------

def _propagated_neg_energy(model, graph):
    with torch.no_grad():
        logits = _gnnsafe_node_logits(model, graph)
        e = T * torch.logsumexp(logits / T, dim=-1)
        e = _energy_propagation(e, graph.edge_index, graph.num_nodes,
                                K=K, alpha=ALPHA)
    return e.detach().cpu().numpy()


def calibrate_one(model, ref_graphs):
    """5th-pct of propagated neg_energy across all clean ref nodes."""
    chunks = [_propagated_neg_energy(model, g) for g in ref_graphs]
    energies = np.concatenate(chunks) if chunks else np.array([])
    if energies.size == 0:
        raise RuntimeError("Empty ref set — cannot calibrate.")
    thr = float(np.percentile(energies, PCT_CUT_LOW))
    return thr, energies


# -------------- smoke tests --------------

def _clean_acc(model, graphs, defense_fn=None, **dk):
    if not graphs:
        return float("nan")
    c = 0
    for g in graphs:
        with torch.no_grad():
            inp = defense_fn(g, **dk) if defense_fn is not None else g
            if getattr(inp, "batch", None) is None or inp.num_nodes == 0:
                # filtered to empty -> classify as wrong
                continue
            pred = model(inp).argmax(-1).item()
            if pred == g.y.item():
                c += 1
    return c / len(graphs)


def _ood_flag_rate(model, graphs, threshold, n_max=30):
    """Inject one zo_free-style OOD node (feat ~ uniform[-1,1] * 2*sqrt(N))
    per graph and check whether GNNSafe flags it (drops it)."""
    from defenses import gnnsafe_defense
    flagged = 0
    total = 0
    for g in graphs[:n_max]:
        N = g.x.shape[0]
        if N == 0:
            continue
        d = g.x.shape[1]
        scale = 2.0 * (N ** 0.5)
        rng = torch.Generator(device="cpu").manual_seed(0xC0DE + total)
        # OOD feature: high-magnitude uniform; connect injected to node 0.
        injected_x = (torch.rand(1, d, generator=rng) * 2 - 1).to(DEVICE) * scale
        new_x = torch.cat([g.x, injected_x], dim=0)
        new_ei = torch.cat([
            g.edge_index,
            torch.tensor([[N, 0], [0, N]], device=DEVICE, dtype=g.edge_index.dtype),
        ], dim=1)
        new_batch = torch.zeros(N + 1, dtype=torch.long, device=DEVICE)
        from torch_geometric.data import Data as _D
        g_inj = _D(x=new_x, edge_index=new_ei, y=g.y, batch=new_batch)
        filtered = gnnsafe_defense(g_inj, victim_model=model, threshold=threshold,
                                   T=T, K=K, alpha=ALPHA)
        # injected node is index N before filtering; it's flagged if num_nodes dropped by >=1
        if filtered.num_nodes < g_inj.num_nodes:
            flagged += 1
        total += 1
    return (flagged / total) if total else float("nan")


# -------------- main --------------

def main():
    out = {"meta": {"pct_cut_low_pct": PCT_CUT_LOW, "T": T, "K": K, "alpha": ALPHA,
                    "note": "neg_energy threshold; node kept iff neg_energy >= thr"},
           "datasets": {}}

    print(f"=== calibrate_gnnsafe (pct_cut={PCT_CUT_LOW}, T={T}, K={K}, alpha={ALPHA}) ===")

    # ---- PROTEINS ----
    t0 = time.time()
    meta_p = json.loads((MODELS_DIR / "meta.json").read_text())
    folds_p = load_proteins_graphs(meta_p)
    ds_rec = {"folds": []}
    for fi, (model, ref, test) in enumerate(folds_p):
        thr, energies = calibrate_one(model, ref)
        clean_acc_undefended = _clean_acc(model, test)
        from defenses import gnnsafe_defense
        clean_acc_gnnsafe = _clean_acc(model, test, defense_fn=gnnsafe_defense,
                                       victim_model=model, threshold=thr,
                                       T=T, K=K, alpha=ALPHA)
        ood_flag = _ood_flag_rate(model, test, thr)
        rec = {"fold": fi, "threshold_neg_energy": thr,
               "n_ref_graphs": len(ref), "n_ref_nodes": int(energies.size),
               "energy_min": float(energies.min()),
               "energy_max": float(energies.max()),
               "energy_median": float(np.median(energies)),
               "clean_acc_undefended": clean_acc_undefended,
               "clean_acc_gnnsafe": clean_acc_gnnsafe,
               "clean_acc_drop": clean_acc_undefended - clean_acc_gnnsafe,
               "ood_flag_rate_zo_free": ood_flag}
        ds_rec["folds"].append(rec)
        print(f"  PROTEINS fold{fi}: thr={thr:+.4f}  "
              f"cleanAcc {clean_acc_undefended:.3f}->{clean_acc_gnnsafe:.3f} "
              f"(Δ={clean_acc_undefended-clean_acc_gnnsafe:+.3f})  "
              f"OOD_flag={ood_flag:.2f}")
    ds_rec["threshold_neg_energy_mean"] = float(np.mean(
        [r["threshold_neg_energy"] for r in ds_rec["folds"]]))
    out["datasets"]["PROTEINS"] = ds_rec
    print(f"  PROTEINS mean thr = {ds_rec['threshold_neg_energy_mean']:+.4f}  "
          f"[{time.time()-t0:.1f}s]")

    # ---- ENZYMES ----
    t0 = time.time()
    meta_e = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    folds_e = load_enzymes_graphs(meta_e)
    ds_rec = {"folds": []}
    for fi, (model, ref, test) in enumerate(folds_e):
        thr, energies = calibrate_one(model, ref)
        clean_acc_undefended = _clean_acc(model, test)
        from defenses import gnnsafe_defense
        clean_acc_gnnsafe = _clean_acc(model, test, defense_fn=gnnsafe_defense,
                                       victim_model=model, threshold=thr,
                                       T=T, K=K, alpha=ALPHA)
        ood_flag = _ood_flag_rate(model, test, thr)
        rec = {"fold": fi, "threshold_neg_energy": thr,
               "n_ref_graphs": len(ref), "n_ref_nodes": int(energies.size),
               "energy_min": float(energies.min()),
               "energy_max": float(energies.max()),
               "energy_median": float(np.median(energies)),
               "clean_acc_undefended": clean_acc_undefended,
               "clean_acc_gnnsafe": clean_acc_gnnsafe,
               "clean_acc_drop": clean_acc_undefended - clean_acc_gnnsafe,
               "ood_flag_rate_zo_free": ood_flag}
        ds_rec["folds"].append(rec)
        print(f"  ENZYMES  fold{fi}: thr={thr:+.4f}  "
              f"cleanAcc {clean_acc_undefended:.3f}->{clean_acc_gnnsafe:.3f} "
              f"(Δ={clean_acc_undefended-clean_acc_gnnsafe:+.3f})  "
              f"OOD_flag={ood_flag:.2f}")
    ds_rec["threshold_neg_energy_mean"] = float(np.mean(
        [r["threshold_neg_energy"] for r in ds_rec["folds"]]))
    out["datasets"]["ENZYMES"] = ds_rec
    print(f"  ENZYMES  mean thr = {ds_rec['threshold_neg_energy_mean']:+.4f}  "
          f"[{time.time()-t0:.1f}s]")

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_PATH}")

    # ---- aggregate smoke verdicts ----
    print("\n=== smoke test summary ===")
    for ds, rec in out["datasets"].items():
        drops = [f["clean_acc_drop"] for f in rec["folds"]]
        oods = [f["ood_flag_rate_zo_free"] for f in rec["folds"]]
        ok_drop = max(drops) <= 0.03 + 1e-9
        ok_ood = min(oods) >= 0.5
        print(f"  {ds:<9}  max clean-acc drop = {max(drops):+.3f} "
              f"({'OK' if ok_drop else 'FAIL'} <=3pp)  "
              f"min OOD-flag = {min(oods):.2f} "
              f"({'OK' if ok_ood else 'WEAK'} >=0.5)")


if __name__ == "__main__":
    main()
