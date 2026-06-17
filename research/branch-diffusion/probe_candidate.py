#!/usr/bin/env python3
"""probe_candidate.py — fast timing+ASR probe of the eb1 candidate configs on the
EXISTING fold0 seed42 PROTEINS GCN victim, all 4 defenses. NOT a gate run; sizes
the full paper-grade matrix. Reuses run_flow_a_vs_sgdi_battery building blocks."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, importlib.util
from pathlib import Path
import torch
from torch_geometric.datasets import TUDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs
import json

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

sgdi_attack = _load("branch_sgdi_attack", HERE / "attack.py")
import attack_flow
import attack_flow_b
from defenses import (DefendedModel, norm_filter_defense, mahalanobis_defense,
                      degree_filter_defense, node_degrees)

DEVICE = "cpu"
DEFENSES = ["none", "normfilter", "mahalanobis", "degfilter"]
DEGFILTER_K = 3.0
CAP = 8

def build_defended(model, defense, ref_graphs):
    if defense == "none": return model
    if defense == "normfilter":
        return DefendedModel(model, norm_filter_defense, "normfilter", percentile=95)
    if defense == "mahalanobis":
        X = torch.cat([g.x for g in ref_graphs], dim=0).float()
        mean_ = X.mean(0); d = X.size(1)
        cov = torch.cov(X.t()) + 1e-3*torch.eye(d); cov_inv = torch.linalg.inv(cov)
        diff = X - mean_; sqm = (diff @ cov_inv * diff).sum(1)
        thr = float(torch.quantile(sqm, 0.995).item())
        return DefendedModel(model, mahalanobis_defense, "mahalanobis", mean=mean_, cov_inv=cov_inv, threshold=thr)
    if defense == "degfilter":
        degs = torch.cat([node_degrees(g) for g in ref_graphs]).float()
        return DefendedModel(model, degree_filter_defense, "degfilter",
                             mean=float(degs.mean()), std=float(degs.std()), k=DEGFILTER_K)
    raise ValueError(defense)

def run_cfg(mod, victim, graphs, overrides):
    saved = {k: mod.CONFIG.get(k) for k in overrides}
    mod.CONFIG.update(overrides)
    try:
        t0 = time.time(); asr = mod.run_attack(victim, graphs, DEVICE); dt = time.time()-t0
    finally:
        mod.CONFIG.update(saved)
    return asr, dt

meta = json.loads((MODELS_DIR / "meta.json").read_text())
ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
fold = meta["folds"][0]
model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
model.load_state_dict(torch.load(fold["model_path"], map_location=DEVICE)); model.eval()
graphs = collect_test_graphs(model, ds, fold["test_idx"])[:CAP]
print(f"probe: PROTEINS GCN fold0 seed42, n_graphs={len(graphs)}, defenses={DEFENSES}", flush=True)

# candidate configs: edge_budget=1 via edge_strategy='topk' (else-branch -> select_targets_topk(nb=1)), fs=1.0
CONFIGS = {
    "FM-A-full(ax2)": (attack_flow, {"edge_strategy": "full", "node_budget": 1, "feat_scale": "auto_x2"}),
    "FM-A-eb1(fs1)":  (attack_flow, {"edge_strategy": "topk", "node_budget": 1, "feat_scale": 1.0}),
    "FM-B-eb1(fs1)":  (attack_flow_b, {"edge_strategy": "topk", "node_budget": 1, "feat_scale": 1.0}),
}
for name, (mod, ov) in CONFIGS.items():
    print(f"\n=== {name} ===", flush=True)
    for dn in DEFENSES:
        victim = build_defended(model, dn, graphs)
        asr, dt = run_cfg(mod, victim, graphs, ov)
        print(f"  {dn:<12} ASR={asr*100:6.2f}%  ({dt:.1f}s)", flush=True)
