#!/usr/bin/env python3
"""Reproduce + diagnose the SGDI-full | degfilter crash (and confirm FM-full=100).
Runs seed 42 fold 0 on CPU. Prints the SGDI traceback so we can get the true value."""
import os, sys, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
from torch_geometric.datasets import TUDataset

import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs
import eval_paper_grade_floor as F
F.DEVICE = "cpu"
import attack_om2c as O

meta = F.load_meta_for_seed(42)
ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
fold = meta["folds"][0]
model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to("cpu")
model.load_state_dict(torch.load(fold["model_path"], map_location="cpu")); model.eval()
graphs = collect_test_graphs(model, ds, fold["test_idx"])[:10]
ref = F.training_ref_graphs(ds, meta, fold)
prior = O.om2c_prior(graphs, "cpu", O.CONFIG["tangent_dim"], O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])

deg_victim = F.build_defended(model, "degfilter", ref)
ATT = {
    "FM-full":  ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "SGDI-full": ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
}
for vname, vic in [("none", model), ("degfilter", deg_victim)]:
    print(f"\n===== victim = {vname} =====", flush=True)
    for aname, (rk, ov) in ATT.items():
        try:
            r = F.run_attack_cell(rk, ov, vic, graphs, prior)
            print(f"  {aname:9s} asr={r['asr']!r}  q={r['q']}  n_elig={r.get('n_eligible')}", flush=True)
        except Exception as e:
            print(f"  {aname:9s} CRASH: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
