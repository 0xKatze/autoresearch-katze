#!/usr/bin/env python3
"""test_bismooth_repro_gpu.py — REQUIRED reproducibility gate for FIX2.

Runs OM2C-nb10 through the bismooth (binding) defense on ONE fold (seed 42, fold 0) on
GPU TWICE, re-applying the FIX2 per-(seed,fold) global seed each time, and confirms the
bismooth ASR + eligibility + flip pattern are IDENTICAL run-to-run. If they are, the
global-seed FIX2 makes bismooth's node-deletion reproducible on GPU (the BS-009 bug — a
CPU torch.Generator no-op on CUDA falling back to an un-seeded global RNG — is closed).

Run: FLOOR_GPU=0 micromamba run -n graph_adversarial python test_bismooth_repro_gpu.py
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_pg_floor_battery6 as B   # GPU-enable hack at import
import torch
import prepare
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs
from torch_geometric.datasets import TUDataset
import attack_om2c as O

F = B.F
DEVICE = "cuda"
SEED = 42
T, P = 20, 0.2


def run_once():
    B._BT, B._BP = T, P
    F.DEVICE = DEVICE
    prepare.DEVICE = DEVICE
    meta = F.load_meta_for_seed(SEED)
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    fold_info = meta["folds"][0]
    fi = fold_info["fold_idx"]
    # FIX2: identical per-(seed,fold) global seed each run
    fold_seed = (int(SEED) * 100003 + int(fi)) % (2**31 - 1)
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)
    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
    model.eval()
    graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:10]
    ref = F.training_ref_graphs(ds, meta, fold_info)
    prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                         O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
    victim = B.build_defended_ext(model, "bismooth", ref)
    r = F.run_attack_cell("om2c", {"node_budget": 10, "feat_scale": 1.0}, victim, graphs, prior)
    return {"asr": r["asr"], "n_eligible": r["n_eligible"], "n_total": r["n_total"]}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable (no silent CPU fallback)")
    a = run_once()
    b = run_once()
    print(f"run#1 bismooth: asr={a['asr']:.6f} eligible={a['n_eligible']}/{a['n_total']}", flush=True)
    print(f"run#2 bismooth: asr={b['asr']:.6f} eligible={b['n_eligible']}/{b['n_total']}", flush=True)
    identical = (abs(a["asr"] - b["asr"]) < 1e-9 and a["n_eligible"] == b["n_eligible"]
                 and a["n_total"] == b["n_total"])
    print("=" * 60, flush=True)
    print(f"REPRODUCIBILITY: {'IDENTICAL (FIX2 verified)' if identical else 'DIVERGENT (FIX2 INSUFFICIENT)'}",
          flush=True)
    (HERE / "results" / "bismooth_repro_gpu_twrun.json").write_text(
        json.dumps({"run1": a, "run2": b, "identical": identical,
                    "fix": "global per-(seed,fold) torch.manual_seed + cuda.manual_seed_all"}, indent=2))
    raise SystemExit(0 if identical else 2)


if __name__ == "__main__":
    main()
