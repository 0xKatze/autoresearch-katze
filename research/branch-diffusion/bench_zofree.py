#!/usr/bin/env python3
"""Benchmark: KATZE-style unconstrained ZO attack (zo_free, feat_scale=auto_x2, ASR@full-edge),
5-fold PROTEINS GCN, with per-fold query counts. Reference point for cdma_graph.py."""
import json, sys, time
from datetime import datetime
from pathlib import Path
import torch
from torch_geometric.datasets import TUDataset
sys.path.insert(0, str(Path(__file__).parent))
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs
import attack

DEVICE = "cpu"
CFG = {"feat_mode": "zo_free", "feat_scale": "auto_x2", "skip_greedy": True, "frac_anchors": [1.0]}

def main():
    meta = json.load(open(MODELS_DIR / "meta.json"))
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    rows = []
    for f in meta["folds"]:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"])
        model.load_state_dict(torch.load(f["model_path"], map_location=DEVICE, weights_only=True))
        model.eval()
        tg = [g.cpu() for g in collect_test_graphs(model, ds, f["test_idx"])]
        t0 = time.time()
        r = attack.run_attack_sweep(model, tg, DEVICE, CFG)
        rows.append({"fold": f["fold_idx"], "asr": r["asr_full"], "n": r["n_graphs"],
                     "queries": r["queries"], "q_per_graph": r["queries"] / max(r["n_graphs"], 1),
                     "wall_s": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    out = {"script": "bench_zofree.py", "timestamp": datetime.now().isoformat(timespec="seconds"),
           "config": CFG, "rows": rows,
           "mean_asr": sum(r["asr"] for r in rows) / len(rows),
           "mean_q_per_graph": sum(r["q_per_graph"] for r in rows) / len(rows)}
    p = Path(__file__).parent / "results/cdma" / f"bench_zofree_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(out, open(p, "w"), indent=2); print("saved", p, out["mean_asr"], out["mean_q_per_graph"])

if __name__ == "__main__":
    main()
