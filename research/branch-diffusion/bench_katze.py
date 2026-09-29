#!/usr/bin/env python3
"""Benchmark: main-project KATZE (GZOO_GraphClassification, feat_scale=auto_x2, full-edge, m=1)
on the branch-diffusion 5-fold victims (PROTEINS discrete or ENZYMES-continuous), with victim
query counting (1 query = 1 graph forwarded). Same settings as run_katze_vs_sgdi.py::run_katze_eb.

usage: python bench_katze.py [PROTEINS|ENZYMES] [cap]
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cdma_data  # noqa: E402  (fold loading shared with cdma_graph.py)
sys.path.insert(0, "/workspace")
from attack.gzoo_graph_classification import GZOO_GraphClassification  # noqa: E402

KATZE_DEFAULTS = {'gen_lr': 5e-3, 'attack_epochs': 50, 'sigma': 1e-3, 'eval_num': 500,
                  'patience': 4, 'kappa': -0.1, 'feat_scale': 'auto_x2', 'grad_method': 'cge',
                  'gen_hid_dim': 128}


class CountingModel(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m; self.q = 0
    def forward(self, data):
        b = getattr(data, "batch", None)
        self.q += int(b.max().item()) + 1 if b is not None and b.numel() else 1
        return self.m(data)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "PROTEINS"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    rows = []
    for fold in cdma_data.load_folds(ds):
        tg = fold["test_graphs"][:cap]
        cm = CountingModel(fold["victim"]).eval()
        att = GZOO_GraphClassification(
            node_budget=1, feat_dim=fold["n_feat"], discrete_feat=False,
            kappa=KATZE_DEFAULTS['kappa'], sigma=KATZE_DEFAULTS['sigma'],
            eval_num=KATZE_DEFAULTS['eval_num'], patience=KATZE_DEFAULTS['patience'],
            gen_hid_dim=KATZE_DEFAULTS['gen_hid_dim'], gen_lr=KATZE_DEFAULTS['gen_lr'],
            attack_epochs=KATZE_DEFAULTS['attack_epochs'], device="cpu",
            edge_strategy='full', edge_top_k=64, dual_gradient=False,
            feat_scale=KATZE_DEFAULTS['feat_scale'], grad_method=KATZE_DEFAULTS['grad_method'],
            inject_topology='none', loss_type='cw', shared_perturbation=False)
        t0 = time.time()
        r = att.run_attack(cm, tg, num_samples=len(tg))
        na = r["num_attacked"]
        rows.append({"fold": fold["fold"], "asr": r["success_rate"], "n": na,
                     "queries": cm.q, "q_per_graph": cm.q / max(na, 1),
                     "wall_s": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    out = {"script": "bench_katze.py", "dataset": ds, "cap": cap,
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "config": {**KATZE_DEFAULTS, "edge_strategy": "full", "node_budget": 1},
           "rows": rows, "mean_asr": sum(r["asr"] for r in rows) / len(rows),
           "mean_q_per_graph": sum(r["q_per_graph"] for r in rows) / len(rows)}
    p = HERE / "results/cdma" / f"bench_katze_{ds}_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(out, open(p, "w"), indent=2)
    print("saved", p, out["mean_asr"], out["mean_q_per_graph"])


if __name__ == "__main__":
    main()
