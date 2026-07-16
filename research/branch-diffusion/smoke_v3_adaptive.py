#!/usr/bin/env python3
"""smoke_v3_adaptive.py — v3 adaptive attack on PROTEINS 30 graphs × gddm_a.

Compares ASR (under the REAL gddm_a defense) for:
  - v3-adaptive (this file): Adam PGD through soft-STE gddm_a surrogate, T_iters=50
  - anchor sgdi: v1 N4 baseline
  - anchor specgdi_v1 / v2 (for context, from prior ablation)

Adaptation note vs v2/v3-C:
  - **Threat model upgraded**: attacker now uses ANALYTICAL gradient through
    the soft-STE gddm_a surrogate (Adam PGD). Real defense still used in the
    forward at final eval (the reported ASR). White-box at defense weights.
  - Sub-graph: same 30 graphs (n ∈ [30, 80]) as v2 ablation for direct
    comparability.
"""
import json
import time
from pathlib import Path

import torch

from prepare import MODELS_DIR, DEVICE
from sweep_stealth import load_fold as load_fold_discrete
from defenses import gddm_a_ckpt_path
from attack_v3_adaptive import run_v3_sweep, SPECGDI_V3_DEFAULTS


def filter_band(graphs):
    return [g for g in graphs if 30 <= g.num_nodes <= 80][:30]


def main():
    print(f"=== v3 ADAPTIVE attack smoke — PROTEINS fold 0, n∈[30,80], 30 graphs ===\n",
          flush=True)
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    graphs = filter_band(graphs)
    print(f"  loaded {len(graphs)} PROTEINS graphs\n", flush=True)

    ckpt = gddm_a_ckpt_path("PROTEINS")
    cfg = dict(SPECGDI_V3_DEFAULTS)
    cfg["v3_T_iters"] = 50           # longer than smoke

    t0 = time.time()
    o = run_v3_sweep(model, graphs, DEVICE, cfg=cfg, gddm_a_ckpt=ckpt)
    wall = time.time() - t0
    asr = o["asr_full"]
    print(f"\n  [v3-adaptive (T={cfg['v3_T_iters']}, K={cfg['v3_K']})] "
          f"ASR_gddm_a={asr:.3f}  wall={wall:.1f}s  per_g={wall/len(graphs):.1f}s",
          flush=True)
    print(f"  Final margins: {sorted([t['final_margin'] for t in o['trajs'] if t['final_margin'] is not None])[:10]}",
          flush=True)
    print(f"  Flipped count: {sum(1 for t in o['trajs'] if t['flipped'])}/{len(graphs)}",
          flush=True)

    out = Path(__file__).parent / "results" / f"v3_adaptive_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"asr": asr, "wall": wall, "n_graphs": len(graphs),
                                "cfg": {k: v for k, v in cfg.items() if not callable(v)},
                                "trajs": o["trajs"]}, indent=2, default=str))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
