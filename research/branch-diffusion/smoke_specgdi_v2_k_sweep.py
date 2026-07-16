#!/usr/bin/env python3
"""smoke_specgdi_v2_k_sweep.py — Complete K × budget grid on PROTEINS to formally
close out the v2 ablation. Result feeds the paper's "we explored the full v2
hyperparameter grid, ASR floor under gddm_a is structural" claim.

Adaptation note vs current smoke_specgdi_v2_ablation.py:
  - 5 K values (1, 3, 5, 7, 10) instead of 1 alone
  - 2 budget values (0.5, 1.0) — drops budget=0.2 (known-broken from ablation)
  - PROTEINS only (ENZYMES gddm_a contaminated; not paper-citable on it)
  - Same 30-graph subset (n ∈ [30, 80]) as the ablation for direct comparability
  - τ=∞ (spec-off): ablation showed spec-projection never triggers in calibrated
    regime, so we keep it OFF to avoid the orthogonal noise source
"""
import json
import math
import time
from pathlib import Path

import torch

from prepare import MODELS_DIR, DEVICE
from sweep_stealth import load_fold as load_fold_discrete
from defenses import DefendedModel, gddm_a_defense, gddm_a_ckpt_path
import attack


SMOKE_MAX_GRAPHS = 30
SMOKE_MIN_N = 30
SMOKE_MAX_N = 80
T_DIFF = 10
SEED = 0
PROTEINS_DENSITY_CAL = 0.6253
TAU_OFF = 1e9


def filter_band(graphs):
    out = [g for g in graphs if SMOKE_MIN_N <= g.num_nodes <= SMOKE_MAX_N]
    return out[:SMOKE_MAX_GRAPHS]


def run_one(label, model, graphs, K, budget_VU):
    cfg = {
        "feat_mode": "specgdi_v2",
        "node_budget": K,
        "skip_greedy": True,
        "zo_epochs": 10,
        "zo_restarts": 1,
        "seed_offset": SEED,
        "sgdi_T_diff": T_DIFF,
        "v2_T_diff": T_DIFF,
        "v2_K": K,
        "v2_budget_VU_per_node": budget_VU,
        "v2_tau_composite_fallback": TAU_OFF,
        "specgdi_dataset_key": "PROTEINS_KSWEEP",
        "v2_density_fallback": PROTEINS_DENSITY_CAL,
    }
    t0 = time.time()
    o = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    dt = time.time() - t0
    asr = o["asr_full"]
    n_g = max(o.get("n_graphs", len(graphs)), 1)
    print(f"  [{label:<32}] ASR={asr:.3f}  n_g={n_g}  wall={dt:.1f}s  per_g={dt/n_g:.2f}s",
          flush=True)
    return {"label": label, "K": K, "budget_VU": budget_VU,
            "asr": asr, "n_g": n_g, "wall_s": dt}


def main():
    print(f"=== v2 K-sweep grid — PROTEINS fold 0, n∈[{SMOKE_MIN_N},{SMOKE_MAX_N}], "
          f"30 graphs, K ∈ {{1,3,5,7,10}} × budget ∈ {{0.5,1.0}} ===\n", flush=True)
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    graphs = filter_band(graphs)
    print(f"  loaded {len(graphs)} PROTEINS graphs\n", flush=True)

    ckpt = gddm_a_ckpt_path("PROTEINS")
    dmodel = DefendedModel(model, gddm_a_defense, "gddm_a",
                           ckpt_path=ckpt, t_star=30)

    results = []
    Ks = [1, 3, 5, 7, 10]
    budgets = [0.5, 1.0]

    # Anchors first
    print("-- anchors (no_def baseline + gddm_a baseline) --", flush=True)
    for label, m in [("anchor_sgdi", model), ("anchor_sgdi_gddm_a", dmodel)]:
        t0 = time.time()
        o = attack.run_attack_sweep(m, graphs, DEVICE,
                                     {"feat_mode": "sgdi", "node_budget": 1,
                                      "skip_greedy": True, "zo_epochs": 10,
                                      "zo_restarts": 1, "seed_offset": SEED,
                                      "sgdi_T_diff": T_DIFF})
        dt = time.time() - t0
        print(f"  [{label:<32}] ASR={o['asr_full']:.3f}  wall={dt:.1f}s", flush=True)
        results.append({"label": label, "K": None, "budget_VU": None,
                        "asr": o["asr_full"], "wall_s": dt})

    # Grid
    for K in Ks:
        for b in budgets:
            print(f"\n-- K={K}, budget_VU={b} --", flush=True)
            # nodef
            results.append(run_one(f"v2_K{K}_b{b}_nodef", model, graphs, K, b))
            # gddm_a
            results.append(run_one(f"v2_K{K}_b{b}_gddm_a", dmodel, graphs, K, b))

    # Summary grid
    print("\n=== GRID SUMMARY (ASR; PROTEINS fold 0, 30 graphs) ===", flush=True)
    print("                              budget_VU=0.5         budget_VU=1.0", flush=True)
    print("  K     no_def  gddm_a       no_def  gddm_a", flush=True)
    by_kb = {}
    for r in results:
        if r["K"] is None:
            continue
        if r["label"].endswith("_gddm_a"):
            suf = "gddm_a"; base = r["label"][:-len("_gddm_a")]
        else:
            suf = "nodef"; base = r["label"][:-len("_nodef")]
        by_kb.setdefault(base, {})[suf] = r["asr"]

    for K in Ks:
        line = f"  K={K:>2} "
        for b in budgets:
            base = f"v2_K{K}_b{b}"
            n = by_kb.get(base, {}).get("nodef", float("nan"))
            g = by_kb.get(base, {}).get("gddm_a", float("nan"))
            line += f"   {n:>5.3f}  {g:>5.3f}     "
        print(line, flush=True)

    # Save
    out = Path(__file__).parent / "results" / f"v2_k_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": results, "Ks": Ks, "budgets": budgets,
                                "n_graphs": len(graphs)}, indent=2, default=str))
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
