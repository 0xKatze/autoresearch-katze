#!/usr/bin/env python3
"""smoke_specgdi_v2_ablation.py

Determine whether SpecGDI v2's PROTEINS gddm_a ASR=0.000 is a tuneable-knob
failure (bug) or a structural failure (real null), by sweeping the two
suspected culprits identified in the post-v2 retro:

  (a) v2_budget_VU_per_node — too tight at 0.2 (cross-edges starved)
  (b) tau_composite spectral projection — drops adversarial edges to stay
      under the random-clean-insertion threshold

Also tests K=1 (collapse to single-node v2 geometry, isolating multi-node
contribution from the joint-update contribution).

PROTEINS only, fold 0, n<=50, 15 graphs. ~5 min wall.
"""
import json
import math
import time
from pathlib import Path

from prepare import MODELS_DIR, DEVICE
from sweep_stealth import load_fold as load_fold_discrete
from defenses import DefendedModel, gddm_a_defense, gddm_a_ckpt_path
import attack


SMOKE_MAX_GRAPHS = 30
SMOKE_MIN_N = 30
SMOKE_MAX_N = 80
T_DIFF = 10
SEED = 0

# PROTEINS calibrated values (from specgdi_v2_thresholds.json)
PROTEINS_TAU_CAL = 1.5845
PROTEINS_DENSITY_CAL = 0.6253

# v1 PROTEINS τ_spec (for K=1 variant only — irrelevant for v2)
TAU_OFF = 1e9


def filter_small(graphs):
    out = [g for g in graphs if SMOKE_MIN_N <= g.num_nodes <= SMOKE_MAX_N]
    return out[:SMOKE_MAX_GRAPHS]


def run_one(label, model, graphs, cfg_extra):
    cfg = {
        "node_budget": cfg_extra.get("v2_K", 3) if cfg_extra.get("feat_mode") == "specgdi_v2" else 1,
        "skip_greedy": True,
        "zo_epochs": 10,
        "zo_restarts": 1,
        "seed_offset": SEED,
        "sgdi_T_diff": T_DIFF,
        "v2_T_diff": T_DIFF,
        # force the fallback path in _load_specgdi_v2_thresholds so our cfg
        # overrides take effect (PROTEINS_ABL is not in the threshold JSON):
        "specgdi_dataset_key": "PROTEINS_ABL",
        "v2_density_fallback": PROTEINS_DENSITY_CAL,
        **cfg_extra,
    }
    t0 = time.time()
    o = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    dt = time.time() - t0
    asr = o["asr_full"]
    n_g = max(o.get("n_graphs", len(graphs)), 1)
    per_g = dt / n_g
    sl = o.get("s_spec_per_graph") or []
    sl_f = [x for x in sl if isinstance(x, float) and not math.isnan(x)]
    s_med = sorted(sl_f)[len(sl_f) // 2] if sl_f else float("nan")
    tau = o.get("tau_spec") or o.get("tau_spec_composite")
    print(f"  [{label:<36}] ASR={asr:.3f}  n_g={n_g}  wall={dt:.1f}s  per_g={per_g:.1f}s  "
          f"S_spec_med={s_med:.4f}  tau={tau}")
    return {
        "label": label, "asr": asr, "n_g": n_g, "wall_s": dt, "per_g_s": per_g,
        "queries": o.get("queries", 0), "s_spec_med": s_med, "tau": tau,
    }


def main():
    print(f"=== SpecGDI v2 ABLATION — PROTEINS fold 0, n<= {SMOKE_MAX_N}, "
          f"up to {SMOKE_MAX_GRAPHS} graphs ===\n")

    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    graphs = filter_small(graphs)
    print(f"  loaded {len(graphs)} PROTEINS small graphs (n<= {SMOKE_MAX_N})\n")

    ckpt = gddm_a_ckpt_path("PROTEINS")
    dmodel = DefendedModel(model, gddm_a_defense, "gddm_a",
                           ckpt_path=ckpt, t_star=30)

    # 4 ablation variants × {no_def, gddm_a} = 8 runs
    variants = [
        # (variant_name, cfg_overrides_for_v2)
        ("v2-original", {
            "v2_K": 3, "v2_budget_VU_per_node": 0.2,
            "v2_tau_composite_fallback": PROTEINS_TAU_CAL,
        }),
        ("v2-loose-budget", {
            "v2_K": 3, "v2_budget_VU_per_node": 0.5,
            "v2_tau_composite_fallback": PROTEINS_TAU_CAL,
        }),
        ("v2-spec-off", {
            "v2_K": 3, "v2_budget_VU_per_node": 0.5,
            "v2_tau_composite_fallback": TAU_OFF,
        }),
        ("v2-K1-joint", {  # K=1 + joint update, no spec/budget limits
            "v2_K": 1, "v2_budget_VU_per_node": 1.0,
            "v2_tau_composite_fallback": TAU_OFF,
        }),
    ]

    results = []
    for vname, cfg_extra in variants:
        feat_cfg = {"feat_mode": "specgdi_v2", **cfg_extra}
        print(f"-- {vname} --")
        results.append(run_one(f"{vname}_nodef",   model, graphs, feat_cfg))
        results.append(run_one(f"{vname}_gddm_a", dmodel, graphs, feat_cfg))
        print()

    # Anchors: v1 specgdi + sgdi (same 15 graphs) for direct comparison
    print("-- anchors (v1, sgdi) --")
    results.append(run_one("anchor_sgdi_nodef",       model, graphs, {"feat_mode": "sgdi"}))
    results.append(run_one("anchor_sgdi_gddm_a",     dmodel, graphs, {"feat_mode": "sgdi"}))
    results.append(run_one("anchor_specgdi_v1_nodef",  model, graphs, {"feat_mode": "specgdi"}))
    results.append(run_one("anchor_specgdi_v1_gddm_a", dmodel, graphs, {"feat_mode": "specgdi"}))

    print(f"\n=== ABLATION SUMMARY (PROTEINS, {SMOKE_MIN_N}<=n<={SMOKE_MAX_N}, {len(graphs)} graphs) ===")
    print(f"  {'variant':<36}  ASR_nodef  ASR_gddm_a  Δ(gddm−no)")
    pair = {}
    for r in results:
        label = r["label"]
        if label.endswith("_gddm_a"):
            key, suffix = label[:-len("_gddm_a")], "gddm_a"
        elif label.endswith("_nodef"):
            key, suffix = label[:-len("_nodef")], "nodef"
        else:
            continue
        pair.setdefault(key, {})[suffix] = r["asr"]
    for k, v in pair.items():
        a0 = v.get("nodef", float("nan")); a1 = v.get("gddm_a", float("nan"))
        d = (a1 - a0) if (a0 == a0 and a1 == a1) else float("nan")
        print(f"  {k:<36}  {a0:>8.3f}   {a1:>8.3f}   {d:>+8.3f}")

    out_path = Path(__file__).parent / "smoke_specgdi_v2_ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
