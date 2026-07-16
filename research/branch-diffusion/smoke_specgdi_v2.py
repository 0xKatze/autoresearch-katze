#!/usr/bin/env python3
"""
smoke_specgdi_v2.py — Smoke test for SpecGDI v2 (joint K-node trojan diffusion).

Runs on PROTEINS fold 0, seed 0, K=3, with a small graph subset (n <= 50, up to
15 graphs). Compares ASR + final S_spec_composite vs SpecGDI v1 and SGDI under:
  - no defense
  - gddm_a (the headline make-or-break defense)
"""
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch

from prepare import MODELS_DIR, DEVICE
from sweep_stealth import load_fold as load_fold_discrete
from defenses import DefendedModel, gddm_a_defense, gddm_a_ckpt_path
import attack


SMOKE_MAX_GRAPHS = 15
SMOKE_MAX_N = 50
K = 3
T_DIFF = 10
SEED = 0


def filter_small(graphs):
    out = [g for g in graphs if g.num_nodes <= SMOKE_MAX_N]
    return out[:SMOKE_MAX_GRAPHS]


def run_one(label, model, graphs, cfg_extra, t_diff=T_DIFF):
    cfg = {
        "node_budget": 1 if cfg_extra.get("feat_mode") != "specgdi_v2" else K,
        "skip_greedy": True,
        "zo_epochs": 10,
        "zo_restarts": 1,
        "seed_offset": SEED,
        "sgdi_T_diff": t_diff,
        "v2_T_diff": t_diff,
        "v2_K": K,
        **cfg_extra,
    }
    t0 = time.time()
    o = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    dt = time.time() - t0
    asr = o["asr_full"]
    n_g = max(o.get("n_graphs", len(graphs)), 1)
    per_g = dt / n_g
    print(f"  [{label:<28}] ASR={asr:.3f}  n_g={n_g}  wall={dt:.1f}s  per_g={per_g:.1f}s")
    return {"label": label, "asr": asr, "wall_s": dt, "per_g_s": per_g,
            "queries": o.get("queries", 0),
            "s_spec_per_graph": o.get("s_spec_per_graph"),
            "tau_spec": o.get("tau_spec") or o.get("tau_spec_composite"),
            }


def _load_enzymes():
    """Load ENZYMES fold 0, correctly-classified attackable graphs (small only)."""
    from prepare import GCN_GraphClassification
    from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
    meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    fi = meta["folds"][0]
    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    mp = fi["model_path"]
    if not Path(mp).exists():
        mp = str(MODELS_CONT_DIR / Path(mp).name)
    model.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    model.eval()
    graphs = []
    for i in fi["test_idx"]:
        g = deepcopy(graphs_all[i]).to(DEVICE)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if model(g).argmax(-1).item() == g.y.item():
                graphs.append(g)
    return model, filter_small(graphs)


def smoke_one_dataset(ds, model, graphs, gddm_ckpt):
    print(f"\n=== {ds} smoke — fold 0, K={K}, T_diff={T_DIFF}, "
          f"n<= {SMOKE_MAX_N}, n_graphs={len(graphs)} ===")
    if not graphs:
        return []
    dmodel = DefendedModel(model, gddm_a_defense, "gddm_a",
                           ckpt_path=gddm_ckpt, t_star=30)
    results = []
    print("-- no defense --")
    results.append(run_one(f"{ds}_sgdi_nodef",       model, graphs, {"feat_mode": "sgdi"}))
    results.append(run_one(f"{ds}_specgdi_v1_nodef", model, graphs, {"feat_mode": "specgdi"}))
    results.append(run_one(f"{ds}_specgdi_v2_nodef", model, graphs, {"feat_mode": "specgdi_v2"}))
    print("-- gddm_a (headline) --")
    results.append(run_one(f"{ds}_sgdi_gddm_a",       dmodel, graphs, {"feat_mode": "sgdi"}))
    results.append(run_one(f"{ds}_specgdi_v1_gddm_a", dmodel, graphs, {"feat_mode": "specgdi"}))
    results.append(run_one(f"{ds}_specgdi_v2_gddm_a", dmodel, graphs, {"feat_mode": "specgdi_v2"}))
    return results


def main():
    print(f"=== SpecGDI v2 smoke — K={K}, T_diff={T_DIFF}, "
          f"n<= {SMOKE_MAX_N}, up to {SMOKE_MAX_GRAPHS} graphs ===")

    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    graphs = filter_small(graphs)
    print(f"  loaded {len(graphs)} PROTEINS small graphs (n<= {SMOKE_MAX_N})")

    ckpt = gddm_a_ckpt_path("PROTEINS")
    results = smoke_one_dataset("PROTEINS", model, graphs, ckpt)

    # also smoke ENZYMES — known stronger SGDI/gddm_a regime
    try:
        model_e, graphs_e = _load_enzymes()
        print(f"  loaded {len(graphs_e)} ENZYMES small graphs")
        ckpt_e = gddm_a_ckpt_path("ENZYMES")
        results += smoke_one_dataset("ENZYMES", model_e, graphs_e, ckpt_e)
    except Exception as e:
        print(f"  ENZYMES smoke skipped: {e}")

    print("\n=== SUMMARY ===")
    for r in results:
        sl = r.get("s_spec_per_graph") or []
        if sl:
            sl_f = [x for x in sl if isinstance(x, float) and not math.isnan(x)]
            med = sorted(sl_f)[len(sl_f)//2] if sl_f else float("nan")
        else:
            med = float("nan")
        print(f"  {r['label']:<26}  ASR={r['asr']:.3f}  per_g={r['per_g_s']:.1f}s  "
              f"S_spec_median={med:.4f}  tau={r.get('tau_spec')}")

    # save raw results
    out_path = Path(__file__).parent / "smoke_specgdi_v2_results.json"
    out_path.write_text(json.dumps([
        {k: (v if k != "s_spec_per_graph" else v) for k, v in r.items()}
        for r in results], indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
