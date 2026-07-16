#!/usr/bin/env python3
"""
smoke_gddm.py — Smoke test for the GDDM-style defense.

Checks two properties:
  (a) clean acc with gddm-purification stays within ~3 pp of no-defense
  (b) under a feature-OOD perturbation (zo_free with large feat_scale),
      ASR drops materially when gddm-purification is applied
"""
import json
import time
from copy import deepcopy
from pathlib import Path
import torch

from prepare import MODELS_DIR, DEVICE
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from sweep_stealth import load_fold as load_fold_discrete
from defenses import DefendedModel, gddm_defense, gddm_ckpt_path
import attack


def clean_acc(model, graphs):
    if not graphs:
        return 0.0
    c = 0
    for g in graphs:
        with torch.no_grad():
            if model(g).argmax(-1).item() == g.y.item():
                c += 1
    return c / len(graphs)


def smoke_one(ds_name, model, graphs, t_star=30):
    ckpt = gddm_ckpt_path(ds_name)
    print(f"\n--- {ds_name} (t_star={t_star}, ckpt={Path(ckpt).name}) ---")
    cln_undef = clean_acc(model, graphs)
    dmodel = DefendedModel(model, gddm_defense, "gddm", ckpt_path=ckpt, t_star=t_star)
    cln_def = clean_acc(dmodel, graphs)
    print(f"  clean_acc no_def = {cln_undef:.3f}   with_gddm = {cln_def:.3f}   "
          f"Δ = {(cln_def - cln_undef)*100:+.1f} pp")

    # Attack: zo_free with default feat_scale (set in attack.py); strong feature-OOD signal.
    cfg = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 20, "zo_restarts": 2,
           "feat_mode": "zo_free", "seed_offset": 0}
    t0 = time.time()
    o_undef = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    t1 = time.time()
    o_def = attack.run_attack_sweep(dmodel, graphs, DEVICE, cfg)
    t2 = time.time()
    print(f"  zo_free ASR  no_def = {o_undef['asr_full']:.3f}  ({t1-t0:.0f}s)")
    print(f"  zo_free ASR  + gddm = {o_def['asr_full']:.3f}  ({t2-t1:.0f}s)")
    print(f"  ASR drop = {(o_undef['asr_full'] - o_def['asr_full'])*100:+.1f} pp")
    return {
        "dataset": ds_name, "t_star": t_star,
        "clean_acc_no_def": cln_undef, "clean_acc_gddm": cln_def,
        "asr_zo_free_no_def": o_undef["asr_full"],
        "asr_zo_free_gddm": o_def["asr_full"],
        "n_graphs": len(graphs),
    }


def main():
    out = []

    # PROTEINS (discrete one-hot, n_feat=3)
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    print(f"PROTEINS fold0: n_graphs={len(graphs)}")
    # cap to ~40 graphs for speed
    graphs_p = graphs[:40]
    out.append(smoke_one("PROTEINS", model, graphs_p, t_star=20))

    # ENZYMES (continuous z-scored, n_feat=21)
    from prepare import GCN_GraphClassification
    meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    fi = meta["folds"][0]
    model_e = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    mp = fi["model_path"]
    if not Path(mp).exists():
        # meta json was written from a different repo root; fall back to local models_cont dir
        mp = str(MODELS_CONT_DIR / Path(mp).name)
    model_e.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    model_e.eval()
    graphs_e = []
    for i in fi["test_idx"]:
        g = deepcopy(graphs_all[i]).to(DEVICE)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if model_e(g).argmax(-1).item() == g.y.item():
                graphs_e.append(g)
    print(f"ENZYMES fold0: n_attackable={len(graphs_e)}")
    graphs_e = graphs_e[:40]
    out.append(smoke_one("ENZYMES", model_e, graphs_e, t_star=30))
    out.append(smoke_one("ENZYMES", model_e, graphs_e, t_star=10))
    out.append(smoke_one("ENZYMES", model_e, graphs_e, t_star=5))

    print("\n=== SUMMARY ===")
    for r in out:
        print(f"  {r['dataset']}: clean Δ={(r['clean_acc_gddm']-r['clean_acc_no_def'])*100:+.1f}pp  "
              f"zo_free ASR {r['asr_zo_free_no_def']:.3f} -> {r['asr_zo_free_gddm']:.3f}")


if __name__ == "__main__":
    main()
