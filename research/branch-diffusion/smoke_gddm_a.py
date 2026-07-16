#!/usr/bin/env python3
"""
smoke_gddm_a.py — Smoke test for the A-channel GDDM defense.

Checks three properties (channel-asymmetry hypothesis):
  (a) clean acc with gddm-A purification stays within ~3 pp of no-defense
  (b) under zo_free (KATZE-family node-injection w/ extreme features):
        GDDM-A should drop ASR *materially* where X-channel GDDM does NOT.
  (c) under sgdi (distribution-matched features, baseline edges):
        GDDM-A should drop ASR SOME — denoiser pruning off-distribution edges.
  (d) under specgdi (spectrum-preserving edges):
        GDDM-A drop should be MINIMAL — that's the SpecGDI thesis.
"""
import json
import time
from copy import deepcopy
from pathlib import Path
import torch

from prepare import MODELS_DIR, DEVICE, GCN_GraphClassification
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from sweep_stealth import load_fold as load_fold_discrete
from defenses import (DefendedModel, gddm_defense, gddm_ckpt_path,
                      gddm_a_defense, gddm_a_ckpt_path)
import attack


FEAT_MODES_FOR_SMOKE = ["zo_free", "sgdi", "specgdi"]
# specgdi attack is the slowest feat_mode (spectral computations per candidate);
# cap to fewer graphs to keep total smoke runtime under ~20 min.
N_GRAPHS_SPECGDI = 20

# Where to dump per-row JSON immediately as we go (so a kill mid-run keeps partials)
_PARTIAL_PATH = Path(__file__).parent / "results" / "smoke_gddm_a.json"


def clean_acc(model, graphs):
    if not graphs:
        return 0.0
    c = 0
    for g in graphs:
        with torch.no_grad():
            if model(g).argmax(-1).item() == g.y.item():
                c += 1
    return c / len(graphs)


def smoke_one(ds_name, model, graphs, t_star_a=30, t_star_x=10):
    ckpt_a = gddm_a_ckpt_path(ds_name)
    ckpt_x = gddm_ckpt_path(ds_name)
    print(f"\n--- {ds_name} (t_star_A={t_star_a}, t_star_X={t_star_x}) ---")

    # Clean-acc rows
    cln_undef = clean_acc(model, graphs)
    dmodel_a = DefendedModel(model, gddm_a_defense, "gddm_a",
                             ckpt_path=ckpt_a, t_star=t_star_a)
    cln_a = clean_acc(dmodel_a, graphs)
    dmodel_x = DefendedModel(model, gddm_defense, "gddm",
                             ckpt_path=ckpt_x, t_star=t_star_x)
    cln_x = clean_acc(dmodel_x, graphs)
    print(f"  clean_acc  no_def={cln_undef:.3f}  +gddm_x={cln_x:.3f}  +gddm_a={cln_a:.3f}")
    print(f"  Δ clean    gddm_x={(cln_x-cln_undef)*100:+.1f}pp   "
          f"gddm_a={(cln_a-cln_undef)*100:+.1f}pp")

    # ASR rows
    rows = []
    for fm in FEAT_MODES_FOR_SMOKE:
        # specgdi is the slowest feat_mode — cap graphs to keep runtime sane.
        gs = graphs[:N_GRAPHS_SPECGDI] if fm == "specgdi" else graphs
        cfg = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 20,
               "zo_restarts": 2, "feat_mode": fm, "seed_offset": 0}
        import sys
        t0 = time.time()
        o_undef = attack.run_attack_sweep(model, gs, DEVICE, cfg); sys.stdout.flush()
        t1 = time.time()
        o_x = attack.run_attack_sweep(dmodel_x, gs, DEVICE, cfg); sys.stdout.flush()
        t2 = time.time()
        o_a = attack.run_attack_sweep(dmodel_a, gs, DEVICE, cfg); sys.stdout.flush()
        t3 = time.time()
        a_no, a_x, a_a = o_undef["asr_full"], o_x["asr_full"], o_a["asr_full"]
        d_x = (a_no - a_x) * 100
        d_a = (a_no - a_a) * 100
        print(f"  [{fm:<8} n={len(gs)}] ASR no={a_no:.3f}  gddm_x={a_x:.3f} (Δ={d_x:+.1f}pp)"
              f"  gddm_a={a_a:.3f} (Δ={d_a:+.1f}pp)   "
              f"[{t1-t0:.0f}s/{t2-t1:.0f}s/{t3-t2:.0f}s]", flush=True)
        rows.append({"feat_mode": fm, "n_graphs": len(gs),
                     "asr_no": a_no, "asr_gddm_x": a_x,
                     "asr_gddm_a": a_a, "drop_gddm_x_pp": d_x,
                     "drop_gddm_a_pp": d_a})
        # Persist partial results so a mid-run kill keeps prior rows
        _PARTIAL_PATH.parent.mkdir(exist_ok=True)
        _PARTIAL_PATH.write_text(json.dumps({
            "partial": True, "dataset": ds_name, "rows": rows,
            "clean_acc_no_def": cln_undef, "clean_acc_gddm_x": cln_x,
            "clean_acc_gddm_a": cln_a}, indent=2))

    return {"dataset": ds_name, "t_star_a": t_star_a, "t_star_x": t_star_x,
            "clean_acc_no_def": cln_undef, "clean_acc_gddm_x": cln_x,
            "clean_acc_gddm_a": cln_a, "n_graphs": len(graphs), "asr_rows": rows}


def main():
    out = []

    def _checkpoint(label):
        _PARTIAL_PATH.parent.mkdir(exist_ok=True)
        _PARTIAL_PATH.write_text(json.dumps(
            {"partial": False, "phase": label, "results": out}, indent=2))

    # PROTEINS (discrete one-hot, n_feat=3)
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    print(f"PROTEINS fold0: n_graphs={len(graphs)}", flush=True)
    graphs_p = graphs[:40]
    out.append(smoke_one("PROTEINS", model, graphs_p, t_star_a=30, t_star_x=20))
    _checkpoint("after-PROTEINS")

    # ENZYMES (continuous z-scored, n_feat=21)
    meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    fi = meta["folds"][0]
    model_e = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    mp = fi["model_path"]
    if not Path(mp).exists():
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
    print(f"ENZYMES fold0: n_attackable={len(graphs_e)}", flush=True)
    graphs_e = graphs_e[:40]
    out.append(smoke_one("ENZYMES", model_e, graphs_e, t_star_a=30, t_star_x=10))
    _checkpoint("after-ENZYMES")

    # ---- summary ----
    print("\n=== SUMMARY ===")
    for r in out:
        ds = r["dataset"]
        print(f"\n  {ds}: clean Δ gddm_x={(r['clean_acc_gddm_x']-r['clean_acc_no_def'])*100:+.1f}pp"
              f"  gddm_a={(r['clean_acc_gddm_a']-r['clean_acc_no_def'])*100:+.1f}pp")
        for row in r["asr_rows"]:
            print(f"    {row['feat_mode']:<10} no={row['asr_no']:.3f}  "
                  f"gddm_x={row['asr_gddm_x']:.3f} (Δ={row['drop_gddm_x_pp']:+.1f}pp)  "
                  f"gddm_a={row['asr_gddm_a']:.3f} (Δ={row['drop_gddm_a_pp']:+.1f}pp)")

    # Save JSON record alongside log
    out_path = Path(__file__).parent / "results" / "smoke_gddm_a.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
