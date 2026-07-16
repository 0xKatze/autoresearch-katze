#!/usr/bin/env python3
"""
smoke_gddm_a_finish.py — run ONLY the rows that smoke_gddm_a.py didn't complete
in the prior run (the specgdi rows + the entire ENZYMES section). Merges into the
partial JSON at results/smoke_gddm_a.json.
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

OUT = Path(__file__).parent / "results" / "smoke_gddm_a.json"


def clean_acc(model, graphs):
    if not graphs:
        return 0.0
    c = 0
    for g in graphs:
        with torch.no_grad():
            if model(g).argmax(-1).item() == g.y.item():
                c += 1
    return c / len(graphs)


def run_one_fm(ds_name, model, graphs, fm, t_star_a, t_star_x, log_path):
    ckpt_a = gddm_a_ckpt_path(ds_name)
    ckpt_x = gddm_ckpt_path(ds_name)
    dmodel_a = DefendedModel(model, gddm_a_defense, "gddm_a",
                             ckpt_path=ckpt_a, t_star=t_star_a)
    dmodel_x = DefendedModel(model, gddm_defense, "gddm",
                             ckpt_path=ckpt_x, t_star=t_star_x)
    cfg = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 20,
           "zo_restarts": 2, "feat_mode": fm, "seed_offset": 0}
    t0 = time.time()
    o_no = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    t1 = time.time()
    o_x = attack.run_attack_sweep(dmodel_x, graphs, DEVICE, cfg)
    t2 = time.time()
    o_a = attack.run_attack_sweep(dmodel_a, graphs, DEVICE, cfg)
    t3 = time.time()
    a_no, a_x, a_a = o_no["asr_full"], o_x["asr_full"], o_a["asr_full"]
    row = {"feat_mode": fm, "n_graphs": len(graphs),
           "asr_no": a_no, "asr_gddm_x": a_x, "asr_gddm_a": a_a,
           "drop_gddm_x_pp": (a_no - a_x) * 100,
           "drop_gddm_a_pp": (a_no - a_a) * 100,
           "secs": [t1 - t0, t2 - t1, t3 - t2]}
    print(f"  [{ds_name}/{fm} n={len(graphs)}] no={a_no:.3f} "
          f"gddm_x={a_x:.3f} (Δ={(a_no-a_x)*100:+.1f}pp) "
          f"gddm_a={a_a:.3f} (Δ={(a_no-a_a)*100:+.1f}pp) "
          f"[{t1-t0:.0f}s/{t2-t1:.0f}s/{t3-t2:.0f}s]", flush=True)
    # checkpoint
    cur = json.loads(OUT.read_text()) if OUT.exists() else {"rows_extra": []}
    cur.setdefault("rows_extra", []).append({"dataset": ds_name, **row})
    OUT.write_text(json.dumps(cur, indent=2))
    return row


def main():
    OUT.parent.mkdir(exist_ok=True)
    # ---- PROTEINS specgdi only (the missing row) ----
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    print(f"PROTEINS fold0: n_attackable={len(graphs)}", flush=True)
    gs = graphs[:10]
    run_one_fm("PROTEINS", model, gs, "specgdi", t_star_a=30, t_star_x=20,
               log_path=None)

    # ---- ENZYMES full (it didn't run before) ----
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
    g40 = graphs_e[:40]

    # clean acc
    ckpt_a = gddm_a_ckpt_path("ENZYMES")
    ckpt_x = gddm_ckpt_path("ENZYMES")
    dmodel_a = DefendedModel(model_e, gddm_a_defense, "gddm_a",
                             ckpt_path=ckpt_a, t_star=30)
    dmodel_x = DefendedModel(model_e, gddm_defense, "gddm",
                             ckpt_path=ckpt_x, t_star=10)
    cln_no = clean_acc(model_e, g40)
    cln_a = clean_acc(dmodel_a, g40)
    cln_x = clean_acc(dmodel_x, g40)
    print(f"ENZYMES clean: no={cln_no:.3f}  gddm_x={cln_x:.3f}  gddm_a={cln_a:.3f}", flush=True)
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur["enzymes_clean"] = {"no": cln_no, "gddm_x": cln_x, "gddm_a": cln_a}
    OUT.write_text(json.dumps(cur, indent=2))

    # ASR rows
    run_one_fm("ENZYMES", model_e, g40, "zo_free", t_star_a=30, t_star_x=10, log_path=None)
    run_one_fm("ENZYMES", model_e, g40, "sgdi", t_star_a=30, t_star_x=10, log_path=None)
    run_one_fm("ENZYMES", model_e, graphs_e[:10], "specgdi",
               t_star_a=30, t_star_x=10, log_path=None)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
