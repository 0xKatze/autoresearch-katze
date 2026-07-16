#!/usr/bin/env python3
"""ENZYMES sgdi + specgdi only, n=15 each. Cheap pass."""
import json, time
from copy import deepcopy
from pathlib import Path
import torch
from prepare import DEVICE, GCN_GraphClassification
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from defenses import (DefendedModel, gddm_defense, gddm_ckpt_path,
                      gddm_a_defense, gddm_a_ckpt_path)
import attack

OUT = Path(__file__).parent / "results" / "smoke_gddm_a_enzymes_rest.json"

def run_fm(model, dm_x, dm_a, graphs, fm):
    cfg = {"node_budget": 1, "skip_greedy": True, "zo_epochs": 20,
           "zo_restarts": 2, "feat_mode": fm, "seed_offset": 0}
    t0 = time.time(); o_no = attack.run_attack_sweep(model, graphs, DEVICE, cfg)
    t1 = time.time(); o_x  = attack.run_attack_sweep(dm_x,  graphs, DEVICE, cfg)
    t2 = time.time(); o_a  = attack.run_attack_sweep(dm_a,  graphs, DEVICE, cfg)
    t3 = time.time()
    r = {"feat_mode": fm, "n_graphs": len(graphs),
         "asr_no": o_no["asr_full"], "asr_gddm_x": o_x["asr_full"],
         "asr_gddm_a": o_a["asr_full"],
         "drop_gddm_x_pp": (o_no["asr_full"]-o_x["asr_full"])*100,
         "drop_gddm_a_pp": (o_no["asr_full"]-o_a["asr_full"])*100,
         "secs": [t1-t0, t2-t1, t3-t2]}
    print(f"  [ENZYMES/{fm} n={len(graphs)}] no={r['asr_no']:.3f} "
          f"gddm_x={r['asr_gddm_x']:.3f} (Δ={r['drop_gddm_x_pp']:+.1f}pp) "
          f"gddm_a={r['asr_gddm_a']:.3f} (Δ={r['drop_gddm_a_pp']:+.1f}pp) "
          f"[{t1-t0:.0f}s/{t2-t1:.0f}s/{t3-t2:.0f}s]", flush=True)
    cur = json.loads(OUT.read_text()) if OUT.exists() else {"rows": []}
    cur["rows"].append(r); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(cur, indent=2))

def main():
    meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    fi = meta["folds"][0]
    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    mp = fi["model_path"]
    if not Path(mp).exists(): mp = str(MODELS_CONT_DIR / Path(mp).name)
    model.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True)); model.eval()
    graphs = []
    for i in fi["test_idx"]:
        g = deepcopy(graphs_all[i]).to(DEVICE)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if model(g).argmax(-1).item() == g.y.item(): graphs.append(g)
    print(f"ENZYMES fold0: n_attackable={len(graphs)}", flush=True)
    g15 = graphs[:15]
    dm_a = DefendedModel(model, gddm_a_defense, "gddm_a", ckpt_path=gddm_a_ckpt_path("ENZYMES"), t_star=30)
    dm_x = DefendedModel(model, gddm_defense, "gddm", ckpt_path=gddm_ckpt_path("ENZYMES"), t_star=10)
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps({"rows": []}, indent=2))
    run_fm(model, dm_x, dm_a, g15, "sgdi")
    run_fm(model, dm_x, dm_a, g15, "specgdi")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
