#!/usr/bin/env python3
"""eval_clean_acc_gddm.py — Clean (no-attack) graph-classification accuracy
under {none, gddm_x, gddm_a} defenses, on the FULL test split (not the
"attackable" subset). Per-fold + 5-fold mean ± std.

Purpose: confirm whether GDDM defenses are usable on the graph-classification
task that KATZE n23 evaluates on, given that GDDM was published for
node-classification (Cora/Citeseer/Pubmed). See `feedback-disclose-adaptations`
memory for the cross-task transplant disclosures.

Datasets:
  - PROTEINS  (TUDataset, 2-class, mean n ~27)
  - ENZYMES   (TUDataset w/ continuous attrs, 6-class, mean n ~33)
"""
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Subset

from prepare import MODELS_DIR, DEVICE, GCN_GraphClassification
from torch_geometric.datasets import TUDataset
from prepare import DATA_DIR
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
from defenses import (DefendedModel, gddm_defense, gddm_a_defense,
                      gddm_ckpt_path, gddm_a_ckpt_path)


T_STAR_X = 20
T_STAR_A = 30


def _eval_full_test(model, graphs):
    correct = 0
    total = 0
    for g in graphs:
        g = g.to(DEVICE)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred = model(g).argmax(dim=-1).item()
        if pred == g.y.item():
            correct += 1
        total += 1
    return correct / total if total else 0.0, total


def load_proteins_fold(fi, meta):
    """Return (model, full_test_graphs) for PROTEINS fold."""
    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE, weights_only=True))
    model.eval()
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    test_graphs = []
    for data in Subset(ds, fi["test_idx"]):
        test_graphs.append(data.clone())
    return model, test_graphs


def load_enzymes_fold(fi, meta):
    """Return (model, full_test_graphs) for ENZYMES fold."""
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    mp = fi["model_path"]
    if not Path(mp).exists():
        mp = str(MODELS_CONT_DIR / Path(mp).name)
    model.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
    model.eval()
    test_graphs = [graphs_all[i].clone() for i in fi["test_idx"]]
    return model, test_graphs


def bench_dataset(name, folds_data, ckpt_x, ckpt_a):
    rows = []
    for fold_idx, (model, graphs) in enumerate(folds_data):
        print(f"\n--- {name} fold {fold_idx} (n_test={len(graphs)}) ---", flush=True)
        # no defense
        t0 = time.time()
        acc_no, n = _eval_full_test(model, graphs)
        print(f"  no_def    acc={acc_no:.4f}  wall={time.time()-t0:.1f}s  n={n}", flush=True)
        # gddm_x
        t0 = time.time()
        dm_x = DefendedModel(model, gddm_defense, "gddm_x",
                             ckpt_path=ckpt_x, t_star=T_STAR_X)
        acc_x, _ = _eval_full_test(dm_x, graphs)
        print(f"  gddm_x    acc={acc_x:.4f}  wall={time.time()-t0:.1f}s  "
              f"Δ={(acc_x-acc_no)*100:+.2f}pp", flush=True)
        # gddm_a
        t0 = time.time()
        dm_a = DefendedModel(model, gddm_a_defense, "gddm_a",
                             ckpt_path=ckpt_a, t_star=T_STAR_A)
        acc_a, _ = _eval_full_test(dm_a, graphs)
        print(f"  gddm_a    acc={acc_a:.4f}  wall={time.time()-t0:.1f}s  "
              f"Δ={(acc_a-acc_no)*100:+.2f}pp", flush=True)
        rows.append({"dataset": name, "fold": fold_idx, "n_test": n,
                     "acc_no": acc_no, "acc_gddm_x": acc_x, "acc_gddm_a": acc_a,
                     "drop_x_pp": (acc_x - acc_no) * 100.0,
                     "drop_a_pp": (acc_a - acc_no) * 100.0})
    return rows


def main():
    print(f"=== Clean-acc benchmark — gddm_x (t*={T_STAR_X}) / gddm_a (t*={T_STAR_A}) ===\n")

    # ---- PROTEINS ----
    meta_p = json.loads((MODELS_DIR / "meta.json").read_text())
    proteins_folds = []
    for fi in meta_p["folds"]:
        m, g = load_proteins_fold(fi, meta_p)
        proteins_folds.append((m, g))
        print(f"  PROTEINS fold {fi['fold_idx']} loaded: {len(g)} test graphs")
    rows = bench_dataset("PROTEINS", proteins_folds,
                          gddm_ckpt_path("PROTEINS"), gddm_a_ckpt_path("PROTEINS"))

    # ---- ENZYMES ----
    meta_e = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    enzymes_folds = []
    for fi in meta_e["folds"]:
        m, g = load_enzymes_fold(fi, meta_e)
        enzymes_folds.append((m, g))
        print(f"  ENZYMES fold {fi['fold_idx']} loaded: {len(g)} test graphs")
    rows += bench_dataset("ENZYMES", enzymes_folds,
                          gddm_ckpt_path("ENZYMES"), gddm_a_ckpt_path("ENZYMES"))

    # ---- aggregate ----
    print("\n=== 5-FOLD MEAN ± STD (full test split) ===")
    print(f"  {'dataset':<10}  {'no_def':>14}  {'gddm_x':>20}  {'gddm_a':>20}")
    summary = {}
    for ds in ("PROTEINS", "ENZYMES"):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        an = [r["acc_no"] for r in ds_rows]
        ax = [r["acc_gddm_x"] for r in ds_rows]
        aa = [r["acc_gddm_a"] for r in ds_rows]
        dx = [r["drop_x_pp"] for r in ds_rows]
        da = [r["drop_a_pp"] for r in ds_rows]
        def ms(xs):
            mu = sum(xs) / len(xs)
            sd = math.sqrt(sum((x-mu)**2 for x in xs) / max(len(xs)-1, 1))
            return mu, sd
        mn, sn = ms(an); mx, sx = ms(ax); ma, sa = ms(aa)
        mdx, sdx = ms(dx); mda, sda = ms(da)
        print(f"  {ds:<10}  {mn:.4f}±{sn:.4f}  "
              f"{mx:.4f}±{sx:.4f} (Δ={mdx:+.2f}±{sdx:.2f}pp)  "
              f"{ma:.4f}±{sa:.4f} (Δ={mda:+.2f}±{sda:.2f}pp)")
        summary[ds] = {"acc_no_mean": mn, "acc_no_std": sn,
                       "acc_gddm_x_mean": mx, "acc_gddm_x_std": sx,
                       "acc_gddm_a_mean": ma, "acc_gddm_a_std": sa,
                       "drop_x_pp_mean": mdx, "drop_x_pp_std": sdx,
                       "drop_a_pp_mean": mda, "drop_a_pp_std": sda}

    out = Path(__file__).parent / "results" / f"clean_acc_gddm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary,
                                "t_star_x": T_STAR_X, "t_star_a": T_STAR_A},
                              indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
