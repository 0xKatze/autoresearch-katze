#!/usr/bin/env python3
"""run_cell.py — single (dataset, feat_mode, defense) cell of the decisive grid.

Stdout final line is JSON record. Writes nothing to disk (caller handles incremental save).
CLI:
  --dataset {ENZYMES,PROTEINS}
  --feat-mode {zo_normcap,sgdi,specgdi,specgdi_signal,...}
  --defense {none,mahalanobis,gnnsafe,gddm,...}
  --seeds 0,1
  --max-n N            # skip test graphs with num_nodes > N (PROTEINS-capped)
  --specgdi-swaps K    # override SPECGDI_DEFAULTS specgdi_swaps_per_step
"""
import argparse, json, sys, time
from copy import deepcopy
import numpy as np
import torch

from prepare import MODELS_DIR, RESULTS_DIR, DEVICE, GCN_GraphClassification
from sweep_stealth import load_fold as load_fold_discrete
from prepare_cont import load_normalized_graphs, MODELS_CONT_DIR
import attack
from eval_paper_grade import build_defended, clean_acc_defended, ATTACK_CFG

# Defensive monkey-patch: node-dropping defenses (gnnsafe) can return batched
# logits whose first dim != len(data_list). Fall back to per-graph eval. Does
# NOT modify attack.py.
_orig_eval_graphs = attack._eval_graphs


def _safe_eval_graphs(model, data_list, true_label, device):
    if not data_list:
        return torch.empty(0)
    try:
        out = _orig_eval_graphs(model, data_list, true_label, device)
        if out.shape[0] == len(data_list):
            return out
    except Exception:
        pass
    margs = []
    for d in data_list:
        try:
            m = _orig_eval_graphs(model, [d], true_label, device)
            if m.shape[0] == 1:
                margs.append(m)
            else:
                margs.append(torch.tensor([0.0], device=device))
        except Exception:
            margs.append(torch.tensor([0.0], device=device))
    return torch.cat(margs, dim=0)


attack._eval_graphs = _safe_eval_graphs


def filter_by_size(graphs, max_n):
    if max_n is None:
        return graphs
    return [g for g in graphs if g.num_nodes <= max_n]


def load_proteins_folds(max_n):
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    out = []
    for fi in meta["folds"]:
        m, gs = load_fold_discrete(fi, meta)
        out.append((m, filter_by_size(gs, max_n)))
    return out


def load_enzymes_folds(max_n):
    meta = json.loads((MODELS_CONT_DIR / "meta_cont_ENZYMES.json").read_text())
    mean = torch.tensor(meta["feat_mean"]).view(1, -1)
    std = torch.tensor(meta["feat_std"]).view(1, -1)
    graphs_all, _, _, _, _ = load_normalized_graphs("ENZYMES", mean, std)
    out = []
    for fi in meta["folds"]:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(torch.load(fi["model_path"], map_location=DEVICE, weights_only=True))
        model.eval()
        gs = []
        for i in fi["test_idx"]:
            g = deepcopy(graphs_all[i]).to(DEVICE)
            g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                if model(g).argmax(-1).item() == g.y.item():
                    gs.append(g)
        out.append((model, filter_by_size(gs, max_n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ENZYMES", "PROTEINS"])
    ap.add_argument("--feat-mode", required=True)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--specgdi-swaps", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()

    if args.dataset == "ENZYMES":
        folds = load_enzymes_folds(args.max_n)
    else:
        folds = load_proteins_folds(args.max_n)
    folds = folds[: args.n_folds]
    sizes = [len(gs) for _, gs in folds]

    # Build defended models per fold (calibrate on filtered ref graphs)
    defended, cleanaccs = [], []
    for (model, graphs) in folds:
        if not graphs:
            defended.append(model)
            cleanaccs.append(0.0)
            continue
        dm = build_defended(model, args.defense, graphs, ds_key=args.dataset)
        defended.append(dm)
        cleanaccs.append(clean_acc_defended(dm, graphs))

    obs, qs = [], []
    for seed in seeds:
        for dm, (_, graphs) in zip(defended, folds):
            if not graphs:
                continue
            cfg = {**ATTACK_CFG, "feat_mode": args.feat_mode, "seed_offset": seed}
            if args.specgdi_swaps is not None:
                cfg["specgdi_swaps_per_step"] = args.specgdi_swaps
            o = attack.run_attack_sweep(dm, graphs, DEVICE, cfg)
            obs.append(o["asr_full"])
            qs.append(o["queries"] / max(o["n_graphs"], 1))

    rec = {
        "status": "ok",
        "dataset": args.dataset,
        "defense": args.defense,
        "feat_mode": args.feat_mode,
        "max_n": args.max_n,
        "specgdi_swaps": args.specgdi_swaps,
        "n_folds": len(folds),
        "fold_sizes": sizes,
        "asr_mean": float(np.mean(obs)) if obs else float("nan"),
        "asr_std": float(np.std(obs)) if obs else float("nan"),
        "asr_obs": [round(x, 4) for x in obs],
        "q_per_graph": float(np.mean(qs)) if qs else float("nan"),
        "clean_acc_def_mean": float(np.mean(cleanaccs)) if cleanaccs else float("nan"),
        "n_obs": len(obs),
        "wall_s": round(time.time() - t0, 2),
    }
    print("CELL_RESULT_JSON:" + json.dumps(rec))


if __name__ == "__main__":
    main()
