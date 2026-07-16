#!/usr/bin/env python3
"""
run_flow_b_defense_battery.py — diagnostic driver (NOT a modification of any
attack/model/defense module). Imports attack_flow_b.run_attack and the existing
DefendedModel/defense fns and runs FM-Velosteal (variant B, TRAINED velocity
field) THROUGH each defense, 3 folds x ~10 graphs, eval_num=200 (RGF), no W&B.

Defenses requested: {none, normfilter, mahalanobis, rspool, agnncert}.
  - none / normfilter / mahalanobis : implemented here via DefendedModel.
  - rspool (RSPoolGIN) / agnncert (official voting) : NOT implemented in this
    branch (only reference PDFs + a smoke-only _MiniRSPool readout). Emitted as
    null so the battery shape is explicit and the gap is auditable.

This is the same harness pattern as defense_eval.py / run_defended_sweep.py:
the attack queries the *defended* model, so it adapts to the filter.

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial python run_flow_b_defense_battery.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.datasets import TUDataset

import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs
DEVICE = "cpu"

import attack_flow_b  # heals env: registers canonical 'attack'
from defenses import DefendedModel, norm_filter_defense, mahalanobis_defense

# ---- diagnostic config -------------------------------------------------------
N_FOLDS = 3
GRAPH_CAP = 10            # ~10/fold -> ~30 total
EVAL_NUM = 200            # honored as RGF sample count (per task: eval_num 200)
RUNNABLE_DEFENSES = ["none", "normfilter", "mahalanobis"]
UNIMPLEMENTED_DEFENSES = ["rspool", "agnncert"]


class QueryCounter:
    """Tally victim graphs-per-forward (same wrapper as smoke_flow_b.py)."""
    def __init__(self, model):
        self.model = model
        self.n = 0
        self._orig = model.forward
    def _graphs(self, inp):
        g = getattr(inp, "num_graphs", None)
        if g is not None:
            return int(g)
        b = getattr(inp, "batch", None)
        if b is not None and b.numel() > 0:
            return int(b.max().item()) + 1
        return 1
    def __enter__(self):
        def counting_forward(*args, **kw):
            if args:
                self.n += self._graphs(args[0])
            self.model.forward = self._orig
            try:
                out = self._orig(*args, **kw)
            finally:
                self.model.forward = counting_forward
            return out
        self.model.forward = counting_forward
        return self
    def __exit__(self, *exc):
        self.model.forward = self._orig


def clean_test_acc(model, ds, test_idx):
    from torch.utils.data import Subset
    c = t = 0
    for data in Subset(ds, test_idx):
        data = data.to(DEVICE)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def build_defended(model, defense, ref_graphs):
    if defense == "none":
        return model
    if defense == "normfilter":
        return DefendedModel(model, norm_filter_defense, "normfilter", percentile=95)
    if defense == "mahalanobis":
        X = torch.cat([g.x for g in ref_graphs], dim=0).to(DEVICE).float()
        mean_ = X.mean(dim=0); d = X.size(1)
        cov = torch.cov(X.t()) + 1e-3 * torch.eye(d, device=DEVICE)
        cov_inv = torch.linalg.inv(cov)
        diff = X - mean_
        sqm = (diff @ cov_inv * diff).sum(dim=1)
        thr = float(torch.quantile(sqm, 0.995).item())
        return DefendedModel(model, mahalanobis_defense, "mahalanobis",
                             mean=mean_, cov_inv=cov_inv, threshold=thr)
    raise ValueError(defense)


def main():
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])

    # eval_num 200 -> RGF with 200 samples (CGE on d=3 would be exact at 6 q;
    # honoring the requested eval_num as the per-step query budget).
    attack_flow_b.CONFIG["grad_method"] = "rgf"
    attack_flow_b.CONFIG["rgf_samples"] = EVAL_NUM
    attack_flow_b.CONFIG["feat_scale"] = "auto_x2"
    attack_flow_b.CONFIG["node_budget"] = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== FM-Velosteal (flow_b) defense battery | {meta['dataset']} GCN | "
          f"{N_FOLDS} folds x<= {GRAPH_CAP} graphs | eval_num(RGF)={EVAL_NUM} | "
          f"device={DEVICE} | {stamp} ===")

    per_fold = {dn: [] for dn in RUNNABLE_DEFENSES}
    total_queries = 0
    victim_accs = []
    fold_records = []
    t0 = time.time()

    for fi in range(N_FOLDS):
        fold_info = meta["folds"][fi]
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
        model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
        model.eval()

        vacc = clean_test_acc(model, ds, fold_info["test_idx"])
        victim_accs.append(vacc)

        graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:GRAPH_CAP]
        rec = {"fold": fi, "victim_acc": round(vacc, 4), "n_graphs": len(graphs), "asr": {}, "queries": {}}
        for dn in RUNNABLE_DEFENSES:
            attack_flow_b._FIELD_CACHE.clear()  # refit field per defended target
            victim = build_defended(model, dn, graphs)
            with QueryCounter(victim) as qc:
                asr = attack_flow_b.run_attack(victim, graphs, DEVICE)
            per_fold[dn].append(asr)
            total_queries += qc.n
            rec["asr"][dn] = round(asr, 4)
            rec["queries"][dn] = qc.n
            print(f"  [fold {fi}] {dn:<12} ASR={asr*100:6.2f}%  q={qc.n:7d}  "
                  f"vacc={vacc:.3f}  n={len(graphs)}")
        fold_records.append(rec)

    asr_by_defense = {dn: round(mean(per_fold[dn]) * 100, 2) for dn in RUNNABLE_DEFENSES}
    for dn in UNIMPLEMENTED_DEFENSES:
        asr_by_defense[dn] = None
    fold_std_by_defense = {dn: round(pstdev(per_fold[dn]) * 100, 2) if len(per_fold[dn]) > 1 else 0.0
                           for dn in RUNNABLE_DEFENSES}
    victim_acc = round(mean(victim_accs), 4)
    headline_fold_std = fold_std_by_defense["none"]

    out = {
        "stamp": stamp,
        "experiment": "FM-Velosteal (attack_flow_b, variant B: trained velocity field)",
        "dataset": meta["dataset"],
        "victim_model": "GCN",
        "mode": "diagnostic",
        "n_folds": N_FOLDS,
        "graph_cap_per_fold": GRAPH_CAP,
        "eval_num_rgf": EVAL_NUM,
        "grad_method": "rgf",
        "feat_scale": "auto_x2",
        "node_budget": 1,
        "wandb": False,
        "asr_by_defense_pct": asr_by_defense,
        "fold_std_by_defense_pct": fold_std_by_defense,
        "victim_acc_mean": victim_acc,
        "victim_acc_per_fold": [round(v, 4) for v in victim_accs],
        "total_queries": total_queries,
        "headline_fold_std_pct": headline_fold_std,
        "unimplemented_defenses": UNIMPLEMENTED_DEFENSES,
        "unimplemented_note": ("RS-Pool (RSPoolGIN) and AGNNCert (official voting) "
                               "are not implemented as runnable defenses in this branch "
                               "(reference PDFs + smoke-only _MiniRSPool readout only); "
                               "emitted as null."),
        "wall_s": round(time.time() - t0, 1),
        "fold_records": fold_records,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flow_b_defense_battery_{meta['dataset']}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("=" * 64)
    print(f"asr_by_defense(%): {asr_by_defense}")
    print(f"fold_std(%):       {fold_std_by_defense}")
    print(f"victim_acc:        {victim_acc}  per-fold={[round(v,3) for v in victim_accs]}")
    print(f"total_queries:     {total_queries}")
    print(f"Saved {out_path}  [{out['wall_s']}s]")


if __name__ == "__main__":
    main()
