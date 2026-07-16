#!/usr/bin/env python3
"""eval_paper_grade_floor.py — DECISIVE paper-grade worst-case-defense FLOOR study
(N28 FM / orthogonal-injection line).

H1 (coordinator-reframed, MULTI-NODE): the orthogonal combination — a MULTI-NODE
(node_budget=10), low-degree-per-node ("edge_budget=1" mass channel: perp evades the
STRUCTURE defense degfilter) + on-manifold features at feat_scale=1.0 (perp evades the
DISTRIBUTION defense mahalanobis) — is the ONLY attack with a HIGH worst-case-defense
floor, while every SINGLE-AXIS attack (and the FEEBLE single-node nb=1 version)
collapses on >=1 defense axis.

floor(attack) := min ASR over EFFECTIVE defenses {normfilter, mahalanobis, degfilter}
(excludes 'none'). H1 holds iff the multi-node candidate's floor >> next-best config's
floor, paired-significant (p<0.01) over 10-fold x 3-seed = 30 (fold,seed) units.

DRIVER ONLY — no attack/model/defense code modified. Imports existing entry points
and runs each THROUGH each defense on the SAME victim per cell. The OM2C multi-node
candidate is run via its SPARSE GREEDY regime (composed from OM2C's OWN exported
functions om2c_prior / om2c_feature_opt / _candidate_pool / _build / _eval_graphs).
The OM2C dense frac_anchors (asr_full, rho=1.0) path is intentionally NOT used: at
rho=1.0 the full-edge nb=10 injection inflates EVERY node's degree past the degfilter
threshold so the whole graph is pruned -> empty-graph batch-size crash. The greedy
sparse regime IS the degree-1 mass channel under test, so this is the correct metric.

attack_flow (FM-A) / attack_flow_b (FM-B) STRUCTURALLY cannot express
"node_budget=10, edge_budget=1" (construct_perturbed_graph ties target-count to
node_budget and connects each injected node to ALL targets -> nb=10 gives 10 nodes of
degree 10, not 10 nodes of degree 1). Hence the multi-node degree-1 mass channel is
realized by OM2C, the existing purpose-built module. FM-A is kept as single-axis
baselines only.

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python eval_paper_grade_floor.py [--seeds 42,7,1027] [--cap 10] [--quick]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, argparse, importlib.util
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.datasets import TUDataset
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs  # noqa: E402
DEVICE = "cpu"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sgdi_attack = _load_module("branch_sgdi_attack", HERE / "attack.py")
import attack_flow          # noqa: E402
import attack_flow_b        # noqa: E402
import attack_om2c as O     # noqa: E402
from defenses import (DefendedModel, norm_filter_defense, mahalanobis_defense,  # noqa: E402
                      degree_filter_defense, node_degrees)

DEFENSES = ["none", "normfilter", "mahalanobis", "degfilter"]
EFFECTIVE = ["normfilter", "mahalanobis", "degfilter"]
DEGFILTER_K = 3.0
NORMFILTER_K = 3.0   # REFERENCE norm threshold = train-fold mean + NORMFILTER_K*std

ATTACKS = {
    "SGDI-full":   ("sgdi",  {"feat_mode": "sgdi", "node_budget": 1, "feat_scale": "auto_x2"}),
    "FM-A-full":   ("flow",  {"edge_strategy": "full", "node_budget": 1, "feat_scale": "auto_x2"}),
    "FM-A-eb1":    ("flow",  {"edge_strategy": "topk", "node_budget": 1, "feat_scale": 1.0}),
    # C1 CONTROL: plain on-manifold-norm flow at nb=10 / topk edges / fs=1.0, WITHOUT
    # OM2C's 2-channel tangent-subspace projection. Tests whether "10 nodes + low
    # norm + sparse edges" alone matches OM2C-nb10 — i.e. whether OM2C's special
    # tangent-projection design is even necessary for a high defense floor.
    "FM-nb10-eb1-lownorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "OM2C-nb1":    ("om2c",  {"node_budget": 1,  "feat_scale": 1.0}),
    "OM2C-nb10":   ("om2c",  {"node_budget": 10, "feat_scale": 1.0}),   # MULTI-NODE candidate
}
CANDIDATE = "OM2C-nb10"


class QueryCounter:
    def __init__(self, model):
        self.model = model; self.n = 0; self._orig = model.forward
    def _graphs(self, inp):
        g = getattr(inp, "num_graphs", None)
        if g is not None: return int(g)
        b = getattr(inp, "batch", None)
        if b is not None and b.numel() > 0: return int(b.max().item()) + 1
        return 1
    def __enter__(self):
        def cf(*a, **k):
            if a: self.n += self._graphs(a[0])
            self.model.forward = self._orig
            try: out = self._orig(*a, **k)
            finally: self.model.forward = cf
            return out
        self.model.forward = cf; return self
    def __exit__(self, *e): self.model.forward = self._orig


def clean_test_acc(model, ds, test_idx):
    c = t = 0
    for data in Subset(ds, test_idx):
        data = data.to(DEVICE)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def build_defended(model, defense, ref_graphs):
    """Wrap `model` with `defense`, deriving all defense statistics from `ref_graphs`.

    `ref_graphs` are the clean TRAINING-fold graphs (see training_ref_graphs) — NOT the
    attacked test graphs. This breaks the circularity (C3): otherwise the mahalanobis
    ellipsoid (and the norm/degree thresholds) would be fit on the same capped test set
    the OM2C prior projects into, so the attack lands exactly inside the ball the defense
    checks. Fitting on the held-out training fold makes every defense independent of the
    graphs it is asked to filter.
    """
    if defense == "none":
        return model
    if defense == "normfilter":
        # C2: REFERENCE norm threshold = train-fold mean + k*std (engages for multi-node
        # injection). A within-graph percentile would let nb injected nodes hide in the
        # graph's own upper percentile and never fire.
        norms = torch.cat([g.x.norm(p=2, dim=1) for g in ref_graphs]).to(DEVICE).float()
        return DefendedModel(model, norm_filter_defense, "normfilter",
                             mean=float(norms.mean().item()),
                             std=float(norms.std().item()), k=NORMFILTER_K)
    if defense == "mahalanobis":
        X = torch.cat([g.x for g in ref_graphs], dim=0).to(DEVICE).float()
        mean_ = X.mean(dim=0); d = X.size(1)
        cov = torch.cov(X.t()) + 1e-3 * torch.eye(d, device=DEVICE)
        cov_inv = torch.linalg.inv(cov)
        diff = X - mean_; sqm = (diff @ cov_inv * diff).sum(dim=1)
        thr = float(torch.quantile(sqm, 0.995).item())
        return DefendedModel(model, mahalanobis_defense, "mahalanobis",
                             mean=mean_, cov_inv=cov_inv, threshold=thr)
    if defense == "degfilter":
        degs = torch.cat([node_degrees(g) for g in ref_graphs]).float()
        return DefendedModel(model, degree_filter_defense, "degfilter",
                             mean=float(degs.mean().item()), std=float(degs.std().item()),
                             k=DEGFILTER_K)
    raise ValueError(defense)


def defended_correct(victim, data, device):
    """FIX1: True iff the (possibly DEFENDED) `victim` classifies the CLEAN graph
    correctly pre-attack (defended margin > 0).

    The floor metric must gate on the DEFENDED model's pre-attack prediction, not the
    base (no-defense) model's. Graphs the defense ITSELF misclassifies at baseline
    (defended-margin < 0) must be excluded from BOTH the ASR numerator and denominator,
    otherwise a defense that drops clean accuracy (e.g. bi-smoothing 0.72 -> ~0.63)
    donates spurious "flips" that inflate the worst-case-defense floor. The pre-screen is
    a defense-side query, run OUTSIDE QueryCounter, so it is NOT charged to the attack.
    """
    d = data
    if getattr(d, "batch", None) is None:
        d = d.clone()
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=d.x.device)
    with torch.no_grad():
        pred = int(victim(d).argmax(dim=-1).item())
    return pred == int(d.y.item())


def eligible_graphs(victim, graphs, device):
    """FIX1: subset of `graphs` the DEFENDED `victim` classifies correctly pre-attack.

    For the 'none' defense victim==base model and `graphs` are already base-correct
    (collect_test_graphs), so eligible == graphs and the path is unchanged (base-correct
    == defended-correct). For an effective defense d, eligible = defended-correct graphs,
    giving ASR_d = (# eligible flipped) / (# eligible) — each defense with its OWN
    defended-correct denominator; floor = min_d ASR_d.
    """
    return [g for g in graphs if defended_correct(victim, g, device)]


def om2c_greedy_flip(model, data, prior, cfg, device):
    """Channel-(a) on-manifold feature opt + channel-(b) SPARSE greedy regime, composed
    from OM2C's OWN functions (om2c_attack_graph lines 270-307). Skips the dense
    frac_anchors path. Returns (flipped, n_edges, max_inj_degree)."""
    n = data.num_nodes; m = cfg["node_budget"]; mu = prior[0]
    O._SGDI._FEAT_SCALE = O._resolve_feat_scale(cfg.get("feat_scale", 1.0), n)
    feats0 = mu.unsqueeze(0).repeat(m, 1)
    inter_full = [(i, j) for i in range(m) for j in range(i + 1, m)] if cfg["use_inter_edges"] else []
    all_v = [(i, v) for i in range(m) for v in range(n)]
    feats, _ = O.om2c_feature_opt(model, data, feats0, all_v, inter_full, prior, cfg, device)
    O._record_in_ball(feats, prior)
    pool = O._candidate_pool(data, cfg["max_edges"], cfg["pool_mult"])
    victim_edges, inter_edges = [], []
    used_v, used_i = set(), set()
    flipped = False; ne = None
    true_label = data.y.item()
    for _ in range(cfg["max_edges"]):
        cands = [("v", (i, v)) for i in range(m) for v in pool if (i, v) not in used_v]
        if cfg["use_inter_edges"] and m >= 2:
            cands += [("e", (i, j)) for i in range(m) for j in range(i + 1, m)
                      if frozenset({i, j}) not in used_i]
        if not cands:
            break
        dl = [O._build(data, feats, victim_edges + [p], inter_edges, device) if k == "v"
              else O._build(data, feats, victim_edges, inter_edges + [p], device) for k, p in cands]
        margins = O._eval_graphs(model, dl, true_label, device)
        b = int(torch.argmin(margins).item()); k, p = cands[b]
        (victim_edges if k == "v" else inter_edges).append(p)
        (used_v if k == "v" else used_i).add(p if k == "v" else frozenset(p))
        if margins[b].item() < 0:
            flipped = True; ne = len(victim_edges) + len(inter_edges); break
    deg = [0] * m
    for (i, v) in victim_edges: deg[i] += 1
    for (i, j) in inter_edges: deg[i] += 1; deg[j] += 1
    return flipped, ne, (max(deg) if deg else 0)


def run_attack_cell(run_key, overrides, victim, graphs, om2c_prior_cache):
    # FIX1 (baseline-margin gating): the ASR denominator for THIS (attack, defense) cell
    # is the DEFENDED-correct eligible set, not the base-correct set. Graphs the defense
    # itself already misclassifies pre-attack are excluded from numerator AND denominator
    # (no free flips). For 'none' victim==base model => eligible==graphs => path unchanged.
    # Both attack entry points (flow/sgdi run_attack -> n_success/len(test_graphs); om2c
    # loop below) use len(eligible) as the denominator, so filtering the input list here
    # is sufficient and no attack code is touched.
    eligible = eligible_graphs(victim, graphs, DEVICE)
    n_total = len(graphs); n_elig = len(eligible)
    if run_key in ("sgdi", "flow", "flowb"):
        mod = {"sgdi": sgdi_attack, "flow": attack_flow, "flowb": attack_flow_b}[run_key]
        saved = {k: mod.CONFIG.get(k) for k in overrides}
        mod.CONFIG.update(overrides)
        nb = nt = 0
        try:
            with QueryCounter(victim) as qc:
                asr = mod.run_attack(victim, eligible, DEVICE) if eligible else 0.0
            if run_key in ("flow", "flowb"):
                st = getattr(mod, "FLOW_STATS", None)
                if st: nb, nt = int(st.get("in_ball", 0)), int(st.get("total", 0))
            return {"asr": asr, "q": qc.n, "in_ball": nb, "in_ball_total": nt,
                    "n_eligible": n_elig, "n_total": n_total, "extra": {}}
        finally:
            mod.CONFIG.update(saved)
    elif run_key == "om2c":
        cfg = {**O.CONFIG, **overrides}
        O._SGDI.QUERY_COUNT = 0; O._reset_stats()
        prior = om2c_prior_cache
        max_deg = 0; flips = 0; edges = []
        with QueryCounter(victim) as qc:
            for g in eligible:
                fl, ne, md = om2c_greedy_flip(victim, g.to(DEVICE), prior, cfg, DEVICE)
                flips += int(fl); max_deg = max(max_deg, md)
                if ne is not None: edges.append(ne)
        asr = flips / n_elig if n_elig else 0.0
        return {"asr": asr, "q": qc.n,
                "in_ball": O.OM2C_STATS["in_ball"], "in_ball_total": O.OM2C_STATS["total"],
                "n_eligible": n_elig, "n_total": n_total,
                "extra": {"max_inj_degree": max_deg,
                          "median_edges_at_flip": (sorted(edges)[len(edges)//2] if edges else None)}}
    raise ValueError(run_key)


def load_meta_for_seed(seed):
    p = MODELS_DIR / f"meta_pg_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing victim meta for seed {seed}: {p}")
    return json.loads(p.read_text())


def training_ref_graphs(ds, meta, fold_info):
    """Clean TRAINING-fold graphs for this fold (C3 fix).

    Reconstructs the FROZEN StratifiedKFold split via prepare.get_folds with the victim
    seed/n_folds (the meta stores only test_idx), then returns the train_idx graphs.
    These are independent of the attacked test graphs, so defense statistics fit on them
    are not circular with the OM2C prior / attack. The reconstructed test split is
    verified against meta['test_idx'] to guarantee we reproduced the exact fold.
    """
    saved_seed, saved_nf = prepare.SEED, prepare.N_FOLDS
    prepare.SEED = meta["seed"]; prepare.N_FOLDS = meta["n_folds"]
    try:
        train_idx = None
        for fidx, tr, va, te in prepare.get_folds(ds):
            if fidx == fold_info["fold_idx"]:
                if sorted(int(i) for i in te) != sorted(int(i) for i in fold_info["test_idx"]):
                    raise RuntimeError(f"fold {fidx}: reconstructed test_idx != meta test_idx")
                train_idx = tr
                break
    finally:
        prepare.SEED = saved_seed; prepare.N_FOLDS = saved_nf
    if train_idx is None:
        raise RuntimeError(f"fold {fold_info['fold_idx']} not found in reconstructed folds")
    graphs = []
    for data in Subset(ds, train_idx):
        data = data.to(DEVICE)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE)
        graphs.append(data)
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = None
    if args.quick:
        seeds = seeds[:1]; cap = 4; fold_limit = 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    in_ball = {a: [0, 0] for a in ATTACKS}
    extra_acc = {a: {"max_inj_degree": 0, "edges": []} for a in ATTACKS}
    floor_per_unit = {a: [] for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; units = []
    meta0 = load_meta_for_seed(seeds[0])

    print(f"=== PAPER-GRADE FLOOR STUDY | {meta0['dataset']} {meta0.get('model_type','GCN')} | "
          f"seeds={seeds} | cap={cap}/fold | defenses={DEFENSES} | attacks={list(ATTACKS)} | "
          f"candidate={CANDIDATE} | device={DEVICE} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = load_meta_for_seed(seed)
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            model.eval()
            vacc = clean_test_acc(model, ds, fold_info["test_idx"]); victim_accs.append(vacc)
            graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
            # C3: defense statistics fit on the held-out TRAINING fold, NOT the attacked
            # test graphs (om2c_prior below still uses test graphs — that is the attack's
            # own prior and is intentionally left untouched).
            ref_graphs = training_ref_graphs(ds, meta, fold_info)
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = build_defended(model, dn, ref_graphs)
                ca = clean_test_acc(v, ds, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                asrs = {}
                for dn in DEFENSES:
                    victim = build_defended(model, dn, ref_graphs)
                    r = run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                    cell_asr[akey][dn].append(r["asr"]); cell_q[akey][dn] += r["q"]
                    if dn == "none":
                        in_ball[akey][0] += r["in_ball"]; in_ball[akey][1] += r["in_ball_total"]
                        ex = r.get("extra", {})
                        if ex.get("max_inj_degree") is not None:
                            extra_acc[akey]["max_inj_degree"] = max(extra_acc[akey]["max_inj_degree"], ex["max_inj_degree"])
                        if ex.get("median_edges_at_flip") is not None:
                            extra_acc[akey]["edges"].append(ex["median_edges_at_flip"])
                    asrs[dn] = round(r["asr"], 4)
                    unit["cells"][f"{akey}|{dn}"] = {"asr": round(r["asr"], 4), "q": r["q"]}
                floor = min(asrs[d] for d in EFFECTIVE)
                floor_per_unit[akey].append(floor)
                unit["cells"][f"{akey}|FLOOR"] = round(floor, 4)
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  " +
                  "  ".join(f"{a}:fl={floor_per_unit[a][-1]*100:.0f}" for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

    floor_summary = {a: agg(floor_per_unit[a]) for a in ATTACKS}
    asr_summary = {a: {d: agg(cell_asr[a][d]) for d in DEFENSES} for a in ATTACKS}

    others = [a for a in ATTACKS if a != CANDIDATE]
    next_best = max(others, key=lambda a: floor_summary[a]["mean_pct"])
    paired = {"candidate": CANDIDATE, "next_best": next_best}
    try:
        from scipy.stats import wilcoxon, ttest_rel
        cand = floor_per_unit[CANDIDATE]; nb = floor_per_unit[next_best]
        diffs = [c - n for c, n in zip(cand, nb)]
        if any(d != 0 for d in diffs):
            try: w_stat, w_p = wilcoxon(cand, nb)
            except Exception as e: w_stat, w_p = None, f"wilcoxon_error:{e}"
        else:
            w_stat, w_p = None, "all_diffs_zero"
        t_stat, t_p = ttest_rel(cand, nb)
        paired.update({
            "candidate_floor_mean_pct": floor_summary[CANDIDATE]["mean_pct"],
            "next_best_floor_mean_pct": floor_summary[next_best]["mean_pct"],
            "mean_diff_pct": round(mean(diffs) * 100, 3),
            "wilcoxon_stat": (None if w_stat is None else float(w_stat)),
            "wilcoxon_p": (w_p if isinstance(w_p, str) else float(w_p)),
            "paired_t_stat": float(t_stat), "paired_t_p": float(t_p), "n_pairs": len(diffs),
        })
    except ImportError:
        paired["error"] = "scipy unavailable"

    out = {
        "stamp": stamp,
        "experiment": "PAPER-GRADE worst-case-defense FLOOR study (N28 multi-node orthogonal)",
        "hypothesis_H1": "MULTI-NODE candidate OM2C-nb10 (10 low-degree nodes + fs=1.0) floor >> next-best floor (p<0.01)",
        "candidate": CANDIDATE,
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE, "degfilter_k": DEGFILTER_K,
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "om2c_metric": "SPARSE greedy regime (degree-1 mass channel); dense frac_anchors NOT used",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "floor_summary": floor_summary, "asr_summary": asr_summary,
        "total_queries_by_attack": {a: sum(cell_q[a].values()) for a in ATTACKS},
        "queries_by_attack_defense": {a: {d: cell_q[a][d] for d in DEFENSES} for a in ATTACKS},
        "in_ball_fraction_by_attack": {
            a: {"in_ball": in_ball[a][0], "total": in_ball[a][1],
                "fraction": round(in_ball[a][0] / in_ball[a][1], 4) if in_ball[a][1] else None}
            for a in ATTACKS},
        "om2c_degree_info": {a: {"max_inj_degree_observed": extra_acc[a]["max_inj_degree"],
                                 "median_edges_at_flip": (sorted(extra_acc[a]["edges"])[len(extra_acc[a]["edges"])//2]
                                                          if extra_acc[a]["edges"] else None)}
                             for a in ATTACKS if ATTACKS[a][0] == "om2c"},
        "paired_test_H1": paired,
        "floor_per_unit": {a: [round(x, 4) for x in floor_per_unit[a]] for a in ATTACKS},
        "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"paper_grade_floor_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 84, flush=True)
    print(f"victim_acc: mean={out['victim_acc_mean']} std={out['victim_acc_std']} min={out['victim_acc_min']} (floor 0.45)", flush=True)
    print("clean acc thru defenses:", {d: out['clean_acc_by_defense'][d]['mean'] for d in DEFENSES}, flush=True)
    print(f"\n{'attack':<12} " + " ".join(f"{d:>13}" for d in DEFENSES) + f" {'FLOOR':>13}", flush=True)
    for a in ATTACKS:
        cells = " ".join(f"{asr_summary[a][d]['mean_pct']:6.1f}±{asr_summary[a][d]['std_pct']:<5.1f}" for d in DEFENSES)
        fl = floor_summary[a]
        print(f"{a:<12} {cells}  {fl['mean_pct']:6.1f}±{fl['std_pct']:<5.1f}", flush=True)
    print(f"\nH1 paired ({CANDIDATE} floor vs next-best={next_best}): "
          f"cand={paired.get('candidate_floor_mean_pct')}% nextbest={paired.get('next_best_floor_mean_pct')}% "
          f"diff={paired.get('mean_diff_pct')}%  wilcoxon_p={paired.get('wilcoxon_p')} "
          f"t_p={paired.get('paired_t_p')} n={paired.get('n_pairs')}", flush=True)
    print(f"om2c degree info: {out['om2c_degree_info']}", flush=True)
    print(f"queries: " + " ".join(f"{a}={sum(cell_q[a].values())}" for a in ATTACKS), flush=True)
    print(f"\nSaved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
