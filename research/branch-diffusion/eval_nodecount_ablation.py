#!/usr/bin/env python3
"""eval_nodecount_ablation.py — "WHY ONE NODE IS NOT ENOUGH" node-count ablation.

Single IV = node_budget nb in {1,2,3,4,5,8,10,15}. Everything else fixed:
  edge_budget=1 (degree-1 sparse-greedy mass channel, stealthy structure),
  feat_scale=1.0 (on-manifold, stealthy features), the OM2C on-manifold attack,
  PROTEINS / GCN victim, SAME victim per (seed,fold) cell.
Rigor: 10-fold x 3 seeds (42,7,1027) = 30 units per nb.

THREE DVs recorded per nb:
  (1) UNDEFENDED ASR (defense=none) — does the attack flip with NO defense?
      separates "too weak to attack" (pooling dilution) from "caught by a defense".
  (2) WORST-CASE-DEFENSE FLOOR = min ASR over {normfilter, mahalanobis, degfilter}.
  (3) POOLED-EMBEDDING SHIFT = L2 distance between the global-mean-pooled readout
      (input to GCN.classifier) of the clean vs the final perturbed graph, measured
      on the UNDEFENDED model, averaged over attacked graphs. Per-graph (shift, flipped)
      pairs are also recorded for a shift-vs-flip scatter.

H1: under the stealth constraint (eb=1, fs=1.0) BOTH undefended ASR and the floor rise
    with nb and cross a threshold near nb~4-5; nb=1 is near 0 because one stealthy node
    contributes ~1/(N+1) of the mean-pooled readout (PROTEINS avg N~39) — too small to
    cross the decision margin. H0: nb doesn't matter.

DRIVER ONLY — no attack/model/defense code modified. The OM2C two-channel attack
(attack_om2c: om2c_feature_opt + sparse greedy) and the defense battery (defenses.py)
are reused verbatim by import. The sparse-greedy helper is re-composed here (faithful to
eval_paper_grade_floor.om2c_greedy_flip) ONLY to additionally return the built perturbed
graph for the pooled-shift DV. All linalg (data_prior inv/cholesky; mahalanobis cov_inv)
is computed on CPU and moved to GPU — torch.linalg on CUDA segfaults in this env once a
CUDA context exists (see attack_om2c.py:96-99); the linalg-free attack/defense forward
runs on GPU.

Usage: micromamba run -n graph_adversarial python3 eval_nodecount_ablation.py \
         [--seeds 42,7,1027] [--cap 10] [--nbs 1,2,3,4,5,8,10,15] [--device cuda] [--quick]
"""
import os, sys, json, time, argparse, traceback
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
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs  # noqa
import attack_om2c as O  # noqa
import numpy as np  # noqa
from defenses import (DefendedModel, norm_filter_defense, mahalanobis_defense,  # noqa
                      degree_filter_defense, node_degrees,
                      jaccard_defense, svd_defense, featclip_defense,
                      gnnsafe_defense, gddm_a_defense, gddm_a_ckpt_path,
                      _gnnsafe_node_logits, _energy_propagation)

DEFENSES = ["none", "normfilter", "mahalanobis", "degfilter"]
EFFECTIVE = ["normfilter", "mahalanobis", "degfilter"]
DEGFILTER_K = 3.0

# --- Extended defense set (branch-diffusion 9-line sweep, 2026-07-01) ---------
# The 5 defenses below are wired ADDITIVELY into build_defended and are ONLY used
# when a driver is invoked with --defenses (module default DEFENSES stays at 4, so
# the other importers of this module are unaffected). See build_defended().
DEFENSES_EXT = ["none", "normfilter", "mahalanobis", "degfilter",
                "gnnsafe", "jaccard", "svd", "featclip", "gddm_adapted"]
# gnnsafe calibration hyperparams — identical to eval_pg_floor_battery5_gpu.py
# (== calibrate_gnnsafe.py / official GraphOOD-GNNSafe repo).
GNNSAFE_T, GNNSAFE_K, GNNSAFE_ALPHA, GNNSAFE_PCT_CUT_LOW = 1.0, 2, 0.5, 5.0
FEATCLIP_K = 2.0            # per-dim clip band = train mean +/- k*std (featclip default)
GDDM_DATASET = "PROTEINS"  # dataset whose gddm_a denoiser ckpt gddm_adapted loads


def calibrate_gnnsafe_threshold(model, ref_graphs, device):
    """5th-percentile of propagated clean-node neg_energy over ref graphs.

    Verbatim reuse of eval_pg_floor_battery5_gpu.calibrate_gnnsafe_threshold
    (T=1, K=2, alpha=0.5; node kept iff neg_energy >= thr). ref_graphs are the
    driver's per-fold reference graphs (already on `device`)."""
    chunks = []
    for g in ref_graphs:
        # prior helpers (om2c/flow_prior_on_cpu) move `graphs` to CPU in place, so a
        # ref graph may be on CPU while the model is on CUDA; align to the model device
        # for this forward (the other 3 defenses are device-agnostic / .cpu() explicitly).
        g = g.to(device)
        with torch.no_grad():
            logits = _gnnsafe_node_logits(model, g)
            e = GNNSAFE_T * torch.logsumexp(logits / GNNSAFE_T, dim=-1)
            e = _energy_propagation(e, g.edge_index, g.num_nodes,
                                    K=GNNSAFE_K, alpha=GNNSAFE_ALPHA)
        chunks.append(e.detach().cpu().numpy())
    energies = np.concatenate(chunks) if chunks else np.array([])
    if energies.size == 0:
        raise RuntimeError("gnnsafe calibration: empty ref set")
    return float(np.percentile(energies, GNNSAFE_PCT_CUT_LOW))


# ----------------------------------------------------------------------------
# Query counter (verbatim from eval_paper_grade_floor.QueryCounter)
# ----------------------------------------------------------------------------
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


def clean_test_acc(model, ds, test_idx, device):
    c = t = 0
    for data in Subset(ds, test_idx):
        data = data.to(device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(data).argmax(dim=-1).item()
        c += int(pred == data.y.item()); t += 1
    return c / max(t, 1)


def build_defended(model, defense, ref_graphs, device):
    """Mirror of eval_paper_grade_floor.build_defended, but mahalanobis cov_inv is
    computed on CPU (linalg-on-CUDA segfaults in this env) then moved to `device`."""
    if defense == "none":
        return model
    if defense == "normfilter":
        return DefendedModel(model, norm_filter_defense, "normfilter", percentile=95)
    if defense == "mahalanobis":
        X = torch.cat([g.x for g in ref_graphs], dim=0).float().cpu()   # linalg on CPU
        mean_ = X.mean(dim=0); d = X.size(1)
        cov = torch.cov(X.t()) + 1e-3 * torch.eye(d)
        cov_inv = torch.linalg.inv(cov)
        diff = X - mean_; sqm = (diff @ cov_inv * diff).sum(dim=1)
        thr = float(torch.quantile(sqm, 0.995).item())
        return DefendedModel(model, mahalanobis_defense, "mahalanobis",
                             mean=mean_.to(device), cov_inv=cov_inv.to(device), threshold=thr)
    if defense == "degfilter":
        degs = torch.cat([node_degrees(g) for g in ref_graphs]).float()
        return DefendedModel(model, degree_filter_defense, "degfilter",
                             mean=float(degs.mean().item()), std=float(degs.std().item()),
                             k=DEGFILTER_K)
    # --- Extended 5 (2026-07-01), additive; existing 4 branches untouched ---
    if defense == "jaccard":            # STRUCTURE, no calibration (cosine-sim edge prune)
        return DefendedModel(model, jaccard_defense, "jaccard",
                             threshold=0.01, similarity="cosine")
    if defense == "svd":                # STRUCTURE, no calibration (low-rank adj recon)
        return DefendedModel(model, svd_defense, "svd", rank=10)
    if defense == "featclip":           # FEATURE; per-dim train mean/std (like normfilter)
        X = torch.cat([g.x for g in ref_graphs], dim=0).float().cpu()
        mean_ = X.mean(dim=0); std_ = X.std(dim=0).clamp_min(1e-6)
        return DefendedModel(model, featclip_defense, "featclip",
                             mean=mean_.to(device), std=std_.to(device), k=FEATCLIP_K)
    if defense == "gnnsafe":            # ENERGY-OOD; ref-graph calibrated threshold
        thr = calibrate_gnnsafe_threshold(model, ref_graphs, device)
        return DefendedModel(model, gnnsafe_defense, "gnnsafe",
                             victim_model=model, threshold=thr,
                             T=GNNSAFE_T, K=GNNSAFE_K, alpha=GNNSAFE_ALPHA)
    if defense == "gddm_adapted":       # DIFFUSION purification (per-graph denoiser; EXPENSIVE)
        return DefendedModel(model, gddm_a_defense, "gddm_adapted",
                             ckpt_path=gddm_a_ckpt_path(GDDM_DATASET), t_star=30)
    raise ValueError(defense)


def om2c_prior_on_cpu(graphs, device):
    """Compute the OM2C prior on CPU (data_prior uses linalg.inv + cholesky which
    segfault on CUDA in this env) then move tensors to `device`."""
    cpu_graphs = [g.cpu() for g in graphs]
    pr = O.om2c_prior(cpu_graphs, "cpu", O.CONFIG["tangent_dim"],
                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
    mu, cov_inv, L, thr, V_k, r2s = pr
    return (mu.to(device), cov_inv.to(device), L.to(device), thr, V_k.to(device), r2s)


# ----------------------------------------------------------------------------
# Pooled-readout hook: input to GCN.classifier == global_mean_pool output.
# ----------------------------------------------------------------------------
def pooled_readout(model, data, device):
    """Return the mean-pooled graph embedding (1, hidden) = the input to model.classifier."""
    cap = {}
    def hook(_m, args):
        cap["x"] = args[0].detach()
    h = model.classifier.register_forward_pre_hook(hook)
    try:
        d = data.to(device)
        if not hasattr(d, "batch") or d.batch is None:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            model(d)
    finally:
        h.remove()
    return cap["x"]


# ----------------------------------------------------------------------------
# Instrumented OM2C sparse-greedy flip (faithful to eval_paper_grade_floor.
# om2c_greedy_flip) that ALSO returns the final perturbed graph for the pooled-shift DV.
# Composed entirely from OM2C's own exported functions — no attack code modified.
# ----------------------------------------------------------------------------
def om2c_greedy_flip_instr(model, data, prior, cfg, device):
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
    perturbed = O._build(data, feats, victim_edges, inter_edges, device)  # final injected graph
    return flipped, ne, (max(deg) if deg else 0), perturbed


def run_om2c_cell(victim, graphs, prior, cfg, device, measure_shift=False, bare_model=None):
    """Run the sparse-greedy OM2C attack over `graphs` through `victim` (possibly defended).
    Returns asr, queries, max_inj_degree, and (if measure_shift) pooled-shift stats from
    the UNDEFENDED bare_model."""
    O._SGDI.QUERY_COUNT = 0; O._reset_stats()
    flips = 0; max_deg = 0; shifts = []; per_graph = []
    with QueryCounter(victim) as qc:
        for g in graphs:
            gd = g.to(device)
            fl, ne, md, perturbed = om2c_greedy_flip_instr(victim, gd, prior, cfg, device)
            flips += int(fl); max_deg = max(max_deg, md)
            if measure_shift and bare_model is not None:
                clean_emb = pooled_readout(bare_model, gd, device)
                pert_emb = pooled_readout(bare_model, perturbed, device)
                shift = float((pert_emb - clean_emb).norm().item())
                shifts.append(shift)
                per_graph.append({"shift": round(shift, 5), "flipped": bool(fl),
                                  "n_nodes": int(gd.num_nodes)})
    asr = flips / max(len(graphs), 1)
    return {"asr": asr, "q": qc.n, "max_inj_degree": max_deg,
            "shifts": shifts, "per_graph": per_graph,
            "in_ball": O.OM2C_STATS["in_ball"], "in_ball_total": O.OM2C_STATS["total"]}


def load_meta_for_seed(seed):
    p = MODELS_DIR / f"meta_pg_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing victim meta for seed {seed}: {p}")
    return json.loads(p.read_text())


def agg(lst):
    return {"mean": round(mean(lst), 4) if lst else None,
            "std": round(pstdev(lst), 4) if len(lst) > 1 else 0.0, "n": len(lst)}


def agg_pct(lst):
    return {"mean_pct": round(mean(lst) * 100, 2) if lst else None,
            "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--nbs", type=str, default="1,2,3,4,5,8,10,15")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # ---- device + GPU probe ----
    want = args.device
    probe = {"requested": want}
    if want == "cuda":
        try:
            t = torch.randn(50, 50).cuda(); _ = float((t @ t).sum().item())
            probe.update({"cuda_available": True, "probe_ok": True,
                          "device_name": torch.cuda.get_device_name(0)})
            device = "cuda"
        except Exception as e:
            probe.update({"cuda_available": torch.cuda.is_available(), "probe_ok": False,
                          "error": repr(e)})
            print(f"[GPU PROBE FAILED] {e!r}\n{traceback.format_exc()}", flush=True)
            device = "cpu"
    else:
        device = "cpu"; probe.update({"probe_ok": None})
    prepare.DEVICE = device
    print(f"[device] requested={want} using={device} probe={probe}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    nbs = [int(x) for x in args.nbs.split(",")]
    cap = args.cap; fold_limit = None
    if args.quick:
        seeds = seeds[:1]; cap = 3; fold_limit = 1; nbs = [int(x) for x in args.nbs.split(",")]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta0 = load_meta_for_seed(seeds[0])

    # per-nb accumulators
    und_asr = {nb: [] for nb in nbs}                 # DV1 per-unit undefended ASR
    floor = {nb: [] for nb in nbs}                   # DV2 per-unit worst-case floor
    asr_by_def = {nb: {d: [] for d in DEFENSES} for nb in nbs}
    shift_mean = {nb: [] for nb in nbs}              # DV3 per-unit mean pooled shift
    all_pairs = {nb: [] for nb in nbs}               # DV3 per-graph (shift, flipped)
    max_deg = {nb: 0 for nb in nbs}
    inball = {nb: [0, 0] for nb in nbs}
    q_by_nb = {nb: 0 for nb in nbs}
    victim_accs = []; clean_acc_def = {d: [] for d in DEFENSES}
    units = []

    print(f"=== NODE-COUNT ABLATION | {meta0['dataset']} {meta0.get('model_type','GCN')} | "
          f"seeds={seeds} cap={cap}/fold nbs={nbs} | defenses={DEFENSES} | "
          f"device={device} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = load_meta_for_seed(seed)
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(device)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=device))
            model.eval()
            vacc = clean_test_acc(model, ds, fold_info["test_idx"], device); victim_accs.append(vacc)
            graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
            prior = om2c_prior_on_cpu(graphs, device)
            # defended clean acc (once per fold)
            defended = {}
            cacc = {}
            for dn in DEFENSES:
                v = build_defended(model, dn, graphs, device)
                defended[dn] = v
                ca = clean_test_acc(v, ds, fold_info["test_idx"], device)
                clean_acc_def[dn].append(ca); cacc[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc, "nb": {}}
            for nb in nbs:
                cfg = {**O.CONFIG, "node_budget": nb, "feat_scale": 1.0}
                asrs = {}
                for dn in DEFENSES:
                    r = run_om2c_cell(defended[dn], graphs, prior, cfg, device,
                                      measure_shift=(dn == "none"), bare_model=model)
                    asrs[dn] = r["asr"]; asr_by_def[nb][dn].append(r["asr"])
                    q_by_nb[nb] += r["q"]
                    if dn == "none":
                        und_asr[nb].append(r["asr"])
                        shift_mean[nb].append(mean(r["shifts"]) if r["shifts"] else 0.0)
                        all_pairs[nb].extend(r["per_graph"])
                        max_deg[nb] = max(max_deg[nb], r["max_inj_degree"])
                        inball[nb][0] += r["in_ball"]; inball[nb][1] += r["in_ball_total"]
                fl = min(asrs[d] for d in EFFECTIVE)
                floor[nb].append(fl)
                unit["nb"][str(nb)] = {"undef_asr": round(asrs["none"], 4),
                                       "floor": round(fl, 4),
                                       "asr_by_def": {d: round(asrs[d], 4) for d in DEFENSES},
                                       "shift_mean": round(shift_mean[nb][-1], 5)}
            units.append(unit)
            tag = "  ".join(f"nb{nb}:U={und_asr[nb][-1]*100:.0f}/F={floor[nb][-1]*100:.0f}/S={shift_mean[nb][-1]:.2f}"
                            for nb in nbs)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  {tag}", flush=True)

    # ---- aggregate ----
    und_summary = {nb: agg_pct(und_asr[nb]) for nb in nbs}
    floor_summary = {nb: agg_pct(floor[nb]) for nb in nbs}
    shift_summary = {nb: agg(shift_mean[nb]) for nb in nbs}
    asr_def_summary = {nb: {d: agg_pct(asr_by_def[nb][d]) for d in DEFENSES} for nb in nbs}

    # knee = nb with the largest first-difference (max marginal gain) on undefended ASR
    def knee(summary):
        ms = [summary[nb]["mean_pct"] for nb in nbs]
        diffs = [(nbs[i + 1], ms[i + 1] - ms[i]) for i in range(len(nbs) - 1)]
        return max(diffs, key=lambda kv: kv[1]) if diffs else (None, None)
    knee_und_nb, knee_und_gain = knee(und_summary)
    knee_floor_nb, knee_floor_gain = knee(floor_summary)

    out = {
        "stamp": stamp,
        "experiment": "NODE-COUNT ABLATION (why one node is not enough) — IV=node_budget",
        "hypothesis_H1": "Under stealth (eb=1, fs=1.0) undefended ASR AND worst-case-defense "
                         "floor rise with nb and cross a threshold near nb~4-5; nb=1 near 0 due "
                         "to ~1/(N+1) mean-pool dilution. H0: nb does not matter.",
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device_requested": want, "device_used": device, "gpu_probe": probe,
        "seeds": seeds, "cap_per_fold": cap, "node_budgets": nbs, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE, "degfilter_k": DEGFILTER_K,
        "controls": {"edge_budget": "1 (degree-1 sparse-greedy mass channel)",
                     "feat_scale": 1.0, "attack": "OM2C on-manifold two-channel",
                     "max_edges_greedy": O.CONFIG["max_edges"],
                     "zo_epochs": O.CONFIG["zo_epochs"], "zo_restarts": O.CONFIG["zo_restarts"],
                     "tangent_dim": O.CONFIG["tangent_dim"], "sigma_mult": O.CONFIG["sigma_mult"]},
        "pooled_shift_def": "L2 distance between mean-pooled readout (input to GCN.classifier) "
                            "of clean vs final perturbed graph, on the UNDEFENDED model",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "DV1_undefended_asr": {str(nb): und_summary[nb] for nb in nbs},
        "DV2_worstcase_floor": {str(nb): floor_summary[nb] for nb in nbs},
        "DV3_pooled_shift": {str(nb): shift_summary[nb] for nb in nbs},
        "asr_by_defense": {str(nb): asr_def_summary[nb] for nb in nbs},
        "knee": {"undefended_asr": {"nb": knee_und_nb, "marginal_gain_pct": round(knee_und_gain, 2) if knee_und_gain is not None else None},
                 "floor": {"nb": knee_floor_nb, "marginal_gain_pct": round(knee_floor_gain, 2) if knee_floor_gain is not None else None}},
        "max_inj_degree_observed": {str(nb): max_deg[nb] for nb in nbs},
        "in_ball_fraction": {str(nb): (round(inball[nb][0] / inball[nb][1], 4) if inball[nb][1] else None) for nb in nbs},
        "queries_by_nb": {str(nb): q_by_nb[nb] for nb in nbs},
        "per_unit": {"undef_asr": {str(nb): [round(x, 4) for x in und_asr[nb]] for nb in nbs},
                     "floor": {str(nb): [round(x, 4) for x in floor[nb]] for nb in nbs},
                     "shift_mean": {str(nb): [round(x, 5) for x in shift_mean[nb]] for nb in nbs}},
        "per_graph_shift_flip": {str(nb): all_pairs[nb] for nb in nbs},
        "wall_s": round(time.time() - t0, 1),
        "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"nodecount_ablation_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 92, flush=True)
    print(f"device_used={device}  victim_acc: mean={out['victim_acc_mean']} "
          f"std={out['victim_acc_std']} min={out['victim_acc_min']}", flush=True)
    print(f"\n{'nb':>4} {'undef_ASR%':>14} {'floor%':>14} {'pooled_shift':>16} {'max_deg':>8}", flush=True)
    for nb in nbs:
        u = und_summary[nb]; f = floor_summary[nb]; s = shift_summary[nb]
        print(f"{nb:>4} {u['mean_pct']:7.1f}±{u['std_pct']:<5.1f} {f['mean_pct']:7.1f}±{f['std_pct']:<5.1f} "
              f"{s['mean']:8.3f}±{s['std']:<6.3f} {max_deg[nb]:>8}", flush=True)
    print(f"\nknee undefended-ASR: nb={knee_und_nb} (max marginal gain "
          f"{out['knee']['undefended_asr']['marginal_gain_pct']}pp)", flush=True)
    print(f"knee floor:          nb={knee_floor_nb} (max marginal gain "
          f"{out['knee']['floor']['marginal_gain_pct']}pp)", flush=True)
    print(f"queries by nb: {out['queries_by_nb']}", flush=True)
    print(f"\nSaved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
