#!/usr/bin/env python3
"""eval_nodecount_2attack.py — TWO-ATTACK node-count ablation: OM2C vs the
FM-eb1-lownorm control, swept over the SAME node budgets, to show OM2C's
degree-1 sparse-greedy design matters at EVERY node count (not just "more nodes").

IV  = node_budget nb in {1,2,3,4,5,8,10,15}, for BOTH attacks:
  * OM2C            : on-manifold two-channel attack — degree-1 sparse-greedy mass
                      channel (tangent + greedy edge selection) + on-manifold feats.
                      ("om2c", feat_scale=1.0)
  * FM-eb1-lownorm  : plain on-manifold flow-matching injection WITHOUT OM2C's
                      tangent+sparse-greedy design. Edge strategy "topk" wires EACH
                      injected node to select_targets_topk(data, node_budget) targets,
                      so the realized per-node degree == node_budget -> degree GROWS
                      with nb. Low-norm (feat_scale=1.0) to match OM2C's on-manifold
                      feature axis. ("flow", edge_strategy="topk", feat_scale=1.0)

Controls (fixed): edge_budget=1 nominal, feat_scale=1.0, PROTEINS/GCN 200-epoch
victims, seeds 42/7/1027, cap>=10 graphs/fold, 10-fold => up to 30 units per (attack,nb).

THREE DVs per (attack, nb) — IDENTICAL harness/definitions to eval_nodecount_ablation.py:
  (U) UNDEFENDED ASR (defense=none).
  (F) WORST-CASE-DEFENSE FLOOR = min ASR over {normfilter, mahalanobis, degfilter}.
  (S) POOLED-EMBEDDING SHIFT = L2 distance between the global-mean-pooled readout
      (input to GCN.classifier) of clean vs final perturbed graph, on the UNDEFENDED model.

H1: OM2C's floor RISES with nb (knee ~4-5, plateau ~55%); the FM-eb1-lownorm control's
    floor STAYS LOW at every nb — its flow wiring connects each injected node to
    ~node_budget targets, so degree grows with nb and degfilter prunes it regardless of
    node count. The gap is the degree-1 SPARSE DESIGN, present at every node count, not
    the node budget. H0: both floors track each other.

DRIVER ONLY — no attack/model/defense code modified.
  * The OM2C side reuses eval_nodecount_ablation.py VERBATIM by import (build_defended,
    om2c_prior_on_cpu, run_om2c_cell, clean_test_acc, pooled_readout, QueryCounter, ...).
  * The FM side composes attack_flow's OWN exported functions: data_prior (computed on
    CPU then moved to device — torch.linalg on CUDA segfaults in this env), select_targets_topk,
    _attack_single_flow, construct_perturbed_graph. The perturbed graph for the pooled-shift
    DV is captured by wrapping attack_flow.construct_perturbed_graph (the function the flow
    ODE itself calls), so the captured graph is exactly the one that determines the outcome.

Usage: micromamba run -n graph_adversarial python3 eval_nodecount_2attack.py \
         [--seeds 42,7,1027] [--cap 10] [--nbs 1,2,3,4,5,8,10,15] [--device cuda] [--quick]
"""
import sys, json, time, math, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch_geometric.datasets import TUDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prepare
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs  # noqa

# Reuse the SINGLE-ATTACK ablation's frozen harness VERBATIM (driver glue, not attack code).
import eval_nodecount_ablation as NB
from eval_nodecount_ablation import (DEFENSES, EFFECTIVE, DEGFILTER_K,  # noqa
                                     build_defended, om2c_prior_on_cpu, run_om2c_cell,
                                     clean_test_acc, pooled_readout, QueryCounter,
                                     load_meta_for_seed, agg, agg_pct)
import attack_om2c as O      # noqa
import attack_flow as F      # noqa


# ----------------------------------------------------------------------------
# FM-eb1-lownorm control: prior on CPU (linalg.inv/cholesky segfault on CUDA), then
# move to device. Mirrors NB.om2c_prior_on_cpu for the flow's data_prior.
# ----------------------------------------------------------------------------
def flow_prior_on_cpu(graphs, device):
    cpu_graphs = [g.cpu() for g in graphs]
    mu, cov_inv, L, thr = F.data_prior(cpu_graphs, "cpu", eps=F.CONFIG["prior_eps"])
    return (mu.to(device), cov_inv.to(device), L.to(device), thr)


def _resolve_fs(fs, num_nodes):
    if fs == "auto":      return math.sqrt(num_nodes)
    if fs == "auto_x2":   return 2.0 * math.sqrt(num_nodes)
    if fs == "auto_x3":   return 3.0 * math.sqrt(num_nodes)
    if fs == "auto_linear": return float(num_nodes)
    return float(fs)


# ----------------------------------------------------------------------------
# Instrumented FM flip: calls attack_flow._attack_single_flow over restarts (mirroring
# attack_flow.run_attack's restart loop) while capturing the perturbed graph that
# determines the outcome by wrapping construct_perturbed_graph. Returns
# (flipped, max_inj_degree, perturbed_graph). No attack code modified.
# ----------------------------------------------------------------------------
def flow_flip_instr(model, data, prior, cfg, device):
    n_feat = data.x.size(1)
    m = cfg["node_budget"]
    fs = _resolve_fs(cfg["feat_scale"], data.num_nodes)
    # edge target selection (same as attack_flow.run_attack)
    if cfg["edge_strategy"] == "full":
        targets = torch.arange(data.num_nodes, device=device)
    elif cfg["edge_strategy"] == "spectral":
        targets = F.select_targets_spectral(data, m, cfg["spectral_top_k_eig"], device)
    else:
        targets = F.select_targets_topk(data, m)

    cap = {}
    orig_cpg = F.construct_perturbed_graph
    def wrapped(d, x, tgts, fscale):
        g = orig_cpg(d, x, tgts, fscale)
        cap["g"] = g
        return g
    F.construct_perturbed_graph = wrapped  # the flow ODE looks this global up at call time
    flipped = False; last_g = None
    try:
        for r in range(cfg["n_restarts"]):
            seed = r * 1000 + data.num_nodes
            cap.pop("g", None)
            res = F._attack_single_flow(model, data, targets, fs, cfg, n_feat, prior, device, seed=seed)
            if cap.get("g") is not None:
                last_g = cap["g"]
            if res:
                flipped = True
                break
    finally:
        F.construct_perturbed_graph = orig_cpg

    # max injected-node degree from the captured (outcome-determining) perturbed graph
    md = 0
    if last_g is not None:
        ei = last_g.edge_index; on = data.num_nodes
        for idx in range(on, on + m):
            md = max(md, int((ei[0] == idx).sum().item()))
    return flipped, md, last_g


def run_flow_cell(victim, graphs, prior, cfg, device, measure_shift=False, bare_model=None):
    """Run the FM-eb1-lownorm control over `graphs` through `victim` (possibly defended).
    Mirror of NB.run_om2c_cell: returns asr, queries, max_inj_degree, pooled shifts."""
    F._reset_stats()
    flips = 0; max_deg = 0; shifts = []; per_graph = []
    with QueryCounter(victim) as qc:
        for g in graphs:
            gd = g.to(device)
            fl, md, perturbed = flow_flip_instr(victim, gd, prior, cfg, device)
            flips += int(fl); max_deg = max(max_deg, md)
            if measure_shift and bare_model is not None and perturbed is not None:
                clean_emb = pooled_readout(bare_model, gd, device)
                pert_emb = pooled_readout(bare_model, perturbed, device)
                shift = float((pert_emb - clean_emb).norm().item())
                shifts.append(shift)
                per_graph.append({"shift": round(shift, 5), "flipped": bool(fl),
                                  "n_nodes": int(gd.num_nodes)})
    asr = flips / max(len(graphs), 1)
    return {"asr": asr, "q": qc.n, "max_inj_degree": max_deg,
            "shifts": shifts, "per_graph": per_graph,
            "in_ball": F.FLOW_STATS["in_ball"], "in_ball_total": F.FLOW_STATS["total"]}


ATTACKS = ["OM2C", "FM-eb1-lownorm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--nbs", type=str, default="1,2,3,4,5,8,10,15")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--defenses", type=str, default=None,
                    help="comma-separated defense-set override (default: module DEFENSES=4). "
                         "Pass the 9-line set for the extended stealth sweep; EFFECTIVE "
                         "(floor) stays the original 3 so DV2 remains comparable.")
    args = ap.parse_args()
    if args.defenses:
        global DEFENSES
        DEFENSES = [d.strip() for d in args.defenses.split(",")]

    # ---- device + GPU probe (same logic as eval_nodecount_ablation) ----
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
        seeds = seeds[:1]; cap = 3; fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta0 = load_meta_for_seed(seeds[0])

    # per-(attack, nb) accumulators
    und = {a: {nb: [] for nb in nbs} for a in ATTACKS}        # U per-unit undefended ASR
    floor = {a: {nb: [] for nb in nbs} for a in ATTACKS}      # F per-unit worst-case floor
    shift = {a: {nb: [] for nb in nbs} for a in ATTACKS}      # S per-unit mean pooled shift
    asr_by_def = {a: {nb: {d: [] for d in DEFENSES} for nb in nbs} for a in ATTACKS}
    pairs = {a: {nb: [] for nb in nbs} for a in ATTACKS}      # per-graph (shift, flipped)
    max_deg = {a: {nb: 0 for nb in nbs} for a in ATTACKS}
    inball = {a: {nb: [0, 0] for nb in nbs} for a in ATTACKS}
    q_by = {a: {nb: 0 for nb in nbs} for a in ATTACKS}
    victim_accs = []; clean_acc_def = {d: [] for d in DEFENSES}
    units = []

    print(f"=== 2-ATTACK NODE-COUNT ABLATION | {meta0['dataset']} {meta0.get('model_type','GCN')} | "
          f"attacks={ATTACKS} | seeds={seeds} cap={cap}/fold nbs={nbs} | defenses={DEFENSES} | "
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
            om2c_prior = om2c_prior_on_cpu(graphs, device)
            flow_prior = flow_prior_on_cpu(graphs, device)
            # defended models + defended clean acc (once per fold; shared across attacks/nbs)
            defended = {}; cacc = {}
            for dn in DEFENSES:
                v = build_defended(model, dn, graphs, device)
                defended[dn] = v
                ca = clean_test_acc(v, ds, fold_info["test_idx"], device)
                clean_acc_def[dn].append(ca); cacc[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc, "nb": {}}

            for nb in nbs:
                unit["nb"][str(nb)] = {}
                for a in ATTACKS:
                    if a == "OM2C":
                        cfg = {**O.CONFIG, "node_budget": nb, "feat_scale": 1.0}
                        runner = run_om2c_cell
                    else:
                        cfg = {**F.CONFIG, "node_budget": nb,
                               "edge_strategy": "topk", "feat_scale": 1.0}
                        runner = run_flow_cell
                    prior = om2c_prior if a == "OM2C" else flow_prior
                    asrs = {}
                    for dn in DEFENSES:
                        r = runner(defended[dn], graphs, prior, cfg, device,
                                   measure_shift=(dn == "none"), bare_model=model)
                        asrs[dn] = r["asr"]; asr_by_def[a][nb][dn].append(r["asr"])
                        q_by[a][nb] += r["q"]
                        if dn == "none":
                            und[a][nb].append(r["asr"])
                            shift[a][nb].append(mean(r["shifts"]) if r["shifts"] else 0.0)
                            pairs[a][nb].extend(r["per_graph"])
                            max_deg[a][nb] = max(max_deg[a][nb], r["max_inj_degree"])
                            inball[a][nb][0] += r["in_ball"]; inball[a][nb][1] += r["in_ball_total"]
                    fl = min(asrs[d] for d in EFFECTIVE)
                    floor[a][nb].append(fl)
                    unit["nb"][str(nb)][a] = {
                        "undef_asr": round(asrs["none"], 4), "floor": round(fl, 4),
                        "asr_by_def": {d: round(asrs[d], 4) for d in DEFENSES},
                        "shift_mean": round(shift[a][nb][-1], 5),
                        "max_inj_degree": max_deg[a][nb]}
            units.append(unit)
            tag = "  ".join(
                f"nb{nb}[OM2C U/F={und['OM2C'][nb][-1]*100:.0f}/{floor['OM2C'][nb][-1]*100:.0f}"
                f" FM U/F={und['FM-eb1-lownorm'][nb][-1]*100:.0f}/{floor['FM-eb1-lownorm'][nb][-1]*100:.0f}]"
                for nb in nbs)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  {tag}", flush=True)

    # ---- aggregate ----
    def summ(d):  # d: {nb: [units]}
        return {str(nb): agg_pct(d[nb]) for nb in nbs}
    def summ_raw(d):
        return {str(nb): agg(d[nb]) for nb in nbs}

    und_s = {a: summ(und[a]) for a in ATTACKS}
    floor_s = {a: summ(floor[a]) for a in ATTACKS}
    shift_s = {a: summ_raw(shift[a]) for a in ATTACKS}
    asr_def_s = {a: {str(nb): {d: agg_pct(asr_by_def[a][nb][d]) for d in DEFENSES} for nb in nbs}
                 for a in ATTACKS}

    def knee(summary):  # max first-difference on the per-nb mean
        ms = [summary[str(nb)]["mean_pct"] for nb in nbs]
        diffs = [(nbs[i + 1], ms[i + 1] - ms[i]) for i in range(len(nbs) - 1)]
        return max(diffs, key=lambda kv: kv[1]) if diffs else (None, None)

    knees = {a: {"undefended_asr": knee(und_s[a]), "floor": knee(floor_s[a])} for a in ATTACKS}

    # H1 verdict: OM2C floor rises (max-min spread), FM floor stays flat-low (max small).
    om2c_floor_means = [floor_s["OM2C"][str(nb)]["mean_pct"] for nb in nbs]
    fm_floor_means = [floor_s["FM-eb1-lownorm"][str(nb)]["mean_pct"] for nb in nbs]
    verdict = {
        "om2c_floor_min_pct": round(min(om2c_floor_means), 2),
        "om2c_floor_max_pct": round(max(om2c_floor_means), 2),
        "om2c_floor_rise_pct": round(max(om2c_floor_means) - min(om2c_floor_means), 2),
        "fm_floor_min_pct": round(min(fm_floor_means), 2),
        "fm_floor_max_pct": round(max(fm_floor_means), 2),
        "fm_floor_rise_pct": round(max(fm_floor_means) - min(fm_floor_means), 2),
        "gap_at_max_nb_pct": round(om2c_floor_means[-1] - fm_floor_means[-1], 2),
        "H1_supported": bool((max(om2c_floor_means) - min(om2c_floor_means)) >= 20.0
                             and max(fm_floor_means) <= 20.0),
        "H1_criterion": "OM2C floor rises >=20pp across nb AND FM floor stays <=20pp at every nb",
    }

    out = {
        "stamp": stamp,
        "experiment": "TWO-ATTACK NODE-COUNT ABLATION (OM2C vs FM-eb1-lownorm) — IV=node_budget x attack",
        "hypothesis_H1": "OM2C floor RISES with nb (knee ~4-5, plateau ~55%); FM-eb1-lownorm control "
                         "floor STAYS LOW at every nb (flow wiring -> degree==node_budget -> degfilter "
                         "prunes regardless of count). Gap = degree-1 sparse design, present at every "
                         "node count, not the node budget. H0: both floors track each other.",
        "attacks": {"OM2C": "om2c, feat_scale=1.0 (degree-1 sparse-greedy mass + on-manifold feats)",
                    "FM-eb1-lownorm": "flow, edge_strategy=topk, feat_scale=1.0 (no tangent/sparse-greedy; "
                                      "realized per-node degree == node_budget)"},
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device_requested": want, "device_used": device, "gpu_probe": probe,
        "seeds": seeds, "cap_per_fold": cap, "node_budgets": nbs, "n_units": len(units),
        "defenses": DEFENSES, "effective_defenses": EFFECTIVE, "degfilter_k": DEGFILTER_K,
        "controls": {"edge_budget": "1 nominal (OM2C degree-1 greedy; FM topk wires deg==nb)",
                     "feat_scale": 1.0, "om2c_max_edges_greedy": O.CONFIG["max_edges"],
                     "flow_T_flow": F.CONFIG["T_flow"], "flow_n_restarts": F.CONFIG["n_restarts"],
                     "flow_grad_method": F.CONFIG["grad_method"]},
        "pooled_shift_def": "L2 distance between mean-pooled readout (input to GCN.classifier) of "
                            "clean vs final perturbed graph, on the UNDEFENDED model",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "DV1_undefended_asr": {a: und_s[a] for a in ATTACKS},
        "DV2_worstcase_floor": {a: floor_s[a] for a in ATTACKS},
        "DV3_pooled_shift": {a: shift_s[a] for a in ATTACKS},
        "asr_by_defense": {a: asr_def_s[a] for a in ATTACKS},
        "knee": {a: {"undefended_asr": {"nb": knees[a]["undefended_asr"][0],
                                        "marginal_gain_pct": round(knees[a]["undefended_asr"][1], 2)
                                        if knees[a]["undefended_asr"][1] is not None else None},
                     "floor": {"nb": knees[a]["floor"][0],
                               "marginal_gain_pct": round(knees[a]["floor"][1], 2)
                               if knees[a]["floor"][1] is not None else None}} for a in ATTACKS},
        "max_inj_degree_observed": {a: {str(nb): max_deg[a][nb] for nb in nbs} for a in ATTACKS},
        "in_ball_fraction": {a: {str(nb): (round(inball[a][nb][0] / inball[a][nb][1], 4)
                                           if inball[a][nb][1] else None) for nb in nbs} for a in ATTACKS},
        "queries_by_nb": {a: {str(nb): q_by[a][nb] for nb in nbs} for a in ATTACKS},
        "H1_verdict": verdict,
        "per_unit": {a: {"undef_asr": {str(nb): [round(x, 4) for x in und[a][nb]] for nb in nbs},
                         "floor": {str(nb): [round(x, 4) for x in floor[a][nb]] for nb in nbs},
                         "shift_mean": {str(nb): [round(x, 5) for x in shift[a][nb]] for nb in nbs}}
                     for a in ATTACKS},
        "per_graph_shift_flip": {a: {str(nb): pairs[a][nb] for nb in nbs} for a in ATTACKS},
        "wall_s": round(time.time() - t0, 1),
        "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"nodecount_2attack_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 100, flush=True)
    print(f"device_used={device}  victim_acc: mean={out['victim_acc_mean']} "
          f"std={out['victim_acc_std']} min={out['victim_acc_min']}", flush=True)
    for a in ATTACKS:
        print(f"\n--- {a} ---", flush=True)
        print(f"{'nb':>4} {'undef_ASR%':>14} {'floor%':>14} {'pooled_shift':>16} {'max_deg':>8}", flush=True)
        for nb in nbs:
            u = und_s[a][str(nb)]; f = floor_s[a][str(nb)]; s = shift_s[a][str(nb)]
            print(f"{nb:>4} {u['mean_pct']:7.1f}±{u['std_pct']:<5.1f} {f['mean_pct']:7.1f}±{f['std_pct']:<5.1f} "
                  f"{s['mean']:8.3f}±{s['std']:<6.3f} {max_deg[a][nb]:>8}", flush=True)
    print(f"\nH1 verdict: {verdict}", flush=True)
    print(f"\nSaved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
