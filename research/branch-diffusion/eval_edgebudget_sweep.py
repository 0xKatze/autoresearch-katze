#!/usr/bin/env python3
"""eval_edgebudget_sweep.py — EDGE-BUDGET (per-node degree) ablation, orthogonal
to the node-count sweep. Holds the NUMBER of injected nodes fixed (nb=10) and
sweeps the per-injected-node DEGREE eb, to isolate the *degree* axis from the
*count* axis. Complements eval_nodecount_2attack.py (which fixed degree and swept
count).

IV  = eb (per-injected-node degree) in {1,3,5,8}, at FIXED nb=10, for the dense
      control attack:
  * FM-dense : plain on-manifold flow-matching injection. edge_strategy wires EACH
               injected node to select_targets_topk(data, eb) targets, so the
               realized per-node degree == eb (verified via max_inj_degree). Low-norm
               (feat_scale=1.0) to match the on-manifold feature axis.
               ("flow", targets=topk(eb), node_budget=10, feat_scale=1.0)

Reference (flat across the eb-axis; not part of the swept IV — shown for context):
  * OM2C     : on-manifold two-channel attack with its NATIVE degree-1 sparse-greedy
               mass channel (adaptive greedy + fraction-of-N anchors). Its per-node
               degree is ~1 BY DESIGN and is NOT driver-controllable, so it is a single
               reference point at nb=10, plotted flat. ("om2c", node_budget=10, fs=1.0)

Controls (fixed): node_budget nb=10, feat_scale=1.0, PROTEINS/GCN 200-epoch victims,
seeds 42/7/1027, cap>=10 graphs/fold, 10-fold => up to 30 units per (attack, eb).

Budget-% annotation (PROTEINS, avg undirected edges = 72.82): total injected edges
  = nb * eb = 10 * eb, as % of the average graph's undirected edges:
  eb=1 -> 10 edges = 13.7% | eb=3 -> 30 = 41.2% | eb=5 -> 50 = 68.7% | eb=8 -> 80 = 109.9%.

THREE DVs per (attack, eb) — identical harness/definitions to eval_nodecount_2attack.py:
  (U) UNDEFENDED ASR (defense=none).
  (F) WORST-CASE-DEFENSE FLOOR = min ASR over {normfilter, mahalanobis, degfilter}.
  (S) POOLED-EMBEDDING SHIFT = L2 distance between the global-mean-pooled readout
      (input to GCN.classifier) of clean vs final perturbed graph, on the UNDEFENDED model.

H1: FM-dense's UNDEFENDED ASR RISES with eb (more edges -> stronger structural push),
    but its FLOOR STAYS LOW at every eb because degfilter (k * mean-degree threshold)
    prunes high-degree injected nodes regardless of how many edges they carry. OM2C's
    degree-1 design is the floor lever: it sits at the sweet spot where degfilter cannot
    distinguish the injected node from real low-degree nodes. H0: FM-dense floor tracks
    its undefended ASR upward with eb.

DRIVER ONLY — no attack/model/defense code modified. The FM side composes attack_flow's
OWN exported functions; degree is set purely by which `targets` the driver passes
(select_targets_topk(data, eb)). The OM2C side reuses eval_nodecount_ablation.run_om2c_cell
VERBATIM.

Usage: micromamba run -n graph_adversarial python3 eval_edgebudget_sweep.py \
         [--seeds 42,7,1027] [--cap 10] [--ebs 1,3,5,8] [--nb 10] [--device cuda] [--quick]
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

import eval_nodecount_ablation as NB
from eval_nodecount_ablation import (DEFENSES, EFFECTIVE, DEGFILTER_K,  # noqa
                                     build_defended, om2c_prior_on_cpu, run_om2c_cell,
                                     clean_test_acc, pooled_readout, QueryCounter,
                                     load_meta_for_seed, agg, agg_pct)
# reuse the 2-attack driver's flow-prior helper + fs resolver (driver glue, not attack code)
from eval_nodecount_2attack import flow_prior_on_cpu, _resolve_fs
import attack_om2c as O      # noqa
import attack_flow as F      # noqa

# PROTEINS average undirected edges (computed from the full TUDataset; see module docstring).
AVG_UNDIRECTED_EDGES = 72.82


def budget_pct(nb, eb):
    """Total injected edges (nb*eb) as a % of the average graph's undirected edges."""
    return round(100.0 * (nb * eb) / AVG_UNDIRECTED_EDGES, 1)


# ----------------------------------------------------------------------------
# Instrumented FM flip with an EXPLICIT per-node degree `eb`: each injected node
# connects to select_targets_topk(data, eb) targets => realized degree == eb. Mirrors
# eval_nodecount_2attack.flow_flip_instr but decouples degree (eb) from the number of
# injected nodes (cfg["node_budget"]). Captures the outcome-determining perturbed graph
# by wrapping construct_perturbed_graph. No attack code modified.
# ----------------------------------------------------------------------------
def flow_flip_instr_eb(model, data, prior, cfg, device, eb):
    n_feat = data.x.size(1)
    m = cfg["node_budget"]                       # number of injected nodes (fixed = nb)
    fs = _resolve_fs(cfg["feat_scale"], data.num_nodes)
    targets = F.select_targets_topk(data, eb)    # per-node degree == eb (driver-controlled)

    cap = {}
    orig_cpg = F.construct_perturbed_graph
    def wrapped(d, x, tgts, fscale):
        g = orig_cpg(d, x, tgts, fscale)
        cap["g"] = g
        return g
    F.construct_perturbed_graph = wrapped
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

    md = 0
    if last_g is not None:
        ei = last_g.edge_index; on = data.num_nodes
        for idx in range(on, on + m):
            md = max(md, int((ei[0] == idx).sum().item()))
    return flipped, md, last_g


def run_flow_cell_eb(victim, graphs, prior, cfg, device, eb, measure_shift=False, bare_model=None):
    """Run FM-dense at per-node degree eb over `graphs` through `victim` (possibly defended)."""
    F._reset_stats()
    flips = 0; max_deg = 0; shifts = []; per_graph = []
    with QueryCounter(victim) as qc:
        for g in graphs:
            gd = g.to(device)
            fl, md, perturbed = flow_flip_instr_eb(victim, gd, prior, cfg, device, eb)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--ebs", type=str, default="1,3,5,8")
    ap.add_argument("--nb", type=int, default=10)
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

    # ---- device + GPU probe (same logic as eval_nodecount_2attack) ----
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
    ebs = [int(x) for x in args.ebs.split(",")]
    nb = args.nb
    cap = args.cap; fold_limit = None
    if args.quick:
        seeds = seeds[:1]; cap = 3; fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta0 = load_meta_for_seed(seeds[0])

    # per-eb accumulators for the swept attack (FM-dense)
    und = {eb: [] for eb in ebs}
    floor = {eb: [] for eb in ebs}
    shift = {eb: [] for eb in ebs}
    asr_by_def = {eb: {d: [] for d in DEFENSES} for eb in ebs}
    pairs = {eb: [] for eb in ebs}
    max_deg = {eb: 0 for eb in ebs}
    inball = {eb: [0, 0] for eb in ebs}
    q_by = {eb: 0 for eb in ebs}
    # OM2C reference (flat across eb) — native degree-1 greedy at nb
    om2c_und = []; om2c_floor = []; om2c_shift = []
    om2c_by_def = {d: [] for d in DEFENSES}; om2c_max_deg = 0; om2c_q = 0
    victim_accs = []; clean_acc_def = {d: [] for d in DEFENSES}
    units = []

    print(f"=== EDGE-BUDGET SWEEP | {meta0['dataset']} {meta0.get('model_type','GCN')} | "
          f"FM-dense eb={ebs} @ nb={nb} (+OM2C native ref) | seeds={seeds} cap={cap}/fold | "
          f"defenses={DEFENSES} | device={device} | {stamp} ===", flush=True)
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
            defended = {}; cacc = {}
            for dn in DEFENSES:
                v = build_defended(model, dn, graphs, device)
                defended[dn] = v
                ca = clean_test_acc(v, ds, fold_info["test_idx"], device)
                clean_acc_def[dn].append(ca); cacc[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc,
                    "fm_dense": {}, "om2c_ref": {}}

            # ---- FM-dense eb-sweep (the IV) ----
            fcfg = {**F.CONFIG, "node_budget": nb, "feat_scale": 1.0}
            for eb in ebs:
                asrs = {}
                for dn in DEFENSES:
                    r = run_flow_cell_eb(defended[dn], graphs, flow_prior, fcfg, device, eb,
                                         measure_shift=(dn == "none"), bare_model=model)
                    asrs[dn] = r["asr"]; asr_by_def[eb][dn].append(r["asr"]); q_by[eb] += r["q"]
                    if dn == "none":
                        und[eb].append(r["asr"])
                        shift[eb].append(mean(r["shifts"]) if r["shifts"] else 0.0)
                        pairs[eb].extend(r["per_graph"])
                        max_deg[eb] = max(max_deg[eb], r["max_inj_degree"])
                        inball[eb][0] += r["in_ball"]; inball[eb][1] += r["in_ball_total"]
                fl = min(asrs[d] for d in EFFECTIVE)
                floor[eb].append(fl)
                unit["fm_dense"][str(eb)] = {
                    "undef_asr": round(asrs["none"], 4), "floor": round(fl, 4),
                    "asr_by_def": {d: round(asrs[d], 4) for d in DEFENSES},
                    "shift_mean": round(shift[eb][-1], 5),
                    "max_inj_degree": max_deg[eb]}

            # ---- OM2C native reference (degree-1 greedy at nb) ----
            ocfg = {**O.CONFIG, "node_budget": nb, "feat_scale": 1.0}
            oasrs = {}
            for dn in DEFENSES:
                r = run_om2c_cell(defended[dn], graphs, om2c_prior, ocfg, device,
                                  measure_shift=(dn == "none"), bare_model=model)
                oasrs[dn] = r["asr"]; om2c_by_def[dn].append(r["asr"]); om2c_q += r["q"]
                if dn == "none":
                    om2c_und.append(r["asr"])
                    om2c_shift.append(mean(r["shifts"]) if r["shifts"] else 0.0)
                    om2c_max_deg = max(om2c_max_deg, r["max_inj_degree"])
            ofl = min(oasrs[d] for d in EFFECTIVE)
            om2c_floor.append(ofl)
            unit["om2c_ref"] = {"undef_asr": round(oasrs["none"], 4), "floor": round(ofl, 4),
                                "asr_by_def": {d: round(oasrs[d], 4) for d in DEFENSES},
                                "max_inj_degree": om2c_max_deg}

            units.append(unit)
            tag = "  ".join(
                f"eb{eb}[U/F={und[eb][-1]*100:.0f}/{floor[eb][-1]*100:.0f} deg={max_deg[eb]}]"
                for eb in ebs)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} n={len(graphs)}  {tag}  "
                  f"OM2C[U/F={om2c_und[-1]*100:.0f}/{om2c_floor[-1]*100:.0f} deg={om2c_max_deg}]",
                  flush=True)

    # ---- aggregate ----
    und_s = {str(eb): agg_pct(und[eb]) for eb in ebs}
    floor_s = {str(eb): agg_pct(floor[eb]) for eb in ebs}
    shift_s = {str(eb): agg(shift[eb]) for eb in ebs}
    asr_def_s = {str(eb): {d: agg_pct(asr_by_def[eb][d]) for d in DEFENSES} for eb in ebs}
    om2c_s = {"undef_asr": agg_pct(om2c_und), "floor": agg_pct(om2c_floor),
              "shift": agg(om2c_shift), "asr_by_def": {d: agg_pct(om2c_by_def[d]) for d in DEFENSES},
              "max_inj_degree": om2c_max_deg}

    fm_floor_means = [floor_s[str(eb)]["mean_pct"] for eb in ebs]
    fm_und_means = [und_s[str(eb)]["mean_pct"] for eb in ebs]
    verdict = {
        "fm_undef_min_pct": round(min(fm_und_means), 2),
        "fm_undef_max_pct": round(max(fm_und_means), 2),
        "fm_undef_rise_pct": round(max(fm_und_means) - min(fm_und_means), 2),
        "fm_floor_min_pct": round(min(fm_floor_means), 2),
        "fm_floor_max_pct": round(max(fm_floor_means), 2),
        "fm_floor_rise_pct": round(max(fm_floor_means) - min(fm_floor_means), 2),
        "om2c_floor_pct": om2c_s["floor"]["mean_pct"],
        "H1_supported": bool(max(fm_floor_means) <= 20.0
                             and om2c_s["floor"]["mean_pct"] >= max(fm_floor_means)),
        "H1_criterion": "FM-dense floor stays <=20pp at every eb (degfilter prunes high-degree "
                        "injection) AND OM2C degree-1 floor >= FM-dense floor at every eb",
    }

    out = {
        "stamp": stamp,
        "experiment": "EDGE-BUDGET SWEEP (FM-dense per-node degree eb @ fixed nb) — IV=eb, orthogonal "
                      "to eval_nodecount_2attack.py (which fixed degree and swept node count)",
        "hypothesis_H1": verdict["H1_criterion"],
        "attacks": {"FM-dense": "flow, targets=select_targets_topk(data, eb), node_budget=nb, "
                                "feat_scale=1.0 (per-node degree == eb, driver-controlled)",
                    "OM2C_ref": "om2c, node_budget=nb, feat_scale=1.0 (native degree-1 sparse-greedy; "
                                "NOT eb-swept — degree is ~1 by design and not driver-controllable)"},
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device_requested": want, "device_used": device, "gpu_probe": probe,
        "seeds": seeds, "cap_per_fold": cap, "node_budget_fixed": nb, "edge_budgets": ebs,
        "n_units": len(units), "defenses": DEFENSES, "effective_defenses": EFFECTIVE,
        "degfilter_k": DEGFILTER_K, "avg_undirected_edges": AVG_UNDIRECTED_EDGES,
        "budget_pct_by_eb": {str(eb): budget_pct(nb, eb) for eb in ebs},
        "budget_pct_def": "total injected edges (nb*eb) as % of avg graph's undirected edges",
        "controls": {"node_budget": nb, "feat_scale": 1.0,
                     "flow_T_flow": F.CONFIG["T_flow"], "flow_n_restarts": F.CONFIG["n_restarts"],
                     "flow_grad_method": F.CONFIG["grad_method"],
                     "om2c_max_edges_greedy": O.CONFIG["max_edges"]},
        "pooled_shift_def": "L2 distance between mean-pooled readout (input to GCN.classifier) of "
                            "clean vs final perturbed graph, on the UNDEFENDED model",
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4),
        "victim_acc_min": round(min(victim_accs), 4),
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4)} for d in DEFENSES},
        "DV1_undefended_asr": {"FM-dense": und_s, "OM2C_ref": om2c_s["undef_asr"]},
        "DV2_worstcase_floor": {"FM-dense": floor_s, "OM2C_ref": om2c_s["floor"]},
        "DV3_pooled_shift": {"FM-dense": shift_s, "OM2C_ref": om2c_s["shift"]},
        "asr_by_defense": {"FM-dense": asr_def_s, "OM2C_ref": om2c_s["asr_by_def"]},
        "max_inj_degree_observed": {"FM-dense": {str(eb): max_deg[eb] for eb in ebs},
                                    "OM2C_ref": om2c_s["max_inj_degree"]},
        "in_ball_fraction": {str(eb): (round(inball[eb][0] / inball[eb][1], 4)
                                       if inball[eb][1] else None) for eb in ebs},
        "queries": {"FM-dense": {str(eb): q_by[eb] for eb in ebs}, "OM2C_ref": om2c_q},
        "H1_verdict": verdict,
        "per_unit": {"FM-dense": {"undef_asr": {str(eb): [round(x, 4) for x in und[eb]] for eb in ebs},
                                  "floor": {str(eb): [round(x, 4) for x in floor[eb]] for eb in ebs}},
                     "OM2C_ref": {"undef_asr": [round(x, 4) for x in om2c_und],
                                  "floor": [round(x, 4) for x in om2c_floor]}},
        "per_graph_shift_flip": {str(eb): pairs[eb] for eb in ebs},
        "wall_s": round(time.time() - t0, 1),
        "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"edgebudget_sweep_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 100, flush=True)
    print(f"device_used={device}  victim_acc: mean={out['victim_acc_mean']} "
          f"std={out['victim_acc_std']} min={out['victim_acc_min']}", flush=True)
    print(f"\n--- FM-dense (per-node degree eb @ nb={nb}) ---", flush=True)
    print(f"{'eb':>4} {'edges':>6} {'budget%':>8} {'undef_ASR%':>14} {'floor%':>14} {'max_deg':>8}", flush=True)
    for eb in ebs:
        u = und_s[str(eb)]; f = floor_s[str(eb)]
        print(f"{eb:>4} {nb*eb:>6} {budget_pct(nb,eb):>7.1f} {u['mean_pct']:7.1f}±{u['std_pct']:<5.1f} "
              f"{f['mean_pct']:7.1f}±{f['std_pct']:<5.1f} {max_deg[eb]:>8}", flush=True)
    print(f"\n--- OM2C native ref (degree-1, nb={nb}) ---", flush=True)
    print(f"  undef_ASR%={om2c_s['undef_asr']['mean_pct']:.1f}±{om2c_s['undef_asr']['std_pct']:.1f}  "
          f"floor%={om2c_s['floor']['mean_pct']:.1f}±{om2c_s['floor']['std_pct']:.1f}  "
          f"max_deg={om2c_s['max_inj_degree']}", flush=True)
    print(f"\nH1 verdict: {verdict}", flush=True)
    print(f"\nSaved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
