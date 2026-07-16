#!/usr/bin/env python3
"""eval_edge_subadditivity.py — N30: diffusion-fighting validation (edge-selection).

GO/NO-GO probe for N31 (learned structural flow). Question: is the current hand-crafted
injected-EDGE selection (topk / OM2C sparse-greedy) sub-optimal vs the exhaustively-optimal
k-edge subset, because of message-passing edge interaction ("diffusion fighting")?

The FEATURE channel is FROZEN (CV): injected node features are fixed ONCE per fold via the
OM2C on-manifold projection of the density mode mu at fs=1.0 (O._project_stealth), IDENTICAL
across all three edge-selection methods. Only the edge-selection method varies (the IV), so
any difference is purely structural — this isolates the edge interaction.

IV (edge-selection method), all choosing exactly k edges from the SAME P-edge candidate pool,
all scored with the SAME O._build + O._eval_graphs CW-margin oracle:
  - topk            : the k pool-edges whose victim node has the HIGHEST degree (naive
                      degree heuristic; pool itself is the OM2C low-degree pool).
  - sparse-greedy   : OM2C's degree-1 sparse-greedy loop (om2c_greedy_flip / om2c_attack_graph
                      lines 281-307), restricted to the pool, run for exactly k picks (each pick
                      = argmin CW-margin over remaining candidates).
  - exhaustive-optimal : best (min-CW-margin) k-subset over the full C(P,k) space.

DVs:
  DV1 sub-additivity r = joint-margin-drop / Σ(single-edge margin-drop) over the k chosen
       edges (primary r uses the GREEDY set; optimal-set r reported as a secondary diagnostic).
       single drop_e = m_base - m({e}); joint = m_base - m(set). r<1 => edges fight.
  DV2 flip-rate + mean CW-margin drop per method.
  DV3 optimality gap = optimal - greedy and optimal - topk (flip-rate pp + CW-margin).
  PRIMARY signal = CW-margin drop (continuous, low-variance); flip-rate secondary.

Reference m_base := CW-margin of the graph with the m injected nodes present but ZERO injected
edges (isolated injected nodes). It is the correct edge-selection baseline. A unit whose m_base
is already < 0 (injected nodes alone flip it) is TRIVIAL — recorded but excluded from the
flip-/margin-gap and r statistics (edge selection is irrelevant there).

DECISION RULE: go to N31 IFF mean greedy-set r < 0.8 (real fighting) AND
(optimal - greedy) flip-gap > 10pp. Otherwise greedy is near-optimal => negative result.

DRIVER ONLY — no attack/victim/defense code modified. Reuses the eval_pg_floor_om2c_fm /
eval_paper_grade_floor harness (load_meta_for_seed, clean_test_acc, collect_test_graphs,
QueryCounter) and OM2C's own primitives (O.om2c_prior, O._project_stealth, O._candidate_pool
theory, O._build, O._eval_graphs).

CPU-pinned (small graphs + exhaustive search; must NOT contend with the busy GPU).

Usage: CUDA_VISIBLE_DEVICES="" micromamba run -n graph_adversarial \
         python eval_edge_subadditivity.py [--seeds 42,7,1027] [--folds 5] [--cap 10] \
         [--m 3] [--k 4] [--pool 10] [--device cpu] [--quick]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU ONLY — do NOT contend with GPU runs
import sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev, median
from itertools import combinations

import torch
from torch_geometric.datasets import TUDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare
prepare.DEVICE = "cpu"
DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs  # noqa: E402

import eval_paper_grade_floor as F          # frozen harness (load_meta/clean_acc/QueryCounter)
import attack_om2c as O                      # OM2C primitives

METHODS = ["topk", "sparse-greedy", "exhaustive-optimal"]


def _edge_pool(data, P, m):
    """P candidate injected-edges from the OM2C low-degree victim pool (mirrors
    O._candidate_pool's lowest-degree theory, but with EXACT size P for exhaustive
    feasibility — O._candidate_pool floors the pool at 16). Each edge (i, v) assigns
    a victim node round-robin across the m injected nodes. Returns (edges, victim_degs)."""
    n = data.num_nodes
    deg = torch.bincount(data.edge_index[0], minlength=n).float()
    victim_pool = torch.argsort(deg)[:P].tolist()          # lowest-degree victims, ascending
    edges = [(j % m, int(v)) for j, v in enumerate(victim_pool)]
    victim_degs = [float(deg[v].item()) for v in victim_pool]
    return edges, victim_degs


def _margin(model, data, feats, edge_list, true_label, device):
    """CW margin of the single graph built by injecting feats + edge_list (no inter edges)."""
    g = O._build(data, feats, list(edge_list), [], device)
    return float(O._eval_graphs(model, [g], true_label, device)[0].item())


def _margins_batch(model, data, feats, edge_sets, true_label, device):
    """CW margins for a list of edge_sets (each a list of (i,v) edges), batched in one query."""
    dl = [O._build(data, feats, list(es), [], device) for es in edge_sets]
    return O._eval_graphs(model, dl, true_label, device).tolist()


def _greedy_select(model, data, feats, pool_edges, k, true_label, device):
    """OM2C-style degree-1 sparse greedy restricted to pool_edges, exactly k picks.
    Each step: argmin CW-margin over remaining candidates (reuses O._build/_eval_graphs)."""
    chosen = []
    remaining = list(range(len(pool_edges)))
    last_margin = None
    for _ in range(k):
        if not remaining:
            break
        cand_sets = [chosen + [pool_edges[idx]] for idx in remaining]
        margins = _margins_batch(model, data, feats, cand_sets, true_label, device)
        b = int(min(range(len(margins)), key=lambda t: margins[t]))
        chosen.append(pool_edges[remaining[b]])
        last_margin = margins[b]
        remaining.pop(b)
    return chosen, last_margin


def _topk_select(pool_edges, victim_degs, k):
    """k pool-edges whose victim node has the HIGHEST degree (naive degree heuristic)."""
    order = sorted(range(len(pool_edges)), key=lambda i: victim_degs[i], reverse=True)
    return [pool_edges[i] for i in order[:k]]


def _exhaustive_select(model, data, feats, pool_edges, k, true_label, device):
    """Best (min CW-margin) k-subset over the full C(P,k) space, batched per-chunk."""
    P = len(pool_edges)
    subsets = list(combinations(range(P), k))
    edge_sets = [[pool_edges[i] for i in sub] for sub in subsets]
    best_margin = None; best_set = None
    CHUNK = 256                                            # cap batch size on CPU
    for s in range(0, len(edge_sets), CHUNK):
        chunk = edge_sets[s:s + CHUNK]
        margins = _margins_batch(model, data, feats, chunk, true_label, device)
        for es, mg in zip(chunk, margins):
            if best_margin is None or mg < best_margin:
                best_margin = mg; best_set = es
    return best_set, best_margin, len(subsets)


def attack_unit(model, data, feats, P, m, k, device):
    """Run all three edge-selection methods on ONE graph with FROZEN feats. Returns a unit dict."""
    true_label = data.y.item()
    pool_edges, victim_degs = _edge_pool(data, P, m)
    P_eff = len(pool_edges)
    if P_eff < k:
        return {"skipped": f"P_eff={P_eff} < k={k} (graph too small)"}

    # baseline: injected nodes present, ZERO injected edges
    m_base = _margin(model, data, feats, [], true_label, device)
    trivial = (m_base < 0.0)                               # injected nodes alone already flip

    # single-edge margins for ALL pool edges (for r + diagnostics)
    single_margins = _margins_batch(
        model, data, feats, [[e] for e in pool_edges], true_label, device)
    single_drops = {tuple(pool_edges[i]): (m_base - single_margins[i]) for i in range(P_eff)}

    # --- the three arms (same feats, same oracle) ---
    topk_set = _topk_select(pool_edges, victim_degs, k)
    m_topk = _margin(model, data, feats, topk_set, true_label, device)

    greedy_set, m_greedy = _greedy_select(model, data, feats, pool_edges, k, true_label, device)

    opt_set, m_opt, n_subsets = _exhaustive_select(
        model, data, feats, pool_edges, k, true_label, device)

    drop = {"topk": m_base - m_topk, "sparse-greedy": m_base - m_greedy,
            "exhaustive-optimal": m_base - m_opt}
    final = {"topk": m_topk, "sparse-greedy": m_greedy, "exhaustive-optimal": m_opt}
    flip = {meth: (final[meth] < 0.0) for meth in METHODS}

    # sanity invariant: optimal drop >= greedy drop and >= topk drop (optimal is the max)
    EPS = 1e-5
    sanity_ok = (drop["exhaustive-optimal"] >= drop["sparse-greedy"] - EPS and
                 drop["exhaustive-optimal"] >= drop["topk"] - EPS)

    def _r(sel_set, joint_margin):
        ssum = sum(single_drops[tuple(e)] for e in sel_set)
        joint = m_base - joint_margin
        if ssum <= 0:
            return None, joint, ssum
        return joint / ssum, joint, ssum

    r_greedy, joint_g, ssum_g = _r(greedy_set, m_greedy)
    r_opt, joint_o, ssum_o = _r(opt_set, m_opt)

    return {
        "true_label": int(true_label), "n_nodes": int(data.num_nodes),
        "P_eff": P_eff, "n_subsets": n_subsets, "trivial": bool(trivial),
        "m_base": round(m_base, 5),
        "victim_degs": [round(d, 2) for d in victim_degs],
        "margin_final": {meth: round(final[meth], 5) for meth in METHODS},
        "margin_drop": {meth: round(drop[meth], 5) for meth in METHODS},
        "flip": flip,
        "single_drops": [round(single_drops[tuple(e)], 5) for e in pool_edges],
        "r_greedy": (round(r_greedy, 5) if r_greedy is not None else None),
        "r_optimal": (round(r_opt, 5) if r_opt is not None else None),
        "joint_drop_greedy": round(joint_g, 5), "sum_singles_greedy": round(ssum_g, 5),
        "sanity_ok": bool(sanity_ok),
    }


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {"mean": round(mean(xs), 5),
            "std": round(pstdev(xs), 5) if len(xs) > 1 else 0.0,
            "median": round(median(xs), 5), "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--folds", type=int, default=5, help="use the first N folds")
    ap.add_argument("--max-folds", type=int, default=None, help="alias/override for --folds")
    ap.add_argument("--cap", type=int, default=10, help="test graphs per fold")
    ap.add_argument("--m", type=int, default=3, help="injected nodes")
    ap.add_argument("--k", type=int, default=4, help="injected edges to select")
    ap.add_argument("--pool", type=int, default=10, help="candidate edge-pool size P")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    F.DEVICE = DEVICE; prepare.DEVICE = DEVICE
    seeds = [int(s) for s in args.seeds.split(",")]
    n_folds = args.max_folds if args.max_folds is not None else args.folds
    cap, m, k, P = args.cap, args.m, args.k, args.pool
    if args.quick:
        seeds = seeds[:1]; n_folds = 1; cap = 4

    O._SGDI._FEAT_SCALE = 1.0                              # fs=1.0 (frozen feature channel)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta0 = F.load_meta_for_seed(seeds[0])
    print(f"=== N30 EDGE SUB-ADDITIVITY | {meta0['dataset']} {meta0.get('model_type','GCN')} | "
          f"seeds={seeds} folds={n_folds} cap={cap} | m={m} k={k} P={P} "
          f"(C(P,k)={len(list(combinations(range(P), k)))}) | feat=FROZEN(OM2C on-manifold,fs=1.0) | "
          f"device={DEVICE} | {stamp} ===", flush=True)
    t0 = time.time()

    units = []; errored = []; victim_accs = []
    total_q = 0

    for seed in seeds:
        meta = F.load_meta_for_seed(seed)
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"][:n_folds]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            model.eval()
            vacc = F.clean_test_acc(model, ds, fold_info["test_idx"])
            victim_accs.append(vacc)
            graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
            # OM2C prior over the fold test graphs -> frozen on-manifold injected features
            prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                 O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            mu = prior[0]
            feats = O._project_stealth(mu.unsqueeze(0).repeat(m, 1), prior)  # FROZEN, fs=1.0
            n_fold_units = 0
            with F.QueryCounter(model) as qc:
                for gi, g in enumerate(graphs):
                    try:
                        u = attack_unit(model, g.to(DEVICE), feats, P, m, k, DEVICE)
                    except Exception as e:
                        tb = traceback.format_exc()
                        errored.append({"seed": seed, "fold": fi, "graph": gi,
                                        "error": f"{type(e).__name__}: {e}",
                                        "traceback": tb.splitlines()[-6:]})
                        print(f"  !! ERR seed{seed} fold{fi} g{gi}: {type(e).__name__}: {e}", flush=True)
                        continue
                    if "skipped" in u:
                        continue
                    u.update({"seed": seed, "fold": fi, "graph": gi})
                    units.append(u); n_fold_units += 1
            total_q += qc.n
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} units={n_fold_units} "
                  f"(q+={qc.n})", flush=True)

    # ---------- aggregation ----------
    nontrivial = [u for u in units if not u["trivial"]]
    trivial_n = len(units) - len(nontrivial)

    by_method = {}
    for meth in METHODS:
        drops = [u["margin_drop"][meth] for u in nontrivial]
        flips = [1.0 if u["flip"][meth] else 0.0 for u in nontrivial]
        by_method[meth] = {
            "margin_drop": _agg(drops),
            "flip_rate_pct": (round(100 * mean(flips), 2) if flips else None),
            "n_flips": int(sum(flips)),
        }

    # DV3 gaps (paired, over non-trivial units)
    def _gap_margin(a, b):
        return _agg([u["margin_drop"][a] - u["margin_drop"][b] for u in nontrivial])

    def _flip_pp(a, b):
        if not nontrivial:
            return None
        fa = mean(1.0 if u["flip"][a] else 0.0 for u in nontrivial)
        fb = mean(1.0 if u["flip"][b] else 0.0 for u in nontrivial)
        return round(100 * (fa - fb), 2)

    gaps = {
        "optimal_minus_greedy": {"flip_pp": _flip_pp("exhaustive-optimal", "sparse-greedy"),
                                 "margin": _gap_margin("exhaustive-optimal", "sparse-greedy")},
        "optimal_minus_topk":   {"flip_pp": _flip_pp("exhaustive-optimal", "topk"),
                                 "margin": _gap_margin("exhaustive-optimal", "topk")},
        "greedy_minus_topk":    {"flip_pp": _flip_pp("sparse-greedy", "topk"),
                                 "margin": _gap_margin("sparse-greedy", "topk")},
    }

    # DV1 sub-additivity ratio distribution (greedy set primary)
    r_greedy_vals = [u["r_greedy"] for u in nontrivial if u["r_greedy"] is not None]
    r_opt_vals = [u["r_optimal"] for u in nontrivial if u["r_optimal"] is not None]
    r_greedy_agg = _agg(r_greedy_vals)
    r_opt_agg = _agg(r_opt_vals)
    frac_fighting = (round(sum(1 for r in r_greedy_vals if r < 0.8) / len(r_greedy_vals), 4)
                     if r_greedy_vals else None)

    sanity_violations = sum(1 for u in units if not u["sanity_ok"])

    # ---------- DECISION RULE ----------
    mean_r = r_greedy_agg["mean"]
    flip_gap = gaps["optimal_minus_greedy"]["flip_pp"]
    cond_r = (mean_r is not None and mean_r < 0.8)
    cond_gap = (flip_gap is not None and flip_gap > 10.0)
    go = bool(cond_r and cond_gap)
    verdict = {
        "go_to_N31": go,
        "rule": "go IFF mean greedy-set r < 0.8 AND (optimal-greedy) flip-gap > 10pp",
        "mean_r_greedy": mean_r, "cond_r_lt_0.8": cond_r,
        "optimal_minus_greedy_flip_pp": flip_gap, "cond_gap_gt_10pp": cond_gap,
        "interpretation": (
            "GO: real diffusion fighting (r<0.8) AND greedy leaves a >10pp flip gap vs optimal "
            "=> a learned structural flow (N31) is worth building."
            if go else
            "NO-GO: greedy/topk is near-optimal on this probe "
            f"(r={mean_r}, flip-gap={flip_gap}pp) => structural flow not justified; negative result."),
    }

    out = {
        "stamp": stamp,
        "experiment": "N30 — diffusion-fighting validation: is hand-crafted injected-edge "
                      "selection (topk / OM2C sparse-greedy) sub-optimal vs exhaustive-optimal "
                      "due to message-passing edge interaction? FEATURE channel FROZEN.",
        "hypotheses": {
            "H1_fighting": "joint k-edge CW-margin drop < Σ single-edge drops => r < 1 "
                           "(well below 1 = real fighting).",
            "H2_greedy_gap": "exhaustive-optimal k-subset beats greedy/topk in flip-rate / "
                             "CW-margin (greedy myopia)."},
        "decision_rule": "go to N31 IFF mean greedy r < 0.8 AND (optimal-greedy) flip-gap > 10pp",
        "verdict": verdict,
        "dataset": meta0["dataset"], "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "n_folds_used": n_folds, "cap_per_fold": cap,
        "m_injected": m, "k_edges": k, "pool_P": P,
        "n_subsets_per_unit": len(list(combinations(range(P), k))),
        "feature_channel": "FROZEN — OM2C on-manifold projection O._project_stealth(mu) at "
                           "fs=1.0; identical across all 3 methods (CV intact).",
        "edge_pool_def": "P lowest-degree victim nodes (mirrors O._candidate_pool theory, exact "
                         "size P), assigned round-robin across m injected nodes.",
        "methods": {
            "topk": "k pool-edges with highest victim-node degree (naive degree heuristic)",
            "sparse-greedy": "OM2C degree-1 sparse-greedy (argmin CW-margin per pick), k picks",
            "exhaustive-optimal": "min-CW-margin k-subset over full C(P,k)"},
        "baseline": "m_base = CW-margin with m injected nodes present + ZERO injected edges; "
                    "trivial units (m_base<0) excluded from gap/r stats.",
        "primary_signal": "CW-margin drop (continuous); flip-rate secondary.",
        "victim_acc_mean": round(mean(victim_accs), 4) if victim_accs else None,
        "victim_acc_std": round(pstdev(victim_accs), 4) if len(victim_accs) > 1 else 0.0,
        "victim_acc_min": round(min(victim_accs), 4) if victim_accs else None,
        "n_units": len(units), "n_units_nontrivial": len(nontrivial),
        "n_units_trivial": trivial_n,
        "per_method": by_method,
        "optimality_gaps": gaps,
        "sub_additivity": {
            "r_greedy": r_greedy_agg, "r_optimal": r_opt_agg,
            "frac_units_r_lt_0.8": frac_fighting,
            "r_greedy_distribution": [round(r, 4) for r in r_greedy_vals]},
        "sanity_violations_optimal_not_max": sanity_violations,
        "total_queries": total_q,
        "errored_units": errored,
        "wall_s": round(time.time() - t0, 1),
        "units": units,
    }

    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "quick" if args.quick else "full"
    out_path = out_dir / f"edge_subadditivity_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    # also write a stable-name copy matching the spec's requested output name
    spec_path = out_dir / f"edge_subadditivity_PROTEINS_{stamp}.json"
    if spec_path != out_path:
        spec_path.write_text(json.dumps(out, indent=2))

    print("=" * 92, flush=True)
    print(f"victim_acc mean={out['victim_acc_mean']} min={out['victim_acc_min']} | "
          f"units={len(units)} (nontrivial={len(nontrivial)}, trivial={trivial_n}) | "
          f"sanity_violations={sanity_violations}", flush=True)
    print(f"\n{'method':<20} {'flip%':>7} {'mean drop':>12} {'std drop':>10}", flush=True)
    for meth in METHODS:
        bm = by_method[meth]
        fr = bm["flip_rate_pct"]
        md = bm["margin_drop"]["mean"]
        sd = bm["margin_drop"]["std"]
        fr_s = "NA" if fr is None else f"{fr:6.1f}"
        md_s = "NA" if md is None else f"{md:11.4f}"
        sd_s = "NA" if sd is None else f"{sd:9.4f}"
        print(f"{meth:<20} {fr_s:>7} {md_s:>12} {sd_s:>10}", flush=True)
    print(f"\nDV1 sub-additivity r (greedy set): mean={r_greedy_agg['mean']} "
          f"median={r_greedy_agg['median']} std={r_greedy_agg['std']} n={r_greedy_agg['n']} "
          f"| frac r<0.8 = {frac_fighting}", flush=True)
    print(f"    r (optimal set): mean={r_opt_agg['mean']} median={r_opt_agg['median']}", flush=True)
    print(f"\nDV3 optimality gaps:", flush=True)
    print(f"  optimal-greedy : flip_gap={gaps['optimal_minus_greedy']['flip_pp']}pp  "
          f"margin_gap={gaps['optimal_minus_greedy']['margin']['mean']}", flush=True)
    print(f"  optimal-topk   : flip_gap={gaps['optimal_minus_topk']['flip_pp']}pp  "
          f"margin_gap={gaps['optimal_minus_topk']['margin']['mean']}", flush=True)
    print(f"  greedy-topk    : flip_gap={gaps['greedy_minus_topk']['flip_pp']}pp  "
          f"margin_gap={gaps['greedy_minus_topk']['margin']['mean']}", flush=True)
    print(f"\n>>> VERDICT: {'GO to N31' if go else 'NO-GO (negative result)'} "
          f"(r={mean_r} <0.8? {cond_r}; flip-gap={flip_gap}pp >10? {cond_gap})", flush=True)
    print(verdict["interpretation"], flush=True)
    print(f"\nerrored_units: {len(errored)} | total_queries={total_q}", flush=True)
    print(f"Saved {out_path}  [{out['wall_s']}s]", flush=True)


if __name__ == "__main__":
    main()
