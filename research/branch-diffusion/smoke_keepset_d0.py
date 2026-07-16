#!/usr/bin/env python3
"""
smoke_keepset_d0.py — Diagnostic D0: the GDDM-A edge-survival census.

THE REFRAME (EXPERIMENT_PROGRAM_DIFFUSION_FEASIBILITY.md §2-3):
  GDDM-A purifies ONLY adjacency (features pass through, defenses.py:605) and its
  reverse step is MONOTONE edge-removal (A_pred = A_pred * A_t, can drop never add,
  defenses.py:594). So feasibility-under-GDDM-A is purely an EDGE-SURVIVAL game:
  if the injected node's attachment edges are pruned, the node is disconnected ->
  inert regardless of features. If >=1 adversarial edge survives, features are FREE
  (GDDM-A-blind) and flow through.

  No prior session measured the keep-set. This does. Architecture-free, decisive.

PER-GRAPH MEASUREMENTS (PROTEINS fold0, K=1, t*=30, full-edge attachment):
  undef_flip      : zo_free flips undefended?            (sanity ~1.0)
  def_flip_full   : zo_free flips defended, full-edge?   (the known floor ~0.03-0.17)
  S0              : # injected edges surviving GDDM-A with CLEAN (mean) features
  S_final         : # injected edges surviving with the ATTACK's adversarial features
                    (tests H0's engine: do adversarial features TRIGGER pruning?)
  greedy_def_flip : zo_free flips defended, attachment RESTRICTED to clean keep-set S0?
                    (the survival-greedy treatment; >> floor => feasibility crack)

DECISION GATE (§3, pre-committed):
  - S0 ~ 0 for most graphs                 -> structural: can't attach cleanly (H0).
  - S0 > 0 but S_final << S0               -> adversarial features trigger pruning (H0 engine).
  - greedy_def_flip ASR >= 0.3 (>> floor)  -> FEASIBILITY CRACK -> build Arch B.
  - greedy_def_flip ASR ~ floor            -> surviving edges can't route signal (H0).
"""
import json
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import (MODELS_DIR, DATA_DIR, DEVICE, GCN_GraphClassification,  # noqa: E402
                     HIDDEN_DIM)
from sweep_stealth import load_fold  # noqa: E402
from defenses import DefendedModel, gddm_a_defense, gddm_a_ckpt_path  # noqa: E402
import attack  # noqa: E402

PER_GRAPH_TIMEOUT_S = 150
OUTER_WALLCLOCK_S = 40 * 60
SMALL_N = 50
N_TARGET = 20
T_STAR_A = 30


def _load_dataset(name):
    """Return (victim_model, test_graphs[fold0]). PROTEINS uses the prepared meta.json
    path; extra datasets (COX2, ...) use the bench_extra_datasets pipeline + models_extra."""
    if name == "PROTEINS":
        meta = json.loads((MODELS_DIR / "meta.json").read_text())
        model, graphs = load_fold(meta["folds"][0], meta)
        return model, graphs
    # extra datasets
    from torch_geometric.datasets import TUDataset
    from bench_extra_datasets import stratified_folds, MODELS_EXTRA
    ds = TUDataset(root=str(DATA_DIR / name), name=name)
    folds = stratified_folds(ds)
    _, _, _, test_idx = folds[0]
    model = GCN_GraphClassification(ds.num_features, ds.num_classes, HIDDEN_DIM).to(DEVICE)
    ckpt = MODELS_EXTRA / f"{name}_gcn_fold0.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    graphs = []
    for i in test_idx:
        g = ds[i].clone().to(DEVICE)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
        graphs.append(g)
    return model, graphs


class _TimeoutErr(Exception):
    pass


def _alarm(signum, frame):
    raise _TimeoutErr("per-graph timeout")


def _cfg(feat_mode):
    c = {
        "feat_mode": feat_mode,
        "node_budget": 1,
        "use_inter_edges": False,
        "zo_epochs": 20, "zo_lr": 0.5, "zo_sigma": 5e-3, "zo_restarts": 2,
        "_norm_radius": float("inf"),
        "seed_offset": 0,
    }
    if feat_mode == "sgdi":
        c["sgdi_eta"] = 0.3
        c["sgdi_temp"] = 0.1
    return c


def _optimize(model, data, feats0, v_edges, device, cfg, prior):
    """Dispatch feature optimization on fixed attachment v_edges. Returns final feats."""
    if cfg["feat_mode"] == "sgdi":
        return attack.sgdi_optimize_features(model, data, feats0, v_edges, [],
                                             cfg, device, prior)
    return attack.zo_optimize_features(model, data, feats0, v_edges, [], cfg, device)


def _injected_survivors(data, feats, v_edges, device, ckpt_a):
    """Build G' (K=1 inj node at index n) + v_edges, run RAW gddm_a, return the set of
    victim nodes whose edge to the injected node SURVIVES purification."""
    n = data.num_nodes
    g_pert = attack._build(data, feats, v_edges, [], device)
    purified = gddm_a_defense(g_pert, ckpt_path=ckpt_a, t_star=T_STAR_A)
    ei = purified.edge_index
    inj = n
    inc = ei[1][ei[0] == inj]
    return sorted(set(inc.tolist()))


def _flips(model, data, feats0, v_edges, device, cfg, prior):
    """Optimize features on fixed attachment v_edges vs `model`; (flipped, final_feats)."""
    if not v_edges:
        return False, feats0
    feats = _optimize(model, data, feats0, v_edges, device, cfg, prior)
    true_label = data.y.item()
    mg = attack._eval_graphs(model, [attack._build(data, feats, v_edges, [], device)],
                             true_label, device)[0].item()
    return bool(mg < 0), feats


def attack_one(model_undef, model_def, g, ckpt_a, device):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PER_GRAPH_TIMEOUT_S)
    try:
        n = g.num_nodes
        stats = attack.data_stats([g], device)
        feats_mean = attack.make_features(stats, 1, "sgdi", n, device)
        prior = attack.data_prior([g], device)          # (mu, cov_inv, L, thr) for sgdi
        c_sgdi = _cfg("sgdi")
        c_free = _cfg("zo_free")
        v_all = [(0, v) for v in range(n)]              # full-edge attachment (rho=1.0)

        # (1) clean-feature keep-set S0 (mean features)
        S0 = _injected_survivors(g, feats_mean, v_all, device, ckpt_a)

        # ---- PRIMARY regime: sgdi (Mahalanobis-bounded, feature-stealthy) ----
        undef_sgdi, feats_adv = _flips(model_undef, g, feats_mean, v_all, device,
                                       c_sgdi, prior)
        # survivors under the sgdi adversarial features (coupling test = H0 engine)
        S_final = _injected_survivors(g, feats_adv, v_all, device, ckpt_a)
        def_full_sgdi, _ = _flips(model_def, g, feats_mean, v_all, device, c_sgdi, prior)
        # survival-greedy: restrict attachment to clean keep-set, sgdi-optimize vs defended
        v_keep = [(0, v) for v in S0]
        greedy_sgdi = (_flips(model_def, g, feats_mean, v_keep, device, c_sgdi, prior)[0]
                       if S0 else False)

        # ---- REFERENCE: zo_free (feature-blind bypass via pool domination) ----
        def_full_free, _ = _flips(model_def, g, feats_mean, v_all, device, c_free, prior)
    finally:
        signal.alarm(0)
    return {
        "n_nodes": int(n), "y": int(g.y.item()),
        "S0": len(S0), "S0_rate": len(S0) / max(n, 1),
        "S_final_sgdi": len(S_final), "S_final_rate": len(S_final) / max(n, 1),
        "undef_sgdi": int(undef_sgdi),
        "def_full_sgdi": int(def_full_sgdi),
        "greedy_sgdi": int(greedy_sgdi),
        "def_full_zofree": int(def_full_free),
        "keepset_empty": int(len(S0) == 0),
    }


def main():
    t_start = time.time()
    quick = "--quick" in sys.argv
    n_target = 2 if quick else N_TARGET
    DATASET = next((a for a in sys.argv[1:] if not a.startswith("-")), "PROTEINS")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"keepset_d0_{DATASET}_{ts}.json"

    model, graphs = _load_dataset(DATASET)
    print(f"[load] {DATASET} fold0 graphs={len(graphs)} quick={quick}", flush=True)

    small = []
    for g in graphs:
        if g.num_nodes > SMALL_N:
            continue
        with torch.no_grad():
            if model(g).argmax(-1).item() != g.y.item():
                continue
        small.append(g)
        if len(small) >= n_target:
            break
    print(f"[filter] small correctly-classified (n<={SMALL_N}): {len(small)}", flush=True)

    ckpt_a = gddm_a_ckpt_path(DATASET)
    model_def = DefendedModel(model, gddm_a_defense, "gddm_a",
                              ckpt_path=ckpt_a, t_star=T_STAR_A)

    rows, skipped = [], []
    for idx, g in enumerate(small):
        if time.time() - t_start > OUTER_WALLCLOCK_S:
            print(f"[outer-timeout] after {len(rows)} graphs", flush=True)
            break
        t0 = time.time()
        try:
            r = attack_one(model, model_def, g, ckpt_a, DEVICE)
            r["idx"] = idx
            r["wall_s"] = time.time() - t0
            rows.append(r)
            print(f"[{idx:02d}] n={r['n_nodes']:>3d} "
                  f"S0={r['S0']:>2d}/{r['n_nodes']:<3d} Sfin_sgdi={r['S_final_sgdi']:>2d} | "
                  f"sgdi: undef={r['undef_sgdi']} defFull={r['def_full_sgdi']} "
                  f"greedy={r['greedy_sgdi']} | zofree_defFull={r['def_full_zofree']}  "
                  f"({r['wall_s']:.0f}s)", flush=True)
        except _TimeoutErr:
            skipped.append({"idx": idx, "n_nodes": int(g.num_nodes), "reason": "timeout"})
            print(f"[{idx:02d}] n={g.num_nodes} SKIPPED (timeout)", flush=True)
        except Exception as e:
            skipped.append({"idx": idx, "n_nodes": int(g.num_nodes),
                            "reason": f"{type(e).__name__}:{e}"})
            print(f"[{idx:02d}] ERROR {type(e).__name__}: {e}", flush=True)

        # aggregate + persist each step
        nd = len(rows)
        mean = lambda k: (sum(r[k] for r in rows) / nd) if nd else None
        agg = {
            "dataset": DATASET, "defense": "gddm_a", "t_star_a": T_STAR_A,
            "K": 1, "small_n_cap": SMALL_N, "n_done": nd, "n_skipped": len(skipped),
            "mean_S0_rate": mean("S0_rate"),
            "mean_S_final_rate_sgdi": mean("S_final_rate"),
            "frac_keepset_empty": mean("keepset_empty"),
            "asr_undef_sgdi": mean("undef_sgdi"),
            "asr_def_full_sgdi": mean("def_full_sgdi"),
            "asr_greedy_sgdi": mean("greedy_sgdi"),
            "asr_def_full_zofree": mean("def_full_zofree"),
            "wall_s": time.time() - t_start,
            "per_graph": rows, "skipped": skipped,
            "floor_reference": {"sgdi_gddm_a_n40": 0.175, "v3_floor_n30": 0.033},
        }
        out_path.write_text(json.dumps(agg, indent=2))

    print("\n=== D0 RESULT (gddm_a, K=1, full-edge) ===", flush=True)
    print(f"  mean clean keep-set rate S0/n              : {agg['mean_S0_rate']:.3f}", flush=True)
    print(f"  mean sgdi-adversarial keep rate Sfin/n     : {agg['mean_S_final_rate_sgdi']:.3f}", flush=True)
    print(f"  frac graphs EMPTY clean keep-set           : {agg['frac_keepset_empty']:.3f}", flush=True)
    print(f"  [sgdi] ASR undefended                      : {agg['asr_undef_sgdi']:.3f}", flush=True)
    print(f"  [sgdi] ASR defended full-edge (FLOOR)      : {agg['asr_def_full_sgdi']:.3f}", flush=True)
    print(f"  [sgdi] ASR defended survival-greedy        : {agg['asr_greedy_sgdi']:.3f}", flush=True)
    print(f"  [zo_free] ASR defended (feature-blind ref) : {agg['asr_def_full_zofree']:.3f}", flush=True)
    print(f"  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
