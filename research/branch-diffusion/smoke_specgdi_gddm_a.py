#!/usr/bin/env python3
"""
smoke_specgdi_gddm_a.py — Single make-or-break number:
  PROTEINS  feat_mode=specgdi  defense=gddm_a  ASR  (small-graph subset).

Reference (from results/smoke_gddm_a.json, n=40, t_star_a=30):
  sgdi   PROTEINS no_def = 0.975
  sgdi   PROTEINS gddm_a = 0.175   (Δ=-80pp)

Decision rule:
  ASR >= 0.6  -> spectrum-preservation thesis CONFIRMED on gddm_a
  ASR <= 0.3  -> thesis FAILED
  otherwise   -> INCONCLUSIVE

Scope (kept lean to fit the wall-clock budget):
  - PROTEINS, fold 0, seed 0
  - up to 15 SMALL graphs (n_nodes <= 50)
  - specgdi_swaps_per_step = 2  (the per-graph cost reducer)
  - gddm_a t_star = 30  (matches the sgdi reference run)
  - per-graph soft timeout via signal.SIGALRM (2 min); outer hard cap 20 min
"""
import json
import os
import signal
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

# Repo-local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import MODELS_DIR, DEVICE  # noqa: E402
from sweep_stealth import load_fold as load_fold_discrete  # noqa: E402
from defenses import DefendedModel, gddm_a_defense, gddm_a_ckpt_path  # noqa: E402
import attack  # noqa: E402


PER_GRAPH_TIMEOUT_S = 120     # skip a graph if it hangs >2 min
OUTER_WALLCLOCK_S = 20 * 60   # 20 min total hard cap
SMALL_N = 50                  # n_nodes <= SMALL_N counts as "small"
N_TARGET = 15                 # cap of small graphs to attack
T_STAR_A = 30
SWAPS_PER_STEP = 2
DATASET = "PROTEINS"


class _TimeoutErr(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutErr("per-graph timeout")


def attack_one(model_undef, model_def, g, cfg):
    """Attack a single graph with feat_mode=specgdi against (undef, def)."""
    # Single-graph list so stats/prior come from this graph; OK for one-hot PROTEINS.
    # Signal-based timeout works because we're in the main thread.
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PER_GRAPH_TIMEOUT_S)
    try:
        o_no = attack.run_attack_sweep(model_undef, [g], DEVICE, dict(cfg))
        o_a = attack.run_attack_sweep(model_def, [g], DEVICE, dict(cfg))
    finally:
        signal.alarm(0)
    return o_no, o_a


def main():
    t_start = time.time()
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"smoke_specgdi_gddm_a_PROTEINS_{ts}.json"

    # ---- load PROTEINS fold 0 ----
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    print(f"[load] PROTEINS fold0 graphs={len(graphs)}", flush=True)

    # Filter to small graphs the model classifies correctly (clean), n <= SMALL_N
    small = []
    for g in graphs:
        if g.num_nodes > SMALL_N:
            continue
        with torch.no_grad():
            if model(g).argmax(-1).item() != g.y.item():
                continue
        small.append(g)
        if len(small) >= N_TARGET:
            break
    print(f"[filter] small (n<={SMALL_N}, correctly-classified): {len(small)}", flush=True)

    # ---- defended model ----
    ckpt_a = gddm_a_ckpt_path(DATASET)
    dmodel_a = DefendedModel(model, gddm_a_defense, "gddm_a",
                             ckpt_path=ckpt_a, t_star=T_STAR_A)

    # ---- specgdi cfg (matches smoke_gddm_a.py contract) ----
    cfg = {
        "node_budget": 1,
        "skip_greedy": True,
        "zo_epochs": 20,
        "zo_restarts": 2,
        "feat_mode": "specgdi",
        "seed_offset": 0,
        "specgdi_swaps_per_step": SWAPS_PER_STEP,
    }

    per_graph = []
    skipped = []
    flips_no = 0
    flips_a = 0
    n_done = 0
    overall_timed_out = False

    for idx, g in enumerate(small):
        # outer wall-clock guard
        if time.time() - t_start > OUTER_WALLCLOCK_S:
            print(f"[outer-timeout] wall={int(time.time()-t_start)}s "
                  f"after {n_done} graphs; breaking.", flush=True)
            overall_timed_out = True
            break

        t0 = time.time()
        try:
            o_no, o_a = attack_one(model, dmodel_a, g, cfg)
            dt = time.time() - t0
            flipped_no = bool(o_no.get("asr_full", 0.0) >= 0.5)
            flipped_a = bool(o_a.get("asr_full", 0.0) >= 0.5)
            flips_no += int(flipped_no)
            flips_a += int(flipped_a)
            n_done += 1
            per_graph.append({
                "idx": idx,
                "n_nodes": int(g.num_nodes),
                "y": int(g.y.item()),
                "asr_no_def": float(o_no.get("asr_full", 0.0)),
                "asr_gddm_a": float(o_a.get("asr_full", 0.0)),
                "rho_no_def": o_no.get("rho_to_flip", [None])[0],
                "rho_gddm_a": o_a.get("rho_to_flip", [None])[0],
                "s_spec_no_def": (o_no.get("s_spec_per_graph") or [None])[0],
                "s_spec_gddm_a": (o_a.get("s_spec_per_graph") or [None])[0],
                "tau_spec": o_no.get("tau_spec"),
                "wall_s": dt,
            })
            print(f"[{idx:02d}] n={g.num_nodes:>3d}  asr_no={float(o_no['asr_full']):.0f}  "
                  f"asr_gddm_a={float(o_a['asr_full']):.0f}   "
                  f"({dt:.0f}s; running flips no/a = {flips_no}/{flips_a})",
                  flush=True)
        except _TimeoutErr:
            dt = time.time() - t0
            skipped.append({"idx": idx, "n_nodes": int(g.num_nodes),
                            "reason": "per_graph_timeout", "wall_s": dt})
            print(f"[{idx:02d}] n={g.num_nodes}  SKIPPED (>{PER_GRAPH_TIMEOUT_S}s)",
                  flush=True)
        except Exception as e:
            dt = time.time() - t0
            skipped.append({"idx": idx, "n_nodes": int(g.num_nodes),
                            "reason": f"exception:{type(e).__name__}:{e}",
                            "wall_s": dt})
            print(f"[{idx:02d}] n={g.num_nodes}  ERROR: {type(e).__name__}: {e}",
                  flush=True)

        # incremental persistence
        cur = {
            "partial": True,
            "dataset": DATASET,
            "feat_mode": "specgdi",
            "defense": "gddm_a",
            "t_star_a": T_STAR_A,
            "specgdi_swaps_per_step": SWAPS_PER_STEP,
            "small_n_cap": SMALL_N,
            "n_target": N_TARGET,
            "n_done": n_done,
            "n_skipped": len(skipped),
            "flips_no_def": flips_no,
            "flips_gddm_a": flips_a,
            "asr_no_def": (flips_no / n_done) if n_done else None,
            "asr_gddm_a": (flips_a / n_done) if n_done else None,
            "wall_s": time.time() - t_start,
            "per_graph": per_graph,
            "skipped": skipped,
        }
        out_path.write_text(json.dumps(cur, indent=2))

    wall = time.time() - t_start
    asr_no = (flips_no / n_done) if n_done else 0.0
    asr_a = (flips_a / n_done) if n_done else 0.0

    final = {
        "partial": False,
        "dataset": DATASET,
        "feat_mode": "specgdi",
        "defense": "gddm_a",
        "t_star_a": T_STAR_A,
        "specgdi_swaps_per_step": SWAPS_PER_STEP,
        "small_n_cap": SMALL_N,
        "n_target": N_TARGET,
        "n_done": n_done,
        "n_skipped": len(skipped),
        "flips_no_def": flips_no,
        "flips_gddm_a": flips_a,
        "asr_no_def": asr_no,
        "asr_gddm_a": asr_a,
        "wall_s": wall,
        "outer_timed_out": overall_timed_out,
        "reference_sgdi_PROTEINS_no_def": 0.975,
        "reference_sgdi_PROTEINS_gddm_a": 0.175,
        "delta_vs_sgdi_pp": (asr_a - 0.175) * 100,
        "per_graph": per_graph,
        "skipped": skipped,
    }
    out_path.write_text(json.dumps(final, indent=2))

    print()
    print(f"RESULT specgdi gddm_a {DATASET} ASR={asr_a:.3f} "
          f"n_graphs={n_done} wall={int(wall)}s "
          f"(skipped={len(skipped)}; undef_ASR={asr_no:.3f}; "
          f"Δ vs sgdi_gddm_a={asr_a-0.175:+.3f})",
          flush=True)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
