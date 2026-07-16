#!/usr/bin/env python3
"""run_camera_ready_grid.py — camera-ready statistical-rigor grid.

Cells: dataset {PROTEINS, ENZYMES} x defense {none, normfilter, jaccard, mahalanobis}
       x feat_mode {zo_normcap, sgdi}.  16 cells total.
Per cell: --seeds {0,1,2} --n-folds 5  =>  3*5 = 15 paired observations.

(Note: only 5 folds exist on disk per dataset, so we use 3 seeds instead of expanding
to 10 folds; paired n = 15 with seed-major ordering aligned across feat_modes.)

Cheap-first order so we maximize information per wall-clock unit.
Writes results JSON incrementally to results/eval_camera_ready_<dataset>_<stamp>.json
(one file per dataset, atomic rename per cell).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
PY = ["micromamba", "run", "-n", "graph_adversarial", "python", str(ROOT / "run_cell.py")]

# Per-defense soft timeout (seconds).  ENZYMES & mahalanobis are the slow ones.
TIMEOUT_BY_KEY = {
    ("PROTEINS", "none"): 700,
    ("PROTEINS", "normfilter"): 700,
    ("PROTEINS", "jaccard"): 700,
    ("PROTEINS", "mahalanobis"): 1500,
    ("ENZYMES", "none"): 2200,
    ("ENZYMES", "normfilter"): 2200,
    ("ENZYMES", "jaccard"): 2200,
    ("ENZYMES", "mahalanobis"): 3300,
}

FEAT_MODES = ["zo_normcap", "sgdi"]  # paired-test order
DEFENSE_ORDER = ["none", "normfilter", "jaccard", "mahalanobis"]  # cheap-first


def atomic_write(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def run_one(cell, log_fp):
    args = [
        "--dataset", cell["dataset"],
        "--feat-mode", cell["feat_mode"],
        "--defense", cell["defense"],
        "--seeds", cell["seeds"],
        "--n-folds", str(cell["n_folds"]),
    ]
    timeout = TIMEOUT_BY_KEY.get((cell["dataset"], cell["defense"]), 1800)
    cmd = ["timeout", str(timeout)] + PY + args
    t0 = time.time()
    log_fp.write(f"[{datetime.now().strftime('%H:%M:%S')}] LAUNCH {cell} timeout={timeout}s\n")
    log_fp.flush()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        log_fp.write(f"  TIMEOUT after {time.time()-t0:.0f}s\n")
        return {**cell, "status": "timeout", "wall_s": round(time.time() - t0, 1)}
    out, err = p.stdout, p.stderr
    for line in out.splitlines():
        if line.startswith("CELL_RESULT_JSON:"):
            rec = json.loads(line[len("CELL_RESULT_JSON:"):])
            rec.setdefault("status", "ok")
            log_fp.write(f"  OK ASR={rec['asr_mean']:.3f} n_obs={rec['n_obs']} wall={rec['wall_s']:.0f}s\n")
            log_fp.flush()
            return rec
    if p.returncode == 124:
        log_fp.write(f"  TIMEOUT (rc=124) {time.time()-t0:.0f}s\n")
        return {**cell, "status": "timeout", "wall_s": round(time.time() - t0, 1),
                "stderr_tail": err[-600:]}
    log_fp.write(f"  ERROR rc={p.returncode} wall={time.time()-t0:.0f}s\n  stderr_tail={err[-600:]}\n")
    log_fp.flush()
    return {**cell, "status": "error", "rc": p.returncode,
            "wall_s": round(time.time() - t0, 1),
            "stderr_tail": err[-600:], "stdout_tail": out[-600:]}


def build_cells(dataset, seeds, n_folds):
    cells = []
    for de in DEFENSE_ORDER:
        for fm in FEAT_MODES:
            cells.append({
                "dataset": dataset, "feat_mode": fm, "defense": de,
                "seeds": seeds, "n_folds": n_folds,
            })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="PROTEINS,ENZYMES")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--budget-s", type=int, default=21600, help="overall wall budget (default 6h)")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"camera_ready_{stamp}.log"
    log_fp = open(log_path, "a", buffering=1)
    log_fp.write(f"=== camera-ready grid {stamp} datasets={args.datasets} seeds={args.seeds} folds={args.n_folds} ===\n")

    datasets = args.datasets.split(",")
    t_start = time.time()

    for ds in datasets:
        cells = build_cells(ds, args.seeds, args.n_folds)
        out_path = RESULTS / f"eval_camera_ready_{ds}_{stamp}.json"
        meta = {
            "stamp": stamp, "dataset": ds, "feat_modes": FEAT_MODES,
            "defenses": DEFENSE_ORDER, "seeds": args.seeds, "n_folds": args.n_folds,
            "note": ("Camera-ready statistical-rigor grid. 3 seeds × 5 folds = 15 paired "
                     "observations per (dataset, defense). Only 5 folds exist on disk per "
                     "dataset; 3 seeds substituted for the originally-requested 10 folds."),
        }
        completed = []
        atomic_write(out_path, {**meta, "cells": completed})
        log_fp.write(f"--- {ds}: {len(cells)} cells -> {out_path}\n")

        for i, cell in enumerate(cells):
            elapsed = time.time() - t_start
            # stop launching at 5h to leave time for finalization
            if elapsed > min(args.budget_s, 5 * 3600):
                log_fp.write(f"[BUDGET] elapsed {elapsed:.0f}s; stop launching at {ds} cell {i}/{len(cells)}\n")
                break
            log_fp.write(f"[{i+1}/{len(cells)}] {ds} {cell['feat_mode']:<11} {cell['defense']:<12} (overall_elapsed={elapsed:.0f}s)\n")
            log_fp.flush()
            rec = run_one(cell, log_fp)
            completed.append(rec)
            atomic_write(out_path, {**meta, "cells": completed})

    log_fp.write(f"=== DONE total_wall={time.time()-t_start:.0f}s ===\n")
    log_fp.close()


if __name__ == "__main__":
    main()
