#!/usr/bin/env python3
"""run_decisive_grid.py — orchestrates the decisive (dataset, feat_mode, defense) grid.

Spawns run_cell.py per cell with `timeout 900` to enforce 15-min per-cell cap.
Writes results JSON incrementally after each cell (atomic rename).
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PY = ["micromamba", "run", "-n", "graph_adversarial", "python", str(ROOT / "run_cell.py")]
PER_CELL_TIMEOUT = 900  # 15 min hard cap


def atomic_write(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def run_one(cell):
    args = [
        "--dataset", cell["dataset"],
        "--feat-mode", cell["feat_mode"],
        "--defense", cell["defense"],
        "--seeds", cell["seeds"],
        "--n-folds", str(cell["n_folds"]),
    ]
    if cell.get("max_n") is not None:
        args += ["--max-n", str(cell["max_n"])]
    if cell.get("specgdi_swaps") is not None:
        args += ["--specgdi-swaps", str(cell["specgdi_swaps"])]
    cmd = ["timeout", str(PER_CELL_TIMEOUT)] + PY + args
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_CELL_TIMEOUT + 30)
    except subprocess.TimeoutExpired:
        return {**cell, "status": "timeout", "wall_s": round(time.time() - t0, 1)}
    out = p.stdout
    err = p.stderr
    # Find marker line
    for line in out.splitlines():
        if line.startswith("CELL_RESULT_JSON:"):
            rec = json.loads(line[len("CELL_RESULT_JSON:"):])
            rec.setdefault("status", "ok")
            return rec
    if p.returncode == 124:  # timeout exit
        return {**cell, "status": "timeout", "wall_s": round(time.time() - t0, 1),
                "stderr_tail": err[-400:]}
    return {**cell, "status": "error", "rc": p.returncode,
            "wall_s": round(time.time() - t0, 1), "stderr_tail": err[-400:],
            "stdout_tail": out[-400:]}


def build_grid(dataset, feat_modes, defenses, seeds, n_folds, max_n=None, specgdi_swaps=None):
    cells = []
    # Order: cheap feat_modes first so we accumulate value before expensive ones
    for fm in feat_modes:
        for de in defenses:
            cells.append({
                "dataset": dataset, "feat_mode": fm, "defense": de,
                "seeds": seeds, "n_folds": n_folds,
                "max_n": max_n,
                "specgdi_swaps": specgdi_swaps if fm in ("specgdi", "specgdi_signal") else None,
            })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ENZYMES", "PROTEINS"])
    ap.add_argument("--feat-modes", default="zo_normcap,sgdi,specgdi")
    ap.add_argument("--defenses", default="none,mahalanobis,gnnsafe,gddm")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--specgdi-swaps", type=int, default=None)
    ap.add_argument("--out", required=True, help="results JSON path")
    ap.add_argument("--budget-s", type=int, default=12600, help="overall wall budget")
    args = ap.parse_args()

    feat_modes = args.feat_modes.split(",")
    defenses = args.defenses.split(",")
    cells = build_grid(args.dataset, feat_modes, defenses, args.seeds, args.n_folds,
                       max_n=args.max_n, specgdi_swaps=args.specgdi_swaps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    meta = {
        "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": args.dataset, "feat_modes": feat_modes, "defenses": defenses,
        "seeds": args.seeds, "n_folds": args.n_folds, "max_n": args.max_n,
        "specgdi_swaps_override": args.specgdi_swaps,
        "per_cell_timeout_s": PER_CELL_TIMEOUT,
        "note": ("PROTEINS-capped subset n<=max_n; "
                 "specgdi_swaps reduced from default 8 to override value (if set). "
                 "1 seed used due to wall-clock budget — fewer paired obs than 2-seed."),
    }
    atomic_write(out_path, {**meta, "cells": completed})

    t_start = time.time()
    for i, cell in enumerate(cells):
        elapsed = time.time() - t_start
        if elapsed > args.budget_s:
            print(f"[BUDGET] reached {elapsed:.0f}s > {args.budget_s}s; stopping at cell {i}/{len(cells)}")
            break
        print(f"[{i+1}/{len(cells)}] {cell['dataset']:<9} {cell['feat_mode']:<14} {cell['defense']:<12} "
              f"(elapsed={elapsed:.0f}s)")
        sys.stdout.flush()
        rec = run_one(cell)
        completed.append(rec)
        if rec.get("status") == "ok":
            print(f"  -> ok ASR={rec['asr_mean']:.3f} clean={rec['clean_acc_def_mean']:.3f} "
                  f"q/g={rec['q_per_graph']:.0f} wall={rec['wall_s']:.0f}s")
        else:
            print(f"  -> {rec.get('status')} wall={rec.get('wall_s', '?')}s")
        atomic_write(out_path, {**meta, "cells": completed})
    print(f"DONE: {len(completed)}/{len(cells)} cells, wall {time.time()-t_start:.0f}s, file {out_path}")


if __name__ == "__main__":
    main()
