#!/usr/bin/env python3
"""run_cells_list.py — run an explicit list of (feat_mode, defense) cells."""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
PY = ["micromamba", "run", "-n", "graph_adversarial", "python", str(ROOT / "run_cell.py")]
PER_CELL_TIMEOUT = 900


def atomic_write(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def run_one(cell):
    args = [
        "--dataset", cell["dataset"],
        "--feat-mode", cell["feat_mode"],
        "--defense", cell["defense"],
        "--seeds", cell.get("seeds", "0"),
        "--n-folds", str(cell.get("n_folds", 5)),
    ]
    if cell.get("max_n") is not None:
        args += ["--max-n", str(cell["max_n"])]
    if cell.get("specgdi_swaps") is not None:
        args += ["--specgdi-swaps", str(cell["specgdi_swaps"])]
    cmd = ["timeout", str(PER_CELL_TIMEOUT)] + PY + args
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_CELL_TIMEOUT + 30)
    for line in p.stdout.splitlines():
        if line.startswith("CELL_RESULT_JSON:"):
            rec = json.loads(line[len("CELL_RESULT_JSON:"):])
            rec.setdefault("status", "ok")
            return rec
    if p.returncode == 124:
        return {**cell, "status": "timeout", "wall_s": round(time.time() - t0, 1),
                "stderr_tail": p.stderr[-400:]}
    return {**cell, "status": "error", "rc": p.returncode, "wall_s": round(time.time() - t0, 1),
            "stderr_tail": p.stderr[-400:], "stdout_tail": p.stdout[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=int, default=12600)
    args = ap.parse_args()

    cells = json.loads(Path(args.cells_json).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    meta = {"stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "cells_planned": len(cells),
            "per_cell_timeout_s": PER_CELL_TIMEOUT}
    atomic_write(out_path, {**meta, "cells": completed})

    t_start = time.time()
    for i, cell in enumerate(cells):
        elapsed = time.time() - t_start
        if elapsed > args.budget_s:
            print(f"[BUDGET] {elapsed:.0f}s > {args.budget_s}s; stop at {i}/{len(cells)}")
            break
        print(f"[{i+1}/{len(cells)}] {cell.get('dataset'):<9} {cell.get('feat_mode'):<14} "
              f"{cell.get('defense'):<12} max_n={cell.get('max_n')} (elapsed={elapsed:.0f}s)")
        sys.stdout.flush()
        rec = run_one(cell)
        completed.append(rec)
        if rec.get("status") == "ok":
            print(f"  -> ok ASR={rec['asr_mean']:.3f} clean={rec['clean_acc_def_mean']:.3f} "
                  f"q/g={rec['q_per_graph']:.0f} wall={rec['wall_s']:.0f}s")
        else:
            print(f"  -> {rec.get('status')} wall={rec.get('wall_s', '?')}s")
        atomic_write(out_path, {**meta, "cells": completed})
    print(f"DONE: {len(completed)}/{len(cells)} cells, wall {time.time()-t_start:.0f}s, out {out_path}")


if __name__ == "__main__":
    main()
