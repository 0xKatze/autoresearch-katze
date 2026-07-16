#!/usr/bin/env python3
"""run_floor_gin10_driver.py — THIN DRIVER (no attack/model/defense code modified).

Runs the validated ENZYMES-continuous / GIN floor sweep (eval_pg_floor_enzymes_gin) against
the 10-fold victims produced by gen_victims_gin10_driver.py (models_cont_10fold/). The stock
floor runner's load_meta() reads models_cont/ with a fixed filename; we override ONLY that
lookup to point at models_cont_10fold/ (so the validated 5-fold metas in models_cont/ stay
untouched), set the requested args (seeds 42,7,1027; cap=20), and call its main() verbatim.
Floor harness is CPU-pinned by design (CUDA_VISIBLE_DEVICES="" set at import in the module).
"""
import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = ["floor", "--seeds", "42,7,1027", "--cap", "20", "--dataset", "ENZYMES"]
import eval_pg_floor_enzymes_gin as M  # noqa: E402

_TENFOLD = HERE / "models_cont_10fold"


def _load_meta_10fold(seed, dataset):
    p = _TENFOLD / f"meta_cont_gin_{dataset}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing 10-fold GIN victim meta for seed {seed}: {p}")
    return json.loads(p.read_text())


M.load_meta = _load_meta_10fold   # redirect lookup; no file edited, 5-fold metas preserved
M.main()
