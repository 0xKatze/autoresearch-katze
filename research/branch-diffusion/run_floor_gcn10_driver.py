#!/usr/bin/env python3
"""run_floor_gcn10_driver.py — THIN DRIVER (no attack/model/defense code modified).

Round-4 disentangle: runs the VALIDATED ENZYMES-continuous floor sweep (eval_pg_floor_enzymes_gin)
against the 10-fold GCN victims (gen_victims_gcn10_driver.py -> models_cont_10fold/), so the GCN
backbone is measured with the EXACT same harness, attacks, 3-effective defense battery, attack-on-
correct ASR, and seed-level (n=3) aggregation as the GIN backbone. We override ONLY:
  (1) the module's victim model class GIN_GraphClassification -> GCN_GraphClassification (the floor
      runner instantiates this name as a module global at call time, so monkeypatching is sufficient),
  (2) load_meta -> read models_cont_10fold/meta_cont_gcn_*  (validated GIN/5-fold metas untouched).
Numbers are then computed against a GCN victim; only two STATIC label strings in the harness's output
dict are hardcoded "GIN", so after main() we relabel victim_model/experiment -> GCN and rename the
output file floor_enzymes_gin_* -> floor_enzymes_gcn_* (label correction, not a number change). The
harness is CPU-pinned at import (CUDA_VISIBLE_DEVICES="").
"""
import sys, json, glob, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = ["floor", "--seeds", "42,7,1027", "--cap", "20", "--dataset", "ENZYMES"]

import eval_pg_floor_enzymes_gin as M  # noqa: E402  (CPU-pinned at import)
from prepare import GCN_GraphClassification  # noqa: E402

_TENFOLD = HERE / "models_cont_10fold"
_RESULTS = HERE / "results"


def _load_meta_gcn_10fold(seed, dataset):
    p = _TENFOLD / f"meta_cont_gcn_{dataset}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing 10-fold GCN victim meta for seed {seed}: {p}")
    return json.loads(p.read_text())


# (1) swap victim backbone, (2) redirect meta lookup — no file edited
M.GIN_GraphClassification = GCN_GraphClassification
M.load_meta = _load_meta_gcn_10fold

_before = set(glob.glob(str(_RESULTS / "floor_enzymes_gin_ENZYMES_full_*.json")))
M.main()
_after = set(glob.glob(str(_RESULTS / "floor_enzymes_gin_ENZYMES_full_*.json")))

new = sorted(_after - _before)
if not new:
    print("!! WARNING: no new floor_enzymes_gin_*full* output found to relabel — check log", flush=True)
else:
    src = new[-1]
    d = json.loads(Path(src).read_text())
    d["victim_model"] = "GCN"
    d["experiment"] = d.get("experiment", "").replace("GIN victim", "GCN victim").replace("GIN", "GCN")
    d["relabeled_from"] = os.path.basename(src)
    d["relabel_note"] = ("Computed by run_floor_gcn10_driver.py: harness eval_pg_floor_enzymes_gin "
                         "run with GCN_GraphClassification monkeypatched in; numbers are GCN. Only the "
                         "static victim_model/experiment label strings were corrected GIN->GCN.")
    dst = src.replace("floor_enzymes_gin_", "floor_enzymes_gcn_")
    Path(dst).write_text(json.dumps(d, indent=2))
    os.remove(src)
    print(f"Relabeled GCN floor output -> {dst}", flush=True)
