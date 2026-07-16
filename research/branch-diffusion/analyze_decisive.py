#!/usr/bin/env python3
"""analyze_decisive.py — paired Wilcoxon contrasts on decisive grid results."""
import argparse, json, sys
from pathlib import Path

import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def load_cells(paths):
    cells = []
    meta = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        cells += d.get("cells", [])
        meta[Path(p).name] = {k: d.get(k) for k in ("dataset", "max_n", "specgdi_swaps_override", "seeds", "n_folds")}
    return cells, meta


def contrast(by, ds, defe, A, B):
    a_rec = by.get((ds, defe, A))
    b_rec = by.get((ds, defe, B))
    if not a_rec or not b_rec or a_rec.get("status") != "ok" or b_rec.get("status") != "ok":
        return None
    a = np.array(a_rec.get("asr_obs", []), dtype=float)
    b = np.array(b_rec.get("asr_obs", []), dtype=float)
    if len(a) == 0 or len(a) != len(b):
        return {"a_mean": float(a.mean()) if len(a) else None,
                "b_mean": float(b.mean()) if len(b) else None, "n": min(len(a), len(b))}
    delta = float(a.mean() - b.mean())
    p = None
    if wilcoxon is not None and np.any(a - b != 0):
        try:
            p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            p = None
    return {"a": A, "b": B, "a_mean": float(a.mean()), "b_mean": float(b.mean()),
            "delta": round(delta, 4), "wilcoxon_p": p, "n": len(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells, meta = load_cells(args.inputs)
    by = {(c.get("dataset"), c.get("defense"), c.get("feat_mode")): c for c in cells}

    defenses = sorted({c.get("defense") for c in cells if c.get("defense")})
    datasets = sorted({c.get("dataset") for c in cells if c.get("dataset")})

    contrasts = []
    for ds in datasets:
        for defe in defenses:
            for (A, B) in [("specgdi", "sgdi"), ("specgdi", "zo_normcap"), ("sgdi", "zo_normcap")]:
                r = contrast(by, ds, defe, A, B)
                if r is not None:
                    contrasts.append({"dataset": ds, "defense": defe, **r})

    # ASR table
    table = {}
    for ds in datasets:
        table[ds] = {}
        for defe in defenses:
            row = {}
            for fm in ("zo_normcap", "sgdi", "specgdi", "specgdi_signal"):
                c = by.get((ds, defe, fm))
                if c and c.get("status") == "ok":
                    row[fm] = {"asr_mean": c["asr_mean"], "asr_std": c.get("asr_std"),
                                "clean_acc_def_mean": c.get("clean_acc_def_mean"),
                                "n_obs": c.get("n_obs")}
            table[ds][defe] = row

    out_payload = {"inputs_meta": meta, "asr_table": table, "contrasts": contrasts}
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(json.dumps(out_payload, indent=2))


if __name__ == "__main__":
    main()
