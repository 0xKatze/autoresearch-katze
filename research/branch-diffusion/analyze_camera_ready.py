#!/usr/bin/env python3
"""analyze_camera_ready.py — paired Wilcoxon + bootstrap CI on the camera-ready grid.

Reads results/eval_camera_ready_<dataset>_<stamp>.json files, runs paired tests
sgdi vs zo_normcap per (dataset, defense), writes
results/camera_ready_wilcoxon_<stamp>.json.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def bootstrap_median_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    boot_medians = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_medians.append(np.median(diffs[idx]))
    boot_medians = np.array(boot_medians)
    lo = float(np.quantile(boot_medians, alpha / 2))
    hi = float(np.quantile(boot_medians, 1 - alpha / 2))
    return float(np.median(diffs)), lo, hi


def bootstrap_mean_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(diffs[idx].mean())
    boot_means = np.array(boot_means)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(diffs.mean()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--datasets", default="PROTEINS,ENZYMES")
    args = ap.parse_args()
    datasets = args.datasets.split(",")

    all_cells = []
    file_map = {}
    for ds in datasets:
        p = RESULTS / f"eval_camera_ready_{ds}_{args.stamp}.json"
        if not p.exists():
            print(f"WARN: {p} missing")
            continue
        d = json.loads(p.read_text())
        for c in d.get("cells", []):
            if c.get("status") == "ok":
                all_cells.append(c)
        file_map[ds] = str(p)

    # Index
    by = {(c["dataset"], c["defense"], c["feat_mode"]): c for c in all_cells}
    out_rows = []
    table_rows = []
    for ds in datasets:
        for defe in ["none", "normfilter", "jaccard", "mahalanobis"]:
            sgdi = by.get((ds, defe, "sgdi"))
            normcap = by.get((ds, defe, "zo_normcap"))
            row = {"dataset": ds, "defense": defe,
                   "sgdi_mean": None, "sgdi_std": None,
                   "normcap_mean": None, "normcap_std": None,
                   "n_pairs": 0, "delta_mean": None,
                   "delta_mean_ci95": None,
                   "delta_median": None, "delta_median_ci95": None,
                   "wilcoxon_p": None, "wilcoxon_stat": None,
                   "sgdi_obs": None, "normcap_obs": None}
            if sgdi:
                row["sgdi_mean"] = sgdi["asr_mean"]
                row["sgdi_std"] = sgdi["asr_std"]
                row["sgdi_obs"] = sgdi["asr_obs"]
                row["sgdi_q_per_graph"] = sgdi.get("q_per_graph")
                row["sgdi_clean_acc"] = sgdi.get("clean_acc_def_mean")
            if normcap:
                row["normcap_mean"] = normcap["asr_mean"]
                row["normcap_std"] = normcap["asr_std"]
                row["normcap_obs"] = normcap["asr_obs"]
                row["normcap_q_per_graph"] = normcap.get("q_per_graph")
                row["normcap_clean_acc"] = normcap.get("clean_acc_def_mean")

            if sgdi and normcap:
                a = np.array(sgdi["asr_obs"], dtype=float)
                b = np.array(normcap["asr_obs"], dtype=float)
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
                row["n_pairs"] = n
                diffs = a - b
                mu, mlo, mhi = bootstrap_mean_ci(diffs)
                med, lo, hi = bootstrap_median_ci(diffs)
                row["delta_mean"] = round(mu, 4)
                row["delta_mean_ci95"] = [round(mlo, 4), round(mhi, 4)]
                row["delta_median"] = round(med, 4)
                row["delta_median_ci95"] = [round(lo, 4), round(hi, 4)]
                if wilcoxon is not None and n > 0 and np.any(diffs != 0):
                    try:
                        w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                        row["wilcoxon_p"] = float(w.pvalue)
                        row["wilcoxon_stat"] = float(w.statistic)
                    except Exception as e:
                        row["wilcoxon_p_error"] = str(e)
            out_rows.append(row)
            table_rows.append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS / f"camera_ready_wilcoxon_{args.stamp}.json"
    out_path.write_text(json.dumps({
        "source_stamp": args.stamp, "analyzed_at": stamp,
        "files": file_map, "rows": out_rows,
    }, indent=2))
    print(f"Saved {out_path}")

    # Print summary
    print()
    print(f"{'dataset':<9} {'defense':<12} {'sgdi':>10} {'normcap':>10} {'Δ':>10} {'CI95':>22} {'p':>10}")
    for r in table_rows:
        s = f"{r['sgdi_mean']:.3f}" if r['sgdi_mean'] is not None else "NA"
        n = f"{r['normcap_mean']:.3f}" if r['normcap_mean'] is not None else "NA"
        d = f"{r['delta_mean']:+.3f}" if r['delta_mean'] is not None else "NA"
        ci = (f"[{r['delta_mean_ci95'][0]:+.3f},{r['delta_mean_ci95'][1]:+.3f}]"
              if r.get('delta_mean_ci95') else "NA")
        p = f"{r['wilcoxon_p']:.4g}" if r['wilcoxon_p'] is not None else "NA"
        print(f"{r['dataset']:<9} {r['defense']:<12} {s:>10} {n:>10} {d:>10} {ci:>22} {p:>10}")


if __name__ == "__main__":
    main()
