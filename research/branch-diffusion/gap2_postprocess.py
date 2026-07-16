#!/usr/bin/env python3
"""gap2_postprocess.py — seed-level FLOOR4 + paired Wilcoxon (n>=8 seeds).

Merges battery-6 result JSONs (existing 3-seed + new 5-seed run, IDENTICAL params:
cap=10, bismooth T=20 p=0.2, same defenses/attacks, device=cuda) into one >=8-seed
analysis. Drops the INVALID rspool test-time SWAP from the floor; effective defenses =
{normfilter, mahalanobis, degfilter, bismooth} (FLOOR4). Computes per-SEED floor4 per
attack (genuine replicate, n=seeds) and paired Wilcoxon OM2C-nb10 vs each baseline.

Per-seed floor convention matches eval_pg_floor_battery6: floor4_unit = min ASR over the
4 effective defenses for that (seed,fold); per-seed floor4 = mean over that seed's folds.

Usage: micromamba run -n graph_adversarial python gap2_postprocess.py \
         results/floor_battery6_..._A.json results/floor_battery6_..._B.json [...]
"""
import sys, json
from statistics import mean, pstdev
from pathlib import Path

EFF4 = ["normfilter", "mahalanobis", "degfilter", "bismooth"]
CANDIDATE = "OM2C-nb10"

# FIX3 — PRE-REGISTERED hypothesis (recorded BEFORE the canonical run; see also the
# stand-alone results/preregistration_8seed.json written at run-launch time). The stat
# test is fixed in advance: one-sided paired Wilcoxon (alternative='greater'), alpha=0.01,
# framed as an Intersection-Union Test so the conjunction "candidate > EVERY baseline"
# needs NO Bonferroni correction. n=8 => min one-sided Wilcoxon p = 1/2^8 = 0.0039 < 0.01,
# attained iff all 8 seeds are concordant.
PREREGISTRATION = {
    "H1": "OM2C-nb10 seed-level floor (min ASR over effective defenses) > EACH baseline's "
          "seed-level floor, for every baseline.",
    "test": "one-sided paired Wilcoxon signed-rank, alternative='greater'",
    "alpha": 0.01,
    "multiplicity": "Intersection-Union Test (conjunction over baselines) => NO Bonferroni; "
                    "global error controlled at alpha by the max per-baseline p-value.",
    "n_seeds": 8,
    "min_attainable_p": 1.0 / (2 ** 8),
    "power_note": "n=8 one-sided Wilcoxon min p = 1/2^8 = 0.00390625 < 0.01, iff all 8 "
                  "seed-level diffs concordant (candidate>baseline on all 8 seeds).",
    "effective_defenses_floor4": EFF4,
    "decision_rule": "H1 supported iff candidate mean floor > every baseline mean floor AND "
                     "max over baselines of one-sided Wilcoxon p < 0.01.",
}


def load_preregistration():
    """Prefer the stand-alone preregistration JSON (timestamped before the run); fall
    back to the embedded PREREGISTRATION constant if it is absent."""
    p = Path(__file__).resolve().parent / "results" / "preregistration_8seed.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            d["_source_file"] = p.name
            return d
        except Exception:
            pass
    return {**PREREGISTRATION, "_source_file": None}


def load_units(paths):
    units = []
    attacks = None
    for p in paths:
        d = json.loads(Path(p).read_text())
        a = list(d["attacks"].keys())
        attacks = a if attacks is None else attacks
        if set(a) != set(attacks):
            print(f"!! WARN attack-set mismatch in {p}: {a} vs {attacks}", flush=True)
        for u in d["units"]:
            units.append({"seed": u["seed"], "fold": u["fold"], "cells": u["cells"],
                          "src": Path(p).name})
    return units, attacks


def per_seed_floor4(units, attacks):
    # seed -> attack -> list of per-fold floor4
    seeds = sorted({u["seed"] for u in units})
    by = {a: {s: [] for s in seeds} for a in attacks}
    skipped = 0
    for u in units:
        for a in attacks:
            asrs = []
            ok = True
            for dn in EFF4:
                cell = u["cells"].get(f"{a}|{dn}")
                if cell is None or cell.get("asr") is None:
                    ok = False; break
                asrs.append(cell["asr"])
            if ok:
                by[a][u["seed"]].append(min(asrs))
            else:
                skipped += 1
    # per-seed mean floor4
    seed_floor = {a: {s: (mean(by[a][s]) if by[a][s] else None) for s in seeds} for a in attacks}
    return seeds, seed_floor, skipped


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: gap2_postprocess.py <battery6_json> [<battery6_json> ...]"); sys.exit(1)
    units, attacks = load_units(paths)
    seeds, seed_floor, skipped = per_seed_floor4(units, attacks)
    n_seeds = len(seeds)

    prereg = load_preregistration()
    print(f"=== PRE-REGISTERED HYPOTHESIS (recorded before run; src="
          f"{prereg.get('_source_file')}) ===", flush=True)
    print(f"  H1: {prereg['H1']}", flush=True)
    print(f"  test={prereg['test']} | alpha={prereg['alpha']} | IUT (no Bonferroni)", flush=True)
    print(f"  n_seeds={prereg['n_seeds']} | min attainable p = 1/2^8 = "
          f"{1.0/2**8:.6f} < 0.01 iff all 8 seeds concordant", flush=True)

    print(f"\n=== GAP-2 seed-level FLOOR4 (drop invalid rspool-swap) ===", flush=True)
    print(f"sources: {[Path(p).name for p in paths]}", flush=True)
    print(f"effective defenses (FLOOR4): {EFF4}", flush=True)
    print(f"seeds (n={n_seeds}): {seeds}   total units={len(units)}  skipped(missing cell)={skipped}",
          flush=True)
    print(f"\n{'attack':<22} {'mean_floor4_pct':>16} {'std':>7} {'n_seeds':>8}", flush=True)
    summ = {}
    for a in attacks:
        vals = [seed_floor[a][s] for s in seeds if seed_floor[a][s] is not None]
        m = round(mean(vals) * 100, 2) if vals else None
        sd = round(pstdev(vals) * 100, 2) if len(vals) > 1 else 0.0
        summ[a] = {"mean_pct": m, "std_pct": sd, "n_seeds": len(vals),
                   "per_seed_pct": {str(s): (round(seed_floor[a][s] * 100, 2)
                                             if seed_floor[a][s] is not None else None)
                                    for s in seeds}}
        print(f"{a:<22} {str(m):>16} {sd:>7} {len(vals):>8}", flush=True)

    # paired Wilcoxon OM2C-nb10 vs each baseline (paired across seeds)
    from scipy.stats import wilcoxon
    print(f"\n--- paired Wilcoxon (seed-level, n={n_seeds}): {CANDIDATE} vs baseline ---", flush=True)
    paired = {}
    cand = [seed_floor[CANDIDATE][s] for s in seeds]
    for a in attacks:
        if a == CANDIDATE:
            continue
        base = [seed_floor[a][s] for s in seeds]
        pairs = [(c, b) for c, b in zip(cand, base) if c is not None and b is not None]
        cc = [c for c, b in pairs]; bb = [b for c, b in pairs]
        diffs = [c - b for c, b in pairs]
        if len(pairs) < 2 or all(d == 0 for d in diffs):
            res = {"n_pairs": len(pairs), "wilcoxon_p": None,
                   "note": "all_diffs_zero" if pairs else "insufficient",
                   "mean_diff_pct": round(mean(diffs) * 100, 2) if diffs else None}
        else:
            try:
                w, p = wilcoxon(cc, bb, alternative="greater")
                res = {"n_pairs": len(pairs), "wilcoxon_stat": float(w),
                       "wilcoxon_p_greater": float(p),
                       "mean_diff_pct": round(mean(diffs) * 100, 2),
                       "all_cand_gt_base": all(d > 0 for d in diffs)}
            except Exception as e:
                res = {"n_pairs": len(pairs), "error": str(e)}
        paired[a] = res
        pv = res.get("wilcoxon_p_greater", res.get("wilcoxon_p"))
        print(f"  vs {a:<22} mean_diff={res.get('mean_diff_pct')}%  "
              f"p(greater)={pv}  all_cand>base={res.get('all_cand_gt_base')}  "
              f"n={res['n_pairs']}", flush=True)

    crit = (summ[CANDIDATE]["mean_pct"] is not None and
            all(summ[CANDIDATE]["mean_pct"] > summ[a]["mean_pct"]
                for a in attacks if a != CANDIDATE and summ[a]["mean_pct"] is not None) and
            all((paired[a].get("wilcoxon_p_greater") is not None and
                 paired[a]["wilcoxon_p_greater"] < 0.01)
                for a in attacks if a != CANDIDATE))
    print(f"\n=== GAP-2 VERDICT: OM2C-nb10 floor4 > every baseline AND all paired "
          f"p<0.01 (n={n_seeds}) ? {'YES' if crit else 'NO'} ===", flush=True)

    out = {"preregistration": prereg,
           "sources": [Path(p).name for p in paths], "effective_defenses_floor4": EFF4,
           "seeds": seeds, "n_seeds": n_seeds, "n_units": len(units),
           "skipped_units_missing_cell": skipped,
           "seed_floor4_summary": summ, "paired_wilcoxon": paired,
           "gap2_criterion_met": crit}
    from datetime import datetime
    op = Path(__file__).resolve().parent / "results" / \
        f"gap2_floor4_wilcoxon_PROTEINS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"Saved {op}", flush=True)


if __name__ == "__main__":
    main()
