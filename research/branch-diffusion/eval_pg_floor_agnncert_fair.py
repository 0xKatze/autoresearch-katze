#!/usr/bin/env python3
"""eval_pg_floor_agnncert_fair.py — DECISIVE fair AGNNCert floor eval.

Identical harness to eval_pg_floor_agnncert.py (same 5 attacks, same FIX1 defended-correct
denominator via F.run_attack_cell / F.eligible_graphs, same per-fold global seed), with ONE
change that resolves the make-or-break FAIRLY:

    the AGNNCert voting classifier wraps the SUBGRAPH-RETRAINED base GCN
    (train_agnncert_victims.py; AGNNCert Sec.4.1 recipe) instead of the mean-trained GCN.

The 'none' (undefended) victim is UNCHANGED — the attacker without a defense still faces the
normal mean-trained GCN (meta_pg_seed{S}.json). So ASR(none) reproduces the prior reference and
ONLY ASR(agnncert) + clean_acc(agnncert) change. This isolates the fairness fix to the defense:
clean acc through AGNNCert should now recover toward the paper's 0.75 (Table 2) instead of the
degenerate 0.5625, and the eligible-graph denominator counts only graphs the FAIR AGNNCert
classifies correctly pre-attack.

Loads:
  * meta_pg_seed{S}.json          -> test_idx + mean-trained 'none' base  (per fold)
  * meta_agnncertT{T}_seed{S}.json-> subgraph-retrained FAIR base         (per fold)

No attack/model/existing-defense file modified. AGNNCert is deterministic (MD5), so GPU is exact.

Usage: micromamba run -n graph_adversarial python3 eval_pg_floor_agnncert_fair.py \
         [--seeds 42,7,1027] [--cap 10] [--T 300] [--device cuda] [--smoke] [--max-folds N] \
         [--attacks OM2C-nb10,FM-nb10-full,...]
"""
import os, sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_GPU = os.environ.get("FLOOR_GPU", "0")
# Only default the GPU mask if the var is UNSET — respect an explicit CUDA_VISIBLE_DEVICES=""
# (CPU). torch.cuda.is_available() segfaults at import on this host, so infer from the env
# instead of probing (see memory feedback_cpu_autograd_segfault / feedback_experiment_gpu_errorcheck).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
import torch  # noqa: E402
_CUDA_OK = os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1")
import prepare  # noqa: E402
import eval_paper_grade_floor as F  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
from torch_geometric.datasets import TUDataset  # noqa: E402
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs  # noqa: E402
import attack_om2c as O  # noqa: E402
from defense_agnncert import AGNNCertModel  # noqa: E402

DEVICE = "cuda"
DEFENSES = ["none", "agnncert"]

ATTACKS = {
    "OM2C-nb10":           ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "OM2C-nb1":            ("om2c", {"node_budget": 1,  "feat_scale": 1.0}),
    "FM-nb10-full":        ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-lownorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "SGDI-nb10-full":      ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
}
CANDIDATE = "OM2C-nb10"
FULL_EDGE = ["FM-nb10-full", "SGDI-nb10-full"]
_T = 300


def load_fair_meta(seed, T):
    p = MODELS_DIR / f"meta_agnncertT{T}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing FAIR (retrained) AGNNCert victim meta: {p}. "
                                f"Run train_agnncert_victims.py --T {T} first.")
    return json.loads(p.read_text())


def build_victim(none_base, fair_base, dn):
    if dn == "none":
        return none_base
    if dn == "agnncert":
        return AGNNCertModel(fair_base, T=_T)
    raise ValueError(dn)


def main():
    global _T, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--attacks", type=str, default=None,
                    help="comma list subset of ATTACKS (default all 5)")
    args = ap.parse_args()
    _T = args.T; DEVICE = args.device
    if DEVICE.startswith("cuda") and not _CUDA_OK:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available()==False "
                           "(NO silent CPU fallback). Check CUDA_VISIBLE_DEVICES.")
    F.DEVICE = DEVICE; prepare.DEVICE = DEVICE

    attacks = ATTACKS if not args.attacks else {a: ATTACKS[a] for a in args.attacks.split(",")}
    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = args.max_folds
    if args.smoke:
        seeds = seeds[:1]
        if fold_limit is None:
            fold_limit = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in attacks}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in attacks}
    asr_by_seed = {a: {d: {s: [] for s in seeds} for d in DEFENSES} for a in attacks}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; units = []; errored = []
    meta0 = F.load_meta_for_seed(seeds[0])
    fair0 = load_fair_meta(seeds[0], _T)

    print(f"=== FAIR AGNNCert-E FLOOR (retrained base) | {meta0['dataset']} "
          f"{meta0.get('model_type','GCN')} | seeds={seeds} | cap={cap}/fold | T={_T} | "
          f"defenses={DEFENSES} | attacks={list(attacks)} | device={DEVICE} | "
          f"fair_clean_mean(meta)={fair0.get('agnncert_voting_clean_test_acc_mean')} | "
          f"{'SMOKE' if args.smoke else 'FULL'} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = F.load_meta_for_seed(seed)
        fair = load_fair_meta(seed, _T)
        fair_by_fold = {f["fold_idx"]: f for f in fair["folds"]}
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            if fi not in fair_by_fold:
                raise RuntimeError(f"seed{seed} fold{fi}: no FAIR base in {fair['folds']}")
            fold_seed = (int(seed) * 100003 + int(fi)) % (2**31 - 1)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)
            none_base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            none_base.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            none_base.eval()
            fair_base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            fair_base.load_state_dict(torch.load(fair_by_fold[fi]["model_path"], map_location=DEVICE))
            fair_base.eval()

            vacc = F.clean_test_acc(none_base, ds, fold_info["test_idx"])
            victim_accs.append(vacc)
            # test graphs are selected as base-correct (undefended), same as the prior eval;
            # the FIX1 eligible_graphs gating inside run_attack_cell re-filters per defense.
            graphs = collect_test_graphs(none_base, ds, fold_info["test_idx"])[:cap]
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = build_victim(none_base, fair_base, dn)
                ca = F.clean_test_acc(v, ds, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in attacks.items():
                for dn in DEFENSES:
                    victim = build_victim(none_base, fair_base, dn)
                    try:
                        r = F.run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                        asr = r["asr"]; q = r["q"]; nelig = r.get("n_eligible")
                    except Exception as e:
                        tb = traceback.format_exc()
                        errored.append({"attack": akey, "defense": dn, "seed": seed, "fold": fi,
                                        "error": f"{type(e).__name__}: {e}", "tb": tb.splitlines()[-6:]})
                        print(f"  !! ERROR {akey}|{dn} seed{seed} fold{fi}: {e}", flush=True)
                        print(tb, flush=True); asr = None; q = 0; nelig = None
                    if asr is not None:
                        cell_asr[akey][dn].append(asr); asr_by_seed[akey][dn][seed].append(asr)
                    cell_q[akey][dn] += q
                    unit["cells"][f"{akey}|{dn}"] = {
                        "asr": (None if asr is None else round(asr, 4)), "q": q, "n_eligible": nelig}
            units.append(unit)
            print(f"  seed{seed} fold{fi} vacc={vacc:.3f} agc_clean={cacc_def['agnncert']:.3f} "
                  f"n={len(graphs)}  " +
                  "  ".join(f"{a}:none="
                            f"{(unit['cells'][f'{a}|none']['asr'] if unit['cells'][f'{a}|none']['asr'] is not None else float('nan'))}"
                            f"/agc="
                            f"{(unit['cells'][f'{a}|agnncert']['asr'] if unit['cells'][f'{a}|agnncert']['asr'] is not None else float('nan'))}"
                            for a in attacks), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

    def seed_level(a, d):
        per = {str(s): (round(mean(asr_by_seed[a][d][s]) * 100, 2) if asr_by_seed[a][d][s] else None)
               for s in seeds}
        vals = [v for v in per.values() if v is not None]
        return {"per_seed_pct": per, "mean_pct": round(mean(vals), 2) if vals else None,
                "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}

    asr_summary = {a: {d: (agg(cell_asr[a][d]) if cell_asr[a][d] else {"mean_pct": None, "n": 0})
                       for d in DEFENSES} for a in attacks}
    seed_summary = {a: {d: seed_level(a, d) for d in DEFENSES} for a in attacks}
    neutralization = {a: (round(seed_summary[a]["none"]["mean_pct"] - seed_summary[a]["agnncert"]["mean_pct"], 2)
                          if (seed_summary[a]["none"]["mean_pct"] is not None
                              and seed_summary[a]["agnncert"]["mean_pct"] is not None) else None)
                      for a in attacks}

    out = {
        "stamp": stamp,
        "experiment": "DECISIVE FAIR AGNNCert-E floor: AGNNCert voting wraps the SUBGRAPH-"
                      "RETRAINED base GCN (AGNNCert Sec.4.1, p.9 recipe) so clean acc recovers "
                      "toward paper Table 2 = 0.75 (vs degenerate 0.5625 of the mean-trained "
                      "wrapper). 'none' = mean-trained GCN (unchanged). FIX1 defended-correct "
                      "denominator (eligible = graphs the FAIR AGNNCert classifies correctly).",
        "fair_retrain_recipe": "train base GCN on training graphs + their MD5 edge-divided "
                               "subgraphs (same labels); stochastic per-epoch augmentation "
                               "(original + 1 uniform-random subgraph/graph). See "
                               "train_agnncert_victims.py + meta_agnncertT{T}_seed{S}.json.",
        "candidate": CANDIDATE, "dataset": meta0["dataset"],
        "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "T_subgraphs": _T,
        "n_units": len(units), "defenses": DEFENSES, "smoke": bool(args.smoke),
        "max_folds": fold_limit, "attacks_run": list(attacks),
        "asr_metric": "FIX1 defended-correct denominator (eligible = graphs the FAIR AGNNCert "
                      "classifies correctly pre-attack).",
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in attacks.items()},
        "victim_acc_mean": round(mean(victim_accs), 4) if victim_accs else None,
        "victim_acc_std": round(pstdev(victim_accs), 4) if len(victim_accs) > 1 else 0.0,
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4) if len(clean_acc_def[d]) > 1 else 0.0}
                                 for d in DEFENSES if clean_acc_def[d]},
        "fair_agnncert_clean_acc_meta_mean": fair0.get("agnncert_voting_clean_test_acc_mean"),
        "paper_clean_acc_PROTEINS_GCN_retrained": {"undefended": 0.75, "AGNNCert_E": 0.75,
                                                   "note": "Table 2; retrained on subgraphs"},
        "asr_summary_fold_level": asr_summary,
        "asr_summary_seed_level": seed_summary,
        "agnncert_neutralization_pct": neutralization,
        "queries_by_attack_defense": {a: {d: cell_q[a][d] for d in DEFENSES} for a in attacks},
        "errored": errored, "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_path = out_dir / f"floor_agnncert_FAIR_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 110, flush=True)
    cad = out["clean_acc_by_defense"]
    print(f"victim_acc(none) mean={out['victim_acc_mean']}  clean thru FAIR AGNNCert="
          f"{cad.get('agnncert', {}).get('mean')} (undef {cad.get('none', {}).get('mean')}; "
          f"paper retrained 0.75; degenerate-wrapper was 0.5625)", flush=True)
    print(f"\n{'attack':<22} {'ASR none':>12} {'ASR fairAGC':>14} {'neutralized':>12}", flush=True)
    for a in attacks:
        n = seed_summary[a]["none"]; g = seed_summary[a]["agnncert"]
        ns = f"{n['mean_pct']:.1f}±{n['std_pct']:.1f}" if n['mean_pct'] is not None else "ERR"
        gs = f"{g['mean_pct']:.1f}±{g['std_pct']:.1f}" if g['mean_pct'] is not None else "ERR"
        nu = f"{neutralization[a]:.1f}" if neutralization[a] is not None else "-"
        print(f"{a:<22} {ns:>12} {gs:>14} {nu:>12}", flush=True)
    if CANDIDATE in seed_summary:
        om = seed_summary[CANDIDATE]["agnncert"]["mean_pct"]
        print(f"\n--- VERDICT INPUT: {CANDIDATE} ASR through FAIR AGNNCert = {om}% "
              f"(clean acc {cad.get('agnncert', {}).get('mean')}) ---", flush=True)
    print(f"\nerrored={len(errored)}  wall={out['wall_s']}s  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
