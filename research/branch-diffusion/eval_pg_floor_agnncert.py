#!/usr/bin/env python3
"""eval_pg_floor_agnncert.py — SEPARATE add-on: AGNNCert as the 6th floor-battery defense.

Mirrors gap1_attack_median.py / eval_pg_floor_battery6.py but evaluates ONE new defense —
AGNNCert-E (Li & Wang, USENIX Security 2025), the only PRIOR certified defense covering
GRAPH-classification node injection (deterministic subgraph-division + majority vote). This
driver DOES NOT touch the canonical battery run (another agent owns eval_pg_floor_battery6);
it produces an AGNNCert ASR column per attack for merging into the final worst-case floor:

    floor6 = min over {normfilter, mahalanobis, degfilter, bismooth, rspool, AGNNCert}.

For every (attack, fold) it runs the attack against BOTH the undefended victim ('none') and
the AGNNCert-wrapped victim, so the table shows neutralization = ASR(none) - ASR(AGNNCert)
directly. Reuses F.run_attack_cell / O.om2c_prior / build harness VERBATIM (FIX1 defended-
correct denominator, train-fold C3 refs). No attack/model/existing-defense code modified.

AGNNCert is DETERMINISTIC (MD5 hash division, no RNG) so — unlike bi-smoothing — the GPU
floor is exact (no CPU/GPU RNG-stream divergence). The per-fold global seed is still pinned
so the underlying stochastic attacks (FM/SGDI/OM2C) are reproducible.

Usage: micromamba run -n graph_adversarial python eval_pg_floor_agnncert.py \
         [--seeds 42,7,1027] [--cap 10] [--T 300] [--device cuda] [--smoke] [--max-folds N]
"""
import os, sys, json, time, argparse, traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- GPU-enable hack (verbatim from eval_pg_floor_battery6.py) ------------------------
_GPU = os.environ.get("FLOOR_GPU", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
import torch  # noqa: E402
_CUDA_OK = torch.cuda.is_available()
if _CUDA_OK:
    _ = (torch.zeros(1, device="cuda") + 1).item()
import prepare  # noqa: E402
import eval_paper_grade_floor as F  # noqa: E402  (re-blanks CUDA_VISIBLE_DEVICES at import)
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
from torch_geometric.datasets import TUDataset  # noqa: E402
from prepare import GCN_GraphClassification, DATA_DIR, collect_test_graphs  # noqa: E402
import attack_om2c as O  # noqa: E402
from defense_agnncert import AGNNCertModel  # noqa: E402

DEVICE = "cuda"
DEFENSES = ["none", "agnncert"]

# Same five attacks as the canonical battery (eval_pg_floor_battery6.ATTACKS).
ATTACKS = {
    "OM2C-nb10":           ("om2c", {"node_budget": 10, "feat_scale": 1.0}),
    "OM2C-nb1":            ("om2c", {"node_budget": 1,  "feat_scale": 1.0}),
    "FM-nb10-full":        ("flow", {"edge_strategy": "full", "node_budget": 10, "feat_scale": "auto_x2"}),
    "FM-nb10-eb1-lownorm": ("flow", {"edge_strategy": "topk", "node_budget": 10, "feat_scale": 1.0}),
    "SGDI-nb10-full":      ("sgdi", {"feat_mode": "sgdi", "node_budget": 10, "feat_scale": "auto_x2"}),
}
CANDIDATE = "OM2C-nb10"
FULL_EDGE = ["FM-nb10-full", "SGDI-nb10-full"]   # attacks whose neutralization-by-partition is the sanity check

_T = 300


def build_victim(model, dn):
    if dn == "none":
        return model
    if dn == "agnncert":
        return AGNNCertModel(model, T=_T)
    raise ValueError(dn)


def main():
    global _T, DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,1027")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--T", type=int, default=300, help="AGNNCert partition count (paper graph-class default=300)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    _T = args.T
    DEVICE = args.device
    if DEVICE.startswith("cuda") and not _CUDA_OK:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available()==False "
                           "(NO silent CPU fallback). Check CUDA_VISIBLE_DEVICES.")
    F.DEVICE = DEVICE; prepare.DEVICE = DEVICE

    seeds = [int(s) for s in args.seeds.split(",")]
    cap = args.cap; fold_limit = args.max_folds
    if args.smoke:
        seeds = seeds[:1]
        if fold_limit is None:
            fold_limit = 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cell_asr = {a: {d: [] for d in DEFENSES} for a in ATTACKS}
    cell_q = {a: {d: 0 for d in DEFENSES} for a in ATTACKS}
    asr_by_seed = {a: {d: {s: [] for s in seeds} for d in DEFENSES} for a in ATTACKS}
    clean_acc_def = {d: [] for d in DEFENSES}
    victim_accs = []; victim_acc_by_seed = {s: [] for s in seeds}
    units = []; errored = []
    meta0 = F.load_meta_for_seed(seeds[0])

    print(f"=== ADD-ON FLOOR: AGNNCert-E (USENIX'25) | {meta0['dataset']} "
          f"{meta0.get('model_type','GCN')} | seeds={seeds} | cap={cap}/fold | T={_T} | "
          f"defenses={DEFENSES} | attacks={list(ATTACKS)} | device={DEVICE} | "
          f"{'SMOKE' if args.smoke else 'FULL'} | {stamp} ===", flush=True)
    t0 = time.time()

    for seed in seeds:
        meta = F.load_meta_for_seed(seed)
        ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
        folds = meta["folds"] if fold_limit is None else meta["folds"][:fold_limit]
        for fold_info in folds:
            fi = fold_info["fold_idx"]
            fold_seed = (int(seed) * 100003 + int(fi)) % (2**31 - 1)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)
            model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
            model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
            model.eval()
            vacc = F.clean_test_acc(model, ds, fold_info["test_idx"])
            victim_accs.append(vacc); victim_acc_by_seed[seed].append(vacc)
            graphs = collect_test_graphs(model, ds, fold_info["test_idx"])[:cap]
            om2c_prior = O.om2c_prior(graphs, DEVICE, O.CONFIG["tangent_dim"],
                                      O.CONFIG["sigma_mult"], eps=O.CONFIG["prior_eps"])
            cacc_def = {}
            for dn in DEFENSES:
                v = build_victim(model, dn)
                ca = F.clean_test_acc(v, ds, fold_info["test_idx"])
                clean_acc_def[dn].append(ca); cacc_def[dn] = round(ca, 4)
            unit = {"seed": seed, "fold": fi, "victim_acc": round(vacc, 4),
                    "n_graphs": len(graphs), "clean_acc_defended": cacc_def, "cells": {}}
            for akey, (rk, ov) in ATTACKS.items():
                for dn in DEFENSES:
                    victim = build_victim(model, dn)
                    try:
                        r = F.run_attack_cell(rk, ov, victim, graphs, om2c_prior)
                        asr = r["asr"]; q = r["q"]; nelig = r.get("n_eligible")
                    except Exception as e:
                        tb = traceback.format_exc()
                        errored.append({"attack": akey, "defense": dn, "seed": seed,
                                        "fold": fi, "error": f"{type(e).__name__}: {e}",
                                        "tb": tb.splitlines()[-6:]})
                        print(f"  !! ERROR {akey}|{dn} seed{seed} fold{fi}: {e}", flush=True)
                        print(tb, flush=True)
                        asr = None; q = 0; nelig = None
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
                            for a in ATTACKS), flush=True)

    def agg(lst):
        return {"mean_pct": round(mean(lst) * 100, 2),
                "std_pct": round(pstdev(lst) * 100, 2) if len(lst) > 1 else 0.0, "n": len(lst)}

    def seed_level(a, d):
        per = {str(s): (round(mean(asr_by_seed[a][d][s]) * 100, 2) if asr_by_seed[a][d][s] else None)
               for s in seeds}
        vals = [v for v in per.values() if v is not None]
        return {"per_seed_pct": per,
                "mean_pct": round(mean(vals), 2) if vals else None,
                "std_pct": round(pstdev(vals), 2) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}

    asr_summary = {a: {d: (agg(cell_asr[a][d]) if cell_asr[a][d] else {"mean_pct": None, "n": 0})
                       for d in DEFENSES} for a in ATTACKS}
    seed_summary = {a: {d: seed_level(a, d) for d in DEFENSES} for a in ATTACKS}
    neutralization = {a: (round(seed_summary[a]["none"]["mean_pct"] - seed_summary[a]["agnncert"]["mean_pct"], 2)
                          if (seed_summary[a]["none"]["mean_pct"] is not None
                              and seed_summary[a]["agnncert"]["mean_pct"] is not None) else None)
                      for a in ATTACKS}

    out = {
        "stamp": stamp,
        "experiment": "ADD-ON FLOOR DEFENSE: AGNNCert-E (Li & Wang, USENIX Security 2025; "
                      "edge-centric MD5 subgraph-division + majority vote, Sec.3.2 / Eq.3,8). "
                      "6th defense for the worst-case floor. SEPARATE from canonical battery6.",
        "defense_impl": "defense_agnncert.AGNNCertModel — AGNNCert-E, T subgraphs via MD5 "
                        "hash of per-graph LOCAL edge endpoints (smaller-first), each subgraph "
                        "keeps all nodes+features with a disjoint edge partition; majority vote "
                        "over T base-GCN predictions. Inference-only wrapper on the existing "
                        "mean-trained GCN (NO subgraph retraining; paper Table 2 retrained "
                        "PROTEINS clean=0.75).",
        "candidate": CANDIDATE, "dataset": meta0["dataset"],
        "victim_model": meta0.get("model_type", "GCN"),
        "device": DEVICE, "seeds": seeds, "cap_per_fold": cap, "T_subgraphs": _T,
        "n_units": len(units), "defenses": DEFENSES,
        "smoke": bool(args.smoke), "max_folds": fold_limit,
        "asr_metric": "FIX1 defended-correct denominator (per (attack,defense) cell; eligible "
                      "= graphs the DEFENDED victim classifies correctly pre-attack).",
        "global_seed_scheme": "torch.manual_seed((seed*100003+fold)%2**31-1)+cuda per fold; "
                              "AGNNCert itself is deterministic (no RNG).",
        "attacks": {a: {"run": rk, "overrides": ov} for a, (rk, ov) in ATTACKS.items()},
        "victim_acc_mean": round(mean(victim_accs), 4),
        "victim_acc_std": round(pstdev(victim_accs), 4) if len(victim_accs) > 1 else 0.0,
        "victim_acc_by_seed": {str(s): {"mean": round(mean(victim_acc_by_seed[s]), 4),
                                        "n_folds": len(victim_acc_by_seed[s])}
                               for s in seeds if victim_acc_by_seed[s]},
        "clean_acc_by_defense": {d: {"mean": round(mean(clean_acc_def[d]), 4),
                                     "std": round(pstdev(clean_acc_def[d]), 4) if len(clean_acc_def[d]) > 1 else 0.0}
                                 for d in DEFENSES},
        "paper_clean_acc_PROTEINS_GCN_retrained": {"undefended": 0.75, "AGNNCert_E": 0.75,
                                                   "note": "Table 2; retrained on subgraphs"},
        "asr_summary_fold_level": asr_summary,
        "asr_summary_seed_level": seed_summary,
        "agnncert_neutralization_pct": neutralization,
        "full_edge_attacks_sanity": {a: {"undef_ASR_pct": seed_summary[a]["none"]["mean_pct"],
                                         "agnncert_ASR_pct": seed_summary[a]["agnncert"]["mean_pct"],
                                         "neutralized_pct": neutralization[a]} for a in FULL_EDGE},
        "queries_by_attack_defense": {a: {d: cell_q[a][d] for d in DEFENSES} for a in ATTACKS},
        "errored": errored, "wall_s": round(time.time() - t0, 1), "units": units,
    }
    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_path = out_dir / f"floor_addon_agnncert_{meta0['dataset']}_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 110, flush=True)
    print(f"victim_acc mean={out['victim_acc_mean']}  clean thru AGNNCert="
          f"{out['clean_acc_by_defense']['agnncert']['mean']} "
          f"(undef-wrapper {out['clean_acc_by_defense']['none']['mean']}; paper retrained 0.75)",
          flush=True)
    print(f"\n{'attack':<22} {'ASR none':>12} {'ASR AGNNCert':>14} {'neutralized':>12}", flush=True)
    for a in ATTACKS:
        n = seed_summary[a]["none"]; g = seed_summary[a]["agnncert"]
        ns = f"{n['mean_pct']:.1f}±{n['std_pct']:.1f}" if n['mean_pct'] is not None else "ERR"
        gs = f"{g['mean_pct']:.1f}±{g['std_pct']:.1f}" if g['mean_pct'] is not None else "ERR"
        nu = f"{neutralization[a]:.1f}" if neutralization[a] is not None else "-"
        print(f"{a:<22} {ns:>12} {gs:>14} {nu:>12}", flush=True)
    print(f"\n--- {CANDIDATE} vs AGNNCert (does OM2C still hold?) ---", flush=True)
    om = seed_summary[CANDIDATE]["agnncert"]["mean_pct"]
    print(f"  OM2C-nb10 ASR through AGNNCert : {om}%", flush=True)
    for a in FULL_EDGE + ["FM-nb10-eb1-lownorm"]:
        print(f"  {a:<22} through AGNNCert : {seed_summary[a]['agnncert']['mean_pct']}%", flush=True)
    print(f"\n--- full-edge neutralization sanity ---", flush=True)
    for a in FULL_EDGE:
        print(f"  {a:<16}: ASR {seed_summary[a]['none']['mean_pct']}% -> "
              f"{seed_summary[a]['agnncert']['mean_pct']}% (neutralized {neutralization[a]} pts)", flush=True)
    print(f"\nerrored={len(errored)}  wall={out['wall_s']}s  Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
