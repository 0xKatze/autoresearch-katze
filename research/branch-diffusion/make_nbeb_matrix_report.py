#!/usr/bin/env python3
"""Consolidation + figures for the nb/eb x method x defense stealth matrix (PROTEINS/GCN).
READ-ONLY over result JSONs. No attack/model code touched. Every cell cites its source JSON.
"""
import json, glob, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"
CH = RES / "charts"; CH.mkdir(parents=True, exist_ok=True)
TS = time.strftime("%Y%m%d_%H%M%S")

def load(f):
    p = RES / f if not str(f).startswith("/") else Path(f)
    return json.load(open(p)), str(p)

def latest(pat):
    fs = sorted(glob.glob(str(RES / pat))); return fs[-1] if fs else None

SRC = {
 "battery6": "floor_battery6_rspool_bismooth_PROTEINS_full_20260622_220259.json",
 "battery5_gnnsafe": "floor_battery5_gpu_PROTEINS_full_20260618_025525.json",
 "om2c_fm": "floor_om2c_fm_PROTEINS_full_20260623_080218.json",
 "agnncert": "floor_agnncert_FAIR_PROTEINS_full_20260623_184603.json",
 "nodecount": "nodecount_2attack_PROTEINS_full_20260618_085159.json",
 "edgebudget": "edgebudget_sweep_PROTEINS_full_20260623_195546.json",
}
DEF_ORDER = ["none","degfilter","normfilter","mahalanobis","bismooth","gnnsafe","rspool","agnncert"]
C = {"stamp":TS,"dataset":"PROTEINS","victim":"GCN(3L)","sources":{},"matrix":{},
     "eb_curve":{},"nb_curve":{},"caveats":[
     "binary-feature: mahalanobis/normfilter have ~no distributional discrimination on "
     "PROTEINS (mean node norm=1.0); degfilter (structure) is the meaningful stealth axis.",
     "auto_x2 confound: KATZE uses feat_scale=auto_x2 and saturates at eb=1 (~1.0); "
     "N4/SGDI/FM use fs=1.0. Do NOT compare KATZE eb-curve SHAPE vs fs=1.0 methods.",
     "cross-driver pairing: victims paired WITHIN each driver, not identical ACROSS drivers; "
     "seeds=42/7/1027 (agnncert & gnnsafe: seed 42 only; katze_vs_sgdi: seeds 0-4)."]}
matrix = {}

def add_rows(key):
    d, path = load(SRC[key])
    C["sources"][key] = {"path":path,"seeds":d.get("seeds"),"victim_acc_mean":d.get("victim_acc_mean")}
    for atk, defs in d.get("asr_summary",{}).items():
        row = matrix.setdefault(f"{atk} [{key}]", {})
        for dn, cell in defs.items():
            if isinstance(cell,dict) and cell.get("mean_pct") is not None:
                row[dn] = [cell["mean_pct"], cell.get("std_pct")]

for key in ("battery6","battery5_gnnsafe","om2c_fm"): add_rows(key)
try:
    d, path = load(SRC["agnncert"])
    C["sources"]["agnncert"] = {"path":path,"seeds":d.get("seeds"),"victim_acc_mean":d.get("victim_acc_mean")}
    asl = d.get("asr_summary_seed_level") or d.get("asr_summary_fold_level") or {}
    for atk, defs in asl.items():
        row = matrix.setdefault(f"{atk} [agnncert]", {})
        for dn, cell in defs.items():
            v = cell.get("mean_pct") if isinstance(cell,dict) else cell
            if v is not None: row[dn] = [v, cell.get("std_pct") if isinstance(cell,dict) else None]
except Exception as e:
    C["sources"]["agnncert_error"] = str(e)
C["matrix"] = matrix

eb = {}
kf = latest("katze_vs_sgdi_ebsweep_PROTEINS_*.json")
if kf:
    kd = json.load(open(kf)); C["sources"]["katze_vs_sgdi"] = {"path":kf,"constraints":kd.get("constraints")}
    m = kd.get("mean",{}); ebk = kd.get("eb_keys",["1","3","5","10","full"])
    for meth in ("katze","n4","sgdi"):
        if meth in m:
            eb[f"{meth} (nb=1)"] = {"x":ebk,"y":[m[meth][k]*100 for k in ebk],
                                    "fs":"auto_x2" if meth=="katze" else "1.0"}
ed, edp = load(SRC["edgebudget"]); C["sources"]["edgebudget"]={"path":edp,"seeds":ed.get("seeds"),"nb":ed.get("node_budget_fixed")}
dv1 = ed.get("DV1_undefended_asr",{})
if "FM-dense" in dv1:
    xs = ed.get("edge_budgets")
    eb["FM-dense (nb=10)"] = {"x":[str(x) for x in xs],"y":[dv1["FM-dense"][str(x)]["mean_pct"] for x in xs],"fs":"1.0"}
if "OM2C_ref" in dv1:
    eb["OM2C_ref (nb=10,flat)"] = {"x":["1","3","5","8"],"y":[dv1["OM2C_ref"]["mean_pct"]]*4,"fs":"1.0"}
C["eb_curve"] = eb

nd, ndp = load(SRC["nodecount"]); C["sources"]["nodecount"]={"path":ndp,"seeds":nd.get("seeds")}
nb = {}; nbs = nd.get("node_budgets")
for atk, cur in nd.get("DV1_undefended_asr",{}).items():
    nb[f"{atk} (undef)"] = {"x":[str(x) for x in nbs],"y":[cur[str(x)]["mean_pct"] for x in nbs]}
for atk, cur in nd.get("DV2_worstcase_floor",{}).items():
    nb[f"{atk} (worst-def floor)"] = {"x":[str(x) for x in nbs],"y":[cur[str(x)]["mean_pct"] for x in nbs]}
C["nb_curve"] = nb

plt.figure(figsize=(7,5))
for name,s in eb.items():
    plt.plot(range(len(s["x"])), s["y"], marker="o", label=f"{name} fs={s['fs']}")
    plt.xticks(range(len(s["x"])), s["x"])
plt.xlabel("edge budget (attach edges/node)"); plt.ylabel("undefended ASR (%)")
plt.title("PROTEINS/GCN - ASR vs edge budget (undefended)\nKATZE nb=1 auto_x2 confound; FM/OM2C nb=10 fs=1.0")
plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.ylim(-3,105); plt.tight_layout()
f1 = CH / f"asr_vs_eb_PROTEINS_{TS}.png"; plt.savefig(f1,dpi=130); plt.close()

plt.figure(figsize=(7,5))
for name,s in nb.items():
    ls = "--" if "floor" in name else "-"
    plt.plot([int(x) for x in s["x"]], s["y"], marker="s", linestyle=ls, label=name)
plt.xlabel("node budget (# injected nodes)"); plt.ylabel("ASR (%)")
plt.title("PROTEINS/GCN - ASR vs node budget (eb=full)\nsolid=undefended, dashed=worst-case defense floor")
plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.ylim(-3,105); plt.tight_layout()
f2 = CH / f"asr_vs_nb_PROTEINS_{TS}.png"; plt.savefig(f2,dpi=130); plt.close()
C["figures"] = [str(f1), str(f2)]
outp = RES / f"nbeb_stealth_matrix_PROTEINS_{TS}.json"; outp.write_text(json.dumps(C,indent=2))

def fmt(c): return (f"{c[0]:.0f}±{c[1]:.0f}" if c and c[1] is not None else (f"{c[0]:.0f}" if c else "-"))
lines = ["| Method [src] | "+" | ".join(DEF_ORDER)+" |","|"+"---|"*(len(DEF_ORDER)+1)]
for atk,row in matrix.items():
    lines.append("| "+atk+" | "+" | ".join(fmt(row.get(dn)) for dn in DEF_ORDER)+" |")
md = "\n".join(lines); (RES / f"nbeb_matrix_table_{TS}.md").write_text(md)
print("CONSOLIDATED:",outp); print("FIG1:",f1); print("FIG2:",f2)
print("\n=== method x defense ASR% ===\n"+md)
print("\n=== eb curve ===")
for k,s in eb.items(): print(" ",k,dict(zip(s["x"],[round(y,1) for y in s["y"]])))
print("=== nb curve ===")
for k,s in nb.items(): print(" ",k,dict(zip(s["x"],[round(y,1) for y in s["y"]])))
