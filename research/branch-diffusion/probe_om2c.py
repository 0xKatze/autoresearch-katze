#!/usr/bin/env python3
"""probe_om2c.py — answer coordinator Q2: does the MULTI-NODE orthogonal mass channel
(OM2C nb=10, feat_scale=1.0) recover a high worst-case-defense floor? Reports both
the dense full-edge endpoint (asr_full, rho=1.0) and the SPARSE greedy regime
(asr_greedy = greedy-flip-through-defense rate, the real degree-1-ish mass channel),
plus injected-node degree info. fold0 seed42 PROTEINS GCN."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, importlib.util, json, math
from pathlib import Path
import torch
from torch_geometric.datasets import TUDataset
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import prepare; prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs
def L(n,p):
    s=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
sgdi=L('branch_sgdi_attack', HERE/'attack.py')
import attack_om2c
from defenses import (DefendedModel, mahalanobis_defense, degree_filter_defense, node_degrees, norm_filter_defense)
DEVICE="cpu"; CAP=8
def build(model, defense, ref):
    if defense=="none": return model
    if defense=="normfilter":
        return DefendedModel(model, norm_filter_defense,"normfilter",percentile=95)
    if defense=="mahalanobis":
        X=torch.cat([g.x for g in ref]).float(); mean_=X.mean(0); d=X.size(1)
        cov=torch.cov(X.t())+1e-3*torch.eye(d); ci=torch.linalg.inv(cov)
        diff=X-mean_; sqm=(diff@ci*diff).sum(1); thr=float(torch.quantile(sqm,0.995))
        return DefendedModel(model, mahalanobis_defense,"mahalanobis",mean=mean_,cov_inv=ci,threshold=thr)
    if defense=="degfilter":
        degs=torch.cat([node_degrees(g) for g in ref]).float()
        return DefendedModel(model, degree_filter_defense,"degfilter",mean=float(degs.mean()),std=float(degs.std()),k=3.0)
meta=json.loads((MODELS_DIR/"meta_pg_seed42.json").read_text())
ds=TUDataset(root=str(DATA_DIR/meta["dataset"]),name=meta["dataset"])
fold=meta["folds"][0]
model=GCN_GraphClassification(meta["num_features"],meta["num_classes"]).to(DEVICE)
model.load_state_dict(torch.load(fold["model_path"],map_location=DEVICE)); model.eval()
graphs=collect_test_graphs(model,ds,fold["test_idx"])[:CAP]
# clean degree stats (degfilter threshold)
alldeg=torch.cat([node_degrees(g) for g in graphs]).float()
print(f"PROTEINS GCN seed42 fold0 n={len(graphs)}; clean node degree mean={alldeg.mean():.2f} std={alldeg.std():.2f} "
      f"=> degfilter(k=3) threshold≈{alldeg.mean()+3*alldeg.std():.2f}", flush=True)
for nb in [1, 10]:
    print(f"\n=== OM2C node_budget={nb}, feat_scale=1.0 ===", flush=True)
    print(f"{'defense':<12} {'asr_full':>9} {'asr_greedy':>11} {'med_greedy_edges':>17} {'med_rho':>8} {'time':>6}", flush=True)
    for dn in ["none","normfilter","mahalanobis","degfilter"]:
        v=build(model,dn,graphs)
        cfg=dict(attack_om2c.CONFIG); cfg["node_budget"]=nb; cfg["feat_scale"]=1.0
        t0=time.time(); rec=attack_om2c.run_attack_sweep(v,graphs,DEVICE,cfg); dt=time.time()-t0
        ge=[g for g in rec["greedy_edges"]]
        flips=[g for g in ge if g is not None]
        asr_greedy=len(flips)/max(len(ge),1)
        med_ge = (sorted(flips)[len(flips)//2] if flips else None)
        rhos=[r for r in rec["rho_to_flip"] if r!=float('inf') and not math.isinf(r)]
        med_rho=(round(sorted(rhos)[len(rhos)//2],3) if rhos else None)
        print(f"{dn:<12} {rec['asr_full']*100:8.1f}% {asr_greedy*100:10.1f}% "
              f"{str(med_ge):>17} {str(med_rho):>8} {dt:5.1f}s  in_ball={rec['stealth_in_ball_frac']:.2f}", flush=True)
