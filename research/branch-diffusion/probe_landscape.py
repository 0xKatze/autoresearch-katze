#!/usr/bin/env python3
"""probe_landscape.py — decompose the two stealth axes (edge_budget x feat_scale)
to locate where FM-A ASR survives, on fold0 seed42 PROTEINS GCN, undefended + degfilter
+ mahalanobis. Honest diagnostic for the H1 floor claim."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, importlib.util, json
from pathlib import Path
import torch
from torch_geometric.datasets import TUDataset
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import prepare; prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
sgdi_attack=_load("branch_sgdi_attack", HERE/"attack.py")
import attack_flow
from defenses import (DefendedModel, mahalanobis_defense, degree_filter_defense, node_degrees)
DEVICE="cpu"; CAP=8
def build(model, defense, ref):
    if defense=="none": return model
    if defense=="mahalanobis":
        X=torch.cat([g.x for g in ref]).float(); mean_=X.mean(0); d=X.size(1)
        cov=torch.cov(X.t())+1e-3*torch.eye(d); ci=torch.linalg.inv(cov)
        diff=X-mean_; sqm=(diff@ci*diff).sum(1); thr=float(torch.quantile(sqm,0.995))
        return DefendedModel(model, mahalanobis_defense,"mahalanobis",mean=mean_,cov_inv=ci,threshold=thr)
    if defense=="degfilter":
        degs=torch.cat([node_degrees(g) for g in ref]).float()
        return DefendedModel(model, degree_filter_defense,"degfilter",mean=float(degs.mean()),std=float(degs.std()),k=3.0)
def runc(victim, graphs, ov):
    saved={k:attack_flow.CONFIG.get(k) for k in ov}; attack_flow.CONFIG.update(ov)
    try: t0=time.time(); a=attack_flow.run_attack(victim,graphs,DEVICE); return a,time.time()-t0
    finally: attack_flow.CONFIG.update(saved)
meta=json.loads((MODELS_DIR/"meta.json").read_text())
ds=TUDataset(root=str(DATA_DIR/meta["dataset"]),name=meta["dataset"])
fold=meta["folds"][0]
model=GCN_GraphClassification(meta["num_features"],meta["num_classes"]).to(DEVICE)
model.load_state_dict(torch.load(fold["model_path"],map_location=DEVICE)); model.eval()
graphs=collect_test_graphs(model,ds,fold["test_idx"])[:CAP]
print(f"landscape: PROTEINS GCN fold0 seed42 n={len(graphs)}",flush=True)
# axes: edge in {full, eb1(topk nb=1)} ; fs in {1.0, auto_x2, auto_x3}
EDGES={"full":{"edge_strategy":"full"},"eb1":{"edge_strategy":"topk"}}
FS=[1.0,"auto_x2","auto_x3"]
print(f"{'edge':<5} {'fs':<9} {'none':>7} {'maha':>7} {'degf':>7}",flush=True)
for ek,eov in EDGES.items():
    for fs in FS:
        ov=dict(eov); ov["node_budget"]=1; ov["feat_scale"]=fs
        row=[]
        for dn in ["none","mahalanobis","degfilter"]:
            v=build(model,dn,graphs); a,dt=runc(v,graphs,ov); row.append(a*100)
        print(f"{ek:<5} {str(fs):<9} {row[0]:7.1f} {row[1]:7.1f} {row[2]:7.1f}",flush=True)
