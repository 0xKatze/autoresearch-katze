"""Fold loading for the CDMA-style graph attack (cdma_graph.py) and its benchmarks.

Per fold returns the frozen victim, its correctly-classified test graphs (as the rest of
branch-diffusion does), and the fold's TRAIN graphs (graph-disjoint from test) that the
attacker uses offline. Splits come from the same get_folds() the victims were trained with,
and the test_idx recorded in the victim meta is asserted equal to the rederived one.
"""
import json
from copy import deepcopy

import torch
from torch_geometric.datasets import TUDataset

import prepare
import prepare_cont
from prepare import GCN_GraphClassification

MODELS_DIR = prepare.MODELS_DIR
CONT_META = "/workspace/autoresearch-katze-n23/research/branch-diffusion/models_cont/meta_cont_ENZYMES.json"


def _prep(g):
    g = deepcopy(g).cpu()
    g.x = g.x.float()
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    return g


def _load_victim(meta, path):
    m = GCN_GraphClassification(meta["num_features"], meta["num_classes"])
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def load_folds(ds_name):
    if ds_name == "PROTEINS":
        meta = json.load(open(MODELS_DIR / "meta.json"))
        ds = TUDataset(root=str(prepare.DATA_DIR / "PROTEINS"), name="PROTEINS")
        graphs = [_prep(ds[i]) for i in range(len(ds))]
        splits = {f: (tr, te) for f, tr, _va, te in prepare.get_folds(ds)}
    elif ds_name == "ENZYMES":
        meta = json.load(open(CONT_META))
        mean = torch.tensor(meta["feat_mean"]).view(1, -1)
        std = torch.tensor(meta["feat_std"]).view(1, -1)
        gl, labels, _, _, _ = prepare_cont.load_normalized_graphs("ENZYMES", mean, std)
        graphs = [_prep(g) for g in gl]
        splits = {f: (tr, te) for f, tr, _va, te in prepare_cont.get_folds(labels)}
    else:
        raise ValueError(ds_name)

    out = []
    for fi in meta["folds"]:
        f = fi["fold_idx"]
        train_idx, test_idx = splits[f]
        assert sorted(map(int, test_idx)) == sorted(map(int, fi["test_idx"])), f"split mismatch fold {f}"
        victim = _load_victim(meta, fi["model_path"])
        test = []
        for i in test_idx:
            g = graphs[i]
            with torch.no_grad():
                if victim(g).argmax(-1).item() == g.y.item():
                    test.append(g)
        out.append({"fold": f, "victim": victim, "test_graphs": test,
                    "train_graphs": [graphs[i] for i in train_idx],
                    "n_feat": meta["num_features"], "n_classes": meta["num_classes"]})
    return out
