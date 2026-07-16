#!/usr/bin/env python3
"""sanity_battery6.py — correctness checks for the two new floor-battery defenses.

Step 1 (RS-Pool): a HIGH-NORM injected node connected to every node flips the mean-pool
readout but must NOT flip the median RS-Pool readout (median ignores a minority outlier).
Reports flip rate for mean-pool vs RS-Pool over clean test graphs.

Step 2 (Bi-smoothing): clean accuracy THROUGH the smoothing defense must stay close to
the undefended clean accuracy (the defense must not wreck clean acc). Reports both.

CPU-only, deterministic, fast (fold0 / seed42 / first 30 graphs).
Usage: micromamba run -n graph_adversarial python sanity_battery6.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, sys
import torch
from types import SimpleNamespace
from torch_geometric.datasets import TUDataset

import prepare
prepare.DEVICE = "cpu"
from prepare import GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs
from defense_battery6 import RSPoolModel, BiSmoothingModel

DEVICE = "cpu"


def main():
    torch.manual_seed(0)
    meta = json.loads((MODELS_DIR / "meta_pg_seed42.json").read_text())
    f0 = meta["folds"][0]
    base = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(DEVICE)
    base.load_state_dict(torch.load(f0["model_path"], map_location=DEVICE))
    base.eval()
    rs = RSPoolModel(base)
    bs = BiSmoothingModel(base, T=20, p=0.2, seed=0)

    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    graphs = collect_test_graphs(base, ds, f0["test_idx"])[:30]

    # mean node-feature norm (for a "high-norm" injection ~ 20x the clean scale)
    all_norms = torch.cat([g.x.norm(p=2, dim=1) for g in graphs])
    mean_norm = float(all_norms.mean())
    inj_dir = torch.cat([g.x for g in graphs], 0).mean(0)
    inj_dir = inj_dir / (inj_dir.norm() + 1e-9)
    inj_feat = (inj_dir * (20.0 * mean_norm)).unsqueeze(0)      # 1 high-norm node

    def pred(model, data):
        d = SimpleNamespace(x=data.x, edge_index=data.edge_index,
                            batch=torch.zeros(data.x.size(0), dtype=torch.long))
        with torch.no_grad():
            return int(model(d).argmax(-1).item())

    def inject(data, degree):
        """Inject ONE high-norm node connected to `degree` randomly-chosen targets
        (degree='all' -> connected to every node)."""
        n = data.x.size(0)
        x2 = torch.cat([data.x, inj_feat], 0)
        new = n
        if degree == "all":
            tgts = list(range(n))
        else:
            k = min(int(degree), n)
            tgts = torch.randperm(n)[:k].tolist()
        extra = torch.tensor([[new] * len(tgts) + tgts,
                              tgts + [new] * len(tgts)], dtype=torch.long)
        ei2 = torch.cat([data.edge_index, extra], 1)
        return SimpleNamespace(x=x2, edge_index=ei2,
                               batch=torch.zeros(x2.size(0), dtype=torch.long))

    # ---- 1a. READOUT-LEVEL test (isolates RS-Pool's actual claim) ----
    # Build clean post-GCN node embeddings H, append ONE giant-norm embedding row, and
    # compare classifier(mean) vs classifier(median). This tests the readout in isolation
    # (NO message-passing contamination), which is exactly what RS-Pool defends.
    b = base
    ro_mean_flips = ro_med_flips = 0
    for g in graphs:
        with torch.no_grad():
            h = g.x
            for i in range(b.num_layers):
                h = torch.relu(b.batch_norms[i](b.conv_layers[i](h, g.edge_index)))
            big_norm = 20.0 * float(h.norm(p=2, dim=1).median())
            d = h.mean(0); d = d / (d.norm() + 1e-9)
            outlier = (d * big_norm).unsqueeze(0)
            h_aug = torch.cat([h, outlier], 0)
            c_clean = int(b.classifier(h.mean(0, keepdim=True)).argmax(-1).item())
            c_mean = int(b.classifier(h_aug.mean(0, keepdim=True)).argmax(-1).item())
            c_med = int(b.classifier(h_aug.median(0, keepdim=True).values).argmax(-1).item())
        ro_mean_flips += int(c_mean != c_clean)
        ro_med_flips += int(c_med != c_clean)
    n = len(graphs)
    print("=" * 70)
    print(f"STEP 1a — READOUT-LEVEL outlier resistance (append 1 giant-norm embedding "
          f"row to the pool; no message passing), N={n}")
    print(f"  mean readout flip rate   : {ro_mean_flips}/{n} = {100*ro_mean_flips/n:.1f}%")
    print(f"  median readout flip rate : {ro_med_flips}/{n} = {100*ro_med_flips/n:.1f}%")
    print(f"  => RS-Pool readout robustness: "
          f"{'PASS' if ro_med_flips < ro_mean_flips else 'FAIL'} (median << mean)")

    # ---- 1b. INPUT-LEVEL injections at degree-1 vs all-connected ----
    for degree in (1, "all"):
        mean_flips = rs_flips = 0
        for g in graphs:
            c_mean = pred(base, g)
            c_rs = pred(rs, g)
            gi = inject(g, degree)
            mean_flips += int(pred(base, gi) != c_mean)
            rs_flips += int(pred(rs, gi) != c_rs)
        print("=" * 70)
        print(f"STEP 1b — INPUT-LEVEL high-norm node (20x mean norm), degree={degree}, N={n}")
        print(f"  mean-pool flip rate : {mean_flips}/{n} = {100*mean_flips/n:.1f}%")
        print(f"  RS-Pool   flip rate : {rs_flips}/{n} = {100*rs_flips/n:.1f}%")
        print(f"  (degree='all' contaminates every embedding via message passing -> "
              f"median cannot help; degree=1 isolates readout dilution)")

    # Step 2 — clean accuracy through bi-smoothing
    c_base = c_rs = c_bs = tot = 0
    for g in graphs:
        y = int(g.y.item())
        tot += 1
        c_base += int(pred(base, g) == y)
        c_rs += int(pred(rs, g) == y)
        c_bs += int(pred(bs, g) == y)
    print("=" * 70)
    print(f"STEP 2 — clean accuracy through defenses (N={tot} graphs, fold0/seed42)")
    print(f"  undefended (mean-pool) clean acc : {100*c_base/tot:.1f}%")
    print(f"  RS-Pool (median)       clean acc : {100*c_rs/tot:.1f}%")
    print(f"  Bi-smoothing T=20 p=0.2 clean acc: {100*c_bs/tot:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
