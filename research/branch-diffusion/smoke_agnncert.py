#!/usr/bin/env python3
"""smoke_agnncert.py — validate the AGNNCert-E wrapper and pick the operating T.

Loads the seed-42 PROTEINS/GCN victim folds and measures CLEAN graph accuracy through
AGNNCertModel at several partition counts T, vs the undefended victim (target ~0.724;
paper Table 2 PROTEINS/GCN AGNNCert-E clean = 0.75 WITH subgraph-retraining). No attacks —
this just checks the wrapper is correct, non-degenerate, and finds the T that keeps clean
accuracy in a sane range for the NON-retrained victim. GPU.
"""
import os, sys, json, time, argparse
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_GPU = os.environ.get("FLOOR_GPU", "0")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
import torch  # noqa: E402
_CUDA_OK = torch.cuda.is_available()
if _CUDA_OK:
    _ = (torch.zeros(1, device="cuda") + 1).item()
import prepare  # noqa: E402
import eval_paper_grade_floor as F  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU
from torch_geometric.datasets import TUDataset  # noqa: E402
from prepare import GCN_GraphClassification, DATA_DIR  # noqa: E402
from defense_agnncert import AGNNCertModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=2)
    ap.add_argument("--Ts", type=str, default="300,100,50,30,15")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    device = args.device
    if device.startswith("cuda") and not _CUDA_OK:
        raise RuntimeError("cuda requested but unavailable (no silent CPU fallback)")
    F.DEVICE = device; prepare.DEVICE = device
    Ts = [int(t) for t in args.Ts.split(",")]

    meta = F.load_meta_for_seed(args.seed)
    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    folds = meta["folds"][:args.max_folds]
    print(f"=== SMOKE AGNNCert clean-acc | {meta['dataset']} {meta.get('model_type','GCN')} "
          f"| seed={args.seed} folds={[f['fold_idx'] for f in folds]} | Ts={Ts} | "
          f"device={device} ===", flush=True)

    base_accs = []
    agc = {T: [] for T in Ts}
    t0 = time.time()
    for fi in folds:
        model = GCN_GraphClassification(meta["num_features"], meta["num_classes"]).to(device)
        model.load_state_dict(torch.load(fi["model_path"], map_location=device))
        model.eval()
        b = F.clean_test_acc(model, ds, fi["test_idx"]); base_accs.append(b)
        line = [f"fold{fi['fold_idx']}: undef={b:.3f}"]
        for T in Ts:
            tt = time.time()
            v = AGNNCertModel(model, T=T)
            ca = F.clean_test_acc(v, ds, fi["test_idx"]); agc[T].append(ca)
            line.append(f"T={T}:{ca:.3f}({time.time()-tt:.1f}s)")
        print("  " + "  ".join(line), flush=True)

    print("=" * 80, flush=True)
    bm = mean(base_accs)
    print(f"undefended clean acc      : {bm:.4f}  (target ~0.724; paper retrained=0.75)", flush=True)
    for T in Ts:
        m = mean(agc[T])
        print(f"AGNNCert-E T={T:<4} clean   : {m:.4f}   (vs undef {bm:.3f}, "
              f"majority-class ~0.595)", flush=True)
    print(f"\nwall={time.time()-t0:.1f}s", flush=True)

    out = {"dataset": meta["dataset"], "victim_model": meta.get("model_type", "GCN"),
           "seed": args.seed, "folds": [f["fold_idx"] for f in folds], "device": device,
           "undefended_clean_acc": round(bm, 4),
           "agnncert_clean_acc_by_T": {str(T): round(mean(agc[T]), 4) for T in Ts},
           "paper_target_clean_acc_PROTEINS_retrained": 0.75}
    p = HERE / "results" / f"smoke_agnncert_cleanacc_{meta['dataset']}_seed{args.seed}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"Saved {p}", flush=True)


if __name__ == "__main__":
    main()
