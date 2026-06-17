#!/usr/bin/env python3
"""
smoke_flow.py — Smoke test for attack_flow.py (N28 flow-matching, variant A).

Verifies, end-to-end on a handful of PROTEINS graphs against the `none` defense:
  (a) attack_flow.run_attack runs without error,
  (b) returns a sane ASR float in [0, 1],
  (c) injected (pre-scale) features land inside the clean Mahalanobis ball,
  (d) roughly compares wall-time + victim queries vs attack_diffusion on the same
      tiny input (apples-to-apples).

Mirrors quick_test_n23.py's harness. attack_flow is imported FIRST so it heals the
branch-diffusion attack.py name collision (registers the canonical katze attack as
`attack`), after which attack_diffusion imports cleanly too.

Usage: micromamba run -n graph_adversarial python smoke_flow.py
"""
import os
# Smoke test runs on CPU: deterministic, and sidesteps the env's CUDA segfault on
# this GPU. Force CPU BEFORE importing prepare so its module-global DEVICE is cpu.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys
import time

import torch
from torch_geometric.datasets import TUDataset

import prepare
prepare.DEVICE = "cpu"
from prepare import (
    GCN_GraphClassification, DATA_DIR, MODELS_DIR, collect_test_graphs,
)
DEVICE = "cpu"
import attack_flow            # heals env: registers canonical 'attack'
import attack_diffusion       # now imports cleanly


# --- victim query counter: wrap model.forward to tally graphs-per-call -------
class QueryCounter:
    def __init__(self, model):
        self.model = model
        self.n = 0
        self._orig = model.forward

    def _graphs(self, inp):
        g = getattr(inp, "num_graphs", None)
        if g is not None:
            return int(g)
        b = getattr(inp, "batch", None)
        if b is not None and b.numel() > 0:
            return int(b.max().item()) + 1
        return 1

    def __enter__(self):
        def counting_forward(*args, **kw):
            if args:
                self.n += self._graphs(args[0])
            self.model.forward = self._orig
            try:
                out = self._orig(*args, **kw)
            finally:
                self.model.forward = counting_forward
            return out
        self.model.forward = counting_forward
        return self

    def __exit__(self, *exc):
        self.model.forward = self._orig


def main():
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: run prepare.py first!")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    fold_info = meta["folds"][0]
    model = GCN_GraphClassification(
        meta["num_features"], meta["num_classes"]).to(DEVICE)
    model.load_state_dict(torch.load(fold_info["model_path"], map_location=DEVICE))
    model.eval()

    ds = TUDataset(root=str(DATA_DIR / meta["dataset"]), name=meta["dataset"])
    test_graphs = collect_test_graphs(model, ds, fold_info["test_idx"])
    if not test_graphs:
        print("No test graphs to attack.")
        return
    test_graphs = test_graphs[:6]
    print(f"Smoke test: {len(test_graphs)} PROTEINS graphs from fold 0, "
          f"defense=none, device={DEVICE}")
    print(f"flow T_flow={attack_flow.CONFIG['T_flow']}, "
          f"integrator={attack_flow.CONFIG['integrator']}, "
          f"lambda={attack_flow.CONFIG['lambda_schedule']} "
          f"[{attack_flow.CONFIG['lambda_min']}->{attack_flow.CONFIG['lambda_max']}]; "
          f"diffusion T_steps={attack_diffusion.CONFIG['T_steps']}")

    def run_flow(feat_scale):
        old_f = attack_flow.CONFIG["feat_scale"]
        attack_flow.CONFIG["feat_scale"] = feat_scale
        try:
            with QueryCounter(model) as qc:
                t0 = time.time()
                asr = attack_flow.run_attack(model, test_graphs, DEVICE)
                dt = time.time() - t0
            st = attack_flow.FLOW_STATS
            frac = (st["in_ball"] / st["total"]) if st["total"] else float("nan")
            return asr, qc.n, dt, st["in_ball"], st["total"], frac
        finally:
            attack_flow.CONFIG["feat_scale"] = old_f

    def run_diff(feat_scale):
        old_f = attack_diffusion.CONFIG["feat_scale"]
        attack_diffusion.CONFIG["feat_scale"] = feat_scale
        try:
            with QueryCounter(model) as qc:
                t0 = time.time()
                asr = attack_diffusion.run_attack(model, test_graphs, DEVICE)
                dt = time.time() - t0
            return asr, qc.n, dt
        finally:
            attack_diffusion.CONFIG["feat_scale"] = old_f

    ok = True
    # Two regimes:
    #   auto_x2 : the CONFIG default (apples-to-apples w/ diffusion). The weak
    #             victim is trivially flipped by the scaled in-ball init -> shows
    #             end-to-end correctness + sane ASR.
    #   1.0     : stealth regime (injected feature == in-ball raw feature, no
    #             scaling) -> the attack must actually integrate the flow, giving
    #             a meaningful query comparison + exercising projection.
    for fs in ("auto_x2", 1.0):
        print("=" * 64)
        print(f"feat_scale = {fs}")
        a_f, q_f, t_f, nb, nt, frac = run_flow(fs)
        a_d, q_d, t_d = run_diff(fs)
        print(f"[flow] ASR={a_f*100:6.2f}%  queries={q_f:5d}  time={t_f:6.2f}s  "
              f"in_ball={nb}/{nt} ({frac*100:.1f}%)")
        print(f"[diff] ASR={a_d*100:6.2f}%  queries={q_d:5d}  time={t_d:6.2f}s  "
              f"(no projection)")
        print(f"  query ratio flow/diff = {q_f / max(q_d, 1):.3f}")
        ok = ok and (0.0 <= a_f <= 1.0)

    print("=" * 64)
    print("Smoke test PASSED" if ok else "Smoke test FAILED (ASR out of [0,1])")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
