#!/usr/bin/env python3
"""
smoke_flow_b.py — Smoke test for attack_flow_b.py (FM-Velosteal, variant B:
TRAINED velocity field).

Verifies, end-to-end on a handful of PROTEINS graphs against the `none` defense:
  (a) attack_flow_b.run_attack runs without error,
  (b) returns a sane ASR float in [0, 1],
  (c) injected (pre-scale) features land inside the clean Mahalanobis ball,
  (d) the velocity field actually trains (flow-matching loss drops), and
  (e) a rough cost/quality comparison vs attack_flow (variant A, analytic) on the
      same tiny input (apples-to-apples).

Mirrors smoke_flow.py's harness. attack_flow_b is imported FIRST so it heals the
branch-diffusion attack.py name collision (registers the canonical katze attack as
`attack`), after which attack_flow imports cleanly too.

Usage: micromamba run -n graph_adversarial python smoke_flow_b.py
"""
import os
# CPU: deterministic + sidesteps the env's CUDA segfault on this GPU. Force CPU
# BEFORE importing prepare so its module-global DEVICE is cpu.
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
import attack_flow_b          # heals env: registers canonical 'attack'
import attack_flow            # variant A, for comparison


class QueryCounter:
    """Tally victim graphs-per-forward — same wrapper as smoke_flow.py."""

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
    print(f"flow_b T_flow={attack_flow_b.CONFIG['T_flow']}, "
          f"integrator={attack_flow_b.CONFIG['integrator']}, "
          f"vf=[{attack_flow_b.CONFIG['vf_layers']}x{attack_flow_b.CONFIG['vf_hidden']}, "
          f"{attack_flow_b.CONFIG['vf_train_steps']} steps], "
          f"lambda={attack_flow_b.CONFIG['lambda_schedule']} "
          f"[{attack_flow_b.CONFIG['lambda_min']}->{attack_flow_b.CONFIG['lambda_max']}]")

    # ---- (d) field trains: flow-matching loss should drop vs init -------------
    print("=" * 64)
    X = torch.cat([g.x for g in test_graphs], dim=0).float()
    in_dim = X.size(1)
    net0 = attack_flow_b.VelocityField(in_dim,
                                       attack_flow_b.CONFIG["vf_hidden"],
                                       attack_flow_b.CONFIG["vf_layers"],
                                       attack_flow_b.CONFIG["vf_time_dim"], seed=0)

    def fm_loss(net, n=2000):
        gen = torch.Generator().manual_seed(777)
        idx = torch.randint(0, X.size(0), (n,), generator=gen)
        x1 = X[idx]
        x0 = torch.randn(n, in_dim, generator=gen)
        t = torch.rand(n, 1, generator=gen)
        with torch.no_grad():
            v = net((1 - t) * x0 + t * x1, t)
        return ((v - (x1 - x0)) ** 2).mean().item()

    l_init = fm_loss(net0)
    net = attack_flow_b.train_velocity_field(test_graphs, DEVICE, attack_flow_b.CONFIG)
    l_fit = fm_loss(net)
    trained_ok = l_fit < l_init
    print(f"flow-matching loss: init={l_init:.4f} -> fit={l_fit:.4f}  "
          f"({'DROPPED' if trained_ok else 'NO DROP'})")

    def run_b(feat_scale):
        old_f = attack_flow_b.CONFIG["feat_scale"]
        attack_flow_b.CONFIG["feat_scale"] = feat_scale
        try:
            with QueryCounter(model) as qc:
                t0 = time.time()
                asr = attack_flow_b.run_attack(model, test_graphs, DEVICE)
                dt = time.time() - t0
            st = attack_flow_b.FLOW_STATS
            frac = (st["in_ball"] / st["total"]) if st["total"] else float("nan")
            return asr, qc.n, dt, st["in_ball"], st["total"], frac
        finally:
            attack_flow_b.CONFIG["feat_scale"] = old_f

    def run_a(feat_scale):
        old_f = attack_flow.CONFIG["feat_scale"]
        attack_flow.CONFIG["feat_scale"] = feat_scale
        try:
            with QueryCounter(model) as qc:
                t0 = time.time()
                asr = attack_flow.run_attack(model, test_graphs, DEVICE)
                dt = time.time() - t0
            st = attack_flow.FLOW_STATS
            frac = (st["in_ball"] / st["total"]) if st["total"] else float("nan")
            return asr, qc.n, dt, frac
        finally:
            attack_flow.CONFIG["feat_scale"] = old_f

    ok = trained_ok
    # Two regimes (mirrors smoke_flow.py):
    #   auto_x2 : CONFIG default; weak victim trivially flipped -> correctness.
    #   1.0     : stealth regime (injected == in-ball raw feature) -> exercises the
    #             trained drift + projection meaningfully.
    for fs in ("auto_x2", 1.0):
        print("=" * 64)
        print(f"feat_scale = {fs}")
        a_b, q_b, t_b, nb, nt, frac_b = run_b(fs)
        a_a, q_a, t_a, frac_a = run_a(fs)
        print(f"[flow_b/trained ] ASR={a_b*100:6.2f}%  queries={q_b:5d}  "
              f"time={t_b:6.2f}s  in_ball={nb}/{nt} ({frac_b*100:.1f}%)")
        print(f"[flow_a/analytic] ASR={a_a*100:6.2f}%  queries={q_a:5d}  "
              f"time={t_a:6.2f}s  in_ball=({frac_a*100:.1f}%)")
        ok = ok and (0.0 <= a_b <= 1.0)
        # stealth contract: every injected/queried feature must be in-ball
        ok = ok and (nb == nt)

    print("=" * 64)
    print("Smoke test PASSED" if ok else "Smoke test FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
