#!/usr/bin/env python3
"""smoke_v3_oneshot.py — Run v3 adaptive attack on a single PROTEINS graph,
verify autograd works end-to-end, print gradient norms + final margin.

Adaptation note vs production v3 run:
  - Single graph (fold 0, first attackable n∈[30,80])
  - Verbose logging on every PGD step
  - No comparisons — just "does it run, do gradients flow, does loss decrease"
"""
import json
import time
import torch

from prepare import MODELS_DIR, DEVICE
from sweep_stealth import load_fold as load_fold_discrete
from defenses import gddm_a_ckpt_path
from attack_v3_adaptive import (specgdi_v3_adaptive_optimize,
                                  SPECGDI_V3_DEFAULTS, gddm_a_soft_apply,
                                  build_soft_composite_A, gcn_dense_forward)
from defenses import _gddm_a_load
from attack import data_prior, CONFIG, SPECGDI_DEFAULTS


def main():
    print("=== v3 adaptive oneshot — PROTEINS fold 0, single graph ===")
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    fi = meta["folds"][0]
    model, graphs = load_fold_discrete(fi, meta)
    # Pick first graph with 30 <= n <= 80
    g = next(g for g in graphs if 30 <= g.num_nodes <= 80)
    print(f"  graph n_nodes={g.num_nodes}, n_edges={g.edge_index.size(1) // 2}")

    cfg = {**CONFIG, **SPECGDI_DEFAULTS, **SPECGDI_V3_DEFAULTS}
    cfg["v3_gddm_a_ckpt"] = gddm_a_ckpt_path("PROTEINS")
    cfg["v3_T_iters"] = 10  # short for smoke
    prior = data_prior(graphs, DEVICE)
    denoiser, sched, _ = _gddm_a_load(cfg["v3_gddm_a_ckpt"], DEVICE)

    # Smoke component test 1: gddm_a_soft_apply on a small soft A
    print("\n[1/3] Sanity-check gddm_a_soft_apply produces a non-trivial output...")
    A_test = torch.rand(10, 10, device=DEVICE, requires_grad=True) * 0.5
    A_test_sym = 0.5 * (A_test + A_test.t())
    A_test_sym = A_test_sym - torch.diag(torch.diag(A_test_sym))
    x_test = torch.randn(10, 3, device=DEVICE)
    t0 = time.time()
    A_out = gddm_a_soft_apply(A_test_sym, x_test, denoiser, sched, t_star=5)
    print(f"  A_out range=[{A_out.min().item():.4f}, {A_out.max().item():.4f}]  "
          f"sum={A_out.sum().item():.4f}  wall={time.time()-t0:.2f}s")
    # Backward test
    loss = A_out.sum()
    g_A_test, = torch.autograd.grad(loss, [A_test])
    print(f"  grad through gddm_a_soft norm={g_A_test.norm().item():.4f} "
          f"({'OK' if g_A_test.norm() > 0 else 'FAIL — grad is zero!'})")

    # Smoke component test 2: dense GCN forward gradient
    print("\n[2/3] Sanity-check gcn_dense_forward gradient w.r.t. A...")
    g_dev = g.to(DEVICE)
    A0 = torch.zeros(g_dev.num_nodes, g_dev.num_nodes, device=DEVICE)
    ei = g_dev.edge_index
    A0[ei[0], ei[1]] = 1.0
    A0_soft = A0.clone().requires_grad_(True)
    logits = gcn_dense_forward(model, g_dev.x.float(), A0_soft)
    print(f"  logits shape={logits.shape}, sum={logits.sum().item():.4f}")
    loss = logits[0].max()
    g_A0, = torch.autograd.grad(loss, [A0_soft])
    print(f"  grad through dense GCN norm={g_A0.norm().item():.4f} "
          f"({'OK' if g_A0.norm() > 0 else 'FAIL'})")

    # Full v3 attack
    print("\n[3/3] Full v3 adaptive attack (10 PGD iters)...")
    t0 = time.time()
    X, A_VV, A_VU, flipped, n_iters, traj = specgdi_v3_adaptive_optimize(
        model, g_dev, denoiser, sched, prior, cfg, DEVICE, log_each=True)
    wall = time.time() - t0
    print(f"\n  iters={n_iters}  flipped={flipped}  wall={wall:.1f}s")
    print("  trajectory (margin per iter):")
    for tr in traj:
        print(f"    it={tr['it']:2d}  margin={tr['margin']:+.4f}  "
              f"X_norm={tr['X_norm']:.3f}  A_VV_sum={tr['A_VV_sum']:.3f}  "
              f"A_VU_sum={tr['A_VU_sum']:.3f}  "
              f"gX={tr.get('gX_norm', 'n/a'):>7.4f}  "
              f"gVV={tr.get('gVV_norm', 'n/a') or 'None':>7}  "
              f"gVU={tr.get('gVU_norm', 'n/a') or 'None':>7}")


if __name__ == "__main__":
    main()
