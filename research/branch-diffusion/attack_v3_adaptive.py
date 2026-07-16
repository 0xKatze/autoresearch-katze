#!/usr/bin/env python3
"""attack_v3_adaptive.py — SpecGDI v3 ADAPTIVE attack: gradient-through-defense.

Threat-model upgrade from v1/v2 (pure black-box at the defended pipeline):
  - v3 is **grey-box at the defended pipeline**: attacker knows the gddm_a
    denoiser weights and the victim GCN weights, and computes ANALYTICAL
    gradients through the full (model ∘ gddm_a) forward.
  - This is the standard "adaptive evaluation" pattern for diffusion-based
    defenses (cf. DiffPure adaptive attack literature). Must be flagged in
    paper §3 as a separate threat model.

Pipeline:
  1. Maintain soft tensors (X*, A_VV, A_VU) for the K-node trojan.
  2. Build the full (N+K, N+K) soft adjacency mixing the victim's binary edges
     with the soft trojan edges.
  3. gddm_a_soft_apply: run the denoiser's reverse loop end-to-end keeping
     A_t as a [0,1]-valued soft tensor. No sampling, no thresholding.
  4. gcn_dense_forward: apply the victim GCN's layers using dense matrix mul
     instead of GCNConv's edge-list dispatch.
  5. PGD-style projected gradient ascent on (X*, A_VV, A_VU) with the CW loss.

What's NOT differentiable in vanilla gddm_a (and how we soften it):
  - `torch.rand < p_flip` (sampling) → expectation: A_t = (1-p)·A0 + p·(1-A0)
  - `(probs > 0.5)` (thresholding) → keep `probs` continuous in [0,1]
  - GSDR `A_pred ∩ A_t` (boolean) → element-wise mult of soft tensors
  - `dense → edge_index` (discrete) → never convert; victim runs on dense A

The victim's BatchNorm uses running stats (model.eval()) — safe for gradient.

Adaptation note vs the v2 design doc §10.5:
  - v3 abandons the 4 v2 defect hypotheses (ablation falsified all 4)
  - v3's hypothesis: ASR floor under gddm_a is structural to "ZO can't see
    the denoiser's edge-pruning gradient". Analytical gradient breaks this.
"""
from __future__ import annotations
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from defenses import (_GDDMAEdgeDenoiser, _gddm_a_make_sched, _gddm_a_load,
                      _data_to_dense_adj, _gddm_sinusoid_t_emb)


# ============================================================
# Differentiable gddm_a
# ============================================================

@torch.no_grad()
def _eigh_unused():
    pass  # kept to signal: we do NOT use eigh in v3


def gddm_a_soft_apply(A0_soft: torch.Tensor, x: torch.Tensor,
                       denoiser: _GDDMAEdgeDenoiser, sched: dict,
                       t_star: int = 30, chunk_pairs: int = 20000,
                       tau_soft: float = 5.0, use_ste: bool = True) -> torch.Tensor:
    """Soft, fully-differentiable gddm_a purification.

    Inputs:
        A0_soft (n, n)   — symmetric, off-diagonal-zero, values in [0,1]. Gradient
                           flows back into this tensor.
        x       (n, d)   — node features (assumed clean; not perturbed by attacker).
        denoiser         — loaded _GDDMAEdgeDenoiser (params NOT requiring grad).
        sched            — schedule dict from _gddm_a_make_sched (cum_flip).
        t_star           — partial-noise level.

    Returns:
        A_t (n, n)       — symmetric soft purified adjacency in [0,1]^{n,n}.
                           Differentiable w.r.t. A0_soft.
    """
    n = A0_soft.shape[0]
    device = A0_soft.device
    T = sched["T"]
    t_star = int(max(1, min(T - 1, t_star)))
    p_flip = sched["cum_flip"][t_star].item()

    # 1. Expected forward diffusion (no sampling). Soft kernel:
    #    E[A_t | A_0] = (1 - p) * A_0 + p * (1 - A_0)
    A_t = (1.0 - p_flip) * A0_soft + p_flip * (1.0 - A0_soft)
    # Force symmetry + zero diag
    A_t = 0.5 * (A_t + A_t.t())
    A_t = A_t - torch.diag(torch.diag(A_t))

    # 2. Pre-compute upper-tri pair index
    triu_mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    all_pairs = triu_mask.nonzero(as_tuple=False)        # (P, 2)
    P = all_pairs.shape[0]

    # 3. Reverse loop — soft predictions, no threshold, soft GSDR intersect
    for t in range(t_star, 0, -1):
        h, te = denoiser.node_embeddings(x, A_t, t)
        # chunked edge logits
        if P <= chunk_pairs:
            logits = denoiser.edge_logits(h, te, all_pairs)
        else:
            chunks = []
            for s in range(0, P, chunk_pairs):
                pl = all_pairs[s:s + chunk_pairs]
                chunks.append(denoiser.edge_logits(h, te, pl))
            logits = torch.cat(chunks, dim=0)
        if use_ste:
            # STE: forward uses hard binary (matches real defense), backward
            # passes gradient through sigmoid(tau * logits). This avoids
            # sigmoid saturation killing A-gradient.
            probs_soft = torch.sigmoid(logits * tau_soft)
            probs_hard = (logits > 0).float()
            probs = probs_hard + (probs_soft - probs_soft.detach())
        else:
            probs = torch.sigmoid(logits * tau_soft)
        # Materialize symmetric soft A_pred
        A_pred = torch.zeros(n, n, device=device, dtype=A_t.dtype)
        A_pred[all_pairs[:, 0], all_pairs[:, 1]] = probs
        A_pred = A_pred + A_pred.t()
        # Soft GSDR: element-wise multiplication (drops anything A_t already 0,
        # softly drops anything A_t low).
        A_pred = A_pred * A_t
        A_t = A_pred
    return A_t


# ============================================================
# Dense GCN forward (mirrors prepare.GCN_GraphClassification.forward)
# ============================================================

def _normalize_adj_dense(A: torch.Tensor) -> torch.Tensor:
    """D^{-1/2}(A + I)D^{-1/2}, batched for a single graph."""
    n = A.shape[0]
    A_hat = A + torch.eye(n, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1).clamp_min(1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    # (D^{-1/2}) A_hat (D^{-1/2}) without forming diag matrices
    return d_inv_sqrt.unsqueeze(-1) * A_hat * d_inv_sqrt.unsqueeze(0)


def gcn_dense_forward(model, x: torch.Tensor, A_dense: torch.Tensor) -> torch.Tensor:
    """Apply prepare.GCN_GraphClassification to (x, dense soft A).

    Mirrors model.forward but uses dense matrix multiplication. Returns logits
    (1, n_classes) — single-graph batch.
    """
    A_norm = _normalize_adj_dense(A_dense)
    h = x
    for i in range(model.num_layers):
        conv = model.conv_layers[i]
        # GCNConv internals: y = A_norm @ (lin(x)) + bias
        h_lin = conv.lin(h)                                # (n, hidden)
        h = A_norm @ h_lin                                 # (n, hidden)
        if conv.bias is not None:
            h = h + conv.bias
        # BatchNorm (eval mode uses running stats — already set by model.eval())
        h = model.batch_norms[i](h)
        h = F.relu(h)
        # dropout: model.eval() makes this a no-op
        h = F.dropout(h, p=model.dropout, training=model.training)
    # global mean pool (single graph → mean over node dim)
    h = h.mean(dim=0, keepdim=True)                        # (1, hidden)
    return model.classifier(h)                             # (1, n_classes)


# ============================================================
# Composite soft adjacency builder
# ============================================================

def build_soft_composite_A(victim_edge_index, victim_n: int,
                            A_VV: torch.Tensor, A_VU: torch.Tensor,
                            device) -> torch.Tensor:
    """Build (N+K, N+K) soft symmetric adjacency mixing binary victim edges
    with soft trojan edges A_VV (K×K) and A_VU (K×N).

    Differentiable w.r.t. (A_VV, A_VU).
    """
    K = A_VV.shape[0]
    N = victim_n
    M = N + K
    A = torch.zeros(M, M, device=device)
    # Victim block: binary, no gradient
    if victim_edge_index is not None and victim_edge_index.numel() > 0:
        with torch.no_grad():
            A[victim_edge_index[0], victim_edge_index[1]] = 1.0
            A[victim_edge_index[1], victim_edge_index[0]] = 1.0
    # Inject A_VV at [N:M, N:M] (symmetric, zero diag) — gradients flow
    A_VV_sym = 0.5 * (A_VV + A_VV.t())
    A_VV_sym = A_VV_sym - torch.diag(torch.diag(A_VV_sym))
    A = A.clone()
    A[N:M, N:M] = A_VV_sym
    # Inject A_VU at [N:M, 0:N] and transpose
    A[N:M, 0:N] = A_VU
    A[0:N, N:M] = A_VU.t()
    return A


# ============================================================
# v3 adaptive attack — PGD on (X*, A_VV, A_VU) through soft (model ∘ gddm_a)
# ============================================================

def specgdi_v3_adaptive_optimize(model, data, denoiser, sched, prior, cfg, device,
                                  log_each: bool = False):
    """PGD-style gradient ascent on (X*, A_VV, A_VU) through soft gddm_a + dense GCN.

    Returns (X_hard, A_VV_hard, A_VU_hard, flipped, n_steps, traj).
    """
    K = cfg["v3_K"]
    T_iters = cfg["v3_T_iters"]
    eps = cfg["v3_eps"]
    eta_feat = cfg["v3_eta_feat"]
    eta_x = cfg["v3_lr_x"]
    eta_a = cfg["v3_lr_a"]
    t_star = cfg["v3_t_star"]
    budget_VU = cfg["v3_budget_VU_per_node"]
    timeout_s = cfg["v3_per_graph_timeout_s"]

    mu, cov_inv, _Lchol, thr = prior
    d_feat = mu.numel()
    N = data.num_nodes

    # Initialize trojan
    X_star = mu.unsqueeze(0).repeat(K, 1).clone().requires_grad_(True)        # (K, d)
    A_VV = torch.rand(K, K, generator=torch.Generator(device='cpu').manual_seed(7),
                      device='cpu').to(device) * 0.3
    A_VV = 0.5 * (A_VV + A_VV.t())
    A_VV.fill_diagonal_(0.0)
    A_VV = A_VV.clone().requires_grad_(True)
    A_VU = (torch.rand(K, N, device=device) * 0.3).clone().requires_grad_(True)

    # Adam optimizers smooth the noisy STE gradients (essential for stability)
    opt_X = torch.optim.Adam([X_star], lr=eta_x)
    opt_A = torch.optim.Adam([A_VV, A_VU], lr=eta_a)

    # Freeze denoiser + victim model params
    for p in denoiser.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)

    true_label = int(data.y.item())
    t0 = time.time()
    traj = []

    for it in range(T_iters):
        if time.time() - t0 > timeout_s:
            break
        # Build composite soft adjacency on (N+K) nodes
        # Compose features: victim x stacked with X_star
        X_all = torch.cat([data.x.float(), X_star], dim=0)                    # (N+K, d)
        A_composite = build_soft_composite_A(
            data.edge_index, N, A_VV, A_VU, device)                            # (N+K, N+K)

        # gddm_a soft purification (composite features as conditioning)
        A_purified = gddm_a_soft_apply(A_composite, X_all, denoiser, sched,
                                        t_star=t_star,
                                        tau_soft=cfg.get("v3_tau_soft", 1.0))   # (N+K, N+K)

        # Dense GCN forward through victim
        logits = gcn_dense_forward(model, X_all, A_purified)                  # (1, C)

        # CW loss: maximize (max_other - true), i.e., minimize (true - max_other)
        true_logit = logits[0, true_label]
        mask = torch.ones(logits.size(-1), dtype=torch.bool, device=device)
        mask[true_label] = False
        max_other = logits[0, mask].max()
        # Margin = true - max_other; want margin < 0 (flipped). Minimize margin.
        margin = true_logit - max_other
        # Mahalanobis prior pull on X_star (keep features stealthy)
        diff = X_star - mu.unsqueeze(0)
        maha = (diff @ cov_inv * diff).sum() / max(K, 1)
        loss = margin + eta_feat * maha

        # Compute gradients
        grads = torch.autograd.grad(loss, [X_star, A_VV, A_VU],
                                     retain_graph=False, create_graph=False,
                                     allow_unused=True)
        gX, gVV, gVU = grads

        # Adam update with per-tensor gradient clipping (STE grads can explode)
        grad_clip = cfg.get("v3_grad_clip", 1.0)
        with torch.no_grad():
            for g in (gX, gVV, gVU):
                if g is None:
                    continue
                gn = g.norm().clamp_min(1e-9)
                g.mul_((grad_clip / gn).clamp(max=1.0))
        # Manually attach clipped grads
        X_star.grad = gX if gX is not None else torch.zeros_like(X_star)
        A_VV.grad = gVV if gVV is not None else torch.zeros_like(A_VV)
        A_VU.grad = gVU if gVU is not None else torch.zeros_like(A_VU)

        opt_X.step(); opt_A.step()
        opt_X.zero_grad(set_to_none=True); opt_A.zero_grad(set_to_none=True)

        # Project after update
        with torch.no_grad():
            from attack import mahalanobis_project_rowwise
            X_star.data = mahalanobis_project_rowwise(X_star.data, mu, cov_inv, thr)
            A_VV.data = A_VV.data.clamp(0, 1)
            A_VV.data = 0.5 * (A_VV.data + A_VV.data.t())
            A_VV.data.fill_diagonal_(0.0)
            A_VU.data = A_VU.data.clamp(0, 1)

        # Log
        if log_each or it == T_iters - 1:
            traj.append({"it": it, "margin": float(margin.item()),
                         "loss": float(loss.item()),
                         "X_norm": float(X_star.detach().norm().item()),
                         "A_VV_sum": float(A_VV.detach().sum().item()),
                         "A_VU_sum": float(A_VU.detach().sum().item()),
                         "gX_norm": float(gX.detach().norm().item()) if gX is not None else None,
                         "gVV_norm": float(gVV.detach().norm().item()) if gVV is not None else None,
                         "gVU_norm": float(gVU.detach().norm().item()) if gVU is not None else None,
                         })

    # Discretize trojan: A_VV top-density, A_VU per-node top-budget
    from attack import discretize_trojan
    A_VV_hard, A_VU_hard = discretize_trojan(
        A_VV.detach(), A_VU.detach(),
        density_VV=cfg.get("v3_density_VV", 0.6),
        budget_VU_per_node=budget_VU,
    )

    # Final hard eval through real (discrete) gddm_a — this is the metric the
    # paper reports (defended ASR under the actual defense, not the soft surrogate)
    from defenses import gddm_a_defense
    from torch_geometric.data import Data as _Data

    n_total = N + K
    A_hard = build_soft_composite_A(data.edge_index, N,
                                      A_VV_hard.to(device), A_VU_hard.to(device),
                                      device)
    # Convert dense to edge_index for the real defense pass
    idx = (A_hard > 0.5).nonzero(as_tuple=False)
    ei_hard = idx.t().contiguous().long()
    X_all_hard = torch.cat([data.x.float(), X_star.detach()], dim=0)
    data_hard = _Data(x=X_all_hard, edge_index=ei_hard,
                       y=data.y, batch=torch.zeros(n_total, dtype=torch.long, device=device))

    # Defended forward (real gddm_a)
    purified = gddm_a_defense(data_hard,
                              ckpt_path=cfg["v3_gddm_a_ckpt"], t_star=t_star)
    with torch.no_grad():
        logits_hard = model(purified)
    margin_hard = (logits_hard[0, true_label] -
                   logits_hard[0, [i for i in range(logits_hard.size(-1)) if i != true_label]].max()).item()
    flipped = margin_hard < 0
    return X_star.detach(), A_VV_hard, A_VU_hard, bool(flipped), it + 1, traj


# ============================================================
# Default config (mirrors SPECGDI_V2_DEFAULTS but for v3 adaptive)
# ============================================================

SPECGDI_V3_DEFAULTS = {
    "v3_K": 3,
    "v3_T_iters": 30,                   # PGD iterations (vs ZO 10 — analytical grad is faster)
    "v3_lr_x": 0.01,                    # Adam step size for X
    "v3_lr_a": 0.05,                    # Adam step size for A (soft tensors in [0,1])
    "v3_grad_clip": 1.0,                # max ℓ2 norm per parameter tensor
    "v3_tau_soft": 3.0,                 # sigmoid sharpness for STE backward (3.0 keeps grad signal alive)
    "v3_use_ste": True,                 # straight-through estimator on the binarization
    "v3_eps": 0.01,
    "v3_eta_feat": 0.05,                # weak Mahalanobis prior pull
    "v3_t_star": 30,
    "v3_budget_VU_per_node": 1.0,       # full edge — let attack pick
    "v3_density_VV": 0.6,
    "v3_per_graph_timeout_s": 60.0,
    # v3_gddm_a_ckpt: set per-dataset by the runner
}


def run_v3_sweep(model, test_graphs, device, cfg=None, gddm_a_ckpt: Optional[str] = None):
    """Top-level runner for v3 adaptive attack."""
    from attack import data_prior, CONFIG, SPECGDI_DEFAULTS
    cfg = {**CONFIG, **SPECGDI_DEFAULTS, **SPECGDI_V3_DEFAULTS, **(cfg or {})}
    if gddm_a_ckpt is not None:
        cfg["v3_gddm_a_ckpt"] = gddm_a_ckpt
    prior = data_prior(test_graphs, device)

    # Load denoiser + sched once
    denoiser, sched, _in_dim = _gddm_a_load(cfg["v3_gddm_a_ckpt"], device)

    n_flip = 0
    trajs = []
    for data in test_graphs:
        data = data.to(device)
        _, _, _, flipped, n_iters, traj = specgdi_v3_adaptive_optimize(
            model, data, denoiser, sched, prior, cfg, device, log_each=False)
        n_flip += int(flipped)
        trajs.append({"flipped": flipped, "n_iters": n_iters,
                      "final_margin": traj[-1]["margin"] if traj else None})

    n = len(test_graphs) or 1
    return {"asr_full": n_flip / n, "n_graphs": n, "trajs": trajs,
            "feat_mode": "specgdi_v3_adaptive"}
