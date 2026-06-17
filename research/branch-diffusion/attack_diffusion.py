#!/usr/bin/env python3
"""
attack_diffusion.py — N23 DiSGIA-NoParam (AdvAD-style non-parametric diffusion)

Drop-in replacement for attack.py with the same run_attack(model, test_graphs, device) -> float
signature, so run.py / executor.py work unchanged.

Core idea (Li et al., NeurIPS 2024, arXiv:2503.09124):
  Cast the attack itself as a non-parametric discrete-time diffusion process.
  Forward:   F_s_T  = sqrt(1 - alpha_T) * eps_0          (pure noise; alpha_T -> 0)
  Reverse:   each step t, denoise via ZO-estimated score of CW loss on victim:
             eps_hat_t = eps_0 - sqrt(1 - alpha_t) * grad_x0 (CW)
             then DDIM transition to F_s_{t-1}.

Key differences vs attack.py (baseline KATZE):
  - No generator network. F_s evolves directly under reverse diffusion.
  - No Adam. Updates are deterministic DDIM transitions + ZO score.
  - Annealed noise schedule (high beta early -> explore; low beta late -> exploit).

Injection-only invariant (Proposition 3.1 from N23 design):
  construct_perturbed_graph() is pure: original_data passed in, new Data returned.
  G is never mutated. F_s lives outside the original feature block.
  Edge additions only between injected nodes and existing nodes (or among injected).
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

# Reuse helpers from baseline attack.py (no duplication)
from attack import (
    construct_perturbed_graph,
    batch_loss,
    get_prediction,
    select_targets_topk,
    select_targets_spectral,
)

# ============================================================
# CONFIG — N23 specific (Pipeline / codegen can tune these)
# ============================================================

CONFIG = {
    # Diffusion schedule
    "T_steps": 50,                # reverse-process steps (replaces attack_epochs)
    "schedule": "cosine",         # "linear" | "cosine"
    "alpha_min": 1e-3,            # alpha at t=T (near-pure noise)
    "alpha_max": 0.999,           # alpha at t=0 (near-clean)

    # ZO score estimation
    "sigma": 5e-3,                # ZO smoothing (same as baseline)
    "grad_method": "cge",         # "cge" | "rgf"
    "rgf_samples": 20,            # only used if grad_method == "rgf"

    # AdvAD-specific
    "guidance_scale": 1.0,        # lambda — strength of adversarial score injection
    "predict_x0": True,           # estimate ZO on predicted x0, not on x_t directly
    "use_AdvAD_X": False,         # if True, apply Dynamic Guidance Injection (DGI)
    "dgi_skip_threshold": 0.01,   # ||x_hat_0 - x_adv_prev|| below this -> skip guidance

    # Inherited from baseline (so executor stays consistent)
    "feat_scale": "auto_x2",
    "loss_type": "cw",
    "kappa": -0.1,
    "edge_strategy": "full",
    "node_budget": 1,
    "spectral_top_k_eig": 10,

    # Multi-restart (AdvAD itself doesn't use restart, but our query budget allows it)
    "n_restarts": 3,              # fewer than baseline (5) since each restart is cheaper
}

# ============================================================
# DDIM noise schedule
# ============================================================

def make_alpha_schedule(T, mode="cosine", alpha_min=1e-3, alpha_max=0.999):
    """Cumulative product alpha_bar_t for DDIM, indexed t=0..T (t=T is most-noisy)."""
    if mode == "linear":
        alphas = torch.linspace(alpha_max, alpha_min, T + 1)
    elif mode == "cosine":
        # Nichol & Dhariwal 2021 cosine schedule
        s = 0.008
        ts = torch.linspace(0, 1, T + 1)
        f = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        alphas = f / f[0]
        alphas = alphas.clamp(alpha_min, alpha_max)
    else:
        raise ValueError(f"Unknown schedule: {mode}")
    return alphas  # shape (T+1,)


def predict_x0_from_xt(x_t, eps, alpha_bar_t):
    """DDIM x0 prediction: x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)."""
    return (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


def ddim_step(x_t, eps_pred, alpha_bar_t, alpha_bar_prev):
    """Deterministic DDIM reverse step: x_t -> x_{t-1}."""
    x0_pred = predict_x0_from_xt(x_t, eps_pred, alpha_bar_t)
    return torch.sqrt(alpha_bar_prev) * x0_pred + \
           torch.sqrt(1.0 - alpha_bar_prev) * eps_pred


# ============================================================
# ZO score estimation on predicted x_0 (the F_s candidate)
# ============================================================

def zo_score_cge(model, original_data, x0, target_indices, sigma, kappa,
                 feat_scale, device, loss_type="cw"):
    """
    CGE estimate of gradient of CW loss w.r.t. x0 (the predicted clean injected feats).

    Returns: (success_flag, grad_tensor of same shape as x0).
    Mirrors estimate_gradient_cge in attack.py but operates on x0 (not generator output).
    """
    n_inject, feat_dim = x0.shape
    true_label = original_data.y.item()
    x0_det = x0.detach()

    data_list = []
    coords = []
    for i in range(n_inject):
        for j in range(feat_dim):
            e = torch.zeros_like(x0_det)
            e[i, j] = 1.0
            data_list.append(construct_perturbed_graph(
                original_data, x0_det + sigma * e, target_indices, feat_scale))
            data_list.append(construct_perturbed_graph(
                original_data, x0_det - sigma * e, target_indices, feat_scale))
            coords.append((i, j))

    loss_list, success = batch_loss(
        model, data_list, true_label, kappa, device,
        loss_type=loss_type, clean_data=original_data)

    grad = torch.zeros_like(x0_det)
    for k, (i, j) in enumerate(coords):
        grad[i, j] = (loss_list[2*k] - loss_list[2*k+1]) / (2 * sigma)
    return success, grad


def zo_score_rgf(model, original_data, x0, target_indices, sigma, n_samples,
                 kappa, feat_scale, device, loss_type="cw"):
    """RGF estimate (cheaper, higher variance) — for ablation."""
    n_inject, feat_dim = x0.shape
    true_label = original_data.y.item()
    x0_det = x0.detach()

    rand = torch.randn(n_samples, n_inject, feat_dim, device=device)
    perturbed = torch.cat([x0_det.unsqueeze(0) + sigma * rand,
                           x0_det.unsqueeze(0) - sigma * rand], dim=0)
    data_list = [construct_perturbed_graph(
        original_data, perturbed[i], target_indices, feat_scale)
        for i in range(2 * n_samples)]

    loss_list, success = batch_loss(
        model, data_list, true_label, kappa, device,
        loss_type=loss_type, clean_data=original_data)

    factors = [(loss_list[i] - loss_list[n_samples + i]) / sigma
               for i in range(n_samples)]
    fac_t = torch.tensor(factors, device=device).reshape(n_samples, 1, 1)
    grad = (fac_t * rand).mean(dim=0) / 2.0
    return success, grad


# ============================================================
# AdvAD-style reverse process for a single graph
# ============================================================

def _attack_single_diffusion(model, data, targets, fs, cfg, n_feat, device, seed=0):
    """
    One run of non-parametric diffusion attack on a single graph.

    Forward (init):  x_T = sqrt(1 - alpha_T) * eps_0    (pure noise; alpha_T ~ 0)
    Reverse loop t=T..1:
        x0_pred = predict_x0(x_t, eps_pred)
        # quick exit if x0_pred already misclassifies
        check victim; if success -> return True
        score = ZO_grad(CW(victim(inject(G, x0_pred))), wrt=x0_pred)
        # AdvAD AMG (Eq. 8 with sign flipped for untargeted CW minimization):
        eps_hat_t = eps_0 - lambda * sqrt(1 - alpha_t) * score
        x_{t-1} = DDIM_step(x_t, eps_hat_t, alpha_t, alpha_{t-1})
    Returns: True if any intermediate x0 misclassifies, else False after T steps.

    Injection-only audit:
      - `data` (original graph) only passed into construct_perturbed_graph(),
        which is pure (no in-place writes).
      - x0_pred is appended to data.x inside construct_perturbed_graph; original
        data.x and data.edge_index are read-only here. ✅
    """
    torch.manual_seed(seed)
    T = cfg["T_steps"]
    kappa = cfg["kappa"]
    m = cfg["node_budget"]

    # Pre-sample noise (AdvAD: eps_0 is fixed throughout the trajectory)
    eps_0 = torch.randn(m, n_feat, device=device)

    # Diffusion schedule
    alphas = make_alpha_schedule(
        T, mode=cfg["schedule"],
        alpha_min=cfg["alpha_min"], alpha_max=cfg["alpha_max"]).to(device)

    # Init x_T  (very noisy)
    alpha_T = alphas[T]
    x_t = torch.sqrt(1.0 - alpha_T) * eps_0   # since x_origin = 0 for fresh injection

    x0_prev = None  # for DGI

    for step in range(T, 0, -1):
        alpha_bar_t = alphas[step]
        alpha_bar_prev = alphas[step - 1]

        # 1. Predict x_0 from x_t
        x0_pred = predict_x0_from_xt(x_t, eps_0, alpha_bar_t)

        # 2. Quick success check on current x_0 prediction
        perturbed = construct_perturbed_graph(data, x0_pred, targets, fs)
        perturbed.batch = torch.zeros(perturbed.num_nodes,
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(perturbed)
        pred = get_prediction(logits)
        if pred != data.y.item():
            return True

        # 3. AdvAD-X DGI: skip guidance if x0 barely changed (saves queries)
        if cfg["use_AdvAD_X"] and x0_prev is not None:
            delta = (x0_pred - x0_prev).norm().item()
            if delta < cfg["dgi_skip_threshold"]:
                # Skip ZO call: reuse eps_0 as the noise prediction
                eps_hat_t = eps_0
            else:
                _, score = _estimate_score(
                    model, data, x0_pred, targets, fs, cfg, device)
                eps_hat_t = eps_0 - cfg["guidance_scale"] * \
                            torch.sqrt(1.0 - alpha_bar_t) * score
        else:
            # 4. ZO score estimation
            _, score = _estimate_score(
                model, data, x0_pred, targets, fs, cfg, device)
            # 5. AMG (AdvAD Eq. 8 adapted for CW minimization)
            eps_hat_t = eps_0 - cfg["guidance_scale"] * \
                        torch.sqrt(1.0 - alpha_bar_t) * score

        x0_prev = x0_pred

        # 6. DDIM step to x_{t-1}
        x_t = ddim_step(x_t, eps_hat_t, alpha_bar_t, alpha_bar_prev)

    # Final check at x_0
    perturbed = construct_perturbed_graph(data, x_t, targets, fs)
    perturbed.batch = torch.zeros(perturbed.num_nodes,
                                  dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(perturbed)
    return get_prediction(logits) != data.y.item()


def _estimate_score(model, data, x0, targets, fs, cfg, device):
    """Dispatch ZO estimator. Returns (success_flag_unused, score_tensor)."""
    if cfg["grad_method"] == "cge":
        return zo_score_cge(
            model, data, x0, targets,
            cfg["sigma"], cfg["kappa"], fs, device, cfg["loss_type"])
    else:
        return zo_score_rgf(
            model, data, x0, targets,
            cfg["sigma"], cfg["rgf_samples"], cfg["kappa"], fs, device,
            cfg["loss_type"])


# ============================================================
# Public entry point — same signature as attack.py
# ============================================================

def run_attack(model, test_graphs, device):
    """Run N23 DiSGIA-NoParam attack on all test graphs, return ASR."""
    cfg = CONFIG
    n_feat = test_graphs[0].x.size(1)
    n_success = 0

    for data in test_graphs:
        data = data.to(device)

        # feat_scale resolution (same as baseline)
        fs = cfg["feat_scale"]
        if fs == "auto":
            fs = math.sqrt(data.num_nodes)
        elif fs == "auto_x2":
            fs = 2.0 * math.sqrt(data.num_nodes)
        elif fs == "auto_x3":
            fs = 3.0 * math.sqrt(data.num_nodes)
        elif fs == "auto_linear":
            fs = float(data.num_nodes)

        # Edge target selection (unchanged from baseline; A_sv is fixed per-strategy)
        if cfg["edge_strategy"] == "full":
            targets = torch.arange(data.num_nodes, device=device)
        elif cfg["edge_strategy"] == "spectral":
            targets = select_targets_spectral(
                data, cfg["node_budget"], cfg["spectral_top_k_eig"], device)
        else:
            targets = select_targets_topk(data, cfg["node_budget"])

        # Multi-restart with different seeds (cheap; T_steps stays the same)
        success = False
        for r in range(cfg["n_restarts"]):
            seed = r * 1000 + data.num_nodes
            if _attack_single_diffusion(
                    model, data, targets, fs, cfg, n_feat, device, seed=seed):
                success = True
                break

        if success:
            n_success += 1

    return n_success / len(test_graphs) if test_graphs else 0.0


# ============================================================
# Module sanity check (run: python attack_diffusion.py)
# ============================================================

if __name__ == "__main__":
    print("attack_diffusion.py — N23 DiSGIA-NoParam")
    print(f"CONFIG keys: {list(CONFIG.keys())}")
    alphas = make_alpha_schedule(50, mode="cosine")
    print(f"alpha schedule: T=50, alpha_max={alphas[0]:.4f}, "
          f"alpha_min={alphas[-1]:.4f}, alpha[25]={alphas[25]:.4f}")
