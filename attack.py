#!/usr/bin/env python3
"""
attack.py — THE FILE AI AGENTS MODIFY

This file contains the full attack pipeline. The AI agent can modify:
- Attack hyperparameters (feat_scale, sigma, lr, etc.)
- Gradient estimation method
- Loss function
- Edge selection strategy
- Feature generation logic
- Any other attack logic

Rules:
1. Must define `run_attack(model, test_graphs, device) -> float` returning ASR
2. Must be runnable in < 5 minutes for a single fold
3. All changes are tracked via git
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch

# ============================================================
# CONFIG — AI agent tunes these
# ============================================================

CONFIG = {
    "feat_scale": "auto_x2",    # float or "auto" | "auto_x2" | "auto_x3"
    "sigma": 5e-3,              # ZOO smoothing parameter
    "gen_lr": 5e-3,             # generator learning rate
    "attack_epochs": 50,        # optimization steps per graph
    "grad_method": "cge",       # "rgf" or "cge"
    "loss_type": "cw",           # "cw" | "cosine" | "hybrid"
    "edge_strategy": "full",    # "topk" | "spectral" | "full"
    "node_budget": 1,           # injected nodes per graph
    "kappa": -0.1,              # CW loss margin
    "gen_hid_dim": 128,         # generator hidden dim
    "spectral_top_k_eig": 10,  # eigenvalues for spectral edge selection
    # --- v3 hybrid (diffusion-as-restart) ---
    "use_diffusion_restart": True,  # if all baseline restarts fail, try DDIM
    "diff_T_steps": 20,             # T=40 tested: identical result, reverted (saturated)
    "diff_alpha_min": 0.5,
    "diff_alpha_max": 0.99,
    "diff_guidance": 2.0,           # score guidance strength
    "diff_n_seeds": 2,              # n=5 tested: identical result, reverted (saturated)
    # --- v4 joint diffusion (N24: X + A_sv with cardinality budget) ---
    "use_joint_diffusion": True,    # at restart-6, run joint X+A diffusion
    "joint_edge_budget": "avg_deg", # int or "avg_deg" — cardinality of A_sv per injected node
}

# ============================================================
# Core attack components
# ============================================================

def get_prediction(logits):
    if logits.dim() > 1:
        logits = logits.squeeze(0) if logits.size(0) == 1 else logits
    if logits.numel() > 1:
        pred = logits.argmax(dim=-1).item()
        return pred
    return 1 if torch.sigmoid(logits).item() > 0.5 else 0


def construct_perturbed_graph(original_data, node_feats, target_indices,
                              feat_scale=1.0):
    device = original_data.x.device
    n_orig = original_data.num_nodes
    m = node_feats.shape[0]
    scaled = node_feats * feat_scale if feat_scale != 1.0 else node_feats
    X = torch.cat([original_data.x, scaled], dim=0)
    # Build edges: each injected node connects to all targets
    src_list, dst_list = [], []
    for i in range(m):
        inj_id = n_orig + i
        t = target_indices
        src_list.append(torch.full((t.shape[0],), inj_id, device=device))
        dst_list.append(t)
        src_list.append(t)
        dst_list.append(torch.full((t.shape[0],), inj_id, device=device))
    new_src = torch.cat(src_list)
    new_dst = torch.cat(dst_list)
    new_edges = torch.stack([new_src, new_dst], dim=0)
    ei = torch.cat([original_data.edge_index, new_edges], dim=1)
    return Data(x=X, edge_index=ei, y=getattr(original_data, 'y', None))


def calculate_cw_loss(logits, true_label, kappa, device):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    true_logit = logits[0, true_label]
    mask = torch.ones(logits.shape[1], dtype=torch.bool, device=device)
    mask[true_label] = False
    max_other = logits[0, mask].max()
    lower = torch.tensor(-kappa, dtype=torch.float32, device=device)
    return torch.max(true_logit - max_other, lower)


def batch_loss(model, data_list, true_label, kappa, device,
               loss_type="cw", clean_data=None):
    """Compute loss for a batch of perturbed graphs."""
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        logits = model(batch)
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)

    losses = []
    success = 0

    if logits.size(-1) > 1:
        preds = logits.argmax(dim=-1)
        true_logits = logits[:, true_label]
        mask = torch.ones(logits.size(-1), dtype=torch.bool, device=device)
        mask[true_label] = False
        max_others = logits[:, mask].max(dim=-1).values
        kappa_val = kappa if isinstance(kappa, float) else kappa.item()
        cw = torch.clamp(true_logits - max_others, min=kappa_val).tolist()

        if loss_type == "cw":
            losses = cw
        elif loss_type == "cosine" and clean_data is not None:
            # Use logits as proxy for embedding distance
            clean_data_b = clean_data.clone().to(device)
            clean_data_b.batch = torch.zeros(clean_data_b.num_nodes,
                                              dtype=torch.long, device=device)
            with torch.no_grad():
                clean_logits = model(clean_data_b)
            if clean_logits.dim() == 1:
                clean_logits = clean_logits.unsqueeze(0)
            cos_sim = F.cosine_similarity(
                clean_logits.expand_as(logits), logits, dim=-1)
            losses = (cos_sim - 1.0).tolist()
        elif loss_type == "hybrid" and clean_data is not None:
            clean_data_b = clean_data.clone().to(device)
            clean_data_b.batch = torch.zeros(clean_data_b.num_nodes,
                                              dtype=torch.long, device=device)
            with torch.no_grad():
                clean_logits = model(clean_data_b)
            if clean_logits.dim() == 1:
                clean_logits = clean_logits.unsqueeze(0)
            cos_sim = F.cosine_similarity(
                clean_logits.expand_as(logits), logits, dim=-1)
            losses = [(c + (cs - 1.0)) for c, cs in zip(cw, cos_sim.tolist())]
        else:
            losses = cw

        if (preds != true_label).any():
            success = 1

    return losses, success


# ============================================================
# Edge selection
# ============================================================

def select_targets_topk(data, n_inject):
    n_orig = data.num_nodes
    degrees = torch.bincount(data.edge_index[0], minlength=n_orig).float()
    k = min(n_inject, n_orig)
    top = torch.topk(degrees, k=k).indices
    if top.shape[0] >= n_inject:
        return top[:n_inject]
    reps = (n_inject + top.shape[0] - 1) // top.shape[0]
    return top.repeat(reps)[:n_inject]


def select_targets_spectral(data, n_inject, top_k_eig=10, device='cpu'):
    data = data.to(device)
    n_orig = data.num_nodes
    k = min(top_k_eig, n_orig)

    # Original eigenvalues
    A = torch.zeros(n_orig, n_orig, device=device)
    A[data.edge_index[0], data.edge_index[1]] = 1.0
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = deg[mask].pow(-0.5)
    D = torch.diag(deg_inv_sqrt)
    L = torch.eye(n_orig, device=device) - D @ A @ D
    orig_eig = torch.linalg.eigvalsh(L)[:k]

    impact = torch.zeros(n_orig, device=device)
    for v in range(n_orig):
        n_total = n_orig + n_inject
        inj_idx = torch.arange(n_orig, n_total, device=device)
        target = torch.full((n_inject,), v, dtype=torch.long, device=device)
        new_edges = torch.stack([
            torch.cat([inj_idx, target]),
            torch.cat([target, inj_idx]),
        ], dim=0)
        ei_aug = torch.cat([data.edge_index, new_edges], dim=1)

        A2 = torch.zeros(n_total, n_total, device=device)
        A2[ei_aug[0], ei_aug[1]] = 1.0
        deg2 = A2.sum(dim=1)
        d2 = torch.zeros_like(deg2)
        m2 = deg2 > 0
        d2[m2] = deg2[m2].pow(-0.5)
        L2 = torch.eye(n_total, device=device) - torch.diag(d2) @ A2 @ torch.diag(d2)
        aug_eig = torch.linalg.eigvalsh(L2)[:k]

        ml = min(len(orig_eig), len(aug_eig))
        impact[v] = torch.norm(orig_eig[:ml] - aug_eig[:ml], p=2)

    if n_orig >= n_inject:
        return impact.topk(n_inject).indices
    top = impact.topk(n_orig).indices
    reps = (n_inject + n_orig - 1) // n_orig
    return top.repeat(reps)[:n_inject]


# ============================================================
# Gradient estimation
# ============================================================

def estimate_gradient_cge(model, original_data, node_feats, target_indices,
                          sigma, kappa, feat_scale, device, loss_type="cw"):
    n_inject, feat_dim = node_feats.shape
    true_label = original_data.y.item()
    feats_det = node_feats.detach()

    data_list = []
    coords = []
    for i in range(n_inject):
        for j in range(feat_dim):
            e = torch.zeros_like(feats_det)
            e[i, j] = 1.0
            data_list.append(construct_perturbed_graph(
                original_data, feats_det + sigma * e, target_indices, feat_scale))
            data_list.append(construct_perturbed_graph(
                original_data, feats_det - sigma * e, target_indices, feat_scale))
            coords.append((i, j))

    loss_list, success = batch_loss(
        model, data_list, true_label, kappa, device,
        loss_type=loss_type, clean_data=original_data)

    grad = torch.zeros_like(feats_det)
    for k, (i, j) in enumerate(coords):
        grad[i, j] = (loss_list[2*k] - loss_list[2*k+1]) / (2 * sigma)
    return success, grad


def estimate_gradient_rgf(model, original_data, node_feats, target_indices,
                          sigma, eval_num, kappa, feat_scale, device, loss_type="cw"):
    n_inject, feat_dim = node_feats.shape
    true_label = original_data.y.item()
    sample_num = eval_num // 2
    feats_det = node_feats.detach()

    rand = torch.randn(sample_num, n_inject, feat_dim, device=device)
    perturbed = torch.cat([
        feats_det.unsqueeze(0) + sigma * rand,
        feats_det.unsqueeze(0) - sigma * rand,
    ], dim=0)

    data_list = []
    for idx in range(eval_num):
        data_list.append(construct_perturbed_graph(
            original_data, perturbed[idx], target_indices, feat_scale))

    loss_list, success = batch_loss(
        model, data_list, true_label, kappa, device,
        loss_type=loss_type, clean_data=original_data)

    factors = [(loss_list[i] - loss_list[sample_num + i]) / sigma
               for i in range(sample_num)]
    fac_t = torch.tensor(factors, device=device).reshape(sample_num, 1, 1)
    grad = (fac_t * rand).mean(dim=0) / 2.0
    return success, grad


# ============================================================
# Generator (simple MLP)
# ============================================================

import torch.nn as nn


class SimpleGenerator(nn.Module):
    def __init__(self, feat_dim, hid_dim, n_nodes):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_nodes = n_nodes
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hid_dim), nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.feat_gen = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2), nn.LeakyReLU(),
            nn.Linear(hid_dim * 2, n_nodes * feat_dim),
        )

    def forward(self, x):
        enc = self.encoder(x).mean(dim=0, keepdim=True)
        feats = torch.tanh(self.feat_gen(enc).reshape(self.n_nodes, self.feat_dim))
        return feats


# ============================================================
# Main attack function — called by run.py
# ============================================================

def _attack_single(model, data, targets, fs, cfg, n_feat, device):
    """Single attempt to attack one graph.

    Returns (success: bool, best_x0: Tensor | None) — best_x0 is the
    generator output with the lowest CW margin observed across epochs,
    used by the hybrid diffusion restart as a non-zero anchor.
    """
    kappa = cfg["kappa"]
    gen = SimpleGenerator(n_feat, cfg["gen_hid_dim"], cfg["node_budget"]).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=cfg["gen_lr"])

    best_x0 = None
    best_margin = float("inf")
    true_label = data.y.item()

    for epoch in range(cfg["attack_epochs"]):
        gen.train()
        opt.zero_grad()
        node_feats = gen(data.x)

        perturbed = construct_perturbed_graph(data, node_feats, targets, fs)
        perturbed.batch = torch.zeros(perturbed.num_nodes,
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(perturbed)
            pred = get_prediction(logits)

        # Track best x0 by CW margin (lower = closer to flipping)
        if logits.dim() == 1:
            lvec = logits.unsqueeze(0)
        else:
            lvec = logits
        if lvec.size(-1) > 1:
            true_l = lvec[0, true_label].item()
            mask = torch.ones(lvec.size(-1), dtype=torch.bool, device=device)
            mask[true_label] = False
            max_other = lvec[0, mask].max().item()
            margin = true_l - max_other
            if margin < best_margin:
                best_margin = margin
                best_x0 = node_feats.detach().clone()

        if pred != true_label:
            return True, best_x0

        if cfg["grad_method"] == "cge":
            _, grad = estimate_gradient_cge(
                model, data, node_feats, targets,
                cfg["sigma"], kappa, fs, device, cfg["loss_type"])
        else:
            _, grad = estimate_gradient_rgf(
                model, data, node_feats, targets,
                cfg["sigma"], 100, kappa, fs, device, cfg["loss_type"])
        node_feats.backward(grad)
        opt.step()
    return False, best_x0


# ============================================================
# Hybrid diffusion-as-restart (Option B / N23-v3)
# ============================================================

def _make_alpha_schedule_linear(T, alpha_min, alpha_max, device):
    """Bounded linear alpha schedule, t=0 clean, t=T noisy."""
    return torch.linspace(alpha_max, alpha_min, T + 1, device=device)


def _ddim_predict_x0(x_t, eps, alpha_bar_t):
    return (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)


def _ddim_step(x_t, eps_pred, alpha_bar_t, alpha_bar_prev):
    x0 = _ddim_predict_x0(x_t, eps_pred, alpha_bar_t)
    return (torch.sqrt(alpha_bar_prev) * x0
            + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred)


def _diffusion_restart(model, data, targets, fs, x_init, cfg, device, seed=0):
    """AdvAD-style bounded DDIM anchored at x_init (best baseline guess).

    Forward:   x_T = sqrt(alpha_T) * x_init + sqrt(1 - alpha_T) * eps_0
    Reverse:   each step t, ZO estimate score = grad_x0 CW; update eps_hat per
               AMG: eps_hat = eps_0 - guidance * sqrt(1 - alpha_t) * score;
               DDIM transition to x_{t-1}; quick-check x0_pred for flip.

    Different from N23-v1/v2 because x_init != 0 — the trajectory has a
    meaningful anchor so the ZO score has somewhere to denoise toward and
    away from. Returns True if any intermediate x0 flips the prediction.
    """
    torch.manual_seed(seed)
    T = cfg["diff_T_steps"]
    kappa = cfg["kappa"]

    alphas = _make_alpha_schedule_linear(
        T, cfg["diff_alpha_min"], cfg["diff_alpha_max"], device)
    eps_0 = torch.randn_like(x_init)
    alpha_T = alphas[T]
    x_t = (torch.sqrt(alpha_T) * x_init
           + torch.sqrt(1.0 - alpha_T) * eps_0)

    for step in range(T, 0, -1):
        alpha_bar_t = alphas[step]
        alpha_bar_prev = alphas[step - 1]
        x0_pred = _ddim_predict_x0(x_t, eps_0, alpha_bar_t)

        # Quick success check
        perturbed = construct_perturbed_graph(data, x0_pred, targets, fs)
        perturbed.batch = torch.zeros(perturbed.num_nodes,
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(perturbed)
        if get_prediction(logits) != data.y.item():
            return True

        # ZO CGE score at x0_pred (gradient of CW w.r.t. x0)
        _, score = estimate_gradient_cge(
            model, data, x0_pred, targets,
            cfg["sigma"], kappa, fs, device, cfg["loss_type"])

        # AMG (CW minimization → subtract score)
        eps_hat = eps_0 - cfg["diff_guidance"] * torch.sqrt(1.0 - alpha_bar_t) * score
        x_t = _ddim_step(x_t, eps_hat, alpha_bar_t, alpha_bar_prev)

    # Final check
    perturbed = construct_perturbed_graph(data, x_t, targets, fs)
    perturbed.batch = torch.zeros(perturbed.num_nodes,
                                  dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(perturbed)
    return get_prediction(logits) != data.y.item()


def _logit_margin(logits, true_label, device):
    """Unclamped CW margin: true_logit - max_other. Lower (more negative) = flip."""
    lv = logits.unsqueeze(0) if logits.dim() == 1 else logits
    true_l = lv[0, true_label]
    mask = torch.ones(lv.size(-1), dtype=torch.bool, device=device)
    mask[true_label] = False
    return (true_l - lv[0, mask].max()).item()


def _attack_joint_diffusion(model, data, fs, k_budget, cfg, n_feat,
                            device, seed=0, x_init=None):
    """N24 — joint diffusion over (X_inj, A_sv) with cardinality budget on A_sv.

    State per step:
      x0_pred ∈ R^{m × d}        — continuous, evolves via bounded DDIM + AMG
      edge_mask ∈ {0,1}^N        — discrete, ||·||_1 <= k_budget, evolves via
                                   greedy one-edge-toggle scoring; projection
                                   keeps only edges with positive marginal
                                   value, floored at 1, capped at k_budget

    Per-step cost: N forwards (edge scoring) + 2·m·d forwards (CGE feature score).
    Total per graph: T_steps × (N + 2·m·d) victim queries.

    Initialization:
      edge_mask = top-k by original-graph degree (warm start; known good init
                  for injection-attack edge selection)
      x0        = x_init (e.g., baseline best-margin guess) or zeros
    """
    torch.manual_seed(seed)
    N = data.num_nodes
    m = cfg["node_budget"]
    T = cfg["diff_T_steps"]
    kappa = cfg["kappa"]
    true_label = data.y.item()
    k = max(1, min(k_budget, N))

    # Warm-start edge mask: top-k by degree
    degrees = torch.bincount(data.edge_index[0], minlength=N).float()
    top_idx = torch.topk(degrees, k=k).indices
    edge_mask = torch.zeros(N, dtype=torch.bool, device=device)
    edge_mask[top_idx] = True

    # Init x0
    if x_init is not None and x_init.shape == (m, n_feat):
        x0 = x_init.clone().to(device)
    else:
        x0 = torch.zeros(m, n_feat, device=device)

    eps_0 = torch.randn(m, n_feat, device=device)
    alphas = _make_alpha_schedule_linear(
        T, cfg["diff_alpha_min"], cfg["diff_alpha_max"], device)
    alpha_T = alphas[T]
    x_t = (torch.sqrt(alpha_T) * x0
           + torch.sqrt(1.0 - alpha_T) * eps_0)

    for step in range(T, 0, -1):
        alpha_bar_t = alphas[step]
        alpha_bar_prev = alphas[step - 1]
        x0_pred = _ddim_predict_x0(x_t, eps_0, alpha_bar_t)

        # Current targets from mask
        cur_targets = torch.where(edge_mask)[0]
        if cur_targets.numel() == 0:
            # Should not happen (we init with k>=1), but guard
            edge_mask[top_idx[0]] = True
            cur_targets = torch.where(edge_mask)[0]

        # Quick check at current state
        perturbed = construct_perturbed_graph(data, x0_pred, cur_targets, fs)
        perturbed.batch = torch.zeros(perturbed.num_nodes,
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            cur_logits = model(perturbed)
        if get_prediction(cur_logits) != true_label:
            return True

        # Edge score: toggle each of N edges, evaluate, project to top-k
        cur_margin = _logit_margin(cur_logits, true_label, device)
        edge_value = torch.full((N,), -1e9, device=device)  # higher = better to be IN

        toggle_data = []
        valid_j = []
        for j in range(N):
            new_mask = edge_mask.clone()
            new_mask[j] = ~new_mask[j]
            new_targets = torch.where(new_mask)[0]
            if new_targets.numel() == 0:
                continue
            toggle_data.append(
                construct_perturbed_graph(data, x0_pred, new_targets, fs))
            valid_j.append(j)

        if toggle_data:
            losses, _ = batch_loss(
                model, toggle_data, true_label, kappa, device,
                loss_type=cfg["loss_type"], clean_data=data)
            for j, loss_after in zip(valid_j, losses):
                # value of edge j being IN the mask:
                #   was-in (mask[j]=1, toggle removes): value = loss_without - cw_current
                #     (positive: removing it hurts → it's useful)
                #   was-out (mask[j]=0, toggle adds): value = cw_current - loss_with
                #     (positive: adding it helps → it should be in)
                if edge_mask[j].item():
                    edge_value[j] = loss_after - cur_margin
                else:
                    edge_value[j] = cur_margin - loss_after

        # Project: keep only edges with positive value (genuinely helpful),
        # up to the cardinality budget k. Floor at 1 to keep the injection
        # connected. This makes k a true upper bound, not an equality.
        order = torch.argsort(edge_value, descending=True)
        n_useful = int((edge_value > 0).sum().item())
        n_keep = min(k, max(1, n_useful))
        new_mask = torch.zeros(N, dtype=torch.bool, device=device)
        new_mask[order[:n_keep]] = True
        edge_mask = new_mask

        # Quick re-check after edge update (cheap: 1 forward)
        new_targets = torch.where(edge_mask)[0]
        perturbed = construct_perturbed_graph(data, x0_pred, new_targets, fs)
        perturbed.batch = torch.zeros(perturbed.num_nodes,
                                      dtype=torch.long, device=device)
        with torch.no_grad():
            logits2 = model(perturbed)
        if get_prediction(logits2) != true_label:
            return True

        # Feature ZO score under updated mask
        _, score = estimate_gradient_cge(
            model, data, x0_pred, new_targets,
            cfg["sigma"], kappa, fs, device, cfg["loss_type"])
        eps_hat = (eps_0
                   - cfg["diff_guidance"]
                   * torch.sqrt(1.0 - alpha_bar_t) * score)
        x_t = _ddim_step(x_t, eps_hat, alpha_bar_t, alpha_bar_prev)

    # Final check
    final_targets = torch.where(edge_mask)[0]
    perturbed = construct_perturbed_graph(data, x_t, final_targets, fs)
    perturbed.batch = torch.zeros(perturbed.num_nodes,
                                  dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(perturbed)
    return get_prediction(logits) != true_label


def run_attack(model, test_graphs, device):
    """Attack with multi-restart: try N_RESTARTS random inits, succeed if any works."""
    cfg = CONFIG
    n_feat = test_graphs[0].x.size(1)
    # (lr, grad_method, epochs) for each restart
    RESTART_CONFIGS = [
        (5e-3, "cge", 50),
        (1e-2, "cge", 50),
        (2e-2, "cge", 75),
        (5e-3, "cge", 100),
        (1e-2, "cge", 100),
    ]
    n_success = 0
    n_baseline_success = 0
    n_diffusion_tried = 0
    n_diffusion_success = 0
    n_joint_tried = 0
    n_joint_success = 0

    for data in test_graphs:
        data = data.to(device)
        fs = cfg["feat_scale"]
        if fs == "auto":
            fs = math.sqrt(data.num_nodes)
        elif fs == "auto_x2":
            fs = 2.0 * math.sqrt(data.num_nodes)
        elif fs == "auto_x3":
            fs = 3.0 * math.sqrt(data.num_nodes)
        elif fs == "auto_linear":
            fs = float(data.num_nodes)

        if cfg["edge_strategy"] == "full":
            targets = torch.arange(data.num_nodes, device=device)
        elif cfg["edge_strategy"] == "spectral":
            targets = select_targets_spectral(
                data, cfg["node_budget"], cfg["spectral_top_k_eig"], device)
        else:
            targets = select_targets_topk(data, cfg["node_budget"])

        success = False
        # Track the best x0 across all baseline restarts, for diffusion anchor
        global_best_x0 = None
        global_best_margin = float("inf")

        for restart, (lr, gm, ep) in enumerate(RESTART_CONFIGS):
            torch.manual_seed(restart * 1000 + data.num_nodes)
            cfg_copy = cfg.copy()
            cfg_copy["gen_lr"] = lr
            cfg_copy["grad_method"] = gm
            cfg_copy["attack_epochs"] = ep
            ok, x0 = _attack_single(model, data, targets, fs, cfg_copy, n_feat, device)
            if ok:
                success = True
                n_baseline_success += 1
                break
            if x0 is not None:
                # Re-evaluate this x0's margin to compare across restarts
                p = construct_perturbed_graph(data, x0, targets, fs)
                p.batch = torch.zeros(p.num_nodes, dtype=torch.long, device=device)
                with torch.no_grad():
                    lg = model(p)
                if lg.dim() == 1:
                    lg = lg.unsqueeze(0)
                if lg.size(-1) > 1:
                    tl = lg[0, data.y.item()].item()
                    mk = torch.ones(lg.size(-1), dtype=torch.bool, device=device)
                    mk[data.y.item()] = False
                    mo = lg[0, mk].max().item()
                    mg = tl - mo
                    if mg < global_best_margin:
                        global_best_margin = mg
                        global_best_x0 = x0

        # Hybrid: if all baseline restarts failed, try diffusion from best x0
        if (not success and cfg.get("use_diffusion_restart", False)
                and global_best_x0 is not None):
            n_diffusion_tried += 1
            for seed in range(cfg["diff_n_seeds"]):
                if _diffusion_restart(
                        model, data, targets, fs, global_best_x0, cfg,
                        device, seed=seed * 7 + data.num_nodes):
                    success = True
                    n_diffusion_success += 1
                    break

        # N24 joint diffusion (X + A_sv with cardinality budget) — last chance
        if not success and cfg.get("use_joint_diffusion", False):
            n_joint_tried += 1
            # Resolve edge_budget = avg_deg per graph (or fixed int)
            eb = cfg["joint_edge_budget"]
            if eb == "avg_deg":
                avg_deg = (2.0 * data.num_edges / data.num_nodes
                           if data.num_nodes else 1.0)
                k_budget = max(1, int(math.ceil(avg_deg)))
            else:
                k_budget = int(eb)
            for seed in range(cfg["diff_n_seeds"]):
                if _attack_joint_diffusion(
                        model, data, fs, k_budget, cfg, n_feat,
                        device, seed=seed * 11 + data.num_nodes,
                        x_init=global_best_x0):
                    success = True
                    n_joint_success += 1
                    break

        if success:
            n_success += 1

    print(f"  [hybrid] baseline solo: {n_baseline_success}, "
          f"diff tried: {n_diffusion_tried}, diff saved: {n_diffusion_success}, "
          f"joint tried: {n_joint_tried}, joint saved: {n_joint_success}")
    asr = n_success / len(test_graphs) if test_graphs else 0.0
    return asr
