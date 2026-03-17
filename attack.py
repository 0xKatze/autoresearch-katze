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
    "feat_scale": "auto",       # float or "auto" (sqrt(N))
    "sigma": 1e-3,              # ZOO smoothing parameter
    "gen_lr": 5e-3,             # generator learning rate
    "attack_epochs": 50,        # optimization steps per graph
    "grad_method": "cge",       # "rgf" or "cge"
    "loss_type": "hybrid",      # "cw" | "cosine" | "hybrid"
    "edge_strategy": "spectral",# "topk" | "spectral"
    "node_budget": 1,           # injected nodes per graph
    "kappa": -0.001,            # CW loss margin
    "gen_hid_dim": 128,         # generator hidden dim
    "spectral_top_k_eig": 10,  # eigenvalues for spectral edge selection
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
    inj = torch.arange(n_orig, n_orig + m, device=device)
    new_edges = torch.cat([
        torch.stack([inj, target_indices], dim=0),
        torch.stack([target_indices, inj], dim=0),
    ], dim=1)
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
    """Single attempt to attack one graph."""
    kappa = cfg["kappa"]
    gen = SimpleGenerator(n_feat, cfg["gen_hid_dim"], cfg["node_budget"]).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=cfg["gen_lr"])

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
        if pred != data.y.item():
            return True

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
    return False


def run_attack(model, test_graphs, device):
    """Attack with multi-restart: try N_RESTARTS random inits, succeed if any works."""
    cfg = CONFIG
    n_feat = test_graphs[0].x.size(1)
    N_RESTARTS = 3
    n_success = 0

    for data in test_graphs:
        data = data.to(device)
        fs = cfg["feat_scale"]
        if fs == "auto":
            fs = math.sqrt(data.num_nodes)

        if cfg["edge_strategy"] == "spectral":
            targets = select_targets_spectral(
                data, cfg["node_budget"], cfg["spectral_top_k_eig"], device)
        else:
            targets = select_targets_topk(data, cfg["node_budget"])

        success = False
        for restart in range(N_RESTARTS):
            torch.manual_seed(restart * 1000 + data.num_nodes)
            if _attack_single(model, data, targets, fs, cfg, n_feat, device):
                success = True
                break

        if success:
            n_success += 1

    asr = n_success / len(test_graphs) if test_graphs else 0.0
    return asr
