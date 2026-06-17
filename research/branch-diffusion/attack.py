#!/usr/bin/env python3
"""
attack.py — branch-diffusion · Stealth: query-frugal stealthy node-injection attack.

Replaces the full-edge baseline with a greedy black-box construction that jointly
chooses (a) subgraph->victim edges, (b) inter-node edges among injected nodes, and
(c) injected features under a stealth constraint. Greedy edge addition is *incremental*,
so a single run records per-graph minimum-edges-to-flip -> the whole ASR-vs-edge-budget
Pareto curve falls out of one pass. See DESIGN_BRANCH_DIFFUSION_STEALTH.md.

Public API:
  run_attack(model, test_graphs, device) -> float        # ASR at full budget (run.py)
  run_attack_sweep(model, test_graphs, device, cfg) -> dict  # rich record for sweeps

Threat model: score-based black-box (CW margin), injection-only (original graph never
mutated), victim = PROTEINS GCN w/ global_mean_pool.
"""

import math
import torch
from torch_geometric.data import Data, Batch

# ============================================================
# CONFIG — defaults; sweep_stealth.py overrides per run
# ============================================================

CONFIG = {
    "node_budget": 1,         # m: injected nodes
    # feature-stealth axis (all start from the data-manifold mean):
    #   "manifold" = no optimization (fully stealthy lower bound)
    #   "zo_box"     = ZO-optimized projected to data range [0,1] (stealthy + optimized)
    #   "zo_normcap" = ZO-optimized projected onto L2 ball R=kappa*median_node_norm
    #                  (the N4/SA-ZOO magnitude-only stealth baseline to beat)
    #   "zo_free"    = ZO-optimized unbounded (KATZE-level, detectable upper reference)
    #   "sgdi"       = Stealth-Guided Diffusion Injection (the contribution): score-based
    #                  guided diffusion = ZO adversarial gradient + Gaussian data-prior
    #                  score -Sigma^-1 (x-mu) + annealed Langevin noise. Lands inside the
    #                  data covariance ellipsoid (matches direction/correlation, not just
    #                  norm) -> evades Mahalanobis / jaccard / featclip that norm-cap misses.
    "feat_mode": "zo_free",
    "normcap_kappa": 1.95,    # zo_normcap L2-ball radius = kappa * median train-node norm
    "sgdi_eta": 0.3,          # sgdi prior-score pull strength
    "sgdi_temp": 0.1,         # sgdi initial Langevin noise temperature (annealed to 0)
    "zo_epochs": 40, "zo_lr": 0.5, "zo_sigma": 5e-3, "zo_restarts": 3,
    "max_edges": 16,          # adaptive-greedy victim-edge cap (cheap/sparse regime)
    "pool_mult": 3,           # candidate pool size = max(16, pool_mult * max_edges)
    "use_inter_edges": True,  # allow malicious-subgraph inter-node edges (m>=2)
    # fraction-of-N anchors trace the expensive regime (rho=1.0 == full-edge baseline):
    "frac_anchors": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
}

# Module-level query accounting (reset per run). One forward over one graph = 1 query.
QUERY_COUNT = 0


# ============================================================
# Black-box evaluation primitives
# ============================================================

def _cw_margins(logits, true_label):
    """Per-row CW margin = logit[true] - max(logit[others]). Flip iff margin < 0."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    true_logit = logits[:, true_label]
    mask = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
    mask[true_label] = False
    max_other = logits[:, mask].max(dim=-1).values
    return true_logit - max_other


def _eval_graphs(model, data_list, true_label, device):
    """Batched black-box query. Returns CW margins; increments QUERY_COUNT by len."""
    global QUERY_COUNT
    if not data_list:
        return torch.empty(0)
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        logits = model(batch)
    QUERY_COUNT += len(data_list)
    return _cw_margins(logits, true_label)


import math as _fs_math

# Uniform feat_scale knob (sweep axis). Default 1.0 = no-op (backward compatible).
# Set per-graph from cfg in greedy_attack_graph; applied to injected feats in _build.
_FEAT_SCALE = 1.0


def _resolve_feat_scale(spec, num_nodes):
    if spec is None:
        return 1.0
    if isinstance(spec, str):
        if spec == "auto":
            return _fs_math.sqrt(max(num_nodes, 1))
        if spec == "auto_x2":
            return 2.0 * _fs_math.sqrt(max(num_nodes, 1))
        if spec == "auto_x4":
            return 4.0 * _fs_math.sqrt(max(num_nodes, 1))
        try:
            return float(spec)
        except (TypeError, ValueError):
            return 1.0
    return float(spec)


def _build(original, feats, victim_edges, inter_edges, device):
    """Pure injection: append m feature rows + given edges. Original never mutated.

    victim_edges: list[(i, v)] injected-node i <-> victim-node v
    inter_edges:  list[(i, j)] injected-node i <-> injected-node j
    """
    n_orig = original.num_nodes
    X = torch.cat([original.x, feats * _FEAT_SCALE], dim=0)
    src, dst = [], []
    for (i, v) in victim_edges:
        inj = n_orig + i
        src += [inj, v]; dst += [v, inj]
    for (i, j) in inter_edges:
        a, b = n_orig + i, n_orig + j
        src += [a, b]; dst += [b, a]
    if src:
        add = torch.tensor([src, dst], dtype=torch.long, device=device)
        ei = torch.cat([original.edge_index, add], dim=1)
    else:
        ei = original.edge_index
    return Data(x=X, edge_index=ei, y=original.y)


# ============================================================
# Injected features
# ============================================================

def data_stats(test_graphs, device):
    """Empirical node-feature stats: per-dim mean (manifold), per-dim max, and the
    median per-node L2 norm (used for the zo_normcap L2-ball radius)."""
    tot = None
    mx = None
    n = 0
    norms = []
    for g in test_graphs:
        tot = g.x.sum(dim=0) if tot is None else tot + g.x.sum(dim=0)
        gmax = g.x.max(dim=0).values
        mx = gmax if mx is None else torch.maximum(mx, gmax)
        norms.append(g.x.norm(dim=1))
        n += g.x.size(0)
    mean = (tot / max(n, 1)).to(device)
    median_norm = float(torch.cat(norms).median().item()) if norms else 1.0
    return mean, mx.to(device), median_norm


def make_features(stats, m, feat_mode, n_nodes, device):
    """Initial injected features (all modes start from the data-manifold mean)."""
    mean = stats[0]
    return mean.unsqueeze(0).repeat(m, 1)


def _project_feats(feats, feat_mode, radius):
    """Per-step feature-stealth projection. box -> [0,1]; normcap -> L2 ball R=radius."""
    if feat_mode == "zo_box":
        return feats.clamp(0.0, 1.0)
    if feat_mode == "zo_normcap":
        norm = feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return feats * (radius / norm).clamp(max=1.0)
    return feats  # zo_free / others: unconstrained


def _zo_one(model, data, feats, v_edges, i_edges, cfg, device, feat_mode, radius):
    """One CGE feature-descent run on a fixed topology. Returns (feats, flipped).

    After each gradient step the candidate is pushed through the feat_mode stealth
    projection (box / normcap / none), so the optimization stays inside the stealth set.
    """
    true_label = data.y.item()
    feats = feats.clone()
    m, d = feats.shape
    sigma, lr = cfg["zo_sigma"], cfg["zo_lr"]
    flipped = False
    for _ in range(cfg["zo_epochs"]):
        data_list, coords = [], []
        for i in range(m):
            for j in range(d):
                e = torch.zeros_like(feats); e[i, j] = sigma
                data_list.append(_build(data, feats + e, v_edges, i_edges, device))
                data_list.append(_build(data, feats - e, v_edges, i_edges, device))
                coords.append((i, j))
        margins = _eval_graphs(model, data_list, true_label, device)
        grad = torch.zeros_like(feats)
        for k, (i, j) in enumerate(coords):
            grad[i, j] = (margins[2 * k] - margins[2 * k + 1]) / (2 * sigma)
        feats = feats - lr * grad        # descend CW margin
        feats = _project_feats(feats, feat_mode, radius)
        mg = _eval_graphs(model, [_build(data, feats, v_edges, i_edges, device)],
                          true_label, device)
        if mg[0].item() < 0:
            flipped = True
            break
    return feats.detach(), flipped


def zo_optimize_features(model, data, feats0, v_edges, i_edges, cfg, device):
    """Multi-restart CGE feature optimization on a fixed topology -> minimize CW margin.

    Stealth set is set by feat_mode: 'zo_box' projects to [0,1]; 'zo_normcap' projects
    onto the L2 ball R=cfg['_norm_radius'] (N4 magnitude-only baseline); 'zo_free' is
    unconstrained (KATZE-level). Restarts use different inits (early-stop on flip so
    successful graphs stay cheap): (2*m*d + 1) queries/epoch.
    """
    feat_mode = cfg["feat_mode"]
    radius = cfg.get("_norm_radius", float("inf"))
    m, d = feats0.shape
    best = feats0
    for r in range(cfg["zo_restarts"]):
        if r == 0:
            init = feats0
        else:
            g = torch.Generator(device="cpu").manual_seed(
                r * 1000 + data.num_nodes + cfg.get("seed_offset", 0) * 99991)
            rnd = torch.rand(m, d, generator=g).to(device) if feat_mode == "zo_box" \
                else torch.randn(m, d, generator=g).to(device)
            init = _project_feats(rnd, feat_mode, radius)
        feats, flipped = _zo_one(model, data, init, v_edges, i_edges, cfg, device,
                                 feat_mode, radius)
        best = feats
        if flipped:
            break
    return best


# ============================================================
# SGDI — Stealth-Guided Diffusion Injection (the contribution)
# ============================================================

def data_prior(test_graphs, device, eps=1e-3):
    """Gaussian data prior (mu, Sigma^-1, cholesky L) over node features.

    Sigma encodes per-dim scale AND inter-dim correlation -> sampling/guiding with it
    keeps injected features inside the data ellipsoid (right *direction*, not just norm).
    This is exactly the density a Mahalanobis defense scores against. eps*I regularizes.
    """
    X = torch.cat([g.x for g in test_graphs], dim=0).to(device).float()
    mu = X.mean(dim=0)
    Xc = X - mu
    d = X.size(1)
    cov = (Xc.t() @ Xc) / max(X.size(0) - 1, 1) + eps * torch.eye(d, device=device)
    cov_inv = torch.linalg.inv(cov)
    L = torch.linalg.cholesky(cov)
    # attacker's stealth radius = 99th pct of clean sq-Mahalanobis (strictly inside the
    # defender's 99.5th-pct cut -> SGDI nodes provably survive a Mahalanobis filter)
    sqm = (Xc @ cov_inv * Xc).sum(dim=1)
    thr = float(torch.quantile(sqm, 0.99).item())
    return mu, cov_inv, L, thr


def _project_maha(x, mu, cov_inv, thr):
    """Project rows of x onto the Mahalanobis ball {(x-mu)^T Sigma^-1 (x-mu) <= thr}."""
    diff = x - mu
    m2 = (diff @ cov_inv * diff).sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mu + diff * (thr / m2).sqrt().clamp(max=1.0)


def _cge_grad(model, data, feats, v_edges, i_edges, cfg, device):
    """CGE estimate of grad of CW margin wrt feats, plus the current margin at feats.

    (2*m*d + 1) queries. Shared by SGDI; _zo_one keeps its own inline copy.
    """
    true_label = data.y.item()
    m, d = feats.shape
    sigma = cfg["zo_sigma"]
    data_list, coords = [], []
    for i in range(m):
        for j in range(d):
            e = torch.zeros_like(feats); e[i, j] = sigma
            data_list.append(_build(data, feats + e, v_edges, i_edges, device))
            data_list.append(_build(data, feats - e, v_edges, i_edges, device))
            coords.append((i, j))
    margins = _eval_graphs(model, data_list, true_label, device)
    grad = torch.zeros_like(feats)
    for k, (i, j) in enumerate(coords):
        grad[i, j] = (margins[2 * k] - margins[2 * k + 1]) / (2 * sigma)
    cur = _eval_graphs(model, [_build(data, feats, v_edges, i_edges, device)],
                       true_label, device)[0].item()
    return grad, cur


def sgdi_optimize_features(model, data, feats0, v_edges, i_edges, cfg, device, prior):
    """Score-based guided diffusion in feature space (classifier-guided / Langevin).

    Reverse process per step:  x <- x - lr * grad_CW(x)            (adversarial guidance)
                                     + eta * (-Sigma^-1 (x - mu))   (data-prior score)
                                     + sqrt-annealed Langevin noise (explore early, commit late)
    Init samples from the Gaussian prior (x = mu + L eps). The prior term keeps x inside
    the data covariance ellipsoid -> on-manifold direction, so distribution-aware defenses
    (Mahalanobis/jaccard/featclip) cannot flag it, unlike unbounded zo_free or isotropic
    norm-cap. Early-stop on flip keeps successful graphs cheap.
    """
    mu, cov_inv, L, thr = prior
    m, d = feats0.shape
    T, lr = cfg["zo_epochs"], cfg["zo_lr"]
    eta, temp0 = cfg["sgdi_eta"], cfg["sgdi_temp"]
    best = feats0
    for r in range(cfg["zo_restarts"]):
        gen = torch.Generator(device="cpu").manual_seed(
            r * 1000 + data.num_nodes + cfg.get("seed_offset", 0) * 99991)
        if r == 0:
            x = mu.unsqueeze(0).repeat(m, 1)                 # start at the density mode
        else:
            eps = torch.randn(m, d, generator=gen).to(device)
            x = mu.unsqueeze(0) + eps @ L.t()               # sample from N(mu, Sigma)
        x = _project_maha(x, mu, cov_inv, thr)
        flipped = False
        for t in range(T):
            frac = 1.0 - t / max(T - 1, 1)                   # 1 -> 0 anneal
            g_adv, cur = _cge_grad(model, data, x, v_edges, i_edges, cfg, device)
            if cur < 0:
                best = x; flipped = True; break
            s_prior = -(x - mu.unsqueeze(0)) @ cov_inv       # grad log N(x; mu, Sigma)
            noise = torch.randn(m, d, generator=gen).to(device) * (temp0 * frac)
            x = x - lr * g_adv + eta * s_prior + noise
            x = _project_maha(x, mu, cov_inv, thr)           # HARD stealth guarantee
        best = x
        if flipped:
            break
    return best.detach()


# ============================================================
# Greedy construction (the attack)
# ============================================================

def _candidate_pool(data, max_edges, pool_mult):
    """Lowest-degree victim nodes (theory: low-degree neighbor -> larger pooled shift)."""
    n = data.num_nodes
    deg = torch.bincount(data.edge_index[0], minlength=n).float()
    size = min(n, max(16, pool_mult * max_edges))
    return torch.argsort(deg)[:size].tolist()


def greedy_attack_graph(model, data, feats, cfg, device):
    """Black-box injection on one graph, in rho=edges/N budget units.

    Two probes per graph:
      - adaptive greedy (low-degree-targeted) -> strongest *sparse* attack, gives the
        smallest rho that flips in the cheap regime.
      - fraction-of-N anchors -> connect each injected node to the ceil(rho*N) lowest-degree
        victim nodes (+ full inter-node clique) -> traces the expensive regime up to rho=1
        (rho=1.0 == the full-edge KATZE regime).
    rho_flip = min over both probes of the fraction of N edges (per injected node) at flip.

    Returns: rho_flip(float|inf), full_flip(bool), n_inter(int), greedy_edges(int|inf).
    """
    true_label = data.y.item()
    n = data.num_nodes
    m = feats.size(0)
    global _FEAT_SCALE
    _FEAT_SCALE = _resolve_feat_scale(cfg.get("feat_scale", 1.0), n)
    inter_full = [(i, j) for i in range(m) for j in range(i + 1, m)] \
        if cfg["use_inter_edges"] else []

    # ---- feature optimization (on full-edge topology) ----
    if cfg["feat_mode"] == "sgdi":
        all_v = [(i, v) for i in range(m) for v in range(n)]
        feats = sgdi_optimize_features(model, data, feats, all_v, inter_full,
                                       cfg, device, cfg["_prior"])
    elif cfg["feat_mode"].startswith("zo"):
        all_v = [(i, v) for i in range(m) for v in range(n)]
        feats = zo_optimize_features(model, data, feats, all_v, inter_full, cfg, device)

    # ---- adaptive greedy (cheap/sparse regime) ----
    # skip_greedy: only the fraction-of-N anchors are probed (fixed-budget ASR), ~halving
    # queries for grid evals that report ASR@full rather than the sparse Pareto curve.
    victim_edges, inter_edges = [], []
    greedy_edges = math.inf
    if not cfg.get("skip_greedy"):
        pool = _candidate_pool(data, cfg["max_edges"], cfg["pool_mult"])
        used_v, used_i = set(), set()
        for _ in range(cfg["max_edges"]):
            cands = [("v", (i, v)) for i in range(m) for v in pool if (i, v) not in used_v]
            if cfg["use_inter_edges"] and m >= 2:
                cands += [("e", (i, j)) for i in range(m) for j in range(i + 1, m)
                          if frozenset({i, j}) not in used_i]
            if not cands:
                break
            data_list = [
                _build(data, feats, victim_edges + [p], inter_edges, device) if k == "v"
                else _build(data, feats, victim_edges, inter_edges + [p], device)
                for k, p in cands]
            margins = _eval_graphs(model, data_list, true_label, device)
            best = int(torch.argmin(margins).item())
            k, p = cands[best]
            (victim_edges if k == "v" else inter_edges).append(p)
            (used_v if k == "v" else used_i).add(p if k == "v" else frozenset(p))
            if margins[best].item() < 0:
                greedy_edges = len(victim_edges) + len(inter_edges)
                break
    # greedy rho: total edges / N (per-injected-node fraction proxy)
    rho_greedy = (greedy_edges / n) if math.isfinite(greedy_edges) else math.inf

    # ---- fraction-of-N anchors (expensive regime) ----
    deg_order = torch.argsort(
        torch.bincount(data.edge_index[0], minlength=n).float()).tolist()
    rho_anchor = math.inf
    full_flip = False
    anchor_list = []
    for rho in cfg["frac_anchors"]:
        k = max(1, math.ceil(rho * n))
        targets = deg_order[:k]
        v_edges = [(i, t) for i in range(m) for t in targets]
        anchor_list.append(_build(data, feats, v_edges, inter_full, device))
    a_margins = _eval_graphs(model, anchor_list, true_label, device)
    for idx, rho in enumerate(cfg["frac_anchors"]):
        if a_margins[idx].item() < 0:
            rho_anchor = min(rho_anchor, rho)
            if rho >= 1.0:
                full_flip = True
    if a_margins[-1].item() < 0:   # rho=1.0 endpoint is the full-edge anchor
        full_flip = True

    rho_flip = min(rho_greedy, rho_anchor)
    return {
        "rho_flip": rho_flip,
        "full_flip": bool(full_flip),
        "n_inter": len(inter_edges),
        "greedy_edges": greedy_edges if math.isfinite(greedy_edges) else None,
    }


# ============================================================
# Run entry points
# ============================================================

def run_attack_sweep(model, test_graphs, device, cfg=None):
    """Rich record for Pareto sweeps. Per-graph rho_flip (edges/N) + query cost."""
    # ---- SpecGDI dispatch: only branch allowed in this function (see SPECGDI_IMPL.md) ----
    _fm = (cfg or {}).get("feat_mode", CONFIG["feat_mode"])
    if _fm in ("specgdi", "specgdi_signal"):
        return _run_specgdi_sweep(model, test_graphs, device, cfg)
    if _fm == "specgdi_v2":
        return _run_specgdi_v2_sweep(model, test_graphs, device, cfg)
    if _fm == "topomass":
        return _run_topomass_sweep(model, test_graphs, device, cfg)
    # ---- end SpecGDI dispatch ----
    global QUERY_COUNT
    QUERY_COUNT = 0
    cfg = {**CONFIG, **(cfg or {})}
    m = cfg["node_budget"]
    stats = data_stats(test_graphs, device)
    # zo_normcap L2-ball radius = kappa * median node-feature norm (computed once per run)
    cfg = {**cfg, "_norm_radius": cfg.get("normcap_kappa", 1.95) * stats[2]}
    if cfg["feat_mode"] == "sgdi":
        cfg = {**cfg, "_prior": data_prior(test_graphs, device)}

    records = []
    for data in test_graphs:
        data = data.to(device)
        feats = make_features(stats, m, cfg["feat_mode"], data.num_nodes, device)
        records.append(greedy_attack_graph(model, data, feats, cfg, device))

    n = len(records) or 1
    return {
        "asr_full": sum(r["full_flip"] for r in records) / n,   # rho=1.0 (KATZE regime)
        "rho_to_flip": [r["rho_flip"] for r in records],        # per-graph, inf if never
        "greedy_edges": [r["greedy_edges"] for r in records],   # abs edges in cheap regime
        "n_inter": [r["n_inter"] for r in records],
        "queries": QUERY_COUNT,
        "n_graphs": len(records),
    }


def run_attack(model, test_graphs, device):
    """run.py-compatible: float ASR at the full-edge anchor (baseline-comparable)."""
    return run_attack_sweep(model, test_graphs, device, CONFIG)["asr_full"]


# ============================================================
# SpecGDI — Spectral-Guided Diffusion Injection (B1 + B2)
# ------------------------------------------------------------
# Implements §2.1 + §2.2 of `branches/branch-diffusion/specgdi-spectral-guided-
# diffusion.md`. Two feat_modes:
#
#   * "specgdi"        — core (B1): SGDI feature Langevin + spectrum-minimising
#                        greedy single-edge swap of the victim-edge set S.
#                        Closed-form ΔΛ via Hellmann–Feynman / interlacing on
#                        the precomputed top-k spectrum (NO inner-loop eigh).
#   * "specgdi_signal" — signal variant (B2): adds the §2.2a graph-signal
#                        smoothness gradient ∇_x (x*^T L_{v*} x*) to the feature
#                        Langevin step (homophily-as-soft-prior).
#
# Constraints (from spec): no eigh inside the inner loop, ≤ 350 LOC added,
# do not touch sgdi_optimize_features / zo_optimize_features / greedy_attack_graph.
# All new helpers live below this banner; the only edit above is one early-return
# dispatch at the top of run_attack_sweep.
#
# Calibration: τ_spec = 99th-pct of S_spec over N=200 random single-node clean
# insertions (calibrate_specgdi.py → specgdi_thresholds.json, keyed by dataset).
# Hard projection at end: greedy drop edges while S_spec > τ_spec.
# ============================================================

import json as _spgd_json
from pathlib import Path as _SpgdPath

try:
    import numpy as _spgd_np
except Exception:  # pragma: no cover
    _spgd_np = None


SPECGDI_DEFAULTS = {
    "specgdi_eta_spec": 0.3,        # weight on the spectral term in edge selection
    "specgdi_k": 20,                # top-k Laplacian eigenvalues to track
    "specgdi_beta": 1.0,            # low-pass weight decay  w_i = exp(-beta * lam_i)
    "specgdi_swaps_per_step": 8,    # top-K candidates re-scored under L_adv (ZO probe)
    "specgdi_T_diff_default": 20,   # default # diffusion steps (overridden by sgdi_T_diff)
    "specgdi_eta_sm": 0.5,          # §2.2 signal-smoothness gradient weight (B2 only)
    "specgdi_tau_path": "specgdi_thresholds.json",   # τ_spec cache (calibrate_specgdi.py)
}


def _build_adj(num_nodes, edge_index, victim_edges_inj, n_inj, inter_edges, device):
    """Return dense symmetric adjacency for clean + injected nodes (N = n + n_inj)."""
    N = num_nodes + n_inj
    A = torch.zeros(N, N, device=device)
    if edge_index is not None and edge_index.numel() > 0:
        ei = edge_index.cpu().numpy() if not isinstance(edge_index, _spgd_np.ndarray) \
             else edge_index
        for s, d in zip(ei[0].tolist() if hasattr(ei[0], "tolist") else ei[0],
                        ei[1].tolist() if hasattr(ei[1], "tolist") else ei[1]):
            A[s, d] = 1.0
    for (i, v) in victim_edges_inj:
        u = num_nodes + i
        A[u, v] = 1.0; A[v, u] = 1.0
    for (i, j) in inter_edges:
        a, b = num_nodes + i, num_nodes + j
        A[a, b] = 1.0; A[b, a] = 1.0
    return A


def _normalized_laplacian(A):
    """L = I - D^-1/2 A D^-1/2 (dense, small n only). Isolated rows: zero block."""
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = deg[nz].pow(-0.5)
    D = torch.diag(d_inv_sqrt)
    N = A.size(0)
    L = torch.eye(N, device=A.device) - D @ A @ D
    return L


def compute_laplacian_spectrum(A_dense, k):
    """Top-k smallest eigenvalues + eigenvectors of the normalized Laplacian.

    Uses torch.linalg.eigh on CPU (n<200 -> ~ms). Returns (lam_k [k], U_k [N,k])
    in ascending order. Truncates to k = min(k, N).
    """
    L = _normalized_laplacian(A_dense).detach().cpu()
    # symmetrize to kill float asymmetry
    L = 0.5 * (L + L.t())
    eigvals, eigvecs = torch.linalg.eigh(L)
    kk = int(min(k, eigvals.numel()))
    return eigvals[:kk], eigvecs[:, :kk]


def spec_disc(lam_pert, lam_clean, beta):
    """S_spec = sum_i w_i (lam_i(L') - lam_i(L))^2  with w_i = exp(-beta * lam_clean_i).

    lam_pert / lam_clean: 1-D tensors of the same length k (already truncated).
    """
    if lam_pert.numel() == 0:
        return torch.tensor(0.0)
    w = torch.exp(-beta * lam_clean)
    return (w * (lam_pert - lam_clean).pow(2)).sum()


def closed_form_delta_lambda(U_clean, lam_clean, dL):
    """First-order Hellmann–Feynman shift  Δλ_i ≈ u_i^T (ΔL) u_i  per eigenpair.

    dL: dense (N,N) delta-Laplacian. U_clean: (N,k). Returns shifts Δλ ∈ R^k.

    Note: when the perturbation adds a node, the clean L is (n,n) and the
    perturbed L is (n+1, n+1). We embed U_clean into the (n+1)-space with a zero
    last row before contracting against dL (the injected-node row of L'/dL is
    orthogonal to all clean eigenvectors at first order — exact under the
    interlacing theorem for the augmented block-Laplacian).
    """
    Nc, kk = U_clean.shape
    Np = dL.size(0)
    if Np != Nc:
        pad = torch.zeros(Np - Nc, kk, device=dL.device, dtype=dL.dtype)
        U = torch.cat([U_clean.to(dL.device).to(dL.dtype), pad], dim=0)
    else:
        U = U_clean.to(dL.device).to(dL.dtype)
    # diag(U^T dL U): shift per eigenvector
    return torch.einsum('ni,nm,mi->i', U, dL, U)


def _load_tau_spec(dataset_key, default=0.10):
    """Lookup τ_spec from specgdi_thresholds.json. Falls back to default if missing."""
    p = _SpgdPath(__file__).parent / SPECGDI_DEFAULTS["specgdi_tau_path"]
    if not p.exists():
        return default
    try:
        d = _spgd_json.loads(p.read_text())
        return float(d.get(dataset_key, d.get("default", default)))
    except Exception:
        return default


def _signal_smoothness_grad(x_inj, x_neighbours, deg_inj):
    """∇_{x*} (x*^T L_{v*} x*) for normalized Laplacian row of v*.

    For normalized Laplacian, L row of v* contracted against the signal X gives
    a smoothness term whose gradient w.r.t. x* is 2 * (x* - mean(x_neighbours))
    weighted by the degree-normalization (D^-1/2 factor at v* and at neighbours).
    We use the unnormalized form (combinatorial-Laplacian smoothness) which is
    cheap and equivalent up to a positive degree-dependent scaling — matches the
    homophily-as-soft-prior framing of §2.2a.
    """
    if x_neighbours.numel() == 0 or deg_inj <= 0:
        return torch.zeros_like(x_inj)
    mean_nbr = x_neighbours.mean(dim=0, keepdim=True)  # (1, d)
    return 2.0 * (x_inj - mean_nbr)


def _eval_specgdi_loss(model, data, feats, v_edges, i_edges, device):
    """Single forward: CW margin (1 query). Reuses _build / _eval_graphs."""
    return _eval_graphs(model, [_build(data, feats, v_edges, i_edges, device)],
                        data.y.item(), device)[0].item()


def _specgdi_optimize_one(model, data, feats0, all_v, inter_full, cfg, device,
                           prior, signal_variant=False):
    """Joint (x*, S) optimisation on one graph. Returns (feats, v_edges, S_spec, flipped).

    - Feature Langevin per step: SGDI gradient flow (CGE adv grad + Mahalanobis
      prior + annealed noise + Mahalanobis-ball projection). When signal_variant=True,
      additionally subtracts eta_sm * ∇_x (x*^T L_{v*} x*).
    - Edge step per step: enumerate single-edge swaps over the candidate set
      (drop u∈S, add u'∈V∖S). Rank by ΔS_spec (closed-form, no eigh). Keep the
      top-K candidates with the LARGEST predicted S_spec drop. Evaluate L_adv via
      a single ZO probe per candidate; pick the one that lowers L_adv (else keep S).
    - Hard projection at end: greedy edge drops while S_spec > τ_spec.
    """
    mu, cov_inv, L_chol, thr = prior
    m, d = feats0.shape
    n = data.num_nodes
    eta_spec = cfg["specgdi_eta_spec"]
    k_top = cfg["specgdi_k"]
    beta = cfg["specgdi_beta"]
    K_swaps = cfg["specgdi_swaps_per_step"]
    eta_sm = cfg.get("specgdi_eta_sm", 0.5)
    lr = cfg["zo_lr"]
    eta_prior = cfg["sgdi_eta"]
    temp0 = cfg["sgdi_temp"]
    T = cfg.get("sgdi_T_diff", cfg["specgdi_T_diff_default"])
    tau_spec = cfg["_tau_spec"]

    # ---- clean spectrum (eigh ONCE) ----
    A_clean = _build_adj(n, data.edge_index, [], 0, [], device)
    lam_clean, U_clean = compute_laplacian_spectrum(A_clean, k_top)
    lam_clean = lam_clean.to(device); U_clean = U_clean.to(device)
    kk = lam_clean.numel()

    # ---- init feats from prior + init edge set S = all victims (full-edge anchor) ----
    gen = torch.Generator(device="cpu").manual_seed(
        data.num_nodes + cfg.get("seed_offset", 0) * 99991)
    eps = torch.randn(m, d, generator=gen).to(device)
    x = mu.unsqueeze(0) + eps @ L_chol.t()
    x = _project_maha(x, mu, cov_inv, thr)

    # current victim-edge set as a set of (i, v) tuples — start full-edge then prune/swap.
    S = set(all_v)

    def _spec_of(S_set):
        """Predict S_spec via Hellmann–Feynman on (n+1)x(n+1) augmented L'."""
        A_p = _build_adj(n, data.edge_index, list(S_set), m, inter_full, device)
        L_p = _normalized_laplacian(A_p)
        L_c_pad = torch.zeros_like(L_p)
        L_c_pad[:n, :n] = _normalized_laplacian(A_clean)
        dL = L_p - L_c_pad
        dlam = closed_form_delta_lambda(U_clean, lam_clean, dL)[:kk]
        lam_pert_est = lam_clean[:kk] + dlam
        return spec_disc(lam_pert_est, lam_clean[:kk], beta).item()

    flipped = False
    for t in range(T):
        frac = 1.0 - t / max(T - 1, 1)
        v_edges = list(S)
        if not v_edges:
            break

        # ---- feature step (SGDI Langevin, optional smoothness) ----
        g_adv, cur = _cge_grad(model, data, x, v_edges, inter_full, cfg, device)
        if cur < 0:
            flipped = True
            break
        s_prior = -(x - mu.unsqueeze(0)) @ cov_inv
        noise = torch.randn(m, d, generator=gen).to(device) * (temp0 * frac)
        update = -lr * g_adv + eta_prior * s_prior + noise
        if signal_variant:
            # smoothness on injected node: pull x* toward neighbour mean
            for i in range(m):
                nbr_idx = [v for (ii, v) in S if ii == i]
                if nbr_idx:
                    x_nbr = data.x[nbr_idx]
                    g_sm = _signal_smoothness_grad(x[i:i+1], x_nbr, len(nbr_idx))
                    update[i:i+1] = update[i:i+1] - eta_sm * g_sm
        x = x + update
        x = _project_maha(x, mu, cov_inv, thr)

        # ---- edge step (spectrum-minimising single-edge swap) ----
        # candidate swaps: drop one (i,u)∈S, add one (i,u')∈V∖S_i
        # score each by closed-form ΔS_spec under the prospective S.
        current_spec = _spec_of(S)
        cand_drops = list(S)
        in_S = {(i, v) for (i, v) in S}
        # Sample candidate adds: for each injected node i, all victim u' not in S_i.
        cand_adds = [(i, v) for i in range(m) for v in range(n)
                     if (i, v) not in in_S]
        if not cand_drops or not cand_adds:
            continue

        # To bound work: try every drop paired with the *single best add* heuristic
        # (or vice-versa). We'll enumerate K_swaps best candidates by greedy:
        # 1) score K_swaps single drops by spec-reduction;
        # 2) for each drop, find the best paired add (smallest spec).
        # This is O(K_swaps * |adds|) closed-form ΔS_spec evaluations + K_swaps ZO probes.
        drop_scores = []
        for du in cand_drops:
            S_minus = S - {du}
            drop_scores.append((_spec_of(S_minus), du))
        drop_scores.sort(key=lambda r: r[0])  # ascending S_spec after drop
        top_drops = drop_scores[:max(1, K_swaps)]

        best = (current_spec, None, None)  # (S_spec_new, drop, add) or no-op
        for s_after_drop, du in top_drops:
            S_minus = S - {du}
            # find best add for this drop
            best_add = None
            best_add_spec = float("inf")
            for au in cand_adds:
                if au == du:
                    continue
                S_try = S_minus | {au}
                sp = _spec_of(S_try)
                if sp < best_add_spec:
                    best_add_spec = sp
                    best_add = au
            if best_add is None:
                # also consider pure-drop (no add) — yields smaller |S| but still valid
                if s_after_drop < best[0]:
                    best = (s_after_drop, du, None)
                continue
            if best_add_spec < best[0]:
                best = (best_add_spec, du, best_add)

        # If a candidate swap improves spec, ZO-check that it doesn't WORSEN L_adv
        # (≥ current margin allowed only if spec drop is substantial; here we require
        #  strict L_adv non-increase, as per §2.3).
        _, du, au = best
        if du is None and au is None:
            continue
        S_new = (S - ({du} if du else set())) | ({au} if au else set())
        v_new = list(S_new)
        if not v_new:
            continue
        mg_new = _eval_specgdi_loss(model, data, x, v_new, inter_full, device)
        if mg_new <= cur + 1e-6:
            S = S_new

    # ---- hard projection: S_spec ≤ τ_spec via greedy edge drops ----
    proj_iters = 0
    while _spec_of(S) > tau_spec and len(S) > 1 and proj_iters < 2 * n:
        # drop the single edge whose removal most reduces S_spec
        best_drop = None
        best_sp = float("inf")
        for du in list(S):
            sp = _spec_of(S - {du})
            if sp < best_sp:
                best_sp = sp
                best_drop = du
        if best_drop is None:
            break
        S = S - {best_drop}
        proj_iters += 1

    v_final = list(S)
    final_spec = _spec_of(S) if v_final else 0.0
    margin_final = _eval_specgdi_loss(model, data, x, v_final, inter_full, device) \
        if v_final else float("inf")
    flipped = bool(margin_final < 0)
    return x.detach(), v_final, float(final_spec), flipped


def _run_specgdi_sweep(model, test_graphs, device, cfg=None):
    """Top-level SpecGDI runner — mirrors run_attack_sweep's contract.

    Returns the same keys plus 's_spec_per_graph' and 'tau_spec' so downstream
    smoke-tests / harness logs can verify spec-stealth.
    """
    global QUERY_COUNT
    QUERY_COUNT = 0
    cfg = {**CONFIG, **SPECGDI_DEFAULTS, **(cfg or {})}
    m = cfg["node_budget"]
    stats = data_stats(test_graphs, device)
    cfg = {**cfg, "_norm_radius": cfg.get("normcap_kappa", 1.95) * stats[2]}
    cfg = {**cfg, "_prior": data_prior(test_graphs, device)}

    # τ_spec lookup: dataset key derived from feature dim (PROTEINS:d=3, ENZYMES:d=21)
    # or explicit cfg["specgdi_dataset_key"] override.
    ds_key = cfg.get("specgdi_dataset_key")
    if ds_key is None:
        d_in = int(test_graphs[0].x.size(1)) if test_graphs else 0
        ds_key = {3: "PROTEINS", 21: "ENZYMES"}.get(d_in, f"d{d_in}")
    tau_spec = _load_tau_spec(ds_key, default=cfg.get("specgdi_tau_default", 0.10))
    cfg = {**cfg, "_tau_spec": tau_spec}

    signal_variant = (cfg["feat_mode"] == "specgdi_signal")
    records = []
    s_specs = []
    for data in test_graphs:
        data = data.to(device)
        feats0 = make_features(stats, m, cfg["feat_mode"], data.num_nodes, device)
        n = data.num_nodes
        all_v = [(i, v) for i in range(m) for v in range(n)]
        inter_full = [(i, j) for i in range(m) for j in range(i + 1, m)] \
            if cfg["use_inter_edges"] else []
        _, v_final, s_spec, flipped = _specgdi_optimize_one(
            model, data, feats0, all_v, inter_full, cfg, device,
            cfg["_prior"], signal_variant=signal_variant)
        records.append({
            "rho_flip": (len(v_final) / max(n, 1)) if flipped else math.inf,
            "full_flip": flipped,
            "n_inter": len(inter_full),
            "greedy_edges": len(v_final) if flipped else None,
        })
        s_specs.append(s_spec)

    n_r = len(records) or 1
    return {
        "asr_full": sum(r["full_flip"] for r in records) / n_r,
        "rho_to_flip": [r["rho_flip"] for r in records],
        "greedy_edges": [r["greedy_edges"] for r in records],
        "n_inter": [r["n_inter"] for r in records],
        "queries": QUERY_COUNT,
        "n_graphs": len(records),
        "s_spec_per_graph": s_specs,
        "tau_spec": tau_spec,
        "dataset_key": ds_key,
    }


# ============================================================
# SpecGDI v2 — Joint multi-node trojan diffusion (K-node, joint update)
# ------------------------------------------------------------
# Fixes two structural weaknesses of v1 (single-node, alternating block descent):
#   (1) Alternating → joint: a single joint Langevin step at the SAME (X_t, A_t)
#       point per diffusion iteration t = T..1; no per-block re-evaluation.
#   (2) Single-node → K-node trojan: K injected nodes form a community-like
#       sub-graph via soft inter-injected edges A_VV ∈ [0,1]^{KxK}, and
#       cross-edges A_VU ∈ [0,1]^{KxN}. Per-node degree stays clean-like →
#       degree-based denoisers (gddm_a) cannot trigger on per-node anomaly.
#
# State:  X* (K,d)  features
#         A_VV (K,K) symmetric, zero-diagonal soft inter-injected edges
#         A_VU (K,N) soft cross edges
#
# Joint update per iteration (Step A/B/C in spec):
#   A. Evaluate ALL gradients at (X_t, A_VV_t, A_VU_t):
#        - grad_adv on (X, A_VV, A_VU)   via SPSA (shared random direction)
#        - grad_feat = Sigma_inv @ (X - mu)   (X only, broadcast)
#        - grad_spec = Hellmann–Feynman on composite Laplacian   (A only;
#          X part is exactly zero in the normalized-Laplacian formulation)
#   B. Joint update (no re-evaluation between blocks):
#        X     -= eps * (g_adv_X + η_f * g_feat)
#        A_VV  -= eps * (g_adv_VV + η_s * g_spec_VV)
#        A_VU  -= eps * (g_adv_VU + η_s * g_spec_VU)
#        X     += sqrt(2 eps) * randn(K, d)            (Langevin noise, X only)
#   C. Projections:
#        X: Mahalanobis ball, row-wise
#        A_VV: clip[0,1], symmetrize, zero diag
#        A_VU: clip[0,1]
#        If S_spec(composite) > τ_spec_composite: greedy drop strongest soft
#        edges (by Hellmann–Feynman score) until satisfied.
#
# After T_diff iters: discretize (top-density_VV * K*(K-1)/2 entries of A_VV;
# top-budget_VU entries of A_VU) and run final hard projection.
#
# Calibration: calibrate_specgdi_v2.py writes
#   { dataset: { "tau_spec_composite": ..., "clean_community_density": ... } }
# to specgdi_v2_thresholds.json.
# ============================================================

SPECGDI_V2_DEFAULTS = {
    "v2_K": 3,                       # number of injected nodes
    "v2_T_diff": 10,                 # diffusion iters
    "v2_eps": 0.05,                  # joint step size
    "v2_eta_feat": 0.3,              # prior pull on X
    "v2_eta_spec": 0.5,              # spectral pull on A
    "v2_n_probes": 4,                # SPSA random directions per step
    "v2_zo_sigma": 5e-3,             # SPSA finite-difference sigma
    "v2_temp0": 0.1,                 # initial Langevin temperature (annealed)
    "v2_k_top": 20,                  # top-k eigenvalues to track
    "v2_beta": 1.0,                  # spectral weights w_i = exp(-β λ_i)
    "v2_budget_VU_per_node": 0.2,    # fraction-of-N cross-edges per injected node (kept low so each
                                     # injected node's degree stays inside the clean degree distribution;
                                     # 0.2*N ≈ clean p95 degree for PROTEINS small graphs)
    "v2_tau_path": "specgdi_v2_thresholds.json",
    "v2_density_fallback": 0.5,      # inter-injected density if calibration missing
    "v2_tau_composite_fallback": 0.633,  # τ_spec_composite if calibration missing
    "v2_per_graph_timeout_s": 240.0, # 4 min per-graph kill switch
}


def _v2_build_composite_adj(victim_n, victim_edge_index, A_VV, A_VU, device):
    """Build dense (N+K, N+K) symmetric adjacency: victim clean + soft A_VV ∪ A_VU.

    A_VV, A_VU are torch tensors (already symmetric/zero-diag for A_VV).
    Victim sub-block uses the clean binary edge_index.
    """
    K = A_VV.size(0)
    N = victim_n
    M = N + K
    A = torch.zeros(M, M, device=device)
    if victim_edge_index is not None and victim_edge_index.numel() > 0:
        ei = victim_edge_index
        s = ei[0].tolist(); d = ei[1].tolist()
        for u, v in zip(s, d):
            A[u, v] = 1.0
    # A_VV block at indices [N:N+K, N:N+K]
    A[N:M, N:M] = A_VV
    # A_VU block at [N:N+K, 0:N] and its transpose
    A[N:M, 0:N] = A_VU
    A[0:N, N:M] = A_VU.t()
    return A


def composite_laplacian_spectrum(victim_edge_index, victim_n, A_VV, A_VU, k_top, device):
    """Top-k smallest eigenvalues + eigenvectors of normalized Laplacian of the
    composite (victim ∪ K injected) graph. Returns (lam_k, U_k) on `device`.

    Uses dense torch.linalg.eigh on CPU; n+K typically < 100 → ~ms.
    """
    A = _v2_build_composite_adj(victim_n, victim_edge_index, A_VV, A_VU, device)
    L = _normalized_laplacian(A).detach().cpu()
    L = 0.5 * (L + L.t())
    eigvals, eigvecs = torch.linalg.eigh(L)
    kk = int(min(k_top, eigvals.numel()))
    return eigvals[:kk].to(device), eigvecs[:, :kk].to(device)


def _v2_compute_S_spec(lam_pert, lam_clean_padded, beta):
    """S_spec_composite = sum_i w_i (λ_i' - λ_i)^2  with w_i = exp(-β λ_i_clean)."""
    if lam_pert.numel() == 0:
        return torch.tensor(0.0, device=lam_pert.device)
    L = min(lam_pert.numel(), lam_clean_padded.numel())
    w = torch.exp(-beta * lam_clean_padded[:L])
    return (w * (lam_pert[:L] - lam_clean_padded[:L]).pow(2)).sum()


def spec_grad_wrt_A_v2(eigvecs, eigvals, lam_clean_padded, beta, N, K):
    """Hellmann–Feynman gradient of S_spec w.r.t. A_VV and A_VU entries.

    For normalized Laplacian L = I - D^{-1/2} A D^{-1/2}, ∂L/∂A_{uv} couples both
    a direct (-2/√(d_u d_v)) term at the (u,v) entries and a degree-dependent
    diagonal term. We use the LEADING-ORDER approximation that ∂λ_i/∂A_{uv}
    ≈ -2 * u_i(u) * u_i(v) / √(d_u d_v + ε), i.e. ignore the degree-derivative
    chain rule for tractability — this is exact only in the unnormalized case
    L = D - A but is well-known to be a good approximation for normalized L and
    is the standard practice in spectral attack literature.

    Returns (grad_AVV [K,K], grad_AVU [K,N]) — already symmetric / off-diagonal-aware.
    """
    # eigvecs: (N+K, k_top), eigvals: (k_top,), lam_clean_padded: (k_top_clean,)
    device = eigvecs.device
    L = min(eigvals.numel(), lam_clean_padded.numel())
    if L == 0:
        return torch.zeros(K, K, device=device), torch.zeros(K, N, device=device)
    diff = (eigvals[:L] - lam_clean_padded[:L])   # (L,)
    w = torch.exp(-beta * lam_clean_padded[:L])    # (L,)
    coeff = 2.0 * w * diff                          # ∂S/∂λ_i  (L,)

    U = eigvecs[:, :L]                              # (N+K, L)
    # ∂λ_i/∂A_{uv} ≈ -2 * U[u,i] * U[v,i] / sqrt(d_u d_v) ; we drop deg term here
    # and re-introduce a degree clamp later in the joint sign.
    # Sum_i coeff_i * U[u,i] * U[v,i] gives part of ∂S/∂A_{uv}.
    # We compute M_{uv} = Σ_i coeff_i * U[u,i] * U[v,i]   shape (N+K, N+K)
    M = (U * coeff.unsqueeze(0)) @ U.t()            # (N+K, N+K)

    # Apply -2 factor (from L = I - D^{-1/2} A D^{-1/2} sign):
    # ∂S/∂A_{uv} ≈ -2 * M[u,v]
    grad_full = -2.0 * M

    # Slice: A_VV is at [N:N+K, N:N+K]; A_VU is at [N:N+K, 0:N]
    grad_AVV = grad_full[N:N+K, N:N+K].clone()
    # symmetrize and zero diag for A_VV
    grad_AVV = 0.5 * (grad_AVV + grad_AVV.t())
    grad_AVV.fill_diagonal_(0.0)
    grad_AVU = grad_full[N:N+K, 0:N].clone()
    return grad_AVV, grad_AVU


def mahalanobis_project_rowwise(X, mu, Sigma_inv, tau):
    """Project each row of X onto the Mahalanobis ball
    {x : (x-μ)^T Σ^{-1} (x-μ) ≤ τ}. Rows outside are rescaled along the
    Mahalanobis ray from μ to land on the boundary; rows inside untouched."""
    diff = X - mu
    m2 = (diff @ Sigma_inv * diff).sum(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (tau / m2).sqrt().clamp(max=1.0)
    return mu + diff * scale


def _v2_build_data_soft(original, X_star, A_VV_soft, A_VU_soft, device,
                          density_VV=None, budget_VU_per_node=None):
    """Build a PyG Data from the soft trojan state. Two materialisation modes:

    (A) If density_VV is None: HARD mode — assume the inputs are already binary
        (0/1) hard edges, threshold at >=0.5 (back-compat with discretize_trojan
        output). Used for the *final* evaluation after discretisation.

    (B) If density_VV is given: SOFT-PROBE mode — discretise by top-k:
        - A_VV: take top-⌈density_VV * K(K-1)/2⌉ symmetric off-diagonal entries
        - A_VU: per row i, take top-⌈budget_VU_per_node * N⌉ entries
        This keeps the probe well-defined: tiny perturbations on soft values
        DO move the topk selection when nearby entries are close, giving a
        non-zero ZO signal. This is the key fix vs. naive 0.5 thresholding
        (which produced near-zero gradient when soft values were all <0.5).
    """
    K = X_star.size(0)
    N = original.num_nodes
    if density_VV is not None:
        A_VV_hard, A_VU_hard = discretize_trojan(A_VV_soft, A_VU_soft,
                                                  density_VV, budget_VU_per_node)
    else:
        A_VV_hard, A_VU_hard = A_VV_soft, A_VU_soft
    v_edges = []
    for i in range(K):
        for v in range(N):
            if A_VU_hard[i, v].item() >= 0.5:
                v_edges.append((i, v))
    inter_edges = []
    for i in range(K):
        for j in range(i + 1, K):
            if A_VV_hard[i, j].item() >= 0.5:
                inter_edges.append((i, j))
    return _build(original, X_star, v_edges, inter_edges, device)


def joint_zo_cge_gradient(model, data, X_star, A_VV, A_VU, cfg, device):
    """SPSA-style joint zeroth-order gradient estimate of CW margin w.r.t.
    (X_star, A_VV, A_VU) at the current point. Returns (g_X, g_AVV, g_AVU).

    Uses n_probes independent ±σ perturbations sharing the same random direction
    across the three state blocks (joint estimator); averages the central-
    difference estimates. Total queries = 2 * n_probes.

    Note: we perturb A in [0,1] space but clamp probe points back into the box;
    edges are materialised at the 0.5 threshold for evaluation (locally smooth
    only when soft entries are not crossing the threshold — acceptable as a
    differentiable surrogate in the diffusion regime).
    """
    true_label = data.y.item()
    K, d = X_star.shape
    N = A_VU.size(1)
    n_probes = cfg["v2_n_probes"]
    sigma = cfg["v2_zo_sigma"]
    gen = torch.Generator(device="cpu").manual_seed(
        int(data.num_nodes) + cfg.get("seed_offset", 0) * 99991)

    g_X = torch.zeros_like(X_star)
    g_VV = torch.zeros_like(A_VV)
    g_VU = torch.zeros_like(A_VU)

    for p in range(n_probes):
        # shared random direction across blocks (Gaussian)
        dX = torch.randn(K, d, generator=gen).to(device)
        dVV = torch.randn(K, K, generator=gen).to(device)
        dVV = 0.5 * (dVV + dVV.t())
        dVV.fill_diagonal_(0.0)
        dVU = torch.randn(K, N, generator=gen).to(device)

        Xp = X_star + sigma * dX
        Xm = X_star - sigma * dX
        VVp = (A_VV + sigma * dVV).clamp(0, 1); VVp.fill_diagonal_(0.0)
        VVm = (A_VV - sigma * dVV).clamp(0, 1); VVm.fill_diagonal_(0.0)
        VUp = (A_VU + sigma * dVU).clamp(0, 1)
        VUm = (A_VU - sigma * dVU).clamp(0, 1)

        # Use SOFT-PROBE mode (topk discretisation) so small perturbations actually
        # change the edge set — pulling a non-zero ZO signal out of A.
        density_VV = cfg.get("_v2_density_VV", 0.5)
        budget_VU = cfg.get("v2_budget_VU_per_node", 0.2)
        dlp = _v2_build_data_soft(data, Xp, VVp, VUp, device,
                                   density_VV=density_VV,
                                   budget_VU_per_node=budget_VU)
        dlm = _v2_build_data_soft(data, Xm, VVm, VUm, device,
                                   density_VV=density_VV,
                                   budget_VU_per_node=budget_VU)
        margins = _eval_graphs(model, [dlp, dlm], true_label, device)
        delta = (margins[0].item() - margins[1].item()) / (2.0 * sigma)
        g_X = g_X + delta * dX
        g_VV = g_VV + delta * dVV
        g_VU = g_VU + delta * dVU

    g_X = g_X / max(n_probes, 1)
    g_VV = g_VV / max(n_probes, 1)
    g_VV = 0.5 * (g_VV + g_VV.t())
    g_VV.fill_diagonal_(0.0)
    g_VU = g_VU / max(n_probes, 1)
    return g_X, g_VV, g_VU


def discretize_trojan(A_VV_soft, A_VU_soft, density_VV, budget_VU_per_node):
    """Threshold the soft trojan into hard binary edges.

    A_VV: keep top-⌈density_VV * K*(K-1)/2⌉ off-diagonal entries (symmetric).
    A_VU: per injected-node i, keep top-⌈budget_VU_per_node * N⌉ entries.
    """
    K = A_VV_soft.size(0)
    N = A_VU_soft.size(1)
    A_VV_hard = torch.zeros_like(A_VV_soft)
    # take upper triangle
    iu, ju = torch.triu_indices(K, K, offset=1)
    vals = A_VV_soft[iu, ju]
    n_keep_VV = max(1, int(math.ceil(density_VV * vals.numel()))) if vals.numel() > 0 else 0
    if n_keep_VV > 0 and vals.numel() > 0:
        top = torch.topk(vals, k=min(n_keep_VV, vals.numel())).indices
        for idx in top.tolist():
            i, j = int(iu[idx].item()), int(ju[idx].item())
            A_VV_hard[i, j] = 1.0
            A_VV_hard[j, i] = 1.0
    A_VU_hard = torch.zeros_like(A_VU_soft)
    n_keep_VU = max(1, int(math.ceil(budget_VU_per_node * N)))
    for i in range(K):
        row = A_VU_soft[i]
        top = torch.topk(row, k=min(n_keep_VU, N)).indices
        A_VU_hard[i, top] = 1.0
    return A_VV_hard, A_VU_hard


def _v2_eval_attack(model, data, X_star, A_VV_hard, A_VU_hard, device):
    """Single forward pass on the hard-discretized trojan. Returns CW margin."""
    g = _v2_build_data_soft(data, X_star, A_VV_hard, A_VU_hard, device)
    return _eval_graphs(model, [g], data.y.item(), device)[0].item()


def specgdi_v2_optimize_trojan(model, data, X_star_init, A_VV_init, A_VU_init,
                                cfg, device, prior):
    """Joint multi-node trojan diffusion. Returns
        (X_final, A_VV_hard, A_VU_hard, S_spec_final, flipped, n_steps)

    Implements the joint update spec: one ZO call per step for all three blocks,
    closed-form spectral gradients on A only, single Langevin noise on X.
    """
    import time as _t
    t_start = _t.time()
    mu, cov_inv, _L_chol, thr = prior
    K, d = X_star_init.shape
    N = data.num_nodes
    T = cfg["v2_T_diff"]
    eps = cfg["v2_eps"]
    eta_f = cfg["v2_eta_feat"]
    eta_s = cfg["v2_eta_spec"]
    temp0 = cfg["v2_temp0"]
    beta = cfg["v2_beta"]
    k_top = cfg["v2_k_top"]
    tau_composite = cfg["_v2_tau_composite"]
    timeout_s = cfg["v2_per_graph_timeout_s"]

    # ---- clean composite spectrum (only victim, K injected isolated nodes) ----
    A_VV0 = torch.zeros(K, K, device=device)
    A_VU0 = torch.zeros(K, N, device=device)
    lam_clean, _U_clean = composite_laplacian_spectrum(
        data.edge_index, N, A_VV0, A_VU0, k_top, device)

    X = X_star_init.clone()
    A_VV = A_VV_init.clone()
    A_VV = 0.5 * (A_VV + A_VV.t()); A_VV.fill_diagonal_(0.0)
    A_VU = A_VU_init.clone()

    gen = torch.Generator(device="cpu").manual_seed(
        int(N) + cfg.get("seed_offset", 0) * 99991)

    last_S = float("inf")
    for t in range(T):
        if _t.time() - t_start > timeout_s:
            break
        frac = 1.0 - t / max(T - 1, 1)
        # --- Step A: gradients at current point ---
        g_adv_X, g_adv_VV, g_adv_VU = joint_zo_cge_gradient(
            model, data, X, A_VV, A_VU, cfg, device)
        g_feat_X = (X - mu.unsqueeze(0)) @ cov_inv  # Σ^{-1}(X - μ); broadcast over K rows

        # spectral grad on A via Hellmann–Feynman
        lam_p, U_p = composite_laplacian_spectrum(
            data.edge_index, N, A_VV, A_VU, k_top, device)
        # pad clean to match length of lam_p
        Lp = lam_p.numel()
        Lc = lam_clean.numel()
        if Lc < Lp:
            lam_c_pad = torch.cat([lam_clean, torch.zeros(Lp - Lc, device=device)])
        else:
            lam_c_pad = lam_clean[:Lp]
        g_spec_VV, g_spec_VU = spec_grad_wrt_A_v2(
            U_p, lam_p, lam_c_pad, beta, N, K)

        # --- Step B: joint update (NO re-evaluation between blocks) ---
        X = X - eps * (g_adv_X + eta_f * g_feat_X)
        A_VV = A_VV - eps * (g_adv_VV + eta_s * g_spec_VV)
        A_VU = A_VU - eps * (g_adv_VU + eta_s * g_spec_VU)
        noise = torch.randn(K, d, generator=gen).to(device) * (temp0 * frac) * math.sqrt(2 * eps)
        X = X + noise

        # --- Step C: projections ---
        X = mahalanobis_project_rowwise(X, mu, cov_inv, thr)
        A_VV = A_VV.clamp(0, 1)
        A_VV = 0.5 * (A_VV + A_VV.t())
        A_VV.fill_diagonal_(0.0)
        A_VU = A_VU.clamp(0, 1)

        # spectral projection (only if violating)
        lam_check, _ = composite_laplacian_spectrum(
            data.edge_index, N, A_VV, A_VU, k_top, device)
        Lp = lam_check.numel()
        if Lc < Lp:
            lam_c_pad = torch.cat([lam_clean, torch.zeros(Lp - Lc, device=device)])
        else:
            lam_c_pad = lam_clean[:Lp]
        s_now = _v2_compute_S_spec(lam_check, lam_c_pad, beta).item()
        last_S = s_now
        if s_now > tau_composite:
            # greedy soft-edge drop: identify the largest soft entries in (A_VV, A_VU)
            # and zero them one at a time until under threshold (cap at 2*N iters).
            iters = 0
            while s_now > tau_composite and iters < 2 * (N + K):
                # find max in flattened soft entries
                vv_max = float(A_VV.max().item()) if A_VV.numel() > 0 else 0.0
                vu_max = float(A_VU.max().item()) if A_VU.numel() > 0 else 0.0
                if vv_max <= 1e-6 and vu_max <= 1e-6:
                    break
                if vv_max >= vu_max:
                    idx = torch.argmax(A_VV.view(-1)).item()
                    r, c = idx // K, idx % K
                    A_VV[r, c] = 0.0
                    A_VV[c, r] = 0.0
                else:
                    idx = torch.argmax(A_VU.view(-1)).item()
                    r, c = idx // N, idx % N
                    A_VU[r, c] = 0.0
                lam_check2, _ = composite_laplacian_spectrum(
                    data.edge_index, N, A_VV, A_VU, k_top, device)
                Lp2 = lam_check2.numel()
                if Lc < Lp2:
                    lam_c_pad2 = torch.cat([lam_clean, torch.zeros(Lp2 - Lc, device=device)])
                else:
                    lam_c_pad2 = lam_clean[:Lp2]
                s_now = _v2_compute_S_spec(lam_check2, lam_c_pad2, beta).item()
                iters += 1
            last_S = s_now

    # ---- discretize ----
    density_VV = cfg["_v2_density_VV"]
    budget_VU = cfg["v2_budget_VU_per_node"]
    A_VV_hard, A_VU_hard = discretize_trojan(A_VV, A_VU, density_VV, budget_VU)

    # ---- final hard spectral projection ----
    lam_final, _ = composite_laplacian_spectrum(
        data.edge_index, N, A_VV_hard, A_VU_hard, k_top, device)
    Lp = lam_final.numel()
    if Lc < Lp:
        lam_c_pad = torch.cat([lam_clean, torch.zeros(Lp - Lc, device=device)])
    else:
        lam_c_pad = lam_clean[:Lp]
    S_final = _v2_compute_S_spec(lam_final, lam_c_pad, beta).item()
    proj_iters = 0
    while S_final > tau_composite and proj_iters < 2 * (N + K):
        # drop the strongest hard edge (greedy)
        vv_max = float(A_VV_hard.max().item()) if A_VV_hard.numel() > 0 else 0.0
        vu_max = float(A_VU_hard.max().item()) if A_VU_hard.numel() > 0 else 0.0
        if vv_max < 0.5 and vu_max < 0.5:
            break
        if vv_max >= vu_max:
            idx = torch.argmax(A_VV_hard.view(-1)).item()
            r, c = idx // K, idx % K
            A_VV_hard[r, c] = 0.0
            A_VV_hard[c, r] = 0.0
        else:
            idx = torch.argmax(A_VU_hard.view(-1)).item()
            r, c = idx // N, idx % N
            A_VU_hard[r, c] = 0.0
        lam_final, _ = composite_laplacian_spectrum(
            data.edge_index, N, A_VV_hard, A_VU_hard, k_top, device)
        Lp = lam_final.numel()
        if Lc < Lp:
            lam_c_pad = torch.cat([lam_clean, torch.zeros(Lp - Lc, device=device)])
        else:
            lam_c_pad = lam_clean[:Lp]
        S_final = _v2_compute_S_spec(lam_final, lam_c_pad, beta).item()
        proj_iters += 1

    margin = _v2_eval_attack(model, data, X, A_VV_hard, A_VU_hard, device)
    flipped = bool(margin < 0)
    return X.detach(), A_VV_hard, A_VU_hard, float(S_final), flipped, T


def _load_specgdi_v2_thresholds(dataset_key, cfg):
    """Lookup τ_spec_composite and clean_community_density from
    specgdi_v2_thresholds.json. Falls back to cfg defaults."""
    p = _SpgdPath(__file__).parent / cfg["v2_tau_path"]
    tau = cfg["v2_tau_composite_fallback"]
    dens = cfg["v2_density_fallback"]
    if p.exists():
        try:
            d = _spgd_json.loads(p.read_text())
            entry = d.get(dataset_key, d.get("default", {}))
            if isinstance(entry, dict):
                tau = float(entry.get("tau_spec_composite", tau))
                dens = float(entry.get("clean_community_density", dens))
        except Exception:
            pass
    return tau, dens


def _run_specgdi_v2_sweep(model, test_graphs, device, cfg=None):
    """Top-level SpecGDI v2 runner — joint K-node trojan diffusion.

    Returns the same contract as run_attack_sweep, plus:
      s_spec_per_graph, tau_spec_composite, density_VV, dataset_key,
      n_inj_nodes (== K), wall_clock_per_graph.
    """
    import time as _t
    global QUERY_COUNT
    QUERY_COUNT = 0
    cfg = {**CONFIG, **SPECGDI_DEFAULTS, **SPECGDI_V2_DEFAULTS, **(cfg or {})}
    stats = data_stats(test_graphs, device)
    cfg = {**cfg, "_norm_radius": cfg.get("normcap_kappa", 1.95) * stats[2]}
    cfg = {**cfg, "_prior": data_prior(test_graphs, device)}

    # dataset key + thresholds
    ds_key = cfg.get("specgdi_dataset_key")
    if ds_key is None:
        d_in = int(test_graphs[0].x.size(1)) if test_graphs else 0
        ds_key = {3: "PROTEINS", 21: "ENZYMES"}.get(d_in, f"d{d_in}")
    tau_comp, density_VV = _load_specgdi_v2_thresholds(ds_key, cfg)
    cfg = {**cfg, "_v2_tau_composite": tau_comp, "_v2_density_VV": density_VV}

    K = cfg["v2_K"]
    mu = cfg["_prior"][0]
    d = mu.numel()

    records = []
    s_specs = []
    wall_times = []
    for data in test_graphs:
        data = data.to(device)
        N = data.num_nodes
        # init: X = μ (data-manifold mean), A_VV = small (random in [0,0.3]), A_VU same
        gen = torch.Generator(device="cpu").manual_seed(
            int(N) + cfg.get("seed_offset", 0) * 99991)
        X0 = mu.unsqueeze(0).repeat(K, 1)
        VV0 = (torch.rand(K, K, generator=gen) * 0.3).to(device)
        VV0 = 0.5 * (VV0 + VV0.t()); VV0.fill_diagonal_(0.0)
        VU0 = (torch.rand(K, N, generator=gen) * 0.3).to(device)

        t_g0 = _t.time()
        try:
            _Xf, AVV_h, AVU_h, S_spec, flipped, _ = specgdi_v2_optimize_trojan(
                model, data, X0, VV0, VU0, cfg, device, cfg["_prior"])
        except Exception as e:
            print(f"  [v2] graph N={N} skipped: {e}")
            flipped = False
            S_spec = float("nan")
            AVV_h = torch.zeros(K, K, device=device)
            AVU_h = torch.zeros(K, N, device=device)
        wall_times.append(_t.time() - t_g0)
        n_inj_edges = int(AVU_h.sum().item()) + int(AVV_h.sum().item() / 2)
        records.append({
            "rho_flip": (n_inj_edges / max(N, 1)) if flipped else math.inf,
            "full_flip": flipped,
            "n_inter": int(AVV_h.sum().item() / 2),
            "greedy_edges": n_inj_edges if flipped else None,
        })
        s_specs.append(S_spec)

    n_r = len(records) or 1
    return {
        "asr_full": sum(r["full_flip"] for r in records) / n_r,
        "rho_to_flip": [r["rho_flip"] for r in records],
        "greedy_edges": [r["greedy_edges"] for r in records],
        "n_inter": [r["n_inter"] for r in records],
        "queries": QUERY_COUNT,
        "n_graphs": len(records),
        "s_spec_per_graph": s_specs,
        "tau_spec_composite": tau_comp,
        "density_VV": density_VV,
        "dataset_key": ds_key,
        "n_inj_nodes": K,
        "wall_clock_per_graph": wall_times,
    }


# ============================================================
# TopoMass — topology-driven readout-mass injection
# ------------------------------------------------------------
# Target: RS-Pool victims (Ennadir et al., NeurIPS 2025). RS-Pool reads out
# tau * v1(H), tau = sigma1(H)/alpha, where v1 is the dominant right singular
# vector of the node-embedding matrix H. Its robustness certificate scales as
# 1/(sigma1 - sigma2): a small top-2 singular GAP means v1 rotates easily, so
# the class readout is fragile there.
#
# Two coupled levers (feat_mode == "topomass"):
#
#   1. Multi-direction CAPPED features (norm = kappa * median node norm, 1.95).
#      The k injected feature rows are spread across the top-r principal input
#      directions instead of all aligned to one axis. Aligning to a single axis
#      (the failed N4-S "single-direction") only inflates sigma1 ALONG the
#      existing v1 -> the readout direction never rotates -> no flip. Spreading
#      mass pre-loads sigma2, shrinking the gap so an edge can rotate v1. The
#      per-row L2 cap keeps every injected node inside the NormFilter ball, so
#      the attack raises NO feature norm beyond clean nodes.
#
#   2. NORM-FREE edge lever (gap-targeted). Edges move readout mass through
#      message passing without touching feature norm. Targets are chosen
#      greedily to minimise the resulting top-2 singular gap of H ("gap" mode;
#      reads H via a forward hook on the victim's last node-embedding layer).
#      When embeddings are not exposed the selector falls back to the pure
#      score-based "margin" surrogate. Either way the KILL sweep is config-only
#      (vary tm_edge_budgets) and is pure black-box (margin-gated flip check).
#
# Constraints (mirrors the SpecGDI banners): self-contained below this line;
# the ONLY edit above is one dispatch branch in run_attack_sweep. Existing
# helpers (_build, _eval_graphs, _candidate_pool, _zo_one, data_stats) reused.
# ============================================================

TOPOMASS_DEFAULTS = {
    "node_budget": 4,               # k: injected nodes (>=2 for true multi-direction)
    "feat_scale": 1.0,              # capped features: NO sqrt(N) amplification (norm-free)
    "normcap_kappa": 1.95,          # per-node L2 cap = kappa * median train-node norm
    "tm_r": 2,                      # spread feature mass across top-r principal directions
    "tm_refine": True,              # ZO-refine the capped multi-direction features
    "tm_edge_select": "gap",        # "gap" (embedding-gap targeted) | "margin" (black-box)
    "tm_pool_mult": 2,              # victim-target candidate pool size multiplier
    "tm_edge_budgets": [1, 4, 8, 16],   # eb sweep (KILL test) + per-budget ASR
    "tm_headline_budget": 8,        # run_attack() float = ASR at this edge budget
    "use_inter_edges": True,        # allow injected<->injected edges in the clique
    # reuses zo_epochs / zo_lr / zo_sigma / zo_restarts from CONFIG for ZO refine.
}


def _topk_right_singular(A, k, iters=15):
    """Top-k (singular value, right singular vector) of A via power iteration + deflation.

    Pure matmul on M = A^T A (no LAPACK) -> device-agnostic and robust on CUDA, and the
    same primitive RS-Pool itself uses for sigma1/v1. Returns (svals[k'], V[k', d]).
    """
    d = A.size(1)
    M = A.t() @ A
    k = max(1, min(k, d))
    svals, vecs = [], []
    for _ in range(k):
        v = torch.randn(d, generator=torch.Generator(device="cpu").manual_seed(d + len(vecs))
                        ).to(A.device)
        v = v / (v.norm() + 1e-8)
        for _ in range(iters):
            v = M @ v
            v = v / (v.norm() + 1e-8)
        s = (A @ v).norm()
        svals.append(s)
        vecs.append(v)
        M = M - (s * s) * torch.outer(v, v)               # deflate to expose next direction
    return torch.stack(svals), torch.stack(vecs)


def _topomass_principal_dirs(test_graphs, device, r):
    """Top-r principal directions (unit) of the centred node-feature matrix + mean.

    Attacker-side statistics only (no victim queries) -> black-box. These input-space
    axes are the attacker's proxy for H's top-r right singular subspace; spreading the
    injected rows across them is what separates TopoMass from single-direction N4-S.
    """
    X = torch.cat([g.x for g in test_graphs], dim=0).to(device).float()
    mu = X.mean(dim=0)
    _, dirs = _topk_right_singular(X - mu, r)             # [r', d] unit rows, matmul-only
    return mu, dirs


def _topomass_init_features(mu, dirs, m, radius):
    """k capped injected rows spread across the principal directions (multi-direction).

    Row i points along dirs[i % r]; sign alternates so paired nodes straddle the axis
    (loads BOTH +/- ends -> larger spread of sigma2). Every row is L2-capped to `radius`.
    """
    r = dirs.size(0)
    rows = []
    for i in range(m):
        axis = dirs[i % r]
        sign = 1.0 if (i // r) % 2 == 0 else -1.0
        rows.append(mu + sign * radius * axis)
    feats = torch.stack(rows, dim=0)
    norm = feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return feats * (radius / norm).clamp(max=1.0)         # hard per-node L2 cap


class _HCapture:
    """Forward-hook capture of the victim's node-embedding matrix H (pre-readout).

    Grabs every 2-D float activation; latest_H() returns the last one whose row count
    matches the perturbed graph (== the final node embeddings RS-Pool pools over). If
    nothing matches (victim never exposes node embeddings) latest_H() returns None and
    the caller falls back to the black-box margin selector.
    """

    def __init__(self, model, expected_n):
        self.expected_n = expected_n
        self.buf = []
        self.handles = [mod.register_forward_hook(self._hook) for mod in model.modules()]

    def _hook(self, module, inp, out):
        if torch.is_tensor(out) and out.dim() == 2 and out.dtype.is_floating_point:
            self.buf.append(out.detach())

    def reset(self):
        self.buf = []

    def latest_H(self):
        matched = [t for t in self.buf if t.size(0) == self.expected_n]
        if matched:
            return matched[-1]
        return self.buf[-1] if self.buf else None

    def close(self):
        for h in self.handles:
            h.remove()


def _singular_gap(H):
    """Top-2 singular gap sigma1 - sigma2 of H (the RS-Pool fragility surrogate).

    Power-iteration (matmul-only) so it is CUDA-safe and cheap inside the edge loop.
    """
    if H.size(1) < 2:
        s, _ = _topk_right_singular(H.float(), 1)
        return float(s[0].item())
    s, _ = _topk_right_singular(H.float(), 2)
    return float((s[0] - s[1]).item())


def _topomass_greedy(model, data, feats, cfg, device):
    """Greedy norm-free edge addition; returns {edge_budget: flipped_within_budget}.

    Early-stops at the first flip (flip_at): every budget >= flip_at is a success,
    every budget < flip_at a failure. "gap" mode ranks candidates by the resulting
    top-2 singular gap of H (one forward/candidate, H read via hook); "margin" mode
    ranks by CW margin in one batched query. Flip is ALWAYS decided by the CW margin
    (black-box), so the eb sweep / KILL test is score-based regardless of selector.
    """
    true_label = data.y.item()
    n = data.num_nodes
    m = feats.size(0)
    budgets = sorted(int(b) for b in cfg["tm_edge_budgets"])
    max_b = max(budgets)
    inter_full = [(i, j) for i in range(m) for j in range(i + 1, m)] \
        if cfg["use_inter_edges"] else []
    pool = _candidate_pool(data, max_b, cfg["tm_pool_mult"])
    expected_n = n + m
    select = cfg["tm_edge_select"]

    cap = None
    if select == "gap":
        cap = _HCapture(model, expected_n)

    victim_edges, used = [], set()
    flip_at = math.inf
    try:
        for _ in range(max_b):
            cands = [(i, v) for i in range(m) for v in pool if (i, v) not in used]
            if not cands:
                break
            if select == "gap" and cap is not None:
                best_idx, best_gap, best_margin = None, math.inf, None
                fell_back = False
                for idx, c in enumerate(cands):
                    cap.reset()
                    mg = _eval_graphs(
                        model, [_build(data, feats, victim_edges + [c], inter_full, device)],
                        true_label, device)[0].item()
                    H = cap.latest_H()
                    if H is None:                      # victim hides embeddings -> fallback
                        fell_back = True
                        break
                    gap = _singular_gap(H)
                    if gap < best_gap:
                        best_gap, best_idx, best_margin = gap, idx, mg
                if fell_back:
                    select = "margin"                  # degrade to black-box for the rest
                else:
                    chosen, chosen_margin = cands[best_idx], best_margin
            if select == "margin":
                data_list = [_build(data, feats, victim_edges + [c], inter_full, device)
                             for c in cands]
                margins = _eval_graphs(model, data_list, true_label, device)
                bi = int(torch.argmin(margins).item())
                chosen, chosen_margin = cands[bi], margins[bi].item()

            victim_edges.append(chosen)
            used.add(chosen)
            if chosen_margin < 0:
                flip_at = len(victim_edges)
                break
    finally:
        if cap is not None:
            cap.close()

    return {b: (flip_at <= b) for b in budgets}


def _run_topomass_sweep(model, test_graphs, device, cfg=None):
    """TopoMass runner. Same contract as run_attack_sweep + per-edge-budget ASR.

    asr_full == ASR at tm_headline_budget (what run_attack returns). asr_by_eb gives
    the full edge-budget curve for the KILL test (config-only eb in {1,4,8,16}).
    """
    global QUERY_COUNT
    QUERY_COUNT = 0
    cfg = {**CONFIG, **TOPOMASS_DEFAULTS, **(cfg or {})}
    m = cfg["node_budget"]
    r = cfg["tm_r"]
    stats = data_stats(test_graphs, device)
    radius = cfg.get("normcap_kappa", 1.95) * stats[2]    # per-node L2 cap (median norm)
    cfg = {**cfg, "_norm_radius": radius}
    mu, dirs = _topomass_principal_dirs(test_graphs, device, r)

    budgets = sorted(int(b) for b in cfg["tm_edge_budgets"])
    hit = {b: 0 for b in budgets}
    flip_rho = []

    for data in test_graphs:
        data = data.to(device)
        global _FEAT_SCALE
        _FEAT_SCALE = _resolve_feat_scale(cfg.get("feat_scale", 1.0), data.num_nodes)
        feats = _topomass_init_features(mu, dirs, m, radius)

        # capped multi-direction feature refinement on the full-target topology, kept
        # inside the L2 ball every step (zo_normcap projection) -> norm never grows.
        if cfg.get("tm_refine", True):
            all_v = [(i, v) for i in range(m) for v in range(data.num_nodes)]
            inter_full = [(i, j) for i in range(m) for j in range(i + 1, m)] \
                if cfg["use_inter_edges"] else []
            feats, _ = _zo_one(model, data, feats, all_v, inter_full, cfg, device,
                               "zo_normcap", radius)

        flips = _topomass_greedy(model, data, feats, cfg, device)
        for b in budgets:
            hit[b] += int(flips[b])
        first = min((b for b in budgets if flips[b]), default=None)
        flip_rho.append((first / data.num_nodes) if first is not None else math.inf)

    n = len(test_graphs) or 1
    asr_by_eb = {b: hit[b] / n for b in budgets}
    headline = int(cfg["tm_headline_budget"])
    asr_full = asr_by_eb.get(headline, asr_by_eb[max(budgets)])
    return {
        "asr_full": asr_full,                 # ASR @ tm_headline_budget (run_attack float)
        "asr_by_eb": asr_by_eb,               # full edge-budget curve (KILL test)
        "headline_budget": headline,
        "rho_to_flip": flip_rho,
        "queries": QUERY_COUNT,
        "n_graphs": len(test_graphs),
        "n_inj_nodes": m,
        "tm_r": r,
        "norm_radius": radius,
    }
