#!/usr/bin/env python3
"""
attack_om2c.py — OM2C: Orthogonal Mass-perp-Manifold Composition.

Drop-in replacement for attack.py with the SAME
    run_attack(model, test_graphs, device) -> float
signature (mirrors attack_flow.py / attack_diffusion.py), so run.py / executor.py
and the SGDI defense harness (DefendedModel, gcn_median RS-Pool checkpoints) all work
unchanged.

------------------------------------------------------------------------------
Idea (decompose the potency-vs-stealth double-bind)
------------------------------------------------------------------------------
The KATZE double-bind: amplifying the injected feature norm (feat_scale) is what
beats mean-pool dilution, but high norm is exactly what NormFilter / FeatClip /
spectral detectors catch; and against a *robust* readout (RS-Pool: median / trimmed
pooling) norm buys nothing at all — one giant-norm node is trimmed regardless.

OM2C splits the injection into two **orthogonal** channels:

  (a) MANIFOLD-TANGENT FEATURE CHANNEL  (norm-bounded, stealthy)
      Injected features are optimized only inside the manifold-tangent subspace
      V_k (top-k principal directions of the clean node-feature covariance) and
      hard-projected into the <= sigma_mult * sigma train-NORM ball AND the clean
      Mahalanobis ball every step. This passes NormFilter (norm capped), FeatClip
      (range), and a low-rank spectral detector (signal lives in V_k) by construction.

  (b) ORTHOGONAL NORM-FREE TOPOLOGY CHANNEL  (carries the readout mass)
      The adversarial mass is delivered by *which* victim nodes the injected node
      connects to (greedy + fraction-of-N anchors), NOT by feature magnitude.
      feat_scale is held at 1.0 — there is NO norm amplification. Against RS-Pool,
      edge mass (how much of the pooled readout is adversarially influenced) is the
      only lever that survives a robust aggregator.

ORTHOGONALITY ENFORCEMENT. At each feature step the black-box CW gradient g is
split  g = g_tan + g_perp  where g_tan = V_k V_k^T g lies in the manifold-tangent
subspace and g_perp is its orthogonal complement. Only g_tan updates the features;
g_perp (the part that the stealth projection would erase) is *discarded from the
feature channel and delivered by the topology channel instead*. Hence the
stealth-projection of (a) cannot erase the adversarial signal — that signal now
lives in (b). This is the only candidate that lifts the worst-case-over-defenses
floor whichever defense binds (detector OR architectural).

Threat model: strict black-box (CW margin only), injection-only (original graph
never mutated). All injection / Mahalanobis / greedy primitives are REUSED from the
local SGDI attack.py — OM2C only adds the tangent-subspace + 2sigma-ball projection
and the orthogonal channel split.

NOTE on the branch-diffusion attack.py name collision: the local attack.py here is
the SGDI variant (owns data_prior / _project_maha / _build / _cge_grad / greedy
helpers). We load it by explicit path so this module is import-order independent.
"""

import math
import os
import sys
import importlib.util

import torch

# ============================================================
# Resolve the local SGDI attack module (injection + Mahalanobis + greedy helpers)
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_sgdi_attack():
    """Return the local SGDI attack module (the attack.py next to this file)."""
    a = sys.modules.get("attack")
    if a is not None and hasattr(a, "data_prior") and hasattr(a, "_build"):
        return a
    return _load_module("om2c_sgdi_base", os.path.join(_HERE, "attack.py"))


_SGDI = _resolve_sgdi_attack()
_build = _SGDI._build
_eval_graphs = _SGDI._eval_graphs
_cge_grad = _SGDI._cge_grad
data_prior = _SGDI.data_prior
_project_maha = _SGDI._project_maha
_resolve_feat_scale = _SGDI._resolve_feat_scale
_candidate_pool = _SGDI._candidate_pool
data_stats = _SGDI.data_stats
# Power-iteration top-k right singular vectors (matmul-only, no LAPACK). REUSED so
# OM2C's manifold-tangent subspace is GPU-safe: torch.linalg.eigh/svd segfault in the
# graph_adversarial env once a CUDA context exists (see agent memory). cov = Xc^T Xc /
# (n-1) + eps*I, so the top-k right singular vectors of Xc ARE cov's top-k eigenvectors.
_topk_right_singular = _SGDI._topk_right_singular


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "node_budget": 1,           # m: injected nodes (>1 adds readout mass for RS-Pool)

    # --- channel (a): manifold-tangent feature, norm-bounded ---
    "tangent_dim": 2,           # k: top-k covariance eigenvectors = manifold-tangent V_k
    "sigma_mult": 2.0,          # 2-sigma train-NORM ball radius = mean_norm + 2*std_norm
    "prior_eps": 1e-3,          # ridge on the clean-feature covariance estimate

    # --- channel (b): orthogonal norm-free topology ---
    # feat_scale held at 1.0 — NO norm amplification; mass comes from edges, not norm.
    "feat_scale": 1.0,
    "max_edges": 16,            # greedy victim-edge cap (cheap/sparse regime)
    "pool_mult": 3,             # candidate pool size = max(16, pool_mult * max_edges)
    "use_inter_edges": True,    # inter-injected-node edges (only matters for m >= 2)
    # fraction-of-N anchors trace the expensive regime (rho=1.0 == full-edge):
    "frac_anchors": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],

    # --- ZO feature optimization (CGE) ---
    "zo_epochs": 30, "zo_lr": 0.5, "zo_sigma": 5e-3, "zo_restarts": 2,
    "skip_greedy": False,       # if True, only frac-anchors are probed (ASR@full only)
}


# ============================================================
# Stealth bookkeeping (opt-in; cheap). Smoke test reads this to report the fraction
# of injected features that land inside the 2-sigma norm ball AND the Maha ball.
# ============================================================

OM2C_STATS = {"in_ball": 0, "total": 0}


def _reset_stats():
    OM2C_STATS["in_ball"] = 0
    OM2C_STATS["total"] = 0


def _record_in_ball(x_raw, prior):
    mu, cov_inv, _L, thr, _Vk, r2s = prior
    diff = x_raw - mu
    m2 = (diff @ cov_inv * diff).sum(dim=1)
    norm = x_raw.norm(dim=1)
    tol_m = thr * (1.0 + 1e-5) + 1e-9
    tol_n = r2s * (1.0 + 1e-5) + 1e-9
    inside = (m2 <= tol_m) & (norm <= tol_n)
    OM2C_STATS["in_ball"] += int(inside.sum().item())
    OM2C_STATS["total"] += int(x_raw.shape[0])


# ============================================================
# OM2C prior: SGDI Gaussian prior + manifold-tangent subspace + 2-sigma norm radius
# ============================================================

def om2c_prior(test_graphs, device, k, sigma_mult, eps=1e-3):
    """Return (mu, cov_inv, L, thr, V_k, r_2sigma).

    (mu, cov_inv, L, thr) are the SGDI Gaussian data prior (reused). V_k holds the
    top-k eigenvectors of the clean node-feature covariance (manifold-tangent
    subspace). r_2sigma = mean_node_norm + sigma_mult * std_node_norm is the
    NORM-ball radius the feature channel is projected into. No victim queries.
    """
    mu, cov_inv, L, thr = data_prior(test_graphs, device, eps=eps)

    X = torch.cat([g.x for g in test_graphs], dim=0).to(device).float()
    d = X.size(1)
    kk = max(1, min(int(k), d))
    # top-k right singular vectors of Xc == top-k eigenvectors of the covariance
    # (manifold-tangent subspace). Power iteration (matmul-only) -> GPU-safe.
    _svals, vecs = _topk_right_singular(X - mu, kk)         # vecs: (kk, d) unit rows
    V_k = vecs.t().contiguous()                             # (d, kk) columns

    node_norms = X.norm(dim=1)
    r_2sigma = float((node_norms.mean() + sigma_mult * node_norms.std()).item())
    return mu, cov_inv, L, thr, V_k, r_2sigma


def _project_tangent(x, mu, V_k):
    """Project rows of x onto the manifold-tangent affine subspace mu + span(V_k)."""
    diff = x - mu
    coeff = diff @ V_k                            # (m, kk)
    return mu + coeff @ V_k.t()


def _project_norm_ball(x, radius):
    """Scale rows of x so their L2 norm <= radius (2-sigma train-norm ball)."""
    norm = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x * (radius / norm).clamp(max=1.0)


def _project_stealth(x, prior):
    """Hard projection onto the feature-channel feasible set:
    manifold-tangent subspace  AND  2-sigma norm ball  AND  Maha ball.

    Applied as a sequence (tangent -> norm-ball -> Maha). One pass is an approximate
    projection onto the intersection; the Maha step is last so the queried/injected
    feature provably satisfies the distribution-aware (Mahalanobis) constraint.
    """
    mu, cov_inv, _L, thr, V_k, r2s = prior
    x = _project_tangent(x, mu, V_k)
    x = _project_norm_ball(x, r2s)
    x = _project_maha(x, mu, cov_inv, thr)
    return x


# ============================================================
# Channel (a): manifold-tangent, norm-bounded feature optimization
# ============================================================

def om2c_feature_opt(model, data, feats0, v_edges, i_edges, prior, cfg, device):
    """ZO/CGE descent on the CW margin restricted to the manifold-tangent channel.

    Orthogonal split per step:  g = g_tan + g_perp,  g_tan = V_k V_k^T g.
    Only g_tan updates the features; g_perp (the off-manifold adversarial signal the
    stealth projection would erase) is dropped from the feature channel and is left
    for the topology channel to carry. Every iterate is hard-projected into the
    stealth set, so the feature ever queried/injected is on-manifold + norm-bounded.

    Returns (feats, flipped).
    """
    mu, cov_inv, L, thr, V_k, _r2s = prior
    m, d = feats0.shape
    lr = cfg["zo_lr"]
    best = _project_stealth(feats0.clone(), prior)
    flipped = False
    for r in range(cfg["zo_restarts"]):
        gen = torch.Generator(device="cpu").manual_seed(
            r * 1000 + data.num_nodes + cfg.get("seed_offset", 0) * 99991)
        if r == 0:
            x = mu.unsqueeze(0).repeat(m, 1)                 # start at the density mode
        else:
            eps = torch.randn(m, d, generator=gen).to(device)
            x = mu.unsqueeze(0) + eps @ L.t()               # sample N(mu, Sigma)
        x = _project_stealth(x, prior)
        for _ in range(cfg["zo_epochs"]):
            g, cur = _cge_grad(model, data, x, v_edges, i_edges, cfg, device)
            if cur < 0:
                best = x
                flipped = True
                break
            g_tan = (g @ V_k) @ V_k.t()                      # keep only tangent component
            x = x - lr * g_tan                               # descend CW within manifold
            x = _project_stealth(x, prior)                   # HARD stealth guarantee
        best = x
        if flipped:
            break
    return best.detach(), flipped


# ============================================================
# Channel (b): orthogonal norm-free topology — greedy + fraction-of-N anchors
# ============================================================

def om2c_attack_graph(model, data, feats0, prior, cfg, device):
    """Two-channel attack on one graph, in rho = edges/N budget units.

    1. Feature channel optimized on the full-edge topology (channel a).
    2. Topology channel carries the readout mass (channel b):
         - adaptive greedy (low-degree-targeted) -> smallest rho that flips, cheaply.
         - fraction-of-N anchors -> rho=1.0 endpoint == full-edge (the RS-Pool /
           dilution-beating regime). full_flip is read at rho=1.0.

    Returns: dict(rho_flip, full_flip, n_inter, greedy_edges).
    """
    true_label = data.y.item()
    n = data.num_nodes
    m = feats0.size(0)
    _SGDI._FEAT_SCALE = _resolve_feat_scale(cfg.get("feat_scale", 1.0), n)
    inter_full = [(i, j) for i in range(m) for j in range(i + 1, m)] \
        if cfg["use_inter_edges"] else []

    # ---- channel (a): on-manifold, norm-bounded feature opt on full topology ----
    all_v = [(i, v) for i in range(m) for v in range(n)]
    feats, _fl = om2c_feature_opt(model, data, feats0, all_v, inter_full,
                                  prior, cfg, device)
    _record_in_ball(feats, prior)

    # ---- channel (b): adaptive greedy (cheap/sparse regime) ----
    victim_edges, inter_edges = [], []
    greedy_edges = math.inf
    if not cfg.get("skip_greedy"):
        pool = _candidate_pool(data, cfg["max_edges"], cfg["pool_mult"])
        used_v, used_i = set(), set()
        for _ in range(cfg["max_edges"]):
            cands = [("v", (i, v)) for i in range(m) for v in pool
                     if (i, v) not in used_v]
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
    rho_greedy = (greedy_edges / n) if math.isfinite(greedy_edges) else math.inf

    # ---- channel (b): fraction-of-N anchors (expensive regime, up to full edge) ----
    deg_order = torch.argsort(
        torch.bincount(data.edge_index[0], minlength=n).float()).tolist()
    rho_anchor = math.inf
    full_flip = False
    anchor_list = []
    for rho in cfg["frac_anchors"]:
        kk = max(1, math.ceil(rho * n))
        targets = deg_order[:kk]
        v_edges = [(i, t) for i in range(m) for t in targets]
        anchor_list.append(_build(data, feats, v_edges, inter_full, device))
    a_margins = _eval_graphs(model, anchor_list, true_label, device)
    for idx, rho in enumerate(cfg["frac_anchors"]):
        if a_margins[idx].item() < 0:
            rho_anchor = min(rho_anchor, rho)
    if a_margins[-1].item() < 0:        # rho=1.0 endpoint == full-edge anchor
        full_flip = True

    rho_flip = min(rho_greedy, rho_anchor)
    return {
        "rho_flip": rho_flip,
        "full_flip": bool(full_flip),
        "n_inter": len(inter_edges),
        "greedy_edges": greedy_edges if math.isfinite(greedy_edges) else None,
    }


# ============================================================
# Run entry points — same contract as attack.py / attack_flow.py
# ============================================================

def run_attack_sweep(model, test_graphs, device, cfg=None):
    """Rich record for Pareto sweeps. Per-graph rho_flip (edges/N) + query cost."""
    cfg = {**CONFIG, **(cfg or {})}
    _SGDI.QUERY_COUNT = 0
    _reset_stats()
    m = cfg["node_budget"]
    stats = data_stats(test_graphs, device)
    prior = om2c_prior(test_graphs, device, cfg["tangent_dim"], cfg["sigma_mult"],
                       eps=cfg["prior_eps"])
    mu = prior[0]

    records = []
    for data in test_graphs:
        data = data.to(device)
        feats0 = mu.unsqueeze(0).repeat(m, 1)        # start every node at the manifold mode
        records.append(om2c_attack_graph(model, data, feats0, prior, cfg, device))

    n = len(records) or 1
    frac = (OM2C_STATS["in_ball"] / OM2C_STATS["total"]) if OM2C_STATS["total"] else float("nan")
    return {
        "asr_full": sum(r["full_flip"] for r in records) / n,   # rho=1.0 (mass regime)
        "rho_to_flip": [r["rho_flip"] for r in records],
        "greedy_edges": [r["greedy_edges"] for r in records],
        "n_inter": [r["n_inter"] for r in records],
        "queries": _SGDI.QUERY_COUNT,
        "n_graphs": len(records),
        "stealth_in_ball_frac": frac,
    }


def run_attack(model, test_graphs, device):
    """run.py-compatible: float ASR at the full-edge anchor (baseline-comparable)."""
    if not test_graphs:
        return 0.0
    return run_attack_sweep(model, test_graphs, device, CONFIG)["asr_full"]


# ============================================================
# Module sanity check (run: python attack_om2c.py)
# ============================================================

if __name__ == "__main__":
    print("attack_om2c.py — OM2C: Orthogonal Mass-perp-Manifold Composition")
    print(f"CONFIG keys: {list(CONFIG.keys())}")
    print(f"sgdi base:   {getattr(_SGDI, '__file__', '?')}")
    # Tiny synthetic prior + projection sanity check (no model needed).
    from torch_geometric.data import Data
    torch.manual_seed(0)
    gs = [Data(x=torch.rand(8, 3),
               edge_index=torch.randint(0, 8, (2, 12)),
               y=torch.tensor([0])) for _ in range(4)]
    pr = om2c_prior(gs, "cpu", k=2, sigma_mult=2.0)
    mu, cov_inv, L, thr, V_k, r2s = pr
    x = torch.randn(3, 3) * 5.0
    xp = _project_stealth(x, pr)
    diff = xp - mu
    m2 = (diff @ cov_inv * diff).sum(dim=1)
    print(f"V_k shape={tuple(V_k.shape)}  r_2sigma={r2s:.3f}  thr={thr:.3f}")
    print(f"projected norms: {[round(v,3) for v in xp.norm(dim=1).tolist()]}  (<= {r2s:.3f})")
    print(f"projected sq-Maha: {[round(v,3) for v in m2.tolist()]}  (<= {thr:.3f})")
