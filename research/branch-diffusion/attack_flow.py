#!/usr/bin/env python3
"""
attack_flow.py — N28 Flow-Matching Stealth Node-Injection (variant A: analytic FM)

Drop-in replacement for attack.py with the SAME
    run_attack(model, test_graphs, device) -> float
signature, mirroring attack_diffusion.py (N23) so run.py / executor.py and the
SGDI defense harness work unchanged. This is the flow-matching analogue of the
non-parametric diffusion attack.

Core idea (design spec 2026-06-16, §3-§4 variant A):
  Cast injected-node-feature synthesis as a flow-matching ODE transporting noise
  x0 ~ N(0,I) to an on-manifold adversarial feature x1 over t in [0,1] in T_flow
  Euler steps:

      dx/dt = v(x, t) = v_prior(x, t) + lambda(t) * g_adv(x)

  - v_prior : analytic rectified / OT straight-line velocity toward the clean-
              feature Gaussian fit (mu, Sigma). For a target anchor x1* on the
              (mu, Sigma) manifold and noise x0, the rectified-flow conditional
              velocity is constant along the straight path:  v_prior = x1* - x0.
              NO trained network (variant A is non-parametric, pure black-box).
  - g_adv   : black-box ZO estimate of  -grad_x CW(victim(inject(G, x)))  via the
              reused zo_score_cge / zo_score_rgf (descent on the CW margin).
  - lambda(t): annealed guidance schedule (explore -> exploit), a CONFIG knob.
  - Per-step Mahalanobis-ball projection (reused) keeps injected feats inside the
              clean 99th-pct Mahalanobis ball  -> on-manifold stealth.

Why FM over diffusion: rectified/OT straight paths need few ODE steps (~10) vs
diffusion's T~50, and each step costs ZO queries -> ~order-fewer victim queries.

Injection-only invariant (unchanged from SGDI / N23):
  construct_perturbed_graph() is pure: original_data in, new Data out. G is never
  mutated; the flow state x lives outside the original feature block; injected
  edges only connect injected<->existing / injected<->injected.

Threat model: strict black-box. The victim is queried ONLY for the CW loss on its
logits (no white-box gradients, no embedding hooks). Apples-to-apples with
attack_diffusion.py.

NOTE on the branch-diffusion attack.py name collision: in this directory the
local attack.py is the *SGDI* variant (it owns data_prior / mahalanobis_project_
rowwise but NOT the injection helpers). The canonical injection helpers
(construct_perturbed_graph, batch_loss, ...) live in the main katze repo's
attack.py. We therefore resolve the helpers from BOTH files explicitly below.
"""

import math
import os
import sys
import importlib.util

import torch

# ============================================================
# Helper resolution (3 sources — see module docstring)
#   * injection helpers  -> canonical katze attack.py
#   * Mahalanobis / prior -> local SGDI attack.py
#   * ZO estimators       -> attack_diffusion.py
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # register before exec (for intra-pkg imports)
    spec.loader.exec_module(mod)
    return mod


def _resolve_canonical_attack():
    """Return the helper-rich katze attack module (with construct_perturbed_graph).

    Prefer whatever `import attack` already resolves to; if that module is the
    local SGDI variant (no injection helpers), load the canonical one by path and
    register it as `attack` so attack_diffusion's `from attack import ...` works.
    """
    a = sys.modules.get("attack")
    if a is None:
        try:
            import attack as a  # noqa: F401
        except Exception:
            a = None
    if a is not None and hasattr(a, "construct_perturbed_graph"):
        return a
    path = os.environ.get("KATZE_ATTACK", "/workspace/autoresearch-katze/attack.py")
    if not os.path.exists(path):
        raise ImportError(
            "Could not locate the canonical katze attack.py (with "
            "construct_perturbed_graph). Set $KATZE_ATTACK to its path.")
    canon = _load_module("attack", path)          # register as 'attack'
    return canon


def _resolve_sgdi_attack():
    """Return the local SGDI attack module (owns data_prior / mahalanobis_project_
    rowwise). It is the attack.py sitting next to this file."""
    local_path = os.path.join(_HERE, "attack.py")
    a = sys.modules.get("attack")
    if a is not None and hasattr(a, "data_prior") and \
            hasattr(a, "mahalanobis_project_rowwise"):
        return a
    return _load_module("sgdi_attack_local", local_path)


_CANON = _resolve_canonical_attack()
construct_perturbed_graph = _CANON.construct_perturbed_graph
batch_loss = _CANON.batch_loss
get_prediction = _CANON.get_prediction
select_targets_topk = _CANON.select_targets_topk
select_targets_spectral = _CANON.select_targets_spectral

_SGDI = _resolve_sgdi_attack()
data_prior = _SGDI.data_prior                       # (mu, Sigma^-1, L, thr)
mahalanobis_project_rowwise = _SGDI.mahalanobis_project_rowwise

# zo_score_cge / zo_score_rgf are defined in attack_diffusion.py; importing it now
# works because the canonical 'attack' is already registered in sys.modules above.
from attack_diffusion import zo_score_cge, zo_score_rgf  # noqa: E402


# ============================================================
# CONFIG — N28 specific (mirrors attack_diffusion.py's shape)
# ============================================================

CONFIG = {
    # Flow-matching integration
    "T_flow": 10,                 # ODE steps (replaces diffusion's T_steps=50)
    "integrator": "euler",        # "euler" (1st order) | "heun" (2nd order, 2x ZO)
    "prior_eps": 1e-3,            # ridge on the clean-feature covariance estimate

    # Guidance schedule lambda(t), t in [0,1]
    "lambda_schedule": "linear",  # "constant" | "linear" (low->high) | "cosine"
    "lambda_min": 0.1,            # guidance at t=0 (explore)
    "lambda_max": 1.0,            # guidance at t=1 (exploit)

    # ZO score estimation (same knobs as attack_diffusion.py)
    "sigma": 5e-3,                # ZO smoothing radius
    "grad_method": "cge",         # "cge" | "rgf"
    "rgf_samples": 20,            # only used if grad_method == "rgf"

    # Inherited from baseline / diffusion (executor stays consistent)
    "feat_scale": "auto_x2",
    "loss_type": "cw",
    "kappa": -0.1,
    "edge_strategy": "full",
    "node_budget": 1,
    "spectral_top_k_eig": 10,

    # Multi-restart (cheap: each restart is only T_flow steps)
    "n_restarts": 3,
}


# ============================================================
# Stealth bookkeeping (opt-in; cheap). The smoke test reads this to report the
# fraction of injected features that land inside the clean Mahalanobis ball.
# ============================================================

FLOW_STATS = {"in_ball": 0, "total": 0}


def _reset_stats():
    FLOW_STATS["in_ball"] = 0
    FLOW_STATS["total"] = 0


def _record_in_ball(x_raw, mu, cov_inv, thr):
    """Tally how many rows of the (pre-scale) flow state sit inside the ball.

    Uses a relative tolerance: hard projection lands outside points exactly on the
    boundary (sq-Mahalanobis == thr), so float round-off can leave m2 a hair above
    thr; an absolute 1e-9 cut would spuriously flag those boundary points.
    """
    diff = x_raw - mu
    m2 = (diff @ cov_inv * diff).sum(dim=1)
    tol = thr * (1.0 + 1e-5) + 1e-9
    FLOW_STATS["in_ball"] += int((m2 <= tol).sum().item())
    FLOW_STATS["total"] += int(x_raw.shape[0])


# ============================================================
# Guidance schedule lambda(t)
# ============================================================

def lambda_at(t, cfg):
    """Annealed guidance strength at flow time t in [0, 1]."""
    lo, hi = cfg["lambda_min"], cfg["lambda_max"]
    mode = cfg["lambda_schedule"]
    if mode == "constant":
        return hi
    if mode == "linear":
        return lo + (hi - lo) * t
    if mode == "cosine":
        # smooth low -> high ramp
        return lo + (hi - lo) * 0.5 * (1.0 - math.cos(math.pi * t))
    raise ValueError(f"Unknown lambda_schedule: {mode}")


# ============================================================
# ZO adversarial drift  g_adv(x) = -grad_x CW(victim(inject(G, x)))
# ============================================================

def _estimate_g_adv(model, data, x, targets, fs, cfg, device):
    """Black-box ZO estimate of the descent direction on the CW margin.

    Returns g_adv with the SAME shape as x. zo_score_* return grad of CW; we
    negate to descend (untargeted CW minimization), matching attack_diffusion.
    """
    if cfg["grad_method"] == "cge":
        _, grad = zo_score_cge(
            model, data, x, targets,
            cfg["sigma"], cfg["kappa"], fs, device, cfg["loss_type"])
    else:
        _, grad = zo_score_rgf(
            model, data, x, targets,
            cfg["sigma"], cfg["rgf_samples"], cfg["kappa"], fs, device,
            cfg["loss_type"])
    return -grad


# ============================================================
# Flow-matching ODE for a single graph
# ============================================================

def _flips(model, data, x, targets, fs, device):
    """True iff injecting (pre-scale) feature x flips the victim's prediction."""
    perturbed = construct_perturbed_graph(data, x, targets, fs)
    perturbed.batch = torch.zeros(perturbed.num_nodes, dtype=torch.long,
                                  device=device)
    with torch.no_grad():
        logits = model(perturbed)
    return get_prediction(logits) != data.y.item()


def _attack_single_flow(model, data, targets, fs, cfg, n_feat, prior, device,
                        seed=0):
    """One flow-matching ODE run on a single graph.

    Rectified-flow transport from noise x0 ~ N(0,I) to an on-manifold anchor x1*
    on the (mu, Sigma) ball, with annealed ZO adversarial guidance and a HARD
    Mahalanobis projection after every Euler (or Heun) step.

    Returns True if any iterate (or the endpoint) flips the victim, else False.

    Injection-only audit: `data` is read-only; it only ever enters
    construct_perturbed_graph()/zo_score_*(), both pure. x lives outside the
    original feature block. (ok)
    """
    mu, cov_inv, L, thr = prior
    m = cfg["node_budget"]
    T = cfg["T_flow"]
    dt = 1.0 / T

    gen = torch.Generator(device="cpu").manual_seed(seed)
    # noise source x0 ~ N(0, I)
    x0 = torch.randn(m, n_feat, generator=gen).to(device)
    # on-manifold target anchor x1*: mode (mu) for the first restart, a Gaussian
    # draw from N(mu, Sigma) for the rest (mirrors SGDI's restart sampling).
    if seed == 0:
        x1_star = mu.unsqueeze(0).repeat(m, 1)
    else:
        eps = torch.randn(m, n_feat, generator=gen).to(device)
        x1_star = mu.unsqueeze(0) + eps @ L.t()
    x1_star = mahalanobis_project_rowwise(x1_star, mu, cov_inv, thr)

    # analytic rectified/OT velocity is CONSTANT along the straight path
    v_prior = x1_star - x0

    # Stealth contract: the feature that is ever injected/queried must lie inside
    # the clean Mahalanobis ball, so we project the iterate BEFORE every check and
    # ZO call (incl. the noise init). This mirrors SGDI, which never evaluates an
    # off-manifold point. The HARD projection is the on-manifold stealth guarantee.
    x = mahalanobis_project_rowwise(x0, mu, cov_inv, thr)
    for step in range(T):
        t = step * dt
        # quick success check on the current (in-ball, pre-scale) iterate
        if _flips(model, data, x, targets, fs, device):
            _record_in_ball(x.detach(), mu, cov_inv, thr)
            return True

        lam = lambda_at(t, cfg)
        g_adv = _estimate_g_adv(model, data, x, targets, fs, cfg, device)

        if cfg["integrator"] == "heun":
            # predictor
            x_pred = x + dt * (v_prior + lam * g_adv)
            x_pred = mahalanobis_project_rowwise(x_pred, mu, cov_inv, thr)
            lam2 = lambda_at(t + dt, cfg)
            g_adv2 = _estimate_g_adv(model, data, x_pred, targets, fs, cfg, device)
            v1 = v_prior + lam * g_adv
            v2 = v_prior + lam2 * g_adv2
            x = x + 0.5 * dt * (v1 + v2)
        else:  # euler
            x = x + dt * (v_prior + lam * g_adv)

        # HARD on-manifold stealth projection after the step
        x = mahalanobis_project_rowwise(x, mu, cov_inv, thr)

    # final endpoint check
    _record_in_ball(x.detach(), mu, cov_inv, thr)
    return _flips(model, data, x, targets, fs, device)


# ============================================================
# Public entry point — same signature as attack.py / attack_diffusion.py
# ============================================================

def run_attack(model, test_graphs, device):
    """Run N28 analytic flow-matching injection on all test graphs, return ASR."""
    cfg = CONFIG
    if not test_graphs:
        return 0.0
    n_feat = test_graphs[0].x.size(1)
    _reset_stats()

    # Clean-feature Gaussian (mu, Sigma) fit + 99th-pct Mahalanobis radius — fit
    # ONCE over the clean test graphs (reused SGDI estimator). No victim queries.
    prior = data_prior(test_graphs, device, eps=cfg["prior_eps"])

    n_success = 0
    for data in test_graphs:
        data = data.to(device)

        # feat_scale resolution (same as baseline / diffusion)
        fs = cfg["feat_scale"]
        if fs == "auto":
            fs = math.sqrt(data.num_nodes)
        elif fs == "auto_x2":
            fs = 2.0 * math.sqrt(data.num_nodes)
        elif fs == "auto_x3":
            fs = 3.0 * math.sqrt(data.num_nodes)
        elif fs == "auto_linear":
            fs = float(data.num_nodes)

        # Edge target selection (fixed per-strategy, same as diffusion)
        if cfg["edge_strategy"] == "full":
            targets = torch.arange(data.num_nodes, device=device)
        elif cfg["edge_strategy"] == "spectral":
            targets = select_targets_spectral(
                data, cfg["node_budget"], cfg["spectral_top_k_eig"], device)
        else:
            targets = select_targets_topk(data, cfg["node_budget"])

        success = False
        for r in range(cfg["n_restarts"]):
            seed = r * 1000 + data.num_nodes
            if _attack_single_flow(model, data, targets, fs, cfg, n_feat,
                                    prior, device, seed=seed):
                success = True
                break
        if success:
            n_success += 1

    return n_success / len(test_graphs)


# ============================================================
# Module sanity check (run: python attack_flow.py)
# ============================================================

if __name__ == "__main__":
    print("attack_flow.py — N28 Flow-Matching (variant A: analytic)")
    print(f"CONFIG keys: {list(CONFIG.keys())}")
    print(f"canonical attack: {getattr(_CANON, '__file__', '?')}")
    print(f"sgdi attack:      {getattr(_SGDI, '__file__', '?')}")
    _cfg = CONFIG
    for mode in ("constant", "linear", "cosine"):
        _cfg2 = dict(_cfg, lambda_schedule=mode)
        vals = [round(lambda_at(t / 4, _cfg2), 3) for t in range(5)]
        print(f"lambda[{mode:8s}] t=0..1: {vals}")
