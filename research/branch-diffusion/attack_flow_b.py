#!/usr/bin/env python3
"""
attack_flow_b.py — FM-Velosteal: Flow-Matching TRAINED velocity field for
manifold-resident node injection (N28 variant B).

Drop-in replacement for attack.py with the SAME
    run_attack(model, test_graphs, device) -> float
signature, mirroring attack_flow.py (variant A, analytic) and attack_diffusion.py
(N23, discrete diffusion) so run.py / executor.py and the SGDI defense harness work
unchanged.

================================================================================
WHAT MAKES THIS "VARIANT B" (and distinct from the two logged cousins)
================================================================================
  * attack_diffusion.py (N23)  : non-parametric DISCRETE diffusion (no network).
  * attack_flow.py     (var A)  : flow-matching with an ANALYTIC rectified/OT
                                  velocity v_prior = x1* - x0 (constant, no net).
  * attack_flow_b.py   (var B)  : flow-matching with a *TRAINED* velocity field
                                  v_theta(x, t), a small MLP fit on the clean node
                                  features by the rectified-flow / OT-CFM loss.

Why train the field? The analytic variant transports noise onto a *Gaussian*
ellipsoid (mu, Sigma) — the right second-order shape, but blind to higher-order
structure of the data cloud (multimodality, binary support, manifold curvature).
A trained v_theta learns the actual conditional transport toward the data
manifold, so injected features land *on the manifold* rather than merely inside
the covariance ball. This is the "manifold-resident" stealth claim: it should
evade NormFilter + FeatClip + a HELD-OUT spectral detector simultaneously,
because the feature looks like a genuine clean node, not a norm-capped artifact.

Mechanism at attack time (per graph, T_flow Euler/Heun steps):

    dx/dt = v_theta(x, t) + lambda(t) * g_adv(x)

  - v_theta : TRAINED velocity field (the manifold drift).  No victim queries.
  - g_adv   : black-box ZO estimate of  -grad_x CW(victim(inject(G, x)))  via the
              reused zo_score_cge / zo_score_rgf (descent on the CW margin).
  - lambda(t): annealed guidance schedule (explore -> exploit), a CONFIG knob.
  - Per-step HARD Mahalanobis-ball projection (reused) is the on-manifold stealth
              guarantee against any Mahalanobis-style / spectral detector — the
              same projection SGDI and variant A use, so the feature that is ever
              injected or queried provably sits inside the clean 99th-pct ball.

Training is BLACK-BOX-SAFE: the velocity field is fit ONLY on clean node features
(the same tensor data_prior() consumes); the victim model is never queried during
training. The field is cached per process (keyed on a feature fingerprint) and may
optionally be persisted to / loaded from CONFIG["ckpt_path"].

Injection-only invariant (unchanged from SGDI / N23 / variant A):
  construct_perturbed_graph() is pure: original_data in, new Data out. G is never
  mutated; the flow state x lives outside the original feature block; injected
  edges only connect injected<->existing / injected<->injected.

SCOPE NOTE (from the idea card): this helps detector-style defenses only. It does
NOT touch the RS-Pool architectural floor — even if the stealth hypothesis holds,
the worst-case pooling floor is unchanged by construction.
"""

import math
import os
import sys
import importlib.util

import torch

# ============================================================
# Helper resolution (3 sources — identical strategy to attack_flow.py)
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
    """Return the helper-rich katze attack module (with construct_perturbed_graph)."""
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

# Velocity network + flow-matching loss + checkpoint I/O live in flow_net.py (shared
# with the offline trainer train_flow.py). VelocityField is re-exported here for the
# smoke / battery scripts that reference attack_flow_b.VelocityField directly.
from flow_net import (  # noqa: E402,F401
    VelocityField, train_field_on_features, load_checkpoint,
)


# ============================================================
# CONFIG — N28 variant B (mirrors attack_flow.py's shape; adds train knobs)
# ============================================================

CONFIG = {
    # Flow-matching integration
    "T_flow": 10,                 # ODE steps
    "integrator": "euler",        # "euler" (1st order) | "heun" (2nd order, 2x ZO)
    "prior_eps": 1e-3,            # ridge on the clean-feature covariance estimate

    # Trained velocity field v_theta(x, t)
    "vf_hidden": 128,             # MLP hidden width
    "vf_layers": 3,               # number of hidden layers
    "vf_time_dim": 32,            # Fourier time-embedding dimension
    "vf_train_steps": 400,        # SGD steps for the flow-matching fit
    "vf_batch": 256,              # rows per flow-matching minibatch
    "vf_lr": 1e-3,                # Adam lr for the field
    "vf_seed": 0,                 # field-init / minibatch RNG seed
    "ckpt_path": None,            # if set: load if present, else train + save here

    # Guidance schedule lambda(t), t in [0,1]
    "lambda_schedule": "linear",  # "constant" | "linear" (low->high) | "cosine"
    "lambda_min": 0.1,            # guidance at t=0 (explore)
    "lambda_max": 1.0,            # guidance at t=1 (exploit)

    # ZO score estimation (same knobs as attack_diffusion / attack_flow)
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
# Stealth bookkeeping (opt-in; cheap). Same contract as attack_flow.py — the smoke
# test reads this to report the fraction of injected features inside the ball.
# ============================================================

FLOW_STATS = {"in_ball": 0, "total": 0}


def _reset_stats():
    FLOW_STATS["in_ball"] = 0
    FLOW_STATS["total"] = 0


def _record_in_ball(x_raw, mu, cov_inv, thr):
    diff = x_raw - mu
    m2 = (diff @ cov_inv * diff).sum(dim=1)
    tol = thr * (1.0 + 1e-5) + 1e-9
    FLOW_STATS["in_ball"] += int((m2 <= tol).sum().item())
    FLOW_STATS["total"] += int(x_raw.shape[0])


# ============================================================
# Trained velocity field v_theta(x, t)
#
# The network (VelocityField), the flow-matching loss, and checkpoint I/O are defined
# ONCE in flow_net.py and shared with the OFFLINE trainer train_flow.py. The preferred
# (and threat-model-clean) workflow is:
#     1. train OFFLINE on benign features:  python train_flow.py --dataset ... --out X
#     2. point CONFIG["ckpt_path"] = "X"  -> the field is LOADED, not refit at attack time.
# The in-process training path below is a convenience fallback (e.g. for the smoke test)
# and is equally black-box safe: it only ever consumes clean node features, never the
# victim. Both paths are cached per process so restarts reuse one field.
# ============================================================

# ----- module-level cache so restarts / repeated run_attack calls reuse the field
_FIELD_CACHE = {}


def _feature_fingerprint(X):
    """Cheap, stable key for the clean-feature tensor (rows, dim, mass)."""
    return (int(X.shape[0]), int(X.shape[1]),
            round(float(X.sum().item()), 4),
            round(float((X * X).sum().item()), 4))


def train_velocity_field(test_graphs, device, cfg):
    """Obtain v_theta for these graphs' feature space (load offline ckpt or fit).

    Resolution order:
      1. CONFIG["ckpt_path"] exists  -> LOAD it (offline-trained field; preferred).
                                        Raises on a feature-dim mismatch.
      2. else                        -> fit in-process on the clean node features via
                                        flow_net's rectified-flow / OT-CFM loss.

    BLACK-BOX SAFE either way: only clean node features are used; the victim is never
    queried during training. Cached per process by feature fingerprint + arch.
    """
    X = torch.cat([g.x for g in test_graphs], dim=0).to(device).float()
    in_dim = X.size(1)
    key = (_feature_fingerprint(X), in_dim, cfg["vf_hidden"], cfg["vf_layers"],
           cfg["vf_time_dim"], cfg["vf_seed"], cfg.get("ckpt_path"))

    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    ckpt = cfg.get("ckpt_path")
    if ckpt and os.path.exists(ckpt):
        # offline-trained field — the threat-model-clean path. load_checkpoint rebuilds
        # the net from the stored arch and guards the feature-dim footgun.
        net, _meta = load_checkpoint(ckpt, device=device, expect_in_dim=in_dim)
        _FIELD_CACHE[key] = net
        return net

    # fallback: fit in-process (smoke / no-ckpt convenience). Single source of truth
    # for the network + loss is flow_net.train_field_on_features.
    net, _hist = train_field_on_features(
        X, device, hidden=cfg["vf_hidden"], n_layers=cfg["vf_layers"],
        time_dim=cfg["vf_time_dim"], train_steps=cfg["vf_train_steps"],
        batch=cfg["vf_batch"], lr=cfg["vf_lr"], seed=cfg["vf_seed"])

    _FIELD_CACHE[key] = net
    return net


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
        return lo + (hi - lo) * 0.5 * (1.0 - math.cos(math.pi * t))
    raise ValueError(f"Unknown lambda_schedule: {mode}")


# ============================================================
# ZO adversarial drift  g_adv(x) = -grad_x CW(victim(inject(G, x)))
# ============================================================

def _estimate_g_adv(model, data, x, targets, fs, cfg, device):
    """Black-box ZO estimate of the descent direction on the CW margin."""
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


def _v_theta(net, x, t, device):
    """Trained manifold drift at flow state x, time t (no grad)."""
    with torch.no_grad():
        tt = torch.tensor([[float(t)]], device=device)
        return net(x, tt)


# ============================================================
# Flow-matching ODE for a single graph (trained field)
# ============================================================

def _flips(model, data, x, targets, fs, device):
    """True iff injecting (pre-scale) feature x flips the victim's prediction."""
    perturbed = construct_perturbed_graph(data, x, targets, fs)
    perturbed.batch = torch.zeros(perturbed.num_nodes, dtype=torch.long,
                                  device=device)
    with torch.no_grad():
        logits = model(perturbed)
    return get_prediction(logits) != data.y.item()


def _attack_single_flow(model, data, targets, fs, cfg, n_feat, prior, net,
                        device, seed=0):
    """One trained-flow ODE run on a single graph.

    Transports noise x0 ~ N(0, I) toward the data manifold under the TRAINED
    velocity v_theta, with annealed ZO adversarial guidance and a HARD Mahalanobis
    projection after every step. Returns True if any iterate (or the endpoint)
    flips the victim, else False.

    Injection-only audit: `data` is read-only; it only ever enters
    construct_perturbed_graph()/zo_score_*(), both pure. x lives outside the
    original feature block. (ok)
    """
    mu, cov_inv, L, thr = prior
    m = cfg["node_budget"]
    T = cfg["T_flow"]
    dt = 1.0 / T

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(m, n_feat, generator=gen).to(device)

    # Project the noise init into the ball: the feature that is ever queried must
    # be on-manifold (same invariant as SGDI / variant A).
    x = mahalanobis_project_rowwise(x0, mu, cov_inv, thr)
    for step in range(T):
        t = step * dt
        if _flips(model, data, x, targets, fs, device):
            _record_in_ball(x.detach(), mu, cov_inv, thr)
            return True

        lam = lambda_at(t, cfg)
        v_prior = _v_theta(net, x, t, device)
        g_adv = _estimate_g_adv(model, data, x, targets, fs, cfg, device)

        if cfg["integrator"] == "heun":
            x_pred = x + dt * (v_prior + lam * g_adv)
            x_pred = mahalanobis_project_rowwise(x_pred, mu, cov_inv, thr)
            t2 = t + dt
            lam2 = lambda_at(t2, cfg)
            v_prior2 = _v_theta(net, x_pred, t2, device)
            g_adv2 = _estimate_g_adv(model, data, x_pred, targets, fs, cfg, device)
            v1 = v_prior + lam * g_adv
            v2 = v_prior2 + lam2 * g_adv2
            x = x + 0.5 * dt * (v1 + v2)
        else:  # euler
            x = x + dt * (v_prior + lam * g_adv)

        x = mahalanobis_project_rowwise(x, mu, cov_inv, thr)

    _record_in_ball(x.detach(), mu, cov_inv, thr)
    return _flips(model, data, x, targets, fs, device)


# ============================================================
# Public entry point — same signature as attack.py / attack_flow.py
# ============================================================

def run_attack(model, test_graphs, device):
    """Run FM-Velosteal (trained flow-matching injection) on all test graphs.

    Returns ASR (float in [0, 1]). Honors run_attack(model, test_graphs, device).
    """
    cfg = CONFIG
    if not test_graphs:
        return 0.0
    n_feat = test_graphs[0].x.size(1)
    _reset_stats()

    # Clean-feature Gaussian (mu, Sigma) + 99th-pct ball — fit ONCE, no victim
    # queries. Same estimator as variant A / SGDI.
    prior = data_prior(test_graphs, device, eps=cfg["prior_eps"])

    # Trained velocity field — fit ONCE on clean features (cached, no victim
    # queries). This is the only thing that distinguishes variant B from variant A.
    net = train_velocity_field(test_graphs, device, cfg)

    n_success = 0
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
        for r in range(cfg["n_restarts"]):
            seed = r * 1000 + data.num_nodes
            if _attack_single_flow(model, data, targets, fs, cfg, n_feat,
                                    prior, net, device, seed=seed):
                success = True
                break
        if success:
            n_success += 1

    return n_success / len(test_graphs)


# ============================================================
# Module sanity check (run: python attack_flow_b.py)
# ============================================================

if __name__ == "__main__":
    print("attack_flow_b.py — FM-Velosteal (variant B: TRAINED velocity field)")
    print(f"CONFIG keys: {list(CONFIG.keys())}")
    print(f"canonical attack: {getattr(_CANON, '__file__', '?')}")
    print(f"sgdi attack:      {getattr(_SGDI, '__file__', '?')}")
    # tiny self-test of the field training + forward on synthetic 3-dim features
    from torch_geometric.data import Data
    g = torch.Generator().manual_seed(0)
    graphs = [Data(x=(torch.rand(20, 3, generator=g) > 0.5).float()) for _ in range(5)]
    net = train_velocity_field(graphs, "cpu", dict(CONFIG, vf_train_steps=50))
    out = net(torch.randn(4, 3), torch.rand(4, 1))
    print(f"velocity field out shape: {tuple(out.shape)}  (expected (4, 3))")
    for mode in ("constant", "linear", "cosine"):
        _cfg2 = dict(CONFIG, lambda_schedule=mode)
        vals = [round(lambda_at(t / 4, _cfg2), 3) for t in range(5)]
        print(f"lambda[{mode:8s}] t=0..1: {vals}")
