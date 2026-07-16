#!/usr/bin/env python3
"""attack_om2c_fm.py — OM2C+FM: flow-matching feature generation + OM2C's sparse-greedy
degree-1 edge channel. The MISSING CELL of the 2x2:

    feature {ZO-tangent, FM} x edge {sparse-greedy-deg1, topk}
      ZO-tangent + sparse-greedy = OM2C                       (floor5 ~22.7%, query-EXPENSIVE)
      FM        + topk           = FM-nb10-eb1-lownorm control (floor5 ~8.7%,  query-cheap)
      FM        + sparse-greedy  = OM2C+FM  <-- THIS MODULE

Hypotheses this module is built to test (driver-level, see eval_pg_floor_om2c_fm.py):
  H_mech: OM2C+FM floor5 ~= OM2C's 22.7% (>> the 8.7% FM+topk control) => the EDGE channel
          (degree-1 sparse-greedy mass), not the feature-gen method, is the source of the
          floor; feature generation is interchangeable.
  H_cost: OM2C+FM total queries << OM2C (FM feature gen is cheaper than OM2C's ZO feature
          opt) => best-of-both.

ADDITIVE COMPOSITION — no attack/model/defense code is modified. This module imports and
reuses, VERBATIM:
  * Feature channel: attack_flow.py's velocity-ODE on-manifold feature generation
    (_estimate_g_adv / lambda_at / mahalanobis_project_rowwise / construct_perturbed_graph
    / _flips). The ONLY change vs attack_flow._attack_single_flow is the return value:
    we return the GENERATED feature tensor (m, d) instead of a flip bool, so OM2C's edge
    channel can deploy it. Then OM2C's hard stealth projection (_project_stealth: tangent
    ∩ 2σ-ball ∩ Mahalanobis-ball) is applied so the injected features are guaranteed
    on-manifold (same feasible set OM2C's own feature channel guarantees).
  * Edge channel: OM2C's sparse-greedy degree-1 channel — the greedy add-best-edge loop
    over the low-degree candidate pool (O._candidate_pool / O._build / O._eval_graphs).
    Copied VERBATIM from eval_paper_grade_floor.om2c_greedy_flip; the ONLY line that
    differs is the feature-generation source (FM instead of O.om2c_feature_opt).

Threat model unchanged: strict black-box (CW margin only), injection-only, fs=1.0 (no norm
amplification — mass comes from edges, not feature norm). PROTEINS/GCN, node_budget=10.
"""

import attack_flow as FM            # FM velocity-ODE feature channel (reused verbatim)
import attack_om2c as O             # OM2C stealth projection + sparse-greedy edge channel

import torch


def fm_generate_features(model, data, m, targets, fs, fm_prior, device,
                         cfg=None):
    """Generate m injected-node features with attack_flow's velocity-ODE.

    Faithful re-implementation of attack_flow._attack_single_flow's ODE loop (rectified-
    flow transport x0~N(0,I) -> on-manifold anchor x1*, with annealed ZO adversarial
    guidance and a HARD Mahalanobis projection after every Euler step), multi-restart,
    returning the GENERATED FEATURE (m, d) instead of a flip bool. Every ODE primitive
    (_estimate_g_adv, lambda_at, mahalanobis_project_rowwise, _flips) is attack_flow's,
    used unchanged. The feature is generated on the FULL-edge topology (targets =
    arange(N)), matching OM2C's own feature channel which optimizes on all_v full edges.

    Returns the iterate that flips on the full topology if one is found (most adversarial),
    else the final-restart ODE endpoint. All iterates are in-ball (pre-scale).
    """
    cfg = cfg or FM.CONFIG
    mu, cov_inv, L, thr = fm_prior
    n_feat = data.x.size(1)
    T = cfg["T_flow"]
    dt = 1.0 / T

    last = None
    for r in range(cfg["n_restarts"]):
        seed = r * 1000 + data.num_nodes
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x0 = torch.randn(m, n_feat, generator=gen).to(device)
        if r == 0:
            x1_star = mu.unsqueeze(0).repeat(m, 1)            # density mode
        else:
            eps = torch.randn(m, n_feat, generator=gen).to(device)
            x1_star = mu.unsqueeze(0) + eps @ L.t()           # draw N(mu, Sigma)
        x1_star = FM.mahalanobis_project_rowwise(x1_star, mu, cov_inv, thr)
        v_prior = x1_star - x0                                # analytic rectified velocity

        x = FM.mahalanobis_project_rowwise(x0, mu, cov_inv, thr)
        for step in range(T):
            t = step * dt
            if FM._flips(model, data, x, targets, fs, device):
                return x.detach()                            # flipping iterate (best)
            lam = FM.lambda_at(t, cfg)
            g_adv = FM._estimate_g_adv(model, data, x, targets, fs, cfg, device)
            if cfg["integrator"] == "heun":
                x_pred = x + dt * (v_prior + lam * g_adv)
                x_pred = FM.mahalanobis_project_rowwise(x_pred, mu, cov_inv, thr)
                lam2 = FM.lambda_at(t + dt, cfg)
                g_adv2 = FM._estimate_g_adv(model, data, x_pred, targets, fs, cfg, device)
                x = x + 0.5 * dt * ((v_prior + lam * g_adv) + (v_prior + lam2 * g_adv2))
            else:                                            # euler
                x = x + dt * (v_prior + lam * g_adv)
            x = FM.mahalanobis_project_rowwise(x, mu, cov_inv, thr)   # HARD on-manifold
        last = x.detach()
        if FM._flips(model, data, last, targets, fs, device):
            return last
    return last


def om2c_fm_greedy_flip(model, data, om2c_prior, fm_prior, cfg, device):
    """OM2C+FM on one graph: FM feature channel + OM2C sparse-greedy degree-1 edge channel.

    Mirrors eval_paper_grade_floor.om2c_greedy_flip EXACTLY, except the feature-generation
    line: O.om2c_feature_opt (ZO-tangent) is replaced by FM feature gen + O._project_stealth.
    The greedy edge loop below is OM2C's degree-1 sparse-greedy channel verbatim.

    Returns (flipped, n_edges_at_flip|None, max_inj_degree).
    """
    n = data.num_nodes
    m = cfg["node_budget"]
    O._SGDI._FEAT_SCALE = O._resolve_feat_scale(cfg.get("feat_scale", 1.0), n)

    # ---- channel (a): FM feature generation on full topology, fs=1.0, THEN OM2C stealth ----
    fs = O._SGDI._FEAT_SCALE                                  # == 1.0 (OM2C regime)
    targets = torch.arange(n, device=device)                 # full-edge topology for gen
    fm_feats = fm_generate_features(model, data, m, targets, fs, fm_prior, device)
    feats = O._project_stealth(fm_feats, om2c_prior)          # tangent ∩ 2σ ∩ Maha guarantee
    O._record_in_ball(feats, om2c_prior)

    # ---- channel (b): OM2C sparse-greedy degree-1 edge channel (verbatim) ----
    pool = O._candidate_pool(data, cfg["max_edges"], cfg["pool_mult"])
    victim_edges, inter_edges = [], []
    used_v, used_i = set(), set()
    flipped = False
    ne = None
    true_label = data.y.item()
    for _ in range(cfg["max_edges"]):
        cands = [("v", (i, v)) for i in range(m) for v in pool if (i, v) not in used_v]
        if cfg["use_inter_edges"] and m >= 2:
            cands += [("e", (i, j)) for i in range(m) for j in range(i + 1, m)
                      if frozenset({i, j}) not in used_i]
        if not cands:
            break
        dl = [O._build(data, feats, victim_edges + [p], inter_edges, device) if k == "v"
              else O._build(data, feats, victim_edges, inter_edges + [p], device)
              for k, p in cands]
        margins = O._eval_graphs(model, dl, true_label, device)
        b = int(torch.argmin(margins).item())
        k, p = cands[b]
        (victim_edges if k == "v" else inter_edges).append(p)
        (used_v if k == "v" else used_i).add(p if k == "v" else frozenset(p))
        if margins[b].item() < 0:
            flipped = True
            ne = len(victim_edges) + len(inter_edges)
            break
    deg = [0] * m
    for (i, v) in victim_edges:
        deg[i] += 1
    for (i, j) in inter_edges:
        deg[i] += 1
        deg[j] += 1
    return flipped, ne, (max(deg) if deg else 0)


if __name__ == "__main__":
    print("attack_om2c_fm.py — OM2C+FM (FM features + OM2C sparse-greedy degree-1 edges)")
    print(f"FM CONFIG keys:   {list(FM.CONFIG.keys())}")
    print(f"OM2C CONFIG keys: {list(O.CONFIG.keys())}")
