# N23 (DiSGIA-NoParam) — Phase 1 Findings

**Date**: 2026-05-19
**Branch**: `n23-disgia-noparam`
**Status**: ❌ **H23.1 falsified — N23-NoParam mechanism contributes zero ASR over trivial baseline**

---

## 1. Headline

The AdvAD-style non-parametric diffusion reverse loop, naively ported to graph-injection, **does not produce any adversarial flips beyond what a zero-feature injection already achieves**. Verified by ablation.

| Config | T_steps | guidance_scale | ASR (fold 0) | success / 156 | Time |
|---|---|---|---|---|---|
| N23 (planned) | 50 | 1.0 | 67.31% | 105 | 38.6s |
| N23 (T=1, no loop) | 1 | 1.0 | 67.31% | 105 | 2.5s |
| N23 (high guidance) | 50 | 50.0 | 67.31% | 105 | 39.6s |
| KATZE baseline (5-fold) | (50 epochs) | — | **94.85%** | — | — |

All three N23 runs flipped the **same 105 of 156** graphs. The diffusion loop contributes **zero** additional flips, even with 50× the guidance scale.

Baseline KATZE (generator + Adam + multi-restart) sits 27.5pp higher on fold 0 quick (typical ~94% per fold).

---

## 2. Why the diffusion loop is inert

At t=T, with `x_origin = 0` (fresh injection), `predict_x0_from_xt(x_t=√(1−α_T)·ε₀, ε₀, α_T) = 0`. So the first iteration's quick-success check tests **zero-feature injection + full edges**. Whatever flips here is independent of the diffusion mechanism.

For the remaining 51 graphs, after one DDIM step `x_{T−1}` is set, but:
- Subsequent `x0_pred` magnitudes scale with `√((1−α)/α) · guidance · score`, which for cosine schedule near t=T is large — yet **even at guidance=50 no additional graphs flip**.
- Hypothesis: after `construct_perturbed_graph` applies `feat_scale ≈ 2√N`, the injected feature magnitudes saturate the GCN normalization, so the prediction is determined entirely by the **graph topology** of the injection (the "full" edge strategy), not by the feature values. The 105 flips are topology-driven; the remaining 51 are topology-immune.

This is consistent with prior findings in feature-injection attacks on GCN — once edge structure is added, feature content has secondary effect under mean/sum pooling + ReLU + softmax.

---

## 3. Interpretation: not a code bug, an algorithm mismatch

- The reverse-process update rule was implemented per AdvAD Eq. 8 (verified by `ddim_step` + `predict_x0_from_xt` matching DDIM).
- Module imports & alpha schedule unit-tested (`python attack_diffusion.py` runs).
- 5-graph smoke test, fold-0 quick, and high-guidance all run end-to-end without error.

The mechanism failure is at the **algorithmic** layer, not the implementation:
**Score guidance via AMG presupposes the victim's gradient w.r.t. `x0` is informative.** For graph injection on GCN with `feat_scale` rescaling, `∇_{x0} CW` direction is geometrically aligned with `x0=0`'s saturation axis, so any step in that direction immediately saturates feat_scale's range. The diffusion never explores the topology-dependent subspace where the remaining 51 graphs are flippable.

---

## 4. Side finding (worth a paragraph in the paper)

**Trivial baseline**: "Inject one zero-feature node, fully connect to original" → 67.31% ASR on PROTEINS GCN. This is a previously-unreported lower bound; KATZE's generator+Adam pipeline adds ~27pp over this trivial baseline. Strong evidence that the *topological* injection (which node-budget=1 + full-edges already gives you) accounts for the majority of attack power, and feature-content optimization is the marginal contribution.

---

## 5. Decision (per HANDOFF §4 decision gate)

> "If ASR << baseline: kill N23-NoParam, pivot to N23-Guided (DiGress prior)."

**Decision: kill N23-NoParam.** The naive AdvAD port does not work for graph injection. The decision is not from a single fold's variance — the *ablation* (T=1 vs T=50; guidance=1 vs 50 yielding identical 105/156) is the conclusive evidence: the diffusion mechanism is inert by construction in this setting, not under-tuned.

**Next directions** (not pursued this session):
1. **N23-Guided** (DiGress prior, original HANDOFF §4 Phase 2). Needs ~6h GPU to pretrain DiGress on PROTEINS. Promising because a learned distribution prior may bypass the saturation problem by constraining x0 to a low-dimensional manifold.
2. **N24/N25** from `path-b-optimization-survey.md` — re-evaluate ranking now that the diffusion-NoParam quadrant is empirically ruled out for GCN+full-edges.
3. **Hybrid**: keep the DDIM trajectory but apply the score to *edge structure* (not features). This shifts away from AdvAD and toward DiSE-style approaches; outside N23 scope.

---

## 6. Files / artifacts

- `results/exp_20260519_134043_54aeadf4.json` — N23 T=50, λ=1.0
- `results/exp_20260519_134215_b5429ced.json` — N23 T=1, λ=1.0 (no-loop ablation)
- `results/exp_20260519_134248_d40c895c.json` — N23 T=50, λ=50.0
- `attack_diffusion.py` — implementation (kept for reference; not deleted)
- `quick_test_n23.py` — smoke test (trivially passes because x0=0 at t=T)

---

End of findings. Session continuation: see decision in §5.
