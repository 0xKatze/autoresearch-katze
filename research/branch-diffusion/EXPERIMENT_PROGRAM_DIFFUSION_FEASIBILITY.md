# Experiment Program — Graph Diffusion Node-Injection Attack Feasibility

> **Status**: DRAFT for user sign-off (handoff §15 requires sign-off before implementation).
> **Author**: session 2026-06-01 (successor to handoff-graph-diffusion-attack-feasibility-2026-05-26).
> **Charge**: Decide whether graph diffusion is *architecturally capable* of producing node-injection attacks that beat non-diffusion baselines under adjacency-aware defenses — or rigorously falsify it. Both outcomes publishable.

---

## 0. TL;DR + the reframe that drives everything

The prior sessions threw **feature-axis** and **spectral-axis** stealth at GDDM-A and got a structural ASR floor ≈ 0.033–0.067 on PROTEINS, robust across optimization signal (ZO vs white-box PGD), trojan content origin (synthetic vs real-motif), and scale (K=1→10). That is already a strong negative.

**But all prior attacks missed a mechanical fact about the wall they were hitting.** Reading `defenses.py::gddm_a_defense`:

1. **GDDM-A purifies *only adjacency*.** It returns `Data(x=data.x, ...)` — **features pass through untouched** (defenses.py:605). SGDI's Mahalanobis feature-whitening and SpecGDI's spectral preservation are *irrelevant* to GDDM-A; it never looks at features.
2. **The reverse step is monotone edge-removal.** `A_pred = A_pred * A_t_cur` — "GSDR-lite: intersect with current noised graph — **can drop, never add**" (defenses.py:594). GDDM-A keeps edge (i,j) iff the per-edge denoiser scores it `p_hat > 0.5` (looks like a clean-distribution edge) at every descent step.

**Therefore feasibility-under-GDDM-A is purely an EDGE-SURVIVAL game:** if the injected node's attachment edges are pruned, the node is disconnected → zero message-passing influence → injection is inert *regardless of features or spectrum*. If even **one** adversarially-useful attachment edge survives, then **features are free** (high-dim, GDDM-A-blind) and SGDI-style feature power flows through it.

The prior 11 attack variants all failed for the **same reason**: their attachment edges (`A_VU`) were chosen adversarially → off the clean-edge manifold → pruned. None of them optimized for *edge survival*. **That is the untested axis**, and it is the only one that matters against GDDM-A.

This reframe makes the feasibility question **cheap and decisive to settle** — see Diagnostic D0 (§3), which no prior session ran.

---

## 1. Falsifiable hypotheses

**H₀ (null, the structural-robustness thesis):**
> Graph diffusion cannot produce node-injection attacks that beat non-diffusion baselines under adjacency-aware defenses (GDDM-A). Specifically: for any single/few-node injection, the set of attachment edges that (a) survive GDDM-A reverse diffusion AND (b) carry adversarial signal is measure-near-zero, because adversarial attachment requires a cross-distribution edge and the denoiser prunes exactly those.

**H₁ (feasibility, the attack thesis):**
> A diffusion process that *conditions on the defense's own edge-distribution prior* can generate attachment edges that survive GDDM-A while routing free (GDDM-A-blind) feature power into the victim, beating the non-diffusion baseline (N4 norm-cap) under GDDM-A by a pre-committed margin.

**Primary metric:** defended ASR (real GDDM-A forward pipeline, not soft surrogate), paired against the same graphs' N4-baseline defended ASR.

**Falsification is a publishable outcome** ("GDDM-A reduces injection to an edge-survival game adversarial edges cannot win" + the D0 measurement + a near-theorem). So is feasibility ("first graph diffusion attack to break GDDM-A"). We do **not** pre-judge.

---

## 2. Mechanistic reframe — the edge-survival game (formal sketch)

Let the victim graph be `G=(X,A)`, injected node `v*` with features `x*` (free) and attachment vector `a ∈ {0,1}^n` (which existing nodes `v*` connects to). GDDM-A applies reverse map `R: A ↦ A'` that keeps edge `(v*,u)` iff `D_θ(emb(v*), emb(u)) > 0.5` where `D_θ` is the clean-trained per-edge denoiser head and `emb(·)` are GCN embeddings on the noised graph.

- **Survival set** `S(x*) = { u : D_θ(emb(v*;x*,a), emb(u)) > 0.5 }` — edges the denoiser keeps. Note it depends on `x*` (through `v*`'s embedding).
- **Adversarial set** `Adv = { u : connecting v* to u moves the victim's prediction toward target class }`.
- Injection succeeds under GDDM-A iff `S(x*) ∩ Adv ≠ ∅` for some achievable `x*`, **after** post-purification feature optimization on the surviving subgraph.

**The tension (H₀'s engine):** `Adv` favors *cross-class / boundary-crossing* edges (high message-passing leverage). `S` favors *clean-distribution* edges (in homophilous classification graphs ≈ same-community edges). For homophilous datasets these sets tend to be disjoint. **H₁'s only crack:** make `v*`'s features mimic a *natural neighbor* of `u` (so the edge looks clean → survives) while the same `x*`, at the mean-pool readout, shifts the class statistic across the boundary. Whether a feature vector can be simultaneously "locally clean-neighbor-like" and "globally class-shifting" is the empirical crux — D0 measures it.

This is exactly the graph analogue of "in-distribution adversarial examples": SGDI proved they exist on the **feature/Mahalanobis** axis (large DoF). The open question is whether they exist on the **edge-survival** axis (low DoF for a singleton node).

---

## 3. Diagnostic D0 — keep-set characterization (RUN FIRST, ~2–3 hr, settles feasibility cheaply)

No prior session measured the keep-set. This is the decisive, architecture-free experiment. **Run before building any attack.**

**D0.1 — Survival census.** On PROTEINS + COX2 (the two clean-usable GDDM-A datasets), for a 30-graph subset per fold: inject `v*`, attach via each strategy below, run real GDDM-A, record **per-edge survival rate** and **P(≥1 attachment edge survives)**:
- random attachment, degree-topk attachment, low-degree attachment, spectral attachment (existing strategies)
- real-motif neighbor mimicry (v3-B style): `x*` ← mean of `u`'s true neighbors' features
- **survival-greedy**: pick `u` maximizing denoiser keep-prob `D_θ` (white-box at defense, grey-box threat model — disclosed)

**D0.2 — Feature-routing ceiling.** For graphs where ≥1 edge survives D0.1: *freeze the surviving subgraph*, then run SGDI-style free feature optimization on `x*` (GDDM-A-blind), measure post-purification ASR. This is the **feasibility ceiling**: the best any feature-only attack can do once an edge survives.

**Decision gate (commit now):**
- If **P(≥1 edge survives) ≈ 0** across all strategies including survival-greedy → **H₀ confirmed structurally**; the floor is an edge-survival impossibility. Stop architecture search; write the falsification paper + near-theorem. (Estimated likely outcome — see §7 prior.)
- If **P(≥1 edge survives) > 0** but D0.2 ceiling ASR ≤ floor → edges survive but can't be routed adversarially → softer H₀; document why.
- If **D0.2 ceiling ASR ≥ 0.3** → feasibility is real and the bottleneck is *generating* survival-greedy edges with a black-box, learnable process → proceed to architecture candidates §4 (esp. Arch B).

---

## 4. Candidate attack architectures (pre-committed, reframed around edge survival)

Built only if D0 clears the gate. Priority by mechanistic alignment with the edge-survival game.

### Arch B — Survival-Maximizing Discrete Attachment Diffusion (DAD-S) — **primary contender**
Train a DiGress-style **discrete** (Bernoulli) diffusion model on clean adjacency (reuse `train_gddm_a.py` edge-head architecture — it *is* a clean-edge denoiser). At attack time, run reverse diffusion to **generate attachment edges `A_VU` that maximize keep-probability under the clean prior** (i.e., generate edges the defense will keep), with per-step ZO classifier guidance steering toward `Adv`. Then free-optimize `x*` on the surviving subgraph.
- Tests handoff Q2 (discrete) + the reframe (survival objective). **The one design that attacks the floor's actual mechanism.**
- Threat: black-box at victim; uses a *trained-on-public-graphs* prior (field-standard).
- Time: ~6–8 hr (mostly reuses existing edge-head + checkpoints).

### Arch A — TrainedPrior-SGDI (DiGress/GDSS feature score) — cheap, tests Q1 on feature-axis defenses
Replace SGDI's analytic Gaussian score `Σ⁻¹(x−μ)` with a *trained* graph-diffusion feature score `−∇ₓ log p_θ(x|G)` (mini-GDSS / 3-layer GCN denoiser, reuse `train_gddm.py` pipeline). Run vs **Mahalanobis + GNNSafe** (the feature/embedding-axis defenses), NOT primarily GDDM-A (which ignores features).
- Tests handoff Q1. Realistic chance of *extending the SGDI headline* on the feature axis, even if irrelevant to GDDM-A.
- Time: ~3 hr.

### Arch C — AdvAD-faithful black-box port (per-step ZO guidance) — only if D0.2 says routing works
SGDI applies the adversarial gradient *once at the end*. AdvAD applies it *every reverse step*. Port per-step guidance via per-step ZO/CGE probes (cheap: 3–37 feature dims). Combine with Arch B's surviving subgraph so the per-step guidance has a live edge to act through.
- Tests handoff Q3. Recipe to be finalized from the image-attack lit agent's findings.
- Time: ~4 hr.

**Deferred** (out of this session's budget): Latent Graph Diffusion (VGAE + latent diffusion), Score Distillation Attack. Note in deliverable as future work.

---

## 5. Defense ladder (pre-committed)

Every candidate runs against the full ladder, not just GDDM-A, so we can separate "wins on feature axis" from "wins on adjacency axis":

| Defense | Axis | Role |
|---|---|---|
| `none` | — | undefended ASR (power, not stealth) |
| `normfilter` (N4 equiv.) | feature magnitude | baseline filter |
| `mahalanobis` | feature distribution | SGDI's home turf (Arch A target) |
| `gnnsafe` | embedding energy | OOD embedding |
| `gddm_a` | **adjacency (edge survival)** | **the wall; Arch B target** |
| `gddm_x` | feature DiffPure | known to backfire — sanity only |
| **`diffsp`** (WWW'25, arXiv 2502.05000) | **adjacency (prior-free purification)** | **2nd, published, prior-free adjacency wall — hardens the verdict** |

`jaccard`, `svd` available but secondary.

**DiffSP (added 2026-06-01, lit-agent finding):** the architecture the user named is a *defense*, not an attack — "Prior-Free Diffusion-Based Structure Purification" (Luo et al., WWW 2025, [repo](https://github.com/RingBDStack/DiffSP)). LID-driven non-isotropic forward noise + graph-transfer-entropy-guided denoising. Including it makes the negative/positive verdict **far more robust**: it is a *second, independent, published* adjacency-purification defense, and being **prior-free** it directly answers the standing caveat that our GDDM-A is just a BCE-per-edge approximation. **Implementation decision deferred to post-D0** (official repo exists; ~4–6 hr port). If D0 confirms structural falsification on GDDM-A, porting DiffSP to confirm the floor on a second defense is the highest-value hardening step.

---

## 6. Datasets, statistical rigor, controls

**Datasets (GDDM-A clean-usable only, t*=30, verified 2026-06-01):**
- **PROTEINS** — GDDM-A clean-acc drop −3.0pp ✅ (primary)
- **COX2** — drop −1.7pp ✅ (**new second dataset — doubles the prior single-dataset floor evidence**)
- ENZYMES (−18.7pp), NCI1 (−18.5pp), DD (−27.2pp) — contaminated at t*=30; **optional t*-band sweep** to recover one as a third dataset (secondary, time-permitting).

**Statistical rigor (carry from MEMORY.md + handoff §7):**
- n ≥ 15 paired observations per cell (3 seeds × 5 folds); aim for n=30 if 10-fold splits get trained.
- Two-sided paired Wilcoxon, target p < 0.01 (floor at n=15 is 6.10e-5).
- 95% bootstrap CI excluding zero for headline claims.
- Per-task wall-clock cap `timeout 1200` or `n ≤ 60` subset for diffusion-family attacks (handoff gotcha #1).
- Report victim model + victim clean accuracy + full budget (node/edge/perturbation/query) every cell (MEMORY: budget reporting mandatory).

**Controls held fixed:** victim = GCN (PROTEINS/COX2 trained, save_best), seed set {0,1,2}, `--force_continuous`, edge budget matched across attack/baseline, GDDM-A t*=30.

**Reference bar to beat:** SGDI vs N4 on PROTEINS×mahalanobis = +61.3pp, p=6.10e-5, W=0. Any new headline must clear its own pre-committed bar (§7).

---

## 7. Decision rules (COMMIT BEFORE SEEING RESULTS)

Per-candidate, declared in advance:

- **Headline feasibility (breaks the wall):** defended ASR ≥ 0.5 under GDDM-A on PROTEINS, AND beats N4 under GDDM-A by ≥ 5pp, paired n ≥ 15, p < 0.01, **replicated on COX2** (same direction, p < 0.05). → "First graph diffusion attack to break GDDM-A."
- **Soft positive:** ASR ≥ 0.3 under GDDM-A on ≥1 dataset, OR a clean win on a feature-axis defense (Arch A extends SGDI by ≥ 5pp under Mahalanobis). → sub-result, reframe as "first sub-defense leak / SGDI extension."
- **Null:** ASR < 0.2 under GDDM-A AND no feature-axis improvement. → honest negative; feeds the "GDDM-A is structurally robust" paper.
- **Structural falsification (strongest negative):** D0.1 shows P(≥1 surviving adversarial edge) ≈ 0 on both PROTEINS + COX2. → near-theorem paper; SGDI×Mahalanobis stays the locked headline.

**Author's pre-registered prior (ultrathink, before any run):** I expect **structural falsification or null** is more likely than headline feasibility, because the edge-survival/adversarial sets are near-disjoint for homophilous graph classification (§2). The *value* of this program is settling it **decisively and cheaply via D0**, and — if null — converting the prior sessions' scattered negatives into one clean mechanistic theorem on two datasets. Stated so the result can prove me wrong.

---

## 8. Deviations / adaptations to disclose (per project policy, BEFORE running)

- **Grey-box at defense** for D0.1 survival-greedy + Arch B keep-prob objective: attacker reads GDDM-A weights. Disclosed; consistent with prior v3-A grey-box. Black-box-at-victim preserved.
- **Trained prior on public graphs** (Arch A/B): attacker trains a diffusion prior on a public clean-graph corpus, not the victim's private train set. Field-standard; disclosed.
- **GDDM-A is our BCE-per-edge approximation** of the paper's MPM+Gumbel denoiser (inherited from prior session; documented in REPORT_v2 §E.5). Verdict robustness to this is itself a stated caveat.
- **Datasets restricted to PROTEINS + COX2** by GDDM-A clean-acc usability; not free dataset choice. Disclosed.
- Any further simplification logged here *before* the run that uses it (no retrospective disclosure — prior session's mistake).

---

## 9. Compute budget + schedule (24–48 hr scope)

| Phase | Task | Est. | Gate |
|---|---|---|---|
| Lit | 3 parallel reviewers → vault refs + cheat sheet (deliverable 1) | running | — |
| **D0** | Keep-set census + feature-routing ceiling on PROTEINS+COX2 | 2–3 hr | **decides whether §4 runs at all** |
| Arch A | TrainedPrior-SGDI smoke vs maha/gnnsafe | 3 hr | only adds SGDI extension |
| Arch B | DAD-S survival-max discrete diffusion smoke | 6–8 hr | only if D0.2 ceiling ≥ 0.3 |
| Arch C | AdvAD per-step ZO port | 4 hr | only if routing works |
| Synth | Verdict + vault writeback + handoff | 2 hr | — |

**Critical path is D0.** If D0 confirms structural falsification, Arch A/B/C collapse to "confirm on a non-diffusion sanity attack" and the session ends early with a clean theorem — the *right* falsificationist outcome.

---

## 10. Literature map (deliverable 1 — DONE 2026-06-01)

Three vault reference pages written: `references/graph-diffusion-generation-survey.md`, `references/graph-diffusion-attack-intersection.md`, `references/image-diffusion-attacks-port-recipe.md`.

### 10.1 Named-architecture resolutions
- **DiffSP** = a **DEFENSE** (Prior-Free Diffusion-Based Structure Purification, WWW'25, arXiv 2502.05000, RingBDStack/DiffSP). Added to defense ladder §5.
- **DPNI** = **NOT FOUND** (no fabrication). No match on GitHub/arXiv/Semantic Scholar/web; absent from vault + paper drafts. Likely a **user-coined working name**. Closest real "poisoning node injection" methods: NIPA (WWW'20), NICKI (arXiv 2301.12277), LPGIA (TCSS'24). **→ ASK USER what DPNI refers to.**

### 10.2 Novelty gap — graph diffusion ATTACK
**Effectively none exists** (June 2026). Near-misses: DM4Steal (diffusion+graph but link-*stealing privacy* attack); GAFNC (generative node-injection but *GAN*, poisoning); DiffAttack/SFIA (naming traps). **Defensible claim (qualified):** *"first black-box node-injection evasion attack on graph classifiers that uses a diffusion/score-based generative process to craft injected-node features."* Do NOT claim "first diffusion graph attack" or "first generative node-injection attack" unqualified.

### 10.3 Generation-architecture cheat sheet (Arch B candidate selection)
| Arch | Discrete/Cont | Conditional? | Repo | Attack-portability |
|---|---|---|---|---|
| **DiGress** (2209.14734) | **Discrete** categorical | **Yes** (graph-level guidance) | cvignac/DiGress | **Best.** Discrete edges (no relaxation gap) + native conditioning; denoiser inherits the clean-edge prior GDDM-A enforces. **→ Arch B base.** |
| GDSS (2202.02514) | Continuous joint X+A | No (unconditional) | harryjo97/GDSS | 2nd. Models X–A coupling; must bolt on conditioning; continuous→binary seam = where GDDM-A catches it. |
| EDM (2203.17003) | 3D coords + types | Yes | ehoogeboom/... | Not portable (no adjacency channel). |
| GeoDiff (2203.02923) | 3D coords | wrong direction | MinkaiXu/GeoDiff | Not portable. |
| ConGress | Continuous X+A | Yes | in DiGress repo | Dominated by DiGress. |
| GraphARM (2307.08849) | Discrete autoregressive | constraint-only | community only | Best *interface* (node+edges = injection op) but NOT a denoiser → can't exploit GDDM-A's reverse prior. Contrast only. |

→ **Arch B uses a DiGress-style discrete edge denoiser** (we already have one: `train_gddm_a.py`'s edge-head IS a clean-edge Bernoulli denoiser — repurpose it attack-side).

### 10.4 Image-attack recipe (Arch A/C)
Only **AdvAD** (NeurIPS'24, 2503.09124) is **non-parametric** → SGDI's ancestor. SGDI dropped AdvAD's **per-step hard projection** (uses only end-projection). The **"AdvAD-ZO-G" port** (per-step ZO guidance + per-step Mahalanobis projection, 2·d·T queries, cheap for d≤7) is worth a small prototype **but measured by Mahalanobis-radius @ fixed ASR, NOT raw ASR** — per-step *feature* guidance cannot break the adjacency ceiling (confirms the reframe §2). See `references/image-diffusion-attacks-port-recipe.md`.

---

## 11. First actions on sign-off
1. Implement D0 census harness (`smoke_keepset_d0.py`) — reuses `gddm_a_defense`, `construct_perturbed_graph`, existing checkpoints.
2. Run D0 on PROTEINS + COX2; evaluate decision gate (§3).
3. Branch per gate: falsify-and-write, or build Arch B.
