# SGDI constraint ablation — what happens when SGDI's stealth constraints are removed

Question: *「SGDI 改為沒有限制的版本如何?」* — does dropping SGDI's stealth machinery buy
benchmark-level ASR, and which of its two constraints (the hard Mahalanobis projection, or
the Gaussian prior pull) actually costs the ASR?

Run 2026-09-29. Script `eval_sgdi_unconstrained.py`; `attack.py` unmodified (the projection is
switched off by swapping `attack._project_maha` for the identity inside a context manager).

## Setup

- Victims: the branch-diffusion 5-fold GCN checkpoints — PROTEINS (discrete, 3-dim) and
  ENZYMES-continuous (21-dim z-scored, `models_cont`).
- Attack: `node_budget=1`, ASR@full-edge, `ATTACK_CFG = {skip_greedy: True, zo_epochs: 20,
  zo_restarts: 2}` (identical to `eval_paper_grade.py`, so cells are comparable to the
  published grid).
- 2 seeds × 5 folds = **n=10 observations per cell**. Defenses: `none`, `mahalanobis`
  (calibrated by `eval_paper_grade.build_defended`, 99.5th-pct clean cut).
- Variants: `sgdi` (original) · `sgdi_noproj` (projection off, prior kept η=0.3) ·
  `sgdi_noproj_noprior` (projection off, η=0) · `sgdi_free_x2` (+ `feat_scale=auto_x2`) ·
  `zo_free` / `zo_free_x2` (plain ZO references, no diffusion terms).

Sources: `results/cdma/sgdi_unconstrained_both_20260929_194010.json` (PROTEINS),
`results/cdma/sgdi_unconstrained_ENZYMES_20260929_200048.json` (ENZYMES).

## Results (ASR@full-edge, mean ± std over n=10)

### PROTEINS

| variant | none | mahalanobis | q/graph (none) |
|---|---|---|---|
| sgdi (original) | 0.670 ± 0.045 | **0.668 ± 0.046** | 106 |
| sgdi −proj (prior kept) | 0.687 ± 0.034 | 0.632 ± 0.031 | 101 |
| **sgdi −proj −prior** | **0.846 ± 0.069** | 0.645 ± 0.025 | 70 |
| sgdi −proj −prior + 2√N | 0.673 ± 0.027 | 0.004 ± 0.006 | 103 |
| zo_free | 0.854 ± 0.068 | 0.042 ± 0.069 | 67 |
| zo_free + 2√N | 0.674 ± 0.028 | 0.000 ± 0.000 | 103 |

### ENZYMES-continuous

| variant | none | mahalanobis | q/graph (none) |
|---|---|---|---|
| sgdi (original) | 0.275 ± 0.068 | **0.294 ± 0.066** | 1307 |
| **sgdi −proj (prior kept)** | **0.994 ± 0.010** | 0.233 ± 0.050 | 150 |
| sgdi −proj −prior | 0.937 ± 0.061 | 0.252 ± 0.057 | 352 |
| sgdi −proj −prior + 2√N | 1.000 ± 0.000 | 0.147 ± 0.041 | 88 |
| zo_free | 0.942 ± 0.055 | 0.197 ± 0.050 | 315 |
| zo_free + 2√N | 1.000 ± 0.000 | 0.020 ± 0.019 | 50 |

## Findings

**F1. Unconstrained SGDI reaches plain-ZO level undefended, and no further.** PROTEINS
0.846 vs `zo_free` 0.854; ENZYMES 0.937–0.994 vs `zo_free` 0.942. The diffusion machinery
(Langevin noise, prior score) contributes **no attack power of its own** once the stealth
constraints are gone — the same conclusion N23 reached, now measured on the SGDI code path.

**F2. Which constraint costs the ASR is dataset-dependent, and the two datasets disagree.**

| | PROTEINS (topology-driven) | ENZYMES (feature-driven) |
|---|---|---|
| drop projection only | +1.7pp (0.670→0.687) | **+71.9pp (0.275→0.994)** |
| also drop prior pull | **+17.6pp** (→0.846) | −5.7pp (→0.937) |

On PROTEINS the bottleneck is the **prior pull** `η·Σ⁻¹(x−μ)`; the hard ball is nearly free,
because full-edge topology carries the attack and the injected feature barely needs to move.
On ENZYMES the bottleneck is the **projection**: the attack must carry its signal in the
21-dim features, so confining them to the clean Mahalanobis ball is what caps ASR at 0.275.
Tuning η is therefore the lever on PROTEINS; the ball radius is the lever on ENZYMES.

**F3. Whether removing the constraints forfeits SGDI's purpose also splits by dataset.**

- **ENZYMES: yes, the floor is lost.** Original `sgdi` is the best variant under Mahalanobis
  (0.294); every unconstrained variant drops to 0.147–0.252. Here the constraints are what buy
  the defended floor, and dropping them trades it away for undefended ASR.
- **PROTEINS: no, the floor survives** — `sgdi −proj −prior` is 0.846 / 0.645 vs original
  0.670 / 0.668, i.e. **+17.6pp undefended for −2.3pp defended** (inside one std at n=10).
  That is close to a free win in this grid. What is lost is not the floor but the *invariance
  property* (none ≈ maha), which no longer holds. The mechanism is unexplained — see **F5**,
  which is the exception to the ENZYMES story, not a footnote to it.

So the direct answer: unconstrained SGDI reaches plain-ZO undefended ASR on both datasets,
but it then has no mechanism of its own (F1) — on ENZYMES it also gives up the defended floor,
on PROTEINS it keeps it for reasons we have not identified. Neither case leaves SGDI's
published framing ([[sgdi-mechanism-tradeoff]]) intact as stated.

**F4. `feat_scale=2√N` amplification helps or hurts along the same dataset split as F2.**

| | PROTEINS (topology-driven) | ENZYMES (feature-driven) |
|---|---|---|
| undefended, amplified vs not | **0.674 vs 0.854 (−18pp)** | **1.000 vs 0.942 (+5.8pp)** |
| under mahalanobis | 0.000–0.004 | 0.020–0.147 |

On PROTEINS the amplified cells land exactly on the 67.3% zero-feature trivial baseline
(`bench_zofree_20260929_192449.json` = 0.673): the victim saturates to a fixed answer and the
optimised direction stops mattering, so amplification actively destroys information. On
ENZYMES, where the attack is feature-carried, amplification instead saturates the attack to
1.000. Under Mahalanobis it is fatal on both. This is the same topology-vs-feature split as F2
and F3, which is the through-line of this ablation. *Caveat:* this is the branch ZO engine;
the main project's KATZE generator engine reaches 1.00 on PROTEINS full-edge with the same
`auto_x2`, so the PROTEINS row does **not** transfer to that claim without a matched run.

**F5. SUPERSEDED — see the Correction below.** The original F5 claimed in-distribution
initialization carried the PROTEINS defended floor. That was wrong: the two code paths start at
the *same* point at r=0, so init cannot be the difference. See **Correction** for the verified
mechanism.

## Caveats

- `zo_epochs=20`/`restarts=2` is the fast grid config, not `attack.py`'s 40/3 default.
- Query counts are not comparable across cells: early-stop-on-flip means a failing variant
  burns more queries (ENZYMES `sgdi` 1307 q/g vs `sgdi_noproj` 150 q/g).
- Only `mahalanobis` was tested. GNNSafe / GDDM-X / GDDM-A are untouched here; the
  GDDM-A weakness in `REPORT_v2_CONSOLIDATED.md` §B still stands.
- F5's mechanism is now settled by a dedicated cell (noise excluded, init implicated); the
  *generality* of the init effect (other datasets, other defenses) is not.

---

# Correction (2026-09-29, later the same day)

Two claims above were wrong. Both are corrected here; the tables in §Results are unaffected.

## The mechanism is an early-stop artifact, not initialization

`sgdi_optimize_features` calls `_cge_grad` at the **top** of its loop and breaks on `cur < 0`
*before* taking any step — so at t=0 it evaluates, and can return, the **unoptimised point μ**.
`_zo_one` (the `zo_free` path) evaluates the margin only **after** `feats = feats - lr*grad`, so
it can never return μ; its first act is always an unbounded gradient step that, under a
Mahalanobis filter, throws the injected node out of the ball where the gradient is ~0.

Discriminator (`results/cdma/f5_retest_manifold_restart1.json`, n=10, `zo_restarts=1` so both
paths have *identical* init and no restart; sgdi run with η=0, temp=0, projection off, making
its update rule literally identical to `zo_free`'s):

| dataset | defense | manifold (μ, **zero** optimisation) | sgdi bare, r=1 | zo_free, r=1 |
|---|---|---|---|---|
| PROTEINS | none | 0.599 ± 0.041 | 0.840 ± 0.072 | **0.840 ± 0.072** |
| PROTEINS | mahalanobis | **0.599 ± 0.041** | **0.601 ± 0.038** | 0.040 ± 0.069 |
| ENZYMES | none | 0.116 ± 0.040 | 0.923 ± 0.063 | **0.923 ± 0.063** |
| ENZYMES | mahalanobis | 0.144 ± 0.036 | 0.238 ± 0.058 | 0.197 ± 0.050 |

- Undefended, the two paths are **bit-identical** (0.840 / 0.840 and 0.923 / 0.923), confirming
  the stripped update rules are the same and that nothing about SGDI's machinery differs there.
- Under Mahalanobis on PROTEINS they still diverge 0.601 vs 0.040 **with identical init**, so
  **initialization is excluded** (as annealed noise already was). The remaining difference is
  the flip-check placement.
- And `sgdi bare` 0.601 ≈ `manifold` 0.599. **The unconstrained variant's entire defended ASR on
  PROTEINS is the trivial μ-injection baseline**, surfaced by the pre-step check. It is not an
  attack result at all.
- On ENZYMES the artifact is absent: both paths beat manifold (0.238 / 0.197 vs 0.144) and agree
  within one std. Consistent with F2 — PROTEINS is topology-driven, so injecting the mean node
  at full edge already flips 60% of graphs; ENZYMES is feature-driven, so it does not.

## F3's "PROTEINS near-free win" is withdrawn

F3 read `sgdi −proj −prior` = 0.846 / 0.645 vs original 0.670 / 0.668 as +17.6pp undefended for
only −2.3pp defended. But 0.645 is the trivial baseline (0.599) plus restart/noise slack, not a
preserved attack floor. **Dropping the constraints does forfeit the defended attack capability on
PROTEINS too** — it just gets masked by a baseline the metric hands out for free. The ENZYMES
half of F3 (floor genuinely lost, 0.294 → 0.147–0.252) stands unchanged.

## Consequence for the published SGDI headline (new, needs the user's attention)

The `manifold` row — inject the per-dim mean node, optimise nothing — is **present in the
published grid data** (`results/eval_paper_grade_20260525_161553.json`) but **absent from the
§A.1 table of `REPORT_v2_CONSOLIDATED.md`**, which lists only `zo_normcap` / `sgdi` / `specgdi`.
Under Mahalanobis it reads:

| dataset | manifold | sgdi | Δ(sgdi − manifold) | Δ(sgdi − zo_normcap) = the published headline |
|---|---|---|---|---|
| PROTEINS | 0.599 | 0.658 | **+5.9pp** | +61.6pp |
| ENZYMES | 0.144 | 0.294 | **+15.0pp** | +9.5pp |

The +61.6pp / +9.5pp claims against N4 norm-cap are arithmetically fine. But the question
"does the guided-diffusion machinery earn its keep against a distribution filter?" is answered by
the manifold column, and there the PROTEINS margin is **+5.9pp, not +61.6pp** — because a
Mahalanobis filter punishes *any* off-distribution feature, so the honest reference is the
cheapest in-distribution injection, not the off-distribution baseline. Recommendation: add the
`manifold` row to the §A.1 table and state Δ against it alongside Δ against N4. This does not
require new experiments — the numbers are already in the published JSON.

## What is settled and what is not

- Settled: F1 (unconstrained SGDI = plain ZO undefended), F2 (which constraint binds flips by
  dataset), F4 (amplification splits the same way), and the mechanism above.
- Withdrawn: F3's PROTEINS half, and F5 in its entirety.
- Not tested: the η curve between 0 and 0.3 (no sweep exists in `results/`); the ENZYMES radius
  sweep; anything under GNNSafe / GDDM-X / GDDM-A.
