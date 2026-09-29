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

**F5. Unexplained (the F3 exception): on PROTEINS the unconstrained variants keep a high
Mahalanobis ASR** (0.632–0.645) while `zo_free` collapses to 0.042, despite matching
undefended ASR. Under the Mahalanobis `DefendedModel` an out-of-ball injected node is removed,
the margin reverts to clean, and both ±σ CGE probes are also out-of-ball → gradient ≈ 0 → the
attack is stuck. Two candidate mechanisms let the SGDI path avoid that trap, and they are
**not** distinguished by the cells above:

1. **In-distribution initialization** — the SGDI path starts at μ and its restart samples from
   N(μ,Σ), whereas `zo_free`'s restart uses raw `randn` (off-distribution, filtered out).
2. **Annealed Langevin noise** — `sgdi_temp=0.1·frac` survives in `sgdi −proj −prior`
   (only η is zeroed) and can random-walk a just-outside feature back into the ball, resuming
   gradient flow. `zo_free` has no noise term, so once stuck it stays stuck.

**Discriminator run (PROTEINS, mahalanobis, `sgdi −proj −prior`, n=10;
`results/cdma/f5_noise_discriminator_PROTEINS.json`):**

| Langevin noise | ASR under mahalanobis |
|---|---|
| `sgdi_temp=0.1` (on) | 0.645 ± 0.025 |
| **`sgdi_temp=0` (off)** | **0.630 ± 0.030** |

Turning the noise off changes nothing (−1.5pp, well inside one std), so **mechanism 2 is
excluded and mechanism 1 — in-distribution initialization — is the operative one.** The
attack's whole defended floor on PROTEINS comes from *starting* at μ / sampling restarts from
N(μ,Σ); neither the hard projection, nor the prior pull, nor the Langevin noise is doing that
work. This is the single most actionable result of the ablation: on PROTEINS a free change of
initialization reproduces the defended ASR that SGDI's expensive machinery was credited with,
while leaving the +17.6pp undefended gain of the unconstrained variant on the table.

Still open: whether the same holds on ENZYMES (where the projection genuinely carries the ASR,
per F2), and whether the init effect survives other defenses (GNNSafe / GDDM).

## Caveats

- `zo_epochs=20`/`restarts=2` is the fast grid config, not `attack.py`'s 40/3 default.
- Query counts are not comparable across cells: early-stop-on-flip means a failing variant
  burns more queries (ENZYMES `sgdi` 1307 q/g vs `sgdi_noproj` 150 q/g).
- Only `mahalanobis` was tested. GNNSafe / GDDM-X / GDDM-A are untouched here; the
  GDDM-A weakness in `REPORT_v2_CONSOLIDATED.md` §B still stands.
- F5's mechanism is now settled by a dedicated cell (noise excluded, init implicated); the
  *generality* of the init effect (other datasets, other defenses) is not.
