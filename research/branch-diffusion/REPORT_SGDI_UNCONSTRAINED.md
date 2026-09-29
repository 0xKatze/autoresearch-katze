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

**F3. Removing the constraints forfeits exactly the one thing SGDI was for.** Under
Mahalanobis, original `sgdi` is the best variant on ENZYMES (0.294; every unconstrained
variant lands at 0.147–0.252) and ties the best on PROTEINS (0.668). So the answer to the
question is: *yes*, unconstrained SGDI reaches benchmark-level undefended ASR — but at that
point it is `zo_free` with extra steps, and its published contribution (defense-invariance,
[[sgdi-mechanism-tradeoff]]) is gone. There is no setting in this grid where dropping the
constraints gives both.

**F4. `feat_scale=2√N` amplification is harmful here, not helpful.** PROTEINS undefended
0.674 amplified vs 0.854 unamplified (−18pp), and the amplified cells land exactly on the
67.3% zero-feature trivial baseline (`bench_zofree_20260929_192449.json` = 0.673), i.e. the
victim saturates to a fixed answer and the optimised direction stops mattering. Under
Mahalanobis, amplification is fatal (0.000–0.147). *Caveat:* this is the branch ZO engine;
the main project's KATZE generator engine reaches 1.00 on PROTEINS full-edge with the same
`auto_x2`, so this does **not** transfer to that claim without a matched run.

**F5. Unexplained: on PROTEINS the unconstrained variants keep a high Mahalanobis ASR**
(0.632–0.645) while `zo_free` collapses to 0.042, despite matching undefended ASR. The
likely cause is initialization — the SGDI path starts at μ and its restart samples from
N(μ,Σ) (in-distribution), whereas `zo_free`'s restart uses raw `randn` (off-distribution,
filtered out, so no gradient signal survives). **This is inferred from the code, not
verified.** A restart-init swap ablation would settle it, and it matters: if true, cheap
in-distribution *initialization* buys most of the defense-invariance that the expensive hard
projection was credited with.

## Caveats

- `zo_epochs=20`/`restarts=2` is the fast grid config, not `attack.py`'s 40/3 default.
- Query counts are not comparable across cells: early-stop-on-flip means a failing variant
  burns more queries (ENZYMES `sgdi` 1307 q/g vs `sgdi_noproj` 150 q/g).
- Only `mahalanobis` was tested. GNNSafe / GDDM-X / GDDM-A are untouched here; the
  GDDM-A weakness in `REPORT_v2_CONSOLIDATED.md` §B still stands.
- F5 is a hypothesis, not a result.
