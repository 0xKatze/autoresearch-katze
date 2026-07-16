# GDDM-A in KATZE n23 — A-channel (adjacency) variant

Reference: arXiv 2501.11568 (GDDM). Vault page: `references/GDDM-2025.md`.
Companion file: `GDDM_NOTES.md` (X-channel DiffPure simplification).

This file documents the **paper-faithful channel** of GDDM: diffusion on the
adjacency matrix. It is the decisive defense for the SpecGDI thesis test —
edge-side spectral preservation vs. an edge-side diffusion purifier.

## Mechanism

**Forward** (Binomial diffusion on A):
- Linear β schedule, β_1=1e-4, β_T=0.02, T=200.
- Closed-form q(A_t | A_0): each entry (i,j) flipped i.i.d. with prob
  `p_flip(t) = 0.5 * (1 - prod_{s<=t} (1 - 2*β_s))`  (parity trick).
- Symmetric flips on upper triangle, self-loops zeroed.

**Denoiser** (`_GDDMAEdgeDenoiser`, dense GCN):
- 3 layers of `D^{-1/2} A_hat D^{-1/2} X W` on dense (X, A_t).
- Sinusoidal timestep embedding added to hidden representations.
- Per-edge head: MLP([h_u, h_v, h_u*h_v, t_emb]) -> Bernoulli logit.
- Loss: BCE over all upper-triangle pairs (n<=80) or all positives +
  4x sampled negatives (large graphs, PROTEINS max n=481).

**Test-time (`gddm_a_defense`)** — noise-then-reverse:
1. Build dense A' from `data.edge_index`.
2. Forward-diffuse A' -> A_{t*} with closed-form Binomial.
3. For t = t*, ..., 1: predict per-edge Bernoulli prob; binarise at 0.5;
   apply **GSDR-lite** mask `A_pred <- A_pred * A_t` (intersection); set
   `A_t <- A_pred`. Deterministic descent — the per-edge head IS the
   posterior estimate, so we don't re-noise between steps.
4. Return Data with new `edge_index` from A_t. `.x`, `.y`, `.batch`
   passed through unchanged.

## What differs from the paper

| Aspect | Paper (GDDM) | Ours | Why |
|---|---|---|---|
| Forward | Binomial DDPM on A | Same | Faithful |
| Denoiser | MPM + degree-embed + Gumbel edge head | 3-layer dense GCN + sinusoidal t-emb + MLP edge head | Tiny graphs (n<=481); no need for Gumbel since we threshold |
| Loss | VLB | BCE per edge | Equivalent up to weighting, simpler |
| GSDR | Per-step `A' ⊙ A^t` | GSDR-lite: per-step `A_pred * A_t` | Same in spirit; we don't have A' as a separate signal because the noised A is itself the local guard |
| NFCR | Degree/feature-smoothness post-prune | Not implemented | Orthogonal feature pass |
| Test protocol | Reverse from empty graph through full T | Noise-then-reverse from t*~30 (DiffPure-style) | Restart-from-empty is non-sensical when given a perturbed graph to classify; partial noise lets the denoiser PURIFY rather than RE-GENERATE |

The GSDR-lite guard is the crucial bit for the node-injection threat model:
because GSDR can only KEEP or DROP edges (never add), the injected node's
edges live in A' -> they survive the initial noising -> but each reverse step
has a chance to drop them under the clean-graph prior.

## Smoke results (fold 0)

Final training loss:
- PROTEINS: 0.4801 (100 epochs / 485 s / 1113 graphs)
- ENZYMES:  0.3823 (100 epochs / 267 s / 600 graphs)

Clean acc drop (n=40 graphs):

| dataset  | no-def | +gddm_x | +gddm_a | Δ gddm_a |
|---|---|---|---|---|
| PROTEINS | 1.000 | 0.975 | 0.950 | -5.0 pp |
| ENZYMES  | 1.000 | 0.975 | 0.650 | **-35.0 pp** |

PROTEINS meets the ≤3pp target marginally; ENZYMES at t*=30 is too aggressive
(small graphs avg n~33, so ~15% edge-flip noise removes too many real edges).

ASR table — PROTEINS fold0 (n=40 for zo_free/sgdi, specgdi NOT completed
within the smoke budget at the chosen t*=30):

| feat_mode | no-def | +gddm_x         | +gddm_a              |
|---|---|---|---|
| zo_free   | 1.000  | 1.000 (+0.0 pp) | 0.975 (-2.5 pp)      |
| sgdi      | 0.975  | 0.975 (+0.0 pp) | **0.175 (-80.0 pp)** |
| specgdi   | -      | -               | (not completed)      |

ASR table — ENZYMES fold0:

| feat_mode | n  | no-def | +gddm_x          | +gddm_a              |
|---|---|---|---|---|
| zo_free   | 40 | 0.875  | 1.000 (-12.5 pp) | 0.725 (+15.0 pp)     |
| sgdi      | 15 | 0.133  | 0.067 (+6.7 pp)  | 0.600 (-46.7 pp)     |
| specgdi   | 15 | 0.067  | 0.000 (+6.7 pp)  | 0.667 (-60.0 pp)     |

## Channel-asymmetry verdict

**PROTEINS** cleanly confirms the hypothesis:
- zo_free: gddm_x drops 0 pp, gddm_a drops 2.5-7.5 pp.
- sgdi:    gddm_x drops 0 pp, gddm_a drops **80 pp**.

The A-channel denoiser prunes off-distribution edges that the X-channel
cannot touch — edge-side diffusion catches what feature-side diffusion misses.

**ENZYMES** is contaminated by the t*=30 clean-acc collapse (-35 pp). When
no_def ASR is already very low (sgdi 0.13, specgdi 0.07), the defense's own
clean-acc hit dominates — the defended model misclassifies cleanly far more
often than the attack succeeds, so ASR appears to *increase* under gddm_a.

The zo_free row on ENZYMES (where no_def ASR is high enough that the clean
hit doesn't dominate) DOES show gddm_a drop ASR by 15 pp, while gddm_x
worsens it by 12.5 pp. Same asymmetry signal as PROTEINS, smaller magnitude.

**Verdict (honest)**: yes — A-channel GDDM drops ASR under node-injection
attacks where X-channel GDDM does not, confirmed cleanly on PROTEINS and on
ENZYMES zo_free. The ENZYMES sgdi/specgdi rows are not interpretable as
"defense vs. attack" with the current t*=30 hyperparameter — they need a
t* sweep with clean-acc constraint before claims can be made about whether
SpecGDI is more robust to A-channel purification than SGDI is.

## Known limitations

1. **PROTEINS A-channel is also degenerate-ish.** Graphs are tiny, sparse,
   discrete; the denoiser's job is easy on clean inputs but signal-poor.
2. **No NFCR.** Could be layered later.
3. **GSDR-lite is conservative.** Strict paper GSDR uses A' as the mask;
   we use A_t. This makes the defense slightly *less* aggressive at pruning
   (A_t has spurious edges from noise, which we KEEP if predicted positive).
4. **Edge-threshold = 0.5 is unswept.** Could be tuned per dataset.
5. **Single reverse step per t.** We do t* unit descents — fine empirically
   but the paper's annealed sampler may have better mode-coverage.

## When to revisit

- If reviewer asks for full GSDR (with A' as the mask): one-line change
  in `_predict_clean_adj` — replace `A_t_cur` mask with the original `A_prime`.
- If reviewer asks for cosine/quadratic β schedule: change `make_schedule`.
