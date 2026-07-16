# GDDM in KATZE n23 — simplification ledger

Reference: arXiv 2501.11568 (Jan 2025). Vault page: `references/GDDM-2025.md`.

We implement the **GDDM paradigm** (clean-only diffusion prior + purification at test time) but **not the bit-exact paper**. This file lists every deliberate divergence so future work knows what to revisit if a full reproduction is needed.

## What we kept (paradigm-faithful)

- Clean-only training. No adversarial examples in the denoiser's training set.
- Per-dataset denoiser. Separate checkpoints for PROTEINS and ENZYMES.
- Test-time purification reads only the (possibly perturbed) graph — no per-graph fine-tuning.
- Conditioning on graph structure via message passing.

## What we changed (and why)

| Aspect | Paper (GDDM) | Ours | Rationale |
|---|---|---|---|
| What is diffused | Adjacency $\mathbf{A}$ only; $\mathbf{X}$ held fixed | Node features $\mathbf{X}$ only; $\mathbf{A}$ held fixed | KATZE n23 is a *node-injection* attack. GSDR's $\mathbf{A}'\odot\mathbf{A}^t$ mask can only remove edges, never the injected node itself. An X-purifier is the meaningful baseline for our threat model. |
| Forward process | Discrete Binomial on $\mathbf{A}$ | Gaussian DDPM on $\mathbf{X}$ (linear $\beta_t$, $T=200$, $\beta_1=10^{-4}$, $\beta_T=0.02$) | Standard, well-conditioned, well-understood on continuous features. |
| Training objective | Variational lower bound (VLB) | L2 noise prediction (DDPM-style) | Equivalent up to weighting; far simpler. |
| Denoiser arch | MPM + degree-embedding + Gumbel edge head | 3-layer GCN (hidden 64) + sinusoidal $t$-embedding, predicts $\epsilon$ | Tiny graphs ($\le 50$ nodes); no need for the paper's edge-prediction head since we don't denoise edges. |
| GSDR (structure refiner) | Per-step $\mathbf{A}^t \leftarrow \mathbf{A}'\odot\mathbf{A}^t$ | Not implemented | We don't diffuse $\mathbf{A}$; nothing to mask. |
| NFCR (feature-smoothness pruning) | Post-hoc degree+smoothness edge prune | Not implemented | Orthogonal to feature purification. |
| Test-time reverse | Reverse from empty/target-removed graph through full $T$, applying GSDR each step | DiffPure-style: noise $\mathbf{X}'$ to $t^\star \approx 30$ ($\sim$15% noise), then ancestral reverse to 0 | Cheaper, single hyperparameter ($t^\star$) to sweep. Faithful in spirit to "use the clean prior to pull the perturbed input back onto the manifold". |
| Training data | Clean adjacency only | Clean features only (PROTEINS one-hot binary; ENZYMES z-scored continuous) | Same paradigm; different channel. |

## Known limitations of this instance

1. **PROTEINS one-hot is nearly degenerate.** Features are 3-dim binary one-hot, so a Gaussian denoiser is over-parameterised for the distribution. We expect modest signal on PROTEINS; the meaningful test is ENZYMES (21-dim continuous z-scored).
2. **No adjacency purification.** A KATZE-style attack that places injected nodes with manifold-aligned features but adversarial *edges* (homophily-aligned U3 picks) would not be touched here. This is a genuine gap relative to the paper's GSDR.
3. **$t^\star$ is fixed.** We default to 30 (\~15% noise on the schedule). The right sweep grid for the paper-grade table is $\{10, 20, 30, 60\}$.
4. **No NFCR.** A feature-smoothness post-pass is cheap and could be layered on top later.

## Empirical smoke test (fold 0, 40 graphs)

| dataset  | $t^\star$ | clean Δ vs no-def | zo_free ASR no-def | zo_free ASR + gddm |
|---|---|---|---|---|
| PROTEINS | 20 | -5.0 pp | 1.000 | 1.000 |
| ENZYMES  | 30 | -15.0 pp | 0.875 | 1.000 (ASR ↑) |
| ENZYMES  | 10 | -5.0 pp | 0.875 | 0.975 (ASR ↑) |
| ENZYMES  | 5  | +0.0 pp | 0.875 | 0.975 (ASR ↑) |

**Honest finding:** at every $t^\star$ tested, ASR under zo_free *increases* slightly after gddm-purification — the Gaussian feature denoiser does not reduce the effectiveness of a node-injection feature attack on a mean-pool GCN. This is consistent with the paper's own framing (GDDM targets *adjacency-modifying* attacks, not node injection) and is a useful negative result: it documents that the "feature-DiffPure" instantiation of the GDDM paradigm is not a meaningful defense against KATZE-family attacks. The right baseline for an *adjacency*-modifying ablation would need GSDR (not implemented here).

## When to revisit

- If reviewer asks for an adjacency-purification baseline: implement Binomial diffusion on $\mathbf{A}$ + GSDR. Estimate \~200 LoC; one-off training pass per dataset.
- If the paper releases code (none in v1 as of 2026-05): port directly.
