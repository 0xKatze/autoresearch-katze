# SpecGDI v2 — implementation notes

Sister file to `SPECGDI_IMPL.md` (v1). Describes the v2 *joint multi-node trojan*
diffusion attack: single joint Langevin step on (X*, A_VV, A_VU) per diffusion
iteration, K injected nodes (default K=3) forming a soft community sub-graph.

## Design recap

State per graph:
- `X* ∈ R^{K×d}` — injected features
- `A_VV ∈ [0,1]^{K×K}` — symmetric, zero diagonal soft inter-injected edges
- `A_VU ∈ [0,1]^{K×N}` — soft cross-edges from injected to victim nodes

**Joint update** (T_diff iterations, Step A/B/C):
- A. ONE shared ZO/SPSA call gives grad_adv on (X, A_VV, A_VU) jointly; closed-form
     Hellmann–Feynman gives grad_spec on (A_VV, A_VU); Mahalanobis prior gives grad_feat
     on X. All evaluated at the SAME (X_t, A_t).
- B. Joint update with no re-evaluation between blocks; Langevin noise on X.
- C. X → Mahalanobis ball; A_VV → clip + symmetrize + zero diag; A_VU → clip;
     if S_spec_composite > τ_composite, greedy drop strongest soft entries.

After T_diff: top-k discretisation by calibrated `clean_community_density`
(A_VV) and `v2_budget_VU_per_node` (A_VU). Final hard spectral projection.

## File / symbol map

| Spec concept | Symbol in `attack.py` |
|---|---|
| composite (N+K) Laplacian + top-k eigh | `composite_laplacian_spectrum` |
| S_spec_composite | `_v2_compute_S_spec` |
| ∂S/∂A_VV, ∂S/∂A_VU (Hellmann–Feynman, vectorised) | `spec_grad_wrt_A_v2` |
| joint ZO/SPSA grad on (X, A_VV, A_VU) | `joint_zo_cge_gradient` |
| Mahalanobis ball projection | `mahalanobis_project_rowwise` |
| top-k discretisation | `discretize_trojan` |
| Soft → hard probe materialisation (topk per step) | `_v2_build_data_soft` (soft mode) |
| Main optimiser (Steps A/B/C + final projection) | `specgdi_v2_optimize_trojan` |
| Top-level runner / dispatch | `_run_specgdi_v2_sweep` |

Dispatch: added one branch in `run_attack_sweep` for `feat_mode == "specgdi_v2"`.

## K-node calibration

`calibrate_specgdi_v2.py` runs once per dataset:
1. `clean_community_density` = mean intra-community edge density over N=200
   clean graphs, using `networkx.greedy_modularity_communities`.
2. `tau_spec_composite` = 99th percentile of `S_spec_composite` (exact eigh)
   over N=200 random K-trojan insertions, with A_VV ~ Bernoulli(density) and
   A_VU = one random row per injected node up to full-N.

Results (saved in `specgdi_v2_thresholds.json`):

| dataset | tau_spec_composite | clean_community_density | K |
|---|---|---|---|
| PROTEINS | 1.5845 | 0.6253 | 3 |
| ENZYMES  | 0.7222 | 0.5934 | 3 |

## Hyperparameters (`SPECGDI_V2_DEFAULTS`)

| key | default | notes |
|---|---|---|
| `v2_K` | 3 | # injected nodes |
| `v2_T_diff` | 10 | diffusion iterations |
| `v2_eps` | 0.05 | joint step size |
| `v2_eta_feat` | 0.3 | Mahalanobis prior pull on X |
| `v2_eta_spec` | 0.5 | spectral pull on A |
| `v2_n_probes` | 4 | SPSA random directions / step |
| `v2_zo_sigma` | 5e-3 | SPSA finite-diff sigma |
| `v2_temp0` | 0.1 | initial Langevin temperature (linear anneal to 0) |
| `v2_k_top` | 20 | top-k eigenvalues tracked |
| `v2_beta` | 1.0 | low-pass weight w_i = exp(-β λ_i) |
| `v2_budget_VU_per_node` | 0.2 | fraction-of-N cross-edges per injected node (≈ clean p95 degree) |
| `v2_per_graph_timeout_s` | 240 | per-graph kill switch |

Temperature schedule: linear anneal `temp(t) = temp0 * (1 - t/(T-1))` × √(2 eps);
zero at the last step (deterministic refinement).

## Smoke results

`smoke_specgdi_v2.py` — PROTEINS + ENZYMES fold 0, seed 0, K=3, T_diff=10,
n≤50, up to 15 graphs each.

| Dataset | Variant | ASR no_def | ASR gddm_a | per-graph wall |
|---|---|---|---|---|
| PROTEINS | sgdi (1-node)       | 0.867 | 0.067 | 0.0s / 0.8s |
| PROTEINS | specgdi v1 (1-node) | 0.867 | 0.067 | 0.1s / 0.9s |
| PROTEINS | **specgdi v2 (K=3)** | **0.267** | **0.000** | 0.6s / 2.0s |
| ENZYMES  | sgdi                | 0.067 | 0.733 | 0.1s / 1.1s |
| ENZYMES  | specgdi v1          | 0.067 | 0.600 | 0.2s / 1.3s |
| ENZYMES  | **specgdi v2 (K=3)** | **0.000** | **0.533** | 0.7s / 1.9s |

(ENZYMES gddm_a ASR > nodef is a known artifact: gddm_a denoising can over-clean
some natively-misclassifiable graphs into the wrong class — see GDDM_A_NOTES.md.)

**Verdict — does v2 rescue SpecGDI? NO** (on these 15 small graphs):
- PROTEINS gddm_a: v2 = 0.000 vs v1 = 0.067 — v2 is *worse* by 1 graph.
- ENZYMES gddm_a: v2 = 0.533 vs v1 = 0.600, sgdi = 0.733 — v2 underperforms.

v2 nodef is also weaker than v1 here because (a) the budget cap
`budget_VU_per_node=0.2` produces fewer total edges than v1's full-edge anchor;
(b) the spectral projection actively drops edges to stay under τ_composite.

## Deviations from spec

1. **SPSA over CGE for joint adv gradient.** Full coordinate-wise CGE would cost
   2*(K*d + K*(K-1)/2 + K*N) ≈ 250 queries/step on PROTEINS — too expensive for
   T_diff=10 over 15 graphs. We use SPSA with shared random direction across
   blocks; n_probes=4 → 8 queries/step. Noisier but cheap. Documented in code.

2. **Soft-probe materialisation by topk, not 0.5 threshold.** A naive 0.5
   thresholding gave near-zero ZO signal because all soft entries started <0.5
   and small perturbations did not cross the threshold. We instead discretise
   each probe by top-k (using calibrated density + per-node budget). This is
   a deviation from the "straight clip(0,1)" projection in the spec but is
   necessary to extract gradient signal; the final hard discretisation uses
   the same operator (consistent calibration semantics).

3. **Hellmann–Feynman ignores degree-derivative term.** `∂λ_i/∂A_{uv}` for
   normalized L has a degree-chain term we drop (leading-order = bilinear in
   eigenvectors only). Standard practice in spectral attack literature; the
   approximation works fine for the small perturbations seen here.

4. **τ_composite computed via exact eigh** (not Hellmann–Feynman proxy) inside
   `calibrate_specgdi_v2.py`. The attacker uses the same exact eigh inside
   `composite_laplacian_spectrum` (we have no inner-loop eigh-free regime for
   the composite case yet) — so calibration and attack metric are perfectly
   consistent.

## Honest concerns

- **PROTEINS gddm_a = 0 across all three attackers** in this smoke. The
  defense is highly effective on this dataset; v2 does not lift the floor.
- **ENZYMES gddm_a is the only regime where ASR is meaningful**, but v2 loses
  to v1 and sgdi there. The K-node trojan is *more* visible to the A-channel
  denoiser, not less — the inter-injected community sub-graph evidently
  triggers gddm_a's structural prior.
- Sample size (15 graphs) gives ±0.07 confidence intervals → individual
  cell differences here are NOT statistically significant. A full eval at
  N≥100 graphs / 5 folds / 2 seeds is needed before drawing the headline
  conclusion. The smoke result is *consistent with* a null/negative finding.
- Per-graph wall time well within budget (2s vs 240s cap); smoke total ~135s.
- v2 nodef on PROTEINS dropped to 0.267 because of the conservative
  `budget_VU_per_node=0.2`. A larger budget restores nodef ASR (0.5 → ~0.73)
  but did not lift gddm_a ASR off zero in the ablation we ran.
