# Spectrum Diagnostic — Clean vs Injected Normalized-Laplacian Spectra

**Script:** `diagnose_spectrum.py`
**Output:** `results/spectrum_diagnostic_20260526_092141.json`
**Setup:** seed=0, 5 folds, top-k=20 eigenvalues, full-edge anchor (rho=1.0),
spectral discrepancy = `||sort(lam_clean)[:k] - sort(lam_pert)[:k]||_2`.

## Headline numbers (mean ± std across all test graphs, single seed)

| dataset  | feat_mode  | mean Δ | median Δ | std Δ  | n graphs |
| -------- | ---------- | ------ | -------- | ------ | -------- |
| PROTEINS | zo_normcap | 0.5262 | 0.4980   | 0.1736 | 812      |
| PROTEINS | zo_free    | 0.5262 | 0.4980   | 0.1736 | 812      |
| PROTEINS | sgdi       | 0.5262 | 0.4980   | 0.1736 | 812      |
| ENZYMES  | zo_normcap | 0.4835 | 0.4653   | 0.0994 | 350      |
| ENZYMES  | zo_free    | 0.4835 | 0.4653   | 0.0994 | 350      |
| ENZYMES  | sgdi       | 0.4835 | 0.4653   | 0.0994 | 350      |

(No defense cell is broken out: the **perturbation topology at rho=1.0 is
defense-independent** — defenses act at inference, not at injection-build time —
so spectrum is a property of (graph, feat_mode) only.)

## Verdict

**SGDI does NOT preserve graph spectrum better than zo_normcap or zo_free —
because at the full-edge anchor used by `eval_paper_grade.py`, *all three feat
modes produce the EXACT same topology*: one injected node connected to every
victim node** (`all_v = [(0, v) for v in range(n)]` in `attack.greedy_attack_graph`).
The symmetric normalized Laplacian `L = I − D^(−1/2) A D^(−1/2)` is a function of
the adjacency only — feature vectors do not enter it — so `lam_pert` is
mathematically identical across `zo_normcap`, `zo_free`, and `sgdi`. The triple
identical means/medians/stds above (to machine precision) are a sanity check, not
a coincidence.

What the numbers DO say:
- Adding one universal-degree node perturbs the top-20 small eigenvalues by
  ~0.48–0.53 in L2, on graphs of 11–100+ nodes. This is a **large** spectral
  shift — adding a node of degree n to a graph of n nodes drastically rebalances
  the degree distribution and the resulting normalized Laplacian.
- PROTEINS Δ slightly larger and noisier than ENZYMES (heavier-tailed graph-size
  distribution; ENZYMES test graphs cluster tighter in n).

## Implication for SpecGDI

The result is **positive for the new lever, not for SGDI as-is**:

1. SGDI's contribution lives **entirely in feature space** — it samples inside
   the data covariance ellipsoid. It does nothing to topology.
2. The full-edge anchor is the worst possible topology for spectral stealth (it
   maximises the perturbation to D and to the Laplacian's smallest eigenvalues).
3. Therefore a **Spectral-Guided variant (SpecGDI)** that selects victim edges
   to *minimise* spectral discrepancy — instead of taking the trivial full-edge
   anchor or the low-degree greedy heuristic — has clear, unexplored headroom:
   roughly a 0.48–0.53 L2 budget to claw back via edge selection.
4. This makes SpecGDI a **genuinely new axis**, not an incremental SGDI tweak.
   The natural next experiment is to (a) ablate over edge-budget rho (already
   logged by `frac_anchors`), (b) build a spectrum-aware victim-edge scorer,
   and (c) re-run this diagnostic + a Laplacian-spectrum defense.

## Issues / caveats

- **No defense axis breakout.** The injection topology is defense-blind, so a
  per-defense breakdown would be redundant (numbers would be identical).
- **Isolated nodes.** Normalized-Laplacian convention used here zeros the
  D^(−1/2) entry on isolated nodes (no eigenvalue normalization tricks). No
  test graph had isolated nodes; reported `iso_clean`/`iso_pert` are 0
  throughout the per-graph JSON.
- **Padding convention.** Clean has n eigenvalues, perturbed has n+1; we
  compare the first k=min(20, n) smallest eigenvalues of each (matched-rank).
  This is one of two reasonable conventions; the alternative (compare *full*
  spectra after zero-padding the clean side) would give a slightly different
  number but the same cross-mode equality.
- **Seed coverage.** Single seed (seed=0). Since the topology is deterministic
  (lowest-degree anchor list `deg_order[:N] == range(N)` up to ties), and the
  feature optimisation does not affect the spectrum, seed variance on Δ is
  exactly zero.
- **PROTEINS one-hot.** PROTEINS feat is binary one-hot; this does not change
  the spectral conclusion (again: features do not enter L).
- **Stale meta path.** `models_cont/meta_cont_ENZYMES.json` records absolute
  `model_path` values that point one level up from where the files live
  (`/workspace/autoresearch-katze-n23/models_cont/...` vs the actual
  `research/branch-diffusion/models_cont/...`). `diagnose_spectrum.py`
  falls back to the local file by basename; no upstream file was modified.
