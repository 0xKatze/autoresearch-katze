# SpecGDI implementation — design ↔ code map

Sister file to `branches/branch-diffusion/specgdi-spectral-guided-diffusion.md`.
This is the as-built mapping: which spec section is implemented by which symbol,
where we deviated, and the per-step complexity.

## File map

| File | Role |
|---|---|
| `attack.py` (new SpecGDI block, ~300 LOC appended) | core attack |
| `calibrate_specgdi.py` | calibrates τ_spec → `specgdi_thresholds.json` |
| `test_specgdi.py` | PROTEINS fold-0 smoke test |
| `eval_paper_grade.py` | 1-line addition: `"specgdi"`, `"specgdi_signal"` in `FEAT_MODES` |

Untouched: `prepare.py`, `prepare_cont.py`, `run.py`, and every pre-SpecGDI
function in `attack.py` (`sgdi_optimize_features`, `zo_optimize_features`,
`greedy_attack_graph`, etc.). The only edit inside an existing function is a
single 4-line early-return at the top of `run_attack_sweep` that dispatches the
two new feat modes — this is the "single dispatch branch" allowance per the
implementation brief.

## Spec → code

| Spec § | Equation / concept | Symbol in `attack.py` |
|---|---|---|
| §2 | $\mathcal S_{\text{spec}} = \sum_i w_i (\lambda_i(L') - \lambda_i(L))^2$, $w_i = e^{-\beta \lambda_i}$ | `spec_disc(lam_pert, lam_clean, beta)` |
| §2 | eigendecomposition of normalized Laplacian (top-k smallest) | `compute_laplacian_spectrum(A_dense, k)` (single eigh on CPU, kept ~1 ms for n<150) |
| §2.1 | Hellmann–Feynman / interlacing first-order shift $\Delta \lambda_i \approx u_i^\top \Delta L u_i$ | `closed_form_delta_lambda(U_clean, lam_clean, dL)` — zero-pads U into the (n+1)-block when an injected node augments L |
| §2.2a | $\nabla_x (x^{*\top} L_{v^*} x^*)$ smoothness gradient | `_signal_smoothness_grad(x_inj, x_neighbours, deg_inj)` (combinatorial form: $2(x^* - \bar{x}_{N(v^*)})$) |
| §2.3 | joint Langevin + greedy single-edge swap update | `_specgdi_optimize_one(...)` |
| §2.3 hard projection | greedy drop edges until $\mathcal S_{\text{spec}} \le \tau_{\text{spec}}$ | tail of `_specgdi_optimize_one` (no eigh inside loop) |
| §2.3 τ_spec calibration | 99-th pct of $\mathcal S_{\text{spec}}$ over $N=200$ random single-node insertions | `calibrate_specgdi.py` → `specgdi_thresholds.json` |
| §3 (combined update) | $S_{t-1} = \arg\min_{S' \in \mathcal N(S_t)} \mathcal S_{\text{spec}}(L'(S'))$ s.t. $\mathcal L_{\text{adv}}(x, S') \le \mathcal L_{\text{adv}}(x, S)$ | edge step inside `_specgdi_optimize_one` — top-$K$ candidates by closed-form $\Delta\mathcal S_{\text{spec}}$ are ZO-probed for $\mathcal L_{\text{adv}}$ |
| §3 dispatch | new feat modes `specgdi` (B1) and `specgdi_signal` (B2) | branch at top of `run_attack_sweep` → `_run_specgdi_sweep` |

## Config (defaults from `SPECGDI_DEFAULTS` in `attack.py`)

| key | default | meaning |
|---|---|---|
| `specgdi_eta_spec` | 0.3 | weight on $\Delta\mathcal S_{\text{spec}}$ when ranking candidate swaps |
| `specgdi_k` | 20 | top-k Laplacian eigenpairs |
| `specgdi_beta` | 1.0 | low-pass weight decay $w_i = e^{-\beta \lambda_i}$ |
| `specgdi_swaps_per_step` | 8 | $K$ in "top-$K$ candidates ZO-probed" |
| `specgdi_T_diff_default` | 20 | $T$ if `sgdi_T_diff` is not in cfg |
| `specgdi_eta_sm` | 0.5 | §2.2 smoothness-gradient weight (B2 only) |
| `specgdi_tau_path` | `specgdi_thresholds.json` | dataset → $\tau_{\text{spec}}$ map |

## Complexity per diffusion step

Per graph with $n$ clean nodes, top-$k$ spectrum, $K$ swap candidates:

- 1 eigh up-front: $O(n^3)$, once per graph. Cached.
- Feature step: $2md + 1$ ZO queries (same as SGDI).
- Edge step:
  - $|S|$ closed-form drop scores: each $O(n^2 k)$ for the dL contraction.
  - For each of top-$K$ drops, $|V \setminus S|$ candidate adds → $O(K \cdot n \cdot n^2 k)$.
  - 1 ZO probe to accept the swap.
- Hard projection: at most $2n$ greedy drops, each $O(|S| \cdot n^2 k)$.

For PROTEINS test graphs (n typically 10–80) the dominant cost is the inner
dL contraction; we keep $K = 8$ (config) and never re-eigh — the spec's "no
eigh inside the inner loop" constraint holds.

## Deviations from the spec

1. **Edge initialisation = full-edge anchor, then *prune/swap*** (B1 brief). The
   spec §2.1 phrases this as "greedy / sub-modular selection of $S$"; we chose
   pruning from the full set because the eval harness's `skip_greedy` regime
   anchors at $\rho = 1.0$, so starting full-edge means the worst-case adversary
   baseline is recovered if $T = 0$.

2. **Pure-drop branch.** The §2.3 swap neighbourhood is "single-edge swap"
   (drop + add); we additionally allow a pure-drop move when no add improves
   $\mathcal S_{\text{spec}}$. This is strictly inside the larger
   $\mathcal N(S_t)$ neighbourhood the spec discusses (interlacing covers
   rank-1 perturbations of either sign) and is needed because edge count is
   not constrained — many graphs hit $\mathcal S_{\text{spec}} \le \tau_{\text{spec}}$ only after net edge removal.

3. **Smoothness gradient uses the *combinatorial* form** $2(x^* - \bar x_{N(v^*)})$
   instead of the strictly-normalized $D^{-1/2}$-weighted form. The spec
   (§2.2a) calls out either is acceptable — the combinatorial form is one
   line of code, positive-semi-definite, and equivalent up to a positive
   degree-dependent scaling absorbed by `specgdi_eta_sm`.

4. **τ_spec proxy ≡ attacker's S_spec proxy** (intentional consistency).
   `calibrate_specgdi.py` uses the same Hellmann–Feynman closed form the
   attacker uses inside `_specgdi_optimize_one`. The spec implies an "exact"
   τ_spec; we use the proxy for both. If we later switch the attacker to a
   full re-eigh, we should re-calibrate.

5. **ASR semantics under the new feat_mode.** `_run_specgdi_sweep` returns
   `asr_full = mean(full_flip)` where `full_flip` is the *final-attack* flip
   (after spec-stealth feature + edge selection), not the rho=1.0-anchor flip.
   This is the only sensible reading because the SpecGDI edge set is in
   general a strict subset of the full-edge anchor. The standard rho=1.0
   anchor evaluation is still available via existing feat modes.

## Open questions / next steps

- B2 (`specgdi_signal`) is wired but the smoothness term hasn't been tuned —
  default `eta_sm = 0.5` is a placeholder. Sweep alongside the §5 ablations.
- For larger graphs (n > 150) the eigh path stays correct but should switch to
  `scipy.sparse.linalg.eigsh` for the one-shot decomposition; the inner loop is
  already eigh-free.
- The `data_prior` (SGDI's Mahalanobis ellipsoid) is computed across `test_graphs`
  — same as SGDI today. For paper-grade isolation we should consider computing
  it on the train split only; this is a generic SGDI/SpecGDI question, not new.
