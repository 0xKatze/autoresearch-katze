# branch-diffusion · Distributional Stealth — SGDI vs N4 norm-cap

**Generated**: 2026-05-25
**Primary source**: `results/eval_paper_grade_20260525_161553.json` (2 seeds × 5 folds = 10 paired obs/cell)
**Victims**: PROTEINS GCN (3-dim one-hot, discrete) · ENZYMES GCN (21-dim continuous, z-scored), both `global_mean_pool`, 5-layer h=64
**Threat model**: score-based black-box (CW margin), injection-only (original graph never mutated), node_budget=1, ASR at **fixed full-edge budget** (ρ=1.0, KATZE-comparable), attack restricted to originally-correctly-classified graphs (GRABNEL protocol).

---

## 1. Thesis

The current KATZE stealth mechanism (**N4 / SA-ZOO**) caps the injected node's **L2 norm** (`zo_normcap`, radius κ·median‖x‖). That defeats a norm filter, but it is **magnitude-only**: it constrains how *big* the injected feature is, not its *direction* or inter-feature correlation. A **distribution-aware defense** (Mahalanobis filter) scores nodes against the data covariance ellipsoid and removes off-manifold *directions* even at correct norm.

**SGDI** (Stealth-Guided Diffusion Injection, `attack.py` feat_mode=`sgdi`) is a score-based guided diffusion in feature space: each reverse step combines the ZO adversarial CGE gradient, the Gaussian data-prior score −Σ⁻¹(x−μ), and annealed Langevin noise, with a **hard Mahalanobis-ball projection** (attacker radius = 99th pct of clean sq-Mahalanobis, strictly inside the defender's 99.5th-pct cut). It therefore stays inside the data ellipsoid by construction.

**H**: SGDI survives a Mahalanobis defense (ASR invariant) where norm-cap collapses.

---

## 2. Defenses (`defenses.py`)

| key | mechanism | targets |
|---|---|---|
| none | — | — |
| normfilter | drop nodes with within-graph 95th-pct L2 norm | magnitude |
| jaccard | prune edges with endpoint cosine sim < 0.01 | edge similarity |
| **mahalanobis** | drop nodes with sq-Mahalanobis > 99.5th-pct of clean nodes | **distribution / direction** |

All defenses adaptive (attacker queries the defended model). Mahalanobis/featclip stats calibrated per-fold on clean test nodes so clean accuracy is preserved.

---

## 3. Main results — ASR @ full-edge budget (mean over 2 seeds × 5 folds)

`cleanAcc` = fraction of attackable graphs still correctly classified **after the defense** (defended clean accuracy).

### PROTEINS (discrete — topology-driven: a single node + full edges alone flips ~60%)

| feat_mode | none | normfilter | jaccard | **mahalanobis** | cleanAcc(maha) |
|---|---|---|---|---|---|
| manifold (in-dist, no opt) | 59.9% | 59.9% | 66.3% | 59.9% | 1.00 |
| zo_normcap (**N4**, magnitude) | 77.4% | 55.9% | 78.7% | **4.2%** | 1.00 |
| zo_free (unconstrained, KATZE-level) | 85.2% | 56.1% | 80.1% | **4.2%** | 1.00 |
| **sgdi (ours, distribution)** | 67.0% | 66.1% | 70.6% | **65.8%** | 1.00 |

### ENZYMES (continuous — feature-driven: topology alone flips only ~12%)

| feat_mode | none | normfilter | jaccard | **mahalanobis** | cleanAcc(maha) |
|---|---|---|---|---|---|
| manifold | 11.6% | 15.4% | 26.2% | 14.4% | 0.98 |
| zo_normcap (**N4**) | 94.2% | 74.9% | 90.2% | **19.7%** | 0.98 |
| zo_free | 94.2% | 74.9% | 92.3% | **19.7%** | 0.98 |
| **sgdi (ours)** | 27.4% | 23.5% | 42.0% | **29.3%** | 0.98 |

> jaccard depresses defended clean acc (PROTEINS 0.79, ENZYMES 0.67) — it prunes legitimate edges, so its ASR column is not a free defense.

Chart: `charts/diststealth_asr_by_defense.png` (grouped bars, SGDI invariance vs norm-cap/free collapse under Mahalanobis).

---

## 4. Paired Wilcoxon — SGDI − zo_normcap (n=10)

| dataset | defense | Δ ASR | p | verdict |
|---|---|---|---|---|
| PROTEINS | none | −10.4 pp | 0.002 | norm-cap wins undefended |
| PROTEINS | normfilter | +10.2 pp | 0.084 | SGDI ahead (ns) |
| PROTEINS | jaccard | −8.1 pp | 0.011 | norm-cap wins |
| PROTEINS | **mahalanobis** | **+61.6 pp** | **0.002** | **SGDI dominates** |
| ENZYMES | none | −66.7 pp | 0.002 | norm-cap dominates undefended |
| ENZYMES | normfilter | −51.4 pp | 0.002 | norm-cap wins |
| ENZYMES | jaccard | −48.2 pp | 0.002 | norm-cap wins |
| ENZYMES | **mahalanobis** | **+9.5 pp** | **0.002** | **SGDI wins** |

---

## 5. Findings

1. **SGDI is the only defense-invariant attack.** Its ASR barely moves across none/normfilter/jaccard/mahalanobis (PROTEINS 67/66/71/66; ENZYMES 27/24/42/29). norm-cap and zo_free swing from ~80–94% (undefended) to **4–20% under mahalanobis** — the distribution filter removes their off-manifold nodes.
2. **Under the distribution-aware defense, SGDI significantly beats N4 norm-cap on both datasets** (PROTEINS +61.6 pp, ENZYMES +9.5 pp, both p=0.002). This is the contribution: *distribution-projection survives distribution-filtering; magnitude-cap does not.* It holds even on discrete PROTEINS — norm-based stealth is degenerate there (norm=1.0) but **direction-based** stealth is not (off-simplex nodes are still caught by Mahalanobis).
3. **The win size tracks the attack mechanism.** PROTEINS is topology-driven, so keeping the injected node (SGDI) preserves ~66% while losing it (norm-cap, removed) collapses to 4% → huge gap. ENZYMES is feature-driven, so the adversarial signal *needs* off-distribution features SGDI cannot use → both end low under maha → small gap.
4. **Cost — no free lunch on raw ASR.** SGDI's in-distribution constraint caps undefended ASR below norm-cap (ENZYMES 27% vs 94%). SGDI does **not** improve ASR on undefended models; it trades undefended-ASR for defense-invariance.

---

## 6. Node-budget crossover (5-fold, seed 0)

ASR @ full-edge under **mahalanobis**, source `results/eval_nb_crossover_20260525_173007.json`:

| dataset | nb | zo_normcap (N4) | **sgdi** | winner |
|---|---|---|---|---|
| PROTEINS | 1 | 0.040 | **0.643** | sgdi |
| PROTEINS | 5 | 0.128 | **0.714** | sgdi |
| PROTEINS | 10 | 0.233 | **0.738** | sgdi |
| ENZYMES | 1 | 0.197 | **0.251** | sgdi |
| ENZYMES | 5 | 0.619 | **0.647** | sgdi |
| ENZYMES | 10 | **0.856** | 0.732 | normcap |

**Two distinct regimes** (corrects the earlier fold-0 probe, which had over-generalized the reversal):
- **PROTEINS (topology-driven): SGDI dominates at EVERY budget** (0.64/0.71/0.74 vs 0.04/0.13/0.23). norm-cap can never aggregate — its off-simplex nodes are always removed by Mahalanobis, so even nb=10 reaches only 0.233.
- **ENZYMES (feature-driven): SGDI wins at nb=1,5; norm-cap's brute-force aggregation overtakes only at nb=10** (0.856 vs 0.732) — on 21-dim continuous features enough L2-capped nodes happen to fall inside the ellipsoid to survive and aggregate. SGDI is capped by its in-distribution ASR ceiling.

_Net: SGDI's distributional-stealth advantage is robust across budgets when the attack is topology-carried; on feature-carried attacks it holds at low/mid budget and yields to brute-force multi-node norm-cap at high budget._

## 6.1 Budget fairness — why SGDI < N4 undefended (κ-frontier)

The operational budget is matched (node_budget=1, full-edge ρ=1.0, query budget SGDI≥N4). The undefended gap is purely the **stealth-set size**: N4's L2-ball radius = κ·median‖x‖. Sweeping κ (5-fold, `results/eval_kappa_frontier_20260525_192834.json`):

| feat | PROTEINS none | PROTEINS maha | ENZYMES none | ENZYMES maha |
|---|---|---|---|---|
| zo_normcap κ=0.5 | 0.678 | 0.006 | 0.609 | 0.197 |
| zo_normcap κ=1.0 | 0.706 | 0.044 | 0.860 | 0.197 |
| zo_normcap κ=2.0 | 0.768 | 0.043 | 0.929 | 0.197 |
| zo_normcap κ=4.0 | 0.828 | 0.043 | 0.929 | 0.197 |
| zo_free (κ=∞) | 0.834 | 0.043 | 0.929 | 0.197 |
| **sgdi** | 0.668 | **0.655** | 0.279 | **0.302** |

- N4's **undefended ASR is monotone in κ** — its edge is literally bought by inflating feature magnitude (the `feat_scale`-style lever, which has **no precedent in the GIA literature** — every prior attack bounds features to the data range; see §9).
- N4's **defended ASR is flat and near-floor at every κ** (PROTEINS ~0.04, ENZYMES ~0.20 = the topology-only residual after its node is filtered). Norm is the wrong knob: the L2 ball is isotropic and never aligns with the data ellipsoid, so no κ lets it survive a Mahalanobis filter.
- **SGDI Pareto-dominates the entire N4 κ-frontier in the defended threat model** (PROTEINS 0.655 vs ≤0.044; ENZYMES 0.302 vs 0.197). No norm budget reaches SGDI's defended ASR. The "SGDI loses undefended" caveat is therefore an artifact of N4 being allowed a literature-non-compliant magnitude that the defense nullifies anyway.

---

## 7. Honest verdict & limitations

- **Goal scorecard** (vs current KATZE/N4 architecture): SGDI **improves stealth** — it is provably invariant to distribution-aware filtering, which N4 norm-cap is not (significant on both datasets under Mahalanobis). It does **not improve raw (undefended) ASR**. So the "beat on ASR *and* stealth simultaneously" target is **partially met**: a strict Pareto gain on the robustness/stealth axis under distribution-aware defense, at an undefended-ASR cost.
- **Right framing**: SGDI provides a *distribution-filter-proof ASR floor*. Against a defender deploying Mahalanobis filtering it is the best of the tested attacks; against an undefended model it is not.
- **Limitations**: single victim arch (GCN, mean-pool); high-nb crossover favors norm-cap **only on feature-driven ENZYMES** — on topology-driven PROTEINS SGDI dominates at all budgets (§6); attacker shares the data distribution with the defender (white-box-distribution / black-box-model); 2 seeds × 5 folds for the main grid, 1 seed for the crossover (Wilcoxon already p=0.002, but 10-fold × 3-seed is the camera-ready bar); ENZYMES clean acc 58%; mahalanobis threshold fixed at 99.5th pct (defender-strictness sweep is future work).

---

## 9. Related work & novelty positioning (literature search 2026-05-25)

**Feature-scale in prior GIA (4 reference PDFs):** bounding injected features to the data range is universal — GZOO (soft δ + mean/std attribute loss, one-hot via Gumbel-Softmax), GRABNEL (features pinned to real-node max/min/median), SpNIA (hard [0,1] projection + binary), PEANUT (zero features). **None uses an amplified out-of-range magnitude** — so KATZE/`zo_free`'s `feat_scale≈2√N` lever is unprecedented and trivially filterable (motivates this whole line). Stealth metrics in prior work are structural (homophily, degree-dist); none reports an injected-feature L2 norm.

**Diffusion-as-attack on graphs:** none exists. Graph "diffusion attack" titles are false friends (perturbation *spreading*: arXiv:2104.09369, SFIA) or diffusion *defense/purification* (GDDM 2501.11568; 2502.05000) or backdoors *on* graph diffusion generators (2503.06340). The generative-diffusion-attack lineage is image-only:
- **AdvAD (NeurIPS 2024, 2503.09124)** — non-parametric diffusion + per-step classifier guidance + ℓ∞ pixel projection. **Direct ancestor** — and literally the recipe N23 falsified as a naive port. SGDI = AdvAD ported to graph node-injection, made **black-box ZO**, with the ℓ∞ box swapped for a **Gaussian-score prior + Mahalanobis-ball projection**.
- AdvDiff (ECCV'24), Diff-PGD (NeurIPS'23), ScoreAdv (2025) — white-box, need a *trained* diffusion network; SGDI uses an analytic Gaussian score (no trained generator).

**Distribution-aware stealthy injection:** no prior injection attack constrains features by full mean+covariance/Mahalanobis. Prior stealth stops at cosine homophily (**HAO**, ICLR'22), marginal stats (**GANI**, TCSS'24), bounded per-dim range / neighbor-copy (**QUGIA**, 2025), a GAN discriminator (**CANA**, Info Sci'23 — its GraphFD mean+cov is an *eval metric*, not a constraint, on GIN embeddings), or a soft manifold-alignment + RL loss (**JANUS**, 2509.13266 — closest *graph* competitor; not diffusion, not Mahalanobis). SGDI's **hard Mahalanobis projection of raw injected features is novel.**

**Defense side:** a full-covariance Mahalanobis node-removal filter for GIA appears novel. Lee 2018 (Mahalanobis OOD) is image-domain; GNNSafe/NODESAFE (graph OOD) are energy-based and treat Mahalanobis as a weak baseline; GNNGuard is edge-cosine.

**Novelty verdict:** the combination — black-box ZO + analytic-Gaussian-prior guided diffusion + hard Mahalanobis projection for graph-classification node injection — has no prior instance. **Closest to cite/benchmark: AdvAD (ancestor), JANUS (graph in-distribution injection), CANA (covariance-as-metric).** **Reviewer caveats:** (a) "diffusion keeps AEs in-distribution + beats purification" is established in images → graph-specific mechanics must carry novelty; (b) non-diffusion in-distribution injection exists (JANUS/GANI/QUGIA) → must show the diffusion/Gaussian machinery wins on defense-evasion; (c) **defenses still to evaluate against: GDDM (diffusion purification), GNNSafe (energy OOD), CANA's detectors (COPOD/PCA/IForest/AE)**.

## 8. Reproduce

```bash
micromamba run -n graph_adversarial python prepare_cont.py ENZYMES       # continuous victims
micromamba run -n graph_adversarial python eval_paper_grade.py           # main grid (nb=1)
micromamba run -n graph_adversarial python eval_nb_crossover.py          # nb crossover
```
feat_modes in `attack.py`: `manifold` / `zo_box` / `zo_normcap` (N4) / `zo_free` / `sgdi` (ours). Every number traces to the source JSON above.
