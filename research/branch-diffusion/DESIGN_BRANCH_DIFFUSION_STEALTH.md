# branch-diffusion · Stealth — Query-Frugal Stealthy Node-Injection with Pareto Analysis

**Date**: 2026-05-23
**Branch**: `branch-diffusion`
**Supersedes direction of**: `FINDINGS_N23.md` (diffusion-NoParam killed — 0 ASR over trivial baseline)

---

## 0. One-paragraph summary

The N23 diffusion attack is dead (adds 0 ASR over "inject 1 zero-feature node + full
edges" = 67.31%). The baseline KATZE pipeline reaches 94.85% but only by connecting
each injected node to **every** original node (`edge_strategy="full"`) with feature
magnitude `≈2√N` — i.e. zero stealth. This work replaces that with a **query-frugal
greedy black-box construction** that jointly optimizes (a) subgraph→victim edges,
(b) inter-node edges among injected nodes, and (c) injected features under a stealth
constraint — and characterizes the **ASR-vs-stealth and ASR-vs-query Pareto frontiers**
instead of chasing a single ASR number.

## 1. Threat model (explicit)

| | |
|---|---|
| Access | score-based black-box (CW margin on logits), query-only |
| Capability | injection-only (GIA): original graph immutable, append ≤ m nodes + edges |
| Victim | PROTEINS GCN: 3× GCNConv + BatchNorm + ReLU + dropout, **global_mean_pool**, MLP head |
| Eval | 5-fold, attack only correctly-classified test graphs |

Mean-pool dilutes an injected node by `1/(N+1)` → this is the surface the attack exploits.

## 2. Mechanism (Approach B — greedy edge + ZO feature)

Per graph, given `m` injected nodes and feature mode:

1. **Features** by `feat_mode`:
   - `stealth`: per-fold data-manifold value, clipped to `[0,1]` (binary-faithful, low norm)
   - `scaled`: stealth feature × `2√N` (detectable high-ASR reference = baseline regime)
   - `zo`: CGE-optimized features on fixed topology (feat_dim=3 → 6 queries/epoch, cheap)
2. **Inter-node edges** (`m≥2`): greedily add the internal edge that most reduces CW,
   over the ≤ `m(m-1)/2` candidates. This is the *malicious-subgraph* lever that
   `program.md` dismissed at m=1 (where it is meaningless).
3. **Victim edges**: adaptive greedy over a **degree-pre-filtered candidate pool**
   (lowest-degree victim nodes — `FINDINGS` theory: low-degree neighbors → larger
   pooled shift). Each step: 1 query/candidate, add the min-CW edge, check flip.
   Record `edges_used` at first flip; cap at `max_edges`.

**Why this is the right shape**: greedy is *incremental*, so a single run records, per
graph, the **minimum edges to flip**. The entire ASR-vs-edge-budget curve
`ASR(b) = frac(graphs flipped with ≤ b edges)` falls out of one pass — no 36-cell grid.

## 3. Metrics (every number → results JSON + timestamp)

- **ASR** mean±std over 5 folds (primary)
- **queries/graph** (secondary objective — minimize)
- **stealth**: edges-added / original-edges; injected-feature norm vs data manifold; m/N

## 4. Hypotheses (paired across folds)

| ID | H1 | H0 |
|---|---|---|
| **H1 subgraph** | at matched total-edge budget, m∈{2,3} + learned inter-node edges > m=1 | inter-node edges add nothing (confirms `program.md`) |
| **H2 stealth** | ∃ sparse `b_v ≪ N` with ASR within Δ of full-edge (full conn. is overkill) | ASR falls ~linearly with budget — no cheap stealthy regime |
| **H3 query** | greedy-edge + ZO-feature ≥ 94.85% baseline at fewer queries than 5-config restart | needs ≥ baseline query budget |

## 5. Deliverables

- `attack.py`: greedy construction + `run_attack_sweep()` (returns asr + per-graph
  min-edges + queries); `run_attack()` kept as float wrapper for `run.py` compatibility.
- `sweep_stealth.py`: iterate (m × feat_mode), 5-fold, write `results/exp_*.json`.
- Pareto report: ASR-vs-stealth + ASR-vs-queries figures (300 DPI) + markdown findings.

## 6. Anchors / sanity

- `scaled` features + full-edge budget must reproduce ≈94.85% (validates implementation).
- Trivial baseline (zero-feat, full-edge) = 67.31% is the documented lower bound.
