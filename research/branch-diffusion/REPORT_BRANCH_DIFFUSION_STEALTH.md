# branch-diffusion · Stealth — Pareto Sweep Report

**Generated**: 20260523_195832  
**Source data**: `results/sweep_stealth_20260523_195832.json`  
**Dataset/victim**: PROTEINS GCN (global_mean_pool), 5-fold  
**Baselines**: trivial(zero-feat,full-edge)=67.31%, KATZE=94.85%

Primary ASR metric = **best budget per graph** (flippable at any rho<=1); the attacker deploys whatever edge count works. `ASR@full` (naive single full-edge anchor) is shown for contrast.

## Per-config results

| m | feat_mode | ASR (best rho) | ASR@full | rho within 5pp | median greedy edges | q/graph | mean inter-edges |
|---|---|---|---|---|---|---|---|
| 1 | manifold | **66.81%** | 59.90% | rho=1.00 | 7.0 | 330.1 | 0.0 |
| 2 | manifold | **83.41%** | 67.77% | rho=1.00 | 2.0 | 721.8 | 0.3 |
| 3 | manifold | **88.52%** | 68.27% | rho=0.70 | 3.0 | 1069.1 | 0.4 |
| 1 | zo_box | **76.10%** | 71.05% | rho=1.00 | 8.0 | 562.4 | 0.0 |
| 2 | zo_box | **82.49%** | 74.77% | rho=0.70 | 2.0 | 1092.7 | 0.5 |
| 3 | zo_box | **85.18%** | 75.86% | rho=0.70 | 1.0 | 1560.1 | 0.9 |
| 1 | zo_free | **92.39%** | 88.74% | rho=0.80 | 2.0 | 382.3 | 0.0 |
| 2 | zo_free | **94.47%** | 88.88% | rho=0.70 | 1.0 | 751.5 | 0.5 |
| 3 | zo_free | **94.86%** | 88.29% | rho=0.50 | 1.0 | 1122.5 | 0.7 |

## Hypothesis verdicts

**H1 (malicious subgraph helps)**: **SUPPORTED only at equal-rho (more total edges); REFUTED at matched budget.**

_Naive view_ — best-budget ASR vs m (note: m nodes each get rho<=1, so m=3 spends ~3x the edges of m=1):

| feat_mode | m=1 | m=2 | m=3 | Δ(m2−m1) | Δ(m3−m1) |
|---|---|---|---|---|---|
| manifold | 66.81% | 83.41% | 88.52% | +16.60% | +21.71% |
| zo_box | 76.10% | 82.49% | 85.18% | +6.39% | +9.08% |
| zo_free | 92.39% | 94.47% | 94.86% | +2.08% | +2.47% |

_Matched total-edge budget_ — tau = total edges / N held constant (m nodes each use rho=tau/m). **m=1 wins in 80% of cases:**

| feat_mode | tau | m=1 | m=2 | m=3 | best |
|---|---|---|---|---|---|
| manifold | 0.2 | 11.1% | 11.9% | 8.1% | **m2** |
| manifold | 0.3 | 17.5% | 17.0% | 16.0% | **m1** |
| manifold | 0.5 | 31.1% | 27.1% | 24.8% | **m1** |
| manifold | 0.7 | 47.0% | 35.6% | 33.1% | **m1** |
| manifold | 1.0 | 66.8% | 54.8% | 45.2% | **m1** |
| zo_box | 0.2 | 20.2% | 22.4% | 27.7% | **m3** |
| zo_box | 0.3 | 28.8% | 25.0% | 31.9% | **m3** |
| zo_box | 0.5 | 54.6% | 35.8% | 35.9% | **m1** |
| zo_box | 0.7 | 68.7% | 50.2% | 45.7% | **m1** |
| zo_box | 1.0 | 76.1% | 72.6% | 60.2% | **m1** |
| zo_free | 0.2 | 46.8% | 45.4% | 45.5% | **m1** |
| zo_free | 0.3 | 60.7% | 48.2% | 50.2% | **m1** |
| zo_free | 0.5 | 76.0% | 62.6% | 54.2% | **m1** |
| zo_free | 0.7 | 86.7% | 72.9% | 65.8% | **m1** |
| zo_free | 1.0 | 92.4% | 86.7% | 76.1% | **m1** |

_Conclusion_: at a fixed stealth footprint, **concentrating all edges on a single optimized injected node dominates** a multi-node malicious subgraph — splitting edges across nodes dilutes each node's message strength and adds m to the mean-pool denominator. Inter-node edges are wasted budget under mean-pool. This confirms and sharpens `program.md`'s 'interconnection no improvement'.

**H2 (cheap stealthy regime exists)** — smallest rho within 5pp of best ASR:
- m=1 manifold: best ASR=66.81%, within 5pp at rho=1.00
- m=2 manifold: best ASR=83.41%, within 5pp at rho=1.00
- m=3 manifold: best ASR=88.52%, within 5pp at rho=0.70
- m=1 zo_box: best ASR=76.10%, within 5pp at rho=1.00
- m=2 zo_box: best ASR=82.49%, within 5pp at rho=0.70
- m=3 zo_box: best ASR=85.18%, within 5pp at rho=0.70
- m=1 zo_free: best ASR=92.39%, within 5pp at rho=0.80
- m=2 zo_free: best ASR=94.47%, within 5pp at rho=0.70
- m=3 zo_free: best ASR=94.86%, within 5pp at rho=0.50

**H3 (query frugality)**: greedy q/graph in table above; KATZE 5-restart CGE baseline ~ up to 1500 q/graph (6/epoch x 50 x 5, early-stop).

## Figures

- `charts/pareto_asr_vs_edges.png`
- `charts/pareto_asr_vs_queries.png`

_Every number traces to the source JSON above._
