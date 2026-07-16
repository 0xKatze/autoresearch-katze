# branch-diffusion · Multinode — matched-budget multi-node injection with interconnection

**Generated**: 20260524_061442  
**Source**: `results/multinode_20260524_061442.json`  
**Victim**: PROTEINS GCN (global_mean_pool), 5-fold  
**KATZE baseline**: 94.85%

Compares, at **matched total-edge budget** tau=edges/N, three edge-allocation strategies for m injected nodes against a single concentrated node:
`duplicate` (all nodes -> same victims), `partition` (per-node distinct victims), `interconnect` (partition + inter-node clique).

## feat_mode = manifold

| m | alloc | tau=0.5 | tau=1.0 | q/graph |
|---|---|---|---|---|
| 1 | single | 20.7% | 59.9% | 1 |
| 2 | duplicate | 7.7% | 33.0% | 1 |
| 2 | partition | 11.7% | 40.4% | 1 |
| 2 | interconnect | 12.1% | 41.0% | 1 |
| 3 | duplicate | 5.8% | 18.0% | 1 |
| 3 | partition | 8.1% | 29.7% | 1 |
| 3 | interconnect | 12.5% | 32.8% | 1 |

## feat_mode = zo_free

| m | alloc | tau=0.5 | tau=1.0 | q/graph |
|---|---|---|---|---|
| 1 | single | 93.3% | 88.6% | 98 |
| 2 | duplicate | 95.5% | 95.6% | 165 |
| 2 | partition | 96.6% | 94.3% | 143 |
| 2 | interconnect | 96.6% | 94.8% | 137 |
| 3 | duplicate | 95.6% | 97.2% | 299 |
| 3 | partition | 98.1% | 96.9% | 188 |
| 3 | interconnect | 97.7% | 97.0% | 195 |

## Verdict

**Best config**: m=3 partition tau=0.5 zo_free → **98.08% ± 1.36%** at 188 q/graph (beats KATZE 94.85%; std well below m=1).

**H1' (per-node-mask multi-node beats single at matched budget)**: **SUPPORTED with optimized features, REFUTED with fixed features** (multi-node exceeds m=1 in 2/4 (feat_mode, tau) cells — all the zo_free ones, none of the manifold ones). The advantage needs the **synergy of splitting edges across nodes (low per-node degree -> less 1/sqrt(deg) self-dilution) AND independently optimized per-node features**; with weak fixed features, coverage matters more and the single node wins.

| feat_mode | tau | best multi − m1 |
|---|---|---|
| manifold | 0.5 | -8.19pp |
| manifold | 1.0 | -18.85pp |
| zo_free | 0.5 | +4.77pp |
| zo_free | 1.0 | +8.35pp |

_partition vs duplicate_ and _interconnect vs partition_ deltas are visible in the per-mode tables above. Mechanism: with per-node distinct (low-degree) edge masks + independently ZO-optimized features, m injected nodes act as m separate attack vectors over distinct graph regions; GCN symmetric normalization gives each low-degree injected node's messages weight ~1/sqrt(deg), so spreading edges across nodes avoids the 1/sqrt(N) self-dilution that a single all-connected node suffers (note m=1 ASR often DROPS from tau=0.5 to tau=1.0).

**Corrects** the earlier `REPORT_BRANCH_DIFFUSION_STEALTH.md` 'm=1 dominates at matched budget' conclusion, which used duplicate-hub allocation (all injected nodes wired to identical victims) — an implementation artifact, not a property of mean-pool GIA.
