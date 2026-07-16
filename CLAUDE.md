# autoresearch-katze

Two separable things live here (split 2026-05-25):

- **`framework/`** — the reusable **autoresearch FRAMEWORK** (generic self-driving research loop), inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) + [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw). Domain-agnostic; drives experiments on *any* research line.
- **`research/branch-diffusion/`** — the **RESEARCH** that applies the framework: diffusion-based node-injection attacks on graph classifiers (N23 → N24 → SGDI). This is a **branch of GZOO-KATZE** (git branch `branch-diffusion`); in the wiki it lives under `projects/gzoo-katze/branches/branch-diffusion/`.

> **Framework = how the search is driven; research = what was found.** Keep them distinct.
> The framework references research artifacts via `framework/_paths.py::RESEARCH_DIR` (default `../research/branch-diffusion`, override with the `RESEARCH_DIR` env var to point at a different research line).

## Quick Start

```bash
# 1. Prepare (once): download data, train victim models
python research/branch-diffusion/prepare.py

# 2a. Manual mode: modify research/branch-diffusion/attack.py, run experiment
python research/branch-diffusion/run.py          # full 5-fold (~3 min)
python research/branch-diffusion/run.py --quick  # 1-fold fast test (~30s)

# 2b. Automated pipeline (framework drives the research dir):
python framework/pipeline.py --max-iterations 20 --target-asr 0.95
python framework/pipeline.py --step / --dry-run / --analyze-only
python framework/pipeline.py --charts / --review / --package

# 2c. Stealth / defense evaluation (this branch's contribution):
python research/branch-diffusion/eval_paper_grade.py        # §4.4 grid (SGDI vs N4 × defenses)
python research/branch-diffusion/eval_nb_crossover.py
python research/branch-diffusion/eval_kappa_frontier.py
```
Scripts use `Path(__file__)`-relative paths, so invoking them by full path from the repo root works (each puts its own dir on `sys.path[0]`).

## Structure

```
framework/                ← reusable autoresearch loop (generic)
  _paths.py               ← FRAMEWORK_DIR + RESEARCH_DIR (the split's seam)
  pipeline.py             ← orchestrator
  knowledge.py hypothesis.py codegen.py executor.py analyzer.py decision.py
  charts.py evolution.py review.py deliverables.py

research/branch-diffusion/  ← the diffusion-attack research (branch of gzoo-katze)
  # Fixed harness (DO NOT MODIFY content; they stay co-located so their
  # Path(__file__)-relative imports resolve):
  prepare.py              ← dataset + victim model
  run.py                  ← runs attack.py, records results/exp_*.json
  attack.py               ← AI modifies this: attack logic + feat_modes (manifold/zo_box/zo_normcap/zo_free/sgdi)
  # Research scripts (this session):
  prepare_cont.py defenses.py defense_eval.py
  eval_paper_grade.py eval_nb_crossover.py eval_kappa_frontier.py make_diststealth_chart.py
  sweep_stealth.py multinode.py
  program.md best.json    ← framework config + state (per research line)
  data/ models_saved/ models_cont/ results/ charts/   ← data + artifacts
  REPORT_BRANCH_DIFFUSION_*.md DESIGN_*.md FINDINGS_N23.md HANDOFF_N23.md
  attack_diffusion.py quick_test_n23.py   ← N23 sub-stage records (attack_diffusion is stale: old attack.py API)

.claude/                  ← project tooling (skills/agents/commands/hooks; see below)
```

## Tooling (`.claude/`, consolidated from GZOO-KATZE 2026-05-25)

Skills: `autoresearch` (the framework loop), `stealth-eval` (this branch's defense/SGDI eval workflow), `research-verification`, `academic-paper-research`, `document-reader-editor`, `notebooklm`, `save`, `ppt-table-generator`. Agents: literature-reviewer / experiment-runner / code-developer / paper-writer / reviewer / lead. Hook `hooks/vault-redflag-check.sh` is present but **not registered** (add to `.claude/settings.json` PostToolUse to activate).

## Environment

```bash
micromamba run -n graph_adversarial python research/branch-diffusion/prepare.py
micromamba run -n graph_adversarial python framework/pipeline.py --max-iterations 20
```

## Research findings (branch-diffusion)

N23 feature-only diffusion = inert (falsified) → N24 joint X+A diffusion = 98.43% PROTEINS GCN → **SGDI** (guided diffusion + hard Mahalanobis projection) = defense-invariant, beats N4 norm-cap under a distribution-aware Mahalanobis filter (PROTEINS +61.6pp, ENZYMES +9.5pp, p=0.002) at an undefended-ASR cost. See `research/branch-diffusion/REPORT_BRANCH_DIFFUSION_DISTSTEALTH.md` and wiki `projects/gzoo-katze/branches/branch-diffusion/`.
