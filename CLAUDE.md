# autoresearch-katze

Automated adversarial attack research framework for graph classification.
Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
and [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw).

## Quick Start

```bash
# 1. Prepare (once): download data, train victim models
python prepare.py

# 2a. Manual mode: modify attack.py, run experiment
python run.py          # full 5-fold (~3 min)
python run.py --quick  # 1-fold fast test (~30s)

# 2b. Automated pipeline (recommended):
python pipeline.py --max-iterations 20 --target-asr 0.95
python pipeline.py --step              # single iteration
python pipeline.py --dry-run           # preview next hypothesis
python pipeline.py --analyze-only      # knowledge base summary

# 2c. Individual features:
python pipeline.py --charts            # generate all charts
python pipeline.py --review            # generate peer review
python pipeline.py --package           # package deliverables
```

## Structure

```
# Core (original)
prepare.py      ← Fixed: dataset + victim model (DO NOT MODIFY)
attack.py       ← AI modifies this: attack logic + hyperparams
run.py          ← Fixed: runs attack.py, records results (DO NOT MODIFY)
program.md      ← Research instructions for AI agent
best.json       ← Tracks best ASR achieved
results/        ← All experiment JSONs

# Pipeline (AutoResearchClaw-inspired)
pipeline.py     ← Main orchestrator: automated research loop
knowledge.py    ← Extract lessons from experiment history → knowledge.json
hypothesis.py   ← Generate ranked experiment hypotheses
codegen.py      ← Modify attack.py CONFIG with backup/restore
executor.py     ← Run experiments with self-healing (timeout/NaN/error → revert)
analyzer.py     ← 4-layer result verification + quality sentinel
decision.py     ← Decision engine: PROMOTE / REFINE / PIVOT / STOP

# New features (AutoResearchClaw-inspired)
charts.py       ← 📊 Auto-generated charts with error bars + 95% CI
evolution.py    ← 🧬 Self-learning lessons (JSONL + prompt overlays)
review.py       ← 📝 Multi-perspective peer review (4 reviewers)
deliverables.py ← 📦 Package all outputs into deliverables/

# Generated artifacts
charts/         ← 7 chart types (300 DPI, colorblind-safe)
evolution/      ← lessons.jsonl (append-only lesson store)
reviews.md      ← Structured peer review report
deliverables/   ← All outputs in one folder
knowledge.json  ← Structured knowledge base
backups/        ← attack.py backups before each edit
pipeline.log    ← Pipeline execution log
```

## AutoResearchClaw Feature Mapping

| AutoResearchClaw Feature | autoresearch-katze Implementation |
|--------------------------|-----------------------------------|
| 🔍 4-layer citation verification | `analyzer.py` — L1 range, L2 cross-fold, L3 historical, L4 alignment |
| 🧪 Experiment sandbox + structured JSON | `executor.py` — self-healing runner + structured results/*.json |
| 📊 Charts with error bars + CI | `charts.py` — 7 chart types with 95% CI error bars |
| 📝 Multi-agent peer review | `review.py` — 4 reviewers (attack, statistical, security, consistency) |
| 🧬 Self-learning evolution | `evolution.py` — JSONL lessons + prompt overlays for hypothesis gen |
| 📦 Deliverables packaging | `deliverables.py` — charts + review + code + summary + evolution |

## Pipeline Flow

```
knowledge.py → hypothesis.py → codegen.py → executor.py → analyzer.py → decision.py
     ↑              ↑                                          ↓              |
     │        evolution.py                                  charts.py         |
     │        (prompt overlay)                              review.py         |
     └────────────────────────── loop ──────────────────────────────────────┘
                                                         deliverables.py (final)
```

## Environment

```bash
micromamba run -n graph_adversarial python prepare.py
micromamba run -n graph_adversarial python pipeline.py --max-iterations 20
```
