# autoresearch-katze

Automated adversarial attack research framework for graph classification.
Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Quick Start

```bash
# 1. Prepare (once): download data, train victim models
python prepare.py

# 2. Run experiment: attack with current attack.py config
python run.py          # full 5-fold (~3 min)
python run.py --quick  # 1-fold fast test (~30s)

# 3. Check results
cat best.json
ls results/
```

## Structure

```
prepare.py      ← Fixed: dataset + victim model (DO NOT MODIFY)
attack.py       ← AI modifies this: attack logic + hyperparams
run.py          ← Fixed: runs attack.py, records results (DO NOT MODIFY)
program.md      ← Research instructions for AI agent
best.json       ← Tracks best ASR achieved
results/        ← All experiment JSONs
models_saved/   ← Trained victim models
data/           ← Downloaded datasets
```

## Auto-Research Loop

The AI agent reads `program.md`, modifies `attack.py`, runs experiments,
and iterates. Each cycle takes ~30s (quick) to ~3min (full).

## Environment

```bash
micromamba run -n graph_adversarial python prepare.py
micromamba run -n graph_adversarial python run.py
```
