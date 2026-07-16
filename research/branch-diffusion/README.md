# autoresearch-katze

> Let an AI agent run adversarial attack experiments on graph neural networks autonomously overnight.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — but for **graph adversarial attacks** instead of LLM training.

## Concept

```
┌─────────────────────────────────────────────┐
│  AI Agent (Claude Code / Codex / etc.)      │
│                                             │
│  1. Read program.md (research instructions) │
│  2. Modify attack.py (attack logic)         │
│  3. Run: python run.py --quick              │
│  4. Check: ASR improved?                    │
│  5. If yes → commit. If no → revert.        │
│  6. Go to 1.                                │
│                                             │
│  ~12 experiments/hour, ~100 overnight       │
└─────────────────────────────────────────────┘
```

One dataset. One victim model. One metric (**Attack Success Rate**). One file to modify (`attack.py`).

## Setup

```bash
pip install torch torch-geometric scikit-learn numpy

# Prepare dataset + train victim models (one-time, ~5 min)
python prepare.py

# Run an experiment (~30s quick, ~3min full)
python run.py --quick   # 1-fold fast test
python run.py           # 5-fold evaluation
```

## How It Works

| File | Role | Modified by AI? |
|------|------|:-:|
| `prepare.py` | Download data, train victim GCN | No |
| `attack.py` | **All attack logic + hyperparams** | **Yes** |
| `run.py` | Load model, run attack, record results | No |
| `program.md` | Research instructions for AI agent | Humans refine |
| `best.json` | Tracks best ASR achieved | Auto-updated |

The AI agent modifies `attack.py` — changing hyperparameters, loss functions, gradient estimation methods, edge selection strategies, or even the entire attack algorithm. After each modification, `run.py` evaluates the attack and records whether ASR improved.

## Research Context

This framework attacks **graph classification** models (GCN, GIN, GraphSAGE) by **injecting adversarial nodes** — a practical threat model where attackers add fake nodes to a graph without modifying existing structure.

The key challenge is the **pooling dilution barrier**: global mean pooling suppresses injected node influence by 1/(N+1). Our baseline KATZE framework addresses this with:

- **√N feature scaling** (Theorem 1: compensate pooling dilution)
- **Coordinate-wise gradient estimation** (Theorem 2: exact gradients in 2d queries)
- **Spectral edge selection** (Theorem 3: maximize Laplacian eigenvalue perturbation)

Current best: **73.5% ASR** on PROTEINS (GCN), up from 15.8% baseline.

## Results

All experiments are saved to `results/` as JSON files with full config, per-fold ASR, and git diff of the changes that produced them.

```bash
# View experiment history
cat best.json | python -m json.tool

# List all experiments
ls results/ | sort
```

## License

MIT
