# KATZE Auto-Research Program

## Goal

Maximize Attack Success Rate (ASR) on graph classification via black-box node injection.

**Current best**: See `best.json`
**Target**: ASR > 85% on PROTEINS (GCN), with std < 5%

## How It Works

1. You modify `attack.py` (the ONLY file you should edit)
2. Run `python run.py --quick` for fast 1-fold test (~30s)
3. If promising, run `python run.py` for full 5-fold evaluation (~3min)
4. Check if ASR improved in `best.json`
5. Repeat

## What You Can Modify in attack.py

### Hyperparameters (CONFIG dict)
- `feat_scale`: Feature amplification factor. "auto" = sqrt(N). Try different values.
- `sigma`: ZOO smoothing parameter. Range [1e-5, 1e-1].
- `gen_lr`: Generator learning rate. Range [1e-5, 1e-2].
- `attack_epochs`: Steps per graph. More = better but slower.
- `grad_method`: "cge" (coordinate-wise, exact) or "rgf" (random, noisy).
- `loss_type`: "cw" (logit margin), "cosine" (embedding distance), "hybrid" (both).
- `edge_strategy`: "topk" (degree-based) or "spectral" (eigenvalue-based).
- `kappa`: CW loss margin. Try [-0.1, -0.001, -5].

### Attack Logic (functions)
- `construct_perturbed_graph()`: How injected nodes are added to the graph
- `estimate_gradient_cge()` / `estimate_gradient_rgf()`: Gradient estimation
- `batch_loss()`: Loss function computation
- `select_targets_spectral()` / `select_targets_topk()`: Edge selection
- `SimpleGenerator`: Feature generator architecture
- `run_attack()`: Main attack loop

### Ideas to Try
1. **Adaptive sigma**: Start large, anneal smaller as optimization converges
2. **Momentum in ZOO**: Add gradient momentum to reduce noise
3. **Multi-restart**: Try multiple random initializations, keep best
4. **Feature clipping**: Clip features to match dataset statistics
5. **Dynamic edge selection**: Re-select target nodes every K epochs
6. **Larger generator**: Add more layers or attention mechanisms
7. **Warm-start**: Initialize generator from a pre-trained model
8. **Loss scheduling**: Start with cosine loss, switch to CW later

## Rules

1. ONLY modify `attack.py`. Never touch `prepare.py` or `run.py`.
2. `run_attack()` must accept `(model, test_graphs, device)` and return float ASR.
3. Single fold must complete in < 5 minutes.
4. Commit after each improvement with descriptive message.
5. If an experiment fails or ASR drops, revert and try a different approach.

## Evaluation

- **Primary metric**: mean ASR across 5 folds (higher = better)
- **Secondary**: std of ASR (lower = more stable)
- **Constraint**: wall time < 5 min per fold

## Current Knowledge (from prior experiments)

### What works:
- feat_scale=sqrt(N) gives ~71% ASR (vs 16% without)
- CGE > RGF for low-dim features (3-dim: 6 queries vs 100)
- Spectral edge selection adds ~3-4pp over topk
- Hybrid loss (CW+cosine) adds ~6pp over CW alone
- More attack epochs helps (50 > 10)

### What doesn't work:
- Injected node interconnection (clique/chain) — no improvement
- feat_scale=N (too aggressive, unstable)
- sigma too small (< 1e-5) — gradient signal vanishes

### Theoretical insights (from Theorem 1-3):
- Mean pooling dilutes injection by 1/(N+1) → scaling compensates
- CGE has zero variance for dim <= 10 → always use for PROTEINS
- Connecting to low-degree nodes causes larger spectral shift
