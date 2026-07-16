#!/usr/bin/env bash
# gap1_launch.sh — MAKE-OR-BREAK: train proper retrained RS-Pool (trimmed-mean robust
# readout) victims on GPU, then attack OM2C-nb10 + FM-full + FM-eb1-lownorm(control)
# against them on CPU. Does the PROPER retrained robust readout drop OM2C below baselines?
set -euo pipefail
cd /workspace/autoresearch-katze-n23/research/branch-diffusion
STAMP=$(date +%Y%m%d_%H%M%S)
TRAINLOG="logs/gap1_train_${STAMP}.log"
ATKLOG="logs/gap1_attack_${STAMP}.log"
mkdir -p logs

echo "[$(date)] GAP-1 step 1/2: TRAIN trimmed-mean(trim=0.2) victims (GPU, 200ep, seeds 42,7,1027)" | tee -a "$TRAINLOG"
micromamba run -n graph_adversarial python3 gap1_train_median.py \
    --seeds 42,7,1027 --epochs 200 --n-folds 10 --device cuda \
    --readout trimmed --trim 0.2 >> "$TRAINLOG" 2>&1
echo "[$(date)] step 1 done." | tee -a "$TRAINLOG"

echo "[$(date)] GAP-1 step 2/2: ATTACK robustpool victims (CPU)" | tee -a "$ATKLOG"
micromamba run -n graph_adversarial python3 gap1_attack_median.py \
    --seeds 42,7,1027 --cap 10 >> "$ATKLOG" 2>&1
echo "[$(date)] step 2 done. GAP-1 complete." | tee -a "$ATKLOG"
