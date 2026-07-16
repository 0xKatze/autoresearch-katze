#!/bin/bash
# Detached launcher for the full OM2C+FM floor comparison (CPU only, gap2 GPU untouched).
cd /workspace/autoresearch-katze-n23/research/branch-diffusion || exit 1
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/floor_om2c_fm_full_${STAMP}.log"
export CUDA_VISIBLE_DEVICES=""
setsid micromamba run -n graph_adversarial python3 -u eval_pg_floor_om2c_fm.py \
    --seeds 42,7,1027 --cap 10 --bismooth-T 20 --bismooth-p 0.2 \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null
echo "PID=$PID"
echo "LOG=/workspace/autoresearch-katze-n23/research/branch-diffusion/$LOG"
