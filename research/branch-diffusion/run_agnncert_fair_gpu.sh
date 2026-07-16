#!/bin/bash
# run_agnncert_fair_gpu.sh — GPU-gated launcher for the DECISIVE FAIR AGNNCert pipeline.
#
# Context: pid 64005 (eval_pg_floor_agnncert.py, T=300, mean-trained / DEGENERATE wrapper,
# clean acc 0.56) currently owns ~15.9GB of the RTX-5080 and will produce the UNFAIR
# OM2C->0 result. CPU retraining is impossible right now: torch's CPU autograd backward
# SEGFAULTS (exit 139) within the first ~250 steps under that job's contention (verified).
# So we GATE: poll GPU free memory and only start once the GPU has >= NEED MiB free for 2
# consecutive checks (i.e. pid 64005 finished or was stopped). NO silent CPU fallback.
#
# Once the GPU frees, run the FAIR pipeline on GPU:
#   1. RETRAIN the base GCN on the subgraph-divided distribution (AGNNCert Sec.4.1 recipe)
#      for seeds 42,7,1027 x 10 folds, T=300  -> clean acc should recover toward 0.75.
#   2. FAST VERDICT: fair eval on seed 42 (all 5 attacks, cap 10) -> early partial JSON.
#   3. FULL: fair eval on seeds 42,7,1027 (all 5 attacks, cap 10) -> decisive JSON.
set -euo pipefail
PYBIN=/home/devuser/micromamba/envs/graph_adversarial/bin/python3   # bypass mamba lock contention
cd /workspace/autoresearch-katze-n23/research/branch-diffusion || exit 9
export CUDA_VISIBLE_DEVICES=0 FLOOR_GPU=0 AGC_DEVICE=cuda AGC_GPU=0
T=300
NEED=8000          # MiB free required before launching
BUDGET_MIN=900     # give up after 15h of waiting (no silent fallback)
INTERVAL=60
waited=0; ok=0; free=0
echo "[gate $(date '+%F %T')] FAIR AGNNCert pipeline: waiting for GPU free >= ${NEED}MiB (pid 64005 owns it)"
while [ "$waited" -lt $((BUDGET_MIN*60)) ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$free" ] && free=0
  if [ "$free" -ge "$NEED" ]; then ok=$((ok+1)); else ok=0; fi
  if [ "$ok" -ge 2 ]; then break; fi
  sleep "$INTERVAL"; waited=$((waited+INTERVAL))
  if [ $((waited % 600)) -eq 0 ]; then echo "[gate $(date '+%F %T')] still waiting (${waited}s); free=${free}MiB"; fi
done
if [ "$ok" -lt 2 ]; then
  echo "[gate $(date '+%F %T')] GPU never freed >= ${NEED}MiB within ${BUDGET_MIN}min — EXIT 3 (no silent CPU fallback)"
  exit 3
fi
echo "[gate $(date '+%F %T')] GPU free=${free}MiB — starting FAIR pipeline (T=${T}) on GPU 0"

echo "==== STEP 1/3  RETRAIN fair AGNNCert victims (seeds 42,7,1027 x10 folds, T=${T}) ===="
"$PYBIN" -u train_agnncert_victims.py --seeds 42,7,1027 --folds 10 --T "$T" \
  --epochs 120 --val-every 40 --device cuda

echo "==== STEP 2/3  FAST VERDICT  fair eval seed 42 (all attacks, cap 10) ===="
"$PYBIN" -u eval_pg_floor_agnncert_fair.py --seeds 42 --cap 10 --T "$T" --device cuda

echo "==== STEP 3/3  FULL  fair eval seeds 42,7,1027 (all attacks, cap 10) ===="
"$PYBIN" -u eval_pg_floor_agnncert_fair.py --seeds 42,7,1027 --cap 10 --T "$T" --device cuda

echo "[done $(date '+%F %T')] FAIR AGNNCert pipeline complete."
