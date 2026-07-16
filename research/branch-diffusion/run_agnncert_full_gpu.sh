#!/bin/bash
# run_agnncert_full_gpu.sh — GPU-availability-gated launcher for the AGNNCert add-on FULL eval.
#
# The canonical 8-seed battery6 (eval_pg_floor_battery6.py --device cuda) owns ~15.7GB of the
# RTX-5080 right now. T=300 AGNNCert needs the GPU (CPU is intractable for the SGDI/FM CGE
# loops at T=300). This launcher POLLS GPU free memory and only launches the eval once the GPU
# has >= NEED MiB free for 2 consecutive checks (i.e. the canonical run finished) — coordinating
# to AVOID OOM. NO silent CPU fallback: if the GPU never frees within the budget it exits non-zero.
set -u
PYBIN=/home/devuser/micromamba/envs/graph_adversarial/bin/python3   # bypass mamba lock contention
cd /workspace/autoresearch-katze-n23/research/branch-diffusion || exit 9
NEED=8000          # MiB free required before launching (T=300 SGDI batches peak a few GB)
BUDGET_MIN=600     # give up after 10h of waiting (no silent fallback)
INTERVAL=60
waited=0; ok=0; free=0
echo "[gate $(date '+%F %T')] waiting for GPU free >= ${NEED}MiB (canonical battery6 owns it now)"
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
echo "[gate $(date '+%F %T')] GPU free=${free}MiB — launching AGNNCert FULL eval on GPU (T=300)"
exec env FLOOR_GPU=0 "$PYBIN" -u eval_pg_floor_agnncert.py \
  --seeds 42,7,1027 --cap 10 --T 300 --device cuda
