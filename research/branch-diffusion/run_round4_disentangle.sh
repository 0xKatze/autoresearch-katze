#!/bin/bash
# run_round4_disentangle.sh — Round-4 backbone-disentangle pipeline (detached orchestrator).
# Sequence (GPU victim-gen, then CPU floor sweeps):
#   1. wait for the in-flight GIN 10-fold victim gen to finish (do NOT contend the GPU)
#   2. train ENZYMES/GCN 10-fold victims, seeds 42/7/1027 (GPU)
#   3. ENZYMES/GIN 10-fold floor sweep (CPU, validated harness)
#   4. ENZYMES/GCN 10-fold floor sweep (CPU, same harness, GCN backbone)
# PROTEINS 3-battery anchor already exists: results/paper_grade_floor_PROTEINS_full_20260617_182213.json
set -u
cd /workspace/autoresearch-katze-n23/research/branch-diffusion
RUN="micromamba run -n graph_adversarial python3"
echo "[round4] START $(date -u +%FT%TZ)"

echo "[round4] step1: wait for GIN 10-fold victim gen (seed1027 meta + process exit)"
while [ ! -f models_cont_10fold/meta_cont_gin_ENZYMES_seed1027.json ]; do sleep 20; done
while pgrep -f gen_victims_gin10_driver >/dev/null 2>&1; do sleep 15; done
echo "[round4] GIN victims ready $(date -u +%FT%TZ)"

echo "[round4] step2: train ENZYMES/GCN 10-fold victims (GPU)"
$RUN gen_victims_gcn10_driver.py --seeds 42,7,1027 --epochs 200 --n-folds 10 \
     --dataset ENZYMES --device cuda --models-dir models_cont_10fold
rc=$?; echo "[round4] GCN victim-gen exit=$rc $(date -u +%FT%TZ)"
if [ ! -f models_cont_10fold/meta_cont_gcn_ENZYMES_seed1027.json ]; then
  echo "[round4] FATAL: GCN victims missing, aborting floor sweeps"; exit 1
fi

echo "[round4] step3: ENZYMES/GIN 10-fold floor sweep (CPU)"
$RUN run_floor_gin10_driver.py
echo "[round4] GIN floor exit=$? $(date -u +%FT%TZ)"

echo "[round4] step4: ENZYMES/GCN 10-fold floor sweep (CPU)"
$RUN run_floor_gcn10_driver.py
echo "[round4] GCN floor exit=$? $(date -u +%FT%TZ)"

echo "[round4] DONE $(date -u +%FT%TZ)"
echo "[round4] results:"
ls -t results/floor_enzymes_gin_ENZYMES_full_*.json results/floor_enzymes_gcn_ENZYMES_full_*.json 2>/dev/null | head -4
