#!/usr/bin/env bash
# gap2_launch.sh — generate 5 NEW-seed PROTEINS/GCN victims (CPU, 200ep, matches the
# existing 42/7/1027 protocol), then run the UNCHANGED battery-6 floor on those seeds
# (GPU, device cuda, same params as the existing 3-seed run => mergeable to >=8 seeds).
# gap2_postprocess.py then merges new+existing JSONs -> seed-level FLOOR4 + Wilcoxon.
set -euo pipefail
cd /workspace/autoresearch-katze-n23/research/branch-diffusion
NEW_SEEDS="1,2,3,5,11"
STAMP=$(date +%Y%m%d_%H%M%S)
GENLOG="logs/gap2_genvictims_${STAMP}.log"
BATLOG="logs/gap2_battery6_newseeds_${STAMP}.log"
mkdir -p logs

echo "[$(date)] GAP-2 step 1/2: generate victims seeds=${NEW_SEEDS} (CPU, 200ep)" | tee -a "$GENLOG"
micromamba run -n graph_adversarial python3 gen_victims_pg.py \
    --seeds "${NEW_SEEDS}" --folds 10 --train-epochs 200 >> "$GENLOG" 2>&1
echo "[$(date)] step 1 done. victims:" | tee -a "$GENLOG"
ls -1 models_saved/meta_pg_seed{1,2,3,5,11}.json >> "$GENLOG" 2>&1 || true

echo "[$(date)] GAP-2 step 2/2: battery-6 floor on new seeds (GPU)" | tee -a "$BATLOG"
FLOOR_GPU=0 micromamba run -n graph_adversarial python3 eval_pg_floor_battery6.py \
    --seeds "${NEW_SEEDS}" --cap 10 --device cuda >> "$BATLOG" 2>&1
echo "[$(date)] step 2 done. GAP-2 raw battery6 (new seeds) complete." | tee -a "$BATLOG"
echo "[$(date)] Next: gap2_postprocess.py <new_json> results/floor_battery6_rspool_bismooth_PROTEINS_full_20260622_220259.json" | tee -a "$BATLOG"
