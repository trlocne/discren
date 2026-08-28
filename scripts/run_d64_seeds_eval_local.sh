#!/usr/bin/env bash
# Evaluate every d=64 seeded checkpoint (downloaded locally) on Modal, using
# eval_local.py so the volume is never touched — checkpoints ride along in
# the container image from ./checkpoints/<config>_d64/seed<N>/.
#
# Prerequisite: bash tools/download_d64_seeds.sh   (populates ./checkpoints/)
#
# Usage:
#   bash run_d64_seeds_eval_local.sh            # eval all 4 configs x 3 seeds
#   SEEDS="42 13" bash run_d64_seeds_eval_local.sh
set -u
cd "$(dirname "$0")"
mkdir -p logs/d64_seeds

SEEDS="${SEEDS:-7 13 42}"

# (config_path, dataset_name)
RUNS=(
  "configs/clothing_full_d64.yaml:Clothing"
  "configs/clothing_wo_llm_d64.yaml:Clothing"
  "configs/sports_full_d64.yaml:Sports"
  "configs/sports_wo_llm_d64.yaml:Sports"
)

pids=()
for entry in "${RUNS[@]}"; do
  cfg="${entry%%:*}"
  ds="${entry##*:}"
  name="$(basename "$cfg" .yaml)"   # e.g. clothing_full_d64
  for s in $SEEDS; do
    log="logs/d64_seeds/eval_${name}_seed${s}.log"
    echo "[eval] ${name} seed=${s} (dataset=${ds})  -> ${log}"
    modal run eval_local.py \
      --config-path "$cfg" \
      --dataset-name "$ds" \
      --seed "$s" \
      > "$log" 2>&1 &
    pids+=($!)
  done
done

echo "Launched ${#pids[@]} local-checkpoint evals."
wait "${pids[@]}"
echo "All d=64 seeded evals done. Logs in logs/d64_seeds/eval_*_seed*.log"
echo "Extract best-per-seed with:"
echo '  grep -o "\"recall@20\": [0-9.]*" logs/d64_seeds/eval_*_seed*.log'
