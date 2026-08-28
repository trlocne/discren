#!/usr/bin/env bash
# Multi-seed sweep for the d=64 controlled comparison (full vs no-LLM).
#
# AFTERMATH of the first d=64 run: the gap between arms was within seed-noise
# (Clothing full 0.0897 vs wo_llm 0.0877; Sports full 0.1079 vs wo_llm
# 0.1046). To claim anything about the LLM stack at d=64, we need mean +- std
# over several seeds, like we did for d=128 in run_seeds.sh.
#
# New hyperparameters (tau & dropedge reduced, patience increased) are
# reflected in configs/*_d64.yaml; this script just repeats them 3 times.
#
# Usage:
#   bash run_d64_seeds.sh         # train all 4 configs x 3 seeds
#   bash run_d64_seeds.sh eval    # evaluate every seeded checkpoint
#   SEEDS="42 7" bash run_d64_seeds.sh    # custom list
set -u
cd "$(dirname "$0")"
mkdir -p logs/d64_seeds

SEEDS="${SEEDS:-42 7 13}"

RUNS=(
  "configs/clothing_full_d64.yaml:Clothing:full"
  "configs/clothing_wo_llm_d64.yaml:Clothing:wo_llm"
  "configs/sports_full_d64.yaml:Sports:full"
  "configs/sports_wo_llm_d64.yaml:Sports:wo_llm"
)

MODE="${1:-train}"

if [[ "$MODE" == "eval" ]]; then
  epids=()
  for entry in "${RUNS[@]}"; do
    cfg="${entry%%:*}"; rest="${entry#*:}"
    ds="${rest%%:*}"; name="${rest##*:}"
    for s in $SEEDS; do
      echo "[eval] ${ds}_${name} seed=${s}"
      modal run eval_modal.py \
        --config-path "$cfg" \
        --dataset-name "$ds" \
        --seed "$s" \
        > "logs/d64_seeds/eval_${ds}_${name}_seed${s}.log" 2>&1 &
      epids+=($!)
    done
  done
  echo "Launched ${#epids[@]} evals. Tail one with: tail -f logs/d64_seeds/eval_Clothing_full_seed42.log"
  wait "${epids[@]}"
  echo "All d=64 seeded evals done."
  exit 0
fi

pids=()
for entry in "${RUNS[@]}"; do
  cfg="${entry%%:*}"; rest="${entry#*:}"
  ds="${rest%%:*}"; name="${rest##*:}"
  for s in $SEEDS; do
    echo "[train] ${ds}_${name} seed=${s}  (log: logs/d64_seeds/train_${ds}_${name}_seed${s}.log)"
    modal run main_modal.py \
      --config-path "$cfg" \
      --dataset-name "$ds" \
      --seed "$s" \
      > "logs/d64_seeds/train_${ds}_${name}_seed${s}.log" 2>&1 &
    pids+=($!)
  done
done

echo "Launched ${#pids[@]} d=64 seeded training runs."
wait "${pids[@]}"
echo "All d=64 seeded training done. Evaluate with: bash run_d64_seeds.sh eval"
