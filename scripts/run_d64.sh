#!/usr/bin/env bash
# d=64 sweep — the CONTROLLED comparison against MMHCL-convention baselines.
#
# WHY: the headline Table tab:main compares DiscReN at d=128 against baseline
# numbers transcribed from the MMHCL paper at d=64. That confounds embedding
# dimension with the architecture. Running both arms (full, wo_llm) at d=64
# removes the dimension confound: same dim, same preprocessing convention,
# only the split difference remains.
#
# Each run writes to /checkpoints/<name>_d64, so nothing collides with the
# d=128 runs.
#
# Usage:
#   bash run_d64.sh          # train all 4 runs in parallel
#   bash run_d64.sh eval     # evaluate all 4 checkpoints in parallel
set -u
cd "$(dirname "$0")"
mkdir -p logs/d64

RUNS=(
  "configs/clothing_full_d64.yaml:Clothing:clothing_full"
  "configs/clothing_wo_llm_d64.yaml:Clothing:clothing_wo_llm"
  "configs/sports_full_d64.yaml:Sports:sports_full"
  "configs/sports_wo_llm_d64.yaml:Sports:sports_wo_llm"
)

MODE="${1:-train}"

if [[ "$MODE" == "eval" ]]; then
  pids=()
  for entry in "${RUNS[@]}"; do
    cfg="${entry%%:*}"; rest="${entry#*:}"
    ds="${rest%%:*}"; name="${rest##*:}"
    echo "[eval] ${name} (d=64)"
    modal run eval_modal.py \
      --config-path "$cfg" \
      --dataset-name "$ds" \
      > "logs/d64/eval_${name}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "All d=64 evals done. Logs in logs/d64/"
  exit 0
fi

pids=()
for entry in "${RUNS[@]}"; do
  cfg="${entry%%:*}"; rest="${entry#*:}"
  ds="${rest%%:*}"; name="${rest##*:}"
  echo "[train] ${name} (d=64)"
  modal run main_modal.py \
    --config-path "$cfg" \
    --dataset-name "$ds" \
    > "logs/d64/train_${name}.log" 2>&1 &
  pids+=($!)
done
echo "Launched ${#pids[@]} d=64 training runs. Tail one with:"
echo "  tail -f logs/d64/train_sports_full.log"
wait "${pids[@]}"
echo "All d=64 training done. Evaluate with: bash run_d64.sh eval"
