#!/usr/bin/env bash
# Multi-seed sweep for the full-vs-no-LLM comparison.
#
# WHY: the single-seed result is Recall@20 0.0949 (full) vs 0.0938 (no-LLM),
# a +1.2% gap. One run per configuration cannot tell that apart from seed
# noise, and the two checkpoints even early-stopped at different epochs
# (70 vs 104). Every claim about the LLM stack needs a mean +- std over
# several seeds instead.
#
# Each seeded run writes to <save_dir>/seed<N>, so nothing overwrites
# anything and the runs can go in parallel.
#
# Usage:
#   bash run_seeds.sh                 # train all configs x all seeds
#   bash run_seeds.sh eval            # evaluate every seeded checkpoint
#   SEEDS="42 7" bash run_seeds.sh    # custom seed list
set -u
cd "$(dirname "$0")"
mkdir -p logs/seeds

DATASET="Clothing"
SEEDS="${SEEDS:-42 7 13 2024 31337}"

# Only the two configurations the LLM claim rests on. lowreg is a probe, not
# part of the comparison, so it is deliberately excluded here.
CONFIGS=(
  "configs/clothing_full.yaml:full"
  "configs/clothing_wo_llm.yaml:wo_llm"
)

MODE="${1:-train}"

if [[ "$MODE" == "eval" ]]; then
  # Parallel, like training: eval only needs to load a checkpoint and score the
  # test split, so there is no reason to pay wall-clock time sequentially for
  # what training already paid for once in parallel.
  epids=()
  for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%:*}"; name="${entry##*:}"
    for s in $SEEDS; do
      echo "[eval] launching ${name} seed=${s}"
      modal run eval_modal.py \
        --config-path "$cfg" \
        --dataset-name "$DATASET" \
        --seed "$s" \
        > "logs/seeds/eval_${name}_seed${s}.log" 2>&1 &
      epids+=($!)
    done
  done
  echo "Launched ${#epids[@]} evals. Tail one with: tail -f logs/seeds/eval_full_seed42.log"
  wait "${epids[@]}"
  echo
  echo "All seeded evaluations done. Aggregate with:"
  echo "  python aggregate_seeds.py"
  exit 0
fi

pids=()
for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"; name="${entry##*:}"
  for s in $SEEDS; do
    echo "[train] ${name} seed=${s}  (log: logs/seeds/${name}_seed${s}.log)"
    modal run main_modal.py \
      --config-path "$cfg" \
      --dataset-name "$DATASET" \
      --seed "$s" \
      > "logs/seeds/${name}_seed${s}.log" 2>&1 &
    pids+=($!)
  done
done

echo "Launched ${#pids[@]} runs. Tail one with: tail -f logs/seeds/full_seed42.log"
wait "${pids[@]}"
echo "All seeded training runs finished."
echo "Next:  bash run_seeds.sh eval  &&  python aggregate_seeds.py"
