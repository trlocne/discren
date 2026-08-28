#!/usr/bin/env bash
# Evaluate ONLY the best-seed checkpoint for each d=64 config, on Modal.
#
# WHY: the 3-seed run logged several checkpoints per config under
# /checkpoints/<name>_d64/seed<N>. For the main paper's comparison against
# MMHCL (at matched d=64), we use the single best-seed per config rather than
# averaging, so this script picks exactly one seed per config.
#
# Best seeds (from logs/d64_seeds):
#   clothing_full : 42  (Recall@20=0.0896)
#   clothing_wo_llm : 42  (0.0858)
#   sports_full : 7  (0.1089)
#   sports_wo_llm : 7  (0.1065)
#
# Usage:
#   bash run_d64_best_eval.sh
set -u
cd "$(dirname "$0")"
mkdir -p logs/d64_best

# (config, dataset, name, best_seed)
RUNS=(
  "configs/clothing_full_d64.yaml:Clothing:clothing_full:42"
  "configs/clothing_wo_llm_d64.yaml:Clothing:clothing_wo_llm:42"
  "configs/sports_full_d64.yaml:Sports:sports_full:7"
  "configs/sports_wo_llm_d64.yaml:Sports:sports_wo_llm:7"
)

pids=()
for entry in "${RUNS[@]}"; do
  cfg="${entry%%:*}"; rest="${entry#*:}"
  ds="${rest%%:*}"; rest2="${rest#*:}"
  name="${rest2%%:*}"; seed="${rest2##*:}"
  log="logs/d64_best/eval_${name}_seed${seed}.log"
  echo "[eval] ${name} (dataset=${ds}, best seed=${seed})"
  modal run eval_modal.py \
    --config-path "$cfg" \
    --dataset-name "$ds" \
    --seed "$seed" \
    > "$log" 2>&1 &
  pids+=($!)
done

wait "${pids[@]}"
echo "All d=64 best-seed evals done. Results in logs/d64_best/"
