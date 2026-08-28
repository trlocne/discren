#!/usr/bin/env bash
# Run all ablation configs IN PARALLEL. Each config writes to its own
# save_dir (set inside the YAML), so the runs never overwrite each other.
#
# Usage:
#   bash run_ablations.sh            # train Clothing ablations in parallel
#   bash run_ablations.sh Clothing   # same, explicit dataset
#   bash run_ablations.sh Sports     # train Sports ablations in parallel
#   bash run_ablations.sh Clothing eval
#   bash run_ablations.sh Sports eval
set -u
cd "$(dirname "$0")"
mkdir -p logs

DATASET="${1:-Clothing}"
MODE="${2:-train}"
LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')
LOGDIR="logs/${DATASET}/ablations"
mkdir -p "$LOGDIR"

# config file  ->  short run name (also the log file name)
# The five no_* arms below are the previously-unmeasured toggles from
# Table tab:ablation-protocol in the paper; each isolates one component.
CONFIGS=(
  "configs/${LOWER}_full.yaml:full"
  "configs/${LOWER}_wo_llm.yaml:wo_llm"
  "configs/ablations/${LOWER}_user_only.yaml:user_only"
  "configs/ablations/${LOWER}_item_only.yaml:item_only"
  "configs/ablations/${LOWER}_no_mae.yaml:no_mae"
  "configs/ablations/${LOWER}_shuffle_llm.yaml:shuffle_llm"
  "configs/ablations/${LOWER}_no_item_struct.yaml:no_item_struct"
  "configs/ablations/${LOWER}_no_modal.yaml:no_modal"
  "configs/ablations/${LOWER}_no_rca.yaml:no_rca"
  "configs/ablations/${LOWER}_no_gate.yaml:no_gate"
  "configs/ablations/${LOWER}_no_mixup.yaml:no_mixup"
)

# The low-reg probe was only defined for Clothing, so keep it as an optional
# extra for that dataset to avoid mixing a one-off probe into other runs.
if [[ "$LOWER" == "clothing" && -f "configs/config_lowreg.yaml" ]]; then
  CONFIGS+=("configs/config_lowreg.yaml:lowreg")
fi

for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"
  if [[ ! -f "$cfg" ]]; then
    echo "missing config: $cfg" >&2
    exit 1
  fi
done

if [[ "$MODE" == "eval" ]]; then
  for entry in "${CONFIGS[@]}"; do
    cfg="${entry%%:*}"; name="${entry##*:}"
    echo "[eval] $name  <-  $cfg"
    modal run eval_modal.py \
      --config-path "$cfg" \
      --dataset-name "$DATASET" \
      2>&1 | tee "${LOGDIR}/eval_${name}.log"
  done
  exit 0
fi

# ── Train: launch every config in the background, wait for all ──
pids=()
for entry in "${CONFIGS[@]}"; do
  cfg="${entry%%:*}"; name="${entry##*:}"
  echo "[train] launching $name  <-  $cfg   (log: ${LOGDIR}/${name}.log)"
  modal run main_modal.py \
    --config-path "$cfg" \
    --dataset-name "$DATASET" \
    > "${LOGDIR}/${name}.log" 2>&1 &
  pids+=("$!")
done

echo "Launched ${#pids[@]} parallel runs. Tail a log with:  tail -f ${LOGDIR}/full.log"
wait "${pids[@]}"
echo "All ${DATASET} ablation runs finished. Logs in ${LOGDIR}/"
