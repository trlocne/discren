#!/usr/bin/env bash
# Train and evaluate one dataset's paired arms (full vs w/o LLM).
#
# Everything a run produces is namespaced by dataset, so Clothing and Sports can
# be in flight at the same time without either overwriting the other:
#
#   configs/<dataset>_full.yaml     configs/<dataset>_wo_llm.yaml
#   /checkpoints/<dataset>_full     /checkpoints/<dataset>_wo_llm
#   logs/<dataset>/full.log         logs/<dataset>/wo_llm.log
#   logs/<dataset>/eval_full.log    logs/<dataset>/eval_wo_llm.log
#
# The dataset name is passed to --dataset-name as well as being part of the
# config filename, because the config selects the hyperparameters while
# --dataset-name selects which data directory the loader reads. Getting those two
# out of step would train Sports hyperparameters on Clothing data and report it
# as Sports, so this script derives both from one argument.
#
# Usage:
#   bash run_dataset.sh Clothing            # train both arms
#   bash run_dataset.sh Clothing eval       # evaluate both arms
#   bash run_dataset.sh Sports
#   bash run_dataset.sh Sports eval
set -u
cd "$(dirname "$0")"

DATASET="${1:-}"
MODE="${2:-train}"

if [[ -z "$DATASET" ]]; then
  echo "usage: bash run_dataset.sh <Clothing|Sports> [train|eval]" >&2
  exit 1
fi

# Config names are lowercase; the loader's dataset key is capitalized.
LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')
ARMS=(full wo_llm)

for arm in "${ARMS[@]}"; do
  cfg="configs/${LOWER}_${arm}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "missing config: $cfg" >&2
    exit 1
  fi
done

LOGDIR="logs/${DATASET}"
mkdir -p "$LOGDIR"

if [[ "$MODE" == "eval" ]]; then
  pids=()
  for arm in "${ARMS[@]}"; do
    echo "[eval] ${DATASET}/${arm}"
    modal run eval_modal.py \
      --config-path "configs/${LOWER}_${arm}.yaml" \
      --dataset-name "$DATASET" \
      > "${LOGDIR}/eval_${arm}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo
  echo "Eval done. Test-set metrics:"
  for arm in "${ARMS[@]}"; do
    echo "  --- ${DATASET}/${arm} ---"
    grep -A12 "^|   K |" "${LOGDIR}/eval_${arm}.log" 2>/dev/null | tail -12 \
      || echo "  (no metric table found; check the log)"
  done
  echo
  echo "Compare both datasets with:  python compare_datasets.py"
  exit 0
fi

pids=()
for arm in "${ARMS[@]}"; do
  echo "[train] ${DATASET}/${arm}  (log: ${LOGDIR}/${arm}.log)"
  modal run main_modal.py \
    --config-path "configs/${LOWER}_${arm}.yaml" \
    --dataset-name "$DATASET" \
    > "${LOGDIR}/${arm}.log" 2>&1 &
  pids+=($!)
done

echo "Launched ${#pids[@]} runs. Tail with:  tail -f ${LOGDIR}/full.log"
wait "${pids[@]}"
echo "Training finished for ${DATASET}."
echo "Next:  bash run_dataset.sh ${DATASET} eval"
