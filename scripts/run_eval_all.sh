#!/usr/bin/env bash
# ============================================================================
#  DISCREN — evaluate EVERY trained checkpoint and log each one properly.
#
#  One entry point that mirrors every training sweep:
#    - seed sweep        (run_seeds.sh)      -> logs/seeds/eval_*_seed*.log
#    - Clothing ablations(run_ablations.sh)  -> logs/Clothing/ablations/eval_*.log
#    - Sports ablations  (run_ablations.sh)  -> logs/Sports/ablations/eval_*.log
#    - d=64 sweep        (run_d64.sh)        -> logs/d64/eval_*.log
#
#  Every eval goes through eval_local.py: it runs on Modal GPUs but loads the
#  checkpoint from the LOCAL ./checkpoints tree (baked into the image), never
#  from the discren-checkpoints volume. It prints a [metrics-json] line, so
#  aggregate_seeds.py can reduce the seed sweep afterwards.
#
#  USAGE
#    bash run_eval_all.sh              # evaluate everything, sequentially
#    bash run_eval_all.sh -j           # evaluate everything, in parallel
#    bash run_eval_all.sh seeds        # only the seed sweep
#    bash run_eval_all.sh ablations    # only Clothing + Sports ablations
#    bash run_eval_all.sh d64          # only the d=64 sweep
#    bash run_eval_all.sh seeds -j     # combine a group with -j
# ============================================================================
set -u
cd "$(dirname "$0")"

SEEDS="${SEEDS:-42 7 13 2024 31337}"

# ── pretty output ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; C=""; N=""
fi
say()  { printf "%s\n" "${C}==>${N} ${B}$*${N}"; }
ok()   { printf "%s\n" "  ${G}✓${N} $*"; }
warn() { printf "%s\n" "  ${Y}!${N} $*"; }

# ── job runner ──────────────────────────────────────────────────────────────
# PARALLEL=1 launches every eval in the background and waits at the end of
# each group; otherwise evals run one at a time so a failure is easy to read.
PARALLEL=0
pids=()

run_eval() {
  # run_eval <config> <dataset> <seed> <logfile>
  local cfg="$1" ds="$2" seed="$3" log="$4"
  mkdir -p "$(dirname "$log")"
  local extra=()
  [[ "$seed" -ge 0 ]] && extra=(--seed "$seed")
  if [[ ! -f "$cfg" ]]; then
    warn "missing config: $cfg — skipped"
    return 0
  fi
  if [[ "$PARALLEL" == "1" ]]; then
    echo "[eval] $ds $(basename "$cfg") seed=${seed:-none}  ->  $log"
    modal run eval_local.py --config-path "$cfg" --dataset-name "$ds" \
      "${extra[@]}" > "$log" 2>&1 &
    pids+=($!)
  else
    printf "  %s %s %s seed=%s\n" "${C}·${N}" "$ds" "$(basename "$cfg")" "${seed:-none}"
    if modal run eval_local.py --config-path "$cfg" --dataset-name "$ds" \
        "${extra[@]}" > "$log" 2>&1; then
      local row; row=$(grep -E '^\|\s+20 ' "$log" | head -1 | tr -s ' ')
      ok "$(basename "$log" .log)  ${row:-(no K=20 row — see $log)}"
    else
      warn "FAILED: $log — last lines:"
      tail -5 "$log" | sed 's/^/      /'
    fi
  fi
}

wait_group() {
  [[ "$PARALLEL" != "1" || ${#pids[@]} -eq 0 ]] && return 0
  say "Waiting for ${#pids[@]} parallel evals…"
  wait "${pids[@]}"
  pids=()
}

# ── groups ──────────────────────────────────────────────────────────────────
eval_seeds() {
  say "Seed sweep (Clothing full vs wo_llm, seeds: $SEEDS)"
  for pair in "configs/clothing_full.yaml:full" "configs/clothing_wo_llm.yaml:wo_llm"; do
    local cfg="${pair%%:*}" name="${pair##*:}"
    for s in $SEEDS; do
      run_eval "$cfg" "Clothing" "$s" "logs/seeds/eval_${name}_seed${s}.log"
    done
  done
  wait_group
  ok "seed sweep evals -> logs/seeds/  (aggregate: python aggregate_seeds.py)"
}

eval_ablations() {
  local ds lower
  for ds in Clothing Sports; do
    lower=$(printf '%s' "$ds" | tr '[:upper:]' '[:lower:]')
    say "${ds} ablations"
    local entries=(
      "configs/${lower}_full.yaml:full"
      "configs/${lower}_wo_llm.yaml:wo_llm"
      "configs/ablations/${lower}_user_only.yaml:user_only"
      "configs/ablations/${lower}_item_only.yaml:item_only"
      "configs/ablations/${lower}_no_mae.yaml:no_mae"
      "configs/ablations/${lower}_shuffle_llm.yaml:shuffle_llm"
      "configs/ablations/${lower}_no_item_struct.yaml:no_item_struct"
      "configs/ablations/${lower}_no_modal.yaml:no_modal"
      "configs/ablations/${lower}_no_rca.yaml:no_rca"
      "configs/ablations/${lower}_no_gate.yaml:no_gate"
      "configs/ablations/${lower}_no_mixup.yaml:no_mixup"
    )
    for pair in "${entries[@]}"; do
      run_eval "${pair%%:*}" "$ds" -1 "logs/${ds}/ablations/eval_${pair##*:}.log"
    done
    wait_group
    ok "${ds} ablation evals -> logs/${ds}/ablations/"
  done
}

eval_d64() {
  say "d=64 sweep"
  local entries=(
    "configs/clothing_full_d64.yaml:Clothing:clothing_full"
    "configs/clothing_wo_llm_d64.yaml:Clothing:clothing_wo_llm"
    "configs/sports_full_d64.yaml:Sports:sports_full"
    "configs/sports_wo_llm_d64.yaml:Sports:sports_wo_llm"
  )
  for e in "${entries[@]}"; do
    local cfg ds name
    cfg=$(cut -d: -f1 <<<"$e"); ds=$(cut -d: -f2 <<<"$e"); name=$(cut -d: -f3 <<<"$e")
    run_eval "$cfg" "$ds" -1 "logs/d64/eval_${name}.log"
  done
  wait_group
  ok "d=64 evals -> logs/d64/"
}

# ── arg parsing ─────────────────────────────────────────────────────────────
# unset first: some environments export a GROUPS array (e.g. from a sourced
# profile), which would silently prepend garbage entries to the list.
unset GROUPS
GROUPS=()
for arg in "$@"; do
  case "$arg" in
    -j|--parallel) PARALLEL=1 ;;
    seeds|ablations|d64) GROUPS+=("$arg") ;;
    all) GROUPS=(seeds ablations d64) ;;
    *) echo "unknown argument: $arg" >&2; sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
done
[[ ${#GROUPS[@]} -eq 0 ]] && GROUPS=(seeds ablations d64)

for g in "${GROUPS[@]}"; do
  case "$g" in
    seeds)     eval_seeds ;;
    ablations) eval_ablations ;;
    d64)       eval_d64 ;;
  esac
  echo
done

say "All requested evaluations finished."
cat <<EOF

  ${B}Next${N}
    - Seed sweep stats:   python aggregate_seeds.py
    - Inspect one eval:   grep -A12 '^|   K |' logs/<group>/eval_<name>.log
EOF
