#!/usr/bin/env bash
# ============================================================================
#  DISCREN — run ONLY the evaluations that are still missing.
#
#  run_eval_all.sh re-runs everything. This script first takes inventory:
#  for every (config, dataset, seed) job it checks
#
#     1. does the local checkpoint exist?           -> else SKIP (not trained)
#     2. does the log already have [metrics-json]?  -> else PENDING
#
#  and then launches only the PENDING jobs on Modal, N at a time.
#  A log that exists but has no [metrics-json] line counts as a failed run
#  and is re-queued (that is how logs/seeds/eval_wo_llm_seed2024.log looked).
#
#  USAGE
#    bash run_eval_missing.sh --list          # inventory only, run nothing
#    bash run_eval_missing.sh                 # run pending jobs, 4 in parallel
#    bash run_eval_missing.sh -j 8            # ... 8 in parallel
#    bash run_eval_missing.sh -j 8 ablations  # only the ablation group
#    bash run_eval_missing.sh --force d64     # re-run d64 even if already done
#
#  GROUPS: seeds | ablations | d64   (default: all three)
# ============================================================================
set -u
cd "$(dirname "$0")"

SEEDS="${SEEDS:-42 7 13 2024 31337}"
JOBS="${JOBS:-4}"          # how many `modal run` processes at once
LIST_ONLY=0
FORCE=0

# ── pretty output ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'
  D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; C=""; D=""; N=""
fi
say()  { printf "%s\n" "${C}==>${N} ${B}$*${N}"; }
ok()   { printf "%s\n" "  ${G}✓${N} $*"; }
warn() { printf "%s\n" "  ${Y}!${N} $*"; }
bad()  { printf "%s\n" "  ${R}✗${N} $*"; }

# ── job table ───────────────────────────────────────────────────────────────
# Each entry is: <config>|<dataset>|<seed>|<logfile>   (seed -1 = unseeded)
JOBS_ALL=()

# eval_local.py resolves seeded runs to checkpoints/seed_<stem>/seed<N>/.
_is_seed_run() { [[ "$1" == "clothing_full" || "$1" == "clothing_wo_llm" ]]; }

ckpt_path_for() {
  # ckpt_path_for <config> <dataset> <seed>  -> echoes expected local .pt path
  local cfg="$1" ds="$2" seed="$3" stem
  stem=$(basename "$cfg" .yaml)
  if [[ "$seed" -ge 0 ]] && _is_seed_run "$stem"; then
    echo "checkpoints/seed_${stem}/seed${seed}/${ds}_best.pt"
  else
    echo "checkpoints/${stem}/${ds}_best.pt"
  fi
}

add_job() {
  # add_job <config> <dataset> <seed> <logfile>
  JOBS_ALL+=("$1|$2|$3|$4")
}

collect_seeds() {
  local pair cfg name s
  for pair in "configs/clothing_full.yaml:full" "configs/clothing_wo_llm.yaml:wo_llm"; do
    cfg="${pair%%:*}"; name="${pair##*:}"
    for s in $SEEDS; do
      add_job "$cfg" "Clothing" "$s" "logs/seeds/eval_${name}_seed${s}.log"
    done
  done
}

collect_ablations() {
  local ds lower pair
  for ds in Clothing Sports; do
    lower=$(printf '%s' "$ds" | tr '[:upper:]' '[:lower:]')
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
      add_job "${pair%%:*}" "$ds" -1 "logs/${ds}/ablations/eval_${pair##*:}.log"
    done
  done
}

collect_d64() {
  local e cfg ds name
  local entries=(
    "configs/clothing_full_d64.yaml:Clothing:clothing_full"
    "configs/clothing_wo_llm_d64.yaml:Clothing:clothing_wo_llm"
    "configs/sports_full_d64.yaml:Sports:sports_full"
    "configs/sports_wo_llm_d64.yaml:Sports:sports_wo_llm"
  )
  for e in "${entries[@]}"; do
    cfg=$(cut -d: -f1 <<<"$e"); ds=$(cut -d: -f2 <<<"$e"); name=$(cut -d: -f3 <<<"$e")
    add_job "$cfg" "$ds" -1 "logs/d64/eval_${name}.log"
  done
}

# ── arg parsing ─────────────────────────────────────────────────────────────
unset GROUPS
GROUPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)     JOBS="${2:?-j needs a number}"; shift 2 ;;
    -j[0-9]*)      JOBS="${1#-j}"; shift ;;
    -l|--list|--dry-run) LIST_ONLY=1; shift ;;
    -f|--force)    FORCE=1; shift ;;
    seeds|ablations|d64) GROUPS+=("$1"); shift ;;
    all)           GROUPS=(seeds ablations d64); shift ;;
    -h|--help)     sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ ${#GROUPS[@]} -eq 0 ]] && GROUPS=(seeds ablations d64)

for g in "${GROUPS[@]}"; do
  case "$g" in
    seeds)     collect_seeds ;;
    ablations) collect_ablations ;;
    d64)       collect_d64 ;;
  esac
done

# ── inventory ───────────────────────────────────────────────────────────────
PENDING=()
n_done=0; n_fail=0; n_skip=0

say "Inventory (${#JOBS_ALL[@]} jobs across: ${GROUPS[*]})"
printf "  %s%-8s %-34s %-9s %-6s %s%s\n" "$D" "STATE" "LOG" "DATASET" "SEED" "REASON" "$N"

for job in "${JOBS_ALL[@]}"; do
  IFS='|' read -r cfg ds seed log <<<"$job"
  short=$(basename "$log" .log)
  ck=$(ckpt_path_for "$cfg" "$ds" "$seed")
  seed_disp=$([[ "$seed" -ge 0 ]] && echo "$seed" || echo "-")

  if [[ ! -f "$cfg" ]]; then
    printf "  %s%-8s%s %-34s %-9s %-6s %s\n" "$D" "SKIP" "$N" "$short" "$ds" "$seed_disp" "no config $cfg"
    ((n_skip++)); continue
  fi
  if [[ ! -f "$ck" ]]; then
    printf "  %s%-8s%s %-34s %-9s %-6s %s\n" "$D" "SKIP" "$N" "$short" "$ds" "$seed_disp" "no checkpoint $ck"
    ((n_skip++)); continue
  fi
  # A 0-byte .pt is a truncated download, not a trained model: torch.load dies
  # with "EOFError: Ran out of input". Catch it here instead of burning a GPU.
  if [[ ! -s "$ck" ]]; then
    printf "  %s%-8s%s %-34s %-9s %-6s %s\n" "$R" "SKIP" "$N" "$short" "$ds" "$seed_disp" "CORRUPT (0 bytes): $ck"
    ((n_skip++)); continue
  fi
  if [[ "$FORCE" != "1" && -f "$log" ]] && grep -qa '\[metrics-json\]' "$log" 2>/dev/null; then
    printf "  %s%-8s%s %-34s %-9s %-6s %s\n" "$G" "DONE" "$N" "$short" "$ds" "$seed_disp" ""
    ((n_done++)); continue
  fi
  reason="never run"
  [[ -f "$log" ]] && reason="log exists but no metrics — failed run"
  [[ "$FORCE" == "1" ]] && reason="forced"
  printf "  %s%-8s%s %-34s %-9s %-6s %s\n" "$Y" "PENDING" "$N" "$short" "$ds" "$seed_disp" "$reason"
  PENDING+=("$job")
  [[ -f "$log" ]] && ((n_fail++))
done

echo
say "Summary: ${G}${n_done} done${N}${B}, ${Y}${#PENDING[@]} pending${N}${B} (${n_fail} of them previously failed), ${D}${n_skip} skipped${N}"

if [[ ${#PENDING[@]} -eq 0 ]]; then
  ok "Nothing to run."
  exit 0
fi
if [[ "$LIST_ONLY" == "1" ]]; then
  echo
  warn "--list given: stopping before launch. Drop the flag to run these ${#PENDING[@]} jobs."
  exit 0
fi

# ── parallel launcher ───────────────────────────────────────────────────────
# A slot-limited fan-out: keep at most $JOBS `modal run` processes alive.
echo
say "Launching ${#PENDING[@]} evals on Modal, ${JOBS} at a time"
START=$(date +%s)
declare -A PID2JOB=()

reap_one() {
  # Block until any child exits, then report it.
  local pid rc
  wait -n -p pid 2>/dev/null; rc=$?
  if [[ -z "${pid:-}" ]]; then
    # bash < 5.1 has no `wait -n -p`; fall back to draining everything.
    for pid in "${!PID2JOB[@]}"; do
      wait "$pid"; report_job "$pid" "$?"
      unset 'PID2JOB[$pid]'
    done
    return
  fi
  report_job "$pid" "$rc"
  unset 'PID2JOB[$pid]'
}

report_job() {
  local pid="$1" rc="$2" job log short row
  job="${PID2JOB[$pid]:-}"
  [[ -z "$job" ]] && return
  IFS='|' read -r _ _ _ log <<<"$job"
  short=$(basename "$log" .log)
  # -a: Modal writes ANSI/control bytes, so grep would otherwise call these
  # logs binary and print "binary file matches" instead of the metrics row.
  if [[ "$rc" -eq 0 ]] && grep -qa '\[metrics-json\]' "$log" 2>/dev/null; then
    row=$(grep -aE '^\|\s+20 ' "$log" | head -1 | tr -s ' ')
    ok "$short  ${row:-(K=20 row not found)}"
  else
    bad "$short  (exit=$rc) — last lines of $log:"
    tail -6 "$log" 2>/dev/null | sed 's/^/        /'
  fi
}

for job in "${PENDING[@]}"; do
  while [[ ${#PID2JOB[@]} -ge $JOBS ]]; do reap_one; done

  IFS='|' read -r cfg ds seed log <<<"$job"
  mkdir -p "$(dirname "$log")"
  extra=(); [[ "$seed" -ge 0 ]] && extra=(--seed "$seed")

  printf "  %s→%s %-34s %s %s\n" "$C" "$N" "$(basename "$log" .log)" "$ds" "${D}$(basename "$cfg")${N}"
  modal run eval_local.py --config-path "$cfg" --dataset-name "$ds" \
    "${extra[@]}" > "$log" 2>&1 &
  PID2JOB[$!]="$job"
done

while [[ ${#PID2JOB[@]} -gt 0 ]]; do reap_one; done

# ── wrap up ─────────────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - START ))
echo
n_ok=0; n_bad=0
for job in "${PENDING[@]}"; do
  IFS='|' read -r _ _ _ log <<<"$job"
  if grep -qa '\[metrics-json\]' "$log" 2>/dev/null; then ((n_ok++)); else ((n_bad++)); fi
done

say "Finished in $((ELAPSED/60))m $((ELAPSED%60))s — ${G}${n_ok} ok${N}${B}, ${R}${n_bad} failed${N}"
if [[ $n_bad -gt 0 ]]; then
  warn "Re-run just the failures with: bash $(basename "$0") -j $JOBS  (they stay PENDING)"
fi
cat <<EOF

  ${B}Next${N}
    - Seed sweep stats:   python aggregate_seeds.py
    - Inspect one eval:   grep -A12 '^|   K |' logs/<group>/eval_<name>.log
EOF
exit $(( n_bad > 0 ))
