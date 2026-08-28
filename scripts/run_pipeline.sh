#!/usr/bin/env bash
# ============================================================================
#  DISCREN — one entry point for the whole train -> eval -> figures pipeline.
#
#  Every step is idempotent and safe to re-run: nothing is destroyed, and a step
#  that has already produced its output is skipped unless you force it.
#
#  USAGE
#    ./run_pipeline.sh status          # where are the runs right now?
#    ./run_pipeline.sh train           # launch training (3 configs, parallel)
#    ./run_pipeline.sh wait            # block until all runs early-stop
#    ./run_pipeline.sh eval            # test-set metrics for every checkpoint
#    ./run_pipeline.sh export          # pull analysis artifacts off the volume
#    ./run_pipeline.sh figures         # regenerate all paper figures
#    ./run_pipeline.sh paper           # compile the PDF
#    ./run_pipeline.sh all             # wait -> eval -> export -> figures -> paper
#
#  TYPICAL SESSION
#    ./run_pipeline.sh status          # check progress whenever you like
#    ./run_pipeline.sh all             # once training is under way, run this and
#                                     # walk away; it waits, then does the rest
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs paper/figures/artifacts

DATASET="Clothing"

# config file : run name : checkpoint dir on the Modal volume
#
# config_lowreg.yaml is retained in the repo but no longer trained by default: it
# was a one-off probe testing whether rho=0.5 / emb_reg=0.005 had been tuned to
# compensate for the DropEdge bias. It had not -- full (0.0949) still beat lowreg
# (0.0940) after the fix -- so the knobs stay as they are. Re-add the line below
# to reproduce that probe.
CONFIGS=(
  "configs/clothing_full.yaml:full:/checkpoints/full"
  "configs/clothing_wo_llm.yaml:wo_llm:/checkpoints/wo_llm"
)

# ── pretty output ───────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; C=""; N=""
fi
say()  { printf "%s\n" "${C}==>${N} ${B}$*${N}"; }
ok()   { printf "%s\n" "  ${G}✓${N} $*"; }
warn() { printf "%s\n" "  ${Y}!${N} $*"; }
die()  { printf "%s\n" "  ${R}✗${N} $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

# Split a "config:name:dir" entry. Uses cut rather than nested ${x#*:} expansions,
# which silently tripped `set -u` when an intermediate variable was reused.
entry_cfg()  { printf '%s' "$1" | cut -d: -f1; }
entry_name() { printf '%s' "$1" | cut -d: -f2; }
entry_dir()  { printf '%s' "$1" | cut -d: -f3; }

# A run is finished when its log contains the early-stopping line.
is_done()    { grep -q "Stopping at epoch" "logs/$1.log" 2>/dev/null; }
is_running() { pgrep -f "main_modal.py.*$1" >/dev/null 2>&1; }

# ── status ──────────────────────────────────────────────────────────────────
cmd_status() {
  say "Training status"
  local n_done=0
  for e in "${CONFIGS[@]}"; do
    local name; name=$(entry_name "$e")
    local log="logs/${name}.log"
    if [[ ! -f "$log" ]]; then
      warn "$(printf '%-8s' "$name") not started"
      continue
    fi
    if is_done "$name"; then
      local line; line=$(grep "Stopping at epoch" "$log" | tail -1)
      ok "$(printf '%-8s' "$name") ${line#*] }"
      n_done=$((n_done + 1))
    else
      local ep best
      ep=$(grep -oE '^Epoch [0-9]+/' "$log" | tail -1 | tr -d 'Epoch /')
      best=$(grep -oE 'Recall@20=[0-9.]+ at epoch [0-9]+' "$log" | tail -1)
      warn "$(printf '%-8s' "$name") epoch ${ep:-?}  ${best:-no eval yet}"
    fi
  done
  echo
  [[ -f compare_runs.py ]] && python compare_runs.py 2>/dev/null | sed -n '/MATCHED/,$p'
  echo
  say "${n_done}/${#CONFIGS[@]} runs finished"
  [[ $n_done -eq ${#CONFIGS[@]} ]]
}

# ── train ───────────────────────────────────────────────────────────────────
cmd_train() {
  need modal
  say "Launching ${#CONFIGS[@]} training runs in parallel"
  for e in "${CONFIGS[@]}"; do
    local cfg name
    cfg=$(entry_cfg "$e"); name=$(entry_name "$e")
    if is_running "$name"; then warn "$name already running — skipped"; continue; fi
    if is_done "$name";    then warn "$name already finished — skipped"; continue; fi
    ok "$name  <-  $cfg"
    modal run main_modal.py --config-path "$cfg" --dataset-name "$DATASET" \
      > "logs/${name}.log" 2>&1 &
  done
  echo
  say "Launched. Watch with:  ./run_pipeline.sh status"
}

# ── wait ────────────────────────────────────────────────────────────────────
cmd_wait() {
  say "Waiting for all runs to early-stop (checking every 2 min)"
  local tick=0
  while true; do
    local n_done=0 n_alive=0
    for e in "${CONFIGS[@]}"; do
      local name; name=$(entry_name "$e")
      is_done "$name"    && n_done=$((n_done + 1))
      is_running "$name" && n_alive=$((n_alive + 1))
    done
    [[ $n_done -eq ${#CONFIGS[@]} ]] && { ok "all ${n_done} runs finished"; return 0; }
    if [[ $n_alive -eq 0 ]]; then
      warn "no run is alive but only ${n_done}/${#CONFIGS[@]} finished"
      warn "a run probably crashed — check the tail of logs/*.log"
      return 1
    fi
    tick=$((tick + 1))
    printf "\r  %s waiting… %d/%d done, %d alive (%d min)   " \
           "$Y!$N" "$n_done" "${#CONFIGS[@]}" "$n_alive" "$((tick * 2))"
    sleep 120
  done
}

# ── eval ────────────────────────────────────────────────────────────────────
cmd_eval() {
  need modal
  say "Evaluating every checkpoint on the test set"
  local failed=0
  for e in "${CONFIGS[@]}"; do
    local cfg name
    cfg=$(entry_cfg "$e"); name=$(entry_name "$e")
    if ! is_done "$name"; then warn "$name has not finished training — skipped"; continue; fi
    printf "  %s evaluating %s…\n" "${C}·${N}" "$name"
    if modal run eval_modal.py --config-path "$cfg" --dataset-name "$DATASET" \
         > "logs/eval_${name}.log" 2>&1; then
      # Pull the K=20 row out of the results table.
      local row; row=$(grep -E '^\|\s+20 ' "logs/eval_${name}.log" | head -1)
      ok "$(printf '%-8s' "$name") ${row:-see logs/eval_${name}.log}"
    else
      warn "$name evaluation FAILED — tail of its log:"
      tail -5 "logs/eval_${name}.log" | sed 's/^/      /'
      failed=$((failed + 1))
    fi
  done
  echo
  say "Test-set summary (K=20)"
  printf "  %-10s %10s %10s %10s %12s  %s\n" run recall ndcg precision coverage age
  for e in "${CONFIGS[@]}"; do
    local name; name=$(entry_name "$e")
    local log="logs/eval_${name}.log"
    [[ -f $log ]] || continue
    # Guard against reading a STALE eval log: if the checkpoint was retrained
    # after this log was written, the numbers below are from the previous model.
    local tag="current"
    if [[ -f "logs/${name}.log" && "logs/${name}.log" -nt "$log" ]]; then
      tag="${R}STALE — predates the current training run${N}"
    fi
    # table columns: K | Recall | Precision | NDCG | MRR | Coverage | ColdRecall
    awk -F'|' -v n="$name" -v t="$tag" '/^\|[[:space:]]+20[[:space:]]/ {
      gsub(/ /,"",$3); gsub(/ /,"",$4); gsub(/ /,"",$5); gsub(/ /,"",$7);
      printf "  %-10s %10s %10s %10s %12s  %s\n", n, $3, $5, $4, $7, t }' "$log"
  done
  echo
  warn "For reference, the numbers measured BEFORE the DropEdge bias fix were:"
  printf "      full 0.0969 / wo_llm 0.0938 (Recall@20). Those are superseded.\n"
  [[ $failed -eq 0 ]]
}

# ── export analysis artifacts ───────────────────────────────────────────────
# tag:config:save_dir -- the plot scripts load exactly the tags below (no
# underscore in "wollm"/"lowreg"), and each tag needs the SAME config that
# trained it so the export rebuilds the matching model/graphs.
EXPORT_ENTRIES=(
  "full:configs/clothing_full.yaml:/checkpoints/full"
  "wollm:configs/clothing_wo_llm.yaml:/checkpoints/wo_llm"
)

cmd_export() {
  need modal
  local script="paper/figures/export_analysis_modal.py"
  [[ -f "$script" ]] || die "missing $script"
  say "Exporting analysis artifacts from the trained checkpoints"
  : > logs/export.log
  for e in "${EXPORT_ENTRIES[@]}"; do
    local tag cfg
    tag=$(cut -d: -f1 <<<"$e"); cfg=$(cut -d: -f2 <<<"$e")
    printf "  %s exporting %s  (%s)…\n" "${C}·${N}" "$tag" "$cfg"
    if modal run "$script" --config-path "$cfg" --tag "$tag" \
         >> logs/export.log 2>&1; then
      ok "$tag exported"
    else
      warn "export for $tag failed — tail of logs/export.log:"
      tail -8 logs/export.log | sed 's/^/      /'
    fi
  done
  say "Downloading to paper/figures/artifacts/"
  for e in "${EXPORT_ENTRIES[@]}"; do
    local tag; tag=$(cut -d: -f1 <<<"$e")
    for ext in npz json; do
      modal volume get discren-checkpoints "/analysis/analysis_${tag}.${ext}" \
        "paper/figures/artifacts/" --force >> logs/export.log 2>&1 \
        && ok "analysis_${tag}.${ext}" \
        || warn "could not fetch analysis_${tag}.${ext}"
    done
  done
}

# ── figures ─────────────────────────────────────────────────────────────────
cmd_figures() {
  say "Regenerating figures"
  ( cd paper/figures && python make_all.py ) 2>&1 | sed 's/^/  /'
  python paper/figures/export_case_study.py 2>&1 | tail -3 | sed 's/^/  /'
  ok "figures written to paper/figures/"
}

# ── paper ───────────────────────────────────────────────────────────────────
cmd_paper() {
  command -v pdflatex >/dev/null 2>&1 || { warn "pdflatex not installed — skipping"; return 0; }
  say "Compiling the paper"
  ( cd paper \
    && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
    && bibtex main >/dev/null 2>&1 \
    && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
    && pdflatex -interaction=nonstopmode main.tex > /tmp/discren_tex.log 2>&1 )
  local errs over
  # `grep -c` already prints a count; the `|| echo 0` fallback used to append a
  # SECOND line when grep exited non-zero on no-match, producing "0\n0".
  errs=$(grep -cE '^!' /tmp/discren_tex.log 2>/dev/null) || errs=0
  over=$(grep -oE 'Overfull \\hbox \([0-9.]+' /tmp/discren_tex.log 2>/dev/null \
         | sed 's/.*(//' | awk '$1>20' | wc -l | tr -d ' ')
  if [[ -f paper/main.pdf ]]; then
    ok "paper/main.pdf — errors: ${errs}, overfull>20pt: ${over}"
  else
    warn "compilation failed; see /tmp/discren_tex.log"
  fi
}

# ── all ─────────────────────────────────────────────────────────────────────
cmd_all() {
  cmd_wait    || die "training did not complete cleanly — fix that first"
  echo; cmd_eval || warn "some evaluations failed; continuing anyway"
  echo; cmd_export
  echo; cmd_figures
  echo; cmd_paper
  echo
  say "Pipeline complete"
  cat <<EOF

  ${B}What to do next${N}
    1. Read the K=20 summary above and compare it against the OLD numbers:
         full 0.0969 / wo_llm 0.0938  (these were measured with the biased
         DropEdge operator and are now superseded)
    2. Check whether ${B}lowreg${N} beats ${B}full${N}. If it does, the rho=0.5 /
       emb_reg=0.005 tuning was compensating for the DropEdge bias rather than
       preventing real overfitting — that is a result worth reporting.
    3. Check gen_gap:  python compare_runs.py
       The pre-fix run reached ~8. If the fixed runs stay low, the original
       "overfitting" diagnosis was largely an artifact.
    4. Tell the assistant the numbers so the paper tables get updated:
         tab:main, tab:ablation, tab:components, tab:embed-geom, tab:full-metrics
       plus paper/bang_kien_truc_vi.tex and paper/AUDIT.md.

EOF
}

case "${1:-status}" in
  status)  cmd_status ;;
  train)   cmd_train ;;
  wait)    cmd_wait ;;
  eval)    cmd_eval ;;
  export)  cmd_export ;;
  figures) cmd_figures ;;
  paper)   cmd_paper ;;
  all)     cmd_all ;;
  *) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
