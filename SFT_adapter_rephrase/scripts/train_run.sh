# Shared launcher for one GPU training run of a rephrase adapter. Sourced by
# rank_<N>/phase_<K>/train.sh, which sets the run's settings and calls
# `train_main "$@"`; never run on its own.
#
# Settings the caller sets before `train_main`:
#   RUN_LABEL        shown in the log, e.g. "rank 16 phase 1"
#   RECIPE           the run's recipe.yml (rank, eval interval, training values)
#   DATA_ROOT        default --input is DATA_ROOT/<style> (or DATA_ROOT/${DATA_DIR_OF[<style>]})
#   RUNS_ROOT        output: RUNS_ROOT/<style>/
#   CONFIG_DIR       the generated run configs: CONFIG_DIR/<style>.yml
#   TB_TITLE         TensorBoard window title
#   STEP_ZERO        what the step-0 evaluation measures (TensorBoard guide)
# Optional:
#   DATA_DIR_OF      associative array style -> data folder name under DATA_ROOT
#   INIT_RUNS_ROOT   continue a finished adapter: the default --init-adapter is
#                    INIT_RUNS_ROOT/<style>/final (unset = train from scratch)
#   REQUIRE_MANIFEST 1 = the data folder must match its manifest.json (SHA-256)
#   REQUIRED_TRAIN_FILES  file names that must exist in <input>/train
#
# Options of every train.sh:
#   --style S          professional | elaborate | concise | friendly | grammar (default professional);
#                      `all` trains the five one after another, skipping finished ones (re-run to resume)
#   --lanes N          with --style all: 1 (default) one style at a time, 2 two styles at a time on the one
#                      GPU (see "Lanes" below)
#   --input DIR        holds train/ and evals/; every *.jsonl under train/ is trained on and every
#                      *.jsonl under evals/ scores the checkpoints (both read recursively)
#   --base-model DIR   the pretrained checkpoint (default <repo>/runs/pretrain/checkpoint-90000)
#   --init-adapter DIR only when the run continues an adapter (INIT_RUNS_ROOT set)
#
# Lanes (--style all --lanes 2, experimental): one run keeps the GPU ~45% busy (a step is ~15,000 small kernel
# launches, so it waits on the CPU); two side by side measured 1.57x the throughput, with the same steps, the same
# effective batch and the same data order. The styles still to train are split into two lanes that finish together
# (all five: professional + friendly | elaborate + concise + grammar; without grammar: professional + concise |
# friendly + elaborate). Each lane gets half of the CPU cores, a GPU memory cap of 18 GB (a lane that outgrows it
# fails alone instead of starving the other, which the shared vGPU does not survive) and smaller evaluation
# batches (scripts/make_run_config.py has the numbers). The GPU must be empty. Each lane's console output goes
# to RUNS_ROOT/lane_<N>.log; the runs' own logs are unchanged.
# Known limit: in testing the GPU hung (no progress, cause not found) in most attempts with two processes
# starting or resuming together. Ctrl-C saves a checkpoint in both lanes, but --lanes refuses a style that has a
# checkpoint: resume it alone with `--style <style>`, or with `--style all` without --lanes.
#
# Behaviour shared by every run:
#   - Everything is checked before the GPU is touched: data folders, cleanup classes, the base
#     checkpoint, the adapter to continue, the manifest, the generated configuration.
#   - Re-running the same command resumes an interrupted run from its last checkpoint. A
#     finished run is never overwritten: move RUNS_ROOT/<style> away to train it again. A run
#     stopped before its first checkpoint is moved aside (<style>.aborted-<time>) and restarted.
#   - TensorBoard (port 6007) and GPU telemetry run beside the training and stop with it. One
#     TensorBoard serves every rank and phase, one section each: runs show up as
#     rank_<N>/phase_<K>/<style> (see tensorboard_logdir_spec).
#   - Rows not trainable are counted, by reason, in the first lines of sft.log: a literal control
#     token in the text, a side over 2048 tokens, a train input that is also an eval input,
#     duplicate inputs, and identity pairs.

set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
PY="${PY:-/root/venv/bin/python}"
TENSORBOARD="${TENSORBOARD:-/root/tensorboard-venv/bin/tensorboard}"
TB_PORT="${TB_PORT:-6007}"
ALL_STYLES="professional elaborate concise friendly grammar"
LANE_MEMORY_GB=18                                        # 2 x 18 GB + the CUDA contexts of the 40 GB GPU
# Rough length of a style's phase-1 run (the rank-16 steps, in thousands, longest first): lanes are balanced by it.
STYLE_ORDER=(professional friendly elaborate concise grammar)
declare -A STYLE_WEIGHT=([professional]=148 [friendly]=146 [elaborate]=129 [concise]=127 [grammar]=93)
STYLES_RE="^(${ALL_STYLES// /|})\$"
CLEANUP_CLASSES="KEEP KEEP_CJK_TAIL KEEP_LONGER_SEGMENT REVIEW_TRUNCATED STRIP_NOTE TRUNCATE_AT_LEAK TRUNCATE_AT_TAG"

die() { echo "error: $*" >&2; exit 2; }

# The caller's header comment is its --help.
CALLER="$(realpath "$0")"
usage() { sed -n '2,/^[^#]/p' "${CALLER}" | sed '$d; s/^# \{0,1\}//'; }

count_jsonl() { find "$1" -type f -name '*.jsonl' -size +0 | wc -l; }

run_status() {
  grep -oE '"status": *"[a-z_]+"' "$1/summary.json" 2>/dev/null | sed -E 's/.*"([a-z_]+)"$/\1/' || true
}

require_settings() {
  local name
  for name in RUN_LABEL RECIPE DATA_ROOT RUNS_ROOT CONFIG_DIR TB_TITLE STEP_ZERO; do
    [[ -n "${!name:-}" ]] || die "train_run.sh: ${name} is not set by $0"
  done
  [[ -f "${RECIPE}" ]] || die "recipe ${RECIPE} does not exist"
}

parse_args() {
  STYLE=professional
  INPUT=""
  INIT_ADAPTER=""
  BASE_MODEL="${REPO_ROOT}/runs/pretrain/checkpoint-90000"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --style) STYLE="${2:?--style needs a value}"; shift 2 ;;
      --input) INPUT="${2:?--input needs a value}"; shift 2 ;;
      --base-model) BASE_MODEL="${2:?--base-model needs a value}"; shift 2 ;;
      --lanes) die "--lanes needs --style all" ;;
      --init-adapter)
        [[ -n "${INIT_RUNS_ROOT:-}" ]] || die "--init-adapter: this run trains a new adapter (see --help)"
        INIT_ADAPTER="${2:?--init-adapter needs a value}"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown option '$1' (see --help)" ;;
    esac
  done

  [[ "${STYLE}" =~ ${STYLES_RE} ]] \
    || die "--style must be professional, elaborate, concise, friendly or grammar (got '${STYLE}')"
  local data_dir="${STYLE}"
  if declare -p DATA_DIR_OF &> /dev/null && [[ -n "${DATA_DIR_OF[${STYLE}]:-}" ]]; then
    data_dir="${DATA_DIR_OF[${STYLE}]}"
  fi
  INPUT="${INPUT:-${DATA_ROOT}/${data_dir}}"
  INPUT="$(realpath -e "${INPUT}" 2>/dev/null)" || die "--input: no such directory"
  BASE_MODEL="$(realpath -e "${BASE_MODEL}" 2>/dev/null)" || die "--base-model: no such directory"
  if [[ -n "${INIT_RUNS_ROOT:-}" ]]; then
    INIT_ADAPTER="${INIT_ADAPTER:-${INIT_RUNS_ROOT}/${STYLE}/final}"
    INIT_ADAPTER="$(realpath -e "${INIT_ADAPTER}" 2>/dev/null)" || die "--init-adapter: no such directory"
  fi
}

preflight() {
  TRAIN_DIR="${INPUT}/train"
  EVAL_DIR="${INPUT}/evals"
  [[ -d "${TRAIN_DIR}" ]] || die "${TRAIN_DIR} does not exist (--input must hold train/ and evals/)"
  [[ -d "${EVAL_DIR}" ]] || die "${EVAL_DIR} does not exist (--input must hold train/ and evals/)"
  (( $(count_jsonl "${TRAIN_DIR}") > 0 )) || die "no non-empty *.jsonl under ${TRAIN_DIR}"
  (( $(count_jsonl "${EVAL_DIR}") > 0 )) || die "no non-empty *.jsonl under ${EVAL_DIR}"
  local name
  for name in "${REQUIRED_TRAIN_FILES[@]+"${REQUIRED_TRAIN_FILES[@]}"}"; do
    [[ -f "${TRAIN_DIR}/${name}" ]] || die "${TRAIN_DIR}/${name} missing (build the data set first)"
  done
  [[ -f "${BASE_MODEL}/model.safetensors" && -f "${BASE_MODEL}/config.json" ]] \
    || die "${BASE_MODEL} is not a checkpoint (model.safetensors / config.json missing)"

  if [[ -n "${INIT_ADAPTER}" ]]; then
    [[ -f "${INIT_ADAPTER}/adapter.safetensors" && -f "${INIT_ADAPTER}/adapter_config.json" ]] \
      || die "${INIT_ADAPTER} is not an adapter (adapter.safetensors / adapter_config.json missing)"
    local init_status
    init_status="$(run_status "$(dirname "${INIT_ADAPTER}")")"
    [[ "${init_status}" == "completed" || "${init_status}" == "early_stopped" ]] \
      || die "the run of ${INIT_ADAPTER} is not finished (status '${init_status}')"
  fi

  if [[ "${REQUIRE_MANIFEST:-0}" == 1 ]]; then
    "${PY}" "${SFT_DIR}/scripts/verify_manifest.py" "${INPUT}" \
      || die "the data in ${INPUT} does not match its manifest.json"
  fi

  # A cleanup class outside the accepted list would be dropped (counted, but lost).
  local unknown
  unknown="$(find "${TRAIN_DIR}" "${EVAL_DIR}" -type f -name '*.jsonl' -exec grep -ohE '"cleanup": *"[A-Za-z_]*"' {} + \
    | sed -E 's/.*"([A-Za-z_]*)"$/\1/' | sort -u | grep -vxF -f <(tr ' ' '\n' <<< "${CLEANUP_CLASSES}") || true)"
  [[ -z "${unknown}" ]] || die "unknown cleanup class(es) in the data, they would be dropped: $(echo ${unknown})"

  EVERY_STEPS="$("${PY}" "${SFT_DIR}/scripts/make_run_config.py" --recipe "${RECIPE}" --get every_steps)" \
    || die "cannot read ${RECIPE}"
  RUN_DIR="${RUNS_ROOT}/${STYLE}"
  MODE=new
  if [[ -d "${RUN_DIR}" ]]; then
    local status
    status="$(run_status "${RUN_DIR}")"
    if [[ "${status}" == "completed" || "${status}" == "early_stopped" ]]; then
      die "${RUN_DIR} is a finished run (${status}); move it away to train ${STYLE} (${RUN_LABEL}) again"
    elif compgen -G "${RUN_DIR}/checkpoints/step-*/trainer_state.pt" > /dev/null; then
      MODE=resume
    else
      local aside
      aside="${RUN_DIR}.aborted-$(date +%Y%m%d-%H%M%S)"
      echo "${RUN_DIR} has no resumable checkpoint (stopped before step ${EVERY_STEPS}); moving it to ${aside}"
      mv "${RUN_DIR}" "${aside}"
    fi
  fi

  CFG="${CONFIG_DIR}/${STYLE}.yml"
  mkdir -p "${RUNS_ROOT}"
  local lane_args=()
  [[ -z "${SFT_LANE:-}" ]] || lane_args=(--lane-cores "$(lane_cores "${SFT_LANE}")" --lane-memory-gb "${LANE_MEMORY_GB}")
  "${PY}" "${SFT_DIR}/scripts/make_run_config.py" --recipe "${RECIPE}" --style "${STYLE}" \
    --runs-root "${RUNS_ROOT}" --out "${CFG}" "${lane_args[@]}" || die "the run configuration is not valid"
}

# The --logdir_spec of the one TensorBoard: a section per rank and phase. The rank-16 runs predate
# the rank_<N> output folders and keep theirs; later runs live in runs/rank_<N>/phase_<K>.
tensorboard_logdir_spec() {
  local -A sections=([rank_16/phase_1]="${SFT_DIR}/runs/adapters" [rank_16/phase_2]="${SFT_DIR}/runs/adapters_phase_2")
  local dir
  for dir in "${SFT_DIR}"/runs/rank_*/phase_*; do
    [[ -d "${dir}" ]] && sections["${dir#"${SFT_DIR}/runs/"}"]="${dir}"
  done
  local name spec=()
  for name in $(printf '%s\n' "${!sections[@]}" | sort -V); do
    [[ -d "${sections[${name}]}" ]] && spec+=("${name}:${sections[${name}]}")
  done
  local IFS=,
  echo "${spec[*]}"
}

# The CPU cores of lane 1 or 2: the first or the second half of the machine's cores.
lane_cores() {
  local half=$(( $(nproc) / 2 ))
  (( half >= 1 )) || die "--lanes 2 needs at least 2 CPU cores"
  echo "$(( ($1 - 1) * half ))-$(( $1 * half - 1 ))"
}

# Lanes share the GPU through memory caps, which only work if nothing else is on it.
require_empty_gpu() {
  local busy
  busy="$("${NVIDIA_SMI:-nvidia-smi}" --query-compute-apps=pid,used_memory --format=csv,noheader 2> /dev/null || true)"
  [[ -z "${busy}" ]] || die "--lanes 2 needs an empty GPU, in use by (pid, memory): $(echo ${busy})"
}

start_tensorboard() {
  if ss -ltn | grep -q ":${TB_PORT} "; then
    echo "TensorBoard: already serving on port ${TB_PORT}"
    return
  fi
  nice -n 19 "${TENSORBOARD}" --logdir_spec "$(tensorboard_logdir_spec)" --host 127.0.0.1 --port "${TB_PORT}" \
    --samples_per_plugin scalars=100000,text=1000,histograms=200 --reload_interval 30 \
    --window_title "${TB_TITLE}" > /dev/null 2>&1 &
  PIDS+=($!)
  echo "TensorBoard: http://127.0.0.1:${TB_PORT}"
}

start_side_processes() {
  PIDS=()
  trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
  start_tensorboard
  nice -n 19 "${PY}" "${SFT_DIR}/scripts/gpu_telemetry.py" "${RUN_DIR}" "$$" \
    --every-steps "${EVERY_STEPS}" --step-zero "${STEP_ZERO}" &
  PIDS+=($!)
}

# One lane (or the whole run): the styles of STYLE_LIST in order, each in its own run of this launcher,
# until one fails. A finished style is skipped, an interrupted one resumes, so the same command continues
# after any stop. Exits with the failing run's code.
run_styles() {
  local styles="$1" style status code i
  shift
  local args=("$@")
  for style in ${styles}; do
    status="$(run_status "${RUNS_ROOT}/${style}")"
    if [[ "${status}" == "completed" || "${status}" == "early_stopped" ]]; then
      echo "== ${style} ${RUN_LABEL}: already finished (${status}), skipped"
      continue
    fi
    for i in "${!args[@]}"; do
      [[ "${args[i]}" == "--style" ]] && args[i+1]="${style}"
    done
    code=0
    bash "${CALLER}" "${args[@]}" || code=$?
    (( code == 0 )) || exit "${code}"
  done
}

# The styles still to train (not finished), longest first, split into two lanes whose summed weights are as equal
# as possible; the longest style is in lane 1. Sets LANE_1 and LANE_2 (space-separated, longest first).
split_lanes() {
  local pending=() style
  for style in "${STYLE_ORDER[@]}"; do
    case " $* " in *" ${style} "*) pending+=("${style}") ;; esac
  done
  local n=${#pending[@]} mask i best=-1 best_mask=0 load_1 load_2 worst
  for (( mask = 0; mask < (1 << n); mask += 2 )); do          # bit 0 stays 0: the longest style is in lane 1
    load_1=0; load_2=0
    for (( i = 0; i < n; i++ )); do
      if (( (mask >> i) & 1 )); then load_2=$(( load_2 + STYLE_WEIGHT[${pending[i]}] ))
      else load_1=$(( load_1 + STYLE_WEIGHT[${pending[i]}] )); fi
    done
    worst=$(( load_1 > load_2 ? load_1 : load_2 ))
    if (( best < 0 || worst < best )); then best=${worst}; best_mask=${mask}; fi
  done
  LANE_1=""; LANE_2=""
  for (( i = 0; i < n; i++ )); do
    if (( (best_mask >> i) & 1 )); then LANE_2+="${pending[i]} "; else LANE_1+="${pending[i]} "; fi
  done
  LANE_1="${LANE_1% }"; LANE_2="${LANE_2% }"
}

stop_lanes() {
  local pid
  for pid in "$@"; do kill -INT -- "-${pid}" 2> /dev/null || true; done
}

# Wait for a lane; a trapped signal interrupts `wait` while the lane is still saving its checkpoint.
wait_for() {
  local code=0
  while true; do
    wait "$1" || code=$?
    kill -0 "$1" 2> /dev/null || return "${code}"
    code=0
  done
}

# --style all [--lanes N]: see "Lanes" in the header.
train_all_styles() {
  local args=() lanes=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --lanes) lanes="${2:?--lanes needs a value}"; shift 2 ;;
      --input|--init-adapter) die "$1: not with --style all (every style has its own data folder)" ;;
      *) args+=("$1"); shift ;;
    esac
  done
  [[ "${lanes}" =~ ^[12]$ ]] || die "--lanes must be 1 or 2 (got '${lanes}')"
  PIDS=()
  trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
  [[ "${DRY_RUN:-0}" == 1 ]] || { mkdir -p "${RUNS_ROOT}"; start_tensorboard; }

  if (( lanes == 1 )); then
    run_styles "${ALL_STYLES}" "${args[@]}"
    echo "== all styles done (${RUN_LABEL})"
    return
  fi

  local pending=() style status
  for style in "${STYLE_ORDER[@]}"; do
    status="$(run_status "${RUNS_ROOT}/${style}")"
    if [[ "${status}" == "completed" || "${status}" == "early_stopped" ]]; then
      echo "== ${style} ${RUN_LABEL}: already finished (${status}), skipped"
    else
      pending+=("${style}")
    fi
  done
  if (( ${#pending[@]} < 2 )); then            # nothing to share the GPU with: one run, at full speed
    run_styles "${pending[*]}" "${args[@]}"
    echo "== all styles done (${RUN_LABEL})"
    return
  fi

  local interrupted=()
  for style in "${pending[@]}"; do
    ! compgen -G "${RUNS_ROOT}/${style}/checkpoints/step-*/trainer_state.pt" > /dev/null || interrupted+=("${style}")
  done
  (( ${#interrupted[@]} == 0 )) || die "interrupted run(s) to resume first, one at a time, with --style <style> or --style all (no --lanes): ${interrupted[*]}"
  [[ "${DRY_RUN:-0}" == 1 ]] || require_empty_gpu
  mkdir -p "${RUNS_ROOT}"
  split_lanes "${pending[@]}"
  local lane lane_pids=() lane_styles=("${LANE_1}" "${LANE_2}") failed=0 code
  # Job control gives each lane its own process group with default signal handling (a background job of a
  # script ignores Ctrl-C), so Ctrl-C or a kill of this command is passed on to the lanes as SIGINT: sft.py
  # then saves a resumable checkpoint, and the same command continues from it.
  set -m
  trap 'stop_lanes "${lane_pids[@]}"' INT TERM
  for lane in 1 2; do
    ( SFT_LANE="${lane}" run_styles "${lane_styles[lane-1]}" "${args[@]}" ) > "${RUNS_ROOT}/lane_${lane}.log" 2>&1 &
    lane_pids+=($!)
    PIDS+=($!)
    echo "lane ${lane}: ${lane_styles[lane-1]} (cores $(lane_cores "${lane}"), GPU memory cap ${LANE_MEMORY_GB} GB) -> ${RUNS_ROOT}/lane_${lane}.log"
  done
  for lane in 1 2; do
    code=0
    wait_for "${lane_pids[lane-1]}" || code=$?
    if (( code == 0 )); then
      echo "== lane ${lane} done (${lane_styles[lane-1]})"
    else
      echo "== lane ${lane} stopped with ${code}: see ${RUNS_ROOT}/lane_${lane}.log; run the same command again to continue" >&2
      failed=${code}
    fi
  done
  (( failed == 0 )) || exit "${failed}"
  echo "== all styles done (${RUN_LABEL}, 2 lanes)"
}

train_main() {
  require_settings
  cd "${REPO_ROOT}"                               # relative paths are taken from the repository root
  if [[ " $* " == *" --style all "* ]]; then
    train_all_styles "$@"
    return
  fi
  parse_args "$@"
  preflight

  echo "${MODE}: ${STYLE} ${RUN_LABEL}"
  echo "  base   ${BASE_MODEL}"
  [[ -z "${INIT_ADAPTER}" ]] || echo "  from   ${INIT_ADAPTER}"
  echo "  train  ${TRAIN_DIR} ($(count_jsonl "${TRAIN_DIR}") files)"
  echo "  evals  ${EVAL_DIR} ($(count_jsonl "${EVAL_DIR}") files)"
  echo "  output ${RUN_DIR}"
  [[ "${DRY_RUN:-0}" != 1 ]] || { echo "dry run: everything checked, nothing trained"; return 0; }

  start_side_processes

  local args=(train --device gpu --config "${CFG}" --style "${STYLE}"
              --train-data "${TRAIN_DIR}" --eval-data "${EVAL_DIR}" --base-checkpoint "${BASE_MODEL}")
  if [[ "${MODE}" == "resume" ]]; then
    args+=(--resume "${RUN_DIR}")                 # the checkpoint carries the adapter state
  else
    args+=(--run-name "${STYLE}")
    [[ -z "${INIT_ADAPTER}" ]] || args+=(--init-adapter "${INIT_ADAPTER}")
  fi

  echo "== $(date '+%F %T') ${STYLE} ${RUN_LABEL}: ${MODE}"
  local code=0
  "${PY}" "${SFT_DIR}/sft.py" "${args[@]}" || code=$?
  if (( code != 0 )); then
    echo "== ${STYLE} ${RUN_LABEL}: sft.py exited with ${code} (re-run the same command to resume)" >&2
    exit "${code}"
  fi
  # Ctrl-C makes sft.py save a resumable checkpoint and exit normally; that is a stop, not a finished run.
  if [[ "$(run_status "${RUN_DIR}")" == "interrupted" ]]; then
    echo "== ${STYLE} ${RUN_LABEL}: interrupted after a checkpoint was saved (re-run the same command to resume)" >&2
    exit 130
  fi
  echo "== $(date '+%F %T') ${STYLE} ${RUN_LABEL}: done -> ${RUN_DIR}/final"
}
