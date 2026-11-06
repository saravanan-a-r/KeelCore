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
#   --style S          professional | elaborate | concise | friendly | grammar (default professional)
#   --input DIR        holds train/ and evals/; every *.jsonl under train/ is trained on and every
#                      *.jsonl under evals/ scores the checkpoints (both read recursively)
#   --base-model DIR   the pretrained checkpoint (default <repo>/runs/pretrain/checkpoint-90000)
#   --init-adapter DIR only when the run continues an adapter (INIT_RUNS_ROOT set)
#
# Behaviour shared by every run:
#   - Everything is checked before the GPU is touched: data folders, cleanup classes, the base
#     checkpoint, the adapter to continue, the manifest, the generated configuration.
#   - Re-running the same command resumes an interrupted run from its last checkpoint. A
#     finished run is never overwritten: move RUNS_ROOT/<style> away to train it again. A run
#     stopped before its first checkpoint is moved aside (<style>.aborted-<time>) and restarted.
#   - TensorBoard (port 6007) and GPU telemetry run beside the training and stop with it.
#   - Rows not trainable are counted, by reason, in the first lines of sft.log: a literal control
#     token in the text, a side over 2048 tokens, a train input that is also an eval input,
#     duplicate inputs, and identity pairs.

set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
PY="${PY:-/root/venv/bin/python}"
TENSORBOARD="${TENSORBOARD:-/root/tensorboard-venv/bin/tensorboard}"
TB_PORT="${TB_PORT:-6007}"
STYLES_RE='^(professional|elaborate|concise|friendly|grammar)$'
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
  "${PY}" "${SFT_DIR}/scripts/make_run_config.py" --recipe "${RECIPE}" --style "${STYLE}" \
    --runs-root "${RUNS_ROOT}" --out "${CFG}" || die "the run configuration is not valid"
}

start_side_processes() {
  PIDS=()
  trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

  if ss -ltn | grep -q ":${TB_PORT} "; then
    echo "TensorBoard: already serving on port ${TB_PORT}"
  else
    nice -n 19 "${TENSORBOARD}" --logdir "${RUNS_ROOT}" --host 127.0.0.1 --port "${TB_PORT}" \
      --samples_per_plugin scalars=100000,text=1000,histograms=200 --reload_interval 30 \
      --window_title "${TB_TITLE}" > /dev/null 2>&1 &
    PIDS+=($!)
    echo "TensorBoard: http://127.0.0.1:${TB_PORT}"
  fi

  nice -n 19 "${PY}" "${SFT_DIR}/scripts/gpu_telemetry.py" "${RUN_DIR}" "$$" \
    --every-steps "${EVERY_STEPS}" --step-zero "${STEP_ZERO}" &
  PIDS+=($!)
}

train_main() {
  require_settings
  cd "${REPO_ROOT}"                               # relative paths are taken from the repository root
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
  echo "== $(date '+%F %T') ${STYLE} ${RUN_LABEL}: done -> ${RUN_DIR}/final"
}
