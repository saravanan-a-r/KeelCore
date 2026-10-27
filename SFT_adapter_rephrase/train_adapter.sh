#!/usr/bin/env bash
# Train one rephrase adapter (LoRA on the frozen pretrained base) on the GPU.
#
# Usage (inside tmux -- a style takes many hours):
#   bash SFT_adapter_rephrase/train_adapter.sh
#   bash SFT_adapter_rephrase/train_adapter.sh --style concise --rank 32
#   bash SFT_adapter_rephrase/train_adapter.sh --style friendly --input /path/to/friendly \
#        --base-model /home/admin1/projects/EchoMeBetter/runs/pretrain/checkpoint-90000
#
# Options:
#   --rank 8|16|32|64     LoRA rank (default 16). alpha stays as in adapter_config.yml
#                         (rsLoRA: scaling = alpha / sqrt(rank)).
#   --style S             professional | elaborate | concise | friendly | grammar
#                         (default professional)
#   --input DIR           holds train/ and evals/ (default
#                         /home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/<style>).
#                         Every *.jsonl under train/ is trained on, every *.jsonl under evals/
#                         scores the checkpoints; both are read recursively.
#   --base-model DIR      the pretrained checkpoint (default
#                         /home/admin1/projects/EchoMeBetter/runs/pretrain/checkpoint-90000)
#
# Output: SFT_adapter_rephrase/runs/adapters/<style>/
#   checkpoints/step-NNNNNN/   every 2000 steps, all kept (adapter + optimizer state = resumable)
#   best/                      adapter with the lowest eval/loss so far
#   final/                     best adapter + metrics on the full eval set (end of training)
#   sft.log, metrics.jsonl     human log, and every scalar as JSON
#   tensorboard/               event files (served on http://127.0.0.1:6007 while training)
#   summary.json, config.json  run summary, resolved configuration
#
# Evaluation every 2000 steps: teacher-forced loss on every eval row, plus greedy generation
# on the first 500 (fixed) eval rows. TensorBoard's TEXT tab ("guide") explains each graph.
#
# Re-running the same command resumes an interrupted run from its last checkpoint. A
# finished run is never overwritten: move runs/adapters/<style> away to train it again.
#
# Data rules (data.py, unchanged): every cleanup class is accepted and nothing is dropped
# for length ratio or span loss. Rows still not trainable are counted, by reason, in the
# first lines of sft.log: a literal control token (<style:x>, <text_to_rewrite>) in the
# text, a side over 2048 tokens (the model's maximum length), a train row whose input is
# also an eval input, and duplicate inputs.
#
# GPU plan, measured on this A100 40 GB (bf16, rank 16 and 64, real rows of this data):
#   Every optimizer step is still exactly 32 examples -- the effective batch the adapter
#   recipe was tuned with -- so the training math is unchanged. Only how a step is fed to
#   the GPU changes (src/trainer.py, both tested to give the same weights):
#   - micro_batch_tokens 64000 / target_token_weight 3: a step is cut by padded cost,
#     rows x (longest source + 3 x longest target), not by a fixed 16 rows. A step of
#     short rows (most of the data) runs as one micro-batch of 32; a step of 2048-token
#     rows as several of ~7-10. A target token costs ~3x a source token in memory (its
#     logits span the 33.7k vocabulary). Worst case ~18 GB.
#   - checkpointing_min_tokens 6144: micro-batches of <= 6144 padded tokens skip gradient
#     checkpointing (it only recomputes activations; same gradient). Worst case 29.7 GB.
#   Measured: 62 examples/s vs 23.6 with 16 x 2 rows (2.6x). Larger budgets are not
#   faster: a 32-example step of typical rows is already one micro-batch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

PY="/root/venv/bin/python"
TENSORBOARD="/root/tensorboard-venv/bin/tensorboard"
TB_PORT=6007
DATA_ROOT="/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final"
RUNS_ROOT="${HERE}/runs/adapters"
CLEANUP_CLASSES="KEEP KEEP_CJK_TAIL KEEP_LONGER_SEGMENT REVIEW_TRUNCATED STRIP_NOTE TRUNCATE_AT_LEAK TRUNCATE_AT_TAG"

RANK=16
STYLE=professional
INPUT=""
BASE_MODEL="${ROOT}/runs/pretrain/checkpoint-90000"

die() { echo "error: $*" >&2; exit 2; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank) RANK="${2:?--rank needs a value}"; shift 2 ;;
    --style) STYLE="${2:?--style needs a value}"; shift 2 ;;
    --input) INPUT="${2:?--input needs a value}"; shift 2 ;;
    --base-model) BASE_MODEL="${2:?--base-model needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done

[[ "${RANK}" =~ ^(8|16|32|64)$ ]] || die "--rank must be 8, 16, 32 or 64 (got '${RANK}')"
[[ "${STYLE}" =~ ^(professional|elaborate|concise|friendly|grammar)$ ]] \
  || die "--style must be professional, elaborate, concise, friendly or grammar (got '${STYLE}')"
INPUT="${INPUT:-${DATA_ROOT}/${STYLE}}"
INPUT="$(realpath -e "${INPUT}" 2>/dev/null)" || die "--input: no such directory"
BASE_MODEL="$(realpath -e "${BASE_MODEL}" 2>/dev/null)" || die "--base-model: no such directory"

# -- preflight: everything is checked before the GPU is touched ------------------
TRAIN_DIR="${INPUT}/train"
EVAL_DIR="${INPUT}/evals"
count_jsonl() { find "$1" -type f -name '*.jsonl' -size +0 | wc -l; }
[[ -d "${TRAIN_DIR}" ]] || die "${TRAIN_DIR} does not exist (--input must hold train/ and evals/)"
[[ -d "${EVAL_DIR}" ]] || die "${EVAL_DIR} does not exist (--input must hold train/ and evals/)"
(( $(count_jsonl "${TRAIN_DIR}") > 0 )) || die "no non-empty *.jsonl under ${TRAIN_DIR}"
(( $(count_jsonl "${EVAL_DIR}") > 0 )) || die "no non-empty *.jsonl under ${EVAL_DIR}"
[[ -f "${BASE_MODEL}/model.safetensors" && -f "${BASE_MODEL}/config.json" ]] \
  || die "${BASE_MODEL} is not a checkpoint (model.safetensors / config.json missing)"

# A cleanup class outside the accepted list would be dropped (counted, but lost).
unknown="$(find "${TRAIN_DIR}" "${EVAL_DIR}" -type f -name '*.jsonl' -exec grep -ohE '"cleanup": *"[A-Za-z_]*"' {} + \
  | sed -E 's/.*"([A-Za-z_]*)"$/\1/' | sort -u | grep -vxF -f <(tr ' ' '\n' <<< "${CLEANUP_CLASSES}") || true)"
[[ -z "${unknown}" ]] || die "unknown cleanup class(es) in the data, they would be dropped: $(echo ${unknown})"

RUN_DIR="${RUNS_ROOT}/${STYLE}"
MODE=new
if [[ -d "${RUN_DIR}" ]]; then
  status="$(grep -oE '"status": *"[a-z_]+"' "${RUN_DIR}/summary.json" 2>/dev/null | sed -E 's/.*"([a-z_]+)"$/\1/' || true)"
  if [[ "${status}" == "completed" || "${status}" == "early_stopped" ]]; then
    die "${RUN_DIR} is a finished run (${status}); move it away to train ${STYLE} again"
  elif compgen -G "${RUN_DIR}/checkpoints/step-*/trainer_state.pt" > /dev/null; then
    MODE=resume
  else
    aside="${RUN_DIR}.aborted-$(date +%Y%m%d-%H%M%S)"
    echo "${RUN_DIR} has no resumable checkpoint (stopped before step 2000); moving it to ${aside}"
    mv "${RUN_DIR}" "${aside}"
  fi
fi

# -- the run's configuration: adapter_config.yml + these overrides -------------------
# (adapter_config.yml itself is never edited)
CFG="${HERE}/logs/train_adapter/${STYLE}_r${RANK}.yml"
mkdir -p "$(dirname "${CFG}")" "${RUNS_ROOT}"
RANK="${RANK}" CLEANUP_CLASSES="${CLEANUP_CLASSES}" RUNS_ROOT="${RUNS_ROOT}" \
SRC_CFG="${HERE}/adapter_config.yml" OUT_CFG="${CFG}" "${PY}" - <<'PYEOF'
import os
import yaml

env = os.environ
doc = yaml.safe_load(open(env["SRC_CFG"], encoding="utf-8"))
everything = 10**9

doc["adapter"]["rank"] = int(env["RANK"])

data = doc["data"]
data["file_glob"] = "**/*.jsonl"                 # recursive
data["allowed_cleanup"] = env["CLEANUP_CLASSES"].split()
data["require_span_preservation"] = False
data["min_length_ratio"] = 0.0
data["max_length_ratio"] = 1.0e9
data["train_examples"] = everything
data["eval_examples"] = everything
data["max_source_tokens"] = 2048                 # the model's maximum length
data["max_target_tokens"] = 2048
# The per-style length bounds are filters too; keep only each style's acceptance bar.
for block in (doc.get("styles") or {}).values():
    if isinstance(block, dict):
        block.pop("data", None)

doc["training"]["gradient_checkpointing"] = True
doc["checkpointing"] = {"every_steps": 2000, "keep_last": everything}
doc["logging"]["runs_dir"] = env["RUNS_ROOT"]
doc["logging"]["logs_dir"] = env["RUNS_ROOT"]    # sft.log + tensorboard/ inside the run dir

# The gpu profile is laid over the shared sections, so its values are the ones to set.
gpu = doc["devices"]["gpu"]["overrides"]
gpu["training"] = {
    "batch_size": 16,                            # effective batch 16 x 2 = 32; eval rows per batch = 2 x 16
    "gradient_accumulation_steps": 2,
    "micro_batch_tokens": 64000,                 # see the GPU plan above
    "target_token_weight": 3.0,
    "checkpointing_min_tokens": 6144,
}
gpu["evaluation"] = {
    "every_steps": 2000,
    "loss_examples": everything,                 # every eval row
    "early_stopping_patience": None,             # every checkpoint is kept and none stops the run
    "generation_examples": 500,
    "generation_batch_size": 64,
}

with open(env["OUT_CFG"], "w", encoding="utf-8") as handle:
    handle.write("# Generated by train_adapter.sh -- do not edit; adapter_config.yml is the source.\n")
    yaml.safe_dump(doc, handle, sort_keys=False)
PYEOF

# Validate it exactly as sft.py will load it.
"${PY}" - "${CFG}" "${STYLE}" <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[2]))
from src.config import load_sft_config

cfg = load_sft_config(sys.argv[1], "gpu", {"data": {"style": sys.argv[2]}})
t, e = cfg.training, cfg.evaluation
assert t.effective_batch_size == 32 and cfg.checkpointing.every_steps == e.every_steps == 2000
assert (t.micro_batch_tokens, t.target_token_weight, t.checkpointing_min_tokens) == (64000, 3.0, 6144)
print(f"config ok: rank {cfg.adapter.rank} (scaling {cfg.adapter.scaling:.4g}), {t.epochs:g} epochs, "
      f"batch {t.effective_batch_size} in micro-batches of <= {t.micro_batch_tokens} padded tokens "
      f"(target x {t.target_token_weight:g}), lr {t.learning_rate:g} {t.lr_schedule}, "
      f"grad checkpointing above {t.checkpointing_min_tokens} tokens, {cfg.resources.precision}, "
      f"eval + checkpoint every {e.every_steps} steps")
PYEOF

echo "${MODE}: ${STYLE} rank ${RANK}"
echo "  base   ${BASE_MODEL}"
echo "  train  ${TRAIN_DIR} ($(count_jsonl "${TRAIN_DIR}") files)"
echo "  evals  ${EVAL_DIR} ($(count_jsonl "${EVAL_DIR}") files)"
echo "  output ${RUN_DIR}"

# -- TensorBoard and GPU telemetry (both stop when this script ends) -----------------
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

if ss -ltn | grep -q ":${TB_PORT} "; then
  echo "TensorBoard: already serving on port ${TB_PORT}"
else
  nice -n 19 "${TENSORBOARD}" --logdir "${RUNS_ROOT}" --host 127.0.0.1 --port "${TB_PORT}" \
    --samples_per_plugin scalars=100000,text=1000,histograms=200 --reload_interval 30 \
    --window_title "EchoMeBetter SFT adapters" > /dev/null 2>&1 &
  PIDS+=($!)
  echo "TensorBoard: http://127.0.0.1:${TB_PORT}"
fi

# Samples nvidia-smi every 30 s into the run's TensorBoard (gpu/*), against the training
# step, and writes the metric guide to the TEXT tab. Read-only towards the training.
nice -n 19 "${PY}" - "${RUN_DIR}" "$$" <<'PYEOF' &
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

run_dir, parent = Path(sys.argv[1]), int(sys.argv[2])

GUIDE = """\
**How to read this run.** One step = one optimizer update on 32 examples. `train/*` and
`perf/*` are logged every 10 steps, `eval/*`, `generation/*` and `quality/*` every 2000
steps (step 0 = the untouched base: the adapter starts as an exact no-op).

| graph | what it is | what to look for |
|---|---|---|
| train/loss | mean NLL per target token on the training batches | falls fast, then slowly; noisy (32 examples) |
| eval/loss | the same on every eval row (never trained on); **selects best/** | should track train/loss; rising while train/loss falls = overfitting |
| eval/perplexity | exp(eval/loss) | same story, easier to read |
| eval/token_accuracy, train/token_accuracy | next token predicted exactly | rises; a gap to train = memorising |
| eval/eos_accuracy | the model stops exactly where the reference stops | should reach ~1.0: it knows when the rewrite is done |
| eval/ece, eval/confidence | calibration: confidence vs accuracy gap | low ECE = probabilities can be trusted |
| eval/loss_by_length/{short,medium,long} | eval loss by input length (<=32, <=128, longer tokens) | long inputs are hardest; all three should fall |
| eval/best_metric | best eval/loss so far | flat = recent checkpoints are not better |
| generation/sari | edit quality vs the reference (keep/add/delete) | rises |
| generation/chrf_reference | character overlap with the reference rewrite | rises |
| generation/chrf_input | overlap with the input: meaning kept | must stay high; a drop = content lost |
| generation/length_ratio, shorter_rate | output length / input length | concise < 1, elaborate > 1, others ~1 |
| generation/span_preservation | numbers, URLs, e-mails, @mentions, `code` kept | should be ~1.0; below = facts dropped |
| generation/copy_rate | output identical to the input | high = the adapter does nothing |
| generation/loop_rate, terminated_rate, empty_rate | repetition loops, finished before the cap, empty output | loop ~0, terminated ~1, empty 0 |
| generation/novel_word_rate, distinct_2 | new words vs the input, output diversity | style dependent; sudden jumps = degeneration |
| quality/acceptable_rate | share of outputs passing every bar at once | **the product number**: rises |
| reference/* | the same metrics on the dataset's own references | the ceiling to compare generation/* with |
| final/* | eval of best/ on all eval rows at the end | the numbers to report |
| train/learning_rate | warmup, then cosine decay | sanity check of the schedule |
| train/grad_norm, grad_norm_max, clipped_fraction | gradient size before clipping (at 1.0) | spikes are fine; a high clipped share for long = lr too high |
| train/skipped_steps | updates skipped on a non-finite gradient | must stay 0 |
| adapter/relative_delta/* | adapter change / base weight size, per projection and group | grows smoothly; one group far above the rest = it dominates |
| adapter/grad_norm/* | gradient norm per target group | which layers are learning |
| adapter/token_delta_norm/* | learned change of the frame-token embeddings | grows, then settles |
| progress/epoch, examples_seen | where the run is | -- |
| perf/seconds_per_step, examples_per_second, tokens_per_second | speed | flat; a drop = something else uses the GPU/CPU |
| perf/padding_efficiency | real / padded tokens | ~0.9 with length grouping |
| perf/micro_batches_per_step | GPU passes per 32-example step (token-budget split) | ~1; higher in windows of long rows |
| perf/eta_hours | time left at the current speed | -- |
| perf/gpu_max_allocated_gb, rss_gb | peak GPU memory (PyTorch), process RAM | GPU must stay below 40 GB |
| perf/eval_seconds | time one evaluation takes | -- |
| gpu/utilization_percent, gpu/memory_used_gb | nvidia-smi every 30 s | how busy the GPU really is |

Histograms: per-example generation/length_ratio, chrf_input, sari (the tails matter).
Text: `generation/samples` = input / reference / adapter output, at every evaluation.
"""


def latest_step(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            handle.seek(max(path.stat().st_size - 4096, 0))
            return int(json.loads(handle.read().splitlines()[-1])["step"])
    except (OSError, ValueError, IndexError, KeyError):
        return 0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


while not (run_dir / "tensorboard").is_dir():   # sft.py creates it once the data is read
    if not alive(parent):
        sys.exit(0)
    time.sleep(10)
writer = SummaryWriter(log_dir=str(run_dir / "tensorboard" / "gpu"))
writer.add_text("guide", GUIDE, 0)
writer.flush()
while alive(parent):
    try:
        line = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout.splitlines()[0]
        utilization, memory = (float(x) for x in line.split(","))
        step = latest_step(run_dir / "metrics.jsonl")
        writer.add_scalar("gpu/utilization_percent", utilization, step)
        writer.add_scalar("gpu/memory_used_gb", memory / 1024, step)
        writer.flush()
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    time.sleep(30)
writer.close()
PYEOF
PIDS+=($!)

# -- train ---------------------------------------------------------------------
args=(train --device gpu --config "${CFG}" --style "${STYLE}"
      --train-data "${TRAIN_DIR}" --eval-data "${EVAL_DIR}" --base-checkpoint "${BASE_MODEL}")
if [[ "${MODE}" == "resume" ]]; then args+=(--resume "${RUN_DIR}"); else args+=(--run-name "${STYLE}"); fi

echo "== $(date '+%F %T') ${STYLE}: ${MODE}"
code=0
"${PY}" "${HERE}/sft.py" "${args[@]}" || code=$?
if (( code != 0 )); then
  echo "== ${STYLE}: sft.py exited with ${code} (re-run the same command to resume)" >&2
  exit "${code}"
fi
echo "== $(date '+%F %T') ${STYLE}: done -> ${RUN_DIR}/final"
