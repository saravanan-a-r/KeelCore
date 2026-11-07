#!/usr/bin/env bash
# Rank 32, phase 1: train one rephrase adapter (LoRA on the frozen pretrained base) from scratch
# on the style's phase-1 teacher data, on the GPU. Settings: recipe.yml next to this file.
#
# Usage (inside tmux -- a style takes many hours):
#   bash SFT_adapter_rephrase/rank_32/phase_1/train.sh --style all --lanes 2   # the five, two at a time on the one GPU (~1.5x faster)
#   bash SFT_adapter_rephrase/rank_32/phase_1/train.sh --style all             # the five, one after another
#   bash SFT_adapter_rephrase/rank_32/phase_1/train.sh
#   bash SFT_adapter_rephrase/rank_32/phase_1/train.sh --style concise
#   bash SFT_adapter_rephrase/rank_32/phase_1/train.sh --style friendly --input /path/to/friendly \
#        --base-model <repo>/runs/pretrain/checkpoint-90000
#   DRY_RUN=1 bash SFT_adapter_rephrase/rank_32/phase_1/train.sh --style grammar   # checks only
#
# Options:
#   --style S          professional | elaborate | concise | friendly | grammar (default professional);
#                      `all` trains the five in that order, skips finished ones, resumes an interrupted one
#   --lanes N          with --style all: 2 = two styles at a time on the one GPU (default 1); see scripts/train_run.sh
#   --input DIR        holds train/ and evals/ (default
#                      /home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/<style>;
#                      friendly: final/friendly_sft). Every *.jsonl under train/ is trained on,
#                      every *.jsonl under evals/ scores the checkpoints; both are read recursively.
#   --base-model DIR   the pretrained checkpoint (default <repo>/runs/pretrain/checkpoint-90000)
#
# Recipe: rank 32, all linears, rsLoRA alpha 8 (scaling 1.414), 2 epochs, lr 3e-4 cosine, effective
# batch 32, eval + checkpoint every 2000 steps (adapter_config.yml + recipe.yml).
#
# Output: SFT_adapter_rephrase/runs/rank_32/phase_1/<style>/
#   checkpoints/step-NNNNNN/   every 2000 steps, the last 4 kept (adapter + optimizer state = resumable)
#   best/                      adapter with the lowest eval/loss so far
#   final/                     best adapter + metrics on the full eval set (end of training)
#   sft.log, metrics.jsonl     human log, and every scalar as JSON
#   tensorboard/               event files (served on http://127.0.0.1:6007 while training)
#   summary.json, config.json  run summary, resolved configuration
#
# Evaluation every 2000 steps: teacher-forced loss on every eval row, plus greedy generation
# on the first 500 (fixed) eval rows. TensorBoard's TEXT tab ("guide") explains each graph.
# Resume, data rules and GPU plan: scripts/train_run.sh and scripts/make_run_config.py.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT="$(cd "${HERE}/../.." && pwd)"

RUN_LABEL="rank 32 phase 1"
RECIPE="${HERE}/recipe.yml"
DATA_ROOT="/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final"
declare -A DATA_DIR_OF=([friendly]=friendly_sft)
RUNS_ROOT="${SFT}/runs/rank_32/phase_1"
CONFIG_DIR="${SFT}/logs/run_configs/rank_32/phase_1"
TB_TITLE="KeelCore SFT adapters"
STEP_ZERO="the untouched base: the adapter starts as an exact no-op"

source "${SFT}/scripts/train_run.sh"
train_main "$@"
