#!/usr/bin/env bash
# Rank 16, phase 2: continue the finished rank-16 phase-1 adapter on the frozen phase-2 data set.
# Read REPHRASE_MODEL/rephrase_dataset/final/phase_2/README.md first: it says what the data is and why.
# Settings: recipe.yml next to this file.
#
# Usage (inside tmux; one style is about 25-40 minutes on the A100):
#   bash SFT_adapter_rephrase/rank_16/phase_2/train.sh                       # professional
#   bash SFT_adapter_rephrase/rank_16/phase_2/train.sh --style grammar
#   bash SFT_adapter_rephrase/rank_16/phase_2/train.sh --style all             # the five, one after another
#   bash SFT_adapter_rephrase/rank_16/phase_2/train.sh --style all --lanes 2   # the five, two at a time
#   DRY_RUN=1 bash SFT_adapter_rephrase/rank_16/phase_2/train.sh --style grammar   # checks only
#
# Options:
#   --style S          professional | elaborate | concise | friendly | grammar (default professional);
#                      `all` trains the five in that order, skips finished ones, resumes an interrupted one
#   --lanes N          with --style all: 2 = two styles at a time on the one GPU (default 1); see scripts/train_run.sh
#   --input DIR        the style's phase-2 folder, laid out like phase 1 (default
#                      /home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/phase_2/<style>).
#                      Every *.jsonl under DIR/train is trained on (phase_2_part_01, phase_2_part_02,
#                      phase_2_augment_tokens, phase_1_dataset); every *.jsonl under DIR/evals scores
#                      the checkpoints. DIR/manifest.json must match the files (SHA-256).
#   --init-adapter DIR the phase-1 adapter to continue (default
#                      SFT_adapter_rephrase/runs/adapters/<style>/final; its run must be completed)
#   --base-model DIR   the pretrained checkpoint (default <repo>/runs/pretrain/checkpoint-90000; it must
#                      be the base the phase-1 adapter was trained on -- the fingerprint is checked)
#
# Recipe (measured in the professional trial of 2026-11-05, see the README):
#   same LoRA shape as phase 1 (rank 16, all linears, rsLoRA alpha 8; rank cannot change when
#   continuing), 1 epoch, lr 5e-5 peak (phase 1 ended at 3e-5), 30 warmup steps, cosine to 10%,
#   effective batch 32, eval + checkpoint every 200 steps, best/ = lowest eval/loss on the mixed
#   eval set (phase-2 held-out + phase-1 evals), so a checkpoint that forgets phase 1 is not picked.
#
# Output: SFT_adapter_rephrase/runs/adapters_phase_2/<style>/ (layout as in phase 1; checkpoints
# every 200 steps). The phase-1 adapter in runs/adapters/<style>/ is only read, never changed.
#
# Rows not trainable are counted by reason in the first lines of sft.log; expect `in_eval_set` =
# the held-out phase-2 rows (they stay in the part files, the trainer drops them from train) and
# `identity` = phase-2 rows whose output equals the input.
# Resume, data rules and GPU plan: scripts/train_run.sh and scripts/make_run_config.py.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT="$(cd "${HERE}/../.." && pwd)"

RUN_LABEL="rank 16 phase 2"
RECIPE="${HERE}/recipe.yml"
DATA_ROOT="/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/phase_2"
RUNS_ROOT="${SFT}/runs/adapters_phase_2"
INIT_RUNS_ROOT="${SFT}/runs/adapters"
REQUIRE_MANIFEST=1
REQUIRED_TRAIN_FILES=(phase_1_dataset.jsonl)
CONFIG_DIR="${SFT}/logs/run_configs/rank_16/phase_2"
TB_TITLE="Thelivu SFT adapters, phase 2"
STEP_ZERO="the phase-1 adapter this run starts from"

source "${SFT}/scripts/train_run.sh"
train_main "$@"
