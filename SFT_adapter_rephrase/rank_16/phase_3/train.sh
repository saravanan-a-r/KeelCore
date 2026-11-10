#!/usr/bin/env bash
# Rank 16, phase 3 (professional, elaborate, concise): continue a finished rank-16 phase-2 adapter on its frozen phase-3
# data set. Read REPHRASE_MODEL/rephrase_dataset/final/phase_3/README.md first: it says what the
# data is and why. Settings: recipe.yml next to this file.
#
# Usage (inside tmux; about 15-20 minutes on the A100):
#   bash SFT_adapter_rephrase/rank_16/phase_3/train.sh                       # professional
#   bash SFT_adapter_rephrase/rank_16/phase_3/train.sh --init-adapter <phase-2 adapter>/final
#   bash SFT_adapter_rephrase/rank_16/phase_3/train.sh --style elaborate --init-adapter <phase-2 adapter>/final
#   DRY_RUN=1 bash SFT_adapter_rephrase/rank_16/phase_3/train.sh             # checks only
#
# Options:
#   --style S          professional | elaborate | concise (the styles with phase-3 data; another one stops at the data check)
#   --input DIR        the phase-3 folder (default
#                      /home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/phase_3/professional).
#                      Every *.jsonl under DIR/train is trained on (phase_3_exact_copy, phase_3_shorthand,
#                      phase_3_typos, phase_2_replay, phase_1_replay; elaborate also
#                      phase_3_numbers); every *.jsonl under DIR/evals scores
#                      the checkpoints. DIR/manifest.json must match the files (SHA-256).
#   --init-adapter DIR the phase-2 adapter to continue (default
#                      SFT_adapter_rephrase/runs/adapters_phase_2/professional/final; its run must be completed)
#   --base-model DIR   the pretrained checkpoint (default <repo>/runs/pretrain/checkpoint-90000; it must
#                      be the base the phase-2 adapter was trained on -- the fingerprint is checked)
#
# Recipe: same LoRA shape as phases 1-2 (rank 16, rsLoRA alpha 8), 1 epoch, lr 3e-5 peak, 20 warmup steps,
#   cosine to 10%, effective batch 32, eval + checkpoint every 50 steps, best/ = lowest eval/loss on the mixed
#   eval set (phase-3 held-out + the phase-2 run's two eval files), so a checkpoint that forgets is not picked.
#
# Output: SFT_adapter_rephrase/runs/rank_16/phase_3/<style>/ (layout as in phase 1). The phase-2 adapter
# is only read, never changed.
# Resume, data rules and GPU plan: scripts/train_run.sh and scripts/make_run_config.py.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT="$(cd "${HERE}/../.." && pwd)"

RUN_LABEL="rank 16 phase 3"
RECIPE="${HERE}/recipe.yml"
DATA_ROOT="/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/phase_3"
RUNS_ROOT="${SFT}/runs/rank_16/phase_3"
INIT_RUNS_ROOT="${SFT}/runs/adapters_phase_2"
REQUIRE_MANIFEST=1
REQUIRED_TRAIN_FILES=(phase_3_exact_copy.jsonl phase_2_replay.jsonl phase_1_replay.jsonl)
CONFIG_DIR="${SFT}/logs/run_configs/rank_16/phase_3"
TB_TITLE="KeelCore SFT adapters, phase 3"
STEP_ZERO="the phase-2 adapter this run starts from"

source "${SFT}/scripts/train_run.sh"
train_main "$@"
