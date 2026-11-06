"""
GPU telemetry for one training run: samples nvidia-smi every 30 s into the
run's TensorBoard (gpu/*) against the training step, and writes the metric
guide to the TEXT tab. Read-only towards the training; exits with the
launcher process.

    python scripts/gpu_telemetry.py <run_dir> <launcher_pid> --every-steps 2000 \
        --step-zero "the untouched base: the adapter starts as an exact no-op"
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

GUIDE = """\
**How to read this run.** One step = one optimizer update on 32 examples. `train/*` and
`perf/*` are logged every 10 steps, `eval/*`, `generation/*` and `quality/*` every {every_steps}
steps (step 0 = {step_zero}).

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


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU telemetry for one training run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("launcher_pid", type=int)
    parser.add_argument("--every-steps", type=int, required=True, help="the run's evaluation interval")
    parser.add_argument("--step-zero", required=True, help="what the step-0 evaluation measures")
    args = parser.parse_args()
    run_dir, parent = args.run_dir, args.launcher_pid
    guide = GUIDE.replace("{every_steps}", str(args.every_steps)).replace("{step_zero}", args.step_zero)

    while not (run_dir / "tensorboard").is_dir():   # sft.py creates it once the data is read
        if not alive(parent):
            sys.exit(0)
        time.sleep(10)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard" / "gpu"))
    writer.add_text("guide", guide, 0)
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


if __name__ == "__main__":
    main()
