"""
scripts/train_run.sh, the launcher every rank_<N>/phase_<K>/train.sh sources:
its checks run before anything is trained (DRY_RUN=1 stops right after them).
Each test writes a small caller script, as a train.sh is, with temporary paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import ROOT

LIBRARY = ROOT / "scripts" / "train_run.sh"
RECIPE = ROOT / "rank_16" / "phase_2" / "recipe.yml"


def _row(text: str, cleanup: str = "KEEP") -> str:
    return json.dumps({"row_index": 0, "input": text, "style": "grammar", "output": text + ".", "cleanup": cleanup}) + "\n"


def _data(root: Path, cleanup: str = "KEEP") -> Path:
    for part in ("train", "evals"):
        (root / part).mkdir(parents=True)
        (root / part / f"{part}.jsonl").write_text(_row(f"a {part} row", cleanup), encoding="utf-8")
    return root


def _manifest(root: Path) -> None:
    entry = lambda p: {"rows": 1, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}  # noqa: E731
    (root / "manifest.json").write_text(json.dumps({
        "train_files": {"train/train.jsonl": entry(root / "train" / "train.jsonl")},
        "eval_files": {"evals/evals.jsonl": entry(root / "evals" / "evals.jsonl")},
        "counts": {"phase_2_train": 1, "phase_1_replay": 0, "phase_2_heldout": 1, "phase_1_eval": 0},
    }), encoding="utf-8")


def _finished_adapter(runs: Path, status: str = "completed") -> Path:
    final = runs / "grammar" / "final"
    final.mkdir(parents=True)
    (final / "adapter.safetensors").write_bytes(b"")
    (final / "adapter_config.json").write_text("{}")
    (runs / "grammar" / "summary.json").write_text(json.dumps({"status": status}))
    return final


@pytest.fixture
def setup(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"")
    (base / "config.json").write_text("{}")
    _data(tmp_path / "data" / "grammar")
    return tmp_path


def _run(tmp_path: Path, *args: str, extra: str = "", style: str = "grammar",
         env_extra: dict | None = None) -> subprocess.CompletedProcess:
    caller = tmp_path / "train.sh"
    caller.write_text(f"""#!/usr/bin/env bash
# A test launcher.
RUN_LABEL="test run"
RECIPE="{RECIPE}"
DATA_ROOT="{tmp_path / 'data'}"
RUNS_ROOT="{tmp_path / 'runs'}"
CONFIG_DIR="{tmp_path / 'configs'}"
TB_TITLE="test"
STEP_ZERO="test"
{extra}
source "{LIBRARY}"
train_main "$@"
""")
    env = {**os.environ, "PY": sys.executable, "DRY_RUN": "1", **(env_extra or {})}
    return subprocess.run(["bash", str(caller), "--style", style, "--base-model", str(tmp_path / "base"), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_a_new_run_passes_every_check_and_writes_its_config(setup):
    result = _run(setup)
    assert result.returncode == 0, result.stderr
    assert "config ok: rank 16" in result.stdout and "new: grammar test run" in result.stdout
    assert "dry run" in result.stdout
    assert (setup / "configs" / "grammar.yml").is_file()


def test_a_finished_run_is_never_overwritten(setup):
    _finished_adapter(setup / "runs")
    result = _run(setup)
    assert result.returncode == 2 and "is a finished run (completed)" in result.stderr


def test_an_interrupted_run_is_resumed(setup):
    checkpoint = setup / "runs" / "grammar" / "checkpoints" / "step-000200"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"")
    result = _run(setup)
    assert result.returncode == 0, result.stderr
    assert "resume: grammar" in result.stdout


def test_a_run_without_a_checkpoint_is_moved_aside(setup):
    (setup / "runs" / "grammar").mkdir(parents=True)
    result = _run(setup)
    assert result.returncode == 0, result.stderr
    assert "stopped before step 200" in result.stdout
    assert list((setup / "runs").glob("grammar.aborted-*"))


@pytest.mark.parametrize("status, code", [("interrupted", 130), ("completed", 0)])
def test_a_run_stopped_by_ctrl_c_is_not_reported_as_done(setup, status, code):
    """sft.py exits normally after saving its checkpoint on Ctrl-C; `--style all` must not go on to the next style."""
    python = setup / "python"
    python.write_text(f"""#!/usr/bin/env bash
if [[ "$1" == *sft.py ]]; then
  mkdir -p "{setup}/runs/grammar"; echo '{{"status": "{status}"}}' > "{setup}/runs/grammar/summary.json"; exit 0
fi
[[ "$1" == *gpu_telemetry.py ]] && exit 0
exec "{sys.executable}" "$@"
""")
    python.chmod(0o755)
    result = _run(setup, env_extra={"PY": str(python), "DRY_RUN": "0", "TENSORBOARD": "true", "TB_PORT": "6999"})
    assert result.returncode == code, result.stderr
    assert ("interrupted after a checkpoint was saved" in result.stderr) is (code == 130)
    assert ("done ->" in result.stdout) is (code == 0)


def _running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:                                         # gone before or while reading (ENOENT, ESRCH)
        return False


def test_tensorboard_and_telemetry_stop_with_the_run(setup):
    pid_file = setup / "tensorboard.pid"
    tensorboard = setup / "tensorboard"
    tensorboard.write_text(f"#!/usr/bin/env bash\necho $$ > {pid_file}\nexec sleep 600\n")
    tensorboard.chmod(0o755)
    python = setup / "python"
    python.write_text(f"""#!/usr/bin/env bash
if [[ "$1" == *sft.py ]]; then
  sleep 1                                                   # long enough for the stub to start
  mkdir -p "{setup}/runs/grammar"; echo '{{"status": "completed"}}' > "{setup}/runs/grammar/summary.json"; exit 0
fi
[[ "$1" == *gpu_telemetry.py ]] && exit 0
exec "{sys.executable}" "$@"
""")
    python.chmod(0o755)
    result = _run(setup, env_extra={"PY": str(python), "DRY_RUN": "0", "TENSORBOARD": str(tensorboard), "TB_PORT": "6998"})
    assert result.returncode == 0, result.stderr
    pid = int(pid_file.read_text())
    for _ in range(50):                                     # the kill is asynchronous
        if not _running(pid):
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        pytest.fail("TensorBoard outlived the run")


def test_an_unknown_cleanup_class_is_refused(setup):
    (setup / "data" / "grammar" / "evals" / "evals.jsonl").write_text(_row("x", cleanup="DROP_ME"))
    result = _run(setup)
    assert result.returncode == 2 and "DROP_ME" in result.stderr


def test_missing_folders_and_checkpoints_are_refused(setup):
    assert "no such directory" in _run(setup, "--input", str(setup / "nowhere")).stderr
    (setup / "base" / "model.safetensors").unlink()
    assert "is not a checkpoint" in _run(setup).stderr


def test_relative_paths_are_taken_from_the_repository_root(setup):
    relative = os.path.relpath(setup / "data" / "grammar", ROOT.parent)
    result = _run(setup, "--input", relative)
    assert result.returncode == 0, result.stderr


def test_the_data_folder_mapping_is_used(setup):
    (setup / "data" / "grammar").rename(setup / "data" / "grammar_sft")
    result = _run(setup, extra="declare -A DATA_DIR_OF=([grammar]=grammar_sft)")
    assert result.returncode == 0, result.stderr
    assert "grammar_sft/train" in result.stdout


def test_init_adapter_only_for_a_continuing_run(setup):
    result = _run(setup, "--init-adapter", str(setup))
    assert result.returncode == 2 and "this run trains a new adapter" in result.stderr


def test_a_continuing_run_needs_a_finished_adapter(setup):
    extra = f'INIT_RUNS_ROOT="{setup / "phase_1"}"'
    final = _finished_adapter(setup / "phase_1", status="running")
    result = _run(setup, extra=extra)
    assert result.returncode == 2 and "is not finished (status 'running')" in result.stderr
    (final.parent / "summary.json").write_text(json.dumps({"status": "completed"}))
    result = _run(setup, extra=extra)
    assert result.returncode == 0, result.stderr
    assert f"from   {final}" in result.stdout


def test_the_manifest_must_match_the_data(setup):
    extra = "REQUIRE_MANIFEST=1"
    data = setup / "data" / "grammar"
    assert "does not match its manifest.json" in _run(setup, extra=extra).stderr  # no manifest
    _manifest(data)
    result = _run(setup, extra=extra)
    assert result.returncode == 0, result.stderr
    assert "data ok:" in result.stdout
    (data / "train" / "train.jsonl").write_text(_row("changed"))
    result = _run(setup, extra=extra)
    assert result.returncode == 2 and "sha256 differs" in result.stderr


def test_required_train_files_are_checked(setup):
    result = _run(setup, extra="REQUIRED_TRAIN_FILES=(phase_1_dataset.jsonl)")
    assert result.returncode == 2 and "phase_1_dataset.jsonl missing" in result.stderr


@pytest.fixture
def all_styles(setup):
    for style in ("professional", "elaborate", "concise", "friendly"):
        _data(setup / "data" / style)
    return setup


def test_style_all_trains_every_style_in_order(all_styles):
    result = _run(all_styles, style="all")
    assert result.returncode == 0, result.stderr
    started = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("new: ")]
    assert started == ["professional", "elaborate", "concise", "friendly", "grammar"]
    assert "all styles done (test run)" in result.stdout


def test_style_all_skips_finished_styles_and_resumes_interrupted_ones(all_styles):
    _finished_adapter(all_styles / "runs")
    checkpoint = all_styles / "runs" / "concise" / "checkpoints" / "step-000200"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"")
    result = _run(all_styles, style="all")
    assert result.returncode == 0, result.stderr
    assert "== grammar test run: already finished (completed), skipped" in result.stdout
    assert "resume: concise test run" in result.stdout and "new: friendly test run" in result.stdout


def test_style_all_stops_at_the_first_failing_style(all_styles):
    (all_styles / "data" / "elaborate" / "train" / "train.jsonl").unlink()
    result = _run(all_styles, style="all")
    assert result.returncode == 2 and "no non-empty *.jsonl" in result.stderr
    assert "new: professional" in result.stdout
    assert "concise" not in result.stdout and "all styles done" not in result.stdout


def test_style_all_refuses_a_single_data_folder(all_styles):
    result = _run(all_styles, "--input", str(all_styles / "data" / "grammar"), style="all")
    assert result.returncode == 2 and "not with --style all" in result.stderr


def _lanes(setup, *args, nvidia_smi: str = "true"):
    """`--style all --lanes 2` as a dry run; the NVIDIA_SMI stand-in prints what is on the GPU."""
    stub = setup / "nvidia-smi"
    stub.write_text(f"#!/usr/bin/env bash\n{nvidia_smi}\n")
    stub.chmod(0o755)
    extra = f'export NVIDIA_SMI="{stub}"'
    return _run(setup, "--lanes", "2", *args, extra=extra, style="all")


def test_two_lanes_split_the_styles_and_limit_each_lane(all_styles):
    result = _lanes(all_styles)
    assert result.returncode == 0, result.stderr
    assert "all styles done (test run, 2 lanes)" in result.stdout
    lane_1 = (all_styles / "runs" / "lane_1.log").read_text()
    lane_2 = (all_styles / "runs" / "lane_2.log").read_text()
    started = lambda log: [line.split()[1] for line in log.splitlines() if line.startswith("new: ")]  # noqa: E731
    assert started(lane_1) == ["professional", "friendly"] and started(lane_2) == ["elaborate", "concise", "grammar"]
    half = (os.cpu_count() or 2) // 2
    assert f"lane: cores 0-{half - 1}, GPU memory capped at 18 GB" in lane_1
    assert f"lane: cores {half}-{2 * half - 1}, GPU memory capped at 18 GB" in lane_2


def test_lanes_skip_finished_styles_and_rebalance_the_rest(all_styles):
    _finished_adapter(all_styles / "runs")                  # grammar
    result = _lanes(all_styles)
    assert result.returncode == 0, result.stderr
    assert "== grammar test run: already finished (completed), skipped" in result.stdout
    started = lambda log: [line.split()[1] for line in log.read_text().splitlines() if line.startswith("new: ")]  # noqa: E731
    assert started(all_styles / "runs" / "lane_1.log") == ["professional", "concise"]
    assert started(all_styles / "runs" / "lane_2.log") == ["friendly", "elaborate"]


def test_lanes_do_not_resume_an_interrupted_style(all_styles):
    checkpoint = all_styles / "runs" / "concise" / "checkpoints" / "step-000200"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"")
    result = _lanes(all_styles)
    assert result.returncode == 2 and "interrupted run(s) to resume first" in result.stderr and "concise" in result.stderr
    assert not (all_styles / "runs" / "lane_1.log").exists()


def test_the_last_style_runs_alone_at_full_speed(all_styles):
    for style in ("professional", "friendly", "elaborate", "concise"):
        (all_styles / "runs" / style).mkdir(parents=True)
        (all_styles / "runs" / style / "summary.json").write_text(json.dumps({"status": "completed"}))
    result = _lanes(all_styles)
    assert result.returncode == 0, result.stderr
    assert "new: grammar test run" in result.stdout and "lane:" not in result.stdout
    assert not (all_styles / "runs" / "lane_1.log").exists()


@pytest.mark.parametrize("pending, lanes", [
    ("professional friendly elaborate concise grammar", ("professional friendly", "elaborate concise grammar")),
    ("professional friendly elaborate concise", ("professional concise", "friendly elaborate")),
    ("grammar professional friendly", ("professional", "friendly grammar")),
    ("elaborate grammar", ("elaborate", "grammar")),
])
def test_lanes_are_balanced_by_the_length_of_the_runs(pending, lanes):
    script = f'source "{LIBRARY}"; split_lanes {pending}; echo "${{LANE_1}}|${{LANE_2}}"'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert tuple(result.stdout.strip().split("|")) == lanes


def test_a_failing_lane_leaves_the_other_lane_running(all_styles):
    (all_styles / "data" / "elaborate" / "train" / "train.jsonl").unlink()
    result = _lanes(all_styles)
    assert result.returncode == 2
    assert "lane 2 stopped with 2" in result.stderr and "== lane 1 done (professional friendly)" in result.stdout
    lane_2 = (all_styles / "runs" / "lane_2.log").read_text()
    assert "no non-empty *.jsonl" in lane_2 and "concise" not in lane_2


@pytest.mark.parametrize("on_the_gpu, refused", [("echo '4242, 31000 MiB'", True), ("true", False)])
def test_lanes_need_an_empty_gpu(tmp_path, on_the_gpu, refused):
    stub = tmp_path / "nvidia-smi"
    stub.write_text(f"#!/usr/bin/env bash\n{on_the_gpu}\n")
    stub.chmod(0o755)
    result = subprocess.run(["bash", "-c", f'source "{LIBRARY}"; NVIDIA_SMI="{stub}"; require_empty_gpu'],
                            capture_output=True, text=True, timeout=60)
    assert (result.returncode == 2) is refused
    assert refused == ("needs an empty GPU" in result.stderr and "4242" in result.stderr)


def test_lanes_option_is_checked(setup):
    assert "--lanes must be 1 or 2" in _run(setup, "--lanes", "3", style="all").stderr
    assert "--lanes needs --style all" in _run(setup, "--lanes", "2").stderr


def test_one_tensorboard_section_per_rank_and_phase(tmp_path):
    runs = tmp_path / "runs"
    for name in ("adapters", "adapters_phase_2", "rank_32/phase_1", "rank_8/phase_2", "p2trial_A"):
        (runs / name).mkdir(parents=True)
    (runs / "rank_32" / "phase_2").touch()                # a file is not a section
    script = f'source "{LIBRARY}"; SFT_DIR="{tmp_path}"; tensorboard_logdir_spec'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split(",") == [
        f"rank_8/phase_2:{runs}/rank_8/phase_2",
        f"rank_16/phase_1:{runs}/adapters",
        f"rank_16/phase_2:{runs}/adapters_phase_2",
        f"rank_32/phase_1:{runs}/rank_32/phase_1",
    ]


def test_a_launcher_must_set_its_settings(setup):
    caller = setup / "bad.sh"
    caller.write_text(f'source "{LIBRARY}"\ntrain_main\n')
    result = subprocess.run(["bash", str(caller)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "RUN_LABEL is not set" in result.stderr


@pytest.mark.parametrize("launcher", sorted(ROOT.glob("rank_*/phase_*/train.sh")),
                         ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_every_launcher_prints_its_help(launcher):
    result = subprocess.run(["bash", str(launcher), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "Usage" in result.stdout and "--style" in result.stdout
