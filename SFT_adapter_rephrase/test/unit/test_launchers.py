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


def _run(tmp_path: Path, *args: str, extra: str = "") -> subprocess.CompletedProcess:
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
    env = {**os.environ, "PY": sys.executable, "DRY_RUN": "1"}
    return subprocess.run(["bash", str(caller), "--style", "grammar", "--base-model", str(tmp_path / "base"), *args],
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
