"""
The run launchers: scripts/make_run_config.py (adapter_config.yml + a recipe)
and every rank_<N>/phase_<K>/recipe.yml shipped next to a train.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))

import make_run_config as mrc  # noqa: E402

RECIPES = sorted(ROOT.glob("rank_*/phase_*/recipe.yml"))


def _write(tmp_path: Path, recipe: dict) -> Path:
    path = tmp_path / "recipe.yml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return path


def test_every_launcher_has_a_recipe_and_every_recipe_a_launcher():
    launchers = sorted(ROOT.glob("rank_*/phase_*/train.sh"))
    assert launchers, "no rank_<N>/phase_<K>/train.sh found"
    assert [p.parent for p in launchers] == [p.parent for p in RECIPES]


@pytest.mark.parametrize("recipe_path", RECIPES, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_shipped_recipes_produce_a_valid_config(recipe_path, tmp_path):
    recipe = mrc.read_recipe(recipe_path)
    assert f"rank_{recipe['rank']}" == recipe_path.parent.parent.name, "the folder name must state the rank"
    for style in ("professional", "concise", "elaborate", "friendly", "grammar"):
        summary = mrc.write_config(recipe_path, style, tmp_path / "runs", tmp_path / f"{style}.yml")
        assert summary.startswith(f"config ok: rank {recipe['rank']} ")


def test_rank_16_recipes_keep_the_released_settings(tmp_path):
    """The rank-16 adapters on Hugging Face were trained with exactly these values."""
    phase_1 = mrc.build_document(mrc.read_recipe(ROOT / "rank_16/phase_1/recipe.yml"), tmp_path)
    phase_2 = mrc.build_document(mrc.read_recipe(ROOT / "rank_16/phase_2/recipe.yml"), tmp_path)
    for doc in (phase_1, phase_2):
        assert doc["adapter"]["rank"] == 16
        assert doc["adapter"]["alpha"] == 8.0 and doc["adapter"]["use_rslora"] is True
        assert doc["devices"]["gpu"]["overrides"]["training"] == mrc.GPU_TRAINING
        assert doc["training"]["gradient_checkpointing"] is True
        assert doc["data"]["allowed_cleanup"] == mrc.CLEANUP_CLASSES
        assert all("data" not in block for block in doc["styles"].values() if isinstance(block, dict))
    assert phase_1["checkpointing"]["every_steps"] == 2000
    assert phase_1["training"]["epochs"] == 2 and phase_1["training"]["learning_rate"] == 3.0e-4
    assert phase_2["checkpointing"]["every_steps"] == 200
    assert {k: phase_2["training"][k] for k in ("epochs", "learning_rate", "lr_schedule", "warmup_steps", "min_lr_ratio")} \
        == {"epochs": 1, "learning_rate": 5.0e-5, "lr_schedule": "cosine", "warmup_steps": 30, "min_lr_ratio": 0.1}


def test_the_source_config_is_never_edited(tmp_path):
    before = mrc.SOURCE_CONFIG.read_bytes()
    mrc.write_config(ROOT / "rank_16/phase_2/recipe.yml", "grammar", tmp_path / "runs", tmp_path / "out.yml")
    assert mrc.SOURCE_CONFIG.read_bytes() == before


def test_runs_and_logs_go_to_the_runs_root(tmp_path):
    doc = mrc.build_document({"rank": 32, "every_steps": 500}, tmp_path / "runs")
    assert doc["logging"]["runs_dir"] == doc["logging"]["logs_dir"] == str(tmp_path / "runs")
    assert doc["devices"]["gpu"]["overrides"]["evaluation"]["every_steps"] == 500
    assert doc["adapter"]["rank"] == 32


@pytest.mark.parametrize("recipe, message", [
    ({"every_steps": 200}, "missing keys"),
    ({"rank": 16, "every_steps": 200, "epochs": 1}, "unknown keys"),
    ({"rank": 0, "every_steps": 200}, "positive integer"),
    ({"rank": 16, "every_steps": "200"}, "positive integer"),
    ({"rank": 16, "every_steps": 200, "training": [1]}, "mapping"),
])
def test_malformed_recipes_are_refused(tmp_path, recipe, message):
    with pytest.raises(mrc.RecipeError, match=message):
        mrc.read_recipe(_write(tmp_path, recipe))


def test_an_unknown_training_key_is_refused_by_the_config_loader(tmp_path):
    path = _write(tmp_path, {"rank": 16, "every_steps": 200, "training": {"no_such_knob": 1}})
    with pytest.raises(Exception, match="no_such_knob"):
        mrc.write_config(path, "grammar", tmp_path / "runs", tmp_path / "out.yml")


def test_get_prints_one_recipe_value(capsys):
    assert mrc.main(["--recipe", str(ROOT / "rank_16/phase_2/recipe.yml"), "--get", "every_steps"]) == 0
    assert capsys.readouterr().out.strip() == "200"
