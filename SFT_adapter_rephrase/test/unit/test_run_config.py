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

sys.path.insert(0, str(ROOT))
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
    ({"rank": 16, "every_steps": 200, "keep_last": 0}, "positive integer"),
    ({"rank": 16, "every_steps": 200, "training": [1]}, "mapping"),
])
def test_malformed_recipes_are_refused(tmp_path, recipe, message):
    with pytest.raises(mrc.RecipeError, match=message):
        mrc.read_recipe(_write(tmp_path, recipe))


def test_an_unknown_training_key_is_refused_by_the_config_loader(tmp_path):
    path = _write(tmp_path, {"rank": 16, "every_steps": 200, "training": {"no_such_knob": 1}})
    with pytest.raises(Exception, match="no_such_knob"):
        mrc.write_config(path, "grammar", tmp_path / "runs", tmp_path / "out.yml")


def test_every_checkpoint_is_kept_unless_the_recipe_says_otherwise(tmp_path):
    every = mrc.build_document({"rank": 16, "every_steps": 200}, tmp_path)
    assert every["checkpointing"] == {"every_steps": 200, "keep_last": mrc.EVERYTHING}
    capped = mrc.build_document({"rank": 32, "every_steps": 2000, "keep_last": 4}, tmp_path)
    assert capped["checkpointing"] == {"every_steps": 2000, "keep_last": 4}
    path = _write(tmp_path, {"rank": 32, "every_steps": 2000, "keep_last": 4})
    assert mrc.write_config(path, "grammar", tmp_path / "runs", tmp_path / "out.yml").startswith("config ok: rank 32")


def test_rank_32_phase_1_differs_from_the_released_rank_16_only_in_rank_and_checkpoint_retention(tmp_path):
    rank_16 = mrc.build_document(mrc.read_recipe(ROOT / "rank_16/phase_1/recipe.yml"), tmp_path)
    rank_32 = mrc.build_document(mrc.read_recipe(ROOT / "rank_32/phase_1/recipe.yml"), tmp_path)
    assert rank_32["adapter"]["rank"] == 32 and rank_32["adapter"]["alpha"] == 8.0 and rank_32["adapter"]["use_rslora"]
    assert rank_32["checkpointing"]["keep_last"] < mrc.EVERYTHING
    for doc in (rank_16, rank_32):
        doc["adapter"].pop("rank")
        doc["checkpointing"].pop("keep_last")
    assert rank_16 == rank_32


def test_a_lane_trains_the_same_steps_inside_its_slice_of_the_gpu(tmp_path):
    lane = mrc.Lane("0-20", 18.0)
    solo = mrc.build_document(mrc.read_recipe(ROOT / "rank_16/phase_1/recipe.yml"), tmp_path)
    shared = mrc.build_document(mrc.read_recipe(ROOT / "rank_16/phase_1/recipe.yml"), tmp_path, lane=lane)
    gpu = shared["devices"]["gpu"]
    assert gpu["overrides"]["training"]["batch_size"] * gpu["overrides"]["training"]["gradient_accumulation_steps"] == mrc.EFFECTIVE_BATCH
    assert gpu["overrides"]["training"]["micro_batch_tokens"] == 32000
    assert gpu["overrides"]["evaluation"]["generation_batch_size"] == 32
    assert (gpu["resources"]["cores"], gpu["resources"]["gpu_memory_gb"], gpu["resources"]["require_exclusive_gpu"]) == ("0-20", 18.0, False)
    for doc in (solo, shared):                       # everything that shapes the result is identical
        for key in ("batch_size", "gradient_accumulation_steps", "micro_batch_tokens", "checkpointing_min_tokens"):
            doc["devices"]["gpu"]["overrides"]["training"].pop(key)
        doc["devices"]["gpu"]["overrides"]["evaluation"].pop("generation_batch_size")
        doc["devices"]["gpu"].pop("resources")
    assert solo == shared


def test_a_lane_config_loads_and_a_solo_one_has_no_cap(tmp_path):
    recipe = ROOT / "rank_16/phase_2/recipe.yml"
    summary = mrc.write_config(recipe, "grammar", tmp_path / "runs", tmp_path / "lane.yml", mrc.Lane("21-41", 18.0))
    assert "lane: cores 21-41, GPU memory capped at 18 GB" in summary
    mrc.write_config(recipe, "grammar", tmp_path / "runs", tmp_path / "solo.yml")
    from src.config import load_sft_config
    assert load_sft_config(tmp_path / "lane.yml", "gpu").resources.gpu_memory_gb == 18.0
    solo = load_sft_config(tmp_path / "solo.yml", "gpu").resources
    assert solo.gpu_memory_gb is None and solo.require_exclusive_gpu is True


def test_lane_cores_and_memory_cap_go_together(tmp_path):
    with pytest.raises(SystemExit):
        mrc.main(["--recipe", str(ROOT / "rank_16/phase_1/recipe.yml"), "--style", "grammar",
                  "--runs-root", str(tmp_path), "--out", str(tmp_path / "x.yml"), "--lane-cores", "0-20"])


def test_get_prints_one_recipe_value(capsys):
    assert mrc.main(["--recipe", str(ROOT / "rank_16/phase_2/recipe.yml"), "--get", "every_steps"]) == 0
    assert capsys.readouterr().out.strip() == "200"
