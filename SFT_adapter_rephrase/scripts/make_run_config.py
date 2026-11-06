"""
Write the configuration of one GPU training run: adapter_config.yml with a
recipe laid over it, validated exactly as sft.py will load it.

    python scripts/make_run_config.py --recipe rank_16/phase_1/recipe.yml \\
        --style grammar --runs-root runs/adapters --out logs/run_configs/.../grammar.yml
    python scripts/make_run_config.py --recipe rank_16/phase_1/recipe.yml --get every_steps

adapter_config.yml is never edited. On top of it every run gets the same data
and GPU rules (below); the recipe adds what differs between runs:

    rank: 16             LoRA rank (alpha stays as in adapter_config.yml; rsLoRA)
    every_steps: 2000    evaluation + checkpoint interval
    training: {...}      values laid over adapter_config.yml's `training` section

Data rules: every cleanup class is accepted, nothing is dropped for length
ratio or span loss, up to 2048 tokens a side (the model's maximum), every
*.jsonl under the data folders is read (recursively).

GPU plan, measured on the A100 40 GB (bf16, rank 16 and 64, real rows):
every optimizer step is exactly 32 examples (16 x 2), the effective batch the
recipe was tuned with. Only how a step is fed to the GPU changes (src/trainer.py,
tested to give the same weights):
  - micro_batch_tokens 64000 / target_token_weight 3: a step is cut by padded
    cost, rows x (longest source + 3 x longest target), not by a fixed 16 rows.
    Short rows run as one micro-batch of 32, 2048-token rows as several of
    ~7-10. A target token costs ~3x a source token (its logits span the 33.7k
    vocabulary). Worst case ~18 GB.
  - checkpointing_min_tokens 6144: micro-batches of <= 6144 padded tokens skip
    gradient checkpointing (it only recomputes activations; same gradient).
    Worst case 29.7 GB.
  Measured: 62 examples/s vs 23.6 with 16 x 2 rows (2.6x).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SFT_DIR = Path(__file__).resolve().parents[1]
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from src.config import load_sft_config  # noqa: E402

SOURCE_CONFIG = SFT_DIR / "adapter_config.yml"
CLEANUP_CLASSES = ["KEEP", "KEEP_CJK_TAIL", "KEEP_LONGER_SEGMENT", "REVIEW_TRUNCATED", "STRIP_NOTE",
                   "TRUNCATE_AT_LEAK", "TRUNCATE_AT_TAG"]
EFFECTIVE_BATCH = 32
GPU_TRAINING = {
    "batch_size": 16,                            # effective batch 16 x 2 = 32; eval rows per batch = 2 x 16
    "gradient_accumulation_steps": 2,
    "micro_batch_tokens": 64000,                 # see the GPU plan above
    "target_token_weight": 3.0,
    "checkpointing_min_tokens": 6144,
}
EVERYTHING = 10**9
RECIPE_KEYS = {"rank", "every_steps", "training"}


class RecipeError(ValueError):
    pass


def read_recipe(path: str | Path) -> dict:
    path = Path(path)
    try:
        recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeError(f"cannot read recipe {path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise RecipeError(f"{path}: a recipe is a mapping")
    unknown = set(recipe) - RECIPE_KEYS
    missing = {"rank", "every_steps"} - set(recipe)
    if unknown or missing:
        raise RecipeError(f"{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    if not isinstance(recipe.get("training", {}), dict):
        raise RecipeError(f"{path}: `training` is a mapping")
    for key in ("rank", "every_steps"):
        if not isinstance(recipe[key], int) or isinstance(recipe[key], bool) or recipe[key] <= 0:
            raise RecipeError(f"{path}: `{key}` must be a positive integer")
    return recipe


def build_document(recipe: dict, runs_root: str | Path, source: Path = SOURCE_CONFIG) -> dict:
    """adapter_config.yml + the shared run rules + the recipe, as a config document."""
    doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    every_steps = recipe["every_steps"]

    doc["adapter"]["rank"] = recipe["rank"]

    data = doc["data"]
    data["file_glob"] = "**/*.jsonl"                 # recursive
    data["allowed_cleanup"] = list(CLEANUP_CLASSES)
    data["require_span_preservation"] = False
    data["min_length_ratio"] = 0.0
    data["max_length_ratio"] = 1.0e9
    data["train_examples"] = EVERYTHING
    data["eval_examples"] = EVERYTHING
    data["max_source_tokens"] = 2048                 # the model's maximum length
    data["max_target_tokens"] = 2048
    # The per-style length bounds are filters too; keep only each style's acceptance bar.
    for block in (doc.get("styles") or {}).values():
        if isinstance(block, dict):
            block.pop("data", None)

    doc["training"].update(recipe.get("training", {}))
    doc["training"]["gradient_checkpointing"] = True
    doc["checkpointing"] = {"every_steps": every_steps, "keep_last": EVERYTHING}
    doc["logging"]["runs_dir"] = str(runs_root)
    doc["logging"]["logs_dir"] = str(runs_root)      # sft.log + tensorboard/ inside the run dir

    # The gpu profile is laid over the shared sections, so its values are the ones to set.
    gpu = doc["devices"]["gpu"]["overrides"]
    gpu["training"] = dict(GPU_TRAINING)
    gpu["evaluation"] = {
        "every_steps": every_steps,
        "loss_examples": EVERYTHING,                 # every eval row
        "early_stopping_patience": None,             # every checkpoint is kept and none stops the run
        "generation_examples": 500,
        "generation_batch_size": 64,
    }
    return doc


def validate(path: Path, recipe: dict, style: str) -> str:
    """Load the written file as sft.py does and check the recipe took effect; returns a one-line summary."""
    cfg = load_sft_config(path, "gpu", {"data": {"style": style}})
    t, e = cfg.training, cfg.evaluation
    problems = []
    if cfg.adapter.rank != recipe["rank"]:
        problems.append(f"rank {cfg.adapter.rank} != {recipe['rank']}")
    if not (cfg.checkpointing.every_steps == e.every_steps == recipe["every_steps"]):
        problems.append(f"eval/checkpoint interval {e.every_steps}/{cfg.checkpointing.every_steps} != {recipe['every_steps']}")
    if t.effective_batch_size != EFFECTIVE_BATCH:
        problems.append(f"effective batch {t.effective_batch_size} != {EFFECTIVE_BATCH}")
    if (t.micro_batch_tokens, t.target_token_weight, t.checkpointing_min_tokens) != (
            GPU_TRAINING["micro_batch_tokens"], GPU_TRAINING["target_token_weight"], GPU_TRAINING["checkpointing_min_tokens"]):
        problems.append("the GPU micro-batch plan was not applied")
    for key, value in recipe.get("training", {}).items():
        if getattr(t, key) != value:
            problems.append(f"training.{key} {getattr(t, key)!r} != {value!r}")
    if problems:
        raise RecipeError(f"{path}: " + "; ".join(problems))
    epochs = "epoch" if t.epochs == 1 else "epochs"
    return (f"config ok: rank {cfg.adapter.rank} (scaling {cfg.adapter.scaling:.4g}), {t.epochs:g} {epochs}, "
            f"batch {t.effective_batch_size} in micro-batches of <= {t.micro_batch_tokens} padded tokens "
            f"(target x {t.target_token_weight:g}), lr {t.learning_rate:g} {t.lr_schedule} (warmup {t.warmup_steps}), "
            f"grad checkpointing above {t.checkpointing_min_tokens} tokens, {cfg.resources.precision}, "
            f"eval + checkpoint every {e.every_steps} steps")


def write_config(recipe_path: Path, style: str, runs_root: Path, out: Path) -> str:
    recipe = read_recipe(recipe_path)
    doc = build_document(recipe, runs_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(f"# Generated from {recipe_path} -- do not edit; adapter_config.yml is the source.\n")
        yaml.safe_dump(doc, handle, sort_keys=False)
    return validate(out, recipe, style)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--get", choices=sorted(RECIPE_KEYS - {"training"}), help="print one recipe value and exit")
    parser.add_argument("--style")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.get:
            print(read_recipe(args.recipe)[args.get])
            return 0
        if not (args.style and args.runs_root and args.out):
            parser.error("--style, --runs-root and --out are required to write a config")
        print(write_config(args.recipe, args.style, args.runs_root, args.out))
    except RecipeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
