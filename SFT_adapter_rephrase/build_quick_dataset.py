#!/usr/bin/env python
"""
Materialize a flat 50k-train / 10k-eval JSONL snapshot for one style, using
the project's own selection logic (src.data.build_splits: allowed_cleanup,
split_salt hashing, token-budget and length-ratio filters, span
preservation) so the "quick" files are exactly the records a live run
against adapter_config.yml would pick -- just pre-materialized into one flat
file per split so a CPU run does not have to scan every part file.

Read-only against the source dataset: never writes into
REPHRASE_MODEL/rephrase_dataset/final.

    python SFT_adapter_rephrase/build_quick_dataset.py --style elaborate

Writes:
    SFT_adapter_rephrase/dataset/quick/train/<style>.jsonl       (train_examples)
    SFT_adapter_rephrase/dataset/quick/evals/<style>_evals.json  (eval_examples)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.codec import TextCodec  # noqa: E402
from src.config import load_sft_config, style_token_for  # noqa: E402
from src.data import DatasetReport, build_splits, iter_records, source_files, split_key  # noqa: E402


def collect_row_index(config, report: DatasetReport) -> dict[str, int]:
    """key -> row_index of the first occurrence, mirroring data._collect's own filters."""
    allowed_cleanup = set(config.allowed_cleanup)
    seen: dict[str, int] = {}
    for record in iter_records(config.train_data, config.file_glob):
        if record.get("style") != config.expected_style:
            continue
        if record.get("cleanup", "KEEP") not in allowed_cleanup:
            continue
        source_text, target = record.get("input"), record.get("output")
        if (not isinstance(source_text, str) or not isinstance(target, str)
                or not source_text.strip() or not target.strip()):
            continue
        source_text, target = source_text.strip(), target.strip()
        if source_text == target:
            continue
        key = split_key(source_text, config.split_salt)
        if key not in seen:
            seen[key] = record.get("row_index")
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", required=True)
    parser.add_argument("--source-dir", default=None,
                         help="default: REPHRASE_MODEL/rephrase_dataset/final/<style>")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "adapter_config.yml"))
    args = parser.parse_args()

    source_dir = Path(args.source_dir) if args.source_dir else (
        Path("/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final") / args.style
    )
    if not source_dir.is_dir():
        raise SystemExit(f"source dir not found: {source_dir}")

    overrides = {"data": {"style": args.style, "train_data": str(source_dir), "eval_data": None}}
    config = load_sft_config(args.config, device="cpu", overrides=overrides)

    style_token = style_token_for(args.style)
    codec = TextCodec.from_tokenizer_dir(config.base_model.tokenizer_dir, style_token)

    print(f"[{args.style}] source files: {len(source_files(source_dir, config.data.file_glob))}", flush=True)
    print(f"[{args.style}] min/max_length_ratio={config.data.min_length_ratio}/{config.data.max_length_ratio} "
          f"max_source/target_tokens={config.data.max_source_tokens}/{config.data.max_target_tokens} "
          f"span_preservation={config.data.require_span_preservation}", flush=True)

    train_split, eval_split, report = build_splits(config.data, codec)
    print(f"[{args.style}] records scanned: {report.records}  rejected: {dict(sorted(report.rejected.items()))}",
          flush=True)
    print(f"[{args.style}] accepted: train={report.train} eval={report.eval}", flush=True)

    row_index = collect_row_index(config.data, report)

    out_dir_train = PROJECT_ROOT / "SFT_adapter_rephrase" / "dataset" / "quick" / "train"
    out_dir_eval = PROJECT_ROOT / "SFT_adapter_rephrase" / "dataset" / "quick" / "evals"
    out_dir_train.mkdir(parents=True, exist_ok=True)
    out_dir_eval.mkdir(parents=True, exist_ok=True)

    train_path = out_dir_train / f"{args.style}.jsonl"
    eval_path = out_dir_eval / f"{args.style}_evals.json"

    def write(path: Path, split) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for example in split:
                rec = {
                    "row_index": row_index.get(example.key),
                    "input": example.source_text,
                    "style": args.style,
                    "output": example.target_text,
                    "cleanup": "KEEP",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        before = len(split)
        after = sum(1 for _ in tmp.open(encoding="utf-8"))
        assert before == after, f"{path}: count mismatch {before} vs {after}"
        tmp.replace(path)

    write(train_path, train_split)
    write(eval_path, eval_split)

    print(f"[{args.style}] wrote {len(train_split)} -> {train_path}", flush=True)
    print(f"[{args.style}] wrote {len(eval_split)} -> {eval_path}", flush=True)


if __name__ == "__main__":
    main()
