"""
Check that a frozen data folder is exactly what its manifest.json recorded:
the same files in train/ and evals/, byte for byte (SHA-256).

    python scripts/verify_manifest.py <data_dir>

Exit 0 and one summary line when it matches, 1 with the first difference otherwise.
The manifest is written by data_prep/build_phase_2_dataset.py.
"""

import hashlib
import json
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(data_dir: Path) -> str:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path} is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {**manifest["train_files"], **manifest["eval_files"]}
    present = {f"{part}/{path.name}" for part in ("train", "evals") for path in (data_dir / part).glob("*.jsonl")}
    if present != set(listed):
        raise ValueError(f"files differ from the manifest: extra {sorted(present - set(listed))}, "
                         f"missing {sorted(set(listed) - present)}")
    for name, info in listed.items():
        if file_sha256(data_dir / name) != info["sha256"]:
            raise ValueError(f"{name}: sha256 differs from the manifest")
    counts = manifest["counts"]
    return (f"data ok: phase-2 train {counts['phase_2_train']} + phase-1 replay {counts['phase_1_replay']}; "
            f"evals: phase-2 held-out {counts['phase_2_heldout']} + phase-1 {counts['phase_1_eval']}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        print(verify(Path(argv[0])))
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
