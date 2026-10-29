"""
Where the raw text comes from: one adapter per corpus in architecture.md §9.1.

A source's entire job is to yield raw strings. Everything that happens to
those strings — cleaning, PII masking, quality filtering, deduplication,
budgeting — belongs to `pipeline.py` and `writer.py` and is identical for
every source. That split is what keeps the fidelity rules defined once
instead of once per source, with that many chances to get them subtly wrong.

Every source here is licensed for commercial use: ODC-By web text, Apache-2.0
synthetic text, permissively licensed code, public-domain books, and two
MIT/CC0 command-line references.

Two kinds of adapter
--------------------
**HuggingFace streaming** covers the five large corpora. `streaming=True`
means we stop the moment the byte budget is met instead of downloading a
terabyte to keep a slice of it. Parallelism comes from sharding: each worker
takes `shard(num_shards, index)` of the stream, which splits by underlying
file where the format allows and therefore actually opens separate
connections rather than pulling one stream through several threads.

**Git repositories** cover the two command-line references, each a small
upstream repo cloned once and then split across workers by file.

Dataset availability, checked rather than assumed
-------------------------------------------------
* **Permissive code.** `codeparrot/github-code-clean` on its main branch is a
  loading-script dataset, which current `datasets` refuses to execute — the
  same failure that forced `deepmind/pg19` to be swapped for a Parquet
  mirror. Its auto-converted Parquet branch works, so this reads
  `refs/convert/parquet` directly and selects only the permissively licensed
  configs (`all-mit`, `all-apache-2.0`, `all-bsd-3-clause`), which is exactly
  the "Apache/permissive" §9.1 asks for.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .errors import ExtractionError, SourceConfigError
from .pipeline import PipelineConfig

# --------------------------------------------------------------------------
# Source specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One corpus: how to read it, how to treat it, and what it is for."""

    source_id: str
    kind: str  # "hf" | "git_tldr" | "git_cheat"
    role: str
    license: str
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    # HF-specific
    hf_dataset: str = ""
    hf_config: str | None = None
    hf_revision: str | None = None
    hf_data_files: tuple[str, ...] = ()
    text_field: str = "text"
    shuffle_buffer: int = 10_000
    # How many underlying files a worker's shuffle buffer draws from at once.
    #
    # This is a *memory* knob, not a quality one, and it is the single largest
    # consumer of RAM in a worker. `datasets` >= 5.0 fills the shuffle buffer
    # from `max_buffer_input_shards` files concurrently, holding a separate
    # open Parquet reader (and its download/decode buffers) for each. Measured
    # on this corpus, one reader costs ~2.4GB of resident memory, and the cost
    # is linear in the number of readers: the library default of 10 puts a
    # single FineWeb-Edu worker at ~24GB, which is what OOM-killed a 125GB box
    # running 20 workers. One reader holds the same worker at ~3GB.
    #
    # `buffer_size` is deliberately *not* the lever here: 1,000 and 10,000
    # both plateau around 24GB at 10 readers, because the memory is in the
    # readers rather than in the buffered rows. So the buffer stays large
    # (good mixing, nearly free) and the reader count comes down.
    #
    # What this costs: the buffer mixes rows within one file at a time instead
    # of across ten. File *order* is still shuffled -- `shuffle()` randomizes
    # the shard order independently of this -- so a budget-limited run still
    # takes a random slice of the dataset rather than a fixed prefix.
    shuffle_input_shards: int = 1


def _hf(source_id: str, dataset: str, **kwargs) -> SourceSpec:
    return SourceSpec(source_id=source_id, kind="hf", hf_dataset=dataset, **kwargs)


# The permissively licensed Parquet configs of github-code-clean. Selected by
# license, not by language: "permissive code" in §9.1 is a licensing
# statement, and MIT/Apache-2.0/BSD-3 is what makes the resulting weights
# cleanly sellable.
CODE_PARQUET_CONFIGS = ("all-mit", "all-apache-2.0", "all-bsd-3-clause")
CODE_PARQUET_GLOB = (
    "hf://datasets/codeparrot/github-code-clean@refs/convert/parquet/{config}/partial-train/*.parquet"
)


SOURCES: dict[str, SourceSpec] = {
    "fineweb_edu": _hf(
        "fineweb_edu",
        "HuggingFaceFW/fineweb-edu",
        hf_config="sample-350BT",
        role="primary high-quality English web backbone",
        license="ODC-By 1.0",
    ),
    "c4": _hf(
        "c4",
        "allenai/c4",
        hf_config="en",
        role="broad web register diversity",
        license="ODC-By 1.0",
    ),
    "cosmopedia": _hf(
        "cosmopedia",
        "HuggingFaceTB/cosmopedia",
        hf_config="web_samples_v2",
        role="clean synthetic instructional prose; boosts fluency for small models",
        license="Apache-2.0",
    ),
    "gutenberg_pg19": _hf(
        "gutenberg_pg19",
        "emozilla/pg19",
        role="long-form literary English; balances short conversational text",
        license="Public domain (pre-1919 US works)",
        shuffle_buffer=2_000,
    ),
    "permissive_code": SourceSpec(
        source_id="permissive_code",
        kind="hf",
        hf_dataset="parquet",
        hf_data_files=tuple(CODE_PARQUET_GLOB.format(config=c) for c in CODE_PARQUET_CONFIGS),
        text_field="code",
        role="teaches code-span handling, for verbatim code preservation",
        license="MIT / Apache-2.0 / BSD-3-Clause (selected configs only)",
        # Code is not English prose. Every quality heuristic would reject it
        # for being exactly what it is meant to be, and an `@decorator` is
        # syntax rather than a mention of a person.
        pipeline=PipelineConfig(quality=False, mask_pii=False),
    ),
    "cli_tldr": SourceSpec(
        source_id="cli_tldr",
        kind="git_tldr",
        role=(
            "plain-English command descriptions paired with real CLI syntax "
            "(Linux/macOS); the CLI-helper downstream task's input/output "
            "shape directly, earmarked for the <cli_reserved_*> vocabulary"
        ),
        # The pages are CC BY 4.0 (attribution required, see NOTICE); only the
        # repository's own scripts are MIT, and none of those are read.
        license="CC BY 4.0",
        # Short structured records (one command, a few example lines), not
        # prose -- the English-fluency quality heuristics would reject a
        # `tar -xzf archive.tar.gz` line for the same reasons they correctly
        # reject permissive_code, and PII masking has nothing to do here:
        # tldr pages contain no user-submitted text, only maintained docs.
        pipeline=PipelineConfig(quality=False, mask_pii=False),
    ),
    "cli_cheat_sheets": SourceSpec(
        source_id="cli_cheat_sheets",
        kind="git_cheat",
        role="community command cheat sheets (the cheat.sh source data); wider tool coverage than tldr",
        license="CC0-1.0 (cheat/cheatsheets)",
        pipeline=PipelineConfig(quality=False, mask_pii=False),
    ),
}


def resolve(source_id: str) -> SourceSpec:
    """Look up a source by id."""
    try:
        return SOURCES[source_id]
    except KeyError:
        raise SourceConfigError(
            f"unknown source {source_id!r}; known sources: {', '.join(sorted(SOURCES))}"
        ) from None


# --------------------------------------------------------------------------
# HuggingFace streaming
# --------------------------------------------------------------------------


def iter_hf_records(
    spec: SourceSpec, *, seed: int, shard_index: int = 0, num_shards: int = 1
) -> Iterator[str]:
    """
    Stream raw text from a Hub dataset, taking only this worker's shard.

    Shuffling uses a bounded buffer rather than a global shuffle — the only
    option when streaming without downloading the whole dataset — which is
    deterministic given the same seed and dataset revision.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        # Include str(exc): this branch fires for ANY ImportError, not only
        # "datasets isn't installed" -- a real install with a broken
        # transitive dependency (pyarrow/numpy ABI mismatch, a partial
        # install, a fork-safety issue surfacing only inside a worker
        # process) raises ImportError too, and would otherwise be
        # misreported as "pip install datasets" even though that's not the
        # actual fix needed.
        raise SourceConfigError(
            f"could not import the `datasets` package ({exc}). If it's "
            f"already installed, this is likely a broken/incompatible "
            f"install (pyarrow/numpy version mismatch is the most common "
            f"cause) rather than a missing package -- try `pip install "
            f"--force-reinstall datasets` or check `pip check`."
        ) from exc

    kwargs: dict = {"split": "train", "streaming": True}
    if spec.hf_data_files:
        kwargs["data_files"] = {"train": list(spec.hf_data_files)}
    elif spec.hf_config:
        kwargs["name"] = spec.hf_config
    if spec.hf_revision:
        kwargs["revision"] = spec.hf_revision

    try:
        dataset = load_dataset(spec.hf_dataset, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surfaced as a config error with context
        raise SourceConfigError(
            f"could not open {spec.hf_dataset} "
            f"(config={spec.hf_config}, revision={spec.hf_revision}): {exc}"
        ) from exc

    if num_shards > 1:
        dataset = _shard(dataset, num_shards, shard_index)
    if spec.shuffle_buffer:
        dataset = _shuffle(
            dataset,
            seed=seed + shard_index,
            buffer_size=spec.shuffle_buffer,
            input_shards=spec.shuffle_input_shards,
        )

    for row in dataset:
        text = row.get(spec.text_field)
        if text:
            yield text


def _shuffle(dataset, *, seed: int, buffer_size: int, input_shards: int):
    """
    Shuffle a streaming dataset, capping how many files feed the buffer.

    `max_buffer_input_shards` is a `datasets` >= 5.0 parameter; see
    `SourceSpec.shuffle_input_shards` for why capping it is what keeps a
    worker inside a few GB instead of a few tens of GB.

    The parameter is passed only when the installed `datasets` actually
    accepts it, checked by signature rather than by catching `TypeError`.
    `requirements.txt` allows `datasets>=2.14`, and a `try/except TypeError`
    around a call whose body is a generator-producing library function would
    also swallow a genuine `TypeError` raised *inside* it -- turning a real
    bug into a silent fallback to the 10-reader default, which is the exact
    failure this function exists to prevent.
    """
    import inspect

    kwargs: dict = {"seed": seed, "buffer_size": buffer_size}
    if "max_buffer_input_shards" in inspect.signature(dataset.shuffle).parameters:
        kwargs["max_buffer_input_shards"] = max(1, input_shards)
    return dataset.shuffle(**kwargs)


def _shard(dataset, num_shards: int, index: int):
    """
    Split by underlying file when the format allows, else stride.

    File-level sharding is what makes parallelism real: each worker opens its
    own files, so N workers move N times the bytes. The modulo fallback still
    divides the *work* correctly, it just does not divide the download — every
    worker would pull the whole stream and discard most of it — so it exists
    only so an unusual dataset degrades to "correct but slow" rather than
    crashing.
    """
    try:
        return dataset.shard(num_shards=num_shards, index=index)
    except (AttributeError, TypeError, ValueError):
        from itertools import islice

        return islice(dataset, index, None, num_shards)


def check_hf_dependencies() -> None:
    """
    Verify `datasets` is importable in *this* process before any worker
    starts.

    Without this, a missing/broken `datasets` install is only ever
    discovered inside `iter_hf_records`, which every HF-source worker only
    calls after two levels of process creation (the outer per-source pool,
    then the inner per-worker pool). This is called from `run_hf_source`
    inside the *outer* worker but before the *inner* pool is created, so it
    still catches the failure one level earlier than it would otherwise
    surface, with one specific real-world case in mind: `python3
    download.py` resolving to a different interpreter than the one `pip
    install -r requirements.txt` targeted (common on machines with
    multiple Pythons on PATH -- conda, pyenv, Homebrew, system Python all
    coexisting). Checking here means that class of mismatch is caught as
    one clear message naming the exact interpreter in use, instead of a
    `ModuleNotFoundError` surfacing from inside a worker with no
    indication of *which* Python hit it.
    """
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        import sys

        raise SourceConfigError(
            f"could not import the `datasets` package under {sys.executable} "
            f"({exc}). This is the Python actually running download.py -- "
            f"if `pip install datasets` succeeded but this still fails, "
            f"you very likely installed it into a *different* Python "
            f"(check `which python3` / `which pip` agree, and that no "
            f"other Python earlier on PATH is shadowing the one you "
            f"activated). Fix: `{sys.executable} -m pip install -r "
            f"requirements.txt`, which installs into this exact interpreter "
            f"regardless of what `pip`/`python3` resolve to."
        ) from exc


# --------------------------------------------------------------------------
# tldr-pages (CLI helper)
# --------------------------------------------------------------------------
#
# The whole repo is one shallow `git clone` (a few tens of MB), so there is
# nothing to shard by file: one clone, then the page list is what gets split
# across workers.

CLI_TLDR_REPO_URL = "https://github.com/tldr-pages/tldr.git"

# Only the English, OS-relevant page sets. `pages` itself holds these plus
# a long tail of `pages.<locale>` translation directories (pages.pt_BR,
# pages.zh, ...) which this excludes by name: a directory-level decision is
# exact and free, where the per-document language filter is only a
# statistical backstop. `common` covers cross-platform tools; `linux`/`osx` are
# the two platforms architecture.md's CLI-helper task actually targets;
# `android`/`sunos`/`freebsd`/`netbsd`/`openbsd`/`windows` are included too
# since they're still English and still real command syntax.
CLI_TLDR_PLATFORM_DIRS = (
    "common", "linux", "osx", "android", "sunos",
    "freebsd", "netbsd", "openbsd", "windows",
)


def clone_git_repo(url: str, dest_dir: Path, *, depth: int = 1) -> Path:
    """Shallow-clone `url` into `dest_dir`, or reuse an existing clone."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    repo_name = url.rstrip("/").rsplit("/", 1)[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[: -len(".git")]
    dest = dest_dir / repo_name
    if (dest / ".git").is_dir():
        return dest
    tmp = dest_dir / f".{repo_name}.part"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", str(depth), "--single-branch", url, str(tmp)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        tmp.replace(dest)
    except subprocess.CalledProcessError as exc:
        raise ExtractionError(
            f"git clone failed for {url}: {exc.stderr.decode('utf-8', 'replace')[:400]}"
        ) from exc
    return dest


def list_tldr_pages(repo_dir: Path) -> list[Path]:
    """Every English-locale page markdown file, sorted for determinism."""
    pages_dir = repo_dir / "pages"
    files: list[Path] = []
    for platform in CLI_TLDR_PLATFORM_DIRS:
        sub = pages_dir / platform
        if sub.is_dir():
            files.extend(sorted(sub.glob("*.md")))
    return files


def iter_tldr_pages(paths: list[Path]) -> Iterator[str]:
    """One yielded string per page: command, description, worked examples."""
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = format_tldr_page(path.stem, raw)
        if text:
            yield text


def format_tldr_page(command: str, raw: str) -> str:
    """
    Turn one tldr-pages markdown file into plain training text.

    tldr's format is fixed and simple: `# command`, then `>` description
    lines, then repeated `- explanation:` / backtick-fenced-command pairs.
    Reformatted as prose-plus-command pairs rather than left as markdown,
    since the CLI-helper task's actual input/output shape is "plain-English
    request in, shell command out" — this is that shape, not a markdown
    rendering exercise.
    """
    description_lines: list[str] = []
    examples: list[str] = []
    pending_desc: str | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("# "):
            continue
        if line.startswith("> "):
            description_lines.append(line[2:].strip())
        elif line.startswith("- "):
            pending_desc = line[2:].strip().rstrip(":")
        elif line.startswith("`") and line.endswith("`") and len(line) >= 2:
            if pending_desc:
                code = line.strip("`")
                examples.append(f"{pending_desc}\n{code}")
                pending_desc = None

    if not examples:
        return ""

    header = command.replace("-", " ")
    if description_lines:
        header = f"{header}: {' '.join(description_lines)}"
    return header + "\n\n" + "\n\n".join(examples)


def check_git_dependencies() -> None:
    """Verify `git` is on PATH before any worker starts."""
    from shutil import which

    if which("git") is None:
        raise SourceConfigError(
            "no `git` binary on PATH; the tldr-pages and cheat-sheets "
            "sources are cloned with `git clone`. Install it with "
            "`apt-get install -y git` (Debian/Ubuntu) or `brew install git` "
            "(macOS)."
        )


# --------------------------------------------------------------------------
# cheat.sh community cheat sheets
# --------------------------------------------------------------------------

CLI_CHEAT_REPO_URL = "https://github.com/cheat/cheatsheets.git"


def list_cheat_sheets(repo_dir: Path) -> list[Path]:
    """Every plain-text sheet file, skipping VCS/dotfiles and directories."""
    files = []
    for path in sorted(repo_dir.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(repo_dir).parts):
            files.append(path)
    return files


def format_cheat_sheet(name: str, raw: str) -> str:
    """
    Turn one cheat-sheet file into plain training text.

    The format is looser than tldr's (no fixed markdown grammar), but the
    convention is consistent: `#`-prefixed lines are prose description,
    blank-separated runs of everything else are the command(s) that
    description explains. Grouped as (description, code-block) pairs the
    same way `format_tldr_page` groups them, for the same reason: this is
    the CLI-helper task's actual input/output shape.
    """
    description_lines: list[str] = []
    blocks: list[str] = []
    current_code: list[str] = []

    def _flush_code() -> None:
        if current_code:
            blocks.append("\n".join(current_code))
            current_code.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            _flush_code()
            comment = stripped.lstrip("#").strip()
            if comment:
                description_lines.append(comment)
        elif stripped:
            current_code.append(stripped)
        else:
            _flush_code()
    _flush_code()

    if not blocks:
        return ""

    header = name.replace("-", " ").replace("_", " ")
    if description_lines:
        header = f"{header}: {' '.join(description_lines)}"
    return header + "\n\n" + "\n\n".join(blocks)


def iter_cheat_sheets(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = format_cheat_sheet(path.name, raw)
        if text:
            yield text
