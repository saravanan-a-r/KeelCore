"""Freeze the phase-2 training set of every rephrase adapter, so the run can be reproduced from the folder alone.

    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_2_dataset.py            # all five styles
    /root/venv/bin/python SFT_adapter_rephrase/data_prep/build_phase_2_dataset.py --styles grammar

For each style, next to the phase-2 rows already in final/phase_2/<style>/ (phase_2_part_01 = date file,
phase_2_part_02 = url file, written by Restyle/export_phase2.py), this writes:

  train/phase_1_dataset.jsonl    replay rows copied from the style's phase-1 train data (frozen copy)
  train/phase_2_augment_tokens.jsonl  phase-2 url rows with the link swapped for a lowercase email, @handle, path or
                                 host, in input and target alike (teaches: copy such strings exactly, even first)
  evals/phase_2_heldout.jsonl    phase-2 rows held out from training (copies; the trainer drops their inputs from train)
  evals/phase_1_eval.jsonl       rows of the style's phase-1 evals, to see that the old behaviour is kept
  manifest.json                  counts, selection rules, seeds and the SHA-256 of every file, sources included
The layout is the phase-1 one: <style>/train/*.jsonl and <style>/evals/*.jsonl (train/ also holds phase_2_part_01/02).

Selection is by SHA-256 of a fixed salt + the input text (or the source row id), never by file order or a random
generator, so rerunning this script on the same sources writes byte-identical files. The phase-1 replay is not a
plain random sample: replay_selection.py keeps only rows whose target keeps every fact of the input and is clean, and
spreads them over the phase-1 data's topics, lengths and fact types in proportion (see its docstring and the README).
Phase-1 eval rows are taken only from cleanup KEEP rows with input != output and no input shared with phase 2.
Needs scikit-learn (topic clusters): PYTHONPATH=<dir with scikit-learn> /root/venv/bin/python ... .
Nothing outside final/phase_2/ is written; the phase-1 folders are only read.
"""
import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path

import replay_selection as rs

FINAL = Path("/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final")
PHASE2 = FINAL / "phase_2"
PHASE2_INPUTS = FINAL / "professional_phase_2"            # the casual inputs every phase-2 style was written from
PHASE1_DIR = {"professional": "professional", "concise": "concise", "elaborate": "elaborate",
              "friendly": "friendly_sft", "grammar": "grammar"}
PARTS = {"date": "phase_2_part_01", "url": "phase_2_part_02"}
SOURCE_FILES = {"date": "input_with_date_qwen.jsonl", "url": "input_with_url_qwen.jsonl"}

HELDOUT = {"date": 1500, "url": 500}       # inputs held out of phase-2 training, the same inputs for every style
PHASE1_EVAL_ROWS = 2000
REPLAY_PER_NEW_ROW = 1.0                   # phase-1 rows per phase-2 training row (see README: the trial measured 1:1)
AUGMENT_ROWS = 1500                        # token-transplant rows per style (v2, see README)
SALT_HELDOUT = "phase2-heldout-v1"
SALT_AUGMENT = "phase2-augment-v1"
SALT_P1EVAL = "phase2-p1eval-v1"


def sha_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    # Iterate lines, not splitlines(): a raw U+2028 inside a JSON string is not a record break.
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def heldout_ids():
    """{name: set(row_index)}: the source rows held out, chosen once from the shared input files."""
    out = {}
    for name, fname in SOURCE_FILES.items():
        n_rows = sum(1 for _ in (PHASE2_INPUTS / fname).open(encoding="utf-8"))
        order = sorted(range(n_rows), key=lambda i: sha_hex(f"{SALT_HELDOUT}/{name}/{i}"))
        out[name] = set(order[:HELDOUT[name]])
    return out


def smallest_by_key(paths, wanted, salt, keep):
    """The `wanted` records of `paths` that `keep` accepts, with the smallest SHA-256(salt + input), sorted by it."""
    heap, seen = [], set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if not keep(r):
                    continue
                inp = r["input"].strip()
                if inp in seen:
                    continue
                seen.add(inp)
                key = sha_hex(salt + inp)
                item = (_Neg(key), key, r)
                if len(heap) < wanted:
                    heapq.heappush(heap, item)
                elif key < heap[0][1]:
                    heapq.heapreplace(heap, item)
    return [r for _, _, r in sorted(heap, key=lambda x: x[1])]


class _Neg:
    """Reverses string order so heapq (a min-heap) keeps the largest key on top."""
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k

    def __lt__(self, other):
        return self.k > other.k


def _versions():
    import numpy
    import sklearn
    return {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__}


URL_RE = re.compile(r"https?://[^\s<>\"')\]]+[^\s<>\"')\].,;:!?]")
FIRST = "anna ravi meera tomas li-wei olu sara deepak keiko marco nadia ivan chloe arjun fatima jonas".split()
LAST = "patel novak okafor lindqvist tanaka moreau reyes kowalski ahmed bauer silva chen".split()
DOMAINS = "northwind.io tidepool.dev orbitlabs.com quarry-tech.net pinecrest.org lumen-ai.co".split()
PROJECTS = "billing infra webapp etl mobile search".split()
FILES = "settings.toml deploy.yaml main.go utils.py index.ts schema.sql".split()


def token_pool():
    """Lowercase strings that must be copied byte-exact. None of them is in eval_sets/common (no leakage)."""
    pool = []
    for i, f in enumerate(FIRST):
        for j, l in enumerate(LAST):
            d = DOMAINS[(i + j) % len(DOMAINS)]
            pool += [f"{f}.{l}@{d}", f"{f[0]}{l}@{d}", f"@{f}_{l}"]
    for i, pj in enumerate(PROJECTS):
        for j, fl in enumerate(FILES):
            pool += [f"~/work/{pj}/{fl}", f"src/{pj}/{fl}", f"{pj}.internal.{DOMAINS[(i + j) % len(DOMAINS)]}:{8000 + 7 * i + j}"]
    return sorted(set(pool))


def augment_rows(rows, style):
    """Swap the single link of a phase-2 url row for a token, in input and target alike."""
    pool = token_pool()
    cand = []
    for r in rows:
        urls = URL_RE.findall(r["input"])
        if len(urls) != 1:
            continue
        u = urls[0]
        if r["input"].count(u) != 1 or r["output"].count(u) != 1 or r["input"].strip() == r["output"].strip():
            continue
        cand.append((sha_hex(SALT_AUGMENT + r["input"]), r, u))
    cand.sort(key=lambda x: x[0])
    out = []
    for key, r, u in cand[:AUGMENT_ROWS]:
        tok = pool[int(key[:8], 16) % len(pool)]
        out.append({"row_index": r["row_index"], "input": r["input"].replace(u, tok), "style": style,
                    "output": r["output"].replace(u, tok), "cleanup": "KEEP"})
    return out, len(cand)


def build(style, held):
    sdir = PHASE2 / style
    p1 = FINAL / PHASE1_DIR[style]
    new_rows, heldout_rows = [], []
    for name, part in PARTS.items():
        for r in read_jsonl(sdir / "train" / f"{part}_{style}.jsonl"):
            (heldout_rows if r["row_index"] in held[name] else new_rows).append(r)
    trainable_new = [r for r in new_rows if r["input"].strip() != r["output"].strip()]
    phase2_inputs = {r["input"].strip() for r in new_rows + heldout_rows}

    p1_eval_files = sorted((p1 / "evals").glob("*.jsonl"))
    p1_eval = smallest_by_key(
        p1_eval_files, PHASE1_EVAL_ROWS, SALT_P1EVAL,
        lambda r: r.get("style") == style and r.get("cleanup", "KEEP") == "KEEP"
        and r["input"].strip() != r["output"].strip() and r["input"].strip() not in phase2_inputs)
    all_p1_eval_inputs = {r["input"].strip() for f in p1_eval_files for r in read_jsonl(f)}

    url_train = [r for r in read_jsonl(sdir / "train" / f"{PARTS['url']}_{style}.jsonl") if r["row_index"] not in held["url"]]
    augment, n_aug_candidates = augment_rows(url_train, style)
    n_replay = round((len(trainable_new) + len(augment)) * REPLAY_PER_NEW_ROW)
    p1_train_files = sorted((p1 / "train").glob("*.jsonl"))
    pop = rs.population(p1_train_files, style, phase2_inputs | all_p1_eval_inputs)
    feats = rs.annotate(pop, style)
    topics = rs.Topics(rs.K_CLUSTERS, seed=0).fit([r["input"] for r in pop])
    replay, sel_info = rs.select(pop, feats, topics.labels([r["input"] for r in pop]), n_replay)
    assert len(replay) == n_replay, (style, len(replay), n_replay)
    pop_stats = {"fact_corrupt_target_%": round(100 * sum(f["fact_bad"] for f in feats) / len(feats), 1),
                 "not_clean_target_%": round(100 * sum(f["clean_bad"] for f in feats) / len(feats), 1)}

    write_jsonl(sdir / "train" / "phase_1_dataset.jsonl", replay)
    write_jsonl(sdir / "train" / "phase_2_augment_tokens.jsonl", augment)
    write_jsonl(sdir / "evals" / "phase_2_heldout.jsonl", heldout_rows)
    write_jsonl(sdir / "evals" / "phase_1_eval.jsonl", p1_eval)

    train_files = [sdir / "train" / f"{p}_{style}.jsonl" for p in PARTS.values()] + [sdir / "train" / "phase_1_dataset.jsonl", sdir / "train" / "phase_2_augment_tokens.jsonl"]
    eval_files = [sdir / "evals" / "phase_2_heldout.jsonl", sdir / "evals" / "phase_1_eval.jsonl"]
    manifest = {
        "style": style,
        "built_by": "SFT_adapter_rephrase/data_prep/build_phase_2_dataset.py",
        "train_files": {f"train/{f.name}": {"rows": sum(1 for _ in f.open(encoding="utf-8")), "sha256": file_sha(f)} for f in train_files},
        "eval_files": {f"evals/{f.name}": {"rows": sum(1 for _ in f.open(encoding="utf-8")), "sha256": file_sha(f)} for f in eval_files},
        "counts": {
            "phase_2_rows_in_part_files": len(new_rows) + len(heldout_rows),
            "phase_2_heldout": len(heldout_rows),
            "phase_2_train": len(new_rows),
            "phase_2_train_identity_pairs (trainer drops)": len(new_rows) - len(trainable_new),
            "phase_2_augment_tokens": len(augment),
            "phase_1_replay": len(replay),
            "phase_1_eval": len(p1_eval),
        },
        "selection": {
            "phase_2_heldout": f"source rows ranked by sha256('{SALT_HELDOUT}/<date|url>/<row_index>'), first "
                               f"{HELDOUT['date']} of the date file and {HELDOUT['url']} of the url file "
                               f"(the same inputs for every style; a style that rejected some of them holds out fewer). "
                               f"They stay in the part files (train/); the trainer drops every train row whose input is an eval input.",
            "phase_1_replay": f"{n_replay} rows chosen by replay_selection.py from a hash-uniform population of "
                              f"{sel_info['population']:,} rows of {p1}/train/*.jsonl (style {style!r}, cleanup KEEP, input != "
                              f"output, input not in phase 2 or the phase-1 evals): only rows whose target keeps every fact of "
                              f"the input and is clean ({sel_info['candidates']:,} of them), allocated over "
                              f"{sel_info['strata']} strata (topic cluster x length x fact class) in proportion to the "
                              f"population, near duplicates skipped, ties by sha256('replay-pick-v1' + input). "
                              f"Count = {REPLAY_PER_NEW_ROW:g} x (trainable phase-2 train rows + augment rows).",
            "phase_2_augment_tokens": f"{len(augment)} of {n_aug_candidates} phase-2 url train rows (held-out excluded) whose "
                                      f"input and target each hold their single link once, ranked by sha256('{SALT_AUGMENT}' + input); "
                                      f"the link is replaced in both by a token from token_pool() picked by that hash.",
            "phase_1_eval": f"rows of {p1}/evals/*.jsonl with cleanup KEEP, input != output, input not in phase 2; the "
                            f"{PHASE1_EVAL_ROWS} with the smallest sha256('{SALT_P1EVAL}' + input).",
        },
        "replay_selection": {**sel_info, "population_before_filter": pop_stats,
                             "libraries": _versions(), "k_clusters": rs.K_CLUSTERS, "population_size": rs.POP},
        "sources": {
            "phase_2_inputs": {f: file_sha(PHASE2_INPUTS / f) for f in SOURCE_FILES.values()},
            "phase_1_train": {f.name: {"bytes": f.stat().st_size, "sha256": file_sha(f)} for f in p1_train_files},
            "phase_1_evals": {f.name: {"bytes": f.stat().st_size, "sha256": file_sha(f)} for f in p1_eval_files},
        },
    }
    (sdir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(style, json.dumps(manifest["counts"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", nargs="+", default=list(PHASE1_DIR), choices=list(PHASE1_DIR))
    args = ap.parse_args()
    held = heldout_ids()
    for style in args.styles:
        build(style, held)


if __name__ == "__main__":
    main()
