"""
data_prep/build_phase_3_dataset.py: the row builders keep every target as it is, and the token pool never
holds a string of eval_sets/. scripts/verify_manifest.py reads a phase-3 manifest.
"""

from __future__ import annotations

import json
import re
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "data_prep"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_phase_3_dataset as p3  # noqa: E402
import verify_manifest  # noqa: E402

KEY = "0123456789abcdef" * 4


def _row(text, output, row_index=7):
    return {"row_index": row_index, "input": text, "style": "professional", "output": output, "cleanup": "KEEP"}


def test_the_token_pool_holds_no_eval_string():
    keeps, _ = p3.eval_strings()
    assert keeps, "eval_sets/ has no keep strings"
    p3.clean_pool(keeps)                                  # raises SystemExit on a leak
    assert all(tokens for tokens in p3.token_pool().values())


def test_a_leak_is_refused():
    with pytest.raises(SystemExit, match="eval strings"):
        p3.clean_pool({"npm ci"})
    assert p3.leaks("C:\\Users\\anna\\Desktop\\report.xlsx", {"C:\\Users\\anna\\Desktop\\report.xlsx"})
    assert not p3.leaks("v1.8.2", {"5.3"})                # a short keep string is not a leak by containment


def test_exact_copy_swaps_the_link_in_input_and_target_alike():
    row = _row("see https://example.com/a for the plan", "Please see https://example.com/a for the plan.")
    out = p3.exact_copy_row(row, KEY, p3.token_pool())
    token = out["input"].split()[1]
    assert "https://" not in out["input"] + out["output"]
    assert out["input"].count(token) == 1 and out["output"].count(token) == 1
    assert out["output"] == row["output"].replace("https://example.com/a", token)
    assert p3.single_link(_row("a https://x.io/1 b https://x.io/2", "A https://x.io/1 B https://x.io/2")) is None


def test_shorthand_changes_only_the_input():
    row = _row("pls send it before the meeting with your manager",
               "Please send it before the meeting with your manager.")
    out = p3.shorthand_row(row, KEY)
    assert out["output"] == row["output"]
    assert re.search(r"\bb4\b", out["input"]) and re.search(r"\bmtg\b", out["input"])
    assert re.search(r"\bmgr\b", out["input"])


def test_shorthand_needs_a_rare_word_and_keeps_meeting_the_verb():
    assert p3.shorthand_row(_row("can you send your notes", "Could you send your notes?"), KEY) is None
    out = p3.shorthand_row(_row("we are meeting people before noon", "We are meeting people before noon."), KEY)
    assert "meeting" in out["input"] and "b4" in out["input"]


def test_a_misspelling_is_never_a_real_word():
    for word in ("broken", "dashboard", "update", "cancelled", "received"):
        typo = p3.misspell(word, KEY)
        assert typo and typo != word and p3.zipf_frequency(typo, "en") == 0


def test_typos_change_only_input_words_the_target_has():
    row = _row("the printer is broken and the dashboard shows nothing since monday",
               "The printer is broken, and the dashboard has shown nothing since Monday.")
    out = p3.typo_row(row, KEY)
    assert out["output"] == row["output"]
    changed = [a for a, b in zip(row["input"].split(), out["input"].split()) if a != b]
    assert 2 <= len(changed) <= 3 and all(re.search(r"\b" + w + r"\b", row["output"]) for w in changed)
    assert p3.typo_row(_row("short text", "Short text."), KEY) is None


def test_the_elaborate_pools_hold_no_eval_string(monkeypatch):
    keeps, _ = p3.eval_strings()
    monkeypatch.setattr(p3, "STYLE", "elaborate")
    p3.clean_pool(keeps)                                  # raises SystemExit on a leak
    pool = p3.token_pool()
    assert set(p3.ELABORATE_KEY_VALUES) <= set(pool["key_value"]) and set(p3.ELABORATE_VERSIONS) <= set(pool["version"])
    assert not set(p3.ELABORATE_KEY_VALUES) & set(p3.token_pool("professional")["key_value"])


def test_the_concise_pool_and_shorthand_hold_no_eval_string(monkeypatch):
    keeps, _ = p3.eval_strings()
    monkeypatch.setattr(p3, "STYLE", "concise")
    p3.clean_pool(keeps)
    assert set(p3.CONCISE_COMMANDS) <= set(p3.token_pool()["command"])
    assert not set(p3.CONCISE_COMMANDS) & set(p3.token_pool("professional")["command"])
    row = {"row_index": 1, "input": "I don't know, I'm working from home right now", "style": "concise",
           "output": "I don't know; I'm working from home right now.", "cleanup": "KEEP"}
    assert p3.shorthand_row(row, "0" * 64)["input"] == "idk, I'm wfh rn"
    monkeypatch.setattr(p3, "STYLE", "professional")
    assert p3.shorthand_row(row, "0" * 64) is None


def test_number_row_swaps_one_plain_number_in_input_and_target_alike():
    row = _row("the quote was 18000 for 2019 pls confirm", "The quote was 18000 for 2019; please confirm.")
    out = p3.number_row(row, KEY, p3.number_pool())
    new = next(n for n in p3.number_pool() if n in out["input"])
    assert "18000" not in out["input"] + out["output"] and "2019" in out["input"] and "2019" in out["output"]
    assert out["output"] == row["output"].replace("18000", new) and "," not in new
    assert p3.number_row(_row("back in 2019 it was", "Back in 2019, it was."), KEY, p3.number_pool()) is None
    assert p3.number_row(_row("pay 18000 now", "Pay 18,000 now."), KEY, p3.number_pool()) is None
    assert p3.number_row(_row("see v1.18000 and 12:18000", "See v1.18000 and 12:18000."), KEY, p3.number_pool()) is None


def test_the_manifest_check_reads_a_phase_3_manifest(tmp_path):
    for part, name in (("train", "phase_3_exact_copy.jsonl"), ("evals", "phase_3_heldout.jsonl")):
        (tmp_path / part).mkdir()
        (tmp_path / part / name).write_text('{"input": "a"}\n', encoding="utf-8")
    sha = verify_manifest.file_sha256(tmp_path / "train" / "phase_3_exact_copy.jsonl")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "train_files": {"train/phase_3_exact_copy.jsonl": {"rows": 1, "sha256": sha}},
        "eval_files": {"evals/phase_3_heldout.jsonl": {"rows": 1, "sha256": sha}},
        "counts": {"phase_3_exact_copy": 1, "phase_3_heldout": 1}}), encoding="utf-8")
    assert verify_manifest.verify(tmp_path) == "data ok: phase_3_exact_copy 1, phase_3_heldout 1"
