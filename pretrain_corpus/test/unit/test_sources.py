from __future__ import annotations

import pytest

from src.errors import SourceConfigError
from src.sources import (
    SOURCES,
    format_cheat_sheet,
    format_tldr_page,
    resolve,
)


def test_resolve_known_source_returns_its_spec():
    spec = resolve("fineweb_edu")
    assert spec.hf_dataset == "HuggingFaceFW/fineweb-edu"


def test_resolve_unknown_source_raises():
    with pytest.raises(SourceConfigError):
        resolve("not_a_real_source")


def test_exactly_the_seven_commercially_licensed_sources_are_registered():
    assert set(SOURCES) == {
        "fineweb_edu", "c4", "cosmopedia", "gutenberg_pg19", "permissive_code",
        "cli_tldr", "cli_cheat_sheets",
    }


def test_every_source_records_its_license():
    for source_id, spec in SOURCES.items():
        assert spec.source_id == source_id
        assert spec.license, source_id


def test_cli_tldr_is_registered_alongside_the_five_large_corpora():
    """The CLI-helper corpus (architecture.md §9.1's later addition), kept
    as its own test so a regression here doesn't get lost inside the
    full-set assertion above."""
    assert "cli_tldr" in SOURCES
    assert SOURCES["cli_tldr"].kind == "git_tldr"


def test_code_source_disables_quality_and_pii_masking():
    spec = SOURCES["permissive_code"]
    assert spec.pipeline.quality is False
    assert spec.pipeline.mask_pii is False


def test_prose_sources_keep_quality_filtering_and_pii_masking():
    for source_id in ("fineweb_edu", "c4", "cosmopedia", "gutenberg_pg19"):
        spec = SOURCES[source_id]
        assert spec.kind == "hf"
        assert spec.pipeline.quality is True
        assert spec.pipeline.mask_pii is True


def test_cli_tldr_source_disables_quality_and_pii_masking():
    """
    Same reasoning as permissive_code: short structured command records, not
    prose, and no user-submitted text to mask.
    """
    spec = SOURCES["cli_tldr"]
    assert spec.pipeline.quality is False
    assert spec.pipeline.mask_pii is False
    assert spec.license == "CC BY 4.0"


def test_cli_cheat_sheets_is_registered_without_prose_filters():
    spec = SOURCES["cli_cheat_sheets"]
    assert spec.kind == "git_cheat"
    assert spec.pipeline.quality is False
    assert spec.pipeline.mask_pii is False
    assert spec.license.startswith("CC0-1.0")


def test_format_tldr_page_pairs_each_description_with_its_command():
    raw = (
        "# tar\n\n"
        "> Archiving utility.\n"
        "> More information: <https://example.com>.\n\n"
        "- Create an archive:\n\n"
        "`tar cf target.tar file1 file2`\n\n"
        "- Extract an archive:\n\n"
        "`tar xf source.tar`\n"
    )
    text = format_tldr_page("tar", raw)
    assert text.startswith("tar: Archiving utility.")
    assert "Create an archive\ntar cf target.tar file1 file2" in text
    assert "Extract an archive\ntar xf source.tar" in text


def test_format_tldr_page_with_no_examples_yields_nothing():
    """A page with only a description and no `- ...` / backtick pairs is
    not useful CLI-helper training data, and must not turn into an empty
    or malformed chunk downstream."""
    raw = "# foo\n\n> Just a description, no examples.\n"
    assert format_tldr_page("foo", raw) == ""


def test_format_cheat_sheet_pairs_comment_blocks_with_code_blocks():
    raw = (
        "# To extract an archive:\n\n"
        "tar xf archive.tar\n\n"
        "# To create an archive:\n\n"
        "tar cf archive.tar files\n"
    )
    text = format_cheat_sheet("tar", raw)
    assert text.startswith("tar: To extract an archive:")
    assert "tar xf archive.tar" in text
    assert "tar cf archive.tar files" in text


def test_format_cheat_sheet_with_no_code_yields_nothing():
    raw = "# Just a comment, no command lines.\n"
    assert format_cheat_sheet("nothing", raw) == ""
