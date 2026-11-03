"""
Decoding: plain greedy must agree with the model package's own greedy, and
n-gram blocking must actually block.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from conftest import build_tiny_model
from src.decoding import UrlTracker, decode_batch, greedy_decode, piece_text
from src.inference import DEFAULT_DECODING
from src.interop import rephrase_model


def _inputs(batch: int = 4, length: int = 10, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 80, (batch, length), generator=generator)
    mask = torch.ones_like(ids)
    mask[0, -3:] = 0
    ids[0, -3:] = 0
    return ids, mask


def test_unblocked_greedy_matches_the_model_packages_greedy():
    """Same argmax path, so the adapter's evaluation is not a different decoder."""
    model = build_tiny_model()
    ids, mask = _inputs()
    ours = greedy_decode(model, ids, mask, max_new_tokens=30, no_repeat_ngram_size=0)
    reference = rephrase_model.greedy_generate(model, ids, mask, max_new_tokens=30).tolist()
    for row, (sequence, done) in enumerate(zip(ours.sequences, ours.terminated)):
        expected = reference[row]
        if 1 in expected:
            assert done and sequence == expected[: expected.index(1)]
        else:
            assert not done and sequence == expected


def test_ngram_blocking_never_repeats_an_ngram():
    """A random tiny model loops readily -- exactly what blocking must stop."""
    model = build_tiny_model(seed=3)
    ids, mask = _inputs(batch=6, seed=5)
    for n in (2, 3):
        result = greedy_decode(model, ids, mask, max_new_tokens=60, no_repeat_ngram_size=n)
        for sequence in result.sequences:
            grams = [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
            assert len(grams) == len(set(grams))


def test_rows_that_hit_the_budget_are_reported_as_unterminated():
    model = build_tiny_model()
    ids, mask = _inputs()
    result = greedy_decode(model, ids, mask, max_new_tokens=1, no_repeat_ngram_size=0)
    for sequence, done in zip(result.sequences, result.terminated):
        assert done == (len(sequence) == 0)


class _ScriptedModel:
    """A model that wants the scripted token at each step (second choice: id 2), so a ban shows."""

    def __init__(self, script, vocab=12):
        self.script, self.vocab = script, vocab
        self.config = type("C", (), {"decoder_start_token_id": 0, "eos_token_id": 1, "pad_token_id": 0, "max_decoder_length": 2048})()

    def encode(self, ids, mask):
        return None

    def __call__(self, encoder_ids, next_input, encoder_attention_mask=None, encoder_hidden_states=None, cache=None, use_cache=True):
        position = cache or 0
        logits = torch.zeros(next_input.shape[0], 1, self.vocab)
        logits[:, 0, 2] = 0.5
        logits[:, 0, self.script[position] if position < len(self.script) else 1] = 1.0
        return type("O", (), {"logits": logits, "cache": position + 1})()


def _scripted(script, input_ids, **kwargs):
    ids = torch.tensor([input_ids])
    result = greedy_decode(_ScriptedModel(script), ids, torch.ones_like(ids), max_new_tokens=12, **kwargs)
    return result.sequences[0]


def test_an_exempt_token_may_repeat_while_other_repeats_stay_blocked():
    script = [7, 7, 7, 7, 7, 1]
    assert _scripted(script, [9], no_repeat_ngram_size=3)[:5] != [7] * 5
    assert _scripted(script, [9], no_repeat_ngram_size=3, no_repeat_exempt_tokens=(7,)) == [7] * 5
    # a token that is not exempt is still blocked
    assert _scripted([6, 6, 6, 6, 6, 1], [9], no_repeat_ngram_size=3, no_repeat_exempt_tokens=(7,))[:5] != [6] * 5


def test_defaults_are_the_plain_ban_on_a_real_looping_model():
    model = build_tiny_model(seed=3)
    ids, mask = _inputs(batch=6, seed=5)
    plain = greedy_decode(model, ids, mask, max_new_tokens=60, no_repeat_ngram_size=3)
    same = greedy_decode(model, ids, mask, max_new_tokens=60, no_repeat_ngram_size=3, no_repeat_exempt_tokens=())
    assert plain.sequences == same.sequences


# --- URLs: the ban is lifted while a URL is being written --------------------------------------

PIECES = {3: "▁https", 4: "://", 5: "ex", 6: ".com", 7: "/a", 8: "▁and", 9: "▁the", 10: "▁www", 11: "<0xC3>"}


def _piece(token):
    return PIECES.get(token, "")


def test_piece_text_maps_spaces_and_bytes():
    from src.decoding import piece_text

    assert piece_text("▁and") == " and"
    assert piece_text("<0x2F>") == "/"
    assert piece_text("<0xC3>") == "�"
    assert piece_text("://") == "://"


@pytest.mark.parametrize(
    "tokens, inside",
    [
        ([3], True),                   # "https": a URL is about to start
        ([3, 4], True),                # "https://"
        ([3, 4, 5, 6, 7], True),
        ([3, 4, 5, 8], False),         # a space ends the URL
        ([3, 4, 5, 8, 3, 4], True),    # and a second one starts
        ([10], True),                  # "www": the same
        ([8], False),                  # "and"
        ([10, 6], True),               # "www.com"
        ([5, 6, 7], True),             # "ex.com/a": host.tld/ is a URL
        ([5, 6], False),               # a bare domain has no path yet
        ([9, 5], False),               # "the ex": ordinary words
    ],
)
def test_url_tracker(tokens, inside):
    tracker = UrlTracker(lambda token: piece_text(_piece(token)))
    for token in tokens:
        tracker.observe(token)
    assert tracker.inside is inside


URL_TWICE = [3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 1]


LINKS_IN = URL_TWICE[:-1]  # an input that holds both links


def test_a_url_may_repeat_its_pieces_when_urls_are_exempt():
    plain = _scripted(URL_TWICE, LINKS_IN, no_repeat_ngram_size=3)
    assert plain != URL_TWICE[:-1]  # the plain ban rewrites the second link
    assert _scripted(URL_TWICE, LINKS_IN, no_repeat_ngram_size=3, url_piece=_piece) == URL_TWICE[:-1]


def test_a_url_repeats_only_as_often_as_the_input_has_it():
    once = [3, 4, 5, 6, 7]
    runaway = once + [6, 7, 6, 7, 6, 7, 6, 7, 1]  # keeps going with "/a" after the link
    out = _scripted(runaway, once, no_repeat_ngram_size=3, url_piece=_piece)
    assert out != runaway[:-1]
    assert out[:5] == once


def test_a_link_the_input_does_not_have_is_not_exempt():
    assert _scripted(URL_TWICE, [9], no_repeat_ngram_size=3, url_piece=_piece) != URL_TWICE[:-1]


def test_plain_text_keeps_the_ban_when_urls_are_exempt():
    looping = [8, 9, 8, 9, 8, 9, 8, 9, 1]
    assert _scripted(looping, looping, no_repeat_ngram_size=3, url_piece=_piece) == _scripted(looping, looping, no_repeat_ngram_size=3)


def test_url_pieces_off_is_the_plain_ban():
    assert _scripted(URL_TWICE, LINKS_IN, no_repeat_ngram_size=3, url_piece=None) == _scripted(URL_TWICE, LINKS_IN, no_repeat_ngram_size=3)


def test_decode_batch_needs_the_piece_lookup_for_url_exemption():
    ids = torch.tensor([LINKS_IN])
    decoding = replace(DEFAULT_DECODING, no_repeat_exempt_urls=True)
    with pytest.raises(ValueError, match="url_piece"):
        decode_batch(_ScriptedModel(URL_TWICE), ids, torch.ones_like(ids), decoding)
    result = decode_batch(_ScriptedModel(URL_TWICE), ids, torch.ones_like(ids), decoding, url_piece=_piece)
    assert result.sequences[0] == URL_TWICE[:-1]
