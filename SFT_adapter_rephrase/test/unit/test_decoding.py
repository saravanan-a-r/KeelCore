"""
Decoding: plain greedy must agree with the model package's own greedy, and
n-gram blocking must actually block.
"""

from __future__ import annotations

import torch

from conftest import build_tiny_model
from src.decoding import greedy_decode
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
