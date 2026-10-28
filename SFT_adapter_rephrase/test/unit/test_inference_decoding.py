"""
The default inference setting exempts the ten digit tokens from the n-gram ban,
the same list the chrome extension's model.json carries.
"""

from __future__ import annotations

import pytest

from src.codec import TextCodec
from src.config import DecodingConfig
from src.inference import DEFAULT_DECODING, RephraseAdapter
from src.interop import DEFAULT_TOKENIZER_DIR

# Ids of "0".."9" in the shipped tokenizer (and in the extension's model.json).
SHIPPED_DIGIT_IDS = (1539, 1540, 1543, 1555, 1560, 1564, 1567, 1569, 1572, 1580)


def _adapter(codec):
    return RephraseAdapter(model=_Noop(), codec=codec, device="cpu")


class _Noop:
    def eval(self):
        return self


def test_default_decoding_exempts_the_digits_and_keeps_everything_else():
    adapter = _adapter(TextCodec(lambda text: [4, 100 + int(text)] if text.isdigit() else [4], lambda ids: "", {"<pad>": 0, "</s>": 1, "<style:concise>": 7, "<text_to_rewrite>": 8, "</text_to_rewrite>": 9}, "<style:concise>"))
    decoding = adapter.default_decoding()
    assert decoding.no_repeat_exempt_tokens == tuple(100 + digit for digit in range(10))
    assert decoding == DecodingConfig(**{**DEFAULT_DECODING.__dict__, "no_repeat_exempt_tokens": decoding.no_repeat_exempt_tokens})
    assert decoding.no_repeat_ngram_size == 3


def test_an_explicit_exempt_list_is_left_alone():
    adapter = _adapter(TextCodec(lambda text: [4, 100 + int(text)] if text.isdigit() else [4], lambda ids: "", {"<pad>": 0, "</s>": 1, "<style:concise>": 7, "<text_to_rewrite>": 8, "</text_to_rewrite>": 9}, "<style:concise>"))
    mine = DecodingConfig(**{**DEFAULT_DECODING.__dict__, "no_repeat_exempt_tokens": (5,)})
    assert adapter.default_decoding(mine).no_repeat_exempt_tokens == (5,)


@pytest.mark.real_tokenizer
def test_real_tokenizer_digit_ids_are_the_ones_in_model_json():
    if not (DEFAULT_TOKENIZER_DIR / "spm.model").is_file():
        pytest.skip("tokenizer artifacts not built")
    adapter = _adapter(TextCodec.from_tokenizer_dir(DEFAULT_TOKENIZER_DIR, "<style:professional>"))
    assert adapter.digit_token_ids == tuple(sorted(SHIPPED_DIGIT_IDS))
