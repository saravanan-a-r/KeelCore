"""
Decoding for evaluation and inference.

`model/src/generation.py` covers greedy, sampling and beam for checkpoint
diagnostics. What it does not have is n-gram blocking, and this base loops
under plain greedy decoding (checkpoint analysis, watch item 4): half of all
greedy outputs on neutral checkpoints repeat a phrase until the length limit.
A rewrite evaluated under plain greedy would mostly measure that loop, not
the adapter. So greedy decoding here bans, per row, any token that would
complete an n-gram the row has already produced (`no_repeat_ngram_size`,
the standard seq2seq setting; 3 is typical for rewriting).

Beam search delegates to the model package's `beam_generate` unchanged.

Both return `DecodeResult`: the generated IDs without the EOS, and whether
each row actually emitted EOS -- a row that hit the length budget is a
decoding failure, and evaluation counts it as one.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Callable

import torch

from .interop import rephrase_model


# A word the model is writing is a URL once it begins like one: `scheme://`,
# `www.`, or `host.tld/` -- or is about to: the scheme or `www` written so far
# (`https`, `https:`, `www`). `chrome_extension/src/engine/generation/logitsProcessors.ts`
# holds the same pattern and the same tracker; keep the two identical.
URL_WORD = re.compile(
    r"^(?:[a-z][a-z0-9+.\-]*://|www\.|(?:[a-z0-9\-]+\.)+[a-z]{2,}/|(?:https?|ftp)(?::/?)?$|www$)", re.IGNORECASE
)
_BYTE_PIECE = re.compile(r"^<0x([0-9A-F]{2})>$")


def piece_text(piece: str) -> str:
    """A vocabulary piece as plain text: `▁` is a space, a byte piece is its ASCII character (else U+FFFD)."""
    byte = _BYTE_PIECE.match(piece)
    if byte:
        value = int(byte.group(1), 16)
        return chr(value) if value < 0x80 else "\ufffd"
    return piece.replace("\u2581", " ")


class UrlTracker:
    """
    Whether the word being written is a URL, tracked token by token.

    A URL repeats its own pieces on purpose -- the host in two links, `/docs/`
    in a path, `utm_` in each parameter -- so the n-gram ban corrupts it
    (`utm_medium` came out `utMid`). While `inside` is true the ban is lifted;
    the first space ends the word and restores it for the text around the URL.
    """

    def __init__(self, text_of: Callable[[int], str]) -> None:
        self._text_of = text_of
        self._word = ""

    def observe(self, token: int) -> None:
        text = self._text_of(token)
        for index in range(len(text) - 1, -1, -1):
            if text[index].isspace():
                self._word = text[index + 1:]
                return
        self._word += text

    @property
    def inside(self) -> bool:
        return URL_WORD.match(self._word) is not None


@dataclass
class DecodeResult:
    sequences: list[list[int]]
    terminated: list[bool]


def _attended(mask: torch.Tensor | None, row: int):
    return slice(None) if mask is None else mask[row].bool()


def _ngram_counts(ids: list[int], n: int) -> collections.Counter:
    return collections.Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))


@torch.no_grad()
def greedy_decode(
    model,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int,
    no_repeat_ngram_size: int = 0,
    no_repeat_exempt_tokens: tuple[int, ...] = (),
    url_piece: Callable[[int], str] | None = None,
) -> DecodeResult:
    """
    The n-gram ban has two narrowings; the extension's `logitsProcessors.ts`
    implements both (default: the plain ban).

    `no_repeat_exempt_tokens`: a token in the list is never banned. The
    tokenizer writes every digit as its own token, so `50000` is "5 0 0 0 0"
    and repeats the 3-gram "0 0 0" legitimately.

    `url_piece` (token id -> vocabulary piece): while the word being written
    is a URL (`UrlTracker`) an n-gram may repeat, but only as often as it
    occurs in the input -- a link is copied whole, and `/a/a/a/...` cannot
    run on past what was asked.

    Allowing every n-gram that occurs in the input was measured and rejected:
    it brings loops back and rewrites plain text. The URL rule is narrower on
    both counts: it applies only inside a URL, and only up to the input's own count.
    """
    config = model.config
    budget = min(max_new_tokens, config.max_decoder_length - 1)
    batch = encoder_input_ids.shape[0]
    device = encoder_input_ids.device
    n = no_repeat_ngram_size
    exempt = frozenset(no_repeat_exempt_tokens)
    urls = [UrlTracker(lambda token: piece_text(url_piece(token))) if url_piece else None for _ in range(batch)]
    # Per row: how often each n-gram occurs in the input and, so far, in the output (start token included).
    in_input = [_ngram_counts(encoder_input_ids[row][_attended(encoder_attention_mask, row)].tolist(), n) if url_piece and n > 0 else None
                for row in range(batch)]
    produced: list[collections.Counter] = [collections.Counter() for _ in range(batch)]

    encoder_hidden_states = model.encode(encoder_input_ids, encoder_attention_mask)
    next_input = torch.full((batch, 1), config.decoder_start_token_id, dtype=torch.long, device=device)
    cache = None
    generated: list[list[int]] = [[] for _ in range(batch)]
    # The decoder input so far, start token included, as HuggingFace's (and the
    # extension's) processors see it; n-grams are taken over this.
    history: list[list[int]] = [[config.decoder_start_token_id] for _ in range(batch)]
    terminated = [False] * batch
    # Per row: (n-1)-gram prefix -> tokens that have followed it.
    seen: list[dict[tuple[int, ...], set[int]]] = [{} for _ in range(batch)]

    for _ in range(budget):
        output = model(
            encoder_input_ids,
            next_input,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            cache=cache,
            use_cache=True,
        )
        cache = output.cache
        logits = output.logits[:, -1, :].float()

        if n > 0:
            for row in range(batch):
                if terminated[row] or len(history[row]) < n - 1:
                    continue
                prefix = tuple(history[row][len(history[row]) - (n - 1):]) if n > 1 else ()
                banned = seen[row].get(prefix)
                if banned:
                    banned = [token for token in banned if token not in exempt]
                    if urls[row] and urls[row].inside:
                        banned = [token for token in banned if produced[row][prefix + (token,)] >= in_input[row].get(prefix + (token,), 0)]
                    if banned:
                        logits[row, banned] = float("-inf")

        tokens = torch.argmax(logits, dim=-1)
        next_tokens = tokens.tolist()
        for row in range(batch):
            if terminated[row]:
                next_tokens[row] = config.pad_token_id
                continue
            token = next_tokens[row]
            if token == config.eos_token_id:
                terminated[row] = True
                continue
            generated[row].append(token)
            history[row].append(token)
            if urls[row]:
                urls[row].observe(token)
            if n > 0 and len(history[row]) >= n:
                gram = history[row][-n:]
                seen[row].setdefault(tuple(gram[:-1]), set()).add(gram[-1])
                produced[row][tuple(gram)] += 1

        if all(terminated):
            break
        next_input = torch.tensor(next_tokens, dtype=torch.long, device=device)[:, None]

    return DecodeResult(generated, terminated)


@torch.no_grad()
def beam_decode(
    model,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
) -> DecodeResult:
    output = rephrase_model.beam_generate(
        model,
        encoder_input_ids,
        encoder_attention_mask,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
    )
    eos, pad = model.config.eos_token_id, model.config.pad_token_id
    sequences, terminated = [], []
    for row in output.tolist():
        if eos in row:
            sequences.append([t for t in row[: row.index(eos)] if t != pad])
            terminated.append(True)
        else:
            sequences.append([t for t in row if t != pad])
            terminated.append(False)
    return DecodeResult(sequences, terminated)


def decode_batch(
    model, encoder_input_ids, encoder_attention_mask, decoding, *, url_piece: Callable[[int], str] | None = None
) -> DecodeResult:
    """Dispatch on a `DecodingConfig`; `url_piece` is required when it asks to exempt URLs."""
    if decoding.no_repeat_exempt_urls and url_piece is None:
        raise ValueError("decoding.no_repeat_exempt_urls needs url_piece, the tokenizer's id -> piece lookup")
    if decoding.strategy == "beam" and decoding.num_beams > 1:
        return beam_decode(
            model,
            encoder_input_ids,
            encoder_attention_mask,
            max_new_tokens=decoding.max_new_tokens,
            num_beams=decoding.num_beams,
            length_penalty=decoding.length_penalty,
        )
    return greedy_decode(
        model,
        encoder_input_ids,
        encoder_attention_mask,
        max_new_tokens=decoding.max_new_tokens,
        no_repeat_ngram_size=decoding.no_repeat_ngram_size,
        no_repeat_exempt_tokens=decoding.no_repeat_exempt_tokens,
        url_piece=url_piece if decoding.no_repeat_exempt_urls else None,
    )
