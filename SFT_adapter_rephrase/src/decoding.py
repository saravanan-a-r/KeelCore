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

from dataclasses import dataclass

import torch

from .interop import rephrase_model


@dataclass
class DecodeResult:
    sequences: list[list[int]]
    terminated: list[bool]


@torch.no_grad()
def greedy_decode(
    model,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int,
    no_repeat_ngram_size: int = 0,
    no_repeat_exempt_tokens: tuple[int, ...] = (),
) -> DecodeResult:
    """
    `no_repeat_exempt_tokens` narrows the n-gram ban: a token in the list is
    never banned. The tokenizer writes every digit as its own token, so
    `50000` is "5 0 0 0 0" and repeats the 3-gram "0 0 0" legitimately. The
    extension's `logitsProcessors.ts` implements the same setting (default:
    the plain ban). Allowing n-grams that occur in the input was measured and
    rejected: it brings loops back and rewrites plain text.
    """
    config = model.config
    budget = min(max_new_tokens, config.max_decoder_length - 1)
    batch = encoder_input_ids.shape[0]
    device = encoder_input_ids.device
    n = no_repeat_ngram_size
    exempt = frozenset(no_repeat_exempt_tokens)

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
            if n > 0 and len(history[row]) >= n:
                gram = history[row][-n:]
                seen[row].setdefault(tuple(gram[:-1]), set()).add(gram[-1])

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


def decode_batch(model, encoder_input_ids, encoder_attention_mask, decoding) -> DecodeResult:
    """Dispatch on a `DecodingConfig`."""
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
    )
