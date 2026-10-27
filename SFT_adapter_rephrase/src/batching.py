"""
Batch order and padding.

One optimizer step = one *step batch* of `effective_batch_size` examples.
The step batch is then cut into micro-batches for gradient accumulation:
`batch_size` rows each, or -- with `micro_batch_tokens` -- as many rows as fit
a padded-token budget (target tokens weighted by `target_token_weight`, as
their vocabulary-wide logits dominate memory), so a step of short rows runs as one large micro-batch
and a step of long rows as several small ones. Either split sums to the same
gradient (the loss is normalized by the step's tokens, see `trainer.py`). Planning at step granularity is what makes the data order
device-independent: a CPU run (16 x 2) and a GPU run (32 x 1) see the same
examples at the same optimizer step, so a run started on the CPU can be
resumed on the GPU, and the two runs' curves are directly comparable.

Length grouping
---------------
Each epoch: shuffle all examples, cut the shuffled order into windows of
`window` step batches, sort each window by length, cut it into step batches,
then shuffle the batches. Examples in a batch have similar lengths, so
padding waste is small (it dominates CPU cost), while the epoch as a whole is
still in random order rather than sorted short-to-long.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch

from .codec import Frame
from .data import SFTExample


class StepBatchPlan:
    """Deterministic per-epoch plan of step batches (lists of example indices)."""

    def __init__(self, lengths: Sequence[int], step_batch_size: int, window: int, seed: int) -> None:
        if step_batch_size < 1 or window < 1:
            raise ValueError("step_batch_size and window must be >= 1")
        if not lengths:
            raise ValueError("cannot plan batches over zero examples")
        self.lengths = list(lengths)
        self.step_batch_size = step_batch_size
        self.window = window
        self.seed = seed

    @property
    def steps_per_epoch(self) -> int:
        return -(-len(self.lengths) // self.step_batch_size)

    def epoch(self, epoch: int) -> list[list[int]]:
        rng = random.Random(f"{self.seed}:{epoch}")
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        chunk = self.step_batch_size * self.window
        batches: list[list[int]] = []
        for start in range(0, len(order), chunk):
            window = sorted(order[start:start + chunk], key=lambda i: (self.lengths[i], i))
            batches.extend(
                window[offset:offset + self.step_batch_size]
                for offset in range(0, len(window), self.step_batch_size)
            )
        rng.shuffle(batches)
        return batches


class StepBatchStream:
    """
    An endless stream of step batches over epochs, with a resumable position.

    `position` is `(epoch, index within epoch)` of the *next* batch.
    """

    def __init__(self, plan: StepBatchPlan) -> None:
        self.plan = plan
        self.epoch = 0
        self.index = 0
        self._cached_epoch: int | None = None
        self._batches: list[list[int]] = []

    def next(self) -> list[int]:
        if self._cached_epoch != self.epoch:
            self._batches = self.plan.epoch(self.epoch)
            self._cached_epoch = self.epoch
        batch = self._batches[self.index]
        self.index += 1
        if self.index == len(self._batches):
            self.epoch, self.index = self.epoch + 1, 0
        return batch

    @property
    def fractional_epoch(self) -> float:
        return self.epoch + self.index / self.plan.steps_per_epoch

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "index": self.index}

    def load_state_dict(self, state: dict) -> None:
        epoch, index = int(state["epoch"]), int(state["index"])
        if index < 0 or index >= self.plan.steps_per_epoch:
            raise ValueError(f"batch index {index} out of range for {self.plan.steps_per_epoch} steps/epoch")
        self.epoch, self.index = epoch, index


def split_micro_batches(indices: Sequence[int], batch_size: int) -> list[list[int]]:
    return [list(indices[i:i + batch_size]) for i in range(0, len(indices), batch_size)]


def split_by_token_budget(
    indices: Sequence[int], shapes: Sequence[tuple[int, int]], budget: int, target_weight: float = 1.0
) -> list[list[int]]:
    """
    Cut a step batch into micro-batches whose padded cost, rows x (longest
    source + `target_weight` x longest target), stays within `budget`.
    `shapes[i]` is example i's (source, target) length. Rows are packed shortest first so
    each micro-batch holds similar lengths; a single row longer than the
    budget still gets a micro-batch of its own. Deterministic.
    """
    order = sorted(indices, key=lambda i: shapes[i][0] + target_weight * shapes[i][1])
    micro_batches: list[list[int]] = []
    current: list[int] = []
    longest_source = longest_target = 0
    for i in order:
        source = max(longest_source, shapes[i][0])
        target = max(longest_target, shapes[i][1])
        if current and (len(current) + 1) * (source + target_weight * target) > budget:
            micro_batches.append(current)
            current, (source, target) = [], shapes[i]
        current.append(i)
        longest_source, longest_target = source, target
    if current:
        micro_batches.append(current)
    return micro_batches


@dataclass
class PaddedBatch:
    encoder_input_ids: torch.Tensor
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor

    @property
    def target_tokens(self) -> int:
        return int((self.labels != -100).sum())

    @property
    def real_tokens(self) -> int:
        return int(self.encoder_attention_mask.sum() + self.decoder_attention_mask.sum())

    @property
    def padded_tokens(self) -> int:
        return self.encoder_attention_mask.numel() + self.decoder_attention_mask.numel()

    def to(self, device: torch.device) -> "PaddedBatch":
        return PaddedBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


def collate(examples: Sequence[SFTExample], frame: Frame, label_pad_id: int = -100) -> PaddedBatch:
    """
    Pad a list of examples. Teacher forcing: the decoder reads the target
    shifted right behind the start token, so position i never sees label i.
    """
    batch = len(examples)
    source_length = max(len(e.encoder_input_ids) for e in examples)
    target_length = max(len(e.target_ids) for e in examples)

    encoder_ids = torch.full((batch, source_length), frame.pad_id, dtype=torch.long)
    encoder_mask = torch.zeros((batch, source_length), dtype=torch.long)
    decoder_ids = torch.full((batch, target_length), frame.pad_id, dtype=torch.long)
    decoder_mask = torch.zeros((batch, target_length), dtype=torch.long)
    labels = torch.full((batch, target_length), label_pad_id, dtype=torch.long)

    for row, example in enumerate(examples):
        source, target = example.encoder_input_ids, example.target_ids
        encoder_ids[row, :len(source)] = torch.tensor(source)
        encoder_mask[row, :len(source)] = 1
        shifted = (frame.decoder_start_id, *target[:-1])
        decoder_ids[row, :len(target)] = torch.tensor(shifted)
        decoder_mask[row, :len(target)] = 1
        labels[row, :len(target)] = torch.tensor(target)

    return PaddedBatch(encoder_ids, encoder_mask, decoder_ids, decoder_mask, labels)


def example_length(example: SFTExample) -> int:
    """What length grouping sorts by: the padded cost of an example."""
    return len(example.encoder_input_ids) + len(example.target_ids)
