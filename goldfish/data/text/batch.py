"""Typed batches for causal language modelling."""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor


@dataclass
class LanguageModelBatch:
    """Token inputs, next-token targets, and valid-position mask."""

    input_ids: Tensor
    target_ids: Tensor
    attention_mask: Tensor

    def to(self, device: torch.device, *, non_blocking: bool = False) -> Self:
        return type(self)(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            target_ids=self.target_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
        )


@dataclass
class PrefixLanguageModelBatch:
    """Language-model batch with a mask selecting positions that contribute loss."""

    input_ids: Tensor
    target_ids: Tensor
    attention_mask: Tensor
    loss_mask: Tensor

    def to(self, device: torch.device, *, non_blocking: bool = False) -> Self:
        return type(self)(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            target_ids=self.target_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
            loss_mask=self.loss_mask.to(device, non_blocking=non_blocking),
        )


def collate_language_model_batches(rows: list[LanguageModelBatch]) -> LanguageModelBatch:
    """Stack fixed-length dataset rows into a rank-two language-model batch."""
    if not rows:
        raise ValueError("Cannot collate an empty batch.")
    return LanguageModelBatch(
        input_ids=torch.stack([row.input_ids for row in rows]),
        target_ids=torch.stack([row.target_ids for row in rows]),
        attention_mask=torch.stack([row.attention_mask for row in rows]),
    )


def collate_prefix_language_model_batches(rows: list[PrefixLanguageModelBatch]) -> PrefixLanguageModelBatch:
    """Stack fixed-length paired prefix-language-model rows."""
    if not rows:
        raise ValueError("Cannot collate an empty batch.")
    return PrefixLanguageModelBatch(
        input_ids=torch.stack([row.input_ids for row in rows]),
        target_ids=torch.stack([row.target_ids for row in rows]),
        attention_mask=torch.stack([row.attention_mask for row in rows]),
        loss_mask=torch.stack([row.loss_mask for row in rows]),
    )
