"""Causal language-model loss and metrics."""

from typing import Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor

from goldfish.core import ModelOutput, StepResult


class CausalLanguageModelBatch(Protocol):
    """Structural batch requirements for causal language modelling."""

    target_ids: Tensor
    attention_mask: Tensor


class CausalLanguageModelTask:
    """Computes masked next-token cross-entropy and perplexity."""

    def compute(self, output: ModelOutput, batch: CausalLanguageModelBatch) -> StepResult:
        """Compute loss over valid target positions identified by a boolean mask."""
        try:
            logits = output.predictions["token_logits"]
        except KeyError as error:
            raise KeyError("CausalLanguageModelTask requires a 'token_logits' prediction.") from error

        targets = batch.target_ids
        attention_mask = batch.attention_mask
        self._validate_inputs(logits, targets, attention_mask)

        valid_logits = logits[attention_mask]
        valid_targets = targets[attention_mask]
        if valid_targets.numel() == 0:
            loss = logits.sum() * 0.0
        else:
            loss = functional.cross_entropy(valid_logits, valid_targets)

        metric_loss = loss.detach()
        return StepResult(
            loss=loss,
            metrics={"loss": metric_loss, "perplexity": torch.exp(metric_loss)},
        )

    @staticmethod
    def _validate_inputs(logits: Tensor, targets: Tensor, attention_mask: Tensor) -> None:
        if logits.ndim != 3:
            raise ValueError("token_logits must have shape [batch, time, vocabulary_size].")
        if targets.ndim != 2:
            raise ValueError("target_ids must have shape [batch, time].")
        if attention_mask.dtype is not torch.bool:
            raise TypeError("attention_mask must be a boolean tensor.")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, time].")
        if logits.shape[:2] != targets.shape:
            raise ValueError("token_logits batch/time dimensions must match target_ids.")
        if attention_mask.shape != targets.shape:
            raise ValueError("attention_mask shape must match target_ids.")
        if targets.dtype != torch.long:
            raise TypeError("target_ids must have dtype torch.long.")
