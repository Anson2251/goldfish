"""Structured model outputs."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class ModelOutput:
    """Predictions and optional intermediate values produced by a model."""

    predictions: dict[str, Tensor]
    representations: Tensor | None = None
    aux_losses: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Tensor | float] = field(default_factory=dict)
