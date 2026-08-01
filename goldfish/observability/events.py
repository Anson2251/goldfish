"""Training lifecycle events for hooks and probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

from torch import nn
from torch.optim import Optimizer

from goldfish.core import Batch

if TYPE_CHECKING:
    from goldfish.training.trainer import EpochResult

ProbePhase = Literal["fit_start", "epoch_end", "fit_end"]


@dataclass(frozen=True)
class HookContext:
    """Read-only training state delivered to hooks at a lifecycle event."""

    model: nn.Module
    optimizer: Optimizer
    scheduler: Any | None
    epoch: int | None
    global_step: int
    result: EpochResult | None
    phase: ProbePhase
    reference_batches: Sequence[Batch] | None = None


class TrainingHook(Protocol):
    """A subscriber to trainer lifecycle events."""

    def on_fit_start(self, context: HookContext) -> None: ...

    def on_epoch_end(self, context: HookContext) -> None: ...

    def on_fit_end(self, context: HookContext) -> None: ...
