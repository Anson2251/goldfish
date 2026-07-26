"""Task contracts for translating outputs and batches into optimization values."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from torch import Tensor

from .batch import Batch
from .output import ModelOutput


@dataclass
class StepResult:
    """The primary loss and scalar metrics for one batch."""

    loss: Tensor
    metrics: dict[str, Tensor | float]


BatchT = TypeVar("BatchT", bound=Batch, contravariant=True)


@runtime_checkable
class Task(Protocol[BatchT]):
    """Computes a task-specific optimization objective and metrics."""

    def compute(self, output: ModelOutput, batch: BatchT) -> StepResult:
        """Calculate the step result for a model output and its source batch."""
        raise NotImplementedError
