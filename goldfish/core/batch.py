"""Modality-agnostic batch contract."""

from typing import Protocol, Self, runtime_checkable

import torch


@runtime_checkable
class Batch(Protocol):
    """A structured batch that can move all of its tensors to a device."""

    def to(self, device: torch.device) -> Self:
        """Return this batch with every tensor moved to ``device``."""
        raise NotImplementedError
