"""Typed batches for point forecasting."""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor


@dataclass(frozen=True)
class ForecastBatch:
    """Normalized feature histories and future target values."""

    inputs: Tensor
    targets: Tensor
    entity_ids: tuple[str, ...] = ()
    cutoff_timestamps: tuple[str, ...] = ()

    def to(self, device: torch.device, *, non_blocking: bool = False) -> Self:
        return type(self)(
            inputs=self.inputs.to(device, non_blocking=non_blocking), targets=self.targets.to(device, non_blocking=non_blocking),
            entity_ids=self.entity_ids, cutoff_timestamps=self.cutoff_timestamps,
        )


def collate_forecast_batches(rows: list[ForecastBatch]) -> ForecastBatch:
    """Stack fixed-shape numeric forecast rows."""
    if not rows:
        raise ValueError("Cannot collate an empty batch.")
    return ForecastBatch(
        inputs=torch.stack([row.inputs for row in rows]),
        targets=torch.stack([row.targets for row in rows]),
        entity_ids=tuple(entity_id for row in rows for entity_id in row.entity_ids),
        cutoff_timestamps=tuple(timestamp for row in rows for timestamp in row.cutoff_timestamps),
    )
