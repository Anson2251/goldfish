"""Trajectory objective for recursive self-modifying point forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional

from goldfish.core import ModelOutput, StepResult
from goldfish.data.numeric import ForecastBatch, StandardNormalizer


class SelfModifyingForecastTask:
    """Weight later within-window one-step predictions and the final forecast."""

    def __init__(
        self,
        normalizer: StandardNormalizer,
        targets: Sequence[str],
        target_feature_indices: Sequence[int],
        *,
        final_loss_weight: float = 1.0,
    ) -> None:
        if len(targets) != len(target_feature_indices):
            raise ValueError("target_feature_indices must match target count")
        if not target_feature_indices or len(set(target_feature_indices)) != len(target_feature_indices):
            raise ValueError("target_feature_indices must be non-empty and distinct")
        if any(index < 0 or index >= len(normalizer.features) for index in target_feature_indices):
            raise ValueError("target_feature_indices must be valid feature positions")
        if final_loss_weight < 0:
            raise ValueError("final_loss_weight must be non-negative")
        self.normalizer = normalizer
        self.targets = tuple(targets)
        self.target_feature_indices = tuple(target_feature_indices)
        self.final_loss_weight = final_loss_weight

    def compute(self, output: ModelOutput, batch: ForecastBatch) -> StepResult:
        try:
            trajectory = output.predictions["trajectory_forecast"]
            forecast = output.predictions["forecast"]
        except KeyError as error:
            raise KeyError("SelfModifyingForecastTask requires 'trajectory_forecast' and 'forecast' predictions.") from error
        if batch.inputs.ndim != 3 or batch.inputs.shape[1] < 2:
            raise ValueError("self-modifying forecast inputs must have shape [batch, lookback >= 2, feature_count]")
        if forecast.ndim != 3:
            raise ValueError("self-modifying final forecast must have shape [batch, horizons, targets]")
        if forecast.shape != batch.targets.shape:
            raise ValueError("final forecast prediction shape must match normalized target shape")
        target_count = len(self.targets)
        expected_trajectory_shape = (batch.inputs.shape[0], batch.inputs.shape[1] - 1, target_count)
        if trajectory.shape != expected_trajectory_shape:
            raise ValueError(f"trajectory forecast must have shape {expected_trajectory_shape}")
        if forecast.shape[-1] != target_count:
            raise ValueError("final forecast target width does not match configured targets")

        trajectory_targets = batch.inputs[:, 1:, list(self.target_feature_indices)]
        step_mae = (trajectory - trajectory_targets).abs().mean(dim=-1)
        weights = torch.arange(1, step_mae.shape[1] + 1, device=step_mae.device, dtype=step_mae.dtype) / step_mae.shape[1]
        trajectory_loss = (step_mae * weights).sum(dim=-1).mean()
        final_loss = functional.l1_loss(forecast, batch.targets)
        loss = trajectory_loss + self.final_loss_weight * final_loss

        raw_predictions = self.normalizer.inverse_transform_targets(forecast, self.targets)
        raw_targets = self.normalizer.inverse_transform_targets(batch.targets, self.targets)
        raw_error = raw_predictions - raw_targets
        return StepResult(
            loss=loss,
            metrics={
                "trajectory_mae": trajectory_loss.detach(),
                "final_mae": final_loss.detach(),
                "mae": raw_error.abs().mean().detach(),
                "rmse": raw_error.square().mean().sqrt().detach(),
            },
        )
