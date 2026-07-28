"""Point-forecast objective and raw-unit metrics."""

from collections.abc import Sequence

from torch.nn import functional

from goldfish.core import ModelOutput, StepResult
from goldfish.data.numeric import ForecastBatch, StandardNormalizer


class PointForecastTask:
    """Optimize normalized MSE while reporting MAE and RMSE in raw target units."""

    def __init__(self, normalizer: StandardNormalizer, targets: Sequence[str]) -> None:
        self.normalizer = normalizer
        self.targets = tuple(targets)

    def compute(self, output: ModelOutput, batch: ForecastBatch) -> StepResult:
        try:
            predictions = output.predictions["forecast"]
        except KeyError as error:
            raise KeyError("PointForecastTask requires a 'forecast' prediction.") from error
        if predictions.shape != batch.targets.shape:
            raise ValueError("forecast prediction shape must match normalized target shape.")
        loss = functional.mse_loss(predictions, batch.targets)
        raw_predictions = self.normalizer.inverse_transform_targets(predictions, self.targets)
        raw_targets = self.normalizer.inverse_transform_targets(batch.targets, self.targets)
        raw_error = raw_predictions - raw_targets
        return StepResult(
            loss=loss,
            metrics={"mse": loss.detach(), "mae": raw_error.abs().mean().detach(), "rmse": raw_error.square().mean().sqrt().detach()},
        )
