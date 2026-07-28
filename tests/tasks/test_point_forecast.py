import pytest
import torch

from goldfish.core import ModelOutput
from goldfish.data.numeric import ForecastBatch, StandardNormalizer
from goldfish.tasks import PointForecastTask


def test_point_forecast_optimizes_normalized_mse_and_reports_raw_metrics() -> None:
    normalizer = StandardNormalizer(("open", "close"), torch.tensor([0.0, 10.0], dtype=torch.float64), torch.tensor([1.0, 10.0], dtype=torch.float64))
    batch = ForecastBatch(inputs=torch.zeros((1, 2, 2)), targets=torch.tensor([[[1.0], [2.0]]]))
    predictions = torch.tensor([[[2.0], [0.0]]], requires_grad=True)

    result = PointForecastTask(normalizer, ["close"]).compute(ModelOutput(predictions={"forecast": predictions}), batch)

    assert result.loss.item() == pytest.approx(2.5)
    assert result.metrics["mae"] == pytest.approx(15.0)
    assert result.metrics["rmse"] == pytest.approx((250.0) ** 0.5)
    result.loss.backward()
    assert predictions.grad is not None


def test_point_forecast_requires_matching_forecast_shape() -> None:
    task = PointForecastTask(StandardNormalizer(("close",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)), ["close"])
    batch = ForecastBatch(inputs=torch.zeros((1, 2, 1)), targets=torch.zeros((1, 2, 1)))

    with pytest.raises(ValueError, match="shape"):
        task.compute(ModelOutput(predictions={"forecast": torch.zeros((1, 1, 1))}), batch)
