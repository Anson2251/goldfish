import pytest
import torch

from goldfish.core import ModelOutput
from goldfish.data.numeric import ForecastBatch, StandardNormalizer
from goldfish.tasks import SelfModifyingForecastTask


def test_self_modifying_task_weights_later_trajectory_predictions_and_final_prediction() -> None:
    normalizer = StandardNormalizer(("a", "b"), torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    batch = ForecastBatch(
        inputs=torch.tensor([[[0.0, 10.0], [1.0, 20.0], [3.0, 30.0], [6.0, 40.0]]]),
        targets=torch.tensor([[[10.0]]]),
    )
    output = ModelOutput(
        predictions={
            "trajectory_forecast": torch.tensor([[[2.0], [5.0], [5.0]]], requires_grad=True),
            "forecast": torch.tensor([[[8.0]]], requires_grad=True),
        }
    )

    result = SelfModifyingForecastTask(normalizer, ["a"], [0], final_loss_weight=2.0).compute(output, batch)

    assert result.metrics["trajectory_mae"] == pytest.approx(8.0 / 3.0)
    assert result.metrics["final_mae"] == pytest.approx(2.0)
    assert result.loss.item() == pytest.approx(20.0 / 3.0)


def test_self_modifying_task_preserves_prediction_gradients() -> None:
    prediction = torch.zeros(1, 2, 1, requires_grad=True)
    final = torch.zeros(1, 1, 1, requires_grad=True)
    task = SelfModifyingForecastTask(
        StandardNormalizer(("a",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
        ["a"], [0],
    )
    result = task.compute(
        ModelOutput(predictions={"trajectory_forecast": prediction, "forecast": final}),
        ForecastBatch(inputs=torch.tensor([[[0.0], [1.0], [2.0]]]), targets=torch.tensor([[[3.0]]])),
    )

    result.loss.backward()

    assert prediction.grad is not None
    assert final.grad is not None


def test_self_modifying_task_accepts_multi_horizon_final_forecast() -> None:
    normalizer = StandardNormalizer(("a",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    task = SelfModifyingForecastTask(normalizer, ["a"], [0])
    batch = ForecastBatch(
        inputs=torch.tensor([[[0.0], [1.0], [3.0]]]),
        targets=torch.tensor([[[4.0], [5.0], [6.0]]]),
    )
    output = ModelOutput(
        predictions={
            "trajectory_forecast": torch.tensor([[[1.0], [2.0]]], requires_grad=True),
            "forecast": torch.tensor([[[3.0], [7.0], [6.0]]], requires_grad=True),
        }
    )

    result = task.compute(output, batch)

    assert result.metrics["trajectory_mae"] == pytest.approx(1.0)
    assert result.metrics["final_mae"] == pytest.approx(1.0)
    assert result.loss.item() == pytest.approx(2.0)


def test_self_modifying_task_rejects_final_forecast_with_wrong_target_shape() -> None:
    task = SelfModifyingForecastTask(
        StandardNormalizer(("a",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
        ["a"], [0],
    )
    batch = ForecastBatch(inputs=torch.zeros(1, 3, 1), targets=torch.zeros(1, 2, 1))
    output = ModelOutput(predictions={"trajectory_forecast": torch.zeros(1, 2, 1), "forecast": torch.zeros(1, 1, 1)})

    with pytest.raises(ValueError, match="shape"):
        task.compute(output, batch)
