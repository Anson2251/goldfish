import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models import GRUForecastModel, LSTMForecastModel, model_registry


def test_gru_forecast_model_is_registered_and_preserves_forecast_dimensions() -> None:
    model = model_registry.create("forecast", "gru", feature_count=3, target_count=2, horizon_count=4, hidden_dim=5)
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, GRUForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 5)


def test_lstm_forecast_model_is_registered_and_preserves_forecast_dimensions() -> None:
    model = model_registry.create("forecast", "lstm", feature_count=3, target_count=2, horizon_count=4, hidden_dim=5)
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, LSTMForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 5)
