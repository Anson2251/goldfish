import pytest
import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models import MultiHeadLSTMForecastModel, model_registry


def test_multihead_lstm_forecast_is_registered_and_preserves_dimensions() -> None:
    model = model_registry.create(
        "forecast",
        "multihead-lstm",
        feature_count=3,
        target_count=2,
        horizon_count=4,
        hidden_dim=8,
        num_heads=4,
        num_layers=2,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, MultiHeadLSTMForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 8)
    mixing = model.mixer.mixing_matrix()
    torch.testing.assert_close(mixing.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(mixing.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=0)
    assert len(model.head_layers) == 4
    assert all(len(head_layers) == 2 for head_layers in model.head_layers)


def test_multihead_lstm_forecast_propagates_gradients() -> None:
    model = MultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=4)
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 2, 1))

    model(batch).predictions["forecast"].square().mean().backward()

    assert model.mixer.logits.grad is not None
    assert torch.isfinite(model.mixer.logits.grad).all()
    assert all(
        layer.weight_ih_l0.grad is not None and torch.isfinite(layer.weight_ih_l0.grad).all()
        for head_layers in model.head_layers
        for layer in head_layers
    )


def test_multihead_lstm_rejects_indivisible_hidden_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadLSTMForecastModel(3, 1, 2, 7, num_heads=4)
