import pytest
import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models import LatentCommunicationMultiHeadLSTMForecastModel, model_registry


def test_latent_communication_multihead_lstm_is_registered_and_preserves_dimensions() -> None:
    model = model_registry.create(
        "forecast",
        "multihead-lstm-latent-communication",
        feature_count=3,
        target_count=2,
        horizon_count=4,
        hidden_dim=8,
        num_heads=4,
        num_layers=2,
        communication_dim=5,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, LatentCommunicationMultiHeadLSTMForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 8)
    assert len(model.latent_communications) == 1
    assert model.latent_communications[0].communication_dim == 5


def test_latent_communication_multihead_lstm_propagates_gradients() -> None:
    model = LatentCommunicationMultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=4, num_layers=2)
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 2, 1))

    model(batch).predictions["forecast"].square().mean().backward()

    communication = model.latent_communications[0]
    assert communication.routing_logits.grad is not None
    assert communication.gate_logits.grad is not None
    assert all(parameter.grad is not None for parameter in communication.source_encoders.parameters())
    assert all(parameter.grad is not None for parameter in communication.destination_decoders.parameters())


def test_latent_communication_multihead_lstm_has_no_communication_for_one_layer() -> None:
    model = LatentCommunicationMultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=4, num_layers=1)
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 2, 1))

    output = model(batch)

    assert len(model.latent_communications) == 0
    assert output.predictions["forecast"].shape == (2, 2, 1)


def test_latent_communication_multihead_lstm_requires_multiple_heads() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        LatentCommunicationMultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=1, num_layers=2)
