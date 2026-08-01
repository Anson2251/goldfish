import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models import InterLayerCommunicationMultiHeadLSTMForecastModel, model_registry


def test_inter_layer_communication_multihead_lstm_is_registered_and_identity_initialized() -> None:
    model = model_registry.create(
        "forecast",
        "multihead-lstm-communication",
        feature_count=3,
        target_count=2,
        horizon_count=4,
        hidden_dim=8,
        num_heads=4,
        num_layers=2,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, InterLayerCommunicationMultiHeadLSTMForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 8)
    assert len(model.communications) == 1

    states = torch.randn(2, 6, 8)
    torch.testing.assert_close(model.communications[0](states), states, atol=0, rtol=0)


def test_inter_layer_communication_multihead_lstm_propagates_gradients() -> None:
    model = InterLayerCommunicationMultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=4, num_layers=2)
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 2, 1))

    model(batch).predictions["forecast"].square().mean().backward()

    communication = model.communications[0]
    assert communication.weight.grad is not None
    assert communication.bias.grad is not None
    assert torch.isfinite(communication.weight.grad).all()
    assert torch.isfinite(communication.bias.grad).all()


def test_inter_layer_communication_multihead_lstm_has_no_communication_for_one_layer() -> None:
    model = InterLayerCommunicationMultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=4, num_layers=1)
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 2, 1))

    output = model(batch)

    assert len(model.communications) == 0
    assert output.predictions["forecast"].shape == (2, 2, 1)
