import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models import UnconstrainedMultiHeadLSTMForecastModel, model_registry


def test_unconstrained_multihead_lstm_is_registered_and_preserves_dimensions() -> None:
    model = model_registry.create(
        "forecast",
        "multihead-lstm-unconstrained",
        feature_count=3,
        target_count=2,
        horizon_count=4,
        hidden_dim=8,
        num_heads=4,
        num_layers=2,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, UnconstrainedMultiHeadLSTMForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 8)
    torch.testing.assert_close(model.mixer.mixing_matrix(), torch.eye(4))


def test_unconstrained_multihead_lstm_mixer_can_leave_the_birkhoff_polytope() -> None:
    model = UnconstrainedMultiHeadLSTMForecastModel(3, 1, 2, 8, num_heads=4)
    model.mixer.mixing.weight.data[0, 0] = 2.0

    mixing = model.mixer.mixing_matrix()

    assert mixing.sum(dim=-1)[0] != 1
