import pytest
import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models import ConvLSTMForecastModel, GRUForecastModel, LinearLSTMForecastModel, LSTMForecastModel, model_registry


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


def test_conv_lstm_forecast_model_is_registered_and_preserves_sequence_length() -> None:
    model = model_registry.create(
        "forecast",
        "conv-lstm",
        feature_count=3,
        target_count=2,
        horizon_count=4,
        hidden_dim=5,
        conv_channels=7,
        conv_kernel_size=5,
        num_layers=2,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, ConvLSTMForecastModel)
    assert model.conv.out_channels == 7
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 5)


def test_linear_lstm_forecast_model_is_registered_and_projects_features() -> None:
    model = model_registry.create(
        "forecast",
        "linear-lstm",
        feature_count=4,
        target_count=2,
        horizon_count=3,
        hidden_dim=5,
        projection_dim=7,
        num_layers=2,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 4), targets=torch.randn(2, 3, 2))

    output = model(batch)

    assert isinstance(model, LinearLSTMForecastModel)
    assert model.projection.in_features == 4
    assert model.projection.out_features == 7
    assert output.predictions["forecast"].shape == (2, 3, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 5)


def test_linear_lstm_forecast_model_rejects_nonpositive_projection_dimension() -> None:
    with pytest.raises(ValueError, match="projection_dim"):
        LinearLSTMForecastModel(4, 1, 2, 5, projection_dim=0)


def test_conv_lstm_forecast_model_supports_explicit_stride_and_padding() -> None:
    model = ConvLSTMForecastModel(
        3,
        1,
        2,
        5,
        conv_channels=5,
        conv_kernel_size=3,
        conv_stride=2,
        conv_padding=1,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 2, 1))

    output = model(batch)

    assert output.predictions["forecast"].shape == (2, 2, 1)
    assert output.representations is not None
    assert output.representations.shape == (2, 3, 5)


def test_conv_lstm_forecast_model_rejects_same_padding_with_stride() -> None:
    with pytest.raises(ValueError, match="conv_padding"):
        ConvLSTMForecastModel(3, 1, 2, 5, conv_stride=2)


def test_conv_lstm_forecast_model_supports_silu_downsampling_layer() -> None:
    model = ConvLSTMForecastModel(
        3,
        1,
        2,
        5,
        conv_channels=7,
        conv_kernel_size=5,
        downsample_kernel_size=3,
        downsample_stride=2,
        downsample_padding=1,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 2, 1))

    output = model(batch)

    assert isinstance(model.encoder_activation, torch.nn.SiLU)
    assert model.downsample is not None
    assert model.downsample.in_channels == 7
    assert model.downsample.out_channels == 7
    assert output.predictions["forecast"].shape == (2, 2, 1)
    assert output.representations is not None
    assert output.representations.shape == (2, 3, 5)
