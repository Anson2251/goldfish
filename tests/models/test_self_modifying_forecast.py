from pathlib import Path

import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.config import create_model_from_config, load_model_profile, resolve_model_config
from goldfish.models import RecursiveSelfModifyingForecastModel, model_registry


def make_model() -> RecursiveSelfModifyingForecastModel:
    return RecursiveSelfModifyingForecastModel(
        feature_count=3,
        target_count=2,
        horizon_count=1,
        target_feature_indices=(0, 2),
        hidden_dim=8,
        meta_hidden_dim=16,
        update_scale=0.05,
    )


def test_self_modifying_profile_uses_compact_dynamic_parameter_budget() -> None:
    profile = load_model_profile(Path("model-profiles/forecast/self-modifying-small.yaml"))
    config = resolve_model_config(
        profile,
        family="forecast",
        runtime_parameters={
            "feature_count": 4,
            "target_count": 1,
            "horizon_count": 1,
            "target_feature_indices": (0,),
        },
    )

    model = create_model_from_config(config)

    assert isinstance(model, RecursiveSelfModifyingForecastModel)
    assert model_registry.get("forecast", "self-modifying") is RecursiveSelfModifyingForecastModel
    assert model.parameter_count <= 2_000
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000


def test_self_modifying_forecast_returns_causal_trajectory_and_final_forecast() -> None:
    model = make_model()
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 1, 2))

    output = model(batch)

    assert output.predictions["trajectory_forecast"].shape == (2, 4, 2)
    assert output.predictions["forecast"].shape == (2, 1, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 5, model.parameter_count)
    assert output.diagnostics["parameter_count"] == float(model.parameter_count)
    assert output.diagnostics["mean_update_norm"].item() > 0


def test_self_modifying_forecast_supports_multi_horizon_final_output() -> None:
    model = RecursiveSelfModifyingForecastModel(
        feature_count=4,
        target_count=1,
        horizon_count=3,
        target_feature_indices=(0,),
        hidden_dim=8,
        meta_hidden_dim=16,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 5, 4), targets=torch.randn(2, 3, 1))

    output = model(batch)

    assert output.predictions["trajectory_forecast"].shape == (2, 4, 1)
    assert output.predictions["forecast"].shape == (2, 3, 1)


def test_manual_dynamic_lstm_matches_torch_lstm_cell() -> None:
    model = RecursiveSelfModifyingForecastModel(
        feature_count=2,
        target_count=1,
        horizon_count=1,
        target_feature_indices=(0,),
        hidden_dim=3,
        meta_hidden_dim=4,
    )
    state = torch.randn(2, model.parameter_count)
    inputs = torch.randn(2, 2)
    hidden = torch.randn(2, 3)
    cell = torch.randn(2, 3)
    weights = model._unpack(state)

    expected_gates = (
        (weights.weight_ih * inputs.unsqueeze(1)).sum(dim=-1)
        + (weights.weight_hh * hidden.unsqueeze(1)).sum(dim=-1)
        + weights.bias_ih
        + weights.bias_hh
    )
    input_gate, forget_gate, cell_gate, output_gate = expected_gates.chunk(4, dim=-1)
    expected_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
    expected_hidden = torch.sigmoid(output_gate) * torch.tanh(expected_cell)
    actual_hidden, actual_cell = model._lstm_cell(inputs, hidden, cell, state)

    torch.testing.assert_close(actual_hidden, expected_hidden)
    torch.testing.assert_close(actual_cell, expected_cell)


def test_final_prediction_uses_pre_update_final_state() -> None:
    model = RecursiveSelfModifyingForecastModel(
        feature_count=1,
        target_count=1,
        horizon_count=1,
        target_feature_indices=(0,),
        hidden_dim=1,
        meta_hidden_dim=2,
        update_scale=1.0,
    )
    with torch.no_grad():
        model.theta_initial.zero_()
        for module in (model.observation_encoder, model.meta_network):
            for parameter in module.parameters():
                parameter.zero_()
        model.meta_network[-1].bias.fill_(1.0)
    batch = ForecastBatch(inputs=torch.tensor([[[1.0], [2.0]]]), targets=torch.zeros(1, 1, 1))

    output = model(batch)

    first_update = torch.tanh(torch.ones_like(model.theta_initial))
    expected_hidden, _ = model._lstm_cell(
        torch.tensor([[2.0]]),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        first_update.unsqueeze(0),
    )
    expected_final_prediction = model._predict(expected_hidden, first_update.unsqueeze(0))

    torch.testing.assert_close(output.predictions["trajectory_forecast"], torch.zeros(1, 1, 1))
    torch.testing.assert_close(output.predictions["forecast"], expected_final_prediction.unsqueeze(1))


def test_zero_meta_network_keeps_dynamic_predictor_fixed() -> None:
    model = make_model()
    with torch.no_grad():
        for module in (model.observation_encoder, model.meta_network):
            for parameter in module.parameters():
                parameter.zero_()
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 1, 2))

    output = model(batch)

    assert output.representations is not None
    torch.testing.assert_close(output.representations, output.representations[:, :1].expand_as(output.representations))
    assert output.diagnostics["mean_update_norm"].item() == 0



def test_self_modifying_forecast_backpropagates_through_parameter_updates() -> None:
    model = make_model()
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 1, 2))

    model(batch).predictions["trajectory_forecast"].square().mean().backward()

    assert model.theta_initial.grad is not None
    assert torch.isfinite(model.theta_initial.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.observation_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.meta_network.parameters())
