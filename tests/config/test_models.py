from pathlib import Path

import pytest

from goldfish.config import create_model_from_config, load_model_profile, resolve_model_config
from goldfish.models import MultiHeadLSTMForecastModel


def test_model_profile_resolves_runtime_dimensions_and_constructs_registered_model(tmp_path: Path) -> None:
    profile_path = tmp_path / "model.yaml"
    profile_path.write_text(
        """\
model:
  family: forecast
  name: multihead-lstm
  parameters:
    hidden_dim: 8
    num_heads: 4
    num_layers: 1
    dropout: 0.0
    sinkhorn_iterations: 2
""",
        encoding="utf-8",
    )

    config = resolve_model_config(
        load_model_profile(profile_path),
        family="forecast",
        runtime_parameters={"feature_count": 3, "target_count": 2, "horizon_count": 4},
    )

    assert config["parameters"]["feature_count"] == 3
    assert isinstance(create_model_from_config(config), MultiHeadLSTMForecastModel)


def test_model_profile_rejects_dataset_derived_parameters(tmp_path: Path) -> None:
    profile_path = tmp_path / "model.yaml"
    profile_path.write_text(
        "model: {family: forecast, name: gru, parameters: {feature_count: 3}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset-derived"):
        resolve_model_config(load_model_profile(profile_path), family="forecast", runtime_parameters={"feature_count": 1})
