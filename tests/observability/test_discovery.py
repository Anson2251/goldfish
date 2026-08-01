"""Tests for named-pattern module discovery with identity deduplication."""

from __future__ import annotations

from torch import nn

from goldfish.models import (
    InterLayerCommunicationMultiHeadLSTMForecastModel,
    LatentCommunicationMultiHeadLSTMForecastModel,
    MultiHeadLSTMForecastModel,
)
from goldfish.observability.discovery import discover_modules


def _mixer_model(*, distinct: bool = False) -> MultiHeadLSTMForecastModel:
    return MultiHeadLSTMForecastModel(
        4, 1, 2, hidden_dim=32, num_heads=4, num_layers=2, use_distinct_mixers=distinct
    )


def test_discover_shared_mixer_by_name() -> None:
    model = _mixer_model()

    matches = discover_modules(model, ("mixer",))

    assert [path for path, _ in matches] == ["mixer"]
    assert matches[0][1] is model.mixer


def test_discover_distinct_mixers_by_pattern() -> None:
    model = _mixer_model(distinct=True)

    matches = discover_modules(model, ("mixers.*",))

    assert [path for path, _ in matches] == ["mixers.0", "mixers.1"]
    assert [module for _, module in matches] == list(model.mixers)


def test_alias_is_deduplicated_to_canonical_longest_path() -> None:
    model = _mixer_model(distinct=True)

    matches = discover_modules(model, ("mixer", "mixers.*"))

    assert [path for path, _ in matches] == ["mixers.0", "mixers.1"]
    assert len(matches) == 2


def test_pattern_matches_segments_not_partial_paths() -> None:
    model = _mixer_model()

    matches = discover_modules(model, ("mixers.*",))

    assert matches == ()


def test_glob_selects_one_recurrent_layer_across_heads() -> None:
    model = _mixer_model()

    first = discover_modules(model, ("head_layers.*.0",))
    second = discover_modules(model, ("head_layers.*.1",))

    assert [path for path, _ in first] == [f"head_layers.{index}.0" for index in range(4)]
    assert [path for path, _ in second] == [f"head_layers.{index}.1" for index in range(4)]


def test_dense_communication_patterns() -> None:
    model = InterLayerCommunicationMultiHeadLSTMForecastModel(
        4, 1, 2, hidden_dim=32, num_heads=4, num_layers=2
    )

    matches = discover_modules(model, ("communications.*",))

    assert [path for path, _ in matches] == ["communications.0"]
    assert matches[0][1] is model.communications[0]


def test_latent_communication_patterns() -> None:
    model = LatentCommunicationMultiHeadLSTMForecastModel(
        4, 1, 2, hidden_dim=32, num_heads=4, num_layers=2
    )

    matches = discover_modules(model, ("latent_communications.*",))

    assert [path for path, _ in matches] == ["latent_communications.0"]
    assert matches[0][1] is model.latent_communications[0]


def test_no_match_returns_empty() -> None:
    model = _mixer_model()

    assert discover_modules(model, ("telemetry.*",)) == ()


def test_multiple_patterns_union() -> None:
    model = _mixer_model()

    matches = discover_modules(model, ("fusion", "forecast_head"))

    assert [path for path, _ in matches] == ["forecast_head", "fusion"]
    assert all(isinstance(module, nn.Module) for _, module in matches)
