"""Tests for the MixerStateProbe (parameter tier)."""

from __future__ import annotations

import math

import pytest
import torch

from goldfish.models import MultiHeadLSTMForecastModel, UnconstrainedMultiHeadLSTMForecastModel
from goldfish.observability.events import HookContext
from goldfish.observability.mixer import MixerStateProbe


def _model(**kwargs) -> MultiHeadLSTMForecastModel:
    return MultiHeadLSTMForecastModel(4, 1, 2, hidden_dim=32, num_heads=4, num_layers=2, **kwargs)


def _context(model) -> HookContext:
    return HookContext(model=model, optimizer=None, scheduler=None, epoch=0, global_step=0, result=None, phase="epoch_end")  # type: ignore[arg-type]


def test_collect_reports_shared_mixer_matrix_and_distances() -> None:
    probe = MixerStateProbe({})
    payload = probe.collect(_context(_model()))

    assert payload is not None
    entry = payload["mixers"][0]
    assert entry["module"] == "mixer"
    assert entry["type"] == "doubly_stochastic"
    assert entry["shape"] == [4, 4]
    assert entry["off_diagonal_mass"] < 1e-3
    assert entry["diagonal_min"] > 0.99
    assert entry["row_sum_max_error"] < 1e-3
    assert entry["column_sum_max_error"] < 1e-3
    assert isinstance(entry["matrix"][0][0], float)


def test_collect_reports_logits_with_identity_initialization() -> None:
    probe = MixerStateProbe({})
    entry = probe.collect(_context(_model()))["mixers"][0]

    assert len(entry["logits"]) == 4
    assert entry["logits"][0][0] > 5.0  # identity strength 10 on the diagonal
    assert abs(entry["logits"][0][1]) < 1.0


def test_logits_distance_to_initial_is_zero_then_grows() -> None:
    model = _model()
    probe = MixerStateProbe({})
    first = probe.collect(_context(model))["mixers"][0]
    assert first["logits_distance_to_initial"] == 0.0

    with torch.no_grad():
        model.mixer.logits.add_(1.0)
    second = probe.collect(_context(model))["mixers"][0]

    assert second["logits_distance_to_initial"] > 1.0


def test_include_matrix_and_logits_can_be_disabled() -> None:
    probe = MixerStateProbe({"include_matrix": False, "include_logits": False})
    entry = probe.collect(_context(_model()))["mixers"][0]

    assert "matrix" not in entry
    assert "logits" not in entry
    assert entry["off_diagonal_mass"] < 1e-3  # derived statistics remain


def test_include_grad_norms_reports_logits_gradient() -> None:
    model = _model()
    model.mixer.logits.grad = torch.full_like(model.mixer.logits, 0.5)
    probe = MixerStateProbe({"include_grad_norms": True})

    entry = probe.collect(_context(model))["mixers"][0]

    assert entry["grad_norms"]["logits"] == pytest.approx(0.5 * 4, rel=1e-6)


def test_grad_norms_are_null_without_gradients() -> None:
    probe = MixerStateProbe({"include_grad_norms": True})
    entry = probe.collect(_context(_model()))["mixers"][0]

    assert entry["grad_norms"]["logits"] is None


def test_unconstrained_mixer_schema() -> None:
    model = UnconstrainedMultiHeadLSTMForecastModel(4, 1, 2, hidden_dim=32, num_heads=4, num_layers=2)
    probe = MixerStateProbe({})

    entry = probe.collect(_context(model))["mixers"][0]

    assert entry["type"] == "unconstrained"
    assert entry["logits"] is None
    assert entry["logits_distance_to_initial"] is None
    assert entry["row_sum_max_error"] is None
    assert entry["column_sum_max_error"] is None
    assert entry["off_diagonal_mass"] < 1e-3  # identity-initialized weight


def test_distinct_mixers_produce_two_entries() -> None:
    probe = MixerStateProbe({})
    payload = probe.collect(_context(_model(use_distinct_mixers=True)))

    assert [entry["module"] for entry in payload["mixers"]] == ["mixers.0", "mixers.1"]


def test_require_match_true_raises_when_nothing_matches() -> None:
    model = _model()
    probe = MixerStateProbe({"include": ["telemetry.*"]})

    with pytest.raises(ValueError, match="no mixer"):
        probe.collect(_context(model))


def test_require_match_false_returns_none_when_nothing_matches() -> None:
    model = _model()
    probe = MixerStateProbe({"include": ["telemetry.*"], "require_match": False})

    assert probe.collect(_context(model)) is None


def test_all_numeric_values_are_json_serializable() -> None:
    probe = MixerStateProbe({"include_grad_norms": True})
    payload = probe.collect(_context(_model()))

    import json

    json.dumps(payload)
    for entry in payload["mixers"]:
        for key, value in entry.items():
            if isinstance(value, float):
                assert math.isfinite(value)


def test_fit_start_refreshes_logits_baseline_for_resumed_runs() -> None:
    model = _model()
    probe = MixerStateProbe({})

    def context(phase: str) -> HookContext:
        return HookContext(model=model, optimizer=None, scheduler=None, epoch=None, global_step=2, result=None, phase=phase)  # type: ignore[arg-type]

    probe.collect(context(phase="fit_start"))
    with torch.no_grad():
        model.mixer.logits.add_(1.0)
    # A resumed fit restarts the baseline at fit_start: distance resets to zero.
    assert probe.collect(context(phase="fit_start"))["mixers"][0]["logits_distance_to_initial"] == 0.0
    # Subsequent epoch records measure from the new baseline.
    with torch.no_grad():
        model.mixer.logits.add_(0.5)
    assert probe.collect(context(phase="epoch_end"))["mixers"][0]["logits_distance_to_initial"] > 0.0


def test_large_mixer_omits_matrix_by_default() -> None:
    from torch import nn

    from goldfish.models.components import DoublyStochasticMixer

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixer = DoublyStochasticMixer(20, initialization="identity", identity_strength=10.0)

    model = Wrapper()
    probe = MixerStateProbe({})
    entry = probe.collect(_context(model))["mixers"][0]
    assert "matrix" not in entry  # N > 16 defaults to summary-only
    assert entry["shape"] == [20, 20]
    assert entry["off_diagonal_mass"] < 0.05  # 19 off-diagonal entries per row of ~4.5e-5

    explicit = MixerStateProbe({"include_matrix": True})
    assert "matrix" in explicit.collect(_context(model))["mixers"][0]
