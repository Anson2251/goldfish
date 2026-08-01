"""Tests for the ActivationStatsProbe (activation tier)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from goldfish.models import (
    InterLayerCommunicationMultiHeadLSTMForecastModel,
    LatentCommunicationMultiHeadLSTMForecastModel,
    MultiHeadLSTMForecastModel,
)
from goldfish.observability.activation import ActivationStatsProbe
from goldfish.observability.events import HookContext

NUM_HEADS = 4
HEAD_DIM = 8
LOOKBACK = 8
FEATURES = 4


@dataclass
class TensorBatch:
    inputs: torch.Tensor

    def to(self, device: torch.device) -> "TensorBatch":
        return TensorBatch(self.inputs.to(device))


def _batches(count: int = 2) -> list[TensorBatch]:
    return [TensorBatch(torch.randn(2, LOOKBACK, FEATURES)) for _ in range(count)]


def _context(model, batches=_batches()) -> HookContext:
    return HookContext(
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        global_step=0,
        result=None,
        phase="epoch_end",
        reference_batches=batches,
    )  # type: ignore[arg-type]


def _mixer_model() -> MultiHeadLSTMForecastModel:
    return MultiHeadLSTMForecastModel(FEATURES, 1, 2, hidden_dim=32, num_heads=NUM_HEADS, num_layers=2)


def _dense_model() -> InterLayerCommunicationMultiHeadLSTMForecastModel:
    return InterLayerCommunicationMultiHeadLSTMForecastModel(
        FEATURES, 1, 2, hidden_dim=32, num_heads=NUM_HEADS, num_layers=2
    )


def _latent_model() -> LatentCommunicationMultiHeadLSTMForecastModel:
    return LatentCommunicationMultiHeadLSTMForecastModel(
        FEATURES, 1, 2, hidden_dim=32, num_heads=NUM_HEADS, num_layers=2, communication_dim=8
    )


def test_declarative_norm_on_lstm_layer_output() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {
                    "path": "head_layers.*.1",
                    "tensors": [{"name": "output", "stats": ["norm", "mean_abs"]}],
                }
            ]
        }
    )

    payload = probe.collect(_context(_mixer_model()))

    entries = payload["points"]
    assert len(entries) == NUM_HEADS
    for entry in entries:
        assert entry["module"].startswith("head_layers.")
        assert entry["tensors"]["output"]["norm"] > 0.0
        assert math.isfinite(entry["tensors"]["output"]["mean_abs"])


def test_declarative_per_head_reduce_keeps_head_dimension() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {
                    "path": "mixer",
                    "tensors": [{"name": "input", "stats": ["norm"], "reduce": "per_head"}],
                }
            ]
        }
    )

    entry = probe.collect(_context(_mixer_model()))["points"][0]

    assert entry["module"] == "mixer"
    assert len(entry["tensors"]["input"]["norm"]) == NUM_HEADS
    assert all(value > 0.0 for value in entry["tensors"]["input"]["norm"])


def test_declarative_hidden_and_cell_statistics() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {
                    "path": "head_layers.*.0",
                    "tensors": [
                        {"name": "hidden", "stats": ["norm"]},
                        {"name": "cell", "stats": ["norm"]},
                    ],
                }
            ]
        }
    )

    entries = probe.collect(_context(_mixer_model()))["points"]

    for entry in entries:
        assert entry["tensors"]["hidden"]["norm"] > 0.0
        assert entry["tensors"]["cell"]["norm"] > 0.0


def test_mixing_displacement_is_near_zero_for_identity_mixer() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "mixer", "quantity": "mixing-displacement"},
            ]
        }
    )

    entry = probe.collect(_context(_mixer_model()))["points"][0]

    assert entry["quantity"] == "mixing-displacement"
    assert entry["input_norm"] > 0.0
    # The Sinkhorn projection of the identity logits is not exactly I; its
    # off-diagonal mass (~1e-4 per entry) produces a small but nonzero ratio.
    assert entry["displacement_ratio"] < 1e-3
    assert len(entry["displacement_ratio_per_head"]) == NUM_HEADS


def test_dense_displacement_is_near_zero_for_identity_init() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "communications.*", "quantity": "dense-displacement"},
            ]
        }
    )

    entry = probe.collect(_context(_dense_model()))["points"][0]

    assert entry["quantity"] == "dense-displacement"
    assert entry["displacement_ratio"] < 1e-6
    assert entry["output_norm"] > 0.0


def test_message_magnitude_reports_injection_and_decoded_ratios() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "latent_communications.*", "quantity": "message-magnitude"},
            ]
        }
    )

    entry = probe.collect(_context(_latent_model()))["points"][0]

    assert entry["quantity"] == "message-magnitude"
    assert len(entry["injection_ratio_per_receiver"]) == NUM_HEADS
    assert len(entry["decoded_ratio_per_receiver"]) == NUM_HEADS
    for injected, decoded in zip(entry["injection_ratio_per_receiver"], entry["decoded_ratio_per_receiver"]):
        assert 0.0 < injected < decoded  # the gate attenuates the decoded message
    assert entry["injection_ratio_mean"] == pytest.approx(
        sum(entry["injection_ratio_per_receiver"]) / NUM_HEADS, rel=1e-6
    )
    assert entry["gated_message_norm"] > 0.0


def test_io_stats_on_fusion() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "fusion", "quantity": "io-stats"},
            ]
        }
    )

    entry = probe.collect(_context(_mixer_model()))["points"][0]

    assert entry["quantity"] == "io-stats"
    assert entry["input_norm"] > 0.0
    assert entry["output_norm"] > 0.0
    assert entry["input_mean_abs"] > 0.0


def test_statistics_aggregate_across_reference_batches() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {
                    "path": "mixer",
                    "tensors": [{"name": "output", "stats": ["norm"]}],
                }
            ]
        }
    )

    single = probe.collect(_context(_mixer_model(), _batches(1)))["points"][0]
    double = probe.collect(_context(_mixer_model(), _batches(2)))["points"][0]

    assert math.isfinite(single["tensors"]["output"]["norm"])
    assert math.isfinite(double["tensors"]["output"]["norm"])


def test_std_max_p95_are_finite() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {
                    "path": "mixer",
                    "tensors": [{"name": "input", "stats": ["std", "max", "p95"]}],
                }
            ]
        }
    )

    entry = probe.collect(_context(_mixer_model()))["points"][0]

    for stat in ("std", "max", "p95"):
        assert math.isfinite(entry["tensors"]["input"][stat])
        assert entry["tensors"]["input"][stat] > 0.0


def test_restores_training_mode_and_leaves_gradients_untouched() -> None:
    model = _mixer_model()
    model.train()
    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "mixer", "quantity": "mixing-displacement"},
            ]
        }
    )

    probe.collect(_context(model))

    assert model.training is True
    assert model.mixer.logits.grad is None


def test_require_match_behavior() -> None:
    model = _mixer_model()
    strict = ActivationStatsProbe({"points": [{"path": "telemetry.*", "quantity": "io-stats"}]})
    relaxed = ActivationStatsProbe(
        {"points": [{"path": "telemetry.*", "quantity": "io-stats"}], "require_match": False}
    )

    with pytest.raises(ValueError, match="no module"):
        strict.collect(_context(model))
    assert relaxed.collect(_context(model)) is None


def test_without_reference_batches_raises() -> None:
    probe = ActivationStatsProbe(
        {"points": [{"path": "mixer", "quantity": "mixing-displacement"}]}
    )

    with pytest.raises(RuntimeError, match="reference"):
        probe.collect(_context(_mixer_model(), batches=None))  # type: ignore[arg-type]


def test_zero_denominators_record_null_not_nan() -> None:
    from goldfish.models.components import DoublyStochasticMixer

    class ZeroMixerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixer = DoublyStochasticMixer(NUM_HEADS, sinkhorn_iterations=20)

        def forward(self, batch):
            return self.mixer(torch.zeros(batch.inputs.shape[0], LOOKBACK, NUM_HEADS, HEAD_DIM))

    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "mixer", "quantity": "mixing-displacement"},
            ]
        }
    )

    entry = probe.collect(_context(ZeroMixerModel(), batches=_batches(1)))["points"][0]

    assert entry["input_norm"] == 0.0  # absolute statistics remain valid
    assert entry["displacement_ratio"] is None
    assert entry["displacement_ratio_per_head"] == [None, None, None, None]


def test_two_points_on_the_same_module_do_not_collide() -> None:
    probe = ActivationStatsProbe(
        {
            "points": [
                {"path": "mixer", "quantity": "io-stats"},
                {"path": "mixer", "quantity": "mixing-displacement"},
            ]
        }
    )

    entries = probe.collect(_context(_mixer_model()))["points"]

    assert [entry["quantity"] for entry in entries] == ["io-stats", "mixing-displacement"]
    assert "input_norm" in entries[0]
    assert "displacement_ratio" in entries[1]


def test_diagnostics_error_includes_module_path_and_quantity() -> None:
    from torch import nn

    class PlainModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, inputs):
            return inputs

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.plain = PlainModule()

        def forward(self, inputs):
            return self.plain(inputs)

    probe = ActivationStatsProbe(
        {"points": [{"path": "plain", "quantity": "mixing-displacement"}]}
    )

    with pytest.raises(ValueError, match="plain.*diagnostics"):
        probe.collect(_context(Wrapper(), batches=_batches(1)))
