"""Tests for the CommunicationStateProbe (parameter tier)."""

from __future__ import annotations

import math

import pytest
import torch

from goldfish.models import (
    InterLayerCommunicationMultiHeadLSTMForecastModel,
    LatentCommunicationMultiHeadLSTMForecastModel,
)
from goldfish.observability.communication import CommunicationStateProbe
from goldfish.observability.events import HookContext

NUM_HEADS = 4
HEAD_DIM = 8
HIDDEN = NUM_HEADS * HEAD_DIM


def _dense_model() -> InterLayerCommunicationMultiHeadLSTMForecastModel:
    return InterLayerCommunicationMultiHeadLSTMForecastModel(
        4, 1, 2, hidden_dim=HIDDEN, num_heads=NUM_HEADS, num_layers=2
    )


def _latent_model() -> LatentCommunicationMultiHeadLSTMForecastModel:
    return LatentCommunicationMultiHeadLSTMForecastModel(
        4, 1, 2, hidden_dim=HIDDEN, num_heads=NUM_HEADS, num_layers=2, communication_dim=8
    )


def _context(model) -> HookContext:
    return HookContext(model=model, optimizer=None, scheduler=None, epoch=0, global_step=0, result=None, phase="epoch_end")  # type: ignore[arg-type]


def test_dense_block_reports_identity_baseline() -> None:
    probe = CommunicationStateProbe({"include": ["communications.*"], "head_dim": HEAD_DIM})
    entry = probe.collect(_context(_dense_model()))["communications"][0]

    assert entry["module"] == "communications.0"
    assert entry["type"] == "dense_linear"
    assert entry["shape"] == [32, 32]
    assert entry["frobenius_distance_to_identity"] == pytest.approx(0.0, abs=1e-6)
    assert entry["block_diagonal_deviation_norm"] == pytest.approx(0.0, abs=1e-6)
    assert entry["block_cross_norm_mean"] == pytest.approx(0.0, abs=1e-6)
    assert entry["bias_norm"] == pytest.approx(0.0, abs=1e-6)


def test_dense_block_partitions_weight_into_head_blocks() -> None:
    model = _dense_model()
    with torch.no_grad():
        # Scale one off-diagonal 8x8 block (dest 0 <- source 1) to 0.5.
        model.communications[0].weight[0:8, 8:16].fill_(0.5)
    probe = CommunicationStateProbe({"include": ["communications.*"], "head_dim": HEAD_DIM})

    entry = probe.collect(_context(model))["communications"][0]

    # One 8x8 block of all 0.5s contributes ||0.5*I_8||_F = 2.0 to the cross norm.
    assert entry["block_cross_norm_max"] == pytest.approx(math.sqrt(8 * 8 * 0.25), rel=1e-6)
    assert entry["block_cross_norm_mean"] > 0.0
    assert entry["block_diagonal_deviation_norm"] < 1e-6  # diagonal blocks untouched


def test_dense_head_dim_falls_back_to_model_attribute() -> None:
    probe = CommunicationStateProbe({"include": ["communications.*"]})
    entry = probe.collect(_context(_dense_model()))["communications"][0]

    assert entry["block_cross_norm_mean"] == pytest.approx(0.0, abs=1e-6)


def test_latent_routing_starts_uniform_over_other_heads() -> None:
    probe = CommunicationStateProbe({"include": ["latent_communications.*"]})
    entry = probe.collect(_context(_latent_model()))["communications"][0]

    assert entry["type"] == "latent_communication"
    routing = entry["routing"]
    assert routing[0][0] == 0.0  # self routes masked
    assert routing[0][1] == pytest.approx(1 / 3, abs=1e-6)
    assert entry["routing_uniformity_distance"] == pytest.approx(0.0, abs=1e-6)
    for entropy in entry["routing_entropy_per_receiver"]:
        assert entropy == pytest.approx(math.log(3), abs=1e-6)


def test_latent_gates_start_at_sigmoid_of_initial_logit() -> None:
    probe = CommunicationStateProbe({"include": ["latent_communications.*"]})
    entry = probe.collect(_context(_latent_model()))["communications"][0]

    assert entry["gate_mean"] == pytest.approx(0.00669, abs=1e-4)
    assert entry["gate_min"] > 0.0
    assert entry["gate_max"] < 0.01


def test_latent_routing_entropy_tracks_nonuniform_routing() -> None:
    model = _latent_model()
    with torch.no_grad():
        # Push receiver 0 to prefer source 1.
        logits = model.latent_communications[0].routing_logits
        logits[0, 1] = 5.0
        logits[0, 2] = -5.0
        logits[0, 3] = -5.0
    probe = CommunicationStateProbe({"include": ["latent_communications.*"]})

    entry = probe.collect(_context(model))["communications"][0]

    entropies = entry["routing_entropy_per_receiver"]
    assert entropies[0] < math.log(3)  # receiver 0 is more peaked
    assert entropies[1] == pytest.approx(math.log(3), abs=1e-6)
    assert entry["routing_uniformity_distance"] > 0.0


def test_latent_encoder_decoder_weight_norms_are_positive() -> None:
    probe = CommunicationStateProbe({"include": ["latent_communications.*"]})
    entry = probe.collect(_context(_latent_model()))["communications"][0]

    assert entry["encoder_weight_norm"] > 0.0
    assert entry["decoder_weight_norm"] > 0.0


def test_grad_norms_cover_all_trainable_parameters() -> None:
    model = _latent_model()
    for parameter in model.latent_communications[0].parameters():
        parameter.grad = torch.full_like(parameter, 1.0)
    probe = CommunicationStateProbe(
        {"include": ["latent_communications.*"], "include_grad_norms": True}
    )

    entry = probe.collect(_context(model))["communications"][0]

    assert entry["grad_norms"]["routing_logits"] > 0.0
    assert entry["grad_norms"]["gate_logits"] > 0.0
    assert any(name.startswith("source_encoders.") for name in entry["grad_norms"])
    assert any(name.startswith("destination_decoders.") for name in entry["grad_norms"])


def test_require_match_behavior() -> None:
    model = _dense_model()
    strict = CommunicationStateProbe({"include": ["telemetry.*"]})
    relaxed = CommunicationStateProbe({"include": ["telemetry.*"], "require_match": False})

    with pytest.raises(ValueError, match="no communication"):
        strict.collect(_context(model))
    assert relaxed.collect(_context(model)) is None


def test_mixed_patterns_collect_both_module_kinds() -> None:
    model = _dense_model()
    probe = CommunicationStateProbe(
        {"include": ["communications.*", "latent_communications.*"], "head_dim": HEAD_DIM}
    )

    payload = probe.collect(_context(model))

    assert [entry["module"] for entry in payload["communications"]] == ["communications.0"]
    assert payload["communications"][0]["type"] == "dense_linear"
