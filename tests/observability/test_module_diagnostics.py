"""Tests for the diagnostics() module API (spec 6.3)."""

from __future__ import annotations

import torch

from goldfish.models.components import (
    DoublyStochasticMixer,
    HeadLatentCommunication,
    UnconstrainedMixer,
)


def test_doubly_stochastic_mixer_diagnostics_match_forward() -> None:
    mixer = DoublyStochasticMixer(4, sinkhorn_iterations=20)
    channels = torch.randn(2, 5, 4, 8)

    diagnostics = mixer.diagnostics(channels)

    assert torch.equal(diagnostics["input"], channels)
    assert torch.equal(diagnostics["output"], mixer(channels))
    assert torch.equal(diagnostics["mixing_matrix"], mixer.mixing_matrix())


def test_unconstrained_mixer_diagnostics_match_forward() -> None:
    mixer = UnconstrainedMixer(4)
    channels = torch.randn(2, 5, 4, 8)

    diagnostics = mixer.diagnostics(channels)

    assert torch.equal(diagnostics["output"], mixer(channels))
    assert torch.equal(diagnostics["mixing_matrix"], mixer.mixing_matrix())


def test_latent_diagnostics_shapes_and_forward_consistency() -> None:
    communication = HeadLatentCommunication(4, 8, communication_dim=8, gate_initial_logit=-5.0)
    states = torch.randn(2, 5, 4, 8)

    diagnostics = communication.diagnostics(states)

    assert diagnostics["states"].shape == (2, 5, 4, 8)
    assert diagnostics["latents"].shape == (2, 5, 4, 8)
    assert diagnostics["messages"].shape == (2, 5, 4, 8)
    assert diagnostics["decoded"].shape == (2, 5, 4, 8)
    assert diagnostics["gated_messages"].shape == (2, 5, 4, 8)
    assert torch.equal(diagnostics["states"], states)
    assert torch.allclose(diagnostics["gated_messages"], communication.gates() * diagnostics["decoded"])
    assert torch.allclose(communication(states), states + diagnostics["gated_messages"])


def test_latent_diagnostics_messages_are_routed_convex_combinations() -> None:
    communication = HeadLatentCommunication(4, 8, communication_dim=8)
    states = torch.randn(2, 5, 4, 8)

    diagnostics = communication.diagnostics(states)

    routing = communication.routing_weights()
    for receiver in range(4):
        expected = torch.einsum("j,btjd->btd", routing[receiver], diagnostics["latents"])
        assert torch.allclose(diagnostics["messages"][..., receiver, :], expected)


def test_diagnostics_is_side_effect_free_and_gradient_free() -> None:
    mixer = DoublyStochasticMixer(4)
    communication = HeadLatentCommunication(4, 8)
    channels = torch.randn(2, 5, 4, 8)

    before_forward = mixer(channels)
    before_logits = mixer.logits.detach().clone()

    with torch.no_grad():
        diagnostics = mixer.diagnostics(channels)
        communication.diagnostics(channels)

    assert diagnostics["output"].requires_grad is False
    assert torch.equal(mixer(channels), before_forward)
    assert torch.equal(mixer.logits.detach(), before_logits)
