import math

import pytest
import torch

from goldfish.models.components import HeadLatentCommunication


def test_head_latent_communication_masks_self_routes_and_normalizes_receivers() -> None:
    communication = HeadLatentCommunication(num_heads=4, head_dim=3, communication_dim=5)

    weights = communication.routing_weights()

    torch.testing.assert_close(weights.diag(), torch.zeros(4), atol=0, rtol=0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(weights[~torch.eye(4, dtype=torch.bool)], torch.full((12,), 1 / 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(communication.gates(), torch.full((4, 3), 1 / (1 + math.exp(5))), atol=1e-7, rtol=0)


def test_head_latent_communication_preserves_shape_and_propagates_gradients() -> None:
    communication = HeadLatentCommunication(num_heads=4, head_dim=3, communication_dim=5)
    states = torch.randn(2, 6, 4, 3, requires_grad=True)

    output = communication(states)
    output.square().mean().backward()

    assert output.shape == states.shape
    assert states.grad is not None
    assert communication.routing_logits.grad is not None
    assert communication.gate_logits.grad is not None
    assert all(parameter.grad is not None for parameter in communication.source_encoders.parameters())
    assert all(parameter.grad is not None for parameter in communication.destination_decoders.parameters())


def test_head_latent_communication_is_exact_identity_when_gates_are_closed() -> None:
    communication = HeadLatentCommunication(num_heads=4, head_dim=3)
    communication.gate_logits.data.fill_(-torch.inf)
    states = torch.randn(2, 6, 4, 3)

    torch.testing.assert_close(communication(states), states, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 1, "head_dim": 3},
        {"num_heads": 4, "head_dim": 0},
        {"num_heads": 4, "head_dim": 3, "communication_dim": 0},
        {"num_heads": 4, "head_dim": 3, "gate_initial_logit": float("inf")},
    ],
)
def test_head_latent_communication_rejects_invalid_configuration(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        HeadLatentCommunication(**kwargs)
