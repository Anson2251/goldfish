import pytest
import torch

from goldfish.models import DoublyStochasticMixer


def test_doubly_stochastic_mixer_preserves_shape_and_projects_matrix() -> None:
    mixer = DoublyStochasticMixer(num_channels=4, sinkhorn_iterations=40)
    channels = torch.randn(2, 3, 4, 5)

    mixed = mixer(channels)
    matrix = mixer.mixing_matrix()

    assert mixed.shape == channels.shape
    assert torch.all(matrix >= 0)
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(matrix, torch.eye(4), atol=1e-3, rtol=0)


def test_doubly_stochastic_mixer_propagates_gradients() -> None:
    mixer = DoublyStochasticMixer(num_channels=3)
    channels = torch.randn(2, 3, 4, requires_grad=True)

    mixer(channels).square().mean().backward()

    assert channels.grad is not None
    assert mixer.logits.grad is not None
    assert torch.isfinite(channels.grad).all()
    assert torch.isfinite(mixer.logits.grad).all()


@pytest.mark.parametrize("kwargs", [{"num_channels": 0}, {"num_channels": 2, "sinkhorn_iterations": 0}])
def test_doubly_stochastic_mixer_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DoublyStochasticMixer(**kwargs)


def test_doubly_stochastic_mixer_rejects_incompatible_input() -> None:
    mixer = DoublyStochasticMixer(num_channels=2)

    with pytest.raises(ValueError, match="channels"):
        mixer(torch.randn(3, 2))
