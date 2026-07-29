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


def test_doubly_stochastic_mixer_random_init_is_not_identity() -> None:
    """With random initialization, the initial matrix should not be identity."""
    torch.manual_seed(42)
    mixer = DoublyStochasticMixer(num_channels=4, initialization="random", random_std=1.0)
    matrix = mixer.mixing_matrix()

    assert torch.all(matrix >= 0)
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=0)
    assert not torch.allclose(matrix, torch.eye(4), atol=1e-2), "random init should not be identity"


def test_doubly_stochastic_mixer_random_init_has_correct_properties() -> None:
    """Verify that random-init mixer still satisfies double-stochastic constraints."""
    torch.manual_seed(7)
    mixer = DoublyStochasticMixer(num_channels=8, initialization="random", random_std=2.0)
    matrix = mixer.mixing_matrix()

    assert matrix.shape == (8, 8)
    assert torch.all(matrix >= 0)
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(8), atol=1e-4, rtol=0)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(8), atol=1e-4, rtol=0)


def test_doubly_stochastic_mixer_rejects_invalid_initialization() -> None:
    with pytest.raises(ValueError, match="initialization"):
        DoublyStochasticMixer(num_channels=4, initialization="invalid")


@pytest.mark.parametrize("kwargs", [{"num_channels": 0}, {"num_channels": 2, "sinkhorn_iterations": 0}, {"num_channels": 2, "initialization": "random", "random_std": 0.0}, {"num_channels": 2, "initialization": "uniform", "uniform_ratio": -0.1}, {"num_channels": 2, "initialization": "uniform", "uniform_ratio": 1.1}])
def test_doubly_stochastic_mixer_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DoublyStochasticMixer(**kwargs)


def test_doubly_stochastic_mixer_rejects_incompatible_input() -> None:
    mixer = DoublyStochasticMixer(num_channels=2)

    with pytest.raises(ValueError, match="channels"):
        mixer(torch.randn(3, 2))


def test_doubly_stochastic_mixer_uniform_init_ratio_zero_is_identity() -> None:
    mixer = DoublyStochasticMixer(num_channels=4, initialization="uniform", uniform_ratio=0.0)
    torch.testing.assert_close(mixer.mixing_matrix(), torch.eye(4), atol=1e-5, rtol=0)


def test_doubly_stochastic_mixer_uniform_init_is_doubly_stochastic() -> None:
    for ratio in (0.1, 0.25, 0.5, 0.75, 1.0):
        mixer = DoublyStochasticMixer(num_channels=4, initialization="uniform", uniform_ratio=ratio)
        P = mixer.mixing_matrix()
        assert torch.all(P >= 0)
        torch.testing.assert_close(P.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
        torch.testing.assert_close(P.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=0)
        torch.testing.assert_close(P.diag(), torch.full((4,), 1 - ratio), atol=1e-5, rtol=0)


def test_doubly_stochastic_mixer_uniform_init_ratio_half_is_not_identity() -> None:
    mixer = DoublyStochasticMixer(num_channels=4, initialization="uniform", uniform_ratio=0.5)
    P = mixer.mixing_matrix()
    assert not torch.allclose(P, torch.eye(4), atol=1e-2)
    torch.testing.assert_close(P, P.T, atol=1e-5, rtol=0)


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.1])
def test_doubly_stochastic_mixer_uniform_init_small_ratio_is_near_identity(ratio: float) -> None:
    mixer = DoublyStochasticMixer(num_channels=4, initialization="uniform", uniform_ratio=ratio)
    P = mixer.mixing_matrix()
    torch.testing.assert_close(P, torch.eye(4), atol=ratio + 1e-5, rtol=0)
