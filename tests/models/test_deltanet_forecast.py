import pytest
import torch
from torch import Tensor

from goldfish.data.numeric import ForecastBatch
from goldfish.models import DeltaNetForecastModel, model_registry
from goldfish.models.components import DeltaNetBackbone, delta_rule_scan


def test_deltanet_forecast_is_registered_and_preserves_dimensions() -> None:
    model = model_registry.create(
        "forecast",
        "deltanet",
        feature_count=3,
        target_count=2,
        horizon_count=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
    )
    batch = ForecastBatch(inputs=torch.randn(2, 6, 3), targets=torch.randn(2, 4, 2))

    output = model(batch)

    assert isinstance(model, DeltaNetForecastModel)
    assert output.predictions["forecast"].shape == (2, 4, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 6, 8)
    assert isinstance(model.backbone, DeltaNetBackbone)
    assert len(model.backbone.layers) == 2


def test_deltanet_forecast_propagates_gradients() -> None:
    model = DeltaNetForecastModel(3, 1, 2, 8, num_heads=4, num_layers=2, short_conv_kernel=3)
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 2, 1))

    model(batch).predictions["forecast"].square().mean().backward()

    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads.values())
    # Gradients must reach the delta-rule weights of every stacked layer.
    assert "backbone.layers.0.qkv.weight" in grads
    assert "backbone.layers.1.qkv.weight" in grads
    assert "backbone.layers.0.beta_logits" in grads
    assert "backbone.layers.1.beta_logits" in grads
    assert "forecast_head.weight" in grads


def test_deltanet_rejects_indivisible_hidden_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DeltaNetForecastModel(3, 1, 2, 7, num_heads=4)


def test_deltanet_rejects_nonpositive_short_conv_kernel() -> None:
    with pytest.raises(ValueError, match="short_conv_kernel"):
        DeltaNetForecastModel(3, 1, 2, 8, short_conv_kernel=0)


def test_delta_rule_scan_recalls_written_associations() -> None:
    """With unit-norm orthogonal keys and beta=1, the memory stores each key-value pair exactly."""
    head_dim = 4
    keys = torch.eye(head_dim)[:2]  # orthonormal k1, k2
    values = torch.tensor([[2.0, -1.0, 0.5, 3.0], [-4.0, 0.0, 1.0, 2.0]])
    q = torch.stack([keys[0], keys[0]])  # query the first association after both writes
    k = keys
    v = values

    outputs, memory = delta_rule_scan(
        q.view(1, 2, 1, head_dim),
        k.view(1, 2, 1, head_dim),
        v.view(1, 2, 1, head_dim),
        torch.tensor([1.0]),
        return_memory=True,
    )

    # o_t = S_t q_t uses the state after the t-th write; querying k1 returns v1.
    torch.testing.assert_close(outputs[0, 1, 0], values[0])
    # The memory itself stores both associations: S k_i = v_i.
    torch.testing.assert_close(torch.einsum("ij,j->i", memory[0, 0], keys[0]), values[0])
    torch.testing.assert_close(torch.einsum("ij,j->i", memory[0, 0], keys[1]), values[1])


def test_delta_rule_scan_with_zero_beta_is_a_noop() -> None:
    """beta=0 keeps the memory untouched, so all outputs are zero."""
    q = torch.randn(1, 3, 1, 4)
    k = torch.randn(1, 3, 1, 4)
    v = torch.randn(1, 3, 1, 4)

    outputs = delta_rule_scan(q, k, v, torch.zeros(1))

    assert isinstance(outputs, Tensor)
    torch.testing.assert_close(outputs, torch.zeros_like(outputs))
