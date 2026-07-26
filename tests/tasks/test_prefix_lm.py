from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as functional

from goldfish.core import ModelOutput
from goldfish.tasks import PrefixLanguageModelTask


@dataclass
class PrefixLanguageModelBatch:
    target_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


def test_compute_uses_only_positions_enabled_by_both_masks() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -1.0],
                [-2.0, -1.0, 4.0],
                [-1.0, 1.0, 2.0],
            ]
        ],
        requires_grad=True,
    )
    batch = PrefixLanguageModelBatch(
        target_ids=torch.tensor([[0, 2, 1, 2]]),
        attention_mask=torch.tensor([[True, True, False, True]]),
        loss_mask=torch.tensor([[True, False, True, True]]),
    )

    result = PrefixLanguageModelTask().compute(
        ModelOutput(predictions={"token_logits": logits}), batch
    )

    expected_loss = functional.cross_entropy(logits[:, [0, 3], :].reshape(-1, 3), torch.tensor([0, 2]))
    assert torch.allclose(result.loss, expected_loss)
    metric_loss = result.metrics["loss"]
    perplexity = result.metrics["perplexity"]
    assert isinstance(metric_loss, torch.Tensor)
    assert isinstance(perplexity, torch.Tensor)
    assert torch.allclose(metric_loss, expected_loss.detach())
    assert torch.allclose(perplexity, torch.exp(expected_loss.detach()))

    result.loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 1:3, :]) == 0


def test_compute_returns_zero_differentiable_loss_when_no_positions_are_enabled() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    batch = PrefixLanguageModelBatch(
        target_ids=torch.tensor([[0, 1, 2], [3, 4, 0]]),
        attention_mask=torch.ones((2, 3), dtype=torch.bool),
        loss_mask=torch.zeros((2, 3), dtype=torch.bool),
    )

    result = PrefixLanguageModelTask().compute(
        ModelOutput(predictions={"token_logits": logits}), batch
    )

    assert result.loss.item() == 0.0
    metric_loss = result.metrics["loss"]
    perplexity = result.metrics["perplexity"]
    assert isinstance(metric_loss, torch.Tensor)
    assert isinstance(perplexity, torch.Tensor)
    assert metric_loss.item() == 0.0
    assert perplexity.item() == 1.0
    result.loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("mask_name", ["attention_mask", "loss_mask"])
def test_compute_requires_boolean_masks(mask_name: str) -> None:
    batch = PrefixLanguageModelBatch(
        target_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[True, True]]),
        loss_mask=torch.tensor([[True, True]]),
    )
    setattr(batch, mask_name, torch.tensor([[1, 1]]))

    with pytest.raises(TypeError, match=rf"{mask_name}.*boolean"):
        PrefixLanguageModelTask().compute(
            ModelOutput(predictions={"token_logits": torch.randn(1, 2, 3)}), batch
        )


@pytest.mark.parametrize("mask_name", ["attention_mask", "loss_mask"])
def test_compute_requires_masks_to_match_target_shape(mask_name: str) -> None:
    batch = PrefixLanguageModelBatch(
        target_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[True, True]]),
        loss_mask=torch.tensor([[True, True]]),
    )
    setattr(batch, mask_name, torch.tensor([[True]]))

    with pytest.raises(ValueError, match=rf"{mask_name} shape.*target_ids"):
        PrefixLanguageModelTask().compute(
            ModelOutput(predictions={"token_logits": torch.randn(1, 2, 3)}), batch
        )


def test_compute_requires_token_logits_prediction() -> None:
    batch = PrefixLanguageModelBatch(
        target_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[True, True]]),
        loss_mask=torch.tensor([[True, True]]),
    )

    with pytest.raises(KeyError, match="token_logits"):
        PrefixLanguageModelTask().compute(ModelOutput(predictions={}), batch)
