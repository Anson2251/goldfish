from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as functional

from goldfish.core import ModelOutput
from goldfish.tasks import CausalLanguageModelTask


@dataclass
class LanguageModelBatch:
    target_ids: torch.Tensor
    attention_mask: torch.Tensor


def test_compute_uses_only_boolean_masked_token_positions() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -1.0],
                [-2.0, -1.0, 4.0],
            ]
        ],
        requires_grad=True,
    )
    batch = LanguageModelBatch(
        target_ids=torch.tensor([[0, 2, 1]]),
        attention_mask=torch.tensor([[True, False, True]]),
    )

    result = CausalLanguageModelTask().compute(ModelOutput(predictions={"token_logits": logits}), batch)

    expected_loss = functional.cross_entropy(logits[:, [0, 2], :].reshape(-1, 3), torch.tensor([0, 1]))
    assert torch.allclose(result.loss, expected_loss)
    metric_loss = result.metrics["loss"]
    perplexity = result.metrics["perplexity"]
    assert isinstance(metric_loss, torch.Tensor)
    assert isinstance(perplexity, torch.Tensor)
    assert torch.allclose(metric_loss, expected_loss.detach())
    assert torch.allclose(perplexity, torch.exp(expected_loss.detach()))

    result.loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 1, :]) == 0


def test_compute_returns_zero_differentiable_loss_for_all_padding() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    batch = LanguageModelBatch(
        target_ids=torch.tensor([[0, 1, 2], [3, 4, 0]]),
        attention_mask=torch.zeros((2, 3), dtype=torch.bool),
    )

    result = CausalLanguageModelTask().compute(ModelOutput(predictions={"token_logits": logits}), batch)

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


def test_compute_requires_a_boolean_attention_mask() -> None:
    batch = LanguageModelBatch(
        target_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[1, 1]]),
    )
    output = ModelOutput(predictions={"token_logits": torch.randn(1, 2, 3)})

    with pytest.raises(TypeError, match="attention_mask.*boolean"):
        CausalLanguageModelTask().compute(output, batch)


def test_compute_requires_token_logits_prediction() -> None:
    batch = LanguageModelBatch(
        target_ids=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[True, True]]),
    )

    with pytest.raises(KeyError, match="token_logits"):
        CausalLanguageModelTask().compute(ModelOutput(predictions={}), batch)
