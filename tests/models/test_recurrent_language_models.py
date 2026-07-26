from dataclasses import dataclass

import pytest
import torch

from goldfish.core import ModelOutput
from goldfish.models import (
    GRUBackbone,
    GRULanguageModel,
    LSTMBackbone,
    LSTMLanguageModel,
)


@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@pytest.mark.parametrize("backbone_class", [GRUBackbone, LSTMBackbone])
def test_recurrent_backbone_returns_contextual_states_and_final_state(
    backbone_class: type[GRUBackbone] | type[LSTMBackbone],
) -> None:
    backbone = backbone_class(input_dim=3, hidden_dim=5, num_layers=2, dropout=0.1)
    embedded = torch.randn(4, 6, 3)

    states, final_state = backbone(embedded)

    assert states.shape == (4, 6, 5)
    if isinstance(final_state, tuple):
        assert final_state[0].shape == (2, 4, 5)
        assert final_state[1].shape == (2, 4, 5)
    else:
        assert final_state.shape == (2, 4, 5)


@pytest.mark.parametrize("model_class", [GRULanguageModel, LSTMLanguageModel])
def test_language_model_returns_core_output_and_reuses_recurrent_state(
    model_class: type[GRULanguageModel] | type[LSTMLanguageModel],
) -> None:
    model = model_class(vocab_size=11, embedding_dim=4, hidden_dim=7, num_layers=2)
    batch = TextBatch(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[True, True, True], [True, True, False]]),
    )

    output = model(batch)
    step_output, state = model.forward_tokens(batch.input_ids[:, :2])
    final_output, _ = model.forward_tokens(batch.input_ids[:, 2:], hidden_state=state)

    assert isinstance(output, ModelOutput)
    assert set(output.predictions) == {"token_logits"}
    assert output.predictions["token_logits"].shape == (2, 3, 11)
    assert output.representations is not None
    assert output.representations.shape == (2, 3, 7)
    torch.testing.assert_close(
        torch.cat((step_output.predictions["token_logits"], final_output.predictions["token_logits"]), dim=1),
        output.predictions["token_logits"],
    )


@pytest.mark.parametrize("model_class", [GRULanguageModel, LSTMLanguageModel])
def test_language_model_rejects_invalid_batch_shapes(
    model_class: type[GRULanguageModel] | type[LSTMLanguageModel],
) -> None:
    model = model_class(vocab_size=5, embedding_dim=3, hidden_dim=4)
    batch = TextBatch(input_ids=torch.tensor([1, 2]), attention_mask=torch.tensor([True, True]))

    with pytest.raises(ValueError, match="input_ids"):
        model(batch)
