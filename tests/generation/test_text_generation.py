

import torch
from torch import nn

from goldfish.core import ModelOutput
from goldfish.generation import generate_text


class Tokenizer:
    eos_token_id = 3

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class StatefulModel(nn.Module):
    vocab_size = 4
    eos_token_id = 3

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def forward_tokens(
        self, input_ids: torch.Tensor, hidden_state: torch.Tensor | None = None
    ) -> tuple[ModelOutput, torch.Tensor]:
        self.calls.append((input_ids.clone(), hidden_state))
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), -100.0)
        next_token = 2 if len(self.calls) == 1 else self.eos_token_id
        logits[..., next_token] = 100.0
        return ModelOutput(predictions={"token_logits": logits}), torch.tensor(len(self.calls))



def test_generate_text_reuses_state_and_stops_at_eos() -> None:
    model = StatefulModel()
    model.train()

    generated = generate_text(model, Tokenizer(), "0 1", max_new_tokens=5)

    assert generated == "0 1 2 3"
    assert model.training
    assert len(model.calls) == 2
    torch.testing.assert_close(model.calls[0][0], torch.tensor([[0, 1]]))
    torch.testing.assert_close(model.calls[1][0], torch.tensor([[2]]))
    assert model.calls[0][1] is None
    assert model.calls[1][1] is not None


def test_generate_text_accepts_sampling_options() -> None:
    model = StatefulModel()

    token_ids = generate_text(
        model,
        Tokenizer(),
        "0",
        max_new_tokens=1,
        temperature=0.7,
        top_k=2,
        return_token_ids=True,
    )

    assert token_ids == [0, 2]
