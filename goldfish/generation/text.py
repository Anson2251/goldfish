"""Autoregressive text generation for recurrent language models."""

from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor, nn

from goldfish.core import ModelOutput
from goldfish.models.components import RecurrentState


class TextTokenizer(Protocol):
    """Minimal tokenizer interface required for text generation."""

    eos_token_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class RecurrentLanguageModel(Protocol):
    """Inference interface shared by recurrent language-model compositions."""

    def forward_tokens(
        self, input_ids: Tensor, hidden_state: RecurrentState | None = None
    ) -> tuple[ModelOutput, RecurrentState]: ...


def generate_text(
    model: nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float | None = None,
    top_k: int | None = None,
    prefix_token_ids: Sequence[int] = (),
    eos_token_id: int | None = None,
    return_token_ids: bool = False,
) -> str | list[int]:
    """Generate from ``prompt`` with greedy or temperature/top-k decoding.

    The prompt is processed once; each subsequent call feeds only the newly sampled
    token along with the previous recurrent state.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    token_ids = list(tokenizer.encode(prompt)) + list(prefix_token_ids)
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")

    eos_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    module = model
    device = next(module.parameters()).device
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            model_output, state = model.forward_tokens(
                torch.tensor([token_ids], dtype=torch.long, device=device)
            )
            logits = model_output.predictions["token_logits"][:, -1, :]

            for _ in range(max_new_tokens):
                next_token = _select_token(logits, temperature=temperature, top_k=top_k)
                token_id = int(next_token.item())
                token_ids.append(token_id)
                if token_id == eos_id:
                    break

                model_output, state = model.forward_tokens(next_token.unsqueeze(1), hidden_state=state)
                logits = model_output.predictions["token_logits"][:, -1, :]
    finally:
        module.train(was_training)

    return token_ids if return_token_ids else tokenizer.decode(token_ids)


def _select_token(logits: Tensor, *, temperature: float | None, top_k: int | None) -> Tensor:
    if temperature is None:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    if top_k is not None:
        k = min(top_k, scaled_logits.shape[-1])
        threshold = torch.topk(scaled_logits, k, dim=-1).values[..., -1, None]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, -torch.inf)
    return torch.multinomial(torch.softmax(scaled_logits, dim=-1), num_samples=1).squeeze(-1)
