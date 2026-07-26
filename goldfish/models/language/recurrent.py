"""Recurrent language-model compositions."""

from typing import Protocol

from torch import Tensor, nn

from goldfish.core import ModelOutput
from goldfish.models.components.recurrent import GRUBackbone, LSTMBackbone, RecurrentState


class TokenBatch(Protocol):
    """Structural protocol for batches accepted by language models."""

    input_ids: Tensor
    attention_mask: Tensor


class _RecurrentLanguageModel(nn.Module):
    backbone: GRUBackbone | LSTMBackbone

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocabulary_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, batch: TokenBatch) -> ModelOutput:
        """Produce vocabulary logits for every input token position."""
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        _validate_token_batch(input_ids, attention_mask)
        output, _ = self.forward_tokens(input_ids)
        return output

    def forward_tokens(
        self, input_ids: Tensor, hidden_state: RecurrentState | None = None
    ) -> tuple[ModelOutput, RecurrentState]:
        """Process token IDs and optionally continue from a recurrent state."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        embedded = self.embedding(input_ids)
        if isinstance(self.backbone, GRUBackbone):
            states, final_state = self.backbone(embedded, hidden_state)
        else:
            states, final_state = self.backbone(embedded, hidden_state)
        return (
            ModelOutput(
                predictions={"token_logits": self.vocabulary_head(states)},
                representations=states,
            ),
            final_state,
        )


class GRULanguageModel(_RecurrentLanguageModel):
    """Token embedding, reusable GRU backbone, and vocabulary projection."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size, embedding_dim, hidden_dim)
        self.backbone = GRUBackbone(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)


class LSTMLanguageModel(_RecurrentLanguageModel):
    """Token embedding, reusable LSTM backbone, and vocabulary projection."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(vocab_size, embedding_dim, hidden_dim)
        self.backbone = LSTMBackbone(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)


def _validate_token_batch(input_ids: Tensor, attention_mask: Tensor) -> None:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
