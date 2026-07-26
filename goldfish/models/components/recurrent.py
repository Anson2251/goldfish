"""Reusable recurrent sequence backbones."""

from typing import TypeAlias

from torch import Tensor, nn

RecurrentState: TypeAlias = Tensor | tuple[Tensor, Tensor]


class GRUBackbone(nn.Module):
    """Contextualize embedded sequences with a batch-first GRU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.recurrent = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, embedded: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return per-position states and the final recurrent state."""
        _validate_embedded_inputs(embedded)
        return self.recurrent(embedded, hidden_state)


class LSTMBackbone(nn.Module):
    """Contextualize embedded sequences with a batch-first LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.recurrent = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, embedded: Tensor, hidden_state: RecurrentState | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return per-position states and final hidden/cell states."""
        _validate_embedded_inputs(embedded)
        return self.recurrent(embedded, hidden_state)


def _validate_embedded_inputs(embedded: Tensor) -> None:
    if embedded.ndim != 3:
        raise ValueError(f"embedded inputs must have shape [batch, time, features], got {tuple(embedded.shape)}")
