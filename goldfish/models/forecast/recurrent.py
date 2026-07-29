"""Recurrent point-forecast model compositions."""

from typing import Protocol

import torch
from torch import Tensor, nn

from goldfish.core import ModelOutput
from goldfish.models.components import DoublyStochasticMixer, GRUBackbone, LSTMBackbone, UnconstrainedMixer


class ForecastBatch(Protocol):
    """Structural batch requirements for numeric point-forecast models."""

    inputs: Tensor


class _RecurrentForecastModel(nn.Module):
    """Encode a history and project its final recurrent state to all horizons."""

    backbone: GRUBackbone | LSTMBackbone

    def __init__(self, feature_count: int, target_count: int, horizon_count: int, hidden_dim: int) -> None:
        super().__init__()
        if min(feature_count, target_count, horizon_count, hidden_dim) <= 0:
            raise ValueError("forecast model dimensions must be positive")
        self.target_count, self.horizon_count = target_count, horizon_count
        self.forecast_head = nn.Linear(hidden_dim, horizon_count * target_count)

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        if batch.inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        states, _ = self.backbone(batch.inputs)
        forecast = self.forecast_head(states[:, -1]).reshape(batch.inputs.shape[0], self.horizon_count, self.target_count)
        return ModelOutput(predictions={"forecast": forecast}, representations=states)


class GRUForecastModel(_RecurrentForecastModel):
    """GRU encoder with a multi-horizon point-forecast projection head."""

    def __init__(self, feature_count: int, target_count: int, horizon_count: int, hidden_dim: int, *, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__(feature_count, target_count, horizon_count, hidden_dim)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.backbone = GRUBackbone(feature_count, hidden_dim, num_layers=num_layers, dropout=dropout)


class LSTMForecastModel(_RecurrentForecastModel):
    """LSTM encoder with a multi-horizon point-forecast projection head."""

    def __init__(self, feature_count: int, target_count: int, horizon_count: int, hidden_dim: int, *, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__(feature_count, target_count, horizon_count, hidden_dim)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.backbone = LSTMBackbone(feature_count, hidden_dim, num_layers=num_layers, dropout=dropout)


class _MultiHeadLSTMForecastModel(nn.Module):
    """Shared parallel LSTM encoder and head-fusion implementation."""

    mixer: DoublyStochasticMixer | UnconstrainedMixer

    def __init__(
        self,
        feature_count: int,
        target_count: int,
        horizon_count: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(feature_count, target_count, horizon_count, hidden_dim, num_heads, num_layers) <= 0:
            raise ValueError("multi-head LSTM model dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_projections = nn.ModuleList(nn.Linear(feature_count, self.head_dim) for _ in range(num_heads))
        self.input_normalizations = nn.ModuleList(nn.LayerNorm(self.head_dim) for _ in range(num_heads))
        self.head_layers = nn.ModuleList(
            nn.ModuleList(
                nn.LSTM(
                    input_size=self.head_dim,
                    hidden_size=self.head_dim,
                    num_layers=1,
                    batch_first=True,
                )
                for _ in range(num_layers)
            )
            for _ in range(num_heads)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fusion = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.forecast_head = nn.Linear(hidden_dim, horizon_count * target_count)
        self.target_count = target_count
        self.horizon_count = horizon_count

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        if batch.inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        head_states = [normalization(projection(batch.inputs)) for projection, normalization in zip(self.input_projections, self.input_normalizations, strict=True)]
        for layer_index in range(len(self.head_layers[0])):
            head_states = [head_layers[layer_index](states)[0] for head_layers, states in zip(self.head_layers, head_states, strict=True)]
            mixed_states = self.mixer(torch.stack(head_states, dim=-2))
            if layer_index + 1 < len(self.head_layers[0]):
                mixed_states = self.dropout(mixed_states)
            head_states = list(mixed_states.unbind(dim=-2))
        representations = self.fusion(mixed_states.flatten(start_dim=-2))
        forecast = self.forecast_head(representations[:, -1]).reshape(
            batch.inputs.shape[0], self.horizon_count, self.target_count
        )
        return ModelOutput(predictions={"forecast": forecast}, representations=representations)


class MultiHeadLSTMForecastModel(_MultiHeadLSTMForecastModel):
    """Parallel LSTM heads fused by a doubly stochastic channel mixer."""

    def __init__(
        self,
        feature_count: int,
        target_count: int,
        horizon_count: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        sinkhorn_iterations: int = 20,
        mixer_initialization: str = "identity",
        mixer_random_std: float = 1.0,
        mixer_uniform_ratio: float = 0.0,
    ) -> None:
        super().__init__(
            feature_count,
            target_count,
            horizon_count,
            hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mixer = DoublyStochasticMixer(num_heads, sinkhorn_iterations=sinkhorn_iterations, initialization=mixer_initialization, random_std=mixer_random_std, uniform_ratio=mixer_uniform_ratio)


class UnconstrainedMultiHeadLSTMForecastModel(_MultiHeadLSTMForecastModel):
    """Parallel LSTM heads fused by an unconstrained learned linear mixer."""

    def __init__(
        self,
        feature_count: int,
        target_count: int,
        horizon_count: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            feature_count,
            target_count,
            horizon_count,
            hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mixer = UnconstrainedMixer(num_heads)
