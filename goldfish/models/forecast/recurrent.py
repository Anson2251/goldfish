"""Recurrent point-forecast model compositions."""

from typing import Protocol

from torch import Tensor, nn

from goldfish.core import ModelOutput
from goldfish.models.components import GRUBackbone, LSTMBackbone


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
