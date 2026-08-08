"""Recurrent point-forecast model compositions."""

from typing import Protocol

import torch
from torch import Tensor, nn

from goldfish.core import ModelOutput
from goldfish.models.components import DeltaNetBackbone, DoublyStochasticMixer, GRUBackbone, HeadLatentCommunication, LSTMBackbone, UnconstrainedMixer


class ForecastBatch(Protocol):
    """Structural batch requirements for numeric point-forecast models."""

    inputs: Tensor


class _RecurrentForecastModel(nn.Module):
    """Encode a history and project its final recurrent state to all horizons."""

    backbone: GRUBackbone | LSTMBackbone | DeltaNetBackbone

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


class ConvLSTMForecastModel(_RecurrentForecastModel):
    """1D-convolutional feature encoder followed by an LSTM forecast model."""

    def __init__(
        self,
        feature_count: int,
        target_count: int,
        horizon_count: int,
        hidden_dim: int,
        *,
        conv_channels: int | None = None,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int | str = "same",
        conv_dilation: int = 1,
        conv_groups: int = 1,
        conv_bias: bool = True,
        downsample_kernel_size: int | None = None,
        downsample_stride: int = 1,
        downsample_padding: int | str = "same",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(feature_count, target_count, horizon_count, hidden_dim)
        conv_channels = hidden_dim if conv_channels is None else conv_channels
        if conv_channels <= 0:
            raise ValueError("conv_channels must be positive")
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if conv_stride <= 0:
            raise ValueError("conv_stride must be positive")
        if conv_dilation <= 0:
            raise ValueError("conv_dilation must be positive")
        if conv_groups <= 0:
            raise ValueError("conv_groups must be positive")
        if feature_count % conv_groups != 0 or conv_channels % conv_groups != 0:
            raise ValueError("conv_groups must divide feature_count and conv_channels")
        if conv_padding == "same" and conv_stride != 1:
            raise ValueError('conv_padding="same" requires conv_stride=1')
        if downsample_kernel_size is not None and downsample_kernel_size <= 0:
            raise ValueError("downsample_kernel_size must be positive")
        if downsample_stride <= 0:
            raise ValueError("downsample_stride must be positive")
        if downsample_padding == "same" and downsample_stride != 1:
            raise ValueError('downsample_padding="same" requires downsample_stride=1')
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.conv = nn.Conv1d(
            in_channels=feature_count,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            bias=conv_bias,
        )
        self.encoder_activation = nn.SiLU()
        self.downsample = (
            nn.Conv1d(
                conv_channels,
                conv_channels,
                kernel_size=downsample_kernel_size,
                stride=downsample_stride,
                padding=downsample_padding,
            )
            if downsample_kernel_size is not None
            else None
        )
        self.backbone = LSTMBackbone(conv_channels, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        if batch.inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        convolved = self.conv(batch.inputs.transpose(1, 2))
        if self.downsample is not None:
            convolved = self.encoder_activation(convolved)
            convolved = self.encoder_activation(self.downsample(convolved))
        states, _ = self.backbone(convolved.transpose(1, 2))
        forecast = self.forecast_head(states[:, -1]).reshape(
            batch.inputs.shape[0], self.horizon_count, self.target_count
        )
        return ModelOutput(predictions={"forecast": forecast}, representations=states)


class LinearLSTMForecastModel(_RecurrentForecastModel):
    """Per-time-step linear feature projection followed by an LSTM forecast model."""

    def __init__(
        self,
        feature_count: int,
        target_count: int,
        horizon_count: int,
        hidden_dim: int,
        *,
        projection_dim: int | None = None,
        projection_bias: bool = True,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(feature_count, target_count, horizon_count, hidden_dim)
        projection_dim = hidden_dim if projection_dim is None else projection_dim
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.projection = nn.Linear(feature_count, projection_dim, bias=projection_bias)
        self.backbone = LSTMBackbone(projection_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        if batch.inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        states, _ = self.backbone(self.projection(batch.inputs))
        forecast = self.forecast_head(states[:, -1]).reshape(
            batch.inputs.shape[0], self.horizon_count, self.target_count
        )
        return ModelOutput(predictions={"forecast": forecast}, representations=states)


class DeltaNetForecastModel(_RecurrentForecastModel):
    """DeltaNet fast-weight encoder with a multi-horizon point-forecast projection head.

    The backbone runs the delta-rule update over the history window; the
    per-position output at the final history position summarizes the written
    memory and is projected to all horizons.
    """

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
        short_conv_kernel: int = 4,
        beta_initial_logit: float = 4.0,
    ) -> None:
        super().__init__(feature_count, target_count, horizon_count, hidden_dim)
        self.backbone = DeltaNetBackbone(
            feature_count,
            hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            short_conv_kernel=short_conv_kernel,
            beta_initial_logit=beta_initial_logit,
        )


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
        num_layers = len(self.head_layers[0])
        for layer_index in range(num_layers):
            head_states = [head_layers[layer_index](states)[0] for head_layers, states in zip(self.head_layers, head_states, strict=True)]
            stacked = torch.stack(head_states, dim=-2)
            # Support optional per-layer distinct mixers
            if hasattr(self, 'mixers'):
                mixed_states = self.mixers[layer_index](stacked)
            else:
                mixed_states = self.mixer(stacked)
            if layer_index + 1 < num_layers:
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
        use_distinct_mixers: bool = False,
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
        mixer_kwargs = dict(
            sinkhorn_iterations=sinkhorn_iterations,
            initialization=mixer_initialization,
            random_std=mixer_random_std,
            uniform_ratio=mixer_uniform_ratio,
        )
        if use_distinct_mixers:
            self.mixers = nn.ModuleList(
                DoublyStochasticMixer(num_heads, **mixer_kwargs)
                for _ in range(num_layers)
            )
            # Backward compatibility: alias self.mixer to the first mixer so
            # existing code/tests that access model.mixer still work.
            self.mixer = self.mixers[0]
        else:
            self.mixer = DoublyStochasticMixer(num_heads, **mixer_kwargs)


class InterLayerCommunicationMultiHeadLSTMForecastModel(_MultiHeadLSTMForecastModel):
    """Parallel LSTM heads with dense, identity-initialized inter-layer communication."""

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
        self.communications = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        )
        for communication in self.communications:
            nn.init.eye_(communication.weight)
            nn.init.zeros_(communication.bias)

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        if batch.inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        head_states = [normalization(projection(batch.inputs)) for projection, normalization in zip(self.input_projections, self.input_normalizations, strict=True)]
        num_layers = len(self.head_layers[0])
        for layer_index in range(num_layers):
            head_states = [head_layers[layer_index](states)[0] for head_layers, states in zip(self.head_layers, head_states, strict=True)]
            stacked = torch.stack(head_states, dim=-2)
            if layer_index + 1 < num_layers:
                communicated = self.communications[layer_index](stacked.flatten(start_dim=-2))
                head_states = list(self.dropout(communicated).unflatten(-1, (self.num_heads, self.head_dim)).unbind(dim=-2))
        representations = self.fusion(stacked.flatten(start_dim=-2))
        forecast = self.forecast_head(representations[:, -1]).reshape(
            batch.inputs.shape[0], self.horizon_count, self.target_count
        )
        return ModelOutput(predictions={"forecast": forecast}, representations=representations)


class LatentCommunicationMultiHeadLSTMForecastModel(_MultiHeadLSTMForecastModel):
    """Parallel LSTM heads with gated latent cross-head communication between layers."""

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
        communication_dim: int | None = None,
        communication_gate_initial_logit: float = -5.0,
    ) -> None:
        if num_heads <= 1:
            raise ValueError("latent communication requires num_heads > 1")
        super().__init__(
            feature_count,
            target_count,
            horizon_count,
            hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.latent_communications = nn.ModuleList(
            HeadLatentCommunication(
                num_heads,
                self.head_dim,
                communication_dim=communication_dim,
                gate_initial_logit=communication_gate_initial_logit,
            )
            for _ in range(num_layers - 1)
        )

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        if batch.inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        head_states = [normalization(projection(batch.inputs)) for projection, normalization in zip(self.input_projections, self.input_normalizations, strict=True)]
        num_layers = len(self.head_layers[0])
        for layer_index in range(num_layers):
            head_states = [head_layers[layer_index](states)[0] for head_layers, states in zip(self.head_layers, head_states, strict=True)]
            stacked = torch.stack(head_states, dim=-2)
            if layer_index + 1 < num_layers:
                communicated = self.latent_communications[layer_index](stacked)
                head_states = list(self.dropout(communicated).unbind(dim=-2))
        representations = self.fusion(stacked.flatten(start_dim=-2))
        forecast = self.forecast_head(representations[:, -1]).reshape(
            batch.inputs.shape[0], self.horizon_count, self.target_count
        )
        return ModelOutput(predictions={"forecast": forecast}, representations=representations)


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
