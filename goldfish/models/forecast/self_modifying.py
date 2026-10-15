"""Causal self-modifying numeric forecast model."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from goldfish.core import ModelOutput

if TYPE_CHECKING:
    from .recurrent import ForecastBatch


@dataclass(frozen=True)
class _DynamicLSTMParameters:
    """Batched, unpacked LSTM cell and direct forecast head parameters."""

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    head_weight: Tensor
    head_bias: Tensor


class _DynamicLSTMLayout:
    """Pack and unpack the complete mutable state of an LSTM forecaster."""

    def __init__(self, feature_count: int, hidden_dim: int, output_dim: int) -> None:
        self.feature_count = feature_count
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_ih_start = 0
        self.weight_hh_start = 4 * hidden_dim * feature_count
        self.bias_ih_start = self.weight_hh_start + 4 * hidden_dim * hidden_dim
        self.bias_hh_start = self.bias_ih_start + 4 * hidden_dim
        self.head_weight_start = self.bias_hh_start + 4 * hidden_dim
        self.head_bias_start = self.head_weight_start + output_dim * hidden_dim
        self.parameter_count = self.head_bias_start + output_dim

    def initial_parameters(self) -> Tensor:
        """Return a standard LSTM-like initialized state and zero forecast bias."""
        values = torch.empty(self.parameter_count)
        bound = 1.0 / sqrt(self.hidden_dim)
        values[self.weight_ih_start : self.weight_hh_start].uniform_(-bound, bound)
        values[self.weight_hh_start : self.bias_ih_start].uniform_(-bound, bound)
        values[self.bias_ih_start : self.bias_hh_start].uniform_(-bound, bound)
        values[self.bias_hh_start : self.head_weight_start].uniform_(-bound, bound)
        head_weight = values[self.head_weight_start : self.head_bias_start].view(self.output_dim, self.hidden_dim)
        nn.init.uniform_(head_weight, -sqrt(6.0 / (self.hidden_dim + self.output_dim)), sqrt(6.0 / (self.hidden_dim + self.output_dim)))
        values[self.head_bias_start :].zero_()
        return values

    def unpack(self, state: Tensor) -> _DynamicLSTMParameters:
        """View packed per-batch parameter states as named LSTM and head tensors."""
        batch_size = state.shape[0]
        return _DynamicLSTMParameters(
            weight_ih=state[:, self.weight_ih_start : self.weight_hh_start].view(batch_size, 4 * self.hidden_dim, self.feature_count),
            weight_hh=state[:, self.weight_hh_start : self.bias_ih_start].view(batch_size, 4 * self.hidden_dim, self.hidden_dim),
            bias_ih=state[:, self.bias_ih_start : self.bias_hh_start],
            bias_hh=state[:, self.bias_hh_start : self.head_weight_start],
            head_weight=state[:, self.head_weight_start : self.head_bias_start].view(batch_size, self.output_dim, self.hidden_dim),
            head_bias=state[:, self.head_bias_start :],
        )


class RecursiveSelfModifyingForecastModel(nn.Module):
    """Forecaster with dynamic LSTM and direct forecast-head parameter state.

    The trainable module parameters are a prior state and a fixed meta-network.
    Each batch row owns graph-connected LSTM parameters plus separate hidden and
    cell activation state. Registered parameters are never mutated in forward.
    """

    def __init__(
        self,
        feature_count: int,
        target_count: int,
        horizon_count: int,
        target_feature_indices: tuple[int, ...],
        hidden_dim: int,
        meta_hidden_dim: int,
        update_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if feature_count <= 0 or target_count <= 0 or horizon_count <= 0:
            raise ValueError("forecast model dimensions must be positive")
        if target_count > feature_count:
            raise ValueError("target_count must not exceed feature_count")
        if hidden_dim <= 0 or meta_hidden_dim <= 0:
            raise ValueError("hidden_dim and meta_hidden_dim must be positive")
        if update_scale <= 0:
            raise ValueError("update_scale must be positive")
        if len(target_feature_indices) != target_count:
            raise ValueError("target_feature_indices must match target_count")
        if len(set(target_feature_indices)) != len(target_feature_indices) or any(
            index < 0 or index >= feature_count for index in target_feature_indices
        ):
            raise ValueError("target_feature_indices must contain distinct feature positions")

        self.feature_count = feature_count
        self.target_count = target_count
        self.horizon_count = horizon_count
        self.hidden_dim = hidden_dim
        self.target_feature_indices = tuple(target_feature_indices)
        self.update_scale = update_scale
        self.layout = _DynamicLSTMLayout(feature_count, hidden_dim, horizon_count * target_count)
        self.parameter_count = self.layout.parameter_count
        self.theta_initial = nn.Parameter(self.layout.initial_parameters())
        # Shared scalar update rule keeps fixed meta-parameter count small while
        # generating a complete delta for every LSTM and forecast-head parameter.
        self.observation_encoder = nn.Sequential(nn.Linear(feature_count, meta_hidden_dim), nn.SiLU())
        self.meta_network = nn.Sequential(
            nn.Linear(meta_hidden_dim + 1, meta_hidden_dim),
            nn.SiLU(),
            nn.Linear(meta_hidden_dim, 1),
        )
        final_layer = self.meta_network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
        nn.init.zeros_(final_layer.bias)

    def _unpack(self, state: Tensor) -> _DynamicLSTMParameters:
        return self.layout.unpack(state)

    def _lstm_cell(self, inputs: Tensor, hidden: Tensor, cell: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Functionally apply a per-batch manual LSTM cell using dynamic parameters."""
        parameters = self._unpack(state)
        gates = (
            (parameters.weight_ih * inputs.unsqueeze(1)).sum(dim=-1)
            + (parameters.weight_hh * hidden.unsqueeze(1)).sum(dim=-1)
            + parameters.bias_ih
            + parameters.bias_hh
        )
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
        next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
        next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
        return next_hidden, next_cell

    def _predict(self, hidden: Tensor, state: Tensor) -> Tensor:
        parameters = self._unpack(state)
        return (parameters.head_weight * hidden.unsqueeze(1)).sum(dim=-1) + parameters.head_bias

    def _parameter_update(self, observation: Tensor, state: Tensor) -> Tensor:
        batch_size = observation.shape[0]
        context = self.observation_encoder(observation)
        expanded_context = context.unsqueeze(1).expand(-1, self.parameter_count, -1)
        update_inputs = torch.cat((expanded_context, state.unsqueeze(-1)), dim=-1)
        raw_update = self.meta_network(update_inputs.flatten(end_dim=1)).view(batch_size, self.parameter_count)
        return self.update_scale * torch.tanh(raw_update)

    def forward(self, batch: ForecastBatch) -> ModelOutput:
        inputs = batch.inputs
        if inputs.ndim != 3:
            raise ValueError("forecast inputs must have shape [batch, lookback, feature_count].")
        if inputs.shape[-1] != self.feature_count:
            raise ValueError("forecast input feature width does not match feature_count")
        batch_size, sequence_length, _ = inputs.shape
        if sequence_length < 2:
            raise ValueError("self-modifying forecast requires lookback of at least two")

        state = self.theta_initial.unsqueeze(0).expand(batch_size, -1)
        initial_state = state
        hidden = inputs.new_zeros(batch_size, self.hidden_dim)
        cell = inputs.new_zeros(batch_size, self.hidden_dim)
        predictions: list[Tensor] = []
        states: list[Tensor] = []
        updates: list[Tensor] = []
        for time_index in range(sequence_length):
            observation = inputs[:, time_index]
            hidden, cell = self._lstm_cell(observation, hidden, cell, state)
            predictions.append(self._predict(hidden, state))
            update = self._parameter_update(observation, state)
            updates.append(update)
            state = state + update
            states.append(state)

        prediction_trajectory = torch.stack(predictions, dim=1).unflatten(-1, (self.horizon_count, self.target_count))
        state_trajectory = torch.stack(states, dim=1)
        update_trajectory = torch.stack(updates, dim=1)
        return ModelOutput(
            predictions={
                "trajectory_forecast": prediction_trajectory[:, :-1, 0],
                "forecast": prediction_trajectory[:, -1].contiguous(),
            },
            representations=state_trajectory,
            diagnostics={
                "parameter_count": float(self.parameter_count),
                "mean_update_norm": update_trajectory.norm(dim=-1).mean().detach(),
                "mean_parameter_norm": state_trajectory.norm(dim=-1).mean().detach(),
                "final_parameter_shift": (state_trajectory[:, -1] - initial_state).norm(dim=-1).mean().detach(),
            },
        )
