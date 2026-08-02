"""DeltaNet: linear attention with a delta-rule fast-weight memory.

DeltaNet (Schlag et al., 2021; Yang et al., 2024) replaces linear attention's
plain key-value accumulation with an error-correcting delta rule. Each head
maintains a matrix-valued memory S in R^{head_dim x head_dim} that is updated
per position as

    S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T
    o_t = S_t q_t

with a learnable per-head beta_t in (0, 1) (the delta-rule learning rate). The
update erases the old association for key k_t and writes a blended replacement,
so the memory stores key-value associations instead of accumulating them. With
unit-norm keys and beta = 1 the transition I - k k^T is an orthogonal
projection that removes only the direction of k, which keeps interference
between stored associations low.

The layer follows the modernized design: pre-norm LayerNorm, a fused
query/key/value projection, a causal depthwise short convolution, SiLU
activation, L2 normalization of queries and keys, the delta-rule scan, output
RMSNorm, and a residual output projection.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional


def delta_rule_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    *,
    return_memory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run the delta-rule fast-weight update over time.

    ``q``, ``k``, ``v`` have shape ``[B, T, H, D]`` and ``beta`` has shape
    ``[H]`` with values in ``(0, 1)``. Returns per-position outputs
    ``[B, T, H, D]`` where ``o_t = S_t q_t`` uses the memory after incorporating
    position ``t``. With ``return_memory`` the final memory ``[B, H, D, D]`` is
    returned alongside the outputs.
    """
    if not (q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape == v.shape):
        raise ValueError("delta-rule inputs must have matching shapes [B, T, H, D]")
    batch_size, seq_len, num_heads, head_dim = q.shape
    if beta.ndim != 1 or beta.shape[0] != num_heads:
        raise ValueError(f"beta must have shape [{num_heads}], got {tuple(beta.shape)}")
    beta = beta.view(1, num_heads, 1, 1)
    memory = q.new_zeros(batch_size, num_heads, head_dim, head_dim)
    outputs: list[Tensor] = []
    for t in range(seq_len):
        k_t = k[:, t]  # [B, H, D]
        # Prediction error of the associative memory: S k_t - v_t.
        error = torch.einsum("bhij,bhj->bhi", memory, k_t) - v[:, t]
        memory = memory - beta * torch.einsum("bhi,bhj->bhij", error, k_t)
        outputs.append(torch.einsum("bhij,bhj->bhi", memory, q[:, t]))
    stacked = torch.stack(outputs, dim=1)
    if return_memory:
        return stacked, memory
    return stacked


class _RMSNorm(nn.Module):
    """Root-mean-square normalization with a learnable per-channel weight."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _DeltaNetLayer(nn.Module):
    """A single DeltaNet token-mixing layer with a pre-norm residual block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        *,
        short_conv_kernel: int = 4,
        beta_initial_logit: float = 4.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_heads <= 0:
            raise ValueError("DeltaNet layer dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if short_conv_kernel <= 0:
            raise ValueError("short_conv_kernel must be positive")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_norm = nn.LayerNorm(input_dim)
        self.qkv = nn.Linear(input_dim, 3 * hidden_dim)
        # Causal depthwise convolution over the concatenated q/k/v channels;
        # left padding of kernel - 1 keeps each position causal.
        self.short_conv = (
            nn.Conv1d(
                3 * hidden_dim,
                3 * hidden_dim,
                short_conv_kernel,
                padding=short_conv_kernel - 1,
                groups=3 * hidden_dim,
            )
            if short_conv_kernel > 1
            else nn.Identity()
        )
        self.output_norm = _RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # The first layer may have a different input dimension; without a
        # matching projection there is no identity path, so the residual is
        # only present when dimensions agree (like torch recurrent layers).
        self.residual = input_dim == hidden_dim
        # Sigmoid of the logit gives the delta-rule learning rate; initialized
        # near one so the memory writes its first association almost completely.
        self.beta_logits = nn.Parameter(torch.full((num_heads,), float(beta_initial_logit)))

    def forward(self, x: Tensor, *, return_memory: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Return ``[B, T, hidden_dim]``; with ``return_memory`` also the final memory."""
        seq_len = x.shape[1]
        residual = x
        x = self.input_norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        mixed = self.short_conv(torch.cat((q, k, v), dim=-1).transpose(1, 2)).transpose(1, 2)[:, :seq_len]
        q, k, v = mixed.chunk(3, dim=-1)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        q = functional.normalize(functional.silu(q), dim=-1)
        k = functional.normalize(functional.silu(k), dim=-1)
        v = functional.silu(v)
        scanned = delta_rule_scan(q, k, v, self.beta_logits.sigmoid(), return_memory=return_memory)
        if return_memory:
            outputs, memory = cast(tuple[Tensor, Tensor], scanned)
        else:
            outputs, memory = cast(Tensor, scanned), None
        outputs = self.output_proj(self.output_norm(outputs.flatten(start_dim=-2)))
        if self.residual:
            outputs = residual + outputs
        if memory is not None:
            return outputs, memory
        return outputs


class DeltaNetBackbone(nn.Module):
    """Contextualize embedded sequences with stacked DeltaNet layers.

    Returns per-position outputs ``[B, T, hidden_dim]`` and, as the final
    state, the last layer's fast-weight memory ``[B, num_heads, head_dim,
    head_dim]``. The memory is written from the inputs, so an external hidden
    state is not accepted.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        short_conv_kernel: int = 4,
        beta_initial_logit: float = 4.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            _DeltaNetLayer(
                input_dim if index == 0 else hidden_dim,
                hidden_dim,
                num_heads,
                short_conv_kernel=short_conv_kernel,
                beta_initial_logit=beta_initial_logit,
            )
            for index in range(num_layers)
        )
        # Like torch recurrent layers, dropout applies between layers only.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, embedded: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return per-position outputs and the final fast-weight memory state."""
        if embedded.ndim != 3:
            raise ValueError(f"embedded inputs must have shape [batch, time, features], got {tuple(embedded.shape)}")
        if hidden_state is not None:
            raise ValueError("DeltaNet writes its memory from inputs; an external hidden state is not supported")
        states = embedded
        memory: Tensor | None = None
        for index, layer in enumerate(self.layers):
            if index + 1 == self.num_layers:
                states, memory = layer(states, return_memory=True)
            else:
                states = self.dropout(layer(states))
        assert memory is not None
        return states, memory
