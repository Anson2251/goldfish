"""Manifold-constrained channel mixing components."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _LogNormalize(nn.Module):
    """Normalize one log-space matrix dimension to a unit probability sum."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, log_matrix: Tensor) -> Tensor:
        return log_matrix - torch.logsumexp(log_matrix, dim=self.dim, keepdim=True)


class _SinkhornProjection(nn.Module):
    """Project logit matrices onto the Birkhoff polytope in log space."""

    def __init__(self, iterations: int) -> None:
        super().__init__()
        self.normalization_steps = nn.Sequential(
            *[_LogNormalize(dim) for _ in range(iterations) for dim in (-1, -2)]
        )

    def forward(self, logits: Tensor) -> Tensor:
        return self.normalization_steps(logits.float()).exp()


class UnconstrainedMixer(nn.Module):
    """Mix channels with an unconstrained learned linear map."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        self.num_channels = num_channels
        self.mixing = nn.Linear(num_channels, num_channels, bias=False)
        nn.init.eye_(self.mixing.weight)

    def mixing_matrix(self) -> Tensor:
        """Return the unconstrained matrix with rows as output channels."""
        return self.mixing.weight

    def forward(self, channels: Tensor) -> Tensor:
        """Mix an input shaped ``[..., N, D]`` without changing its shape."""
        if channels.ndim < 2:
            raise ValueError(f"channels must have shape [..., {self.num_channels}, features], got {tuple(channels.shape)}")
        if channels.shape[-2] != self.num_channels:
            raise ValueError(
                f"channels must have {self.num_channels} channels at dimension -2, got {channels.shape[-2]}"
            )
        if channels.shape[-1] == 0:
            raise ValueError("channels must have a non-empty feature dimension")
        return torch.matmul(self.mixing.weight.to(dtype=channels.dtype), channels)


class DoublyStochasticMixer(nn.Module):
    """Mix channels with a learnable doubly stochastic matrix.

    The final two dimensions of the input are interpreted as ``[channels,
    features]``.  Every output channel is a non-negative convex combination of
    all input channels, while every input channel has unit total contribution
    across the output channels.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        sinkhorn_iterations: int = 20,
        identity_strength: float = 10.0,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if identity_strength <= 0:
            raise ValueError("identity_strength must be positive")

        self.num_channels = num_channels
        self.sinkhorn_iterations = sinkhorn_iterations
        self.projection = _SinkhornProjection(sinkhorn_iterations)
        self.logits = nn.Parameter(torch.eye(num_channels) * identity_strength)

    def mixing_matrix(self) -> Tensor:
        """Return the learned channel mixing matrix with shape ``[N, N]``.

        Rows index output channels and columns index input channels.
        Log-space Sinkhorn normalization is used to avoid numerical overflow
        while projecting the parameter logits onto the Birkhoff polytope.
        """
        return self.projection(self.logits)

    def forward(self, channels: Tensor) -> Tensor:
        """Mix an input shaped ``[..., N, D]`` without changing its shape."""
        if channels.ndim < 2:
            raise ValueError(f"channels must have shape [..., {self.num_channels}, features], got {tuple(channels.shape)}")
        if channels.shape[-2] != self.num_channels:
            raise ValueError(
                f"channels must have {self.num_channels} channels at dimension -2, got {channels.shape[-2]}"
            )
        if channels.shape[-1] == 0:
            raise ValueError("channels must have a non-empty feature dimension")

        mixing = self.mixing_matrix().to(dtype=channels.dtype)
        return torch.matmul(mixing, channels)
