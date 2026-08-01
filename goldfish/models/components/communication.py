"""Head-local latent message-passing components."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class HeadLatentCommunication(nn.Module):
    """Route translated cross-head messages into head-local recurrent states.

    Each input head is encoded into a communication latent. Every destination
    head uses a static softmax over the *other* source heads, decodes the
    resulting message into its local feature space, and injects it through a
    small residual gate. The local residual path is exact at initialization.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        communication_dim: int | None = None,
        gate_initial_logit: float = -5.0,
    ) -> None:
        super().__init__()
        if num_heads <= 1:
            raise ValueError("head latent communication requires at least two heads")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if communication_dim is None:
            communication_dim = head_dim
        if communication_dim <= 0:
            raise ValueError("communication_dim must be positive")
        if not math.isfinite(gate_initial_logit):
            raise ValueError("gate_initial_logit must be finite")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.communication_dim = communication_dim
        self.source_encoders = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, communication_dim),
                nn.GELU(),
                nn.Linear(communication_dim, communication_dim),
            )
            for _ in range(num_heads)
        )
        self.destination_decoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(communication_dim, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, head_dim),
            )
            for _ in range(num_heads)
        )
        self.routing_logits = nn.Parameter(torch.zeros(num_heads, num_heads))
        self.gate_logits = nn.Parameter(torch.full((num_heads, head_dim), gate_initial_logit))
        self.register_buffer("self_route_mask", torch.eye(num_heads, dtype=torch.bool), persistent=False)

    def routing_weights(self) -> Tensor:
        """Return receiver-by-source static routing weights with self routes masked."""
        return torch.softmax(self.routing_logits.masked_fill(self.self_route_mask, -torch.inf), dim=-1)

    def gates(self) -> Tensor:
        """Return per-destination, per-feature residual communication gates."""
        return torch.sigmoid(self.gate_logits)

    def forward(self, states: Tensor) -> Tensor:
        """Inject translated cross-head messages into ``[batch, time, heads, features]`` states."""
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, time, heads, features]")
        if states.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"states must have trailing shape ({self.num_heads}, {self.head_dim}), got {tuple(states.shape[-2:])}"
            )
        return states + self.gates().to(dtype=states.dtype) * self._decode(self._encode(states))[1]

    def diagnostics(self, states: Tensor) -> dict[str, Tensor]:
        """Return named intermediate tensors for observability.

        ``gated_messages`` equals ``gates * decoded``, the actual residual
        injected by :meth:`forward`.
        """
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, time, heads, features]")
        if states.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"states must have trailing shape ({self.num_heads}, {self.head_dim}), got {tuple(states.shape[-2:])}"
            )
        latents = self._encode(states)
        messages, decoded = self._decode(latents)
        gated = self.gates().to(dtype=states.dtype) * decoded
        return {
            "states": states,
            "latents": latents,
            "messages": messages,
            "decoded": decoded,
            "gated_messages": gated,
        }

    def _encode(self, states: Tensor) -> Tensor:
        return torch.stack(
            [encoder(states[..., index, :]) for index, encoder in enumerate(self.source_encoders)],
            dim=-2,
        )

    def _decode(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        messages = torch.einsum("ij,...jd->...id", self.routing_weights().to(dtype=latents.dtype), latents)
        decoded = torch.stack(
            [decoder(messages[..., index, :]) for index, decoder in enumerate(self.destination_decoders)],
            dim=-2,
        )
        return messages, decoded
