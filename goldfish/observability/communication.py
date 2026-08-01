"""Communication-state probe: parameter-tier observation of inter-head communication."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from goldfish.models.components import HeadLatentCommunication
from goldfish.observability.discovery import discover_modules
from goldfish.observability.events import HookContext


class CommunicationStateProbe:
    """Record dense ``nn.Linear`` communication blocks and gated latent blocks.

    Dense weights are reported with per-head block norms when ``head_dim`` is
    available (option or ``model.head_dim``). Latent blocks report routing
    weights, routing entropy, gates, and encoder/decoder weight norms. Gradient
    norms cover every trainable ``named_parameters`` entry when enabled.
    """

    name = "communication-state"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.include = tuple(options.get("include", ()))
        if not self.include:
            raise ValueError("communication-state requires 'include' patterns")
        self.include_grad_norms = bool(options.get("include_grad_norms", False))
        self.head_dim = options.get("head_dim")
        self.require_match = bool(options.get("require_match", True))

    def collect(self, context: HookContext) -> Mapping[str, Any] | None:
        matches = discover_modules(context.model, self.include)
        if not matches:
            if self.require_match:
                raise ValueError(f"communication-state found no communication module matching patterns {list(self.include)}")
            return None
        return {"communications": [self._extract(path, module, context.model) for path, module in matches]}

    def _extract(self, path: str, module: nn.Module, model: nn.Module) -> dict[str, Any]:
        if isinstance(module, nn.Linear):
            return self._extract_dense(path, module, model)
        if isinstance(module, HeadLatentCommunication):
            return self._extract_latent(path, module)
        raise ValueError(f"communication-state does not support module {path!r} of type {type(module).__name__}")

    def _extract_dense(self, path: str, module: nn.Linear, model: nn.Module) -> dict[str, Any]:
        weight = module.weight.detach().to(dtype=torch.float64)
        identity = torch.eye(weight.shape[0], dtype=weight.dtype)
        difference = weight - identity
        head_dim = self.head_dim or getattr(model, "head_dim", None)
        entry: dict[str, Any] = {
            "module": path,
            "type": "dense_linear",
            "shape": list(weight.shape),
            "weight_frobenius_norm": float(weight.norm()),
            "bias_norm": float(module.bias.detach().to(dtype=torch.float64).norm()) if module.bias is not None else None,
            "frobenius_distance_to_identity": float(difference.norm()),
            "spectral_distance_to_identity": float(torch.linalg.matrix_norm(difference, ord=2)),
            "max_abs_distance_to_identity": float(difference.abs().max()),
        }
        if isinstance(head_dim, int) and head_dim > 0 and weight.shape[0] % head_dim == 0:
            block_diagonal, block_cross = _block_norms(weight, head_dim)
            entry["block_diagonal_deviation_norm"] = block_diagonal
            entry["block_cross_norm_mean"] = block_cross[0]
            entry["block_cross_norm_max"] = block_cross[1]
        else:
            entry["block_diagonal_deviation_norm"] = None
            entry["block_cross_norm_mean"] = None
            entry["block_cross_norm_max"] = None
        if self.include_grad_norms:
            entry["grad_norms"] = {
                "weight": _grad_norm(module.weight),
                "bias": _grad_norm(module.bias) if module.bias is not None else None,
            }
        return entry

    def _extract_latent(self, path: str, module: HeadLatentCommunication) -> dict[str, Any]:
        routing = module.routing_weights().detach().to(dtype=torch.float64)
        gates = module.gates().detach().to(dtype=torch.float64)
        entry: dict[str, Any] = {
            "module": path,
            "type": "latent_communication",
            "routing": routing.tolist(),
            "routing_entropy_per_receiver": _routing_entropies(routing),
            "routing_uniformity_distance": _uniformity_distance(routing),
            "gate_min": float(gates.min()),
            "gate_mean": float(gates.mean()),
            "gate_max": float(gates.max()),
            "encoder_weight_norm": _linear_weight_norm(module.source_encoders),
            "decoder_weight_norm": _linear_weight_norm(module.destination_decoders),
        }
        if self.include_grad_norms:
            entry["grad_norms"] = {
                name: _grad_norm(parameter) for name, parameter in module.named_parameters()
            }
        return entry


def _block_norms(weight: Tensor, head_dim: int) -> tuple[float, tuple[float, float]]:
    """Return (mean diagonal deviation, (mean, max) cross-head norms)."""
    num_blocks = weight.shape[0] // head_dim
    identity = torch.eye(head_dim, dtype=weight.dtype)
    diagonal: list[float] = []
    cross: list[float] = []
    for row in range(num_blocks):
        for column in range(num_blocks):
            block = weight[row * head_dim : (row + 1) * head_dim, column * head_dim : (column + 1) * head_dim]
            if row == column:
                diagonal.append(float((block - identity).norm()))
            else:
                cross.append(float(block.norm()))
    mean_cross = float(torch.tensor(cross).mean()) if cross else 0.0
    max_cross = float(torch.tensor(cross).max()) if cross else 0.0
    mean_diagonal = float(torch.tensor(diagonal).mean()) if diagonal else 0.0
    return mean_diagonal, (mean_cross, max_cross)


def _routing_entropies(routing: Tensor) -> list[float]:
    """Shannon entropy (nats) of each receiver row, ignoring zero entries."""
    entropies = []
    for row in routing:
        nonzero = row[row > 0]
        if nonzero.numel() == 0:
            entropies.append(0.0)
            continue
        entropies.append(float(-(nonzero * nonzero.log()).sum()))
    return entropies


def _uniformity_distance(routing: Tensor) -> float:
    """Mean over receivers of the L1 distance to uniform non-self routing."""
    num_sources = routing.shape[1] - 1
    uniform = 1.0 / num_sources
    distances = []
    for row in routing:
        nonzero = row[row > 0]
        distances.append(float((nonzero - uniform).abs().sum()))
    return float(torch.tensor(distances).mean()) if distances else 0.0


def _linear_weight_norm(modules: nn.ModuleList) -> float:
    total = 0.0
    for module in modules:
        for parameter in module.parameters():
            if parameter.ndim == 2:
                total += float(parameter.detach().norm())
    return total


def _grad_norm(parameter: nn.Parameter | None) -> float | None:
    if parameter is None or parameter.grad is None:
        return None
    return float(parameter.grad.detach().norm())
