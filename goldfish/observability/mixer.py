"""Mixer-state probe: parameter-tier observation of mixing matrices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from goldfish.models.components import DoublyStochasticMixer, UnconstrainedMixer
from goldfish.observability.discovery import discover_modules
from goldfish.observability.events import HookContext


class MixerStateProbe:
    """Record doubly stochastic and unconstrained mixer matrices and derived distances.

    The projected matrix is obtained through the module's public
    ``mixing_matrix()``; logits and gradients are read from the parameters
    without mutation. The first observation of a mixer caches its logits as the
    baseline for ``logits_distance_to_initial``.
    """

    name = "mixer-state"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.include = tuple(options.get("include", ("mixer", "mixers.*")))
        self.include_matrix = bool(options.get("include_matrix", True))
        self.include_logits = bool(options.get("include_logits", True))
        self.include_grad_norms = bool(options.get("include_grad_norms", False))
        self.require_match = bool(options.get("require_match", True))
        self._initial_logits: dict[int, Tensor] = {}

    def collect(self, context: HookContext) -> Mapping[str, Any] | None:
        matches = discover_modules(context.model, self.include)
        if not matches:
            if self.require_match:
                raise ValueError(f"mixer-state found no mixer matching patterns {list(self.include)}")
            return None
        return {"mixers": [self._extract(path, module) for path, module in matches]}

    def _extract(self, path: str, module: nn.Module) -> dict[str, Any]:
        if isinstance(module, DoublyStochasticMixer):
            return self._extract_doubly_stochastic(path, module)
        if isinstance(module, UnconstrainedMixer):
            return self._extract_unconstrained(path, module)
        raise ValueError(f"mixer-state does not support module {path!r} of type {type(module).__name__}")

    def _extract_doubly_stochastic(self, path: str, module: DoublyStochasticMixer) -> dict[str, Any]:
        projected = module.mixing_matrix().detach().to(dtype=torch.float64)
        entry = {
            "module": path,
            "type": "doubly_stochastic",
            "shape": list(projected.shape),
            **_matrix_statistics(projected),
        }
        if self.include_matrix:
            entry["matrix"] = _nested_list(projected)
        if self.include_logits:
            logits = module.logits.detach().to(dtype=torch.float64)
            entry["logits"] = _nested_list(logits)
            baseline = self._initial_logits.setdefault(id(module), logits.clone())
            entry["logits_distance_to_initial"] = float((logits - baseline).norm())
        else:
            entry["logits_distance_to_initial"] = None
        if self.include_grad_norms:
            entry["grad_norms"] = {"logits": _grad_norm(module.logits)}
        return entry

    def _extract_unconstrained(self, path: str, module: UnconstrainedMixer) -> dict[str, Any]:
        weight = module.mixing_matrix().detach().to(dtype=torch.float64)
        entry = {
            "module": path,
            "type": "unconstrained",
            "shape": list(weight.shape),
            **_matrix_statistics(weight),
            "row_sum_max_error": None,
            "column_sum_max_error": None,
        }
        if self.include_matrix:
            entry["matrix"] = _nested_list(weight)
        entry["logits"] = None
        entry["logits_distance_to_initial"] = None
        if self.include_grad_norms:
            entry["grad_norms"] = {"weight": _grad_norm(module.mixing.weight)}
        return entry


def _matrix_statistics(matrix: Tensor) -> dict[str, Any]:
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
    difference = matrix - identity
    return {
        "frobenius_distance_to_identity": float(difference.norm()),
        "spectral_distance_to_identity": float(torch.linalg.matrix_norm(difference, ord=2)),
        "max_abs_distance_to_identity": float(difference.abs().max()),
        "off_diagonal_mass": float(matrix.sum() - torch.trace(matrix)),
        "diagonal_min": float(matrix.diag().min()),
        "diagonal_max": float(matrix.diag().max()),
        "row_sum_max_error": float((matrix.sum(dim=1) - 1).abs().max()),
        "column_sum_max_error": float((matrix.sum(dim=0) - 1).abs().max()),
    }


def _grad_norm(parameter: nn.Parameter) -> float | None:
    if parameter.grad is None:
        return None
    return float(parameter.grad.detach().norm())


def _nested_list(tensor: Tensor) -> list[Any]:
    return tensor.tolist()
