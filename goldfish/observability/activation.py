"""Activation-tier probe: reference-forward statistics of intermediate tensors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from goldfish.observability.discovery import discover_modules
from goldfish.observability.events import HookContext

_STATS = frozenset({"norm", "mean_abs", "std", "max", "p95"})
_REDUCES = frozenset({"overall", "per_head"})
_QUANTITIES = frozenset({"message-magnitude", "mixing-displacement", "dense-displacement", "io-stats"})


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    stats: tuple[str, ...]
    reduce: str = "overall"


@dataclass(frozen=True)
class _Point:
    path: str
    quantity: str | None = None
    tensors: tuple[_TensorSpec, ...] = ()


class ActivationStatsProbe:
    """Compute intermediate-tensor statistics on the reference forward pass.

    Each point observes one pattern of modules; a point declares either a
    composite ``quantity`` or declarative ``tensors`` (name, stats, reduce).
    Statistics are aggregated over the reference batches and the batch/time
    dimensions; raw tensors are never persisted.
    """

    name = "activation-stats"

    def __init__(self, options: Mapping[str, Any]) -> None:
        points = options.get("points")
        if not points:
            raise ValueError("activation-stats requires a non-empty 'points' list")
        self.require_match = bool(options.get("require_match", True))
        self.points = [_parse_point(entry) for entry in points]

    def collect(self, context: HookContext) -> Mapping[str, Any] | None:
        batches = context.reference_batches
        if batches is None:
            raise RuntimeError("activation-stats requires reference batches in the hook context")
        model = context.model

        resolved: list[tuple[_Point, tuple[tuple[str, nn.Module], ...]]] = []
        for point in self.points:
            modules = discover_modules(model, (point.path,))
            if not modules and self.require_match:
                raise ValueError(f"activation-stats found no module matching patterns [{point.path}]")
            resolved.append((point, modules))
        if not any(modules for _, modules in resolved):
            return None

        device = next(model.parameters()).device
        accumulators: dict[tuple[int, int], _Accumulator] = {}
        handles = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for point_index, (point, modules) in enumerate(resolved):
                    for path, module in modules:
                        accumulator = _Accumulator(point)
                        accumulators[(point_index, id(module))] = accumulator
                        handles.append(module.register_forward_hook(_make_hook(accumulator, path)))
                for batch in batches:
                    model(batch.to(device))
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        entries = []
        for point_index, (point, modules) in enumerate(resolved):
            for path, module in modules:
                entries.append(accumulators[(point_index, id(module))].finalize(path))
        return {"points": entries}


def _parse_point(entry: Any) -> _Point:
    if not isinstance(entry, Mapping):
        raise ValueError("activation-stats points entries must be mappings")
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("activation-stats points require a non-empty 'path'")
    quantity = entry.get("quantity")
    tensors = entry.get("tensors")
    if quantity is not None and tensors is not None:
        raise ValueError("activation-stats points must declare exactly one of 'quantity' or 'tensors'")
    if quantity is not None:
        if quantity not in _QUANTITIES:
            raise ValueError(f"activation-stats unknown quantity {quantity!r}")
        return _Point(path=path, quantity=quantity)
    if tensors is not None:
        specs = []
        for index, spec in enumerate(tensors):
            if not isinstance(spec, Mapping):
                raise ValueError(f"activation-stats tensors[{index}] must be a mapping")
            name = spec.get("name")
            stats = spec.get("stats")
            if not isinstance(name, str) or not name:
                raise ValueError(f"activation-stats tensors[{index}].name must be a non-empty string")
            if not isinstance(stats, list) or not stats:
                raise ValueError(f"activation-stats tensors[{index}].stats must be a non-empty list")
            for stat in stats:
                if stat not in _STATS:
                    raise ValueError(f"activation-stats tensors[{index}].stats entries must be one of {sorted(_STATS)}")
            reduce = spec.get("reduce", "overall")
            if reduce not in _REDUCES:
                raise ValueError(f"activation-stats tensors[{index}].reduce must be 'overall' or 'per_head'")
            specs.append(_TensorSpec(name=name, stats=tuple(stats), reduce=reduce))
        return _Point(path=path, tensors=tuple(specs))
    raise ValueError("activation-stats points must declare exactly one of 'quantity' or 'tensors'")


def _make_hook(accumulator: _Accumulator, path: str):
    def hook(module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
        if accumulator.point.quantity is not None:
            accumulator.add_quantity(module, inputs, outputs, path)
        else:
            for spec in accumulator.point.tensors:
                tensor = _resolve_tensor(spec.name, module, inputs, outputs, path)
                accumulator.add_tensor(spec, tensor, path)

    return hook


def _resolve_tensor(name: str, module: nn.Module, inputs: tuple[Any, ...], outputs: Any, path: str) -> Tensor:
    if name == "input":
        return _as_tensor(inputs[0])
    if name == "output":
        return _as_tensor(outputs[0] if isinstance(outputs, tuple) else outputs)
    if name in ("hidden", "cell"):
        if not isinstance(outputs, tuple) or len(outputs) < 2 or not isinstance(outputs[1], tuple):
            raise ValueError(f"module {path!r} does not return the LSTM contract needed for tensor {name!r}")
        return _as_tensor(outputs[1][0 if name == "hidden" else 1])
    if not hasattr(module, "diagnostics"):
        raise ValueError(f"tensor name {name!r} on module {path!r} requires a diagnostics() method, which {type(module).__name__} lacks")
    diagnostics = module.diagnostics(_as_tensor(inputs[0]))  # type: ignore[attr-defined]
    if name not in diagnostics:
        raise ValueError(f"tensor name {name!r} is not produced by diagnostics() of module {path!r}")
    return diagnostics[name]


def _as_tensor(value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"expected a tensor, got {type(value).__name__}")
    return value


class _Accumulator:
    """Collect per-element statistics and composite quantity accumulators.

    Ratio quantities keep a per-element validity mask: elements whose
    denominator is zero are excluded from every reduction, and a field whose
    elements are all invalid is recorded as ``None`` (spec §7.3 aggregation
    rules).
    """

    def __init__(self, point: _Point) -> None:
        self.point = point
        self._values: dict[str, list[Tensor]] = defaultdict(list)  # key -> per-batch element values
        self._per_head: dict[str, list[Tensor]] = defaultdict(list)  # key -> [B, T, N] values
        self._ratios: dict[str, list[tuple[Tensor, Tensor]]] = defaultdict(list)  # key -> [(values, mask)]
        self._per_head_ratios: dict[str, list[tuple[Tensor, Tensor]]] = defaultdict(list)

    def add_tensor(self, spec: _TensorSpec, tensor: Tensor, path: str) -> None:
        for stat in spec.stats:
            values = _per_element(tensor, stat)
            if spec.reduce == "per_head":
                if values.ndim != 3:
                    raise ValueError(
                        f"per_head reduce on {path!r} tensor {spec.name!r} requires a [B, T, N, D] tensor, got shape {tuple(tensor.shape)}"
                    )
                self._per_head[f"{spec.name}.{stat}"].append(values)
            else:
                self._values[f"{spec.name}.{stat}"].append(values.flatten())

    def add_quantity(self, module: nn.Module, inputs: tuple[Any, ...], outputs: Any, path: str) -> None:
        quantity = self.point.quantity
        if quantity == "io-stats":
            input_tensor = _as_tensor(inputs[0])
            output_tensor = _as_tensor(outputs[0] if isinstance(outputs, tuple) else outputs)
            self._add_stat("input_norm", input_tensor, "norm")
            self._add_stat("output_norm", output_tensor, "norm")
            self._add_stat("input_mean_abs", input_tensor, "mean_abs")
            self._add_stat("output_mean_abs", output_tensor, "mean_abs")
        elif quantity == "dense-displacement":
            input_tensor = _as_tensor(inputs[0])
            output_tensor = _as_tensor(outputs[0] if isinstance(outputs, tuple) else outputs)
            self._add_stat("input_norm", input_tensor, "norm")
            self._add_stat("output_norm", output_tensor, "norm")
            difference = (output_tensor - input_tensor).norm(dim=-1)
            self._add_ratio("displacement_ratio", difference, input_tensor.norm(dim=-1), path)
        elif quantity == "mixing-displacement":
            diagnostics = _diagnostics(module, inputs, path, quantity)
            input_tensor = diagnostics["input"]
            output_tensor = diagnostics["output"]
            difference = (output_tensor - input_tensor).norm(dim=-1)
            self._add_stat("input_norm", input_tensor, "norm")
            self._add_ratio("displacement_ratio", difference, input_tensor.norm(dim=-1), path)
            self._add_per_head_ratio("displacement_ratio_per_head", difference, input_tensor.norm(dim=-1), path)
        elif quantity == "message-magnitude":
            diagnostics = _diagnostics(module, inputs, path, quantity)
            states = diagnostics["states"]
            decoded = diagnostics["decoded"]
            gated = diagnostics["gated_messages"]
            states_norm = states.norm(dim=-1)
            self._add_per_head_ratio("injection_ratio_per_receiver", gated.norm(dim=-1), states_norm, path)
            self._add_per_head_ratio("decoded_ratio_per_receiver", decoded.norm(dim=-1), states_norm, path)
            self._add_stat("gated_message_norm", gated, "norm")
        else:  # pragma: no cover - guarded by _parse_point
            raise ValueError(f"unknown quantity {quantity!r}")

    def _add_stat(self, key: str, tensor: Tensor, stat: str) -> None:
        self._values[key].append(_per_element(tensor, stat).flatten())

    def _add_ratio(self, key: str, numerator: Tensor, denominator: Tensor, path: str) -> None:
        values, mask = _safe_ratio(numerator, denominator, path, key)
        self._ratios[key].append((values.flatten(), mask.flatten()))

    def _add_per_head_ratio(self, key: str, numerator: Tensor, denominator: Tensor, path: str) -> None:
        if numerator.ndim != 3:
            raise ValueError(f"per_head reduce on {path!r} requires [B, T, N] values, got shape {tuple(numerator.shape)}")
        values, mask = _safe_ratio(numerator, denominator, path, key)
        self._per_head_ratios[key].append((values, mask))

    def finalize(self, path: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"module": path}
        if self.point.quantity is not None:
            entry["quantity"] = self.point.quantity
            per_head = _mean_per_head(self._per_head)
            entry.update(_mean_overall(self._values))
            entry.update(_mean_overall_ratios(self._ratios))
            entry.update(per_head)
            entry.update(_mean_per_head_ratios(self._per_head_ratios))
            for key, values in per_head.items():
                if key.endswith("_per_receiver"):
                    entry[key.removesuffix("_per_receiver") + "_mean"] = _mean_or_none(values)
            for key, values in _mean_per_head_ratios(self._per_head_ratios).items():
                if key.endswith("_per_receiver"):
                    entry[key.removesuffix("_per_receiver") + "_mean"] = _mean_or_none(values)
            return entry
        tensors: dict[str, dict[str, Any]] = {}
        for key, samples in self._values.items():
            name, stat = key.rsplit(".", 1)
            tensors.setdefault(name, {})[stat] = _mean(samples)
        for key, samples in self._per_head.items():
            name, stat = key.rsplit(".", 1)
            tensors.setdefault(name, {})[stat] = _mean_per_head_values(samples)
        entry["tensors"] = tensors
        return entry


def _diagnostics(module: nn.Module, inputs: tuple[Any, ...], path: str, quantity: str) -> dict[str, Tensor]:
    """Call diagnostics() with a clear error context for the quantity."""
    if not hasattr(module, "diagnostics"):
        raise ValueError(f"quantity {quantity!r} on module {path!r} requires a diagnostics() method, which {type(module).__name__} lacks")
    diagnostics = module.diagnostics(_as_tensor(inputs[0]))  # type: ignore[attr-defined]
    required = {"mixing-displacement": {"input", "output"}, "message-magnitude": {"states", "decoded", "gated_messages"}}[quantity]
    missing = required.difference(diagnostics)
    if missing:
        raise ValueError(f"quantity {quantity!r} on module {path!r}: diagnostics() is missing tensors {sorted(missing)}")
    return diagnostics


def _safe_ratio(numerator: Tensor, denominator: Tensor, path: str, key: str) -> tuple[Tensor, Tensor]:
    """Return (ratios, mask) with non-finite ratios excluded by the mask."""
    mask = denominator > 0
    safe = torch.where(mask, denominator, torch.ones_like(denominator))
    ratios = numerator / safe
    return ratios, mask


def _per_element(tensor: Tensor, stat: str) -> Tensor:
    x = tensor.detach().float()
    if stat == "norm":
        return x.norm(dim=-1)
    if stat == "mean_abs":
        return x.abs().mean(dim=-1)
    if stat == "std":
        return x.std(dim=-1, correction=0)
    if stat == "max":
        return x.abs().amax(dim=-1)
    if stat == "p95":
        return torch.quantile(x.abs(), 0.95, dim=-1)
    raise ValueError(f"unknown stat {stat!r}")  # pragma: no cover - guarded


def _mean(samples: list[Tensor]) -> float:
    if not samples:
        return 0.0
    return float(torch.cat(samples).mean())


def _mean_per_head_values(samples: list[Tensor]) -> list[float]:
    if not samples:
        return []
    return torch.cat(samples, dim=0).mean(dim=(0, 1)).tolist()


def _mean_overall(values: dict[str, list[Tensor]]) -> dict[str, float]:
    return {key: _mean(samples) for key, samples in values.items()}


def _mean_per_head(per_head: dict[str, list[Tensor]]) -> dict[str, list[float]]:
    return {key: _mean_per_head_values(samples) for key, samples in per_head.items()}


def _mean_overall_ratios(ratios: dict[str, list[tuple[Tensor, Tensor]]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, samples in ratios.items():
        values = torch.cat([item[0] for item in samples]).flatten()
        masks = torch.cat([item[1] for item in samples]).flatten()
        valid = values[masks]
        result[key] = float(valid.mean()) if valid.numel() else None
    return result


def _mean_per_head_ratios(ratios: dict[str, list[tuple[Tensor, Tensor]]]) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    for key, samples in ratios.items():
        values = torch.cat([item[0] for item in samples], dim=0)
        masks = torch.cat([item[1] for item in samples], dim=0)
        head_count = values.shape[-1]
        result[key] = []
        for head in range(head_count):
            valid = values[masks[:, head], head]
            result[key].append(float(valid.mean()) if valid.numel() else None)
    return result


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(torch.tensor(valid).mean())
