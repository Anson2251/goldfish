"""Validated probe configuration for the observability system.

The probe selections live in the model profile (``observability.probes``);
the run configuration only enables observability, configures the reference
data source, and may override profile declarations by name. Resolution and
validation follow the OBSERVABILITY.md specification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

_PROBE_NAMES = frozenset({"mixer-state", "communication-state", "activation-stats"})
_TENSOR_STATS = frozenset({"norm", "mean_abs", "std", "max", "p95"})
_PROBE_FIELDS = frozenset(
    {
        "name",
        "include",
        "include_matrix",
        "include_logits",
        "include_grad_norms",
        "head_dim",
        "points",
        "every_n_epochs",
        "epochs",
        "include_initial",
        "include_final",
        "require_match",
    }
)


@dataclass(frozen=True)
class ReferenceConfig:
    """The run-level deterministic input set for activation probes."""

    split: Literal["val", "test"]
    batches: int
    selection: Literal["first"] = "first"


@dataclass(frozen=True)
class ScheduleConfig:
    """Sampling schedule for one probe."""

    every_n_epochs: int | None = None
    epochs: tuple[int, ...] | None = None
    include_initial: bool = True
    include_final: bool = True


@dataclass(frozen=True)
class TensorStatConfig:
    """One declarative tensor statistic."""

    name: str
    stats: tuple[str, ...]
    reduce: Literal["overall", "per_head"] = "overall"


@dataclass(frozen=True)
class ActivationPointConfig:
    """One activation-stats point: a pattern plus a quantity or tensor stats."""

    path: str
    quantity: str | None = None
    tensors: tuple[TensorStatConfig, ...] = ()


@dataclass(frozen=True)
class ProbeConfig:
    """One resolved probe declaration."""

    name: str
    include: tuple[str, ...] | None = None
    include_matrix: bool = True
    include_logits: bool = True
    include_grad_norms: bool = False
    head_dim: int | None = None
    points: tuple[ActivationPointConfig, ...] = ()
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    require_match: bool = True


@dataclass(frozen=True)
class ResolvedObservabilityConfig:
    """The fully validated observability configuration for a run."""

    reference: ReferenceConfig | None = None
    probes: tuple[ProbeConfig, ...] = ()


def resolve_observability_config(
    profile_observability: Mapping[str, Any] | None,
    run_observability: Mapping[str, Any] | None,
) -> ResolvedObservabilityConfig:
    """Resolve profile probe declarations and run-level overrides.

    ``profile_observability`` is the optional ``observability`` block of a
    model profile; ``run_observability`` is the run-level ``observability``
    block. Run-level ``probes`` entries must name a probe declared by the
    profile and replace that declaration entirely.
    """
    profile = _mapping(profile_observability, "observability") if profile_observability is not None else {}
    run = _mapping(run_observability, "observability") if run_observability is not None else {}
    _reject_unknown(run, {"enabled", "reference", "probes"}, "observability")
    _reject_unknown(profile, {"probes"}, "observability")

    enabled = _bool(run.get("enabled", False), "observability.enabled")
    if not enabled:
        if "reference" in run:
            raise ValueError("observability.enabled must be true when a reference is configured")
        if "probes" in run:
            raise ValueError("observability.enabled must be true when probes are overridden")
        return ResolvedObservabilityConfig()

    profile_probes = _probes(profile.get("probes", []), "observability.probes")
    run_entries = run.get("probes", [])
    if run_entries and not profile_probes:
        raise ValueError("observability.probes overrides a probe that the model profile does not declare; the profile declares no probes")
    declared_names = {probe.name for probe in profile_probes}
    for entry in run_entries:
        raw_name = _raw_probe_name(entry, "observability.probes")
        if raw_name not in declared_names:
            raise ValueError(f"observability.probes names probe {raw_name!r} which is not declared by the model profile")
    run_probes = _probes(run_entries, "observability.probes")

    run_by_name = {probe.name: probe for probe in run_probes}
    probes = tuple(run_by_name.get(probe.name, probe) for probe in profile_probes)

    reference = _reference(run["reference"], "observability.reference") if "reference" in run else None
    if reference is None and any(probe.name == "activation-stats" for probe in probes):
        raise ValueError("observability.reference is required when activation-stats probes are active")
    return ResolvedObservabilityConfig(reference=reference, probes=probes)


def _probes(values: Any, prefix: str) -> tuple[ProbeConfig, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{prefix} must be a list")
    probes = tuple(_probe(entry, prefix) for entry in values)
    names = [probe.name for probe in probes]
    if len(names) != len(set(names)):
        raise ValueError(f"{prefix} declares duplicate probe name(s): {sorted({name for name in names if names.count(name) > 1})}")
    return probes


def _raw_probe_name(values: Any, prefix: str) -> str:
    """Extract only the probe name for declaration checks before full validation."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{prefix} entries must be mappings")
    return _name(values.get("name"), f"{prefix}.name")


def _probe(values: Any, prefix: str) -> ProbeConfig:
    if not isinstance(values, Mapping):
        raise ValueError(f"{prefix} entries must be mappings")
    _reject_unknown(values, _PROBE_FIELDS, prefix)
    name = _name(values.get("name"), f"{prefix}.name")
    if name not in _PROBE_NAMES:
        raise ValueError(f"{prefix}.name must be one of {', '.join(sorted(_PROBE_NAMES))}; got {name!r}")

    include = values.get("include")
    include_tuple = tuple(include) if include is not None else None
    if include is not None and (not isinstance(include, list) or not include):
        raise ValueError(f"{prefix}.include must be a non-empty list of patterns")
    if include_tuple is None and name == "mixer-state":
        include_tuple = ("mixer", "mixers.*")

    points = _points(values.get("points"), f"{prefix}.points") if "points" in values else ()
    if name == "activation-stats" and not points:
        raise ValueError(f"{prefix} activation-stats requires a non-empty 'points' list")
    if name == "communication-state" and include_tuple is None:
        raise ValueError(f"{prefix} communication-state requires 'include' patterns")

    if "every_n_epochs" in values and "epochs" in values:
        raise ValueError(f"{prefix} 'every_n_epochs' and 'epochs' are mutually exclusive")
    if "epochs" in values:
        every_n_epochs = None
        epochs = tuple(_positive_int(item, f"{prefix}.epochs") for item in values["epochs"])
        if any(second <= first for first, second in zip(epochs, epochs[1:])):
            raise ValueError(f"{prefix}.epochs must be strictly increasing")
    else:
        every_n_epochs = _positive_int(values.get("every_n_epochs", 1), f"{prefix}.every_n_epochs")
        epochs = None
    schedule = ScheduleConfig(
        every_n_epochs=every_n_epochs,
        epochs=epochs,
        include_initial=_bool(values.get("include_initial", True), f"{prefix}.include_initial"),
        include_final=_bool(values.get("include_final", True), f"{prefix}.include_final"),
    )

    return ProbeConfig(
        name=name,
        include=include_tuple,
        include_matrix=_bool(values.get("include_matrix", True), f"{prefix}.include_matrix"),
        include_logits=_bool(values.get("include_logits", True), f"{prefix}.include_logits"),
        include_grad_norms=_bool(values.get("include_grad_norms", False), f"{prefix}.include_grad_norms"),
        head_dim=_positive_int(values["head_dim"], f"{prefix}.head_dim") if "head_dim" in values else None,
        points=points,
        schedule=schedule,
        require_match=_bool(values.get("require_match", True), f"{prefix}.require_match"),
    )


def _points(values: Any, prefix: str) -> tuple[ActivationPointConfig, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"{prefix} must be a list")
    points = []
    for index, entry in enumerate(values):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{prefix}[{index}] must be a mapping")
        path = _name(entry.get("path"), f"{prefix}[{index}].path")
        quantity = entry.get("quantity")
        tensors = entry.get("tensors")
        if quantity is not None and tensors is not None:
            raise ValueError(f"{prefix}[{index}] must declare exactly one of 'quantity' or 'tensors'")
        if quantity is not None:
            if not isinstance(quantity, str):
                raise ValueError(f"{prefix}[{index}].quantity must be a string")
            points.append(ActivationPointConfig(path=path, quantity=quantity))
        elif tensors is not None:
            points.append(ActivationPointConfig(path=path, tensors=tuple(_tensor(item, f"{prefix}[{index}].tensors") for item in tensors)))
        else:
            raise ValueError(f"{prefix}[{index}] must declare exactly one of 'quantity' or 'tensors'")
    return tuple(points)


def _tensor(values: Any, prefix: str) -> TensorStatConfig:
    if not isinstance(values, Mapping):
        raise ValueError(f"{prefix} entries must be mappings")
    name = _name(values.get("name"), f"{prefix}.name")
    stats = values.get("stats")
    if not isinstance(stats, list) or not stats:
        raise ValueError(f"{prefix}.stats must be a non-empty list")
    for stat in stats:
        if stat not in _TENSOR_STATS:
            raise ValueError(f"{prefix}.stats entries must be one of {', '.join(sorted(_TENSOR_STATS))}; got {stat!r}")
    reduce = values.get("reduce", "overall")
    if reduce not in {"overall", "per_head"}:
        raise ValueError(f"{prefix}.reduce must be 'overall' or 'per_head'")
    return TensorStatConfig(name=name, stats=tuple(stats), reduce=reduce)


def _reference(values: Any, prefix: str) -> ReferenceConfig:
    if not isinstance(values, Mapping):
        raise ValueError(f"{prefix} must be a mapping")
    _reject_unknown(values, {"split", "batches", "selection"}, prefix)
    split = values.get("split", "val")
    if split not in {"val", "test"}:
        raise ValueError(f"{prefix}.split must be 'val' or 'test'")
    selection = values.get("selection", "first")
    if selection != "first":
        raise ValueError(f"{prefix}.selection must be 'first'")
    return ReferenceConfig(
        split=split,
        batches=_positive_int(values["batches"], f"{prefix}.batches") if "batches" in values else 8,
        selection=selection,
    )


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    for key in values:
        if key not in allowed:
            raise ValueError(f"{section} has unknown field {key!r} or it is not applicable to the selected schema")


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
