"""The probe hook: applies schedules, runs probes, and records observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from typing import Any

from torch import nn

from goldfish.config.observability import (
    ProbeConfig,
    ResolvedObservabilityConfig,
    ScheduleConfig,
)
from goldfish.observability.discovery import discover_modules
from goldfish.observability.events import HookContext, TrainingHook
from goldfish.observability.probes import ProbeRegistry, probe_registry
from goldfish.observability.recorder import JsonlRecorder, SCHEMA_VERSION

ReferenceFactory = Callable[[], Sequence[Any]]
ManifestFactory = Callable[[nn.Module, Sequence[Any] | None], Mapping[str, Any]]


def should_sample(schedule: ScheduleConfig, epoch_one_based: int) -> bool:
    """Decide whether an epoch_end observation should be taken."""
    if schedule.epochs is not None:
        return epoch_one_based in schedule.epochs
    if schedule.every_n_epochs is None:
        raise ValueError("schedule must declare either every_n_epochs or epochs")
    return epoch_one_based % schedule.every_n_epochs == 0


class ProbeHook(TrainingHook):
    """Invoke configured probes at lifecycle events and record their payloads.

    The reference batch set is built once at ``fit_start`` through the injected
    factory (when provided) and attached to the contexts of later events, so
    activation probes evaluate the same inputs across the trajectory.
    """

    def __init__(
        self,
        recorder: JsonlRecorder,
        probes: Sequence[tuple[Any, ScheduleConfig]],
        *,
        reference_factory: ReferenceFactory | None = None,
        manifest_factory: ManifestFactory | None = None,
    ) -> None:
        self.recorder = recorder
        self.probes = tuple(probes)
        self.reference_factory = reference_factory
        self.manifest_factory = manifest_factory
        self._reference_batches: Sequence[Any] | None = None

    def on_fit_start(self, context: HookContext) -> None:
        if self.reference_factory is not None:
            self._reference_batches = tuple(self.reference_factory())
        enriched = self._with_reference_batches(context)
        for probe, schedule in self.probes:
            if schedule.include_initial:
                payload = probe.collect(enriched)
                if payload is not None:
                    self.recorder.write_record(probe.name, "fit_start", 0, context.global_step, payload)
        if self.manifest_factory is not None:
            # Written after the collect pass so discovery and reference capture
            # have run; the manifest carries the resolved matched modules.
            self.recorder.write_manifest(self.manifest_factory(context.model, self._reference_batches))

    def on_epoch_end(self, context: HookContext) -> None:
        if context.epoch is None:
            raise ValueError("epoch_end events require an epoch")
        enriched = self._with_reference_batches(context)
        for probe, schedule in self.probes:
            if not should_sample(schedule, context.epoch + 1):
                continue
            payload = probe.collect(enriched)
            if payload is not None:
                self.recorder.write_record(probe.name, "epoch_end", context.epoch + 1, context.global_step, payload)

    def on_fit_end(self, context: HookContext) -> None:
        if context.epoch is None:
            raise ValueError("fit_end events require an epoch")
        enriched = self._with_reference_batches(context)
        for probe, schedule in self.probes:
            if not schedule.include_final:
                continue
            payload = probe.collect(enriched)
            if payload is not None:
                self.recorder.write_record(probe.name, "fit_end", context.epoch + 1, context.global_step, payload)

    def _with_reference_batches(self, context: HookContext) -> HookContext:
        if self._reference_batches is None:
            return context
        return replace(context, reference_batches=self._reference_batches)


def build_probe_hook(
    config: ResolvedObservabilityConfig,
    recorder: JsonlRecorder,
    *,
    registry: ProbeRegistry = probe_registry,
    reference_factory: ReferenceFactory | None = None,
    source_paths: Mapping[str, str] | None = None,
    split_fingerprint: str | None = None,
) -> ProbeHook | None:
    """Assemble a ``ProbeHook`` from a resolved observability configuration.

    Returns ``None`` when the configuration declares no probes, preserving the
    no-observability training path. Activation probes require an injected
    reference provider; the failure surfaces here, before training starts.
    """
    if not config.probes:
        return None
    if any(probe.name == "activation-stats" for probe in config.probes) and reference_factory is None:
        raise ValueError("activation-stats probes require an injected reference provider")
    probes = [(registry.create(probe.name, _options_mapping(probe)), probe.schedule) for probe in config.probes]
    return ProbeHook(
        recorder,
        probes,
        reference_factory=reference_factory,
        manifest_factory=_make_manifest_factory(
            config,
            source_paths or {},
            split_fingerprint,
        ),
    )


def _make_manifest_factory(
    config: ResolvedObservabilityConfig,
    source_paths: Mapping[str, str],
    split_fingerprint: str | None,
) -> ManifestFactory:
    def build(model: nn.Module, reference_batches: Sequence[Any] | None) -> Mapping[str, Any]:
        return build_manifest(
            config,
            model,
            source_paths=source_paths,
            split_fingerprint=split_fingerprint,
        )

    return build


def build_manifest(
    config: ResolvedObservabilityConfig,
    model: nn.Module,
    *,
    source_paths: Mapping[str, str],
    split_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build the resolved probe manifest from a configuration and the live model.

    ``matched_modules`` are resolved by discovery against the model at
    ``fit_start``; ``source_paths`` maps probe names to ``"profile"`` or
    ``"run_override"`` for provenance.
    """
    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if config.reference is not None:
        reference = _jsonable(asdict(config.reference))
        if split_fingerprint is not None:
            reference["split_fingerprint"] = split_fingerprint
        manifest["reference"] = reference
    probes: list[dict[str, Any]] = []
    for probe in config.probes:
        entry: dict[str, Any] = {
            "name": probe.name,
            "schedule": _jsonable(asdict(probe.schedule)),
            "options": _jsonable({key: value for key, value in _options_mapping(probe).items() if key not in {"include", "points"}}),
            "patterns": list(probe.include) if probe.include else [],
            "source": source_paths.get(probe.name, "profile"),
        }
        patterns = list(probe.include) if probe.include else []
        if probe.points:
            entry["points"] = []
            for point in probe.points:
                entry["points"].append(
                    {
                        "pattern": point.path,
                        "quantity": point.quantity,
                        "matched_modules": [path for path, _ in discover_modules(model, (point.path,))],
                    }
                )
        if patterns:
            entry["matched_modules"] = [path for path, _ in discover_modules(model, tuple(patterns))]
        probes.append(entry)
    manifest["probes"] = probes
    return manifest


def _options_mapping(probe: ProbeConfig) -> dict[str, Any]:
    options = {
        key: value
        for key, value in asdict(probe).items()
        if key not in {"name", "schedule", "points"}
    }
    if probe.points:
        options["points"] = [_point_mapping(point) for point in probe.points]
    return options


def _point_mapping(point) -> dict[str, Any]:
    if point.quantity is not None:
        return {"path": point.path, "quantity": point.quantity}
    return {"path": point.path, "tensors": [asdict(spec) for spec in point.tensors]}


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and tuples to JSON-friendly structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
