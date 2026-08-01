"""The probe hook: applies schedules, runs probes, and records observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from typing import Any

from goldfish.config.observability import (
    ProbeConfig,
    ResolvedObservabilityConfig,
    ScheduleConfig,
)
from goldfish.observability.events import HookContext, TrainingHook
from goldfish.observability.probes import ProbeRegistry, probe_registry
from goldfish.observability.recorder import JsonlRecorder, SCHEMA_VERSION

ReferenceFactory = Callable[[], Sequence[Any]]


def should_sample(schedule: ScheduleConfig, epoch_one_based: int) -> bool:
    """Decide whether an epoch_end observation should be taken."""
    if schedule.epochs is not None:
        return epoch_one_based in schedule.epochs
    assert schedule.every_n_epochs is not None
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
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.recorder = recorder
        self.probes = tuple(probes)
        self.reference_factory = reference_factory
        self.manifest = dict(manifest) if manifest is not None else None
        self._reference_batches: Sequence[Any] | None = None

    def on_fit_start(self, context: HookContext) -> None:
        if self.reference_factory is not None:
            self._reference_batches = tuple(self.reference_factory())
        if self.manifest is not None:
            self.recorder.write_manifest(self.manifest)
        enriched = self._with_reference_batches(context)
        for probe, schedule in self.probes:
            if schedule.include_initial:
                payload = probe.collect(enriched)
                if payload is not None:
                    self.recorder.write_record(probe.name, "fit_start", 0, context.global_step, payload)

    def on_epoch_end(self, context: HookContext) -> None:
        assert context.epoch is not None
        enriched = self._with_reference_batches(context)
        for probe, schedule in self.probes:
            if not should_sample(schedule, context.epoch + 1):
                continue
            payload = probe.collect(enriched)
            if payload is not None:
                self.recorder.write_record(probe.name, "epoch_end", context.epoch + 1, context.global_step, payload)

    def on_fit_end(self, context: HookContext) -> None:
        assert context.epoch is not None
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
    manifest: Mapping[str, Any] | None = None,
) -> ProbeHook | None:
    """Assemble a ``ProbeHook`` from a resolved observability configuration.

    Returns ``None`` when the configuration declares no probes, preserving the
    no-observability training path.
    """
    if not config.probes:
        return None
    probes = [(registry.create(probe.name, _options_mapping(probe)), probe.schedule) for probe in config.probes]
    return ProbeHook(
        recorder,
        probes,
        reference_factory=reference_factory,
        manifest=manifest,
    )


def build_manifest(
    config: ResolvedObservabilityConfig,
    *,
    source_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Build the resolved probe manifest from a configuration.

    ``source_paths`` maps probe names to ``"profile"`` or ``"run_override"``
    for provenance; concrete ``matched_modules`` are filled in by discovery at
    ``fit_start`` in a later phase.
    """
    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if config.reference is not None:
        manifest["reference"] = _jsonable(asdict(config.reference))
    probes: list[dict[str, Any]] = []
    for probe in config.probes:
        entry: dict[str, Any] = {
            "name": probe.name,
            "schedule": _jsonable(asdict(probe.schedule)),
            "options": _jsonable({key: value for key, value in _options_mapping(probe).items() if key != "include"}),
            "patterns": list(probe.include) if probe.include else [],
            "source": source_paths.get(probe.name, "profile"),
        }
        if probe.points:
            entry["points"] = [_jsonable(asdict(point)) for point in probe.points]
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
