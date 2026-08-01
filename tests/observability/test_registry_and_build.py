"""Tests for the probe registry, reference provider, and config-to-hook assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from goldfish.config.observability import (
    ProbeConfig,
    ReferenceConfig,
    ResolvedObservabilityConfig,
    ScheduleConfig,
)
from goldfish.observability.hooks import ProbeHook, build_probe_hook, build_manifest
from goldfish.observability.probes import ProbeRegistry
from goldfish.observability.recorder import JsonlRecorder
from goldfish.observability.reference import take_first_batches
from tests.observability.test_probe_hook import StubProbe


def test_registry_registers_and_creates_probes() -> None:
    registry = ProbeRegistry()
    registry.register("mixer-state", lambda options: StubProbe("mixer-state", {"ok": True}))

    probe = registry.create("mixer-state", {})

    assert isinstance(probe, StubProbe)
    assert probe.name == "mixer-state"


def test_registry_rejects_duplicate_registration() -> None:
    registry = ProbeRegistry()
    registry.register("mixer-state", lambda options: StubProbe("mixer-state", None))

    with pytest.raises(ValueError, match="already registered"):
        registry.register("mixer-state", lambda options: StubProbe("mixer-state", None))


def test_registry_rejects_unknown_probe_with_available_names() -> None:
    registry = ProbeRegistry()
    registry.register("mixer-state", lambda options: StubProbe("mixer-state", None))

    with pytest.raises(ValueError, match="unknown.*mixer-state"):
        registry.create("telemetry", {})


def test_take_first_batches_returns_configured_count() -> None:
    batches = [f"batch-{index}" for index in range(5)]

    selected = take_first_batches(iter(batches), 3)

    assert selected == ("batch-0", "batch-1", "batch-2")


def test_take_first_batches_fails_when_split_is_too_short() -> None:
    with pytest.raises(ValueError, match="fewer than 3"):
        take_first_batches(iter(["only"]), 3)


def test_build_probe_hook_returns_none_without_probes(tmp_path: Path) -> None:
    config = ResolvedObservabilityConfig(reference=None, probes=())

    hook = build_probe_hook(config, JsonlRecorder(tmp_path / "probes"))

    assert hook is None


def test_build_probe_hook_constructs_probes_and_binds_schedules(tmp_path: Path) -> None:
    registry = ProbeRegistry()
    registry.register("mixer-state", lambda options: StubProbe("mixer-state", {"ok": True}))
    config = ResolvedObservabilityConfig(
        reference=None,
        probes=(ProbeConfig(name="mixer-state", schedule=ScheduleConfig(every_n_epochs=2)),),
    )

    hook = build_probe_hook(config, JsonlRecorder(tmp_path / "probes"), registry=registry)

    assert isinstance(hook, ProbeHook)
    assert hook.probes[0][0].name == "mixer-state"
    assert hook.probes[0][1] == ScheduleConfig(every_n_epochs=2)


def test_build_probe_hook_forwards_reference_factory(tmp_path: Path) -> None:
    registry = ProbeRegistry()
    registry.register("activation-stats", lambda options: StubProbe("activation-stats", {"ok": True}))
    config = ResolvedObservabilityConfig(
        reference=ReferenceConfig(split="val", batches=8),
        probes=(ProbeConfig(name="activation-stats", points=(), schedule=ScheduleConfig(every_n_epochs=1)),),
    )

    hook = build_probe_hook(
        config,
        JsonlRecorder(tmp_path / "probes"),
        registry=registry,
        reference_factory=lambda: ("batch-a",),
    )

    assert hook is not None
    assert hook.reference_factory is not None


def test_build_probe_hook_rejects_activation_probe_without_reference_provider(tmp_path: Path) -> None:
    registry = ProbeRegistry()
    registry.register("activation-stats", lambda options: StubProbe("activation-stats", {"ok": True}))
    config = ResolvedObservabilityConfig(
        reference=None,
        probes=(ProbeConfig(name="activation-stats", points=(), schedule=ScheduleConfig(every_n_epochs=1)),),
    )

    with pytest.raises(ValueError, match="reference provider"):
        build_probe_hook(config, JsonlRecorder(tmp_path / "probes"), registry=registry)


def test_build_manifest_contains_reference_and_probe_options() -> None:
    from torch import nn

    model = nn.Module()
    config = ResolvedObservabilityConfig(
        reference=ReferenceConfig(split="test", batches=4),
        probes=(ProbeConfig(name="mixer-state", include=("mixer",), include_grad_norms=True),),
    )

    manifest = build_manifest(config, model, source_paths={"mixer-state": "profile"}, split_fingerprint="sha256:abc")

    assert manifest["schema_version"] == 2
    assert manifest["reference"] == {"split": "test", "batches": 4, "selection": "first", "split_fingerprint": "sha256:abc"}
    assert manifest["probes"][0]["name"] == "mixer-state"
    assert manifest["probes"][0]["patterns"] == ["mixer"]
    assert manifest["probes"][0]["matched_modules"] == []
    assert manifest["probes"][0]["source"] == "profile"
