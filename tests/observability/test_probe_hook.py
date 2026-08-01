"""Tests for the ProbeHook: scheduling, probe invocation, and recording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goldfish.observability.events import HookContext
from goldfish.observability.hooks import ProbeHook
from goldfish.observability.recorder import JsonlRecorder
from goldfish.config.observability import ScheduleConfig


class StubProbe:
    """A minimal probe that returns a fixed payload and records its contexts."""

    def __init__(self, name: str, payload: dict[str, Any] | None) -> None:
        self.name = name
        self.payload = payload
        self.contexts: list[HookContext] = []

    def collect(self, context: HookContext) -> dict[str, Any] | None:
        self.contexts.append(context)
        return self.payload


def _context(phase: str = "epoch_end", epoch: int = 0, global_step: int = 0) -> HookContext:
    return HookContext(
        model=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,
        epoch=epoch,
        global_step=global_step,
        result=None,
        phase=phase,  # type: ignore[arg-type]
    )


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hook(tmp_path: Path, probes, **kwargs) -> ProbeHook:
    return ProbeHook(JsonlRecorder(tmp_path / "artifacts" / "probes"), probes, **kwargs)


def test_fit_start_and_fit_end_records_are_written_per_include_flags(tmp_path: Path) -> None:
    probe = StubProbe("mixer-state", {"ok": True})
    hook = _hook(tmp_path, [(probe, ScheduleConfig(every_n_epochs=1))])

    hook.on_fit_start(_context(phase="fit_start", epoch=None, global_step=0))
    hook.on_fit_end(_context(phase="fit_end", epoch=0, global_step=1))

    records = _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")
    assert [(record["phase"], record["epoch"]) for record in records] == [("fit_start", 0), ("fit_end", 1)]
    assert records[0]["payload"] == {"ok": True}
    assert records[0]["global_step"] == 0


def test_include_initial_and_final_can_be_disabled(tmp_path: Path) -> None:
    probe = StubProbe("mixer-state", {"ok": True})
    schedule = ScheduleConfig(every_n_epochs=1, include_initial=False, include_final=False)
    hook = _hook(tmp_path, [(probe, schedule)])

    hook.on_fit_start(_context(phase="fit_start", epoch=None))
    hook.on_epoch_end(_context(phase="epoch_end", epoch=0))
    hook.on_fit_end(_context(phase="fit_end", epoch=0))

    records = _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")
    assert [(record["phase"], record["epoch"]) for record in records] == [("epoch_end", 1)]


def test_every_n_epochs_schedule_samples_divisible_epochs(tmp_path: Path) -> None:
    probe = StubProbe("mixer-state", {"ok": True})
    hook = _hook(tmp_path, [(probe, ScheduleConfig(every_n_epochs=2, include_initial=False, include_final=False))])

    for epoch in range(5):
        hook.on_epoch_end(_context(phase="epoch_end", epoch=epoch, global_step=epoch))

    records = _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")
    # one-based epochs 2 and 4 are sampled.
    assert [record["epoch"] for record in records] == [2, 4]


def test_explicit_epoch_list_samples_exactly_those_epochs(tmp_path: Path) -> None:
    probe = StubProbe("mixer-state", {"ok": True})
    hook = _hook(
        tmp_path,
        [(probe, ScheduleConfig(epochs=(1, 3, 10), include_initial=False, include_final=False))],
    )

    for epoch in range(5):
        hook.on_epoch_end(_context(phase="epoch_end", epoch=epoch))

    records = _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")
    assert [record["epoch"] for record in records] == [1, 3]


def test_fit_end_is_recorded_even_when_final_epoch_was_sampled(tmp_path: Path) -> None:
    probe = StubProbe("mixer-state", {"ok": True})
    hook = _hook(tmp_path, [(probe, ScheduleConfig(every_n_epochs=1))])

    hook.on_epoch_end(_context(phase="epoch_end", epoch=2, global_step=6))
    hook.on_fit_end(_context(phase="fit_end", epoch=2, global_step=6))

    records = _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")
    assert [(record["phase"], record["epoch"]) for record in records] == [("epoch_end", 3), ("fit_end", 3)]


def test_probe_returning_none_writes_no_record(tmp_path: Path) -> None:
    probe = StubProbe("silent", None)
    hook = _hook(tmp_path, [(probe, ScheduleConfig(every_n_epochs=1))])

    hook.on_epoch_end(_context(phase="epoch_end", epoch=0))

    path = tmp_path / "artifacts" / "probes" / "silent.jsonl"
    assert not path.exists()


def test_multiple_probes_write_their_own_files(tmp_path: Path) -> None:
    first = StubProbe("mixer-state", {"a": 1})
    second = StubProbe("activation-stats", {"b": 2})
    hook = _hook(tmp_path, [(first, ScheduleConfig(every_n_epochs=1)), (second, ScheduleConfig(every_n_epochs=1))])

    hook.on_epoch_end(_context(phase="epoch_end", epoch=0))

    assert _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")[0]["payload"] == {"a": 1}
    assert _records(tmp_path / "artifacts" / "probes" / "activation-stats.jsonl")[0]["payload"] == {"b": 2}


def test_reference_factory_runs_once_at_fit_start_and_populates_contexts(tmp_path: Path) -> None:
    calls: list[int] = []

    def factory():
        calls.append(1)
        return ["batch-a", "batch-b"]

    probe = StubProbe("activation-stats", {"ok": True})
    hook = _hook(
        tmp_path,
        [(probe, ScheduleConfig(every_n_epochs=1))],
        reference_factory=factory,
    )

    hook.on_fit_start(_context(phase="fit_start", epoch=None))
    hook.on_epoch_end(_context(phase="epoch_end", epoch=0))

    assert calls == [1]
    assert probe.contexts[0].reference_batches == ("batch-a", "batch-b")  # fit_start is enriched too
    assert probe.contexts[1].reference_batches == ("batch-a", "batch-b")


def test_manifest_is_written_at_fit_start(tmp_path: Path) -> None:
    from torch import nn

    probe = StubProbe("mixer-state", {"ok": True})
    manifest = {"schema_version": 2, "probes": [{"name": "mixer-state"}]}
    hook = ProbeHook(
        JsonlRecorder(tmp_path / "artifacts" / "probes"),
        [(probe, ScheduleConfig(every_n_epochs=1))],
        manifest_factory=lambda model, batches: manifest,
    )

    hook.on_fit_start(_context(phase="fit_start", epoch=None))

    assert json.loads((tmp_path / "artifacts" / "probes" / "manifest.json").read_text(encoding="utf-8")) == manifest
