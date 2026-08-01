"""Tests for the JSONL probe recorder and manifest writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from goldfish.observability.recorder import JsonlRecorder


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_write_record_creates_directory_and_envelope(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path / "artifacts" / "probes")

    recorder.write_record("mixer-state", "fit_start", 0, 0, {"mixers": []})

    path = tmp_path / "artifacts" / "probes" / "mixer-state.jsonl"
    assert _records(path) == [
        {
            "schema_version": 2,
            "probe": "mixer-state",
            "phase": "fit_start",
            "epoch": 0,
            "global_step": 0,
            "payload": {"mixers": []},
        }
    ]


def test_write_record_appends_independent_json_lines(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path / "artifacts" / "probes")
    recorder.write_record("mixer-state", "fit_start", 0, 0, {"a": 1})
    recorder.write_record("mixer-state", "epoch_end", 1, 10, {"a": 2})

    records = _records(tmp_path / "artifacts" / "probes" / "mixer-state.jsonl")
    assert [record["epoch"] for record in records] == [0, 1]
    assert records[1]["payload"] == {"a": 2}


def test_different_probes_write_different_files(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path / "artifacts" / "probes")
    recorder.write_record("mixer-state", "fit_start", 0, 0, {})
    recorder.write_record("activation-stats", "fit_start", 0, 0, {})

    assert (tmp_path / "artifacts" / "probes" / "mixer-state.jsonl").is_file()
    assert (tmp_path / "artifacts" / "probes" / "activation-stats.jsonl").is_file()


def test_write_manifest_writes_json_file(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path / "artifacts" / "probes")
    manifest = {"schema_version": 2, "probes": [{"name": "mixer-state"}]}

    recorder.write_manifest(manifest)

    path = tmp_path / "artifacts" / "probes" / "manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_recorder_rejects_non_json_payload(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path / "artifacts" / "probes")

    with pytest.raises(TypeError):
        recorder.write_record("mixer-state", "epoch_end", 1, 1, {"matrix": torch.zeros(4, 4)})

    path = tmp_path / "artifacts" / "probes" / "mixer-state.jsonl"
    assert not path.exists()
