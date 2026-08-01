"""Append-only JSONL recorder and manifest writer for probe records."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from goldfish.observability.events import ProbePhase

SCHEMA_VERSION = 2


class JsonlRecorder:
    """Persist probe records as append-only JSONL under an artifacts/probes directory.

    Each probe gets its own ``<probe>.jsonl`` file; records are flushed and
    ``fsync``ed after every write. Records are validated for JSON serializability
    before the file is touched, so a rejected record never partially writes.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def write_record(
        self,
        probe: str,
        phase: ProbePhase,
        epoch: int,
        global_step: int,
        payload: Mapping[str, Any],
    ) -> None:
        """Append one probe record with the common envelope."""
        record = {
            "schema_version": SCHEMA_VERSION,
            "probe": probe,
            "phase": phase,
            "epoch": epoch,
            "global_step": global_step,
            "payload": dict(payload),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        path = self.directory / f"{probe}.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Write the resolved probe configuration manifest."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "manifest.json").write_text(
            json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
