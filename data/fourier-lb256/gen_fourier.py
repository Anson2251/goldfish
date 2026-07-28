"""Generate the checked-in Fourier-series numeric forecasting demo.

The target signal is deterministic and noiseless. It combines a linear trend,
a slow oscillation, and twelve Fourier components with distinct periods, phases,
and amplitudes. The short periods make one-step forecasting non-trivial, while
the incommensurate longer periods require a model to retain longer history.

``phase_sin`` and ``phase_cos`` expose a slow seasonal phase as observed
features. Files are deliberately partitioned into ordered shards so the demo
also exercises history across train-shard and split boundaries.
"""

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
START = datetime(2024, 1, 1, tzinfo=UTC)
ENTITY_ID = "fourier-demo"

# (relative path, first row index, number of rows)
SHARDS = (
    ("train/01-fourier.csv", 0, 10_000),
    ("train/02-fourier.csv", 10_000, 10_000),
    ("val/01-fourier.csv", 20_000, 4_000),
    ("test/01-fourier.csv", 24_000, 4_000),
)


COMPONENTS = (
    # (amplitude, period in rows, phase radians, use cosine)
    (1.00, 31.0, 0.15, False),
    (0.82, 47.0, 1.20, True),
    (0.68, 73.0, -0.70, False),
    (0.55, 109.0, 2.00, True),
    (0.43, 157.0, -1.30, False),
    (0.35, 233.0, 0.80, True),
    (0.28, 347.0, -2.10, False),
    (0.22, 521.0, 1.70, True),
    (0.18, 787.0, -0.40, False),
    (0.14, 1_181.0, 2.50, True),
    (0.11, 1_771.0, -1.80, False),
    (0.08, 2_657.0, 0.60, True),
)
SLOW_PERIOD = 4_093.0


def values(index: int) -> tuple[float, float, float, float]:
    """Return a multi-frequency Fourier-series signal and observed covariates."""
    phase = 2.0 * math.pi * index / SLOW_PERIOD
    trend = 0.00015 * index
    series = sum(
        amplitude * (math.cos(2.0 * math.pi * index / period + offset) if cosine else math.sin(2.0 * math.pi * index / period + offset))
        for amplitude, period, offset, cosine in COMPONENTS
    )
    signal = trend + 0.30 * math.sin(phase) + series
    return signal, trend, math.sin(phase), math.cos(phase)


def write_shard(relative_path: str, start: int, count: int) -> None:
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "entity_id", "signal", "trend", "phase_sin", "phase_cos"))
        for index in range(start, start + count):
            signal, trend, phase_sin, phase_cos = values(index)
            timestamp = (START + timedelta(minutes=index)).isoformat().replace("+00:00", "Z")
            writer.writerow((timestamp, ENTITY_ID, f"{signal:.12f}", f"{trend:.12f}", f"{phase_sin:.12f}", f"{phase_cos:.12f}"))


def main() -> None:
    for shard in SHARDS:
        write_shard(*shard)


if __name__ == "__main__":
    main()
