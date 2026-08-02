"""Deterministic reference batch capture for activation probes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

BatchT = TypeVar("BatchT")


def take_first_batches(iterator: Iterator[BatchT], batches: int) -> tuple[BatchT, ...]:
    """Take the first ``batches`` items from a fresh split iterator.

    Raises ``ValueError`` when the split yields fewer batches than requested, so
    a misconfigured reference never silently produces a smaller input set. The
    error reports the available batch count and the fix.
    """
    selected: list[BatchT] = []
    for _ in range(batches):
        try:
            selected.append(next(iterator))
        except StopIteration:
            available = len(selected)
            raise ValueError(
                f"reference split yielded {available} batch" + ("" if available == 1 else "es")
                + f", fewer than the configured {batches}; "
                f"reduce --observability-batches to {available} or fewer"
            ) from None
    return tuple(selected)
