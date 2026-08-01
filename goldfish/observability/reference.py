"""Deterministic reference batch capture for activation probes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

BatchT = TypeVar("BatchT")


def take_first_batches(iterator: Iterator[BatchT], batches: int) -> tuple[BatchT, ...]:
    """Take the first ``batches`` items from a fresh split iterator.

    Raises ``ValueError`` when the split yields fewer batches than requested, so
    a misconfigured reference never silently produces a smaller input set.
    """
    selected: list[BatchT] = []
    for _ in range(batches):
        try:
            selected.append(next(iterator))
        except StopIteration:
            raise ValueError(f"reference split yielded fewer than {batches} batches") from None
    return tuple(selected)
