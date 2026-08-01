"""Named-pattern module discovery with identity deduplication."""

from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatchcase

from torch import nn


def discover_modules(model: nn.Module, patterns: Sequence[str]) -> tuple[tuple[str, nn.Module], ...]:
    """Return canonical ``(path, module)`` pairs matching any pattern.

    Patterns are globs matched per dotted segment, so ``mixers.*`` matches
    ``mixers.0`` but not ``mixers.0.1``. Modules exposed under several paths
    (for example a backward-compatible alias) are deduplicated by object
    identity, keeping the canonical longest path. Results are sorted by path
    for determinism.
    """
    matches: dict[int, tuple[str, nn.Module]] = {}
    for path, module in model.named_modules():
        if path == "":
            continue
        if any(_segment_match(pattern, path) for pattern in patterns):
            existing = matches.get(id(module))
            if existing is None or len(path) > len(existing[0]):
                matches[id(module)] = (path, module)
    return tuple(sorted(matches.values(), key=lambda item: item[0]))


def _segment_match(pattern: str, path: str) -> bool:
    pattern_parts = pattern.split(".")
    path_parts = path.split(".")
    if len(pattern_parts) != len(path_parts):
        return False
    return all(fnmatchcase(path_part, pattern_part) for path_part, pattern_part in zip(path_parts, pattern_parts, strict=True))
