"""Probe protocol and registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from goldfish.observability.events import HookContext


class Probe(Protocol):
    """A read-only plugin extracting a JSON-serializable observation."""

    name: str

    def collect(self, context: HookContext) -> Mapping[str, Any] | None:
        """Return the JSON-serializable payload, or None when there is no observation."""
        ...


ProbeFactory = Callable[[Mapping[str, Any]], Probe]


class ProbeRegistry:
    """Map stable probe names to factory callables."""

    def __init__(self) -> None:
        self._factories: dict[str, ProbeFactory] = {}

    def register(self, name: str, factory: ProbeFactory) -> None:
        normalized = self._normalize(name)
        if normalized in self._factories:
            raise ValueError(f"probe {normalized!r} is already registered")
        self._factories[normalized] = factory

    def create(self, name: str, options: Mapping[str, Any]) -> Probe:
        normalized = self._normalize(name)
        try:
            return self._factories[normalized](options)
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(f"unknown probe {normalized!r}; available: {available}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if not normalized:
            raise ValueError("probe name must not be empty")
        return normalized


probe_registry = ProbeRegistry()
