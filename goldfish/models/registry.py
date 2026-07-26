"""Registry for discoverable, configurable model compositions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

ModelFactory = Callable[..., nn.Module]


class ModelRegistry:
    """Map stable model names to factory callables.

    Registries are scoped by model family so, for example, text language-model
    names do not collide with future numeric forecasting model names.
    """

    def __init__(self) -> None:
        self._families: dict[str, dict[str, ModelFactory]] = {}

    def register(self, family: str, name: str, factory: ModelFactory) -> None:
        normalized_family = self._normalize(family, "family")
        normalized_name = self._normalize(name, "name")
        models = self._families.setdefault(normalized_family, {})
        if normalized_name in models:
            raise ValueError(f"model {normalized_name!r} is already registered for family {normalized_family!r}")
        models[normalized_name] = factory

    def create(self, family: str, name: str, **kwargs: Any) -> nn.Module:
        return self.get(family, name)(**kwargs)

    def get(self, family: str, name: str) -> ModelFactory:
        normalized_family = self._normalize(family, "family")
        normalized_name = self._normalize(name, "name")
        try:
            return self._families[normalized_family][normalized_name]
        except KeyError as error:
            available = ", ".join(self.names(normalized_family)) or "none"
            raise ValueError(
                f"unknown {normalized_family!r} model {normalized_name!r}; available: {available}"
            ) from error

    def names(self, family: str) -> tuple[str, ...]:
        normalized_family = self._normalize(family, "family")
        return tuple(sorted(self._families.get(normalized_family, {})))

    @staticmethod
    def _normalize(value: str, kind: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if not normalized:
            raise ValueError(f"{kind} must not be empty")
        return normalized


model_registry = ModelRegistry()
