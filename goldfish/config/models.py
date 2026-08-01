"""Model profile loading, runtime resolution, and registry construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from goldfish.models import model_registry


def load_model_profile(path: Path) -> dict[str, Any]:
    """Load a model profile containing a family, name, and architecture parameters."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Model profile not found: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid model profile: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("Model profile must be a mapping")
    model = value.get("model", value)
    if not isinstance(model, Mapping):
        raise ValueError("Model profile must contain a model mapping")
    family, name, parameters = model.get("family"), model.get("name"), model.get("parameters")
    if not isinstance(family, str) or not family:
        raise ValueError("Model profile family must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("Model profile name must be a non-empty string")
    if not isinstance(parameters, Mapping):
        raise ValueError("Model profile parameters must be a mapping")
    resolved = {"family": family, "name": name, "parameters": dict(parameters)}
    if "observability" in value:
        resolved["observability"] = value["observability"]
    return resolved


def resolve_model_config(profile: Mapping[str, Any], *, family: str, runtime_parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Merge dataset-derived dimensions into a profile and validate its family."""
    if profile.get("family") != family:
        raise ValueError(f"Model profile family must be {family!r}")
    parameters = profile.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Model profile parameters must be a mapping")
    overlap = set(parameters).intersection(runtime_parameters)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Model profile must not set dataset-derived parameters: {names}")
    return {"family": family, "name": profile["name"], "parameters": {**parameters, **runtime_parameters}}


def create_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Create a registered model from a resolved or legacy flat model configuration."""
    family, name = config.get("family"), config.get("name")
    if not isinstance(family, str) or not isinstance(name, str):
        raise ValueError("Model config must contain string 'family' and 'name' fields.")

    parameters = config.get("parameters")
    if parameters is None:
        # Runs created before model profiles stored constructor arguments directly
        # under ``model``. Keep those managed checkpoints usable after migration.
        parameters = {key: value for key, value in config.items() if key not in {"family", "name", "parameters"}}
    if not isinstance(parameters, Mapping):
        raise ValueError(
            f"Model config for {family!r}/{name!r} has an invalid 'parameters' field; "
            "expected a mapping of constructor arguments."
        )
    if not parameters:
        raise ValueError(
            f"Model config for {family!r}/{name!r} has no constructor parameters. "
            "Use a run created with --model-profile, or add a 'parameters' mapping to config.yaml."
        )
    try:
        return model_registry.create(family, name, **parameters)
    except TypeError as error:
        raise ValueError(
            f"Cannot construct {family!r}/{name!r} from this run's model configuration: {error}"
        ) from error
