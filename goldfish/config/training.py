"""Validated training optimization and scheduler configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import yaml

OptimizerName: TypeAlias = Literal["adamw", "adam", "sgd"]
SchedulerName: TypeAlias = Literal["none", "cosine", "step", "exponential", "plateau"]
StepTiming: TypeAlias = Literal["batch", "epoch", "validation"]


@dataclass(frozen=True)
class AdamConfig:
    name: Literal["adam", "adamw"]
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    fused: bool | None = None


@dataclass(frozen=True)
class SGDConfig:
    name: Literal["sgd"] = "sgd"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    maximize: bool = False
    foreach: bool | None = None
    fused: bool | None = None


OptimizerConfig: TypeAlias = AdamConfig | SGDConfig


@dataclass(frozen=True)
class NoSchedulerConfig:
    name: Literal["none"] = "none"
    step_timing: None = None


@dataclass(frozen=True)
class CosineSchedulerConfig:
    name: Literal["cosine"] = "cosine"
    step_timing: Literal["batch", "epoch"] = "epoch"
    t_max: int = 1
    eta_min: float = 0.0
    last_epoch: int = -1


@dataclass(frozen=True)
class StepSchedulerConfig:
    name: Literal["step"] = "step"
    step_timing: Literal["epoch"] = "epoch"
    step_size: int = 1
    gamma: float = 0.1
    last_epoch: int = -1


@dataclass(frozen=True)
class ExponentialSchedulerConfig:
    name: Literal["exponential"] = "exponential"
    step_timing: Literal["epoch"] = "epoch"
    gamma: float = 0.1
    last_epoch: int = -1


@dataclass(frozen=True)
class PlateauSchedulerConfig:
    name: Literal["plateau"] = "plateau"
    step_timing: Literal["validation"] = "validation"
    monitor: str = "validation/loss"
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-8


SchedulerConfig: TypeAlias = (
    NoSchedulerConfig | CosineSchedulerConfig | StepSchedulerConfig | ExponentialSchedulerConfig | PlateauSchedulerConfig
)


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """The fully materialized v1 optimizer and scheduler settings for a run."""

    optimization: OptimizerConfig
    scheduler: SchedulerConfig

    def to_mapping(self) -> dict[str, dict[str, Any]]:
        """Return YAML-safe mappings with every effective setting present."""
        return {"optimization": asdict(self.optimization), "scheduler": asdict(self.scheduler)}


def resolve_training_config(overrides: Mapping[str, Any] | None = None) -> ResolvedTrainingConfig:
    """Resolve v1 optimizer and scheduler settings from nested CLI-style overrides.

    Only the ``optimization`` and ``scheduler`` sections are in scope. Unknown and
    optimizer/scheduler-inapplicable fields are rejected rather than silently ignored.
    """
    source = _mapping(overrides or {}, "overrides")
    _reject_unknown(source, {"optimization", "scheduler"}, "overrides")
    optimization = _resolve_optimizer(_section(source, "optimization"))
    scheduler = _resolve_scheduler(_section(source, "scheduler"))
    return ResolvedTrainingConfig(optimization=optimization, scheduler=scheduler)


def dump_resolved_config(config: ResolvedTrainingConfig) -> str:
    """Serialize a resolved config as stable, human-readable YAML."""
    if not isinstance(config, ResolvedTrainingConfig):
        raise TypeError("config must be a ResolvedTrainingConfig")
    return yaml.safe_dump(config.to_mapping(), sort_keys=False)


def _resolve_optimizer(values: Mapping[str, Any]) -> OptimizerConfig:
    name = values.get("name", "adamw")
    if name not in {"adamw", "adam", "sgd"}:
        raise ValueError(f"optimization.name must be one of adamw, adam, sgd; got {name!r}")
    if name in {"adamw", "adam"}:
        allowed = {"name", "learning_rate", "weight_decay", "betas", "eps", "amsgrad", "maximize", "foreach", "fused"}
        _reject_unknown(values, allowed, "optimization")
        defaults = AdamConfig(name=name, weight_decay=0.0001 if name == "adamw" else 0.0)
        config = AdamConfig(
            name=name,
            learning_rate=_positive_float(values.get("learning_rate", defaults.learning_rate), "optimization.learning_rate"),
            weight_decay=_nonnegative_float(values.get("weight_decay", defaults.weight_decay), "optimization.weight_decay"),
            betas=_betas(values.get("betas", defaults.betas)),
            eps=_positive_float(values.get("eps", defaults.eps), "optimization.eps"),
            amsgrad=_bool(values.get("amsgrad", defaults.amsgrad), "optimization.amsgrad"),
            maximize=_bool(values.get("maximize", defaults.maximize), "optimization.maximize"),
            foreach=_optional_bool(values.get("foreach", defaults.foreach), "optimization.foreach"),
            fused=_optional_bool(values.get("fused", defaults.fused), "optimization.fused"),
        )
        return config

    allowed = {"name", "learning_rate", "weight_decay", "momentum", "dampening", "nesterov", "maximize", "foreach", "fused"}
    _reject_unknown(values, allowed, "optimization")
    config = SGDConfig(
        learning_rate=_positive_float(values.get("learning_rate", 0.001), "optimization.learning_rate"),
        weight_decay=_nonnegative_float(values.get("weight_decay", 0.0), "optimization.weight_decay"),
        momentum=_nonnegative_float(values.get("momentum", 0.0), "optimization.momentum"),
        dampening=_nonnegative_float(values.get("dampening", 0.0), "optimization.dampening"),
        nesterov=_bool(values.get("nesterov", False), "optimization.nesterov"),
        maximize=_bool(values.get("maximize", False), "optimization.maximize"),
        foreach=_optional_bool(values.get("foreach", None), "optimization.foreach"),
        fused=_optional_bool(values.get("fused", None), "optimization.fused"),
    )
    if config.nesterov and (config.momentum <= 0 or config.dampening != 0):
        raise ValueError("optimization.nesterov requires positive momentum and zero dampening")
    return config


def _resolve_scheduler(values: Mapping[str, Any]) -> SchedulerConfig:
    name = values.get("name", "none")
    timings: dict[str, set[StepTiming | None]] = {
        "none": {None}, "cosine": {"batch", "epoch"}, "step": {"epoch"}, "exponential": {"epoch"}, "plateau": {"validation"}
    }
    if name not in timings:
        raise ValueError(f"scheduler.name must be one of {', '.join(timings)}; got {name!r}")
    timing = values.get("step_timing", None if name == "none" else _default_timing(name))
    if timing not in timings[name]:
        expected = ", ".join(repr(item) for item in sorted(timings[name], key=str))
        raise ValueError(f"scheduler.step_timing for {name!r} must be {expected}; got {timing!r}")
    if name == "none":
        _reject_unknown(values, {"name", "step_timing"}, "scheduler")
        return NoSchedulerConfig()
    if name == "cosine":
        _reject_unknown(values, {"name", "step_timing", "t_max", "eta_min", "last_epoch"}, "scheduler")
        return CosineSchedulerConfig(step_timing="batch" if timing == "batch" else "epoch", t_max=_positive_int(values.get("t_max", 1), "scheduler.t_max"), eta_min=_nonnegative_float(values.get("eta_min", 0.0), "scheduler.eta_min"), last_epoch=_last_epoch(values.get("last_epoch", -1)))
    if name == "step":
        _reject_unknown(values, {"name", "step_timing", "step_size", "gamma", "last_epoch"}, "scheduler")
        return StepSchedulerConfig(step_size=_positive_int(values.get("step_size", 1), "scheduler.step_size"), gamma=_positive_float(values.get("gamma", 0.1), "scheduler.gamma"), last_epoch=_last_epoch(values.get("last_epoch", -1)))
    if name == "exponential":
        _reject_unknown(values, {"name", "step_timing", "gamma", "last_epoch"}, "scheduler")
        return ExponentialSchedulerConfig(gamma=_positive_float(values.get("gamma", 0.1), "scheduler.gamma"), last_epoch=_last_epoch(values.get("last_epoch", -1)))
    _reject_unknown(values, {"name", "step_timing", "monitor", "mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lr", "eps"}, "scheduler")
    mode = values.get("mode", "min")
    threshold_mode = values.get("threshold_mode", "rel")
    if mode not in {"min", "max"}:
        raise ValueError("scheduler.mode must be 'min' or 'max'")
    if threshold_mode not in {"rel", "abs"}:
        raise ValueError("scheduler.threshold_mode must be 'rel' or 'abs'")
    monitor = values.get("monitor", "validation/loss")
    if not isinstance(monitor, str) or not monitor:
        raise ValueError("scheduler.monitor must be a non-empty string")
    return PlateauSchedulerConfig(monitor=monitor, mode=mode, factor=_positive_float(values.get("factor", 0.1), "scheduler.factor"), patience=_nonnegative_int(values.get("patience", 10), "scheduler.patience"), threshold=_nonnegative_float(values.get("threshold", 1e-4), "scheduler.threshold"), threshold_mode=threshold_mode, cooldown=_nonnegative_int(values.get("cooldown", 0), "scheduler.cooldown"), min_lr=_nonnegative_float(values.get("min_lr", 0.0), "scheduler.min_lr"), eps=_positive_float(values.get("eps", 1e-8), "scheduler.eps"))


def _default_timing(name: str) -> StepTiming:
    return "validation" if name == "plateau" else "epoch"


def _section(source: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(source.get(name, {}), name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    for key in values:
        if key not in allowed:
            raise ValueError(f"{section} has unknown field {key!r} or it is not applicable to the selected schema")


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_bool(value: Any, name: str) -> bool | None:
    return None if value is None else _bool(value, name)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _last_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError("scheduler.last_epoch must be an integer greater than or equal to -1")
    return value


def _positive_float(value: Any, name: str) -> float:
    result = _float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("optimization.betas must contain exactly two values")
    beta1 = _float(value[0], "optimization.betas[0]")
    beta2 = _float(value[1], "optimization.betas[1]")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("optimization.betas values must be in [0, 1)")
    return beta1, beta2
