"""PyTorch optimizer and learning-rate scheduler factories."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .training import (
    AdamConfig,
    CosineSchedulerConfig,
    ExponentialSchedulerConfig,
    OptimizerConfig,
    PlateauSchedulerConfig,
    SchedulerConfig,
    SGDConfig,
    StepSchedulerConfig,
)


def create_optimizer(parameters: Iterable[Tensor], config: OptimizerConfig) -> Optimizer:
    """Build the configured PyTorch optimizer for one parameter group."""
    if isinstance(config, AdamConfig):
        if config.name == "adamw":
            return torch.optim.AdamW(
                parameters, lr=config.learning_rate, betas=config.betas, eps=config.eps,
                weight_decay=config.weight_decay, amsgrad=config.amsgrad, maximize=config.maximize,
                foreach=config.foreach, fused=config.fused,
            )
        return torch.optim.Adam(
            parameters, lr=config.learning_rate, betas=config.betas, eps=config.eps,
            weight_decay=config.weight_decay, amsgrad=config.amsgrad, maximize=config.maximize,
            foreach=config.foreach, fused=config.fused,
        )
    if isinstance(config, SGDConfig):
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            dampening=config.dampening,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
            maximize=config.maximize,
            foreach=config.foreach,
            fused=config.fused,
        )
    raise TypeError(f"unsupported optimizer config: {type(config).__name__}")


def create_scheduler(optimizer: Optimizer, config: SchedulerConfig) -> LRScheduler | ReduceLROnPlateau | None:
    """Build the configured scheduler; ``none`` intentionally returns ``None``."""
    if config.name == "none":
        return None
    if isinstance(config, CosineSchedulerConfig):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=config.eta_min, last_epoch=config.last_epoch)
    if isinstance(config, StepSchedulerConfig):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma, last_epoch=config.last_epoch)
    if isinstance(config, ExponentialSchedulerConfig):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma, last_epoch=config.last_epoch)
    if isinstance(config, PlateauSchedulerConfig):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
        )
    raise TypeError(f"unsupported scheduler config: {type(config).__name__}")
