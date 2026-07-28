"""Shared DataLoader resource planning and construction options."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoaderSettings:
    """Resolved worker and host-to-device transfer settings for one data module."""

    train_workers: int
    validation_workers: int
    test_workers: int
    pin_memory: bool
    prefetch_factor: int | None
    persistent_workers: bool


def resolve_loader_settings(
    *,
    device: torch.device | str | None = None,
    num_workers: int | None = None,
    train_workers: int | None = None,
    validation_workers: int | None = None,
    cpu_count: int | None = None,
    prefetch_factor: int = 2,
) -> LoaderSettings:
    """Resolve CPU-safe loader settings with a 60:40 train/validation budget split.

    ``num_workers=None`` means auto: reserve 20% of logical CPUs for the
    operating system, then allocate the remaining 80% 60:40 between training
    and validation. Train and validation execute sequentially, so the split is
    a phase-specific cap rather than simultaneous CPU consumption.
    """
    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if train_workers is not None and train_workers < 0:
        raise ValueError("train_workers must be non-negative.")
    if validation_workers is not None and validation_workers < 0:
        raise ValueError("validation_workers must be non-negative.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive.")
    cores = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    budget = max(1, int(cores * 0.8)) if num_workers is None else num_workers
    default_train = int(budget * 0.6)
    default_validation = budget - default_train
    if budget > 1:
        default_train = max(1, default_train)
        default_validation = max(1, default_validation)
    resolved_train = default_train if train_workers is None else train_workers
    resolved_validation = default_validation if validation_workers is None else validation_workers
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    is_cuda = resolved_device.type == "cuda"
    max_workers = max(resolved_train, resolved_validation)
    return LoaderSettings(
        train_workers=resolved_train,
        validation_workers=resolved_validation,
        test_workers=resolved_validation,
        pin_memory=is_cuda,
        prefetch_factor=prefetch_factor if max_workers > 0 else None,
        persistent_workers=max_workers > 0,
    )
