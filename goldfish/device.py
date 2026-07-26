"""Runtime device selection shared by training and inference entry points."""

from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["cpu", "cuda", "mps"]


def best_available_device() -> DeviceName:
    """Return the preferred supported device for the current platform.

    Goldfish prefers CUDA, then Apple Metal Performance Shaders, then CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: DeviceName | None = None) -> torch.device:
    """Resolve a user request, or choose the best available platform device.

    Explicit unavailable devices fail rather than silently falling back, so a user
    cannot accidentally run a requested accelerator experiment on CPU.
    """
    selected = best_available_device() if requested is None else requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if selected == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    return torch.device(selected)
