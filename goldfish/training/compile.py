"""Optional torch.compile integration with checkpoint-stable state dictionaries."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class CompiledModel(nn.Module):
    """Compile forward execution while retaining the original checkpoint layout."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.original_model = model
        self.compiled_forward = torch.compile(model)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.compiled_forward(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        return self.original_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, Tensor], *args: Any, **kwargs: Any) -> Any:
        return self.original_model.load_state_dict(state_dict, *args, **kwargs)


def compile_model(model: nn.Module) -> CompiledModel:
    """Return a checkpoint-compatible compiled model."""
    return CompiledModel(model)
