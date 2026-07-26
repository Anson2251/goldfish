from __future__ import annotations

import pytest
import torch
from torch import nn

from goldfish.config import create_optimizer, create_scheduler, resolve_training_config


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)],
)
def test_create_optimizer_builds_requested_torch_optimizer(name: str, expected_type: type[torch.optim.Optimizer]) -> None:
    model = nn.Linear(2, 1)
    config = resolve_training_config({"optimization": {"name": name}}).optimization

    optimizer = create_optimizer(model.parameters(), config)

    assert isinstance(optimizer, expected_type)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config.learning_rate)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(config.weight_decay)


@pytest.mark.parametrize(
    ("scheduler_config", "expected_type"),
    [
        ({"name": "none"}, type(None)),
        ({"name": "cosine", "t_max": 5}, torch.optim.lr_scheduler.CosineAnnealingLR),
        ({"name": "step", "step_size": 5}, torch.optim.lr_scheduler.StepLR),
        ({"name": "exponential"}, torch.optim.lr_scheduler.ExponentialLR),
        ({"name": "plateau"}, torch.optim.lr_scheduler.ReduceLROnPlateau),
    ],
)
def test_create_scheduler_builds_requested_torch_scheduler(
    scheduler_config: dict[str, object], expected_type: type[object]
) -> None:
    model = nn.Linear(2, 1)
    resolved = resolve_training_config({"scheduler": scheduler_config})
    optimizer = create_optimizer(model.parameters(), resolved.optimization)

    scheduler = create_scheduler(optimizer, resolved.scheduler)

    assert isinstance(scheduler, expected_type)
