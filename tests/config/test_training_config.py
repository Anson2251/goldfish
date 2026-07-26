from __future__ import annotations

import yaml
import pytest

from goldfish.config import dump_resolved_config, resolve_training_config


def test_resolve_training_config_materializes_adamw_and_none_defaults() -> None:
    config = resolve_training_config({})

    assert config.optimization.name == "adamw"
    assert config.optimization.learning_rate == pytest.approx(0.001)
    assert config.optimization.weight_decay == pytest.approx(0.0001)
    assert config.optimization.betas == (0.9, 0.999)
    assert config.optimization.eps == pytest.approx(1e-8)
    assert config.optimization.amsgrad is False
    assert config.optimization.maximize is False
    assert config.optimization.foreach is None
    assert config.optimization.fused is None
    assert config.scheduler.name == "none"
    assert config.scheduler.step_timing is None


def test_resolution_accepts_nested_overrides_and_only_retains_applicable_fields() -> None:
    config = resolve_training_config(
        {
            "optimization": {
                "name": "sgd",
                "learning_rate": 0.1,
                "momentum": 0.9,
                "nesterov": True,
            },
            "scheduler": {"name": "step", "step_timing": "epoch", "step_size": 4, "gamma": 0.5},
        }
    )

    assert config.optimization.name == "sgd"
    assert config.optimization.learning_rate == pytest.approx(0.1)
    assert config.optimization.weight_decay == pytest.approx(0.0)
    assert config.optimization.momentum == pytest.approx(0.9)
    assert config.optimization.dampening == pytest.approx(0.0)
    assert config.optimization.nesterov is True
    assert not hasattr(config.optimization, "betas")
    assert config.scheduler.step_size == 4
    assert config.scheduler.gamma == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        ({"name": "cosine", "step_timing": "validation", "t_max": 4}, "step_timing"),
        ({"name": "step", "step_timing": "batch", "step_size": 2}, "step_timing"),
        ({"name": "plateau", "step_timing": "epoch"}, "step_timing"),
        ({"name": "none", "step_timing": "epoch"}, "step_timing"),
    ],
)
def test_resolution_rejects_invalid_scheduler_timing(scheduler: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_training_config({"scheduler": scheduler})


def test_resolution_rejects_unknown_or_inapplicable_fields() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        resolve_training_config({"optimization": {"name": "adam", "bogus": True}})

    with pytest.raises(ValueError, match="not applicable"):
        resolve_training_config({"optimization": {"name": "adam", "momentum": 0.9}})


def test_dump_resolved_config_is_loadable_and_contains_explicit_defaults() -> None:
    dumped = dump_resolved_config(resolve_training_config({"scheduler": {"name": "cosine", "t_max": 10}}))

    document = yaml.safe_load(dumped)
    assert document["optimization"]["weight_decay"] == pytest.approx(0.0001)
    assert document["scheduler"] == {
        "name": "cosine",
        "step_timing": "epoch",
        "t_max": 10,
        "eta_min": 0.0,
        "last_epoch": -1,
    }
