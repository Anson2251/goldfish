from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from goldfish.core import ModelOutput, StepResult
from goldfish.training import EpochResult, Trainer


@dataclass
class RegressionBatch:
    inputs: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> "RegressionBatch":
        return RegressionBatch(self.inputs.to(device), self.targets.to(device))


def collate(rows: list[tuple[torch.Tensor, torch.Tensor]]) -> RegressionBatch:
    inputs, targets = zip(*rows, strict=True)
    return RegressionBatch(torch.stack(inputs), torch.stack(targets))


class RegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, batch: RegressionBatch) -> ModelOutput:
        return ModelOutput(predictions={"value": self.linear(batch.inputs)})


class RegressionTask:
    def compute(self, output: ModelOutput, batch: RegressionBatch) -> StepResult:
        loss = torch.mean((output.predictions["value"] - batch.targets) ** 2)
        return StepResult(loss=loss, metrics={"mae": torch.mean(torch.abs(output.predictions["value"] - batch.targets))})


def make_loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    rows = [(torch.tensor([x]), torch.tensor([2.0 * x])) for x in (1.0, 2.0, 3.0, 4.0)]
    return DataLoader(rows, batch_size=2, shuffle=False, collate_fn=collate)


def test_fit_trains_and_records_train_and_validation_metrics() -> None:
    torch.manual_seed(0)
    model = RegressionModel()
    trainer = Trainer(model, RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")

    result = trainer.fit(make_loader(), val_loader=make_loader(), epochs=3)

    assert len(result.history) == 3
    assert result.history[-1].epoch == 2
    assert result.history[-1].global_step == 6
    assert result.history[-1].train["loss"] < result.history[0].train["loss"]
    assert result.history[-1].validation is not None
    assert "mae" in result.history[-1].validation
    assert result.history[-1].validation["loss"] >= 0


def test_gradient_clipping_limits_parameter_update() -> None:
    model = RegressionModel()
    model.linear.weight.data.zero_()
    trainer = Trainer(
        model,
        RegressionTask(),
        torch.optim.SGD(model.parameters(), lr=1.0),
        device="cpu",
        gradient_clip_norm=0.5,
    )

    trainer.train_epoch(make_loader())

    assert 0.9 <= model.linear.weight.item() <= 1.0


def test_checkpoint_resume_restores_model_optimizer_epoch_and_global_step(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = RegressionModel()
    trainer = Trainer(model, RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    trainer.fit(make_loader(), epochs=1)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    saved_weight = model.linear.weight.detach().clone()

    restored_model = RegressionModel()
    restored = Trainer(restored_model, RegressionTask(), torch.optim.SGD(restored_model.parameters(), lr=0.1), device="cpu")
    state = restored.load_checkpoint(checkpoint)

    assert state.epoch == 0
    assert state.global_step == 2
    assert torch.equal(restored_model.linear.weight.detach(), saved_weight)

    result = restored.fit(make_loader(), epochs=1)

    assert result.history[0].epoch == 1
    assert result.history[0].global_step == 4


class RecordingScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.calls: list[float | None] = []
        self.value = 0

    def step(self, metric: float | None = None) -> None:
        self.calls.append(metric)
        self.value += 1
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.5

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.value = int(state_dict["value"])


def test_batch_scheduler_steps_after_each_optimizer_update_and_records_learning_rate() -> None:
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingScheduler(optimizer)
    trainer = Trainer(
        model,
        RegressionTask(),
        optimizer,
        scheduler=scheduler,
        scheduler_step_timing="batch",
        device="cpu",
    )

    result = trainer.fit(make_loader())

    assert scheduler.calls == [None, None]
    assert result.history[0].learning_rates == (0.025,)


def test_epoch_scheduler_steps_after_training_epoch_before_epoch_callback() -> None:
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingScheduler(optimizer)
    observed: list[EpochResult] = []
    trainer = Trainer(
        model,
        RegressionTask(),
        optimizer,
        scheduler=scheduler,
        scheduler_step_timing="epoch",
        on_epoch_end=observed.append,
        device="cpu",
    )

    result = trainer.fit(make_loader())

    assert scheduler.calls == [None]
    assert observed == result.history
    assert observed[0].learning_rates == (0.05,)


def test_validation_scheduler_receives_configured_validation_metric() -> None:
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingScheduler(optimizer)
    trainer = Trainer(
        model,
        RegressionTask(),
        optimizer,
        scheduler=scheduler,
        scheduler_step_timing="validation",
        scheduler_metric="validation/mae",
        device="cpu",
    )

    result = trainer.fit(make_loader(), val_loader=make_loader())

    assert result.history[0].validation is not None
    assert scheduler.calls == [result.history[0].validation["mae"]]


def test_scheduler_checkpoint_state_is_restored_when_a_scheduler_is_configured(tmp_path: Path) -> None:
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = RecordingScheduler(optimizer)
    trainer = Trainer(
        model,
        RegressionTask(),
        optimizer,
        scheduler=scheduler,
        scheduler_step_timing="epoch",
        device="cpu",
    )
    trainer.fit(make_loader())
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)

    restored_model = RegressionModel()
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = RecordingScheduler(restored_optimizer)
    restored = Trainer(
        restored_model,
        RegressionTask(),
        restored_optimizer,
        scheduler=restored_scheduler,
        scheduler_step_timing="epoch",
        device="cpu",
    )
    restored.load_checkpoint(checkpoint)

    assert restored_scheduler.value == 1


def test_fit_rejects_non_positive_epoch_count() -> None:
    model = RegressionModel()
    trainer = Trainer(model, RegressionTask(), torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")

    with pytest.raises(ValueError, match="epochs"):
        trainer.fit(make_loader(), epochs=0)
