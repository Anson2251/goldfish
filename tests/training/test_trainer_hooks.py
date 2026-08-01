"""Tests for the Trainer hook lifecycle (fit_start / epoch_end / fit_end)."""

from __future__ import annotations

import torch
from torch import nn

from goldfish.observability.events import HookContext, ProbePhase
from goldfish.training import EpochResult, Trainer
from tests.training.test_trainer import RegressionModel, RegressionTask, make_loader


class RecordingHook:
    """Implements TrainingHook and records every received context."""

    def __init__(self, name: str = "hook") -> None:
        self.name = name
        self.events: list[tuple[str, HookContext]] = []

    def on_fit_start(self, context: HookContext) -> None:
        self.events.append(("fit_start", context))

    def on_epoch_end(self, context: HookContext) -> None:
        self.events.append(("epoch_end", context))

    def on_fit_end(self, context: HookContext) -> None:
        self.events.append(("fit_end", context))


def make_trainer(*hooks, **kwargs) -> Trainer:
    model = RegressionModel()
    return Trainer(
        model,
        RegressionTask(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        hooks=hooks,
        device="cpu",
        **kwargs,
    )


def test_hooks_receive_fit_start_once_then_epoch_end_per_epoch_then_fit_end() -> None:
    hook = RecordingHook()
    trainer = make_trainer(hook)

    trainer.fit(make_loader(), epochs=3)

    assert [phase for phase, _ in hook.events] == ["fit_start", "epoch_end", "epoch_end", "epoch_end", "fit_end"]


def test_fit_start_context_has_no_epoch_or_result() -> None:
    hook = RecordingHook()
    trainer = make_trainer(hook)
    trainer.fit(make_loader(), epochs=1)

    phase, context = hook.events[0]
    assert phase == "fit_start"
    assert context.phase == "fit_start"
    assert context.epoch is None
    assert context.result is None
    assert context.global_step == 0
    assert isinstance(context.model, nn.Module)
    assert context.scheduler is None


def test_epoch_end_context_matches_epoch_result() -> None:
    hook = RecordingHook()
    trainer = make_trainer(hook)
    result = trainer.fit(make_loader(), val_loader=make_loader(), epochs=2)

    epoch_events = [context for phase, context in hook.events if phase == "epoch_end"]
    assert len(epoch_events) == 2
    for index, context in enumerate(epoch_events):
        assert context.phase == "epoch_end"
        assert context.epoch == index
        assert context.result is result.history[index]
        assert context.global_step == 2 * (index + 1)
        assert context.result.validation is not None


def test_fit_end_context_carries_final_result() -> None:
    hook = RecordingHook()
    trainer = make_trainer(hook)
    result = trainer.fit(make_loader(), epochs=2)

    phase, context = hook.events[-1]
    assert phase == "fit_end"
    assert context.phase == "fit_end"
    assert context.epoch == 1
    assert context.result is result.history[-1]
    assert context.global_step == 4


def test_hooks_run_before_legacy_epoch_callback() -> None:
    order: list[str] = []

    class OrderHook(RecordingHook):
        def on_epoch_end(self, context: HookContext) -> None:
            order.append("hook")
            super().on_epoch_end(context)

    hook = OrderHook()

    def legacy_callback(_: EpochResult) -> None:
        order.append("legacy")

    trainer = make_trainer(hook, on_epoch_end=legacy_callback)
    trainer.fit(make_loader(), epochs=1)

    assert [phase for phase, _ in hook.events] == ["fit_start", "epoch_end", "fit_end"]
    assert order == ["hook", "legacy"]


def test_multiple_hooks_run_in_registration_order() -> None:
    first, second = RecordingHook("first"), RecordingHook("second")
    trainer = make_trainer(first, second)

    trainer.fit(make_loader(), epochs=1)

    assert [event[0] for event in first.events] == ["fit_start", "epoch_end", "fit_end"]
    assert [event[0] for event in second.events] == ["fit_start", "epoch_end", "fit_end"]


def test_resume_fit_start_reports_restored_global_step(tmp_path) -> None:
    model = RegressionModel()
    trainer = Trainer(
        model,
        RegressionTask(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        device="cpu",
    )
    trainer.fit(make_loader(), epochs=1)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)

    restored_model = RegressionModel()
    restored = Trainer(
        restored_model,
        RegressionTask(),
        torch.optim.SGD(restored_model.parameters(), lr=0.1),
        device="cpu",
    )
    restored.load_checkpoint(checkpoint)
    hook = RecordingHook()
    restored.hooks = (hook,)

    restored.fit(make_loader(), epochs=1)

    phase, context = hook.events[0]
    assert phase == "fit_start"
    assert context.global_step == 2
    assert context.epoch is None
    # The resumed epoch_end carries the restored counters.
    epoch_phase, epoch_context = hook.events[1]
    assert epoch_phase == "epoch_end"
    assert epoch_context.epoch == 1
    assert epoch_context.global_step == 4


def test_no_hooks_keeps_training_behavior_unchanged() -> None:
    trainer = make_trainer()
    result = trainer.fit(make_loader(), val_loader=make_loader(), epochs=2)

    assert len(result.history) == 2
    assert result.history[-1].validation is not None
