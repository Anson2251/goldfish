"""A small, task- and modality-agnostic PyTorch trainer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

from goldfish.core import Batch, ModelOutput, Task


@dataclass(frozen=True)
class EpochResult:
    """Aggregated metrics recorded after one training epoch."""

    epoch: int
    global_step: int
    train: dict[str, float]
    validation: dict[str, float] | None = None
    learning_rates: tuple[float, ...] = ()


class Scheduler(Protocol):
    """Minimal scheduler contract accepted by :class:`Trainer`."""

    def step(self, metric: float | None = None) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...


EpochCallback = Callable[[EpochResult], None]
SchedulerStepTiming = Literal["batch", "epoch", "validation"]


@dataclass(frozen=True)
class FitResult:
    """History and final counters from a call to :meth:`Trainer.fit`."""

    history: list[EpochResult]
    epoch: int
    global_step: int


@dataclass(frozen=True)
class CheckpointState:
    """Training counters restored from a checkpoint."""

    epoch: int
    global_step: int


BatchT = TypeVar("BatchT", bound=Batch)


class Trainer(Generic[BatchT]):
    """Train a torch model using the generic ``Batch -> Output -> Task`` lifecycle."""

    def __init__(
        self,
        model: nn.Module,
        task: Task[BatchT],
        optimizer: Optimizer,
        *,
        scheduler: Scheduler | None = None,
        scheduler_step_timing: SchedulerStepTiming | None = None,
        scheduler_metric: str | None = None,
        on_epoch_end: EpochCallback | None = None,
        device: torch.device | str | None = None,
        gradient_clip_norm: float | None = None,
        aux_loss_weights: Mapping[str, float] | None = None,
        progress: bool = True,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if scheduler is None and (scheduler_step_timing is not None or scheduler_metric is not None):
            raise ValueError("scheduler_step_timing and scheduler_metric require a scheduler")
        if scheduler is not None and scheduler_step_timing is None:
            raise ValueError("scheduler_step_timing is required when a scheduler is configured")
        if scheduler_step_timing not in (None, "batch", "epoch", "validation"):
            raise ValueError("scheduler_step_timing must be 'batch', 'epoch', or 'validation'")
        if scheduler_step_timing == "validation" and scheduler_metric is None:
            raise ValueError("scheduler_metric is required for validation scheduler stepping")
        if scheduler_step_timing != "validation" and scheduler_metric is not None:
            raise ValueError("scheduler_metric is only valid for validation scheduler stepping")

        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_timing = scheduler_step_timing
        self.scheduler_metric = scheduler_metric
        self.on_epoch_end = on_epoch_end
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.gradient_clip_norm = gradient_clip_norm
        self.aux_loss_weights = dict(aux_loss_weights or {})
        self.progress = progress
        self.epoch = -1
        self.global_step = 0
        self._display_final_epoch: int | None = None
        self.model.to(self.device)

    def train_epoch(self, loader: Iterable[BatchT]) -> dict[str, float]:
        """Run one optimization epoch and return mean batch metrics."""
        self.model.train()
        metrics: list[dict[str, float]] = []
        progress = self._batches(loader, description=self._progress_description("train"), leave=True)
        for batch in progress:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch)
            result = self.task.compute(output, batch)
            loss = self._total_loss(result.loss, output)
            loss.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            if self.scheduler_step_timing == "batch":
                self._step_scheduler()
            self.global_step += 1
            metrics.append(self._as_metrics(loss, result.metrics))
            progress.set_postfix(self._progress_metrics(metrics, include_learning_rate=True))
        return self._mean_metrics(metrics)

    def validate(self, loader: Iterable[BatchT]) -> dict[str, float]:
        """Run one validation epoch without gradient tracking."""
        self.model.eval()
        metrics: list[dict[str, float]] = []
        progress = self._batches(loader, description=self._progress_description("val"), leave=False)
        with torch.no_grad():
            for batch in progress:
                batch = batch.to(self.device)
                output = self.model(batch)
                result = self.task.compute(output, batch)
                loss = self._total_loss(result.loss, output)
                metrics.append(self._as_metrics(loss, result.metrics))
                progress.set_postfix(self._progress_metrics(metrics, include_learning_rate=False))
        return self._mean_metrics(metrics)

    def fit(
        self,
        train_loader: Iterable[BatchT],
        *,
        val_loader: Iterable[BatchT] | None = None,
        epochs: int = 1,
    ) -> FitResult:
        """Train for ``epochs`` additional epochs and return their history."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.scheduler_step_timing == "validation" and val_loader is None:
            raise ValueError("a validation scheduler requires val_loader")

        history: list[EpochResult] = []
        self._display_final_epoch = self.epoch + epochs + 1
        for _ in range(epochs):
            self.epoch += 1
            train = self.train_epoch(train_loader)
            if self.scheduler_step_timing == "epoch":
                self._step_scheduler()
            validation = self.validate(val_loader) if val_loader is not None else None
            if self.scheduler_step_timing == "validation":
                self._step_scheduler(self._scheduler_validation_metric(validation))
            epoch_result = EpochResult(
                self.epoch,
                self.global_step,
                train,
                validation,
                self.learning_rates,
            )
            history.append(epoch_result)
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch_result)
        return FitResult(history, self.epoch, self.global_step)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save enough state to continue training from the next epoch."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "epoch": self.epoch,
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> CheckpointState:
        """Restore model, optimizer, and counters from ``path``."""
        checkpoint: dict[str, Any] = torch.load(Path(path), map_location=self.device, weights_only=False)
        required = {"model", "optimizer", "epoch", "global_step"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler_state = checkpoint.get("scheduler")
        if self.scheduler is not None:
            if scheduler_state is None:
                raise ValueError("checkpoint has no scheduler state for the configured scheduler")
            self.scheduler.load_state_dict(scheduler_state)
        elif scheduler_state is not None:
            raise ValueError("checkpoint contains scheduler state but no scheduler is configured")
        self.epoch = int(checkpoint["epoch"])
        self.global_step = int(checkpoint["global_step"])
        return CheckpointState(self.epoch, self.global_step)

    @property
    def learning_rates(self) -> tuple[float, ...]:
        """Current learning rate for each optimizer parameter group."""
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)

    def _step_scheduler(self, metric: float | None = None) -> None:
        if self.scheduler is None:
            raise RuntimeError("cannot step an unconfigured scheduler")
        if metric is None:
            self.scheduler.step()
        else:
            self.scheduler.step(metric)

    def _scheduler_validation_metric(self, validation: dict[str, float] | None) -> float:
        if validation is None or self.scheduler_metric is None:
            raise RuntimeError("validation metrics are required for validation scheduler stepping")
        prefix = "validation/"
        if not self.scheduler_metric.startswith(prefix):
            raise ValueError("scheduler_metric must use the 'validation/<metric>' form")
        metric_name = self.scheduler_metric.removeprefix(prefix)
        if metric_name not in validation:
            raise KeyError(f"validation metric {self.scheduler_metric!r} was not recorded")
        return validation[metric_name]

    def _progress_description(self, phase: str) -> str:
        """Build a human-readable phase label without changing training semantics."""
        current_epoch = self.epoch + 1
        if self._display_final_epoch is None:
            return f"Epoch {current_epoch} {phase}"
        return f"Epoch {current_epoch}/{self._display_final_epoch} {phase}"

    def _progress_metrics(self, batches: list[dict[str, float]], *, include_learning_rate: bool) -> dict[str, str]:
        """Return generic running means for tqdm without task-specific metric names."""
        means = self._mean_metrics(batches)
        display = {name: f"{value:.4g}" for name, value in sorted(means.items())}
        if include_learning_rate:
            rates = self.learning_rates
            display["lr"] = f"{rates[0]:.3g}" if len(rates) == 1 else ",".join(f"{rate:.3g}" for rate in rates)
        return display

    def _batches(self, loader: Iterable[BatchT], *, description: str, leave: bool) -> tqdm:
        """Create a per-epoch progress bar with the requested terminal retention."""
        return tqdm(loader, desc=description, leave=leave, disable=not self.progress)

    def _total_loss(self, primary_loss: Tensor, output: ModelOutput) -> Tensor:
        loss = primary_loss
        for name, weight in self.aux_loss_weights.items():
            if name not in output.aux_losses:
                raise KeyError(f"auxiliary loss {name!r} was configured but not returned by the model")
            loss = loss + weight * output.aux_losses[name]
        return loss

    @staticmethod
    def _as_metrics(loss: Tensor, metrics: Mapping[str, Tensor | float]) -> dict[str, float]:
        values = {"loss": float(loss.detach().item())}
        for name, value in metrics.items():
            values[name] = float(value.detach().item()) if isinstance(value, Tensor) else float(value)
        return values

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            raise ValueError("data loader produced no batches")
        names = set().union(*(item.keys() for item in metrics))
        return {name: sum(item[name] for item in metrics if name in item) / sum(name in item for item in metrics) for name in names}
