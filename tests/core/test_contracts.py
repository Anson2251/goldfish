from dataclasses import dataclass

import torch

from goldfish.core import Batch, ModelOutput, StepResult, Task


@dataclass
class ToyBatch:
    values: torch.Tensor

    def to(self, device: torch.device) -> "ToyBatch":
        return ToyBatch(values=self.values.to(device))


class ToyTask:
    def compute(self, output: ModelOutput, batch: ToyBatch) -> StepResult:
        return StepResult(loss=output.predictions["value"].mean(), metrics={"score": 1.0})


def test_core_contracts_support_structured_model_and_step_outputs() -> None:
    output = ModelOutput(predictions={"value": torch.tensor([2.0])})
    result = ToyTask().compute(output, ToyBatch(torch.tensor([1.0])))

    assert output.representations is None
    assert output.aux_losses == {}
    assert output.diagnostics == {}
    assert result.loss.item() == 2.0
    assert result.metrics == {"score": 1.0}
    assert isinstance(ToyBatch(torch.tensor([1.0])), Batch)
    assert isinstance(ToyTask(), Task)


def test_batch_to_returns_a_batch_on_the_requested_device() -> None:
    batch = ToyBatch(torch.tensor([1.0]))

    moved = batch.to(torch.device("cpu"))

    assert isinstance(moved, ToyBatch)
    assert moved.values.device.type == "cpu"
