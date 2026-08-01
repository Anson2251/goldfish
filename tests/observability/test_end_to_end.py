"""End-to-end test: model profile -> resolved config -> ProbeHook -> Trainer -> artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from goldfish.config import load_model_profile, resolve_observability_config
from goldfish.core import ModelOutput, StepResult
from goldfish.models import LatentCommunicationMultiHeadLSTMForecastModel
from goldfish.observability import (
    JsonlRecorder,
    build_manifest,
    build_probe_hook,
)
from goldfish.training import Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "model-profiles" / "forecast" / "multihead-lstm-small-latent-communication.yaml"


@dataclass
class TensorBatch:
    inputs: torch.Tensor

    def to(self, device: torch.device) -> "TensorBatch":
        return TensorBatch(self.inputs.to(device))


class ForecastTask:
    def compute(self, output: ModelOutput, batch: TensorBatch) -> StepResult:
        loss = torch.mean(output.predictions["forecast"] ** 2)
        return StepResult(loss=loss, metrics={})


def _batches(count: int, rows: int) -> list[TensorBatch]:
    return [TensorBatch(torch.randn(rows, 8, 4)) for _ in range(count)]


def test_training_with_profile_probes_writes_probe_artifacts(tmp_path: Path) -> None:
    torch.manual_seed(0)
    profile = load_model_profile(PROFILE)
    config = resolve_observability_config(
        profile["observability"],
        {"enabled": True, "reference": {"split": "val", "batches": 2}},
    )
    model = LatentCommunicationMultiHeadLSTMForecastModel(
        4, 1, 2, hidden_dim=32, num_heads=4, num_layers=2, communication_dim=8, communication_gate_initial_logit=-5.0
    )
    probe_dir = tmp_path / "artifacts" / "probes"
    recorder = JsonlRecorder(probe_dir)
    manifest = build_manifest(config, model, source_paths={probe.name: "profile" for probe in config.probes})
    reference = _batches(2, rows=4)
    hook = build_probe_hook(config, recorder, reference_factory=lambda: reference)
    assert hook is not None

    trainer = Trainer(
        model,
        ForecastTask(),
        torch.optim.SGD(model.parameters(), lr=0.01),
        hooks=[hook],
        device="cpu",
        progress=False,
    )
    trainer.fit(_batches(4, rows=4), val_loader=_batches(2, rows=4), epochs=2)

    assert (probe_dir / "manifest.json").is_file()
    manifest_content = json.loads((probe_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_content["reference"]["batches"] == 2
    assert [entry["name"] for entry in manifest_content["probes"]] == [
        "communication-state",
        "activation-stats",
    ]
    communication_entry = manifest_content["probes"][0]
    assert communication_entry["matched_modules"] == ["latent_communications.0"]
    activation_points = manifest_content["probes"][1]["points"]
    assert activation_points[0]["pattern"] == "latent_communications.*"
    assert activation_points[0]["matched_modules"] == ["latent_communications.0"]

    communication = [json.loads(line) for line in (probe_dir / "communication-state.jsonl").read_text().splitlines()]
    assert [(record["phase"], record["epoch"]) for record in communication] == [
        ("fit_start", 0),
        ("epoch_end", 1),
        ("epoch_end", 2),
        ("fit_end", 2),
    ]
    assert communication[1]["payload"]["communications"][0]["type"] == "latent_communication"
    assert len(communication[1]["payload"]["communications"][0]["routing_entropy_per_receiver"]) == 4

    activation = [json.loads(line) for line in (probe_dir / "activation-stats.jsonl").read_text().splitlines()]
    assert [(record["phase"], record["epoch"]) for record in activation] == [
        ("fit_start", 0),
        ("epoch_end", 1),
        ("epoch_end", 2),
        ("fit_end", 2),
    ]
    payload = activation[1]["payload"]
    quantities = {entry["quantity"]: entry for entry in payload["points"]}
    assert "message-magnitude" in quantities
    assert len(quantities["message-magnitude"]["injection_ratio_per_receiver"]) == 4
    assert "io-stats" in quantities
    assert quantities["io-stats"]["input_norm"] > 0.0
