from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[2]
FORECAST_PROFILE = ROOT / "model-profiles" / "forecast" / "gru-small.yaml"


def load_entry(name: str):
    spec = importlib.util.spec_from_file_location(f"numeric_{name}", ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def write_numeric_bundle(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
    (root / "manifest.yaml").write_text(
        """\
name: numeric-integration
version: "1.0"
modality: numeric
builder: numeric_files_forecast
task: point_forecast
format:
  file_type: csv
  timestamp_column: timestamp
  entity_column: entity
  sort_order: ascending
schema:
  features: [value]
  targets: [value]
  dtypes: {timestamp: datetime, entity: string, value: double}
window: {lookback: 2, horizons: [1]}
normalization:
  name: standard
  fit_split: train
  artifact: preprocessing/normalizer.json
  lock: preprocessing/normalizer-lock.json
splits:
  train: {files: [train/01.csv]}
  val: {files: [val/01.csv]}
  test: {files: [test/01.csv]}
locking: {dataset_lock: dataset-lock.json}
""",
        encoding="utf-8",
    )
    for split, start, count in (("train", 0, 5), ("val", 5, 3), ("test", 8, 3)):
        rows = "timestamp,entity,value\n" + "".join(
            f"2024-01-01T00:{index:02d}:00Z,a,{index}.0\n" for index in range(start, start + count)
        )
        (root / split / "01.csv").write_text(rows, encoding="utf-8")


def test_deterministic_training_requires_a_seed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    write_numeric_bundle(dataset)
    assert load_entry("prepare")([str(dataset)]) == 0

    with pytest.raises(ValueError, match="deterministic requires --seed"):
        load_entry("train")([str(dataset), "--deterministic"])


def test_prepare_and_train_numeric_forecast_run(tmp_path: Path, capsys) -> None:
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    write_numeric_bundle(dataset)

    assert load_entry("prepare")([str(dataset)]) == 0
    assert load_entry("train")([
        str(dataset), "--runs-dir", str(runs), "--name", "numeric", "--epochs", "1", "--batch-size", "2",
        "--device", "cpu", "--num-workers", "0", "--model-profile", str(FORECAST_PROFILE), "--max-new-tokens", "0", "--deterministic", "--seed", "7",
    ]) == 0

    run = runs / "exp1-numeric"
    checkpoint = torch.load(run / "checkpoints" / "final.pt", weights_only=False)
    data = json.loads((run / "data.json").read_text(encoding="utf-8"))
    assert (run / "metrics.jsonl").is_file()
    assert (run / "artifacts" / "plots" / "training-curves.png").is_file()
    assert checkpoint["provenance"]["normalizer_fingerprint"] == data["normalizer"]["fingerprint"]
    assert checkpoint["provenance"]["tokenizer_fingerprint"] is None
    assert data["runtime"]["features"] == ["value"]
    config = (run / "config.yaml").read_text(encoding="utf-8")
    assert "train_workers: 0" in config
    assert "pin_memory: false" in config
    assert "deterministic: true" in config
    assert "seed: 7" in config
    brief = capsys.readouterr().out
    assert "Device:" in brief
    assert "Loader:" in brief
    assert "Reproduce:  deterministic=True, seed=7" in brief


def test_numeric_training_strictly_resumes_from_latest_checkpoint(tmp_path: Path) -> None:
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    write_numeric_bundle(dataset)
    assert load_entry("prepare")([str(dataset)]) == 0
    assert load_entry("train")([
        str(dataset), "--runs-dir", str(runs), "--name", "resume", "--epochs", "1", "--batch-size", "2", "--num-workers", "0", "--model-profile", str(FORECAST_PROFILE),
    ]) == 0

    run = runs / "exp1-resume"
    assert load_entry("train")([str(dataset), "--resume", str(run), "--epochs", "1"]) == 0

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "checkpoints" / "final.pt", weights_only=False)
    records = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["status"] == "completed"
    assert summary["last_epoch"] == 2
    assert summary["global_step"] == 4
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 4
    assert len(records) == 2

    with pytest.raises(ValueError, match="model-profile cannot be used when resuming"):
        load_entry("train")([str(dataset), "--resume", str(run), "--epochs", "1", "--model-profile", str(ROOT / "model-profiles" / "forecast" / "lstm-small.yaml")])


def test_numeric_loose_resume_allows_learning_rate_override_and_resets_optimizer_state(tmp_path: Path) -> None:
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    write_numeric_bundle(dataset)
    assert load_entry("prepare")([str(dataset)]) == 0
    assert load_entry("train")([
        str(dataset), "--runs-dir", str(runs), "--name", "loose", "--epochs", "1", "--batch-size", "2",
        "--num-workers", "0", "--model-profile", str(FORECAST_PROFILE),
    ]) == 0

    run = runs / "exp1-loose"
    before = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert before["optimizer"]["state"]
    with pytest.raises(ValueError, match="resume optimization configuration"):
        load_entry("train")([str(dataset), "--resume", str(run), "--epochs", "1", "--learning-rate", "0.0001"])

    assert load_entry("train")([
        str(dataset), "--resume", str(run), "--resume-loose", "--epochs", "1", "--learning-rate", "0.0001",
        "--batch-size", "1", "--num-workers", "0",
    ]) == 0

    checkpoint = torch.load(run / "checkpoints" / "final.pt", weights_only=False)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 5
    assert records[-1]["learning_rate"] == pytest.approx(0.0001)
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0001)
    assert checkpoint["optimizer"]["state"]


def test_forecast_exports_raw_unit_predictions_and_text_infer_rejects_numeric_run(tmp_path: Path) -> None:
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    write_numeric_bundle(dataset)
    assert load_entry("prepare")([str(dataset)]) == 0
    assert load_entry("train")([str(dataset), "--runs-dir", str(runs), "--name", "numeric", "--epochs", "1", "--batch-size", "2", "--num-workers", "0", "--model-profile", str(FORECAST_PROFILE)]) == 0

    run = runs / "exp1-numeric"
    output = tmp_path / "predictions.jsonl"
    assert load_entry("forecast")([str(run), "--checkpoint", "final", "--split", "test", "--output", str(output)]) == 0

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["entity_id"] == "a"
    assert [record["cutoff_timestamp"] for record in records] == ["2024-01-01T00:08:00Z", "2024-01-01T00:09:00Z"]
    assert records[0]["horizons"] == [1]
    assert records[0]["targets"] == ["value"]
    assert isinstance(records[0]["prediction"][0][0], float)
    assert summary["raw_units"] is True
    assert summary["window_count"] == 2

    plot = tmp_path / "forecast.png"
    assert load_entry("forecast")([
        str(run), "--checkpoint", "final", "--split", "test", "--output", str(output), "--plot", str(plot),
    ]) == 0
    assert plot.is_file()
    assert plot.stat().st_size > 0

    with pytest.raises(ValueError, match="goldfish forecast"):
        load_entry("infer")([str(run), "--prompt", "not applicable"])

    (dataset / "test" / "01.csv").write_text("timestamp,entity,value\n2024-01-01T00:08:00Z,a,800.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset lock"):
        load_entry("forecast")([str(run), "--checkpoint", "final", "--split", "test", "--output", str(output)])
