from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from goldfish.experiments import (
    CheckpointManager,
    ExperimentRun,
    build_data_provenance,
    collect_environment,
    sanitize_run_name,
)


def test_run_creation_writes_canonical_layout_and_never_overwrites(tmp_path: Path) -> None:
    run = ExperimentRun.create(
        tmp_path / "runs",
        name="Alphabet GRU!",
        config={"training": {"epochs": 2}},
        data={"dataset": {"name": "alphabet"}},
    )

    assert run.run_id == "exp1-alphabet-gru"
    assert run.path == tmp_path / "runs" / "exp1-alphabet-gru"
    assert (run.path / "checkpoints").is_dir()
    assert (run.path / "artifacts" / "samples").is_dir()
    assert (run.path / "metrics.jsonl").read_text() == ""
    assert json.loads((run.path / "summary.json").read_text())["status"] == "created"
    assert "run_id: exp1-alphabet-gru" in (run.path / "config.yaml").read_text()

    (tmp_path / "runs" / "exp2-taken").mkdir()
    next_run = ExperimentRun.create(tmp_path / "runs", name="next", config={}, data={})
    assert next_run.run_id == "exp3-next"


def test_run_lifecycle_metrics_and_failure_summary(tmp_path: Path) -> None:
    run = ExperimentRun.create(tmp_path, name="trial", config={}, data={})
    run.start()
    run.append_metrics({"epoch": 1, "global_step": 2, "train": {"loss": 1.0}})
    run.complete(last_epoch=0, global_step=2, final={"checkpoint": "checkpoints/final.pt"})

    record = json.loads((run.path / "metrics.jsonl").read_text())
    summary = json.loads((run.path / "summary.json").read_text())
    assert record["epoch"] == 1
    assert summary["last_epoch"] == 1
    assert summary["status"] == "completed"
    assert summary["finished_at"] is not None

    failed = ExperimentRun.create(tmp_path, name="failed", config={}, data={})
    failed.fail(RuntimeError("boom"))
    failed_summary = json.loads((failed.path / "summary.json").read_text())
    assert failed_summary["status"] == "failed"
    assert failed_summary["error"] == {"type": "RuntimeError", "message": "boom"}


def test_provenance_builder_uses_validated_mapping_snapshots() -> None:
    provenance = build_data_provenance(
        manifest={
            "name": "alphabet", "version": "1.0", "modality": "text", "builder": "text_files_lm",
        },
        dataset_lock={
            "fingerprint": "dataset-fp",
            "splits": {"train": {"fingerprint": "train-fp"}, "val": {"fingerprint": "val-fp"}},
        },
        tokenizer_lock={
            "fingerprint": "tokenizer-fp",
            "tokenizer": {
                "path": "tokenizer.json", "sha256": "artifact-fp", "name": "character", "vocab_size": 28,
                "special_token_ids": {"pad": 0, "eos": 1},
            },
        },
        runtime_metadata={"sequence_length": 13, "sample_counts": {"train": 6, "val": 2}},
        dataset_root="data/alphabet",
    )

    assert provenance["locking"]["dataset_fingerprint"] == "dataset-fp"
    assert provenance["locking"]["split_fingerprints"] == {"train": "train-fp", "val": "val-fp"}
    assert provenance["tokenizer"]["pad_token_id"] == 0
    assert provenance["runtime"]["train_samples"] == 6


def test_environment_collection_is_robust_outside_a_git_repository(tmp_path: Path) -> None:
    environment = collect_environment(device="cpu", repository=tmp_path)
    assert environment["python"]
    assert environment["device"] == "cpu"
    assert environment["git_commit"] is None
    assert environment["git_dirty"] is None


def test_checkpoint_manager_saves_latest_best_final_and_periodic(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path, monitor="validation/loss", mode="min", save_frequency=2)

    manager.on_epoch_end(model, optimizer, epoch=0, global_step=1, metrics={"validation": {"loss": 2.0}})
    manager.on_epoch_end(model, optimizer, epoch=1, global_step=2, metrics={"validation": {"loss": 1.0}})
    manager.save_final(model, optimizer, epoch=1, global_step=2, metrics={"validation": {"loss": 1.0}})

    for filename in ("latest.pt", "best.pt", "final.pt", "epoch-0002.pt"):
        payload = torch.load(tmp_path / "checkpoints" / filename, weights_only=False)
        assert payload["format"] == "goldfish-checkpoint-v1"
        assert payload["model"]
        assert payload["optimizer"]
    assert manager.best_value == 1.0
    assert manager.best_epoch == 1
    assert manager.best_summary()["epoch"] == 2


def test_checkpoint_manager_rejects_missing_configured_monitor(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path, monitor="validation/f1", mode="max")

    with pytest.raises(ValueError, match="validation/f1"):
        manager.on_epoch_end(model, optimizer, epoch=0, global_step=1, metrics={"validation": {"loss": 1.0}})


def test_sanitize_run_name() -> None:
    assert sanitize_run_name("  Café__Model / v2  ") == "cafe-model-v2"
    assert sanitize_run_name("!!!") == "run"
