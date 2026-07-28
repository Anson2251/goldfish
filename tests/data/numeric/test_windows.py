from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from goldfish.core import Batch
from goldfish.data.numeric import ForecastBatch, NumericFilesForecastDataModule, StandardNormalizer


MANIFEST = {
    "name": "windows",
    "modality": "numeric",
    "format": {"timestamp_column": "timestamp", "entity_column": "entity", "delimiter": ","},
    "schema": {
        "features": ["open", "close"],
        "targets": ["close"],
        "dtypes": {"timestamp": "datetime", "entity": "string", "open": "double", "close": "double"},
    },
    "window": {"lookback": 2, "horizons": [1, 2]},
    "normalization": {"name": "standard", "artifact": "preprocessing/normalizer.json", "lock": "preprocessing/normalizer-lock.json"},
    "splits": {"train": {"files": ["train.csv"]}, "val": {"files": ["val.csv"]}, "test": {"files": ["test.csv"]}},
}


def write_bundle(root: Path) -> dict[str, object]:
    root.joinpath("train.csv").write_text(
        "timestamp,entity,open,close\n"
        "2024-01-01T00:00:00Z,a,0,0\n"
        "2024-01-01T00:01:00Z,a,1,10\n"
        "2024-01-01T00:00:00Z,b,5,50\n"
        "2024-01-01T00:01:00Z,b,6,60\n"
        "2024-01-01T00:02:00Z,a,2,20\n"
        "2024-01-01T00:03:00Z,a,3,30\n",
        encoding="utf-8",
    )
    root.joinpath("val.csv").write_text(
        "timestamp,entity,open,close\n2024-01-01T00:04:00Z,a,4,40\n2024-01-01T00:05:00Z,a,5,50\n",
        encoding="utf-8",
    )
    root.joinpath("test.csv").write_text(
        "timestamp,entity,open,close\n2024-01-01T00:06:00Z,a,6,60\n2024-01-01T00:07:00Z,a,7,70\n",
        encoding="utf-8",
    )
    artifact = {
        "format": "goldfish-normalizer-v1", "name": "standard", "features": ["open", "close"],
        "means": [1.0, 10.0], "scales": [2.0, 10.0], "train_row_count": 6,
        "source": {"train_fingerprint": "fixture"}, "config": {"name": "standard", "fit_split": "train"},
    }
    (root / "preprocessing").mkdir()
    (root / "preprocessing" / "normalizer.json").write_text(json.dumps(artifact), encoding="utf-8")
    return copy.deepcopy(MANIFEST)


def test_standard_normalizer_transforms_and_inverse_transforms_targets(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    normalization = cast(dict[str, Any], manifest["normalization"])
    normalizer = StandardNormalizer.load(tmp_path / str(normalization["artifact"]))

    assert normalizer.transform_features([[3.0, 30.0]]).tolist() == [[1.0, 2.0]]
    restored = normalizer.inverse_transform_targets(torch.tensor([[[2.0]]]), ["close"])
    assert restored.tolist() == [[[30.0]]]


def test_windows_respect_entities_split_ownership_and_horizon_order(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    module = NumericFilesForecastDataModule(tmp_path, manifest, batch_size=2, verify_locks=False)

    assert len(module.train_dataset) == 1
    assert len(module.val_dataset) == 0
    assert len(module.test_dataset) == 0
    train = module.train_dataset[0]
    assert train.inputs.tolist() == [[-0.5, -1.0], [0.0, 0.0]]
    assert train.targets.tolist() == [[1.0], [2.0]]
    assert module.runtime_metadata["sample_counts"] == {"train": 1, "val": 0, "test": 0}
    assert module.runtime_metadata["zero_window_entities"] == {"train": ["b"], "val": ["a"], "test": ["a"]}


def test_validation_windows_use_train_history_but_own_cutoff_and_targets(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    manifest["window"] = {"lookback": 2, "horizons": [1]}
    (tmp_path / "val.csv").write_text(
        "timestamp,entity,open,close\n"
        "2024-01-01T00:04:00Z,a,4,40\n"
        "2024-01-01T00:05:00Z,a,5,50\n",
        encoding="utf-8",
    )
    module = NumericFilesForecastDataModule(tmp_path, manifest, batch_size=1, verify_locks=False)

    assert len(module.val_dataset) == 1
    window = module.val_dataset[0]
    assert window.inputs.tolist() == [[1.0, 2.0], [1.5, 3.0]]
    assert window.targets.tolist() == [[4.0]]


def test_forecast_batch_collates_and_moves_tensors(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    module = NumericFilesForecastDataModule(tmp_path, manifest, batch_size=1, verify_locks=False)
    batch = next(iter(module.train_dataloader()))

    assert isinstance(batch, Batch)
    assert isinstance(batch, ForecastBatch)
    assert batch.inputs.shape == (1, 2, 2)
    assert batch.targets.shape == (1, 2, 1)
    assert batch.to(torch.device("cpu")).targets.device.type == "cpu"


def test_data_module_requires_frozen_normalizer_artifact(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    (tmp_path / "preprocessing" / "normalizer.json").unlink()

    with pytest.raises(ValueError, match="Normalizer artifact"):
        NumericFilesForecastDataModule(tmp_path, manifest, batch_size=1, verify_locks=False)
