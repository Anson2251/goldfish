from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import pytest

from goldfish.data.numeric import NumericDataValidationError, prepare_numeric_forecast_dataset
from goldfish.data.validation import (
    validate_dataset_lock,
    validate_normalizer_lock,
    validator_registry,
)


MANIFEST = """\
name: bars
version: "1.0"
modality: numeric
builder: numeric_files_forecast
task: point_forecast
format:
  file_type: csv
  delimiter: ","
  timestamp_column: timestamp
  entity_column: entity
  sort_order: ascending
schema:
  features: [open, close, volume]
  targets: [close]
  dtypes:
    timestamp: datetime
    entity: string
    open: double
    close: double
    volume: int
window:
  lookback: 2
  horizons: [1]
normalization:
  name: standard
  fit_split: train
  artifact: preprocessing/normalizer.json
  lock: preprocessing/normalizer-lock.json
splits:
  train:
    files: [train/02.csv, train/01.csv]
  val:
    files: [val/01.csv]
  test:
    files: [test/01.csv]
locking:
  dataset_lock: dataset-lock.json
"""


def write_bundle(root: Path) -> dict[str, object]:
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
    (root / "manifest.yaml").write_text(MANIFEST, encoding="utf-8")
    (root / "train" / "02.csv").write_text(
        "timestamp,entity,open,close,volume,ignored\n"
        "2024-01-01T00:00:00Z,a,1,10,2,x\n"
        "2024-01-01T00:00:00Z,b,5,50,6,x\n",
        encoding="utf-8",
    )
    (root / "train" / "01.csv").write_text(
        "timestamp,entity,open,close,volume\n"
        "2024-01-01T00:01:00+00:00,a,3,30,4\n"
        "2024-01-01T00:01:00Z,b,7,70,8\n",
        encoding="utf-8",
    )
    (root / "val" / "01.csv").write_text(
        "timestamp,entity,open,close,volume\n2024-01-01T00:02:00Z,a,9,90,10\n", encoding="utf-8"
    )
    (root / "test" / "01.csv").write_text(
        "timestamp,entity,open,close,volume\n2024-01-01T00:03:00Z,a,11,110,12\n", encoding="utf-8"
    )
    return validator_registry.validate_manifest(root)


def test_prepares_ordered_csv_shards_and_train_only_normalizer(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    prepared = prepare_numeric_forecast_dataset(tmp_path, manifest)
    dataset_lock = validate_dataset_lock(tmp_path, manifest)

    artifact = json.loads((tmp_path / "preprocessing" / "normalizer.json").read_text(encoding="utf-8"))
    assert prepared.train_row_count == 4
    assert artifact["features"] == ["open", "close", "volume"]
    assert artifact["means"] == [4.0, 40.0, 5.0]
    assert artifact["scales"] == pytest.approx([2.23606797749979, 22.360679774997898, 2.23606797749979])
    artifact_source = cast(dict[str, Any], artifact["source"])
    dataset_splits = cast(dict[str, Any], dataset_lock["splits"])
    train_split = cast(dict[str, Any], dataset_splits["train"])
    assert artifact_source["train_fingerprint"] == train_split["fingerprint"]
    assert validate_dataset_lock(tmp_path, manifest) == dataset_lock
    normalizer_lock = validate_normalizer_lock(tmp_path, manifest)
    locked_normalizer = cast(dict[str, Any], normalizer_lock["normalizer"])
    assert locked_normalizer["features"] == ["open", "close", "volume"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda root: (root / "train" / "01.csv").write_text("timestamp,entity,open,close,volume\n2023-12-31T23:59:00Z,a,3,30,4\n", encoding="utf-8"), "strictly increasing"),
        (lambda root: (root / "train" / "01.csv").write_text("timestamp,entity,open,close,volume\n2024-01-01T00:01:00,a,3,30,4\n", encoding="utf-8"), "RFC 3339"),
        (lambda root: (root / "train" / "01.csv").write_text("timestamp,entity,open,close,volume\n2024-01-01T00:01:00Z,a,NaN,30,4\n", encoding="utf-8"), "finite"),
        (lambda root: (root / "train" / "01.csv").write_text("timestamp,entity,open,close,volume\n2024-01-01T00:01:00Z,a,3,30,4.5\n", encoding="utf-8"), "integer"),
    ],
)
def test_rejects_invalid_csv_content(tmp_path: Path, mutate, match: str) -> None:
    manifest = write_bundle(tmp_path)
    mutate(tmp_path)

    with pytest.raises(NumericDataValidationError, match=match):
        prepare_numeric_forecast_dataset(tmp_path, manifest)


def test_prepare_entrypoint_writes_numeric_locks_without_a_tokenizer(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    repository_root = Path(__file__).parents[3]
    spec = importlib.util.spec_from_file_location("goldfish_prepare", repository_root / "prepare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main([str(tmp_path)]) == 0
    assert (tmp_path / "dataset-lock.json").is_file()
    assert (tmp_path / "preprocessing" / "normalizer.json").is_file()
    assert (tmp_path / "preprocessing" / "normalizer-lock.json").is_file()
    assert not (tmp_path / "tokenizer").exists()
    assert validator_registry.validate_manifest(tmp_path) == manifest


def test_rejects_cross_split_time_leakage(tmp_path: Path) -> None:
    manifest = write_bundle(tmp_path)
    (tmp_path / "val" / "01.csv").write_text(
        "timestamp,entity,open,close,volume\n2023-12-31T00:00:00Z,a,9,90,10\n", encoding="utf-8"
    )

    with pytest.raises(NumericDataValidationError, match="earlier split"):
        prepare_numeric_forecast_dataset(tmp_path, manifest)
