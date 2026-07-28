"""Preparation-time validation and train-only normalization for numeric CSV datasets."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from goldfish.data.validation import write_dataset_lock, write_normalizer_lock


class NumericDataValidationError(ValueError):
    """Raised when numeric CSV content violates the dataset contract."""


@dataclass(frozen=True)
class PreparedNumericDataset:
    """Preparation result used to create a normalizer lock."""

    normalizer_path: Path
    train_row_count: int


def prepare_numeric_forecast_dataset(root: Path | str, manifest: Mapping[str, Any]) -> PreparedNumericDataset:
    """Validate every declared shard and write a frozen train-only standard normalizer."""
    root = Path(root)
    rows_by_split = _read_and_validate_rows(root, manifest)
    dataset_lock = write_dataset_lock(root, dict(manifest))
    normalizer = _fit_normalizer(rows_by_split["train"], manifest, dataset_lock)
    normalization = _mapping(manifest, "normalization")
    artifact_path = root / _string(normalization, "artifact")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(normalizer, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    write_normalizer_lock(root, dict(manifest))
    return PreparedNumericDataset(normalizer_path=artifact_path, train_row_count=len(rows_by_split["train"]))


def _read_and_validate_rows(root: Path, manifest: Mapping[str, Any]) -> dict[str, list[dict[str, object]]]:
    """Strict prepare-time reader that validates all content and temporal invariants."""
    return _read_rows(root, manifest, validate_content=True)


def read_prepared_rows(root: Path, manifest: Mapping[str, Any]) -> dict[str, list[dict[str, object]]]:
    """Load lock-verified CSV values for runtime windowing without quality revalidation."""
    return _read_rows(root, manifest, validate_content=False)


def _read_rows(root: Path, manifest: Mapping[str, Any], *, validate_content: bool) -> dict[str, list[dict[str, object]]]:
    format_config = _mapping(manifest, "format")
    schema = _mapping(manifest, "schema")
    timestamp_column = _string(format_config, "timestamp_column")
    entity_column = _string(format_config, "entity_column")
    dtypes = _mapping(schema, "dtypes")
    features = _string_list(schema, "features")
    required_columns = [timestamp_column, entity_column, *features]
    delimiter = format_config.get("delimiter", ",")
    assert isinstance(delimiter, str)

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "val", "test"):
        split_rows: list[dict[str, object]] = []
        split_config = _mapping(manifest, "splits")[split]
        if not isinstance(split_config, Mapping):
            raise NumericDataValidationError(f"Numeric manifest split {split!r} must be a mapping.")
        files = split_config.get("files")
        # The validator has already established this type and non-emptiness.
        assert isinstance(files, list)
        for relative_path in files:
            if not isinstance(relative_path, str):
                raise NumericDataValidationError(f"{split} shard path must be a string.")
            split_rows.extend(_read_shard(root / relative_path, relative_path, delimiter, required_columns, dtypes, timestamp_column, entity_column, validate_content=validate_content))
        if validate_content:
            _validate_split_order(split_rows, split, entity_column, timestamp_column)
        rows_by_split[split] = split_rows
    if validate_content:
        _validate_cross_split_order(rows_by_split, entity_column, timestamp_column)
    return rows_by_split


def _read_shard(
    path: Path,
    relative_path: str,
    delimiter: str,
    required_columns: list[str],
    dtypes: Mapping[str, Any],
    timestamp_column: str,
    entity_column: str,
    *,
    validate_content: bool,
) -> list[dict[str, object]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise NumericDataValidationError(f"Could not read CSV shard {relative_path!r}: {error}") from error
    with handle:
        try:
            reader = csv.DictReader(handle, delimiter=delimiter)
            header = reader.fieldnames
            if validate_content and (not header or any(not name for name in header) or len(set(header)) != len(header)):
                raise NumericDataValidationError(f"CSV shard {relative_path!r} has an empty or duplicate header.")
            if header is None:
                raise NumericDataValidationError(f"CSV shard {relative_path!r} has no header.")
            if validate_content:
                missing = set(required_columns).difference(header)
                if missing:
                    raise NumericDataValidationError(f"CSV shard {relative_path!r} is missing declared columns: {sorted(missing)}.")
            result: list[dict[str, object]] = []
            for row_number, row in enumerate(reader, start=2):
                if validate_content and None in row:
                    raise NumericDataValidationError(f"CSV shard {relative_path!r} has a ragged row at line {row_number}.")
                result.append(_parse_row(row, relative_path, row_number, dtypes, timestamp_column, entity_column))
            return result
        except csv.Error as error:
            raise NumericDataValidationError(f"Could not parse CSV shard {relative_path!r}: {error}") from error


def _parse_row(
    row: Mapping[str | None, str | None], path: str, line: int, dtypes: Mapping[str, Any], timestamp_column: str, entity_column: str
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for column, dtype in dtypes.items():
        value = row.get(column)
        if value is None or value == "":
            raise NumericDataValidationError(f"CSV shard {path!r} line {line} has an empty {column!r} value.")
        if column == timestamp_column:
            parsed[column] = _timestamp(value, path, line)
        elif column == entity_column:
            if not value.strip():
                raise NumericDataValidationError(f"CSV shard {path!r} line {line} has an empty entity ID.")
            parsed[column] = value
        elif dtype == "double":
            try:
                number = float(value)
            except ValueError as error:
                raise NumericDataValidationError(f"CSV shard {path!r} line {line} has an invalid numeric value for {column!r}.") from error
            if not math.isfinite(number):
                raise NumericDataValidationError(f"CSV shard {path!r} line {line} has a non-finite value for {column!r}.")
            parsed[column] = number
        elif dtype == "int":
            try:
                number = int(value)
            except ValueError as error:
                raise NumericDataValidationError(f"CSV shard {path!r} line {line} has an invalid integer value for {column!r}.") from error
            if str(number) != value and str(number) != value.lstrip("+"):
                raise NumericDataValidationError(f"CSV shard {path!r} line {line} has a non-integral integer value for {column!r}.")
            if not -(2**63) <= number < 2**63:
                raise NumericDataValidationError(f"CSV shard {path!r} line {line} has an integer outside the signed 64-bit range.")
            parsed[column] = number
    return parsed


def _timestamp(value: str, path: str, line: int) -> datetime:
    try:
        timestamp_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(timestamp_value)
    except ValueError as error:
        raise NumericDataValidationError(f"CSV shard {path!r} line {line} has an invalid RFC 3339 timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NumericDataValidationError(f"CSV shard {path!r} line {line} timestamp must be RFC 3339 with an offset or Z.")
    return parsed.astimezone(UTC)


def _validate_split_order(rows: list[dict[str, object]], split: str, entity_column: str, timestamp_column: str) -> None:
    seen: set[tuple[object, object]] = set()
    previous: dict[object, datetime] = {}
    for row in rows:
        entity, timestamp = row[entity_column], row[timestamp_column]
        assert isinstance(timestamp, datetime)
        key = (entity, timestamp)
        if key in seen:
            raise NumericDataValidationError(f"Duplicate (entity, timestamp) key in {split} split.")
        seen.add(key)
        if entity in previous and timestamp <= previous[entity]:
            raise NumericDataValidationError(f"Timestamps must be strictly increasing within each entity in {split} split.")
        previous[entity] = timestamp


def _validate_cross_split_order(rows_by_split: Mapping[str, list[dict[str, object]]], entity_column: str, timestamp_column: str) -> None:
    latest: dict[object, datetime] = {}
    for split in ("train", "val", "test"):
        earliest: dict[object, datetime] = {}
        for row in rows_by_split[split]:
            entity, timestamp = row[entity_column], row[timestamp_column]
            assert isinstance(timestamp, datetime)
            earliest.setdefault(entity, timestamp)
        for entity, timestamp in earliest.items():
            if entity in latest and timestamp <= latest[entity]:
                raise NumericDataValidationError("Timestamps in an earlier split must strictly precede every later split timestamp for each entity.")
        for row in rows_by_split[split]:
            entity, timestamp = row[entity_column], row[timestamp_column]
            assert isinstance(timestamp, datetime)
            latest[entity] = timestamp


def _fit_normalizer(rows: list[dict[str, object]], manifest: Mapping[str, Any], dataset_lock: Mapping[str, Any]) -> dict[str, object]:
    features = _string_list(_mapping(manifest, "schema"), "features")
    counts = [0] * len(features)
    means = [0.0] * len(features)
    m2 = [0.0] * len(features)
    for row in rows:
        for index, feature in enumerate(features):
            raw_value = row[feature]
            if not isinstance(raw_value, (int, float)):
                raise NumericDataValidationError(f"Numeric feature {feature!r} has an invalid parsed value.")
            value = float(raw_value)
            counts[index] += 1
            delta = value - means[index]
            means[index] += delta / counts[index]
            m2[index] += delta * (value - means[index])
    scales = [math.sqrt(value / count) if value else 1.0 for value, count in zip(m2, counts, strict=True)]
    train = _mapping(_mapping(dataset_lock, "splits"), "train")
    normalization = _mapping(manifest, "normalization")
    return {
        "format": "goldfish-normalizer-v1",
        "name": normalization["name"],
        "features": features,
        "means": means,
        "scales": scales,
        "train_row_count": len(rows),
        "source": {"train_fingerprint": train["fingerprint"]},
        "config": {"name": normalization["name"], "fit_split": normalization["fit_split"]},
    }


def _mapping(value: Mapping[str, Any] | Mapping[str, object], field: str) -> Mapping[str, Any]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise NumericDataValidationError(f"Numeric manifest {field!r} must be a mapping.")
    return result


def _string(value: Mapping[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str):
        raise NumericDataValidationError(f"Numeric manifest {field!r} must be a string.")
    return result


def _string_list(value: Mapping[str, Any], field: str) -> list[str]:
    result = value.get(field)
    if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
        raise NumericDataValidationError(f"Numeric manifest {field!r} must be a list of strings.")
    return result
