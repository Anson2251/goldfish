"""Frozen-normalizer numeric forecast datasets and data module."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from goldfish.data.validation import validate_dataset_lock, validate_normalizer_lock

from .batch import ForecastBatch, collate_forecast_batches
from .prepare import NumericDataValidationError, read_prepared_rows


@dataclass(frozen=True)
class StandardNormalizer:
    """Ordered standard feature normalizer loaded from a frozen preparation artifact."""

    features: tuple[str, ...]
    means: Tensor
    scales: Tensor

    @classmethod
    def load(cls, path: Path | str) -> StandardNormalizer:
        path = Path(path)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Normalizer artifact not found: {path}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSON in normalizer artifact {path}: {error}") from error
        if not isinstance(artifact, Mapping) or artifact.get("format") != "goldfish-normalizer-v1" or artifact.get("name") != "standard":
            raise ValueError("Unsupported normalizer artifact.")
        features, means, scales = artifact.get("features"), artifact.get("means"), artifact.get("scales")
        if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
            raise ValueError("Normalizer artifact must contain ordered feature names.")
        if not isinstance(means, list) or not isinstance(scales, list) or len(means) != len(features) or len(scales) != len(features):
            raise ValueError("Normalizer artifact means and scales must match features.")
        try:
            means_tensor = torch.tensor(means, dtype=torch.float64)
            scales_tensor = torch.tensor(scales, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Normalizer artifact means and scales must be numeric.") from error
        if not torch.isfinite(means_tensor).all() or not torch.isfinite(scales_tensor).all() or (scales_tensor <= 0).any():
            raise ValueError("Normalizer artifact means must be finite and scales positive.")
        return cls(tuple(features), means_tensor, scales_tensor)

    def transform_features(self, values: Sequence[Sequence[float]] | Tensor) -> Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.shape[-1] != len(self.features):
            raise ValueError("Feature values do not match normalizer feature count.")
        return (tensor - self.means.to(dtype=tensor.dtype)) / self.scales.to(dtype=tensor.dtype)

    def inverse_transform_features(self, values: Tensor) -> Tensor:
        if values.shape[-1] != len(self.features):
            raise ValueError("Feature values do not match normalizer feature count.")
        return values * self.scales.to(device=values.device, dtype=values.dtype) + self.means.to(device=values.device, dtype=values.dtype)

    def transform_targets(self, values: Tensor, target_features: Sequence[str]) -> Tensor:
        means, scales = self._target_parameters(values, target_features)
        return (values - means) / scales

    def inverse_transform_targets(self, values: Tensor, target_features: Sequence[str]) -> Tensor:
        means, scales = self._target_parameters(values, target_features)
        return values * scales + means

    def _target_parameters(self, values: Tensor, target_features: Sequence[str]) -> tuple[Tensor, Tensor]:
        indices = self._feature_indices(target_features)
        means = self.means[list(indices)].to(device=values.device, dtype=values.dtype)
        scales = self.scales[list(indices)].to(device=values.device, dtype=values.dtype)
        return means, scales

    def _feature_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        positions = {name: index for index, name in enumerate(self.features)}
        try:
            return tuple(positions[name] for name in names)
        except KeyError as error:
            raise ValueError(f"Target {error.args[0]!r} is not a normalized feature.") from error


class NumericForecastDataset(Dataset[ForecastBatch]):
    """Materialized same-entity forecast windows for one requested output split."""

    def __init__(
        self,
        rows_by_split: Mapping[str, list[dict[str, object]]],
        manifest: Mapping[str, Any],
        normalizer: StandardNormalizer,
        split: str,
    ) -> None:
        self._rows, self.window_entities = self._make_windows(rows_by_split, manifest, normalizer, split)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> ForecastBatch:
        return self._rows[index]

    @staticmethod
    def _make_windows(
        rows_by_split: Mapping[str, list[dict[str, object]]], manifest: Mapping[str, Any], normalizer: StandardNormalizer, split: str
    ) -> tuple[list[ForecastBatch], set[str]]:
        format_config = _mapping(manifest, "format")
        schema = _mapping(manifest, "schema")
        window = _mapping(manifest, "window")
        entity_column = _string(format_config, "entity_column")
        timestamp_column = _string(format_config, "timestamp_column")
        features, targets = _strings(schema, "features"), _strings(schema, "targets")
        lookback, horizons = window["lookback"], window["horizons"]
        assert isinstance(lookback, int) and isinstance(horizons, list) and all(isinstance(item, int) for item in horizons)
        split_order = ("train", "val", "test")

        by_entity: dict[object, list[tuple[str, dict[str, object]]]] = defaultdict(list)
        for name in split_order[: split_order.index(split) + 1]:
            for row in rows_by_split[name]:
                by_entity[row[entity_column]].append((name, row))

        windows: list[ForecastBatch] = []
        window_entities: set[str] = set()
        for entity, entity_rows in by_entity.items():
            for cutoff_index in range(lookback - 1, len(entity_rows)):
                target_indices = [cutoff_index + horizon for horizon in horizons]
                if target_indices[-1] >= len(entity_rows):
                    continue
                cutoff_split = entity_rows[cutoff_index][0]
                target_split_names = [entity_rows[index][0] for index in target_indices]
                if cutoff_split != split or any(name != split for name in target_split_names):
                    continue
                history = [row for _, row in entity_rows[cutoff_index - lookback + 1 : cutoff_index + 1]]
                future = [entity_rows[index][1] for index in target_indices]
                inputs = normalizer.transform_features([[_numeric_value(row[feature], feature) for feature in features] for row in history])
                target_values = torch.tensor([[_numeric_value(row[target], target) for target in targets] for row in future], dtype=torch.float32)
                cutoff_timestamp = entity_rows[cutoff_index][1][timestamp_column]
                windows.append(ForecastBatch(
                    inputs=inputs,
                    targets=normalizer.transform_targets(target_values, targets),
                    entity_ids=(str(entity),),
                    cutoff_timestamps=(_timestamp_string(cutoff_timestamp),),
                ))
                window_entities.add(str(entity))
        return windows, window_entities


class NumericFilesForecastDataModule:
    """Load numeric forecasting windows using a verified frozen normalizer artifact."""

    def __init__(self, root: Path | str, manifest: Mapping[str, Any], *, batch_size: int, num_workers: int = 0, verify_locks: bool = True) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.root, self.manifest = Path(root), manifest
        if verify_locks:
            validate_dataset_lock(self.root, dict(manifest))
            validate_normalizer_lock(self.root, dict(manifest))
        normalization = _mapping(manifest, "normalization")
        self.normalizer = StandardNormalizer.load(self.root / _string(normalization, "artifact"))
        if self.normalizer.features != tuple(_strings(_mapping(manifest, "schema"), "features")):
            raise ValueError("Normalizer feature order does not match numeric manifest.")
        try:
            rows_by_split = read_prepared_rows(self.root, manifest)
        except NumericDataValidationError as error:
            raise ValueError(f"Could not load numeric forecast data: {error}") from error
        self.train_dataset = NumericForecastDataset(rows_by_split, manifest, self.normalizer, "train")
        self.val_dataset = NumericForecastDataset(rows_by_split, manifest, self.normalizer, "val")
        self.test_dataset = NumericForecastDataset(rows_by_split, manifest, self.normalizer, "test")
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_workers = num_workers
        self.validation_workers = num_workers
        self.pin_memory = False
        self.prefetch_factor: int | None = None
        self.persistent_workers = False
        self.runtime_metadata = {
            "name": _string(manifest, "name"), "modality": "numeric", "features": list(self.normalizer.features),
            "targets": _strings(_mapping(manifest, "schema"), "targets"), "lookback": _mapping(manifest, "window")["lookback"],
            "horizons": _mapping(manifest, "window")["horizons"],
            "sample_counts": {"train": len(self.train_dataset), "val": len(self.val_dataset), "test": len(self.test_dataset)},
            "zero_window_entities": self._zero_window_entities(rows_by_split),
        }

    def train_dataloader(self) -> DataLoader[ForecastBatch]:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[ForecastBatch]:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[ForecastBatch]:
        return self._loader(self.test_dataset, shuffle=False)

    def configure_loading(
        self, *, train_workers: int, validation_workers: int, pin_memory: bool, prefetch_factor: int | None, persistent_workers: bool
    ) -> None:
        """Apply resolved process and memory-transfer settings before creating loaders."""
        self.train_workers = train_workers
        self.validation_workers = validation_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def _loader(self, dataset: NumericForecastDataset, *, shuffle: bool) -> DataLoader[ForecastBatch]:
        workers = self.train_workers if shuffle else self.validation_workers
        if workers == 0:
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0,
                pin_memory=self.pin_memory, collate_fn=collate_forecast_batches,
            )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=workers,
            pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor, collate_fn=collate_forecast_batches,
        )

    def _zero_window_entities(self, rows_by_split: Mapping[str, list[dict[str, object]]]) -> dict[str, list[str]]:
        entity_column = _string(_mapping(self.manifest, "format"), "entity_column")
        result: dict[str, list[str]] = {}
        for split, dataset in (("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)):
            eligible = {str(row[entity_column]) for row in rows_by_split[split]}
            result[split] = sorted(eligible.difference(dataset.window_entities))
        return result


def _timestamp_string(value: object) -> str:
    if not isinstance(value, datetime):
        raise TypeError("Numeric row has an invalid parsed timestamp.")
    return value.isoformat().replace("+00:00", "Z")


def _numeric_value(value: object, feature: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Numeric row has an invalid value for feature {feature!r}.")
    return float(value)


def _mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise TypeError(f"Numeric manifest {field!r} must be a mapping.")
    return result


def _string(value: Mapping[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str):
        raise TypeError(f"Numeric manifest {field!r} must be a string.")
    return result


def _strings(value: Mapping[str, Any], field: str) -> list[str]:
    result = value.get(field)
    if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
        raise TypeError(f"Numeric manifest {field!r} must be a list of strings.")
    return result
