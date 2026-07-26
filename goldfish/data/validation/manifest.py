"""Versioned dataset-manifest validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Protocol

import yaml

Manifest = dict[str, Any]


class ManifestValidationError(ValueError):
    """Raised when a dataset manifest is missing or violates its versioned schema."""


class ManifestValidator(Protocol):
    """A validator for one manifest schema version."""

    def validate(self, root: Path, manifest: Manifest) -> Manifest:
        """Validate and return the parsed manifest."""
        ...


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"manifest {field!r} must be a mapping.")
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"manifest {field!r} must be a non-empty string.")
    return value


def manifest_file_path(entry: object, field: str) -> str:
    """Return and validate a simple or metadata-rich manifest file entry path."""
    path = entry if isinstance(entry, str) else _require_mapping(entry, field).get("path")
    path = _require_nonempty_string(path, f"{field}.path")
    pure_path = PurePath(path)
    windows_path = PureWindowsPath(path)
    if pure_path.is_absolute() or windows_path.is_absolute() or ".." in pure_path.parts or ".." in windows_path.parts:
        raise ManifestValidationError(
            f"manifest file path {path!r} must be relative and must not contain traversal ('..')."
        )
    return path


def manifest_file_pair_paths(entry: object, field: str) -> tuple[str, str]:
    """Return validated input/output paths from a file-pair manifest entry."""
    pair = _require_mapping(entry, field)
    return (
        manifest_file_path(pair.get("input"), f"{field}.input"),
        manifest_file_path(pair.get("output"), f"{field}.output"),
    )


class ManifestValidatorV1:
    """Validator for manifest version ``1.0``."""

    version = "1.0"
    _required_fields = ("name", "version", "modality", "builder", "task")

    def validate(self, root: Path, manifest: Manifest) -> Manifest:
        del root  # Paths are validated syntactically; existence is checked by lock verification.
        for field in self._required_fields:
            _require_nonempty_string(manifest.get(field), field)

        format_declaration = _require_mapping(manifest.get("format"), "format")
        encoding = _require_nonempty_string(format_declaration.get("encoding"), "format.encoding")
        if encoding != "utf-8":
            raise ManifestValidationError("manifest format.encoding must be 'utf-8'.")
        document_unit = _require_nonempty_string(format_declaration.get("document_unit"), "format.document_unit")
        if document_unit not in {"file", "file-pair"}:
            raise ManifestValidationError("manifest format.document_unit must be 'file' or 'file-pair'.")

        splits = _require_mapping(manifest.get("splits"), "splits")
        split_names = ("train", "val", "test") if document_unit == "file-pair" else ("train", "val")
        for split_name in split_names:
            split = _require_mapping(splits.get(split_name), f"splits.{split_name}")
            files = split.get("files")
            if not isinstance(files, list) or not files:
                raise ManifestValidationError(f"manifest splits.{split_name}.files must be a non-empty list.")
            for index, entry in enumerate(files):
                field = f"splits.{split_name}.files[{index}]"
                if document_unit == "file-pair":
                    manifest_file_pair_paths(entry, field)
                else:
                    manifest_file_path(entry, field)

        if manifest["modality"] == "text":
            self._validate_text_declarations(manifest, document_unit)
        return manifest

    @staticmethod
    def _validate_text_declarations(manifest: Manifest, document_unit: str) -> None:
        tokenizer = _require_mapping(manifest.get("tokenizer"), "tokenizer")
        _require_nonempty_string(tokenizer.get("name"), "tokenizer.name")
        for field in ("artifact", "lock"):
            path = _require_nonempty_string(tokenizer.get(field), f"tokenizer.{field}")
            manifest_file_path(path, f"tokenizer.{field}")
        if tokenizer.get("fit_split") != "train": 
            raise ManifestValidationError("manifest tokenizer.fit_split must be 'train' for text datasets.")
        special_tokens = _require_mapping(tokenizer.get("special_tokens"), "tokenizer.special_tokens")
        token_names = ("pad", "eos", "sep") if document_unit == "file-pair" else ("pad", "eos")
        for token_name in token_names:
            _require_nonempty_string(special_tokens.get(token_name), f"tokenizer.special_tokens.{token_name}")

        locking = _require_mapping(manifest.get("locking"), "locking")
        lock_path = _require_nonempty_string(locking.get("dataset_lock"), "locking.dataset_lock")
        manifest_file_path(lock_path, "locking.dataset_lock")


class ValidatorRegistry:
    """Routes parsed manifests to their registered version validators."""

    def __init__(self) -> None:
        self._validators: dict[str, ManifestValidator] = {}

    def register(self, version: str, validator: ManifestValidator) -> None:
        self._validators[version] = validator

    def validate_manifest(self, root: Path | str) -> Manifest:
        root = Path(root)
        manifest_path = root / "manifest.yaml"
        try:
            loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ManifestValidationError(f"Dataset manifest not found: {manifest_path}") from error
        except UnicodeDecodeError as error:
            raise ManifestValidationError(f"Dataset manifest is not UTF-8: {manifest_path}") from error
        except yaml.YAMLError as error:
            raise ManifestValidationError(f"Invalid YAML in dataset manifest {manifest_path}: {error}") from error

        manifest = dict(_require_mapping(loaded, "root"))
        version = manifest.get("version")
        if not isinstance(version, str):
            raise ManifestValidationError("manifest 'version' must be a string.")
        try:
            validator = self._validators[version]
        except KeyError as error:
            raise ManifestValidationError(f"Unsupported manifest version: {version!r}.") from error
        return validator.validate(root, manifest)


validator_registry = ValidatorRegistry()
validator_registry.register(ManifestValidatorV1.version, ManifestValidatorV1())
