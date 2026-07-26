"""Dataset-lock v1 construction and verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import (
    Manifest,
    ManifestValidationError,
    manifest_file_pair_paths,
    manifest_file_path,
    validator_registry,
)

LOCK_FORMAT_V1 = "goldfish-dataset-lock-v1"
TOKENIZER_LOCK_FORMAT_V1 = "goldfish-tokenizer-lock-v1"
_ALGORITHM = "sha256"


class DatasetLockValidationError(ValueError):
    """Raised when a dataset lock does not match its manifest or raw files."""


class TokenizerLockValidationError(ValueError):
    """Raised when a tokenizer lock does not match its artifact or manifest."""



def canonical_json(value: object) -> bytes:
    """Encode a value using the stable JSON representation used for fingerprints."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lock_path(root: Path, manifest: Mapping[str, Any]) -> Path:
    locking = manifest.get("locking")
    if isinstance(locking, Mapping) and isinstance(locking.get("dataset_lock"), str):
        return root / locking["dataset_lock"]
    return root / "dataset-lock.json"


def _file_snapshot(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    try:
        contents = path.read_bytes()
    except FileNotFoundError as error:
        raise DatasetLockValidationError(f"Manifest-listed file is missing: {relative_path}") from error
    if not path.is_file():
        raise DatasetLockValidationError(f"Manifest-listed path is not a file: {relative_path}")
    return {"path": relative_path, "sha256": sha256_bytes(contents), "bytes": len(contents)}


def _split_fingerprint(split_name: str, files: list[dict[str, object]]) -> str:
    return sha256_bytes(canonical_json({"split": split_name, "files": files}))


def _document_unit(manifest: Mapping[str, Any]) -> str:
    format_declaration = manifest.get("format")
    if not isinstance(format_declaration, Mapping) or not isinstance(format_declaration.get("document_unit"), str):
        raise DatasetLockValidationError("Cannot build lock: manifest format.document_unit must be declared.")
    return format_declaration["document_unit"]


def _lock_manifest_semantics(manifest: Mapping[str, Any]) -> dict[str, object]:
    return {
        field: manifest[field]
        for field in ("name", "version", "modality", "builder", "task", "format", "tokenizer")
        if field in manifest
    }


def build_dataset_lock(root: Path | str, manifest: Manifest) -> dict[str, object]:
    """Build an unsigned dataset-lock v1 snapshot from declared manifest files."""
    root = Path(root)
    splits_value = manifest.get("splits")
    if not isinstance(splits_value, Mapping):
        raise DatasetLockValidationError("Cannot build lock: manifest 'splits' must be a mapping.")

    document_unit = _document_unit(manifest)
    locked_splits: dict[str, object] = {}
    for split_name, split_value in splits_value.items():
        if not isinstance(split_name, str) or not isinstance(split_value, Mapping):
            raise DatasetLockValidationError("Cannot build lock: manifest splits must be named mappings.")
        files_value = split_value.get("files")
        if not isinstance(files_value, list):
            raise DatasetLockValidationError(f"Cannot build lock: splits.{split_name}.files must be a list.")
        files: list[dict[str, object]] = []
        for index, entry in enumerate(files_value):
            field = f"splits.{split_name}.files[{index}]"
            if document_unit == "file-pair":
                input_path, output_path = manifest_file_pair_paths(entry, field)
                files.append(
                    {
                        "input": _file_snapshot(root, input_path),
                        "output": _file_snapshot(root, output_path),
                    }
                )
            else:
                files.append(_file_snapshot(root, manifest_file_path(entry, field)))
        locked_splits[split_name] = {"fingerprint": _split_fingerprint(split_name, files), "files": files}

    lock: dict[str, object] = {
        "format": LOCK_FORMAT_V1,
        "algorithm": _ALGORITHM,
        "dataset": {"name": manifest.get("name"), "version": manifest.get("version")},
        "manifest": _lock_manifest_semantics(manifest),
        "splits": locked_splits,
    }
    lock["fingerprint"] = sha256_bytes(canonical_json(lock))
    return lock


def write_dataset_lock(root: Path | str, manifest: Manifest | None = None) -> dict[str, object]:
    """Build and write the dataset lock declared by a validated manifest."""
    root = Path(root)
    manifest = manifest if manifest is not None else validator_registry.validate_manifest(root)
    lock = build_dataset_lock(root, manifest)
    path = _lock_path(root, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return lock


def validate_dataset_lock(root: Path | str, manifest: Manifest) -> dict[str, object]:
    """Verify the on-disk v1 lock and raw files against a validated manifest."""
    root = Path(root)
    path = _lock_path(root, manifest)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetLockValidationError(f"Dataset lock not found: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetLockValidationError(f"Invalid JSON in dataset lock {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise DatasetLockValidationError("Dataset lock must be a JSON object.")
    if loaded.get("format") != LOCK_FORMAT_V1:
        raise DatasetLockValidationError(f"Unsupported dataset lock format: {loaded.get('format')!r}.")
    if loaded.get("algorithm") != _ALGORITHM:
        raise DatasetLockValidationError("Dataset lock algorithm must be 'sha256'.")

    expected = build_dataset_lock(root, manifest)
    if loaded.get("dataset") != expected["dataset"]:
        raise DatasetLockValidationError("Dataset lock dataset identity does not match manifest.")
    if loaded.get("manifest") != expected["manifest"]:
        raise DatasetLockValidationError("Dataset lock manifest semantics do not match manifest.yaml.")
    if loaded.get("splits") != expected["splits"]:
        _raise_split_mismatch(loaded, expected)
    if loaded.get("fingerprint") != expected["fingerprint"]:
        raise DatasetLockValidationError("Dataset lock overall fingerprint mismatch.")
    return loaded


def _tokenizer_paths(root: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any]]:
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise TokenizerLockValidationError("Text manifest is missing tokenizer declarations.")
    try:
        artifact_path = manifest_file_path(tokenizer["artifact"], "tokenizer.artifact")
        lock_path = manifest_file_path(tokenizer["lock"], "tokenizer.lock")
    except (KeyError, ManifestValidationError) as error:
        raise TokenizerLockValidationError("Tokenizer artifact and lock paths must be declared.") from error
    return root / artifact_path, root / lock_path, tokenizer


def _read_tokenizer_artifact(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        contents = path.read_bytes()
    except FileNotFoundError as error:
        raise TokenizerLockValidationError(f"Tokenizer artifact not found: {path}") from error
    try:
        artifact = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerLockValidationError(f"Invalid JSON in tokenizer artifact {path}: {error}") from error
    if not isinstance(artifact, dict):
        raise TokenizerLockValidationError("Tokenizer artifact must be a JSON object.")
    return artifact, contents


def _tokenizer_metadata(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, object], Path, Mapping[str, Any]]:
    artifact_path, lock_path, config = _tokenizer_paths(root, manifest)
    artifact, contents = _read_tokenizer_artifact(artifact_path)
    special_token_ids = {"pad": artifact.get("pad_token_id"), "eos": artifact.get("eos_token_id")}
    if not all(isinstance(token_id, int) for token_id in special_token_ids.values()):
        raise TokenizerLockValidationError("Tokenizer artifact must declare integer pad_token_id and eos_token_id.")
    vocab_size = artifact.get("vocab_size")
    if not isinstance(vocab_size, int):
        token_to_id = artifact.get("token_to_id")
        if not isinstance(token_to_id, Mapping):
            raise TokenizerLockValidationError("Tokenizer artifact must declare vocab_size or token_to_id.")
        vocab_size = len(token_to_id) + 2
    return (
        {
            "path": str(artifact_path.relative_to(lock_path.parent)),
            "sha256": sha256_bytes(contents),
            "name": config.get("name"),
            "vocab_size": vocab_size,
            "special_token_ids": special_token_ids,
        },
        lock_path,
        config,
    )


def build_tokenizer_lock(root: Path | str, manifest: Manifest) -> dict[str, object]:
    """Build a tokenizer-lock v1 bound to the verified dataset training split."""
    root = Path(root)
    dataset_lock = validate_dataset_lock(root, manifest)
    tokenizer, _, config = _tokenizer_metadata(root, manifest)
    splits = dataset_lock.get("splits")
    train = splits.get("train") if isinstance(splits, Mapping) else None
    train_fingerprint = train.get("fingerprint") if isinstance(train, Mapping) else None
    if not isinstance(train_fingerprint, str):
        raise TokenizerLockValidationError("Dataset lock does not contain a train fingerprint.")
    lock: dict[str, object] = {
        "format": TOKENIZER_LOCK_FORMAT_V1,
        "algorithm": _ALGORITHM,
        "tokenizer": tokenizer,
        "source": {
            "dataset_name": manifest.get("name"),
            "dataset_version": manifest.get("version"),
            "train_fingerprint": train_fingerprint,
        },
        "config": {
            "fit_split": config.get("fit_split"),
            "special_tokens": config.get("special_tokens"),
        },
    }
    lock["fingerprint"] = sha256_bytes(canonical_json(lock))
    return lock


def write_tokenizer_lock(root: Path | str, manifest: Manifest | None = None) -> dict[str, object]:
    """Build and write the tokenizer lock declared by a validated manifest."""
    root = Path(root)
    manifest = manifest if manifest is not None else validator_registry.validate_manifest(root)
    lock = build_tokenizer_lock(root, manifest)
    _, lock_path, _ = _tokenizer_metadata(root, manifest)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return lock


def validate_tokenizer_lock(root: Path | str, manifest: Manifest) -> dict[str, object]:
    """Verify a tokenizer lock against its artifact, manifest configuration, and dataset lock."""
    root = Path(root)
    _, lock_path, _ = _tokenizer_metadata(root, manifest)
    try:
        loaded = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TokenizerLockValidationError(f"Tokenizer lock not found: {lock_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerLockValidationError(f"Invalid JSON in tokenizer lock {lock_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise TokenizerLockValidationError("Tokenizer lock must be a JSON object.")
    if loaded.get("format") != TOKENIZER_LOCK_FORMAT_V1:
        raise TokenizerLockValidationError(f"Unsupported tokenizer lock format: {loaded.get('format')!r}.")
    if loaded.get("algorithm") != _ALGORITHM:
        raise TokenizerLockValidationError("Tokenizer lock algorithm must be 'sha256'.")
    expected = build_tokenizer_lock(root, manifest)
    if loaded.get("tokenizer") != expected["tokenizer"]:
        raise TokenizerLockValidationError("Tokenizer artifact sha256 or metadata does not match tokenizer lock.")
    if loaded.get("source") != expected["source"]:
        raise TokenizerLockValidationError("Tokenizer lock source does not match the current dataset train fingerprint.")
    if loaded.get("config") != expected["config"]:
        raise TokenizerLockValidationError("Tokenizer lock configuration does not match manifest.yaml.")
    if loaded.get("fingerprint") != expected["fingerprint"]:
        raise TokenizerLockValidationError("Tokenizer lock fingerprint mismatch.")
    return loaded


def _raise_split_mismatch(loaded: Mapping[str, object], expected: Mapping[str, object]) -> None:
    actual_splits = loaded.get("splits")
    expected_splits = expected["splits"]
    if not isinstance(actual_splits, Mapping) or not isinstance(expected_splits, Mapping):
        raise DatasetLockValidationError("Dataset lock split entries do not match manifest files.")
    for split_name, expected_split in expected_splits.items():
        actual_split = actual_splits.get(split_name)
        if not isinstance(actual_split, Mapping) or not isinstance(expected_split, Mapping):
            raise DatasetLockValidationError(f"Dataset lock is missing split {split_name!r}.")
        actual_files = actual_split.get("files")
        expected_files = expected_split.get("files")
        if actual_files != expected_files:
            if isinstance(actual_files, list) and isinstance(expected_files, list):
                for index, expected_file in enumerate(expected_files):
                    actual_file = actual_files[index] if index < len(actual_files) else None
                    if actual_file != expected_file:
                        relative_path = expected_file.get("path", "<unknown>") if isinstance(expected_file, Mapping) else "<unknown>"
                        if isinstance(actual_file, Mapping) and actual_file.get("path") == relative_path:
                            raise DatasetLockValidationError(f"Dataset lock sha256 mismatch for {relative_path!r}.")
                        raise DatasetLockValidationError(f"Dataset lock file order or membership mismatch in split {split_name!r}.")
            raise DatasetLockValidationError(f"Dataset lock file entries mismatch in split {split_name!r}.")
        if actual_split.get("fingerprint") != expected_split.get("fingerprint"):
            raise DatasetLockValidationError(f"Dataset lock split fingerprint mismatch for {split_name!r}.")
    raise DatasetLockValidationError("Dataset lock split entries do not match manifest files.")
