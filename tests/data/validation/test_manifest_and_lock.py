from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from goldfish.data.validation import (
    DatasetLockValidationError,
    Manifest,
    ManifestValidationError,
    TokenizerLockValidationError,
    build_dataset_lock,
    build_tokenizer_lock,
    validate_dataset_lock,
    validate_tokenizer_lock,
    validator_registry,
    write_dataset_lock,
    write_tokenizer_lock,
)


def write_manifest(root: Path, text: str) -> Manifest:
    (root / "manifest.yaml").write_text(text, encoding="utf-8")
    return validator_registry.validate_manifest(root)


VALID_TEXT_MANIFEST = """\
name: example
version: "1.0"
modality: text
builder: text_files_lm
task: causal_language_model
splits:
  train:
    files: [train/a.txt]
  val:
    files: [val/a.txt]
format:
  encoding: utf-8
  document_unit: file
  append_eos: true
tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens:
    pad: <pad>
    eos: <eos>
locking:
  dataset_lock: dataset-lock.json
"""


def prepare_dataset(root: Path) -> Manifest:
    (root / "train").mkdir()
    (root / "val").mkdir()
    (root / "tokenizer").mkdir()
    (root / "train" / "a.txt").write_bytes(b"train")
    (root / "val" / "a.txt").write_bytes(b"val")
    (root / "tokenizer" / "tokenizer.json").write_text(
        json.dumps({"type": "character", "pad_token_id": 0, "eos_token_id": 1, "token_to_id": {"a": 2}}),
        encoding="utf-8",
    )
    return write_manifest(root, VALID_TEXT_MANIFEST)


def test_validates_v1_manifest_and_returns_parsed_data(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path)

    assert manifest["version"] == "1.0"
    assert manifest["splits"]["train"]["files"] == ["train/a.txt"]


@pytest.mark.parametrize("path", ["/absolute.txt", "../escape.txt", "train/../../escape.txt", r"C:\\escape.txt"])
def test_rejects_unsafe_manifest_file_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ManifestValidationError, match="relative.*traversal"):
        write_manifest(tmp_path, VALID_TEXT_MANIFEST.replace("train/a.txt", path))


@pytest.mark.parametrize("field", ["artifact", "lock"])
def test_requires_text_tokenizer_artifact_and_lock_paths(tmp_path: Path, field: str) -> None:
    with pytest.raises(ManifestValidationError, match=field):
        write_manifest(tmp_path, VALID_TEXT_MANIFEST.replace(f"  {field}: tokenizer/tokenizer{'-lock' if field == 'lock' else ''}.json\n", ""))


def test_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="Unsupported manifest version"):
        write_manifest(tmp_path, VALID_TEXT_MANIFEST.replace('version: "1.0"', 'version: "2.0"'))


def test_builds_writes_and_verifies_canonical_v1_dataset_lock(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path)

    assert build_dataset_lock(tmp_path, manifest) == write_dataset_lock(tmp_path)
    lock = cast(dict[str, Any], write_dataset_lock(tmp_path, manifest))

    assert lock["format"] == "goldfish-dataset-lock-v1"
    assert lock["splits"]["train"]["files"] == [
        {"path": "train/a.txt", "sha256": hashlib.sha256(b"train").hexdigest(), "bytes": 5}
    ]
    assert validate_dataset_lock(tmp_path, manifest) == lock


def test_dataset_lock_verification_detects_modified_content(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path)
    write_dataset_lock(tmp_path, manifest)
    (tmp_path / "train" / "a.txt").write_text("after", encoding="utf-8")

    with pytest.raises(DatasetLockValidationError, match="sha256 mismatch.*train/a.txt"):
        validate_dataset_lock(tmp_path, manifest)


def test_builds_writes_and_verifies_tokenizer_lock(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path)
    dataset_lock = write_dataset_lock(tmp_path, manifest)

    assert build_tokenizer_lock(tmp_path, manifest) == write_tokenizer_lock(tmp_path, manifest)
    lock = cast(dict[str, Any], write_tokenizer_lock(tmp_path, manifest))

    assert lock["format"] == "goldfish-tokenizer-lock-v1"
    assert lock["tokenizer"]["special_token_ids"] == {"pad": 0, "eos": 1}
    assert lock["source"]["train_fingerprint"] == cast(dict[str, Any], dataset_lock["splits"])["train"]["fingerprint"]
    assert validate_tokenizer_lock(tmp_path, manifest) == lock


def test_tokenizer_lock_verification_detects_artifact_and_train_lock_changes(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path)
    write_dataset_lock(tmp_path, manifest)
    write_tokenizer_lock(tmp_path, manifest)
    (tmp_path / "tokenizer" / "tokenizer.json").write_text(
        json.dumps({"type": "character", "pad_token_id": 0, "eos_token_id": 1, "token_to_id": {"a": 2, "b": 3}}),
        encoding="utf-8",
    )

    with pytest.raises(TokenizerLockValidationError, match="artifact.*sha256"):
        validate_tokenizer_lock(tmp_path, manifest)


def test_tokenizer_lock_binds_the_current_dataset_train_fingerprint(tmp_path: Path) -> None:
    manifest = prepare_dataset(tmp_path)
    write_dataset_lock(tmp_path, manifest)
    write_tokenizer_lock(tmp_path, manifest)
    (tmp_path / "train" / "a.txt").write_text("changed training data", encoding="utf-8")
    write_dataset_lock(tmp_path, manifest)

    with pytest.raises(TokenizerLockValidationError, match="train fingerprint"):
        validate_tokenizer_lock(tmp_path, manifest)
