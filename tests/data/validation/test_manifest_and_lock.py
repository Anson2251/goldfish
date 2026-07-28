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
    build_normalizer_lock,
    build_tokenizer_lock,
    validate_dataset_lock,
    validate_normalizer_lock,
    validate_tokenizer_lock,
    validator_registry,
    write_dataset_lock,
    write_normalizer_lock,
    write_tokenizer_lock,
)


def write_manifest(root: Path, text: str) -> Manifest:
    (root / "manifest.yaml").write_text(text, encoding="utf-8")
    return validator_registry.validate_manifest(root)


VALID_FILE_PAIR_TEXT_MANIFEST = """\
name: paired-example
version: "1.0"
modality: text
builder: text_files_lm
task: causal_language_model
splits:
  train:
    files:
      - input: train/source-a.txt
        output: train/target-a.txt
      - input: train/source-b.txt
        output: train/target-b.txt
  val:
    files:
      - input: val/source-a.txt
        output: val/target-a.txt
  test:
    files:
      - input: test/source-a.txt
        output: test/target-a.txt
format:
  encoding: utf-8
  document_unit: file-pair
tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens:
    pad: <pad>
    eos: <eos>
    sep: <sep>
locking:
  dataset_lock: dataset-lock.json
"""


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


def prepare_file_pair_dataset(root: Path) -> Manifest:
    for split in ("train", "val", "test", "tokenizer"):
        (root / split).mkdir()
    for path, contents in {
        "train/source-a.txt": b"source a",
        "train/target-a.txt": b"target a",
        "train/source-b.txt": b"source b",
        "train/target-b.txt": b"target b",
        "val/source-a.txt": b"validation source",
        "val/target-a.txt": b"validation target",
        "test/source-a.txt": b"test source",
        "test/target-a.txt": b"test target",
    }.items():
        (root / path).write_bytes(contents)
    (root / "tokenizer" / "tokenizer.json").write_text(
        json.dumps({"type": "character", "pad_token_id": 0, "eos_token_id": 1, "token_to_id": {"a": 2}}),
        encoding="utf-8",
    )
    return write_manifest(root, VALID_FILE_PAIR_TEXT_MANIFEST)


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


@pytest.mark.parametrize(
    ("format_declaration", "match"),
    [
        ("format:\n  encoding: latin-1\n  document_unit: file", "format.encoding"),
        ("format:\n  encoding: utf-8\n  document_unit: record", "format.document_unit"),
        ("format:\n  encoding: utf-8\n", "format.document_unit"),
    ],
)
def test_requires_supported_format_declarations(tmp_path: Path, format_declaration: str, match: str) -> None:
    manifest = VALID_TEXT_MANIFEST.replace("format:\n  encoding: utf-8\n  document_unit: file", format_declaration)

    with pytest.raises(ManifestValidationError, match=match):
        write_manifest(tmp_path, manifest)


@pytest.mark.parametrize(
    "entry",
    [
        "      - input: train/source-a.txt\n",
        "      - input: ../source-a.txt\n        output: train/target-a.txt",
        "      - input: train/source-a.txt\n        output: /target-a.txt",
        "      - train/source-a.txt",
    ],
)
def test_file_pair_requires_safe_input_and_output_entries(tmp_path: Path, entry: str) -> None:
    manifest = VALID_FILE_PAIR_TEXT_MANIFEST.replace(
        "      - input: train/source-a.txt\n        output: train/target-a.txt", entry, 1
    )

    with pytest.raises(ManifestValidationError, match="input|output|mapping|relative"):
        write_manifest(tmp_path, manifest)


def test_file_pair_requires_a_nonempty_separator_token(tmp_path: Path) -> None:
    manifest = VALID_FILE_PAIR_TEXT_MANIFEST.replace("    sep: <sep>\n", "")

    with pytest.raises(ManifestValidationError, match="special_tokens.sep"):
        write_manifest(tmp_path, manifest)


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


def test_builds_and_verifies_ordered_file_pair_dataset_lock(tmp_path: Path) -> None:
    manifest = prepare_file_pair_dataset(tmp_path)
    lock = write_dataset_lock(tmp_path, manifest)

    pairs = cast(dict[str, Any], cast(dict[str, Any], lock["splits"])["train"])["files"]
    assert pairs == [
        {
            "input": {"path": "train/source-a.txt", "sha256": hashlib.sha256(b"source a").hexdigest(), "bytes": 8},
            "output": {"path": "train/target-a.txt", "sha256": hashlib.sha256(b"target a").hexdigest(), "bytes": 8},
        },
        {
            "input": {"path": "train/source-b.txt", "sha256": hashlib.sha256(b"source b").hexdigest(), "bytes": 8},
            "output": {"path": "train/target-b.txt", "sha256": hashlib.sha256(b"target b").hexdigest(), "bytes": 8},
        },
    ]
    assert validate_dataset_lock(tmp_path, manifest) == lock


def test_file_pair_lock_detects_pair_identity_order_and_membership_changes(tmp_path: Path) -> None:
    manifest = prepare_file_pair_dataset(tmp_path)
    write_dataset_lock(tmp_path, manifest)
    lock_path = tmp_path / "dataset-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["splits"]["train"]["files"][0]["input"], lock["splits"]["train"]["files"][0]["output"] = (
        lock["splits"]["train"]["files"][0]["output"],
        lock["splits"]["train"]["files"][0]["input"],
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(DatasetLockValidationError, match="file order or membership|file entries"):
        validate_dataset_lock(tmp_path, manifest)


def test_file_pair_tokenizer_lock_binds_train_fingerprint_and_config(tmp_path: Path) -> None:
    manifest = prepare_file_pair_dataset(tmp_path)
    write_dataset_lock(tmp_path, manifest)
    lock = write_tokenizer_lock(tmp_path, manifest)
    assert cast(dict[str, Any], lock["config"])["special_tokens"] == {"pad": "<pad>", "eos": "<eos>", "sep": "<sep>"}

    (tmp_path / "train" / "source-a.txt").write_text("changed source", encoding="utf-8")
    write_dataset_lock(tmp_path, manifest)
    with pytest.raises(TokenizerLockValidationError, match="train fingerprint"):
        validate_tokenizer_lock(tmp_path, manifest)

    write_tokenizer_lock(tmp_path, manifest)
    manifest["tokenizer"]["special_tokens"]["sep"] = "<different-sep>"
    write_dataset_lock(tmp_path, manifest)
    with pytest.raises(TokenizerLockValidationError, match="configuration"):
        validate_tokenizer_lock(tmp_path, manifest)


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


VALID_NUMERIC_MANIFEST = """\
name: example-bars
version: "1.0"
modality: numeric
builder: numeric_files_forecast
task: point_forecast
format:
  file_type: csv
  timestamp_column: timestamp
  entity_column: entity_id
  sort_order: ascending
schema:
  features: [open, close, volume]
  targets: [close]
  dtypes:
    timestamp: datetime
    entity_id: string
    open: double
    close: double
    volume: int
window:
  lookback: 3
  horizons: [1, 2]
normalization:
  name: standard
  fit_split: train
  artifact: preprocessing/normalizer.json
  lock: preprocessing/normalizer-lock.json
splits:
  train:
    files: [train/01-bars.csv]
  val:
    files: [val/01-bars.csv]
  test:
    files: [test/01-bars.csv]
locking:
  dataset_lock: dataset-lock.json
"""


def prepare_numeric_dataset(root: Path) -> Manifest:
    for split in ("train", "val", "test", "preprocessing"):
        (root / split).mkdir()
    for split in ("train", "val", "test"):
        (root / split / "01-bars.csv").write_text(
            "timestamp,entity_id,open,close,volume\\n2024-01-01T00:00:00Z,a,1.5,2.5,3\\n",
            encoding="utf-8",
        )
    return write_manifest(root, VALID_NUMERIC_MANIFEST)


def write_normalizer_artifact(root: Path, manifest: Manifest) -> None:
    (root / manifest["normalization"]["artifact"]).write_text(
        json.dumps(
            {
                "format": "goldfish-normalizer-v1",
                "name": "standard",
                "features": ["open", "close", "volume"],
                "means": [1.5, 2.5, 3.0],
                "scales": [1.0, 1.0, 1.0],
                "train_row_count": 1,
                "source": {"train_fingerprint": "placeholder"},
                "config": {"name": "standard", "fit_split": "train"},
            }
        ),
        encoding="utf-8",
    )


def test_validates_numeric_forecast_manifest_and_numeric_lock_semantics(tmp_path: Path) -> None:
    manifest = prepare_numeric_dataset(tmp_path)
    lock = write_dataset_lock(tmp_path, manifest)

    lock_manifest = cast(dict[str, Any], lock["manifest"])
    assert lock_manifest["schema"] == manifest["schema"]
    assert lock_manifest["window"] == {"lookback": 3, "horizons": [1, 2]}
    assert lock_manifest["normalization"] == manifest["normalization"]
    assert validate_dataset_lock(tmp_path, manifest) == lock


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("  targets: [missing]", "target"),
        ("  horizons: [1, 1]", "unique"),
        ("  lookback: 0", "positive"),
        ("  sort_order: descending", "sort_order"),
        ("    volume: string", "double.*int"),
    ],
)
def test_rejects_invalid_numeric_forecast_declarations(tmp_path: Path, replacement: str, match: str) -> None:
    source, _, _ = replacement.partition(":")
    manifest = VALID_NUMERIC_MANIFEST.replace(next(line for line in VALID_NUMERIC_MANIFEST.splitlines() if line.startswith(source)), replacement)

    with pytest.raises(ManifestValidationError, match=match):
        write_manifest(tmp_path, manifest)


def test_normalizer_lock_binds_artifact_config_and_train_fingerprint(tmp_path: Path) -> None:

    manifest = prepare_numeric_dataset(tmp_path)
    dataset_lock = write_dataset_lock(tmp_path, manifest)
    write_normalizer_artifact(tmp_path, manifest)

    assert build_normalizer_lock(tmp_path, manifest) == write_normalizer_lock(tmp_path, manifest)
    lock = validate_normalizer_lock(tmp_path, manifest)
    assert lock["format"] == "goldfish-normalizer-lock-v1"
    normalizer = cast(dict[str, Any], lock["normalizer"])
    source = cast(dict[str, Any], lock["source"])
    dataset_splits = cast(dict[str, Any], dataset_lock["splits"])
    assert normalizer["features"] == ["open", "close", "volume"]
    assert source["train_fingerprint"] == dataset_splits["train"]["fingerprint"]

    artifact_path = tmp_path / "preprocessing" / "normalizer.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["means"] = [9.0, 2.5, 3.0]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact.*sha256"):
        validate_normalizer_lock(tmp_path, manifest)
