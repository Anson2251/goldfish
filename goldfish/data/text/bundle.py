"""Manifest-driven text dataset preparation and frozen-tokenizer loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .batch import LanguageModelBatch, collate_language_model_batches
from .corpus import CausalLanguageModelDataset
from .tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class PreparedTextDataset:
    """Preparation output for callers that persist their own dataset lock."""

    tokenizer: CharacterTokenizer
    tokenizer_path: Path
    train_document_count: int

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def prepare_text_dataset(root: Path, manifest: Mapping[str, Any]) -> PreparedTextDataset:
    """Fit a character tokenizer from manifest-listed training documents and save it.

    This intentionally performs no locking or manifest mutation. A validation layer can
    use the returned artifact path and metadata when it writes its own lock.
    """
    root = Path(root)
    train_documents = _read_split_documents(root, manifest, "train", required=True)
    assert train_documents is not None
    tokenizer = CharacterTokenizer()
    tokenizer.fit(train_documents)
    tokenizer_path = root / _tokenizer_path(manifest)
    tokenizer.save(tokenizer_path)
    return PreparedTextDataset(tokenizer, tokenizer_path, len(train_documents))


class TextFilesLanguageModelDataModule:
    """Load text splits using a frozen tokenizer artifact referenced by a manifest.

    Supported manifest fields are ``name``, ``modality``, ``splits.<split>.files``
    (ordered relative UTF-8 path strings or mappings with ``path``), and
    ``tokenizer.artifact``. The manifest is assumed to have been validated by a
    higher-level API; basic structural errors are still reported clearly at this boundary.
    """

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        *,
        sequence_length: int,
        batch_size: int,
        num_workers: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        self.root = Path(root)
        self.manifest = manifest
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = CharacterTokenizer.load(self.root / _tokenizer_path(manifest))

        self.train_dataset = self._build_split("train", required=True)
        self.val_dataset = self._build_split("val")
        self.test_dataset = self._build_split("test")
        self.runtime_metadata = {
            "name": _string_field(manifest, "name"),
            "modality": _string_field(manifest, "modality"),
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "sample_counts": {
                name: len(dataset)
                for name, dataset in (
                    ("train", self.train_dataset),
                    ("val", self.val_dataset),
                    ("test", self.test_dataset),
                )
                if dataset is not None
            },
        }

    def train_dataloader(self) -> DataLoader[LanguageModelBatch]:
        dataset = self.train_dataset
        assert dataset is not None
        return self._loader(dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[LanguageModelBatch] | None:
        dataset = self.val_dataset
        return self._loader(dataset, shuffle=False) if dataset is not None else None

    def test_dataloader(self) -> DataLoader[LanguageModelBatch] | None:
        dataset = self.test_dataset
        return self._loader(dataset, shuffle=False) if dataset is not None else None

    def _build_split(self, split: str, *, required: bool = False) -> CausalLanguageModelDataset | None:
        documents = _read_split_documents(self.root, self.manifest, split, required=required)
        if documents is None:
            return None
        try:
            return CausalLanguageModelDataset(documents, self.tokenizer, self.sequence_length)
        except ValueError as error:
            paths = _split_paths(self.manifest, split, required=True)
            assert paths is not None
            raise ValueError(f"{error} while encoding {split} split ({', '.join(paths)}).") from error

    def _loader(self, dataset: CausalLanguageModelDataset, *, shuffle: bool) -> DataLoader[LanguageModelBatch]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_language_model_batches,
        )


def _read_split_documents(
    root: Path, manifest: Mapping[str, Any], split: str, *, required: bool = False
) -> tuple[str, ...] | None:
    paths = _split_paths(manifest, split, required=required)
    if paths is None:
        return None
    documents: list[str] = []
    for relative_path in paths:
        path = root / relative_path
        try:
            documents.append(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"Could not read {split} text file {relative_path!r}: {error}") from error
    return tuple(documents)


def _split_paths(manifest: Mapping[str, Any], split: str, *, required: bool) -> tuple[str, ...] | None:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Text manifest must contain a 'splits' mapping.")
    if split not in splits:
        if required:
            raise ValueError(f"Text manifest is missing required {split!r} split.")
        return None
    split_config = splits[split]
    if not isinstance(split_config, Mapping):
        raise ValueError(f"Text manifest split {split!r} must be a mapping containing 'files'.")
    files = split_config.get("files")
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
        raise ValueError(f"Text manifest split {split!r} 'files' must be an ordered sequence.")

    paths: list[str] = []
    for entry in files:
        if isinstance(entry, str):
            paths.append(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
            paths.append(entry["path"])
        else:
            raise ValueError(f"Text manifest split {split!r} file entries must be paths or mappings with 'path'.")
    return tuple(paths)


def _tokenizer_path(manifest: Mapping[str, Any]) -> Path:
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("Text manifest must contain a 'tokenizer' mapping.")
    value = tokenizer.get("artifact")
    if not isinstance(value, str) or not value:
        raise ValueError("Text manifest tokenizer must contain a non-empty 'artifact' path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Text manifest tokenizer artifact must be a relative path within the dataset root.")
    return path


def _string_field(manifest: Mapping[str, Any], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Text manifest must contain a non-empty {field!r} string.")
    return value
