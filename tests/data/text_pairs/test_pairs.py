import json
from pathlib import Path

import pytest
import torch

from goldfish.core import Batch
from goldfish.data.text import (
    CharacterTokenizer,
    FilePairPrefixLanguageModelDataset,
    PrefixLanguageModelBatch,
    create_file_pair_prefix_language_model_data_module,
    prepare_file_pair_prefix_language_model_dataset,
)


def _manifest() -> dict[str, object]:
    return {
        "name": "tiny-pairs",
        "modality": "text",
        "format": {"document_unit": "file-pair"},
        "splits": {
            "train": {"files": [{"input": "train/input.txt", "output": "train/output.txt"}]},
            "val": {"files": [{"input": "val/input.txt", "output": "val/output.txt"}]},
        },
        "tokenizer": {"name": "character", "artifact": "artifacts/tokenizer.json", "fit_split": "train"},
    }


def _write_pairs(root: Path) -> None:
    for split in ("train", "val"):
        (root / split).mkdir(parents=True)
    (root / "train/input.txt").write_text("ab", encoding="utf-8")
    (root / "train/output.txt").write_text("cd", encoding="utf-8")
    (root / "val/input.txt").write_text("a", encoding="utf-8")
    (root / "val/output.txt").write_text("d", encoding="utf-8")


def test_sep_is_opt_in_and_old_artifacts_remain_loadable(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer(with_sep_token=True)
    tokenizer.fit(["ab"])

    assert tokenizer.sep_token_id == 2
    assert tokenizer.vocab_size == 5
    assert tokenizer.encode("ab") == [3, 4]
    assert tokenizer.decode([3, 2, 4, 1, 0]) == "ab"

    old_artifact = tmp_path / "old.json"
    old_artifact.write_text(
        json.dumps({"type": "character", "pad_token_id": 0, "eos_token_id": 1, "token_to_id": {"a": 2}}),
        encoding="utf-8",
    )
    loaded = CharacterTokenizer.load(old_artifact)
    assert loaded.sep_token_id is None
    assert loaded.vocab_size == 3
    assert loaded.encode("a") == [2]


def test_file_pair_dataset_encodes_prefix_and_masks_only_output_targets() -> None:
    tokenizer = CharacterTokenizer(with_sep_token=True)
    tokenizer.fit(["ab", "cd"])
    dataset = FilePairPrefixLanguageModelDataset([("ab", "cd")], tokenizer, sequence_length=6)

    row = dataset[0]

    assert isinstance(row, PrefixLanguageModelBatch)
    assert isinstance(row, Batch)
    assert row.input_ids.tolist() == [3, 4, 2, 5, 6, 0]
    assert row.target_ids.tolist() == [4, 2, 5, 6, 1, 0]
    assert row.attention_mask.tolist() == [True, True, True, True, True, False]
    assert row.loss_mask.tolist() == [False, False, True, True, True, False]
    assert row.to(torch.device("cpu")).loss_mask.dtype is torch.bool


def test_pair_prepare_fits_both_train_sides_and_module_exposes_metadata(tmp_path: Path) -> None:
    _write_pairs(tmp_path)
    manifest = _manifest()

    prepared = prepare_file_pair_prefix_language_model_dataset(tmp_path, manifest)
    module = create_file_pair_prefix_language_model_data_module(
        tmp_path, manifest, sequence_length=5, batch_size=1
    )

    assert prepared.train_document_count == 1
    assert prepared.tokenizer.encode("abcd") == [3, 4, 5, 6]
    assert module.runtime_metadata == {
        "name": "tiny-pairs",
        "modality": "text",
        "document_unit": "file-pair",
        "vocab_size": 7,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "sep_token_id": 2,
        "sample_counts": {"train": 1, "val": 1},
    }
    val_loader = module.val_dataloader()
    assert val_loader is not None
    batch = next(iter(val_loader))
    assert batch.input_ids.tolist() == [[3, 2, 6, 0, 0]]
    assert batch.target_ids.tolist() == [[2, 6, 1, 0, 0]]
    assert batch.attention_mask.tolist() == [[True, True, True, False, False]]
    assert batch.loss_mask.tolist() == [[False, True, True, False, False]]


def test_pair_module_reports_unknown_validation_characters_and_requires_file_pair_format(tmp_path: Path) -> None:
    _write_pairs(tmp_path)
    manifest = _manifest()
    prepare_file_pair_prefix_language_model_dataset(tmp_path, manifest)
    (tmp_path / "val/output.txt").write_text("z", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unknown character: 'z'.*val file-pair"):
        create_file_pair_prefix_language_model_data_module(tmp_path, manifest, sequence_length=5, batch_size=1)

    manifest["format"] = {"document_unit": "file"}
    with pytest.raises(ValueError, match="file-pair"):
        prepare_file_pair_prefix_language_model_dataset(tmp_path, manifest)
