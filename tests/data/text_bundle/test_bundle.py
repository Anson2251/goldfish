from pathlib import Path

import pytest

from goldfish.data.text import CharacterTokenizer, TextFilesLanguageModelDataModule, prepare_text_dataset


def _manifest() -> dict[str, object]:
    return {
        "name": "tiny-text",
        "modality": "text",
        "splits": {
            "train": {"files": ["train/second.txt", {"path": "train/first.txt", "source": "fixture"}]},
            "val": {"files": ["val.txt"]},
            "test": {"files": ["test.txt"]},
        },
        "tokenizer": {"name": "character", "artifact": "artifacts/tokenizer.json", "fit_split": "train"},
    }


def _write_corpus(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "train" / "first.txt").write_text("ab", encoding="utf-8")
    (root / "train" / "second.txt").write_text("bc", encoding="utf-8")
    (root / "val.txt").write_text("ca", encoding="utf-8")
    (root / "test.txt").write_text("ab", encoding="utf-8")


def test_character_tokenizer_load_round_trips_saved_state(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["cab"])
    artifact = tmp_path / "tokenizer.json"
    tokenizer.save(artifact)

    loaded = CharacterTokenizer.load(artifact)

    assert loaded.encode("cab") == tokenizer.encode("cab")
    assert loaded.decode([4, 2, 3]) == "cab"
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.pad_token_id == tokenizer.pad_token_id
    assert loaded.eos_token_id == tokenizer.eos_token_id


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ('{"type": "other", "pad_token_id": 0, "eos_token_id": 1, "token_to_id": {}}', "type"),
        ('{"type": "character", "pad_token_id": 2, "eos_token_id": 1, "token_to_id": {}}', "special"),
        ('{"type": "character", "pad_token_id": 0, "eos_token_id": 1, "token_to_id": {"a": 3}}', "contiguous"),
    ],
)
def test_character_tokenizer_load_rejects_invalid_artifacts(tmp_path: Path, artifact: str, message: str) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text(artifact, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CharacterTokenizer.load(path)


def test_prepare_text_dataset_fits_train_in_manifest_order_and_writes_tokenizer(tmp_path: Path) -> None:
    _write_corpus(tmp_path)

    prepared = prepare_text_dataset(tmp_path, _manifest())

    assert prepared.tokenizer_path == tmp_path / "artifacts/tokenizer.json"
    assert prepared.train_document_count == 2
    assert prepared.tokenizer.decode(prepared.tokenizer.encode("abc")) == "abc"
    assert prepared.tokenizer_path.is_file()


def test_data_module_loads_frozen_tokenizer_and_optional_split_loaders(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    manifest = _manifest()
    prepare_text_dataset(tmp_path, manifest)

    module = TextFilesLanguageModelDataModule(tmp_path, manifest, sequence_length=2, batch_size=2)

    train = module.train_dataset
    val_loader = module.val_dataloader()
    test_loader = module.test_dataloader()
    assert val_loader is not None
    assert test_loader is not None
    val = next(iter(val_loader))
    test = next(iter(test_loader))

    assert train is not None
    assert train[0].input_ids.tolist() == [3, 4]  # "bc" is listed before "ab".
    assert train[1].input_ids.tolist() == [1, 2]
    assert val.input_ids.tolist() == [[4, 2]]
    assert test.input_ids.tolist() == [[2, 3]]
    assert module.runtime_metadata == {
        "name": "tiny-text",
        "modality": "text",
        "vocab_size": 5,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "sample_counts": {"train": 3, "val": 1, "test": 1},
    }


def test_data_module_does_not_fit_and_reports_val_unknown_characters(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    manifest = _manifest()
    prepare_text_dataset(tmp_path, manifest)
    (tmp_path / "val.txt").write_text("z", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unknown character: 'z'.*val.txt"): 
        TextFilesLanguageModelDataModule(tmp_path, manifest, sequence_length=2, batch_size=1)


def test_data_module_omits_missing_optional_splits(tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("ab", encoding="utf-8")
    manifest = {
        "name": "train-only",
        "modality": "text",
        "splits": {"train": {"files": ["train.txt"]}},
        "tokenizer": {"name": "character", "artifact": "tokenizer.json", "fit_split": "train"},
    }
    prepare_text_dataset(tmp_path, manifest)

    module = TextFilesLanguageModelDataModule(tmp_path, manifest, sequence_length=2, batch_size=1)

    assert module.val_dataloader() is None
    assert module.test_dataloader() is None
    assert module.runtime_metadata["sample_counts"] == {"train": 1}
