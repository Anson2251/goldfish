from collections.abc import Iterable

import pytest
import torch
from torch.utils.data import DataLoader

from goldfish.core import Batch
from goldfish.data.text import (
    CausalLanguageModelDataset,
    CharacterTokenizer,
    LanguageModelBatch,
    Tokenizer,
    build_train_validation_datasets,
    collate_language_model_batches,
)


def test_character_tokenizer_has_deterministic_mappings_and_special_token_semantics() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["cab", "bad"])

    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.vocab_size == 6
    assert tokenizer.encode("cab") == [4, 2, 3]
    assert tokenizer.decode([4, 2, 3]) == "cab"
    assert tokenizer.decode([4, tokenizer.eos_token_id, tokenizer.pad_token_id, 2]) == "ca"


def test_character_tokenizer_requires_fit_and_rejects_unknown_characters() -> None:
    tokenizer = CharacterTokenizer()

    with pytest.raises(RuntimeError, match="fit"):
        tokenizer.encode("a")

    tokenizer.fit(["a"])
    with pytest.raises(ValueError, match="Unknown"):
        tokenizer.encode("b")


def test_language_model_batch_moves_all_tensors_and_meets_batch_contract() -> None:
    batch = LanguageModelBatch(
        input_ids=torch.tensor([[2, 3]]),
        target_ids=torch.tensor([[3, 1]]),
        attention_mask=torch.tensor([[True, True]]),
    )

    moved = batch.to(torch.device("cpu"))

    assert isinstance(batch, Batch)
    assert moved is not batch
    assert moved.input_ids.device.type == "cpu"
    assert moved.target_ids.device.type == "cpu"
    assert moved.attention_mask.device.type == "cpu"
    assert moved.attention_mask.dtype is torch.bool


def test_dataset_inserts_eos_and_creates_shifted_fixed_length_windows() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["ab", "c"])
    dataset = CausalLanguageModelDataset(["ab", "c"], tokenizer, sequence_length=3)

    row = dataset[0]

    assert len(dataset) == 2
    assert row.input_ids.tolist() == [2, 3, 1]
    assert row.target_ids.tolist() == [3, 1, 4]
    assert row.attention_mask.tolist() == [True, True, True]


def test_dataset_pads_a_tail_window_and_masks_its_padding() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["abcd"])
    dataset = CausalLanguageModelDataset(["abcd"], tokenizer, sequence_length=3)

    assert len(dataset) == 2
    tail = dataset[1]

    assert tail.input_ids.tolist() == [5, tokenizer.pad_token_id, tokenizer.pad_token_id]
    assert tail.target_ids.tolist() == [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.pad_token_id]
    assert tail.attention_mask.tolist() == [True, False, False]


def test_collate_creates_a_rank_two_language_model_batch() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["abcd"])
    dataset = CausalLanguageModelDataset(["abcd"], tokenizer, sequence_length=3)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_language_model_batches)

    batch = next(iter(loader))

    assert batch.input_ids.shape == (2, 3)
    assert batch.target_ids.shape == (2, 3)
    assert batch.attention_mask.shape == (2, 3)


def test_train_validation_builder_fits_only_training_documents() -> None:
    train_documents = iter(["ab"])
    validation_documents = iter(["aa"])

    train_dataset, validation_dataset, tokenizer = build_train_validation_datasets(
        train_documents,
        validation_documents,
        sequence_length=2,
    )

    assert train_dataset[0].input_ids.tolist() == [2, 3]
    assert validation_dataset[0].input_ids.tolist() == [2, 2]
    assert validation_dataset[0].target_ids.tolist() == [2, tokenizer.eos_token_id]
    assert validation_dataset[0].attention_mask.tolist() == [True, True]
    assert tokenizer.vocab_size == 4


def test_dataset_accepts_one_shot_document_iterables() -> None:
    def documents() -> Iterable[str]:
        yield "ab"

    tokenizer = CharacterTokenizer()
    dataset = CausalLanguageModelDataset(documents(), tokenizer, sequence_length=2, fit_tokenizer=True)

    assert tokenizer.decode(dataset[0].input_ids.tolist()) == "ab"
