"""Causal language-model corpora built from text documents."""

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from .batch import LanguageModelBatch, PrefixLanguageModelBatch
from .tokenizer import CharacterTokenizer, Tokenizer


class FilePairPrefixLanguageModelDataset(Dataset[PrefixLanguageModelBatch]):
    """Fixed-length prefix-LM rows for ``input -> output`` text-file pairs."""

    def __init__(
        self,
        pairs: Iterable[tuple[str, str]],
        tokenizer: CharacterTokenizer,
        sequence_length: int,
        *,
        fit_tokenizer: bool = False,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if tokenizer.pad_token_id is None:
            raise ValueError("FilePairPrefixLanguageModelDataset requires a tokenizer with a PAD token.")
        if tokenizer.sep_token_id is None:
            raise ValueError("FilePairPrefixLanguageModelDataset requires a tokenizer with a SEP token.")

        pairs = tuple(pairs)
        if fit_tokenizer:
            tokenizer.fit(text for pair in pairs for text in pair)

        self._rows = [self._encode_pair(input_text, output_text, tokenizer, sequence_length) for input_text, output_text in pairs]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> PrefixLanguageModelBatch:
        return self._rows[index]

    @staticmethod
    def _encode_pair(
        input_text: str, output_text: str, tokenizer: CharacterTokenizer, sequence_length: int
    ) -> PrefixLanguageModelBatch:
        tokens = tokenizer.encode(input_text) + [tokenizer.sep_token_id] + tokenizer.encode(output_text) + [tokenizer.eos_token_id]
        inputs, targets = tokens[:-1], tokens[1:]
        if len(inputs) > sequence_length:
            raise ValueError(
                f"File pair requires {len(inputs)} tokens, exceeding sequence_length {sequence_length}."
            )

        valid_length = len(inputs)
        output_start = len(tokenizer.encode(input_text))
        loss_mask = [False] * output_start + [True] * (valid_length - output_start)
        padding = sequence_length - valid_length
        return PrefixLanguageModelBatch(
            input_ids=torch.tensor(inputs + [tokenizer.pad_token_id] * padding, dtype=torch.long),
            target_ids=torch.tensor(targets + [tokenizer.pad_token_id] * padding, dtype=torch.long),
            attention_mask=torch.tensor([True] * valid_length + [False] * padding, dtype=torch.bool),
            loss_mask=torch.tensor(loss_mask + [False] * padding, dtype=torch.bool),
        )


class CausalLanguageModelDataset(Dataset[LanguageModelBatch]):
    """Contiguous fixed-length next-token windows over EOS-separated documents."""

    def __init__(
        self,
        documents: Iterable[str],
        tokenizer: Tokenizer,
        sequence_length: int,
        *,
        fit_tokenizer: bool = False,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if tokenizer.pad_token_id is None:
            raise ValueError("CausalLanguageModelDataset requires a tokenizer with a PAD token.")

        documents = tuple(documents)
        if fit_tokenizer:
            tokenizer.fit(documents)

        token_ids: list[int] = []
        for document in documents:
            token_ids.extend(tokenizer.encode(document))
            token_ids.append(tokenizer.eos_token_id)

        self._sequence_length = sequence_length
        self._pad_token_id = tokenizer.pad_token_id
        self._windows = self._make_windows(token_ids)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> LanguageModelBatch:
        input_ids, target_ids, attention_mask = self._windows[index]
        return LanguageModelBatch(input_ids, target_ids, attention_mask)

    def _make_windows(self, token_ids: list[int]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(token_ids) < 2:
            return []

        windows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        inputs_stream = token_ids[:-1]
        targets_stream = token_ids[1:]
        for start in range(0, len(inputs_stream), self._sequence_length):
            inputs = inputs_stream[start : start + self._sequence_length]
            targets = targets_stream[start : start + self._sequence_length]
            valid_length = len(inputs)
            if valid_length < self._sequence_length:
                padding = self._sequence_length - valid_length
                inputs.extend([self._pad_token_id] * padding)
                targets.extend([self._pad_token_id] * padding)
            windows.append(
                (
                    torch.tensor(inputs, dtype=torch.long),
                    torch.tensor(targets, dtype=torch.long),
                    torch.tensor(
                        [True] * valid_length + [False] * (self._sequence_length - valid_length),
                        dtype=torch.bool,
                    ),
                )
            )
        return windows


def build_train_validation_datasets(
    train_documents: Iterable[str],
    validation_documents: Iterable[str],
    sequence_length: int,
    *,
    tokenizer: Tokenizer | None = None,
) -> tuple[CausalLanguageModelDataset, CausalLanguageModelDataset, Tokenizer]:
    """Fit on training documents, then independently encode train and validation corpora."""
    tokenizer = tokenizer or CharacterTokenizer()
    train_documents = tuple(train_documents)
    validation_documents = tuple(validation_documents)
    tokenizer.fit(train_documents)
    return (
        CausalLanguageModelDataset(train_documents, tokenizer, sequence_length),
        CausalLanguageModelDataset(validation_documents, tokenizer, sequence_length),
        tokenizer,
    )
