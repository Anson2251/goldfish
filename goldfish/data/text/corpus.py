"""Causal language-model corpora built from text documents."""

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from .batch import LanguageModelBatch
from .tokenizer import CharacterTokenizer, Tokenizer


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
