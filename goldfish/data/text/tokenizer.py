"""Tokenizers for text data."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Text-to-token-ID mapping used by language-model datasets."""

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID, when the tokenizer defines one."""
        ...

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""
        ...

    @property
    def vocab_size(self) -> int:
        """Number of tokens, including special tokens."""
        ...

    def fit(self, texts: Iterable[str]) -> None:
        """Fit tokenizer state from training text only."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode one document without adding special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs, omitting special tokens."""
        ...

    def save(self, path: Path) -> None:
        """Persist tokenizer state."""
        ...


class CharacterTokenizer:
    """A deterministic character tokenizer with PAD and EOS special tokens."""

    _pad_token_id = 0
    _eos_token_id = 1

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._fitted = False

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id) + 2

    def fit(self, texts: Iterable[str]) -> None:
        """Create an ID mapping sorted lexicographically by character."""
        characters = {character for text in texts for character in text}
        self._token_to_id = {character: index for index, character in enumerate(sorted(characters), start=2)}
        self._id_to_token = {index: character for character, index in self._token_to_id.items()}
        self._fitted = True

    def encode(self, text: str) -> list[int]:
        self._require_fitted()
        try:
            return [self._token_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: Sequence[int]) -> str:
        self._require_fitted()
        characters: list[str] = []
        for token_id in token_ids:
            if token_id in (self.pad_token_id, self.eos_token_id):
                continue
            try:
                characters.append(self._id_to_token[token_id])
            except KeyError as error:
                raise ValueError(f"Unknown token ID: {token_id}") from error
        return "".join(characters)

    def save(self, path: Path) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "character",
                    "pad_token_id": self.pad_token_id,
                    "eos_token_id": self.eos_token_id,
                    "token_to_id": self._token_to_id,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        """Load and validate a tokenizer artifact written by :meth:`save`."""
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load character tokenizer artifact {path}: {error}") from error

        if not isinstance(artifact, dict):
            raise ValueError("Character tokenizer artifact must be a JSON object.")
        if artifact.get("type") != "character":
            raise ValueError("Character tokenizer artifact has an unsupported type.")
        if artifact.get("pad_token_id") != cls._pad_token_id or artifact.get("eos_token_id") != cls._eos_token_id:
            raise ValueError("Character tokenizer artifact has incompatible special token IDs.")

        token_to_id = artifact.get("token_to_id")
        if not isinstance(token_to_id, dict) or not all(
            isinstance(character, str) and len(character) == 1 and isinstance(token_id, int)
            for character, token_id in token_to_id.items()
        ):
            raise ValueError("Character tokenizer artifact has an invalid token_to_id mapping.")
        ids = list(token_to_id.values())
        if len(set(ids)) != len(ids) or set(ids) != set(range(2, len(ids) + 2)):
            raise ValueError("Character tokenizer artifact token IDs must be contiguous starting at 2.")

        tokenizer = cls()
        tokenizer._token_to_id = dict(token_to_id)
        tokenizer._id_to_token = {token_id: character for character, token_id in token_to_id.items()}
        tokenizer._fitted = True
        return tokenizer

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fit before encoding or decoding.")
