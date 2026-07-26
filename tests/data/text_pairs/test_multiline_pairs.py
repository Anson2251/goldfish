from pathlib import Path

import pytest

from goldfish.data.text.bundle import _read_split_file_pairs


def _manifest() -> dict:
    return {
        "format": {"document_unit": "file-pair"},
        "splits": {
            "train": {
                "files": [
                    {"input": "train/inputs.txt", "output": "train/outputs.txt"},
                ]
            }
        },
    }


def test_file_pair_shard_aligns_each_input_line_with_corresponding_output_line(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "inputs.txt").write_text("first\n\nthird\n", encoding="utf-8")
    (tmp_path / "train" / "outputs.txt").write_text("one\n\nthree\n", encoding="utf-8")

    pairs = _read_split_file_pairs(tmp_path, _manifest(), "train", required=True)

    assert pairs == (("first", "one"), ("", ""), ("third", "three"))


def test_file_pair_shard_rejects_mismatched_line_counts(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "inputs.txt").write_text("first\nsecond\n", encoding="utf-8")
    (tmp_path / "train" / "outputs.txt").write_text("one\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched line counts: input=2, output=1"):
        _read_split_file_pairs(tmp_path, _manifest(), "train", required=True)
