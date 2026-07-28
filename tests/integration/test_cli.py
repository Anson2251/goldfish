import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

_REPOSITORY_ROOT = Path(__file__).parents[2]
LANGUAGE_PROFILE = _REPOSITORY_ROOT / "model-profiles" / "language" / "gru-small.yaml"
sys.path.insert(0, str(_REPOSITORY_ROOT))
_ENTRY_POINT = _REPOSITORY_ROOT / "main.py"
_SPEC = importlib.util.spec_from_file_location("goldfish_main", _ENTRY_POINT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def _write_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
    (root / "manifest.yaml").write_text(
        """\
name: alphabet
version: "1.0"
modality: text
builder: text_files_lm
task: causal_language_model
format:
  encoding: utf-8
  document_unit: file
  append_eos: true
splits:
  train:
    files:
      - train/01-alphabet.txt
  val:
    files:
      - val/01-alphabet.txt
  test:
    files:
      - test/01-alphabet.txt
tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens:
    pad: "<pad>"
    eos: "<eos>"
locking:
  dataset_lock: dataset-lock.json
""",
        encoding="utf-8",
    )
    (root / "train" / "01-alphabet.txt").write_text("abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8")
    (root / "val" / "01-alphabet.txt").write_text("bcdefghijklmnopqrstuvwxyza\n", encoding="utf-8")
    (root / "test" / "01-alphabet.txt").write_text("zabcdefghijklmnopqrstuvwxy\n", encoding="utf-8")


try:
    import goldfish.data.validation as validation
except ImportError:
    _DATA_WORKFLOW_READY = False
else:
    _DATA_WORKFLOW_READY = all(
        callable(getattr(validation, name, None))
        for name in ("validate_tokenizer_lock", "write_dataset_lock", "write_tokenizer_lock")
    )


@pytest.mark.xfail(
    not _DATA_WORKFLOW_READY,
    reason="Dataset/tokenizer lock APIs have not landed yet.",
    strict=False,
)
def test_prepare_creates_dataset_and_tokenizer_locks(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    assert main(["prepare", str(tmp_path)]) == 0

    assert (tmp_path / "dataset-lock.json").is_file()
    assert (tmp_path / "tokenizer" / "tokenizer.json").is_file()
    assert (tmp_path / "tokenizer" / "tokenizer-lock.json").is_file()


@pytest.mark.xfail(
    not _DATA_WORKFLOW_READY,
    reason="Dataset/tokenizer lock APIs have not landed yet.",
    strict=False,
)
def test_train_consumes_prepared_dataset_and_uses_a_nonempty_default_prompt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_dataset(tmp_path)
    assert main(["prepare", str(tmp_path)]) == 0

    assert main(["train", str(tmp_path), "--sequence-length", "4", "--batch-size", "2", "--epochs", "1", "--model-profile", str(LANGUAGE_PROFILE), "--max-new-tokens", "0"]) == 0

    output = capsys.readouterr().out
    assert "Validation metrics:" in output
    assert "Generated sample: abcdefghijklmnopqrstuvwxyz" in output


def test_train_creates_an_auditable_run_and_strictly_resumes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    runs = tmp_path / "runs"
    _write_dataset(dataset)
    assert main(["prepare", str(dataset)]) == 0

    assert main([
        "train", str(dataset), "--runs-dir", str(runs), "--name", "tiny", "--sequence-length", "4",
        "--batch-size", "2", "--epochs", "1", "--model-profile", str(LANGUAGE_PROFILE),
        "--sample-frequency", "1", "--checkpoint-frequency", "1", "--max-new-tokens", "0",
    ]) == 0

    run = runs / "exp1-tiny"
    assert (run / "config.yaml").is_file()
    assert (run / "data.json").is_file()
    assert (run / "environment.json").is_file()
    assert (run / "artifacts" / "samples" / "epoch-0001.txt").is_file()
    assert (run / "artifacts" / "samples" / "final.txt").is_file()
    for filename in ("latest.pt", "best.pt", "final.pt", "epoch-0001.pt"):
        assert (run / "checkpoints" / filename).is_file()
    payload = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert payload["format"] == "goldfish-checkpoint-v1"
    assert len(payload["provenance"]["config_fingerprint"]) == 64
    assert len(payload["provenance"]["dataset_fingerprint"]) == 64
    assert len(payload["provenance"]["tokenizer_fingerprint"]) == 64

    assert main(["train", str(dataset), "--resume", str(run), "--epochs", "1", "--max-new-tokens", "0"]) == 0

    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in metrics] == [1, 2]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["last_epoch"] == 2


def test_train_rejects_legacy_corpus_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([str(tmp_path / "corpus.txt")])
