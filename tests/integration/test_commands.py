import importlib.util
from pathlib import Path
import sys

import torch

_REPOSITORY_ROOT = Path(__file__).parents[2]
LANGUAGE_PROFILE = _REPOSITORY_ROOT / "model-profiles" / "language" / "gru-small.yaml"
sys.path.insert(0, str(_REPOSITORY_ROOT))


def _load_entry(name: str):
    path = _REPOSITORY_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"goldfish_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


main = _load_entry("main")
train_main = _load_entry("train")
infer_main = _load_entry("infer")


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
    files: [train/01.txt]
  val:
    files: [val/01.txt]
  test:
    files: [test/01.txt]
tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens: {pad: "<pad>", eos: "<eos>"}
locking:
  dataset_lock: dataset-lock.json
""",
        encoding="utf-8",
    )
    (root / "train" / "01.txt").write_text("abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8")
    (root / "val" / "01.txt").write_text("bcdefghijklmnopqrstuvwxyza\n", encoding="utf-8")
    (root / "test" / "01.txt").write_text("zabcdefghijklmnopqrstuvwxy\n", encoding="utf-8")


def test_dispatcher_forwards_train_arguments_unchanged(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    runs = tmp_path / "runs"
    _write_dataset(dataset)

    assert main(["prepare", str(dataset)]) == 0
    assert main([
        "train", str(dataset), "--runs-dir", str(runs), "--name", "dispatch", "--epochs", "1",
        "--sequence-length", "4", "--batch-size", "2", "--model-profile", str(LANGUAGE_PROFILE),
        "--max-new-tokens", "0",
    ]) == 0

    assert (runs / "exp1-dispatch" / "checkpoints" / "final.pt").is_file()
    assert (runs / "exp1-dispatch" / "artifacts" / "plots" / "training-curves.png").is_file()


def test_train_entrypoint_accepts_train_arguments_without_subcommand(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    runs = tmp_path / "runs"
    _write_dataset(dataset)
    assert main(["prepare", str(dataset)]) == 0

    assert train_main([
        str(dataset), "--runs-dir", str(runs), "--name", "direct", "--epochs", "1", "--sequence-length", "4",
        "--batch-size", "2", "--model-profile", str(LANGUAGE_PROFILE), "--max-new-tokens", "0",
    ]) == 0


def test_infer_generates_from_a_managed_run(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "dataset"
    runs = tmp_path / "runs"
    _write_dataset(dataset)
    assert main(["prepare", str(dataset)]) == 0
    assert train_main([
        str(dataset), "--runs-dir", str(runs), "--name", "infer", "--epochs", "1", "--sequence-length", "4",
        "--batch-size", "2", "--model-profile", str(LANGUAGE_PROFILE), "--max-new-tokens", "0", "--seed", "0",
    ]) == 0

    run = runs / "exp1-infer"
    assert infer_main([str(run), "--checkpoint", "final", "--prompt", "abc", "--max-new-tokens", "2"]) == 0

    assert capsys.readouterr().out.splitlines()[-1].startswith("abc")
    checkpoint = torch.load(run / "checkpoints" / "final.pt", weights_only=False)
    assert "model" in checkpoint
