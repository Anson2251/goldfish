import importlib.util
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).parents[2]
LANGUAGE_PROFILE = _REPOSITORY_ROOT / "model-profiles" / "language" / "gru-small.yaml"
sys.path.insert(0, str(_REPOSITORY_ROOT))


def _load_entry(name: str):
    path = _REPOSITORY_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"goldfish_pair_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


main = _load_entry("main")


def _write_pair_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
    (root / "manifest.yaml").write_text(
        """\
name: reverse_pairs
version: "1.0"
modality: text
builder: text_file_pairs
task: prefix_language_model
format:
  encoding: utf-8
  document_unit: file-pair
splits:
  train:
    files:
      - {input: train/01-input.txt, output: train/01-output.txt}
  val:
    files:
      - {input: val/01-input.txt, output: val/01-output.txt}
  test:
    files:
      - {input: test/01-input.txt, output: test/01-output.txt}
tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens: {pad: "<pad>", eos: "<eos>", sep: "<sep>"}
locking:
  dataset_lock: dataset-lock.json
""",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (root / split / "01-input.txt").write_text("abc", encoding="utf-8")
        (root / split / "01-output.txt").write_text("cba", encoding="utf-8")


def test_file_pair_prefix_lm_prepare_train_and_infer(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "pairs"
    runs = tmp_path / "runs"
    _write_pair_dataset(dataset)

    assert main(["prepare", str(dataset)]) == 0
    assert (dataset / "tokenizer" / "tokenizer.json").is_file()
    assert main([
        "train", str(dataset), "--runs-dir", str(runs), "--name", "pairs", "--epochs", "1",
        "--sequence-length", "8", "--batch-size", "1", "--model-profile", str(LANGUAGE_PROFILE),
        "--max-new-tokens", "1",
    ]) == 0

    run = runs / "exp1-pairs"
    assert (run / "checkpoints" / "final.pt").is_file()
    assert main(["infer", str(run), "--checkpoint", "final", "--prompt", "abc", "--max-new-tokens", "1"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("abc")
