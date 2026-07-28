"""Generate text from a managed Goldfish language-model experiment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from goldfish.data.text import FilePairPrefixLanguageModelDataModule, TextFilesLanguageModelDataModule
from goldfish.device import resolve_device
from goldfish.data.validation import validate_dataset_lock, validate_tokenizer_lock, validator_registry
from goldfish.experiments import CHECKPOINT_FORMAT
from goldfish.generation import generate_text
from goldfish.generation.text import TextTokenizer
from goldfish.config import create_model_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Experiment run directory containing config.yaml and checkpoints.")
    parser.add_argument("--checkpoint", choices=("best", "latest", "final"), default="best", help="Checkpoint to load (default: best).")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), help="Execution device; defaults to CUDA, then MPS, then CPU.")
    parser.add_argument("--prompt", required=True, help="Prompt to continue, or the input side for a file-pair run.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum generated tokens (default: 100).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; omit for greedy decoding.")
    parser.add_argument("--top-k", type=int, help="Sample only from the top K candidates.")
    return parser


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Experiment config not found: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("Experiment config.yaml must be a mapping.")
    return config


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Experiment config is missing mapping {key!r}.")
    return value


def _build_model(config: Mapping[str, Any], tokenizer: TextTokenizer) -> torch.nn.Module:
    model_config = _mapping(config, "model")
    model = create_model_from_config(model_config)
    parameters = _mapping(model_config, "parameters")
    if int(parameters["vocab_size"]) != tokenizer.vocab_size:
        raise ValueError("Experiment model vocabulary size does not match the current dataset tokenizer.")
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 0:
        raise ValueError("max new tokens must be non-negative.")
    if args.temperature is not None and args.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("top k must be positive.")

    device = resolve_device(args.device)
    config = _load_config(args.run_dir)
    dataset_config = _mapping(config, "dataset")
    loader_config = _mapping(config, "loader")
    dataset_root = Path(cast(str, dataset_config["root"]))
    manifest = validator_registry.validate_manifest(dataset_root)
    if manifest.get("builder") == "numeric_files_forecast":
        raise ValueError("Numeric runs do not support text inference; use 'goldfish forecast' instead.")
    validate_dataset_lock(dataset_root, manifest)
    validate_tokenizer_lock(dataset_root, manifest)
    document_unit = manifest["format"]["document_unit"]
    data_module_class = FilePairPrefixLanguageModelDataModule if document_unit == "file-pair" else TextFilesLanguageModelDataModule
    data_module = data_module_class(
        dataset_root,
        manifest,
        sequence_length=int(loader_config["sequence_length"]),
        batch_size=int(loader_config["batch_size"]),
    )
    tokenizer = data_module.tokenizer
    model = _build_model(config, tokenizer)

    checkpoint_path = args.run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as error:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}") from error
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    prefix_token_ids = [sep_token_id] if document_unit == "file-pair" and sep_token_id is not None else []
    text = generate_text(
        model,
        cast(TextTokenizer, tokenizer),
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        prefix_token_ids=prefix_token_ids,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
    return 0


if __name__ == "__main__":
    main()
