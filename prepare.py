"""Prepare a manifest-driven text or numeric dataset and its lock artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from goldfish.data.numeric import prepare_numeric_forecast_dataset
from goldfish.data.text import prepare_file_pair_prefix_language_model_dataset, prepare_text_dataset
from goldfish.data.validation import validator_registry, write_dataset_lock, write_tokenizer_lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Directory containing manifest.yaml and dataset splits.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = validator_registry.validate_manifest(args.dataset_root)
    if manifest["builder"] == "numeric_files_forecast":
        prepare_numeric_forecast_dataset(args.dataset_root, manifest)
    else:
        document_unit = manifest["format"]["document_unit"]
        if document_unit == "file-pair":
            prepare_file_pair_prefix_language_model_dataset(args.dataset_root, manifest)
        else:
            prepare_text_dataset(args.dataset_root, manifest)
        write_dataset_lock(args.dataset_root, manifest)
        write_tokenizer_lock(args.dataset_root, manifest)
    print(f"Prepared dataset: {args.dataset_root}")
    return 0


if __name__ == "__main__":
    main()
