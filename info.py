"""Print a torchinfo model summary from a Goldfish model profile."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torchinfo import summary

from goldfish.config import create_model_from_config, load_model_profile, resolve_model_config
from goldfish.data.numeric import ForecastBatch


class _LanguageSummaryModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output, _ = self.model.forward_tokens(input_ids)  # type: ignore[attr-defined]
        return output.predictions["token_logits"]


class _ForecastSummaryModel(nn.Module):
    """Expose forecast architectures as tensor-to-tensor modules for torchinfo."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = ForecastBatch(
            inputs=inputs,
            targets=inputs.new_zeros(inputs.shape[0], 1, 1),
        )
        output = self.model.forward(batch)
        return output.predictions["forecast"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="YAML model profile path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size for the summary (default: 1).")
    parser.add_argument("--sequence-length", type=int, default=32, help="Synthetic sequence/lookback length (default: 32).")
    parser.add_argument("--vocab-size", type=int, default=128, help="Synthetic vocabulary size for language profiles (default: 128).")
    parser.add_argument("--feature-count", type=int, default=8, help="Synthetic feature count for forecast profiles (default: 8).")
    parser.add_argument("--target-count", type=int, default=1, help="Synthetic target count for forecast profiles (default: 1).")
    parser.add_argument("--horizon-count", type=int, default=1, help="Synthetic horizon count for forecast profiles (default: 1).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for value, name in ((args.batch_size, "batch size"), (args.sequence_length, "sequence length"), (args.vocab_size, "vocab size"), (args.feature_count, "feature count"), (args.target_count, "target count"), (args.horizon_count, "horizon count")):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    profile = load_model_profile(args.profile)
    if profile["family"] == "language":
        config = resolve_model_config(profile, family="language", runtime_parameters={"vocab_size": args.vocab_size})
        model = _LanguageSummaryModel(create_model_from_config(config))
        input_data = torch.zeros(args.batch_size, args.sequence_length, dtype=torch.long)
    elif profile["family"] == "forecast":
        config = resolve_model_config(
            profile,
            family="forecast",
            runtime_parameters={"feature_count": args.feature_count, "target_count": args.target_count, "horizon_count": args.horizon_count},
        )
        model = _ForecastSummaryModel(create_model_from_config(config))
        input_data = torch.zeros(args.batch_size, args.sequence_length, args.feature_count)
    else:
        raise ValueError(f"Unsupported model profile family: {profile['family']!r}")

    print(f"Profile: {args.profile}")
    print(f"Model:   {config['family']}/{config['name']}")
    print(summary(model, input_data=input_data, depth=4, verbose=0))
    return 0


if __name__ == "__main__":
    main()
