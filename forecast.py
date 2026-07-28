"""Export raw-unit forecasts from a managed numeric Goldfish experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from goldfish.data.numeric import NumericFilesForecastDataModule
from goldfish.data.validation import validate_dataset_lock, validate_normalizer_lock, validator_registry
from goldfish.device import resolve_device
from goldfish.experiments import CHECKPOINT_FORMAT
from goldfish.models import model_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Numeric experiment run directory.")
    parser.add_argument("--checkpoint", choices=("best", "latest", "final"), default="best")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, help="JSONL destination; defaults under the run artifacts directory.")
    parser.add_argument("--batch-size", type=int, help="Prediction batch size; defaults to the saved run loader batch size.")
    parser.add_argument("--plot", type=Path, help="Optional PNG output showing one history, prediction, and actual trajectory.")
    parser.add_argument("--plot-window", type=int, default=0, help="Zero-based forecast window to plot (default: 0).")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{description} not found: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping: {path}")
    return value


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise ValueError(f"Run config is missing mapping {name!r}.")
    return result


def _output_path(run_dir: Path, checkpoint: str, split: str, requested: Path | None) -> Path:
    return requested if requested is not None else run_dir / "artifacts" / "forecasts" / f"{split}-{checkpoint}.jsonl"


def _plot_forecast(
    path: Path,
    *,
    entity_id: str,
    cutoff_timestamp: str,
    features: list[str],
    targets: list[str],
    horizons: list[int],
    history: torch.Tensor,
    prediction: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    """Save raw-unit history, forecast, and actual target trajectories as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    target_positions = {name: index for index, name in enumerate(features)}
    history_positions = [target_positions[name] for name in targets]
    history_targets = history[:, history_positions]
    lookback = history.shape[0]
    horizon_positions = list(horizons)
    figure, axes = pyplot.subplots(len(targets), 1, figsize=(11, 3.5 * len(targets)), sharex=True, squeeze=False)
    for target_index, target in enumerate(targets):
        axis = axes[target_index, 0]
        history_x = list(range(-lookback + 1, 1))
        axis.plot(history_x, history_targets[:, target_index].tolist(), color="C0", label="history")
        axis.plot([0, *horizon_positions], [history_targets[-1, target_index].item(), *actual[:, target_index].tolist()], color="C2", marker="o", label="actual")
        axis.plot([0, *horizon_positions], [history_targets[-1, target_index].item(), *prediction[:, target_index].tolist()], color="C1", linestyle="--", marker="x", label="forecast")
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_ylabel(target)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("Row offset from cutoff")
    figure.suptitle(f"Forecast: {entity_id} at {cutoff_timestamp}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    pyplot.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch size must be positive.")
    if args.plot_window < 0:
        raise ValueError("plot window must be non-negative.")
    config = _load_mapping(args.run_dir / "config.yaml", "experiment config")
    data = _load_mapping(args.run_dir / "data.json", "experiment data")
    dataset_config, model_config, loader_config = _mapping(config, "dataset"), _mapping(config, "model"), _mapping(config, "loader")
    if model_config.get("family") != "forecast":
        raise ValueError("goldfish forecast requires a numeric forecasting run.")
    dataset_root = Path(cast(str, dataset_config["root"]))
    manifest = validator_registry.validate_manifest(dataset_root)
    dataset_lock = validate_dataset_lock(dataset_root, manifest)
    normalizer_lock = validate_normalizer_lock(dataset_root, manifest)
    locking = _mapping(data, "locking")
    stored_normalizer = _mapping(data, "normalizer")
    if locking.get("dataset_fingerprint") != dataset_lock.get("fingerprint"):
        raise ValueError("forecast dataset fingerprint does not match the managed run.")
    if stored_normalizer.get("fingerprint") != normalizer_lock.get("fingerprint"):
        raise ValueError("forecast normalizer fingerprint does not match the managed run.")

    batch_size = args.batch_size or int(loader_config["batch_size"])
    data_module = NumericFilesForecastDataModule(dataset_root, manifest, batch_size=batch_size)
    loader = data_module.val_dataloader() if args.split == "val" else data_module.test_dataloader()
    model = model_registry.create(
        "forecast", str(model_config["name"]), feature_count=int(model_config["feature_count"]),
        target_count=int(model_config["target_count"]), horizon_count=int(model_config["horizon_count"]),
        hidden_dim=int(model_config["hidden_dim"]), num_layers=int(model_config["num_layers"]), dropout=float(model_config["dropout"]),
    )
    checkpoint_path = args.run_dir / "checkpoints" / f"{args.checkpoint}.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except FileNotFoundError as error:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}") from error
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}.")
    model.load_state_dict(checkpoint["model"])
    device = resolve_device(args.device)
    model.to(device).eval()

    output_path = _output_path(args.run_dir, args.checkpoint, args.split, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = data_module.runtime_metadata["targets"]
    horizons = data_module.runtime_metadata["horizons"]
    if not isinstance(targets, list) or not isinstance(horizons, list):
        raise ValueError("Numeric runtime metadata is invalid.")
    absolute_errors: list[torch.Tensor] = []
    squared_errors: list[torch.Tensor] = []
    count = 0
    selected_plot: tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    with output_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in loader:
            output = model(batch.to(device))
            raw_history = data_module.normalizer.inverse_transform_features(batch.inputs).cpu()
            predictions = data_module.normalizer.inverse_transform_targets(output.predictions["forecast"], targets).cpu()
            actuals = data_module.normalizer.inverse_transform_targets(batch.targets, targets).cpu()
            errors = predictions - actuals
            absolute_errors.append(errors.abs().reshape(-1))
            squared_errors.append(errors.square().reshape(-1))
            for index, (entity_id, cutoff) in enumerate(zip(batch.entity_ids, batch.cutoff_timestamps, strict=True)):
                record = {"entity_id": entity_id, "cutoff_timestamp": cutoff, "split": args.split, "targets": targets, "horizons": horizons, "prediction": predictions[index].tolist(), "target": actuals[index].tolist()}
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                if count == args.plot_window:
                    selected_plot = (entity_id, cutoff, raw_history[index], predictions[index], actuals[index])
                count += 1
    if count == 0:
        mae = rmse = None
    else:
        mae = float(torch.cat(absolute_errors).mean())
        rmse = float(torch.cat(squared_errors).mean().sqrt())
    summary = {"checkpoint": args.checkpoint, "split": args.split, "window_count": count, "targets": targets, "horizons": horizons, "metrics": {"mae": mae, "rmse": rmse}, "raw_units": True}
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        if selected_plot is None:
            raise ValueError(f"plot window {args.plot_window} is unavailable; {count} forecast windows were exported.")
        entity_id, cutoff, history, prediction, actual = selected_plot
        _plot_forecast(args.plot, entity_id=entity_id, cutoff_timestamp=cutoff, features=list(data_module.normalizer.features), targets=targets, horizons=horizons, history=history, prediction=prediction, actual=actual)
    print(f"Exported {count} {args.split} forecasts: {output_path}")
    return 0


if __name__ == "__main__":
    main()
