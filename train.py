"""Train or strictly resume a character language-model experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy
import torch
import yaml
from torchinfo import summary

from goldfish.config import create_model_from_config, create_optimizer, create_scheduler, load_model_profile, resolve_model_config, resolve_training_config
from goldfish.config.observability import resolve_observability_config
from goldfish.observability import JsonlRecorder, build_probe_hook, take_first_batches
from goldfish.data.loading import resolve_loader_settings
from goldfish.core import Task
from goldfish.device import resolve_device
from goldfish.experiments import CHECKPOINT_FORMAT, CheckpointManager, ExperimentRun, build_data_provenance, collect_environment
from goldfish.generation import generate_text
from goldfish.generation.text import TextTokenizer

from goldfish.tasks import CausalLanguageModelTask, PointForecastTask, PrefixLanguageModelTask
from goldfish.training import Trainer, compile_model


def _future_attribute(module_name: str, name: str) -> Any:
    """Load an attribute supplied by the dataset-workflow implementation."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise RuntimeError(f"Dataset workflow support is unavailable: expected {module_name}.{name}.") from error
        raise
    attribute = getattr(module, name, None)
    if attribute is None:
        raise RuntimeError(f"Dataset workflow support is unavailable: expected {module_name}.{name}.")
    return attribute


def _future_api(module_name: str, name: str) -> Callable[..., Any]:
    api = _future_attribute(module_name, name)
    if not callable(api):
        raise RuntimeError(f"Dataset workflow support is unavailable: expected {module_name}.{name}.")
    return api


def _validate_manifest(dataset_root: Path) -> Any:
    validator_registry = _future_attribute("goldfish.data.validation", "validator_registry")
    validate_manifest = getattr(validator_registry, "validate_manifest", None)
    if not callable(validate_manifest):
        raise RuntimeError("Dataset workflow support is unavailable: expected validator_registry.validate_manifest.")
    return validate_manifest(dataset_root)


def _default_prompt(dataset_root: Path, manifest: Any) -> str:
    """Return the first non-empty manifest-listed training document for generation."""
    try:
        entries = manifest["splits"]["train"]["files"]
    except (KeyError, TypeError) as error:
        raise ValueError("Manifest must declare train split files to select a generation prompt.") from error
    if not isinstance(entries, list):
        raise ValueError("Manifest train split files must be a list.")
    document_unit = manifest.get("format", {}).get("document_unit") if isinstance(manifest, Mapping) else None
    for entry in entries:
        if document_unit == "file-pair" and isinstance(entry, Mapping):
            relative_path = entry.get("input")
        else:
            relative_path = entry if isinstance(entry, str) else entry.get("path") if isinstance(entry, Mapping) else None
        if not isinstance(relative_path, str):
            raise ValueError("Manifest train split entries must provide a text path or file-pair input path.")
        document = (dataset_root / relative_path).read_text(encoding="utf-8")
        if document:
            return document
    raise ValueError("Manifest-listed training documents must contain text for generation.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Prepared dataset directory containing manifest.yaml.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Experiment run directory (default: runs).")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), help="Execution device; defaults to CUDA, then MPS, then CPU.")
    parser.add_argument("--name", help="Optional name appended to a new run directory.")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="Resume an existing run from latest.pt.")
    parser.add_argument("--resume-loose", action="store_true", help="Allow loader and optimization overrides while preserving the run's model and data locks.")
    parser.add_argument("--sequence-length", type=int, default=64, help="Tokens in each training window (default: 64).")
    parser.add_argument("--batch-size", type=int, default=32, help="Training and validation batch size (default: 32).")
    parser.add_argument("--num-workers", default="auto", help="Total DataLoader worker budget: 'auto' reserves 20%% CPU and splits the rest 60:40 (default: auto).")
    parser.add_argument("--train-workers", type=int, help="Override training DataLoader workers.")
    parser.add_argument("--val-workers", type=int, help="Override validation/test DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker (default: 2).")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train; on resume these are additional epochs (default: 5).")
    parser.add_argument("--model-profile", type=Path, help="YAML model architecture profile; required for new runs.")
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-3, help="Optimizer learning rate (default: 0.001).")
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    parser.add_argument("--weight-decay", type=float, help="Optimizer weight decay (optimizer default when omitted).")
    parser.add_argument("--momentum", type=float, help="SGD momentum.")
    parser.add_argument("--scheduler", choices=("none", "cosine", "step", "exponential", "plateau"), default="none")
    parser.add_argument("--scheduler-step-timing", choices=("batch", "epoch", "validation"))
    parser.add_argument("--scheduler-t-max", type=int, help="Cosine scheduler T_max.")
    parser.add_argument("--scheduler-eta-min", type=float, help="Cosine scheduler minimum learning rate.")
    parser.add_argument("--scheduler-step-size", type=int, help="Step scheduler interval.")
    parser.add_argument("--scheduler-gamma", type=float, help="Step/exponential scheduler gamma.")
    parser.add_argument("--scheduler-factor", type=float, help="Plateau scheduler factor.")
    parser.add_argument("--scheduler-patience", type=int, help="Plateau scheduler patience.")
    parser.add_argument("--gradient-clip-norm", type=float, help="Maximum gradient norm.")
    parser.add_argument("--sample-frequency", type=int, default=1, help="Write a sample every N epochs (default: 1).")
    parser.add_argument("--checkpoint-frequency", type=int, help="Write an epoch-NNNN checkpoint every N epochs.")
    parser.add_argument("--checkpoint-path", "--output-path", type=Path, help="Optional legacy checkpoint destination.")
    parser.add_argument("--prompt", help="Generation prompt; defaults to the first training document.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate (default: 100).")
    parser.add_argument("--seed", type=int, help="Optional random seed; required with --deterministic.")
    parser.add_argument("--deterministic", action="store_true", help="Require deterministic PyTorch algorithms; requires --seed.")
    parser.add_argument("--compile", action="store_true", help="Compile model forward execution with torch.compile.")
    parser.add_argument("--observability", action="store_true", help="Enable probe observability declared by the model profile.")
    parser.add_argument("--observability-batches", type=int, default=8, help="Reference batches captured for activation probes (default: 8).")
    return parser


def _configure_reproducibility(*, deterministic: bool, seed: int | None) -> None:
    if deterministic and seed is None:
        raise ValueError("--deterministic requires --seed.")
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
    if deterministic:
        # Must be set before the first CUDA GEMM for deterministic cuBLAS execution.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = not deterministic


def _worker_budget(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError("num-workers must be 'auto' or a non-negative integer.") from error
    if result < 0:
        raise ValueError("num-workers must be non-negative.")
    return result


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")




def _optimization_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {"name": args.optimizer, "learning_rate": args.learning_rate}
    for argument, key in (("weight_decay", "weight_decay"), ("momentum", "momentum")):
        value = getattr(args, argument)
        if value is not None:
            values[key] = value
    return values


def _scheduler_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {"name": args.scheduler}
    option_names = {
        "scheduler_step_timing": "step_timing", "scheduler_t_max": "t_max", "scheduler_eta_min": "eta_min",
        "scheduler_step_size": "step_size", "scheduler_gamma": "gamma", "scheduler_factor": "factor",
        "scheduler_patience": "patience",
    }
    for argument, key in option_names.items():
        value = getattr(args, argument)
        if value is not None:
            values[key] = value
    return values


def _resolved_config(args: argparse.Namespace, manifest: Mapping[str, Any], model: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    resolved = resolve_training_config({"optimization": _optimization_overrides(args), "scheduler": _scheduler_overrides(args)})
    config = {
        "experiment": {"name": args.name, "seed": args.seed, "deterministic": args.deterministic},
        "dataset": {"root": str(args.dataset_root), "name": manifest.get("name"), "version": manifest.get("version"), "manifest": str(args.dataset_root / "manifest.yaml"), "builder": manifest.get("builder"), "document_unit": manifest.get("document_unit", manifest.get("format", {}).get("document_unit"))},
        "loader": {"sequence_length": args.sequence_length, "batch_size": args.batch_size, "num_workers": 0, "shuffle_train": True},
        "model": dict(model),
        "task": {"name": manifest.get("task")},
        **resolved.to_mapping(),
        "training": {"epochs": args.epochs, "device": args.device, "gradient_clip_norm": args.gradient_clip_norm, "compile": args.compile},
        "checkpointing": {"monitor": "validation/loss", "mode": "min", "save_frequency": args.checkpoint_frequency},
        "generation": {"prompt": prompt, "max_new_tokens": args.max_new_tokens, "sample_frequency": args.sample_frequency},
    }
    return config


def _fingerprint(value: Any) -> str:
    """Fingerprint config independently of the run ID injected by ExperimentRun."""
    normalized = dict(cast(Mapping[str, Any], value))
    experiment = normalized.get("experiment")
    if isinstance(experiment, Mapping):
        normalized["experiment"] = {key: item for key, item in experiment.items() if key != "run_id"}
    canonical_json = _future_api("goldfish.data.validation", "canonical_json")
    return hashlib.sha256(canonical_json(normalized)).hexdigest()


def _load_run_config(run: ExperimentRun) -> dict[str, Any]:
    loaded = yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("resume run config.yaml must be a mapping")
    return loaded


def _load_mapping_json(path: Path, description: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{description} not found: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {description}: {path}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{description} must be a mapping")
    return loaded



def _resume_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Reject explicit immutable overrides; saved config remains the source of truth."""
    provided = getattr(args, "_provided", set())
    if "model_profile" in provided:
        raise ValueError("--model-profile cannot be used when resuming; the run's saved model config is authoritative.")
    option_sections = {
        "experiment": {"seed", "deterministic"},
        "training": {"compile"},
        "model": {"model_profile"},
        "loader": {"sequence_length", "batch_size"},
        "optimization": {"optimizer", "learning_rate", "weight_decay", "momentum"},
        "scheduler": {"scheduler", "scheduler_step_timing", "scheduler_t_max", "scheduler_eta_min", "scheduler_step_size", "scheduler_gamma", "scheduler_factor", "scheduler_patience"},
    }
    requested = _resolved_config(args, config["dataset"], config["model"], str(config["generation"]["prompt"]))
    for section, options in option_sections.items():
        if provided.intersection(options) and requested[section] != config.get(section):
            raise ValueError(f"resume {section} configuration is incompatible with the existing run")
    return config


def _restore_experiment_checkpoint(
    trainer: Trainer[Any],
    path: Path,
    *,
    run: ExperimentRun,
    provenance: Mapping[str, Any],
    restore_optimizer_state: bool,
) -> None:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("resume checkpoint has an unsupported format")
    checkpoint_provenance = checkpoint.get("provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise ValueError("resume checkpoint is missing provenance")
    provenance_keys = ("run_id", "dataset_fingerprint", "tokenizer_fingerprint", "normalizer_fingerprint", "model_family", "model_name")
    if restore_optimizer_state:
        provenance_keys = ("config_fingerprint", *provenance_keys)
    for key in provenance_keys:
        if checkpoint_provenance.get(key) != provenance.get(key):
            raise ValueError(f"resume checkpoint provenance mismatch for {key}")
    trainer.model.load_state_dict(checkpoint["model"])
    if restore_optimizer_state:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler_state = checkpoint.get("scheduler")
        if trainer.scheduler is None:
            if scheduler_state is not None:
                raise ValueError("resume checkpoint has a scheduler but this run does not")
        elif scheduler_state is None:
            raise ValueError("resume checkpoint has no scheduler for this run")
        else:
            trainer.scheduler.load_state_dict(scheduler_state)
    trainer.epoch = int(checkpoint["epoch"])
    trainer.global_step = int(checkpoint["global_step"])


def _write_training_plot(run: ExperimentRun) -> Path:
    """Render per-epoch train/validation metric trajectories for a completed run."""
    records: list[Mapping[str, Any]] = []
    for line in (run.path / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if isinstance(record, Mapping):
            records.append(record)
    metric_names = sorted({
        name
        for record in records
        for phase in ("train", "validation")
        if isinstance(record.get(phase), Mapping)
        for name, value in cast(Mapping[str, Any], record[phase]).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    if not records or not metric_names:
        raise ValueError("Cannot plot training curves: metrics journal contains no scalar train/validation metrics.")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(len(metric_names), 1, figsize=(10, 3.2 * len(metric_names)), sharex=True, squeeze=False)
    epochs = [int(record["epoch"]) for record in records]
    for index, metric in enumerate(metric_names):
        axis = axes[index, 0]
        for phase, color in (("train", "C0"), ("validation", "C1")):
            points = [
                (epoch, float(cast(Mapping[str, Any], record[phase])[metric]))
                for epoch, record in zip(epochs, records, strict=True)
                if isinstance(record.get(phase), Mapping) and isinstance(cast(Mapping[str, Any], record[phase]).get(metric), (int, float))
            ]
            if points:
                axis.plot([epoch for epoch, _ in points], [value for _, value in points], color=color, label=phase)
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1, 0].set_xlabel("Epoch")
    figure.suptitle(f"Training curves: {run.run_id}")
    figure.tight_layout()
    path = run.path / "artifacts" / "plots" / "training-curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    pyplot.close(figure)
    return path


def _format_metrics(metrics: Mapping[str, float] | None) -> str:
    if not metrics:
        return "n/a"
    return " | ".join(f"{name} {value:.4f}" for name, value in sorted(metrics.items()))




def _print_run_header(
    *,
    run: ExperimentRun,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    model: torch.nn.Module,
    resume: bool,
    resumed_epoch: int | None = None,
    resumed_step: int | None = None,
) -> None:
    dataset = cast(Mapping[str, Any], config["dataset"])
    loader = cast(Mapping[str, Any], config["loader"])
    model_config = cast(Mapping[str, Any], config["model"])
    optimization = cast(Mapping[str, Any], config["optimization"])
    scheduler = cast(Mapping[str, Any], config["scheduler"])
    runtime = cast(Mapping[str, Any], data["runtime"])
    locking = cast(Mapping[str, Any], data["locking"])
    tokenizer = cast(Mapping[str, Any], data.get("tokenizer", {}))
    experiment = cast(Mapping[str, Any], config["experiment"])
    mode = "resuming" if resume else "new run"
    print(f"Goldfish — {mode}")
    print(f"  Run:        {run.path}")
    print(f"  Dataset:    {dataset['name']} v{dataset['version']} ({dataset['builder']})")
    data_shape = f"seq_len={loader['sequence_length']}, vocab={tokenizer.get('vocab_size', '?')}" if "sequence_length" in loader else f"lookback={runtime.get('lookback', '?')}, features={len(runtime.get('features', []))}, targets={len(runtime.get('targets', []))}"
    model_parameters = cast(Mapping[str, Any], model_config["parameters"])
    model_shape = f"embedding={model_parameters['embedding_dim']}, " if "embedding_dim" in model_parameters else ""
    print(f"  Device:     {config['training']['device']}")
    print(f"  Data:       train={runtime.get('train_samples', '?')}, val={runtime.get('val_samples', '?')}, {data_shape}")
    print(f"  Loader:     train_workers={loader.get('train_workers', loader.get('num_workers', 0))}, val_workers={loader.get('validation_workers', loader.get('num_workers', 0))}, pin_memory={loader.get('pin_memory', False)}, prefetch={loader.get('prefetch_factor')}, persistent={loader.get('persistent_workers', False)}")
    details = ", ".join(f"{key}={value}" for key, value in model_parameters.items())
    print(f"  Model:      {model_config['family']}/{model_config['name']} ({model_shape}{details})")
    print(f"  Compile:    {bool(cast(Mapping[str, Any], config['training']).get('compile', False))}")
    print(f"  Reproduce:  deterministic={experiment.get('deterministic', False)}, seed={experiment.get('seed')}")
    if config["training"]["device"] == "cuda":
        print(f"  CUDA opt:   cudnn_benchmark={torch.backends.cudnn.benchmark}, tf32={torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32}")
    print("  Architecture:")
    for line in str(summary(model, depth=3, verbose=0)).splitlines():
        print(f"    {line}")
    print(f"  Optimize:   {optimization['name']} (lr={optimization['learning_rate']:.3g}, weight_decay={optimization['weight_decay']:.3g})")
    scheduler_description = scheduler['name']
    if scheduler['name'] != 'none':
        scheduler_description += f" (step={scheduler['step_timing']})"
    print(f"  Schedule:   {scheduler_description}")
    print(f"  Monitor:    validation/loss (min)")
    print(f"  Data lock:  {str(locking.get('dataset_fingerprint', ''))[:12]}…")
    if tokenizer.get("fingerprint"):
        print(f"  Tokenizer:  {str(tokenizer['fingerprint'])[:12]}…")
    if resume:
        print(f"  Resume:     epoch={resumed_epoch}, step={resumed_step}; additional_epochs={config['training']['epochs']}")


def _write_sample(run: ExperimentRun, *, epoch: int | None, model: Any, tokenizer: TextTokenizer, config: Mapping[str, Any]) -> str:
    generation = cast(Mapping[str, Any], config["generation"])
    document_unit = cast(Mapping[str, Any], config["dataset"]).get("document_unit")
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    prefix_token_ids = [sep_token_id] if document_unit == "file-pair" and sep_token_id is not None else []
    text = cast(str, generate_text(model, tokenizer, str(generation["prompt"]), max_new_tokens=int(generation["max_new_tokens"]), prefix_token_ids=prefix_token_ids))
    filename = "final.txt" if epoch is None else f"epoch-{epoch + 1:04d}.txt"
    contents = f"run_id: {run.run_id}\nepoch: {'final' if epoch is None else epoch}\nprompt: {generation['prompt']}\n\n{text}\n"
    (run.path / "artifacts" / "samples" / filename).write_text(contents, encoding="utf-8")
    return text


def _loose_resume_config(args: argparse.Namespace, config: dict[str, Any], *, numeric: bool) -> tuple[dict[str, Any], bool]:
    """Apply loose runtime overrides while retaining immutable model and data identity."""
    requested = (
        _numeric_config(
            args,
            config["dataset"],
            {
                "features": [None] * int(cast(Mapping[str, Any], config["model"]["parameters"])["feature_count"]),
                "targets": [None] * int(cast(Mapping[str, Any], config["model"]["parameters"])["target_count"]),
                "horizons": [None] * int(cast(Mapping[str, Any], config["model"]["parameters"])["horizon_count"]),
            },
            config["model"],
        )
        if numeric
        else _resolved_config(args, config["dataset"], config["model"], str(config["generation"]["prompt"]))
    )
    updated = dict(config)
    for section in ("loader", "optimization", "scheduler", "training"):
        updated[section] = requested[section]
    updated["training"] = {**cast(Mapping[str, Any], updated["training"]), "resume_mode": "loose"}
    optimizer_changed = updated["optimization"] != config["optimization"] or updated["scheduler"] != config["scheduler"]
    return updated, optimizer_changed


def _numeric_resume_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Reject explicit numeric resume overrides; saved config remains authoritative."""
    provided = getattr(args, "_provided", set())
    if "model_profile" in provided:
        raise ValueError("--model-profile cannot be used when resuming; the run's saved model config is authoritative.")
    options = {
        "experiment": {"seed", "deterministic"},
        "training": {"compile"},
        "model": {"model_profile"},
        "loader": {"batch_size", "num_workers", "train_workers", "val_workers", "prefetch_factor"},
        "optimization": {"optimizer", "learning_rate", "weight_decay", "momentum"},
        "scheduler": {"scheduler", "scheduler_step_timing", "scheduler_t_max", "scheduler_eta_min", "scheduler_step_size", "scheduler_gamma", "scheduler_factor", "scheduler_patience"},
        "observability": {"observability", "observability_batches"},
    }
    model_parameters = cast(Mapping[str, Any], config["model"]["parameters"])
    runtime = {"features": [None] * int(model_parameters["feature_count"]), "targets": [None] * int(model_parameters["target_count"]), "horizons": [None] * int(model_parameters["horizon_count"])}
    requested = _numeric_config(args, config["dataset"], runtime, config["model"])
    # Resume cannot reconstruct the profile's probe declarations, so carry the
    # saved declarations into the requested block before comparing.
    if args.observability and isinstance(config.get("observability"), Mapping):
        cast(dict[str, Any], requested["observability"])["probes"] = config["observability"].get("probes")
    for section, section_options in options.items():
        if provided.intersection(section_options) and requested[section] != config.get(section):
            raise ValueError(f"resume {section} configuration is incompatible with the existing run")
    return config


def _build_probe_hook(config: Mapping[str, Any], data_module: Any, dataset_lock: Mapping[str, Any], run: Any) -> Any:
    """Assemble the probe hook from the run's persisted observability block.

    The reference provider captures the first reference batches from a fresh
    validation loader iterator; the split fingerprint is read from the dataset
    lock for reproducibility anchors.
    """
    observability = config.get("observability")
    if not isinstance(observability, Mapping):
        return None
    probes_declared = observability.get("probes")
    run_block = {key: value for key, value in observability.items() if key != "probes"}
    resolved = resolve_observability_config(probes_declared, run_block)
    if not resolved.probes:
        return None

    def reference_factory() -> Any:
        assert resolved.reference is not None
        return take_first_batches(iter(data_module.val_dataloader()), resolved.reference.batches)

    split_fingerprint: str | None = None
    splits = dataset_lock.get("splits")
    if isinstance(splits, Mapping):
        val_split = splits.get("val")
        if isinstance(val_split, Mapping):
            fingerprint = val_split.get("fingerprint")
            if isinstance(fingerprint, str):
                split_fingerprint = fingerprint
    return build_probe_hook(
        resolved,
        JsonlRecorder(run.path / "artifacts" / "probes"),
        reference_factory=reference_factory if resolved.reference is not None else None,
        source_paths={probe.name: "profile" for probe in resolved.probes},
        split_fingerprint=split_fingerprint,
    )


def _numeric_config(args: argparse.Namespace, manifest: Mapping[str, Any], runtime: Mapping[str, Any], model: Mapping[str, Any], *, observability_probes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = resolve_training_config({"optimization": _optimization_overrides(args), "scheduler": _scheduler_overrides(args)})
    features = runtime["features"]
    targets = runtime["targets"]
    horizons = runtime["horizons"]
    if not isinstance(features, list) or not isinstance(targets, list) or not isinstance(horizons, list):
        raise ValueError("numeric runtime metadata is missing feature, target, or horizon declarations")
    config = {
        "experiment": {"name": args.name, "seed": args.seed, "deterministic": args.deterministic},
        "dataset": {"root": str(args.dataset_root), "name": manifest.get("name"), "version": manifest.get("version"), "manifest": str(args.dataset_root / "manifest.yaml"), "builder": manifest.get("builder")},
        "loader": {"batch_size": args.batch_size, "num_workers": 0, "train_workers": 0, "validation_workers": 0, "pin_memory": False, "prefetch_factor": None, "persistent_workers": False, "shuffle_train": True},
        "model": dict(model),
        "task": {"name": manifest.get("task"), "loss": "normalized_mse", "metrics": ["mae", "rmse"]},
        **resolved.to_mapping(),
        "training": {"epochs": args.epochs, "device": args.device, "gradient_clip_norm": args.gradient_clip_norm, "compile": args.compile},
        "checkpointing": {"monitor": "validation/loss", "mode": "min", "save_frequency": args.checkpoint_frequency},
    }
    if args.observability:
        observability: dict[str, Any] = {
            "enabled": True,
            "reference": {"split": "val", "batches": args.observability_batches, "selection": "first"},
        }
        if observability_probes:
            observability["probes"] = dict(observability_probes)
        config["observability"] = observability
    return config


def _train_numeric(args: argparse.Namespace, manifest: Mapping[str, Any], device: torch.device, dataset_lock: Mapping[str, Any]) -> int:
    validate_normalizer_lock = _future_api("goldfish.data.validation", "validate_normalizer_lock")
    data_module_class = _future_api("goldfish.data.numeric", "NumericFilesForecastDataModule")
    normalizer_lock = validate_normalizer_lock(args.dataset_root, manifest)
    data_module = data_module_class(args.dataset_root, manifest, batch_size=args.batch_size)
    settings = resolve_loader_settings(device=device, num_workers=_worker_budget(args.num_workers), train_workers=args.train_workers, validation_workers=args.val_workers, prefetch_factor=args.prefetch_factor)
    data_module.configure_loading(train_workers=settings.train_workers, validation_workers=settings.validation_workers, pin_memory=settings.pin_memory, prefetch_factor=settings.prefetch_factor, persistent_workers=settings.persistent_workers)
    if args.resume is None:
        if args.model_profile is None:
            raise ValueError("--model-profile is required for a new run.")
        profile = load_model_profile(args.model_profile)
        model = resolve_model_config(profile, family="forecast", runtime_parameters={"feature_count": len(data_module.runtime_metadata["features"]), "target_count": len(data_module.runtime_metadata["targets"]), "horizon_count": len(data_module.runtime_metadata["horizons"])})
        observability_probes = profile.get("observability")
    else:
        model = {}
        observability_probes = None
    config = _numeric_config(args, manifest, data_module.runtime_metadata, model, observability_probes=observability_probes)
    config_loader = cast(dict[str, Any], config["loader"])
    config_loader.update({"num_workers": max(settings.train_workers, settings.validation_workers), "train_workers": settings.train_workers, "validation_workers": settings.validation_workers, "pin_memory": settings.pin_memory, "prefetch_factor": settings.prefetch_factor, "persistent_workers": settings.persistent_workers})
    cast(dict[str, Any], config["training"])["device"] = str(device)
    data = build_data_provenance(manifest=manifest, dataset_lock=dataset_lock, tokenizer_lock=None, normalizer_lock=normalizer_lock, runtime_metadata=data_module.runtime_metadata, dataset_root=args.dataset_root)
    reset_optimizer_state = False
    if args.resume is None:
        run = ExperimentRun.create(args.runs_dir, name=args.name, config=config, data=data, environment=collect_environment(device=str(device), repository=Path(__file__).parent))
    else:
        run = ExperimentRun(args.resume)
        if not run.path.is_dir():
            raise ValueError(f"resume run directory does not exist: {run.path}")
        saved_config = _load_run_config(run)
        if args.resume_loose:
            config, reset_optimizer_state = _loose_resume_config(args, saved_config, numeric=True)
        else:
            config, reset_optimizer_state = _numeric_resume_config(args, saved_config), False
        stored_data = _load_mapping_json(run.path / "data.json", "resume data")
        if stored_data.get("locking", {}).get("dataset_fingerprint") != data["locking"]["dataset_fingerprint"]:
            raise ValueError("resume dataset fingerprint is incompatible with the existing run")
        if stored_data.get("normalizer", {}).get("fingerprint") != data.get("normalizer", {}).get("fingerprint"):
            raise ValueError("resume normalizer fingerprint is incompatible with the existing run")
        loader = cast(Mapping[str, Any], config["loader"])
        data_module = data_module_class(args.dataset_root, manifest, batch_size=int(loader["batch_size"]))
        saved_train_workers = int(loader.get("train_workers", loader.get("num_workers", 0)))
        saved_validation_workers = int(loader.get("validation_workers", loader.get("num_workers", 0)))
        data_module.configure_loading(train_workers=saved_train_workers, validation_workers=saved_validation_workers, pin_memory=bool(loader.get("pin_memory", device.type == "cuda")), prefetch_factor=cast(int | None, loader.get("prefetch_factor")), persistent_workers=bool(loader.get("persistent_workers", saved_train_workers > 0 or saved_validation_workers > 0)))
    model_config = cast(Mapping[str, Any], config["model"])
    resolved_training = resolve_training_config({"optimization": config["optimization"], "scheduler": config["scheduler"]})
    model = create_model_from_config(model_config)
    if bool(cast(Mapping[str, Any], config["training"]).get("compile", False)):
        model = compile_model(model)
    probe_hook = _build_probe_hook(config, data_module, dataset_lock, run)
    optimizer = create_optimizer(model.parameters(), resolved_training.optimization)
    scheduler = create_scheduler(optimizer, resolved_training.scheduler)
    provenance = {"run_id": run.run_id, "config_fingerprint": _fingerprint(config), "dataset_fingerprint": data["locking"]["dataset_fingerprint"], "tokenizer_fingerprint": None, "normalizer_fingerprint": data["normalizer"]["fingerprint"], "model_family": model_config["family"], "model_name": model_config["name"]}
    checkpoints = CheckpointManager(run.path, monitor="validation/loss", mode="min", save_frequency=cast(Mapping[str, Any], config["checkpointing"])["save_frequency"], provenance=provenance)
    summary_path = run.path / "summary.json"
    summary = _load_mapping_json(summary_path, "run summary")
    if isinstance(summary.get("best"), Mapping):
        checkpoints.best_value = summary["best"].get("value")
        best_epoch = summary["best"].get("epoch")
        checkpoints.best_epoch = int(best_epoch) - 1 if isinstance(best_epoch, int) else None
    started = time.monotonic()

    def on_epoch_end(result: Any) -> None:
        metrics = {"train": result.train, "validation": result.validation}
        run.append_metrics({"epoch": result.epoch + 1, "global_step": result.global_step, "wall_time_seconds": time.monotonic() - started, "learning_rate": result.learning_rates[0], **metrics})
        checkpoints.on_epoch_end(model, optimizer, epoch=result.epoch, global_step=result.global_step, metrics=metrics, scheduler=scheduler)

    trainer = Trainer(model, PointForecastTask(data_module.normalizer, data_module.runtime_metadata["targets"]), optimizer, scheduler=cast(Any, scheduler), scheduler_step_timing=resolved_training.scheduler.step_timing, scheduler_metric=getattr(resolved_training.scheduler, "monitor", None), device=device, gradient_clip_norm=cast(Mapping[str, Any], config["training"])["gradient_clip_norm"], on_epoch_end=on_epoch_end, progress=True, hooks=[probe_hook] if probe_hook is not None else [])
    run.start()
    try:
        resumed_epoch: int | None = None
        resumed_step: int | None = None
        if args.resume is not None:
            _restore_experiment_checkpoint(
                trainer,
                run.path / "checkpoints" / "latest.pt",
                run=run,
                provenance=provenance,
                restore_optimizer_state=not reset_optimizer_state,
            )
            resumed_epoch, resumed_step = trainer.epoch, trainer.global_step
        _print_run_header(run=run, config=config, data=data, model=model, resume=args.resume is not None, resumed_epoch=resumed_epoch, resumed_step=resumed_step)
        result = trainer.fit(data_module.train_dataloader(), val_loader=data_module.val_dataloader(), epochs=args.epochs)
        final_metrics = {"train": result.history[-1].train, "validation": result.history[-1].validation}
        final_path = checkpoints.save_final(model, optimizer, epoch=result.epoch, global_step=result.global_step, metrics=final_metrics, scheduler=scheduler)
        run.complete(last_epoch=result.epoch, global_step=result.global_step, best=checkpoints.best_summary(), final={"checkpoint": str(final_path.relative_to(run.path)), "validation": result.history[-1].validation})
        plot_path = _write_training_plot(run)
        print(f"Training curves: {plot_path}")
        print(f"Experiment run: {run.path}")
        return 0
    except BaseException as error:
        run.fail(error, last_epoch=trainer.epoch, global_step=trainer.global_step)
        raise


def _train(args: argparse.Namespace) -> int:
    if args.deterministic and args.seed is None:
        raise ValueError("--deterministic requires --seed.")
    if args.resume is None:
        _configure_reproducibility(deterministic=args.deterministic, seed=args.seed)
    else:
        run_config = _load_run_config(ExperimentRun(args.resume))
        experiment = cast(Mapping[str, Any], run_config["experiment"])
        _configure_reproducibility(deterministic=bool(experiment.get("deterministic", False)), seed=cast(int | None, experiment.get("seed")))
    validate_dataset_lock = _future_api("goldfish.data.validation", "validate_dataset_lock")
    validate_tokenizer_lock = _future_api("goldfish.data.validation", "validate_tokenizer_lock")
    text_data = importlib.import_module("goldfish.data.text")
    data_module_class = _future_api("goldfish.data.text", "TextFilesLanguageModelDataModule")
    for value, name in ((args.sequence_length, "sequence length"), (args.batch_size, "batch size"), (args.epochs, "epochs"), (args.learning_rate, "learning rate"), (args.sample_frequency, "sample frequency")):
        _positive(value, name)
    if args.gradient_clip_norm is not None:
        _positive(args.gradient_clip_norm, "gradient clip norm")
    if args.checkpoint_frequency is not None:
        _positive(args.checkpoint_frequency, "checkpoint frequency")

    if args.max_new_tokens < 0:
        raise ValueError("max new tokens must be non-negative.")

    device = resolve_device(args.device)
    manifest = _validate_manifest(args.dataset_root)
    dataset_lock = validate_dataset_lock(args.dataset_root, manifest)
    if manifest.get("builder") == "numeric_files_forecast":
        return _train_numeric(args, manifest, device, dataset_lock)
    tokenizer_lock = validate_tokenizer_lock(args.dataset_root, manifest)
    document_unit = manifest["format"]["document_unit"]
    if document_unit == "file-pair":
        data_module_class = getattr(text_data, "FilePairPrefixLanguageModelDataModule")
        task = PrefixLanguageModelTask()
    else:
        task = CausalLanguageModelTask()
    data_module = data_module_class(args.dataset_root, manifest, sequence_length=args.sequence_length, batch_size=args.batch_size)
    prompt = args.prompt or _default_prompt(args.dataset_root, manifest)
    if args.resume is None:
        if args.model_profile is None:
            raise ValueError("--model-profile is required for a new run.")
        model = resolve_model_config(load_model_profile(args.model_profile), family="language", runtime_parameters={"vocab_size": data_module.tokenizer.vocab_size})
    else:
        model = {}
    config = _resolved_config(args, manifest, model, prompt)
    cast(dict[str, Any], config["training"])["device"] = str(device)
    data = build_data_provenance(manifest=manifest, dataset_lock=dataset_lock, tokenizer_lock=tokenizer_lock, runtime_metadata=data_module.runtime_metadata, dataset_root=args.dataset_root)

    reset_optimizer_state = False
    if args.resume is None:
        run = ExperimentRun.create(args.runs_dir, name=args.name, config=config, data=data, environment=collect_environment(device=str(device), repository=Path(__file__).parent))
    else:
        run = ExperimentRun(args.resume)
        if not run.path.is_dir():
            raise ValueError(f"resume run directory does not exist: {run.path}")
        saved_config = _load_run_config(run)
        if args.resume_loose:
            config, reset_optimizer_state = _loose_resume_config(args, saved_config, numeric=False)
        else:
            config = _resume_config(args, saved_config)
        stored_data = json.loads((run.path / "data.json").read_text(encoding="utf-8"))
        if not isinstance(stored_data, Mapping) or stored_data.get("locking", {}).get("dataset_fingerprint") != data["locking"]["dataset_fingerprint"]:
            raise ValueError("resume dataset fingerprint is incompatible with the existing run")
        if stored_data.get("tokenizer", {}).get("fingerprint") != data.get("tokenizer", {}).get("fingerprint"):
            raise ValueError("resume tokenizer fingerprint is incompatible with the existing run")
        # Rebuild using the immutable saved loader settings, not parser defaults.
        loader = cast(Mapping[str, Any], config["loader"])
        data_module = data_module_class(args.dataset_root, manifest, sequence_length=int(loader["sequence_length"]), batch_size=int(loader["batch_size"]))

    model_config = cast(Mapping[str, Any], config["model"])
    optimizer_config = cast(Mapping[str, Any], config["optimization"])
    scheduler_config = cast(Mapping[str, Any], config["scheduler"])
    resolved_training = resolve_training_config({"optimization": optimizer_config, "scheduler": scheduler_config})
    model = create_model_from_config(model_config)
    if bool(cast(Mapping[str, Any], config["training"]).get("compile", False)):
        model = compile_model(model)
    optimizer = create_optimizer(model.parameters(), resolved_training.optimization)
    scheduler = create_scheduler(optimizer, resolved_training.scheduler)
    provenance = {"run_id": run.run_id, "config_fingerprint": _fingerprint(config), "dataset_fingerprint": data["locking"]["dataset_fingerprint"], "tokenizer_fingerprint": data.get("tokenizer", {}).get("fingerprint"), "normalizer_fingerprint": None, "model_family": model_config["family"], "model_name": model_config["name"]}
    checkpoints = CheckpointManager(run.path, monitor="validation/loss", mode="min", save_frequency=cast(Mapping[str, Any], config["checkpointing"])["save_frequency"], provenance=provenance)
    summary_path = run.path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(summary.get("best"), Mapping):
        checkpoints.best_value = summary["best"].get("value")
        best_epoch = summary["best"].get("epoch")
        checkpoints.best_epoch = int(best_epoch) - 1 if isinstance(best_epoch, int) else None

    epoch_total = int(cast(Mapping[str, Any], config["training"])["epochs"])

    def on_epoch_end(result: Any) -> None:
        metrics = {"train": result.train, "validation": result.validation}
        run.append_metrics({"epoch": result.epoch + 1, "global_step": result.global_step, "wall_time_seconds": time.monotonic() - started, "learning_rate": result.learning_rates[0] if len(result.learning_rates) == 1 else list(result.learning_rates), **metrics})
        is_best = checkpoints.on_epoch_end(model, optimizer, epoch=result.epoch, global_step=result.global_step, metrics=metrics, scheduler=scheduler)
        frequency = int(cast(Mapping[str, Any], config["generation"])["sample_frequency"])
        wrote_sample = (result.epoch + 1) % frequency == 0
        if wrote_sample:
            _write_sample(run, epoch=result.epoch, model=model, tokenizer=cast(TextTokenizer, data_module.tokenizer), config=config)
        best = checkpoints.best_summary()
        lr = result.learning_rates[0] if len(result.learning_rates) == 1 else result.learning_rates
        print(f"Epoch {result.epoch + 1}/{epoch_total} | {time.monotonic() - started:.2f}s | step {result.global_step} | lr {lr}")
        print(f"  train      {_format_metrics(result.train)}")
        print(f"  validation {_format_metrics(result.validation)}")
        if best is not None:
            saved = " ✓ saved best.pt" if is_best else ""
            print(f"  best {best['metric']}: {best['value']:.4f} (epoch {best['epoch'] + 1}){saved}")
        notices: list[str] = ["saved latest.pt"]
        if checkpoints.save_frequency is not None and (result.epoch + 1) % checkpoints.save_frequency == 0:
            notices.append(f"saved epoch-{result.epoch + 1:04d}.pt")
        if wrote_sample:
            notices.append(f"sample epoch-{result.epoch + 1:04d}.txt")
        print(f"  artifacts  {', '.join(notices)}")

    trainer = Trainer(model, cast(Task[Any], task), optimizer, scheduler=cast(Any, scheduler), scheduler_step_timing=resolved_training.scheduler.step_timing, scheduler_metric=getattr(resolved_training.scheduler, "monitor", None), device=device, gradient_clip_norm=cast(Mapping[str, Any], config["training"])["gradient_clip_norm"], on_epoch_end=on_epoch_end, progress=True)
    started = time.monotonic()
    run.start()
    try:
        resumed_epoch: int | None = None
        resumed_step: int | None = None
        if args.resume is not None:
            _restore_experiment_checkpoint(
                trainer,
                run.path / "checkpoints" / "latest.pt",
                run=run,
                provenance=provenance,
                restore_optimizer_state=not reset_optimizer_state,
            )
            resumed_epoch, resumed_step = trainer.epoch, trainer.global_step
        _print_run_header(run=run, config=config, data=data, model=model, resume=args.resume is not None, resumed_epoch=resumed_epoch, resumed_step=resumed_step)
        result = trainer.fit(data_module.train_dataloader(), val_loader=data_module.val_dataloader(), epochs=args.epochs)
        final_metrics = {"train": result.history[-1].train, "validation": result.history[-1].validation}
        final_path = checkpoints.save_final(model, optimizer, epoch=result.epoch, global_step=result.global_step, metrics=final_metrics, scheduler=scheduler)
        sample = _write_sample(run, epoch=None, model=model, tokenizer=cast(TextTokenizer, data_module.tokenizer), config=config)
        run.complete(last_epoch=result.epoch, global_step=result.global_step, best=checkpoints.best_summary(), final={"checkpoint": str(final_path.relative_to(run.path)), "validation": result.history[-1].validation})
        print("Validation metrics: " + ", ".join(f"{name}={value:.4f}" for name, value in sorted((result.history[-1].validation or {}).items())))
        print(f"Generated sample: {sample}")
        plot_path = _write_training_plot(run)
        print(f"Training curves: {plot_path}")
        if args.checkpoint_path is not None:
            trainer.save_checkpoint(args.checkpoint_path)
            print(f"Saved checkpoint: {args.checkpoint_path}")
        print(f"Experiment run: {run.path}")
        return 0
    except BaseException as error:
        run.fail(error, last_epoch=trainer.epoch, global_step=trainer.global_step)
        raise


class TrainingInterrupted(KeyboardInterrupt):
    """Signal-triggered interruption that preserves the originating signal number."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(f"training interrupted by signal {signal.Signals(signal_number).name}")
        self.signal_number = signal_number


def _install_signal_handlers() -> None:
    """Translate termination signals into a catchable training interruption."""
    def handler(signal_number: int, _frame: Any) -> None:
        raise TrainingInterrupted(signal_number)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _provided_options(argv: Sequence[str]) -> set[str]:
    result: set[str] = set()
    aliases = {"--lr": "learning_rate", "--output-path": "checkpoint_path"}
    for item in argv:
        if item.startswith("--"):
            result.add(aliases.get(item.split("=", 1)[0], item[2:].split("=", 1)[0].replace("-", "_")))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(actual_argv)
    args._provided = _provided_options(actual_argv)
    _install_signal_handlers()
    try:
        return _train(args)
    except TrainingInterrupted as error:
        print(f"Training interrupted ({signal.Signals(error.signal_number).name}); run state was recorded.", file=sys.stderr)
        return 128 + error.signal_number


if __name__ == "__main__":
    main()
