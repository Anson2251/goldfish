"""Train or strictly resume a character language-model experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from goldfish.config import create_optimizer, create_scheduler, resolve_training_config
from goldfish.core import Task
from goldfish.device import resolve_device
from goldfish.experiments import CHECKPOINT_FORMAT, CheckpointManager, ExperimentRun, build_data_provenance, collect_environment
from goldfish.generation import generate_text
from goldfish.generation.text import TextTokenizer
from goldfish.models import model_registry
from goldfish.tasks import CausalLanguageModelTask, PrefixLanguageModelTask
from goldfish.training import Trainer


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
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="Strictly resume an existing run from latest.pt.")
    parser.add_argument("--sequence-length", type=int, default=64, help="Tokens in each training window (default: 64).")
    parser.add_argument("--batch-size", type=int, default=32, help="Training and validation batch size (default: 32).")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train; on resume these are additional epochs (default: 5).")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Character embedding width (default: 64).")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Recurrent hidden width (default: 128).")
    parser.add_argument("--num-layers", type=int, default=1, help="Recurrent layer count (default: 1).")
    parser.add_argument("--dropout", type=float, default=0.0, help="Recurrent dropout (default: 0).")
    parser.add_argument("--model", choices=model_registry.names("language"), default="gru", help="Language-model architecture (default: gru).")
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
    parser.add_argument("--seed", type=int, default=0, help="Torch random seed (default: 0).")
    return parser


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


def _resolved_config(args: argparse.Namespace, manifest: Mapping[str, Any], vocab_size: int, prompt: str) -> dict[str, Any]:
    resolved = resolve_training_config({"optimization": _optimization_overrides(args), "scheduler": _scheduler_overrides(args)})
    config = {
        "experiment": {"name": args.name, "seed": args.seed},
        "dataset": {"root": str(args.dataset_root), "name": manifest.get("name"), "version": manifest.get("version"), "manifest": str(args.dataset_root / "manifest.yaml"), "builder": manifest.get("builder"), "document_unit": manifest.get("document_unit", manifest.get("format", {}).get("document_unit"))},
        "loader": {"sequence_length": args.sequence_length, "batch_size": args.batch_size, "num_workers": 0, "shuffle_train": True},
        "model": {"family": "language", "name": args.model, "vocab_size": vocab_size, "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim, "num_layers": args.num_layers, "dropout": args.dropout},
        "task": {"name": manifest.get("task")},
        **resolved.to_mapping(),
        "training": {"epochs": args.epochs, "device": args.device, "gradient_clip_norm": args.gradient_clip_norm},
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


def _resume_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Reject explicit immutable overrides; saved config remains the source of truth."""
    provided = getattr(args, "_provided", set())
    option_sections = {
        "model": {"model", "embedding_dim", "hidden_dim", "num_layers", "dropout"},
        "loader": {"sequence_length", "batch_size"},
        "optimization": {"optimizer", "learning_rate", "weight_decay", "momentum"},
        "scheduler": {"scheduler", "scheduler_step_timing", "scheduler_t_max", "scheduler_eta_min", "scheduler_step_size", "scheduler_gamma", "scheduler_factor", "scheduler_patience"},
    }
    requested = _resolved_config(args, config["dataset"], int(config["model"]["vocab_size"]), str(config["generation"]["prompt"]))
    for section, options in option_sections.items():
        if provided.intersection(options) and requested[section] != config.get(section):
            raise ValueError(f"resume {section} configuration is incompatible with the existing run")
    return config


def _restore_experiment_checkpoint(trainer: Trainer[Any], path: Path, *, run: ExperimentRun, provenance: Mapping[str, Any]) -> None:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("resume checkpoint has an unsupported format")
    checkpoint_provenance = checkpoint.get("provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise ValueError("resume checkpoint is missing provenance")
    for key in ("run_id", "config_fingerprint", "dataset_fingerprint", "tokenizer_fingerprint", "model_family", "model_name"):
        if checkpoint_provenance.get(key) != provenance.get(key):
            raise ValueError(f"resume checkpoint provenance mismatch for {key}")
    trainer.model.load_state_dict(checkpoint["model"])
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


def _format_metrics(metrics: Mapping[str, float] | None) -> str:
    if not metrics:
        return "n/a"
    return " | ".join(f"{name} {value:.4f}" for name, value in sorted(metrics.items()))


def _parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


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
    total, trainable = _parameter_count(model)
    mode = "resuming" if resume else "new run"
    print(f"Goldfish — {mode}")
    print(f"  Run:        {run.path}")
    print(f"  Dataset:    {dataset['name']} v{dataset['version']} ({dataset['builder']})")
    print(
        f"  Data:       train={runtime.get('train_samples', '?')}, val={runtime.get('val_samples', '?')}, "
        f"seq_len={loader['sequence_length']}, vocab={tokenizer.get('vocab_size', '?')}"
    )
    print(
        f"  Model:      {model_config['family']}/{model_config['name']} "
        f"(embedding={model_config['embedding_dim']}, hidden={model_config['hidden_dim']}, "
        f"layers={model_config['num_layers']}, dropout={model_config['dropout']})"
    )
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
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


def _train(args: argparse.Namespace) -> int:
    validate_dataset_lock = _future_api("goldfish.data.validation", "validate_dataset_lock")
    validate_tokenizer_lock = _future_api("goldfish.data.validation", "validate_tokenizer_lock")
    text_data = importlib.import_module("goldfish.data.text")
    data_module_class = _future_api("goldfish.data.text", "TextFilesLanguageModelDataModule")
    for value, name in ((args.sequence_length, "sequence length"), (args.batch_size, "batch size"), (args.epochs, "epochs"), (args.embedding_dim, "embedding dimension"), (args.hidden_dim, "hidden dimension"), (args.learning_rate, "learning rate"), (args.num_layers, "num layers"), (args.sample_frequency, "sample frequency")):
        _positive(value, name)
    if args.gradient_clip_norm is not None:
        _positive(args.gradient_clip_norm, "gradient clip norm")
    if args.checkpoint_frequency is not None:
        _positive(args.checkpoint_frequency, "checkpoint frequency")
    if args.dropout < 0:
        raise ValueError("dropout must be non-negative.")
    if args.max_new_tokens < 0:
        raise ValueError("max new tokens must be non-negative.")

    device = resolve_device(args.device)
    manifest = _validate_manifest(args.dataset_root)
    dataset_lock = validate_dataset_lock(args.dataset_root, manifest)
    tokenizer_lock = validate_tokenizer_lock(args.dataset_root, manifest)
    document_unit = manifest["format"]["document_unit"]
    if document_unit == "file-pair":
        data_module_class = getattr(text_data, "FilePairPrefixLanguageModelDataModule")
        task = PrefixLanguageModelTask()
    else:
        task = CausalLanguageModelTask()
    data_module = data_module_class(args.dataset_root, manifest, sequence_length=args.sequence_length, batch_size=args.batch_size)
    prompt = args.prompt or _default_prompt(args.dataset_root, manifest)
    config = _resolved_config(args, manifest, data_module.tokenizer.vocab_size, prompt)
    cast(dict[str, Any], config["training"])["device"] = str(device)
    data = build_data_provenance(manifest=manifest, dataset_lock=dataset_lock, tokenizer_lock=tokenizer_lock, runtime_metadata=data_module.runtime_metadata, dataset_root=args.dataset_root)

    if args.resume is None:
        run = ExperimentRun.create(args.runs_dir, name=args.name, config=config, data=data, environment=collect_environment(device=str(device), repository=Path(__file__).parent))
    else:
        run = ExperimentRun(args.resume)
        if not run.path.is_dir():
            raise ValueError(f"resume run directory does not exist: {run.path}")
        config = _resume_config(args, _load_run_config(run))
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
    torch.manual_seed(int(cast(Mapping[str, Any], config["experiment"])["seed"]))
    model = model_registry.create("language", str(model_config["name"]), vocab_size=int(model_config["vocab_size"]), embedding_dim=int(model_config["embedding_dim"]), hidden_dim=int(model_config["hidden_dim"]), num_layers=int(model_config["num_layers"]), dropout=float(model_config["dropout"]))
    optimizer = create_optimizer(model.parameters(), resolved_training.optimization)
    scheduler = create_scheduler(optimizer, resolved_training.scheduler)
    provenance = {"run_id": run.run_id, "config_fingerprint": _fingerprint(config), "dataset_fingerprint": data["locking"]["dataset_fingerprint"], "tokenizer_fingerprint": data.get("tokenizer", {}).get("fingerprint"), "model_family": model_config["family"], "model_name": model_config["name"]}
    checkpoints = CheckpointManager(run.path, monitor="validation/loss", mode="min", save_frequency=cast(Mapping[str, Any], config["checkpointing"])["save_frequency"], provenance=provenance)
    summary_path = run.path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(summary.get("best"), Mapping):
        checkpoints.best_value = summary["best"].get("value")
        checkpoints.best_epoch = summary["best"].get("epoch")

    epoch_total = int(cast(Mapping[str, Any], config["training"])["epochs"])

    def on_epoch_end(result: Any) -> None:
        metrics = {"train": result.train, "validation": result.validation}
        run.append_metrics({"epoch": result.epoch, "global_step": result.global_step, "wall_time_seconds": time.monotonic() - started, "learning_rate": result.learning_rates[0] if len(result.learning_rates) == 1 else list(result.learning_rates), **metrics})
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
            _restore_experiment_checkpoint(trainer, run.path / "checkpoints" / "latest.pt", run=run, provenance=provenance)
            resumed_epoch, resumed_step = trainer.epoch, trainer.global_step
        _print_run_header(run=run, config=config, data=data, model=model, resume=args.resume is not None, resumed_epoch=resumed_epoch, resumed_step=resumed_step)
        result = trainer.fit(data_module.train_dataloader(), val_loader=data_module.val_dataloader(), epochs=args.epochs)
        final_metrics = {"train": result.history[-1].train, "validation": result.history[-1].validation}
        final_path = checkpoints.save_final(model, optimizer, epoch=result.epoch, global_step=result.global_step, metrics=final_metrics, scheduler=scheduler)
        sample = _write_sample(run, epoch=None, model=model, tokenizer=cast(TextTokenizer, data_module.tokenizer), config=config)
        run.complete(last_epoch=result.epoch, global_step=result.global_step, best=checkpoints.best_summary(), final={"checkpoint": str(final_path.relative_to(run.path)), "validation": result.history[-1].validation})
        print("Validation metrics: " + ", ".join(f"{name}={value:.4f}" for name, value in sorted((result.history[-1].validation or {}).items())))
        print(f"Generated sample: {sample}")
        if args.checkpoint_path is not None:
            trainer.save_checkpoint(args.checkpoint_path)
            print(f"Saved checkpoint: {args.checkpoint_path}")
        print(f"Experiment run: {run.path}")
        return 0
    except BaseException as error:
        run.fail(error, last_epoch=trainer.epoch, global_step=trainer.global_step)
        raise


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
    return _train(args)


if __name__ == "__main__":
    main()
