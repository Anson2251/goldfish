"""Auditable filesystem-backed experiment run management."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml


CHECKPOINT_FORMAT = "goldfish-checkpoint-v1"
_RUN_PATTERN = re.compile(r"^exp([1-9][0-9]*)-.*$")


def sanitize_run_name(name: str) -> str:
    """Convert a user-facing run name into a stable, path-safe slug."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "run"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _mapping(value: Mapping[str, Any] | dict[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


class ExperimentRun:
    """Owns a single canonical Goldfish run directory and its lifecycle artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id = path.name

    @classmethod
    def create(
        cls,
        base_dir: str | Path = "runs",
        *,
        name: str | None = None,
        config: Mapping[str, Any],
        data: Mapping[str, Any],
        environment: Mapping[str, Any] | None = None,
    ) -> "ExperimentRun":
        """Allocate a new run directory without ever reusing an existing one."""
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        suffix = f"-{sanitize_run_name(name)}" if name else ""
        number = _next_run_number(base)
        while True:
            run_id = f"exp{number}{suffix}"
            path = base / run_id
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1

        try:
            (path / "checkpoints").mkdir()
            (path / "artifacts" / "samples").mkdir(parents=True)
            config_snapshot = dict(config)
            config_snapshot.setdefault("experiment", {})
            if not isinstance(config_snapshot["experiment"], Mapping):
                raise ValueError("config.experiment must be a mapping when supplied")
            config_snapshot["experiment"] = {**config_snapshot["experiment"], "run_id": run_id}
            (path / "config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
            _json_write(path / "data.json", _mapping(data, "data"))
            _json_write(path / "environment.json", _mapping(environment or collect_environment(), "environment"))
            (path / "metrics.jsonl").touch(exist_ok=False)
            _json_write(path / "summary.json", {"run_id": run_id, "status": "created", "started_at": None, "finished_at": None})
            (path / "run.log").write_text(f"{_utc_now()} created\n", encoding="utf-8")
        except Exception:
            # The directory is intentionally retained for forensic inspection; it is never reused.
            raise
        return cls(path)

    def start(self) -> None:
        self._update_summary(status="running", started_at=_utc_now(), finished_at=None)
        self._log("running")

    def append_metrics(self, record: Mapping[str, Any]) -> None:
        """Append one independently valid JSON epoch record to the metrics journal."""
        if not isinstance(record, Mapping):
            raise TypeError("metrics record must be a mapping")
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def complete(self, *, last_epoch: int, global_step: int, best: Mapping[str, Any] | None = None, final: Mapping[str, Any] | None = None) -> None:
        """Record completed lifecycle metadata with one-based user-facing epochs."""
        changes: dict[str, Any] = {"status": "completed", "finished_at": _utc_now(), "last_epoch": last_epoch + 1, "global_step": global_step}
        if best is not None:
            changes["best"] = dict(best)
        if final is not None:
            changes["final"] = dict(final)
        self._update_summary(**changes)
        self._log("completed")

    def fail(self, error: BaseException, *, last_epoch: int | None = None, global_step: int | None = None) -> None:
        changes: dict[str, Any] = {"status": "failed", "finished_at": _utc_now(), "error": {"type": type(error).__name__, "message": str(error)}}
        if last_epoch is not None:
            changes["last_epoch"] = max(0, last_epoch + 1)
        if global_step is not None:
            changes["global_step"] = global_step
        self._update_summary(**changes)
        self._log(f"failed: {type(error).__name__}: {error}")

    def _update_summary(self, **changes: Any) -> None:
        summary_path = self.path / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(changes)
        _json_write(summary_path, summary)

    def _log(self, message: str) -> None:
        with (self.path / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} {message}\n")


def _next_run_number(base_dir: Path) -> int:
    numbers = []
    for child in base_dir.iterdir():
        if child.is_dir() and (match := _RUN_PATTERN.match(child.name)):
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def build_data_provenance(
    *,
    manifest: Mapping[str, Any],
    dataset_lock: Mapping[str, Any],
    tokenizer_lock: Mapping[str, Any] | None,
    runtime_metadata: Mapping[str, Any],
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
    normalizer_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance snapshot from already-validated lock and runtime mappings."""
    manifest, dataset_lock, runtime = (_mapping(manifest, "manifest"), _mapping(dataset_lock, "dataset_lock"), _mapping(runtime_metadata, "runtime_metadata"))
    splits = _mapping(dataset_lock.get("splits", {}), "dataset_lock.splits")
    split_fingerprints = {name: split["fingerprint"] for name, value in splits.items() if isinstance(value, Mapping) and isinstance((split := dict(value)).get("fingerprint"), str)}
    dataset = {key: manifest.get(key) for key in ("name", "version", "modality", "builder")}
    dataset["root"] = str(dataset_root)
    dataset["manifest"] = str(manifest_path or Path(dataset_root) / "manifest.yaml")
    dataset["manifest_version"] = manifest.get("version")
    result: dict[str, Any] = {"dataset": dataset, "locking": {"dataset_lock": manifest.get("locking", {}).get("dataset_lock") if isinstance(manifest.get("locking"), Mapping) else None, "dataset_fingerprint": dataset_lock.get("fingerprint"), "split_fingerprints": split_fingerprints}, "runtime": _runtime_provenance(runtime)}
    if tokenizer_lock is not None:
        token_lock = _mapping(tokenizer_lock, "tokenizer_lock")
        tokenizer = _mapping(token_lock.get("tokenizer", {}), "tokenizer_lock.tokenizer")
        ids = _mapping(tokenizer.get("special_token_ids", {}), "tokenizer_lock.tokenizer.special_token_ids")
        result["tokenizer"] = {"artifact": tokenizer.get("path"), "lock": manifest.get("tokenizer", {}).get("lock") if isinstance(manifest.get("tokenizer"), Mapping) else None, "fingerprint": token_lock.get("fingerprint"), "artifact_sha256": tokenizer.get("sha256"), "name": tokenizer.get("name"), "vocab_size": tokenizer.get("vocab_size"), "pad_token_id": ids.get("pad"), "eos_token_id": ids.get("eos")}
    if normalizer_lock is not None:
        lock = _mapping(normalizer_lock, "normalizer_lock")
        normalizer = _mapping(lock.get("normalizer", {}), "normalizer_lock.normalizer")
        result["normalizer"] = {"artifact": normalizer.get("path"), "lock": manifest.get("normalization", {}).get("lock") if isinstance(manifest.get("normalization"), Mapping) else None, "fingerprint": lock.get("fingerprint"), "artifact_sha256": normalizer.get("sha256"), "name": normalizer.get("name"), "features": normalizer.get("features")}
    return result


def _runtime_provenance(runtime: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(runtime)
    counts = result.pop("sample_counts", None)
    if isinstance(counts, Mapping):
        for split in ("train", "val", "test"):
            if split in counts:
                result[f"{split}_samples"] = counts[split]
    return result


def collect_environment(*, device: str | None = None, repository: str | Path | None = None) -> dict[str, Any]:
    """Collect runtime provenance; unavailable source-control details remain ``None``."""
    cuda_available = torch.cuda.is_available()
    resolved_device = device or ("cuda:0" if cuda_available else "cpu")
    git_commit, git_dirty = _git_info(Path(repository) if repository is not None else Path.cwd())
    return {
        "goldfish_version": "0.1.0",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device": resolved_device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _git_info(repository: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


class CheckpointManager:
    """Save versioned latest/best/final checkpoints from generic stateful objects."""

    def __init__(self, run_path: str | Path, *, monitor: str, mode: str, save_frequency: int | None = None, provenance: Mapping[str, Any] | None = None) -> None:
        if not monitor:
            raise ValueError("checkpoint monitor must be non-empty")
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be 'min' or 'max'")
        if save_frequency is not None and save_frequency <= 0:
            raise ValueError("save_frequency must be positive")
        root = Path(run_path)
        self.directory = root if root.name == "checkpoints" else root / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor, self.mode, self.save_frequency, self.provenance = monitor, mode, save_frequency, dict(provenance or {})
        self.best_value: float | None = None
        self.best_epoch: int | None = None

    def on_epoch_end(self, model: Any, optimizer: Any, *, epoch: int, global_step: int, metrics: Mapping[str, Any], scheduler: Any | None = None, scaler: Any | None = None) -> bool:
        """Save latest, evaluate the configured monitor, and conditionally save best/periodic."""
        metric = _metric_value(metrics, self.monitor)
        payload = self._payload(model, optimizer, scheduler, scaler, epoch, global_step, metrics)
        self._save("latest.pt", payload)
        is_best = self.best_value is None or (metric < self.best_value if self.mode == "min" else metric > self.best_value)
        if is_best:
            self.best_value, self.best_epoch = metric, epoch
            self._save("best.pt", payload)
        if self.save_frequency is not None and (epoch + 1) % self.save_frequency == 0:
            self._save(f"epoch-{epoch + 1:04d}.pt", payload)
        return is_best

    def save_final(self, model: Any, optimizer: Any, *, epoch: int, global_step: int, metrics: Mapping[str, Any], scheduler: Any | None = None, scaler: Any | None = None) -> Path:
        path = self.directory / "final.pt"
        self._save(path.name, self._payload(model, optimizer, scheduler, scaler, epoch, global_step, metrics))
        return path

    def best_summary(self) -> dict[str, Any] | None:
        if self.best_value is None or self.best_epoch is None:
            return None
        return {"checkpoint": "checkpoints/best.pt", "epoch": self.best_epoch + 1, "metric": self.monitor, "mode": self.mode, "value": self.best_value}

    def _payload(self, model: Any, optimizer: Any, scheduler: Any | None, scaler: Any | None, epoch: int, global_step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None, "amp_scaler": scaler.state_dict() if scaler is not None else None, "epoch": epoch, "global_step": global_step, "metrics": dict(metrics), "provenance": self.provenance}

    def _save(self, filename: str, payload: Mapping[str, Any]) -> None:
        target = self.directory / filename
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f".{filename}.", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(dict(payload), temporary_path)
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)


def _metric_value(metrics: Mapping[str, Any], path: str) -> float:
    current: Any = metrics
    for part in path.split("/"):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"configured checkpoint monitor {path!r} is absent from epoch metrics")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"configured checkpoint monitor {path!r} must be numeric")
    return float(current)
