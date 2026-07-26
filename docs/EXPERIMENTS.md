# Goldfish Experiment Management Specification

This document defines Goldfish's reproducible experiment lifecycle. It adopts the discoverability of YOLO-style `runs/exp<N>-<name>/` directories while treating a run as an auditable bundle of code, configuration, data identity, metrics, checkpoints, and task artifacts.

## Goals

An experiment run must make it possible to answer:

- Which model, task, optimizer, scheduler, and training settings produced this result?
- Which exact dataset, split, tokenizer, and preprocessing artifacts were used?
- Can training safely resume from this checkpoint?
- Is a comparison between two runs fair, or did data/task semantics change?
- Which checkpoint was best, according to which metric and direction?

## Run directories

New runs are created under a configurable base directory, defaulting to `runs/`:

```text
runs/
├── exp1-alphabet-gru/
├── exp2-alphabet-lstm/
└── exp3-shakespeare-gru/
```

The generated identifier has the form:

```text
exp<N>-<sanitized-name>
```

- `N` is the next positive integer after existing `exp<N>-*` directories.
- `name` is supplied by the user or generated from dataset/task/model names.
- Existing run directories must never be overwritten by a new run.
- If no name is supplied, Goldfish may use `exp<N>`.

A run directory is created before training begins. Its immutable provenance artifacts are written before the first optimization step.

## Canonical run layout

```text
runs/<run-id>/
├── config.yaml
├── data.json
├── environment.json
├── metrics.jsonl
├── summary.json
├── run.log
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   ├── final.pt
│   └── epoch-0010.pt             # Optional periodic checkpoint
└── artifacts/
    └── samples/
        ├── epoch-0001.txt
        ├── epoch-0010.txt
        └── final.txt
```

| Artifact | Purpose |
|---|---|
| `config.yaml` | Fully resolved experiment configuration. |
| `data.json` | Verified dataset/tokenizer/provenance snapshot. |
| `environment.json` | Runtime, package, device, and source-control information. |
| `metrics.jsonl` | One append-only record per completed epoch. |
| `summary.json` | Final status, best metric/checkpoint, and summary metrics. |
| `run.log` | Human-readable lifecycle and warning/error log. |
| `checkpoints/` | Resume and model-selection artifacts. |
| `artifacts/samples/` | Task-specific qualitative artifacts, initially generated text. |

`runs/` is runtime output and should be ignored by source control unless a user intentionally archives a run elsewhere.

## Resolved configuration

`config.yaml` is the final configuration actually consumed by the run. It contains defaults, dataset-derived values, config-file values, and explicit CLI overrides after resolution.

It must not omit a value merely because it used a default.

```yaml
experiment:
  run_id: exp1-alphabet-gru
  name: alphabet-gru
  seed: 42

source:
  # Optional audit record only. It is not used to resume or reproduce a run.
  command: "uv run python main.py train data/alphabet --model gru --epochs 40"

dataset:
  root: data/alphabet
  name: alphabet
  version: "1.0"
  manifest: data/alphabet/manifest.yaml
  builder: text_files_lm

loader:
  sequence_length: 13
  batch_size: 32
  num_workers: 0
  shuffle_train: true

model:
  family: language
  name: gru
  vocab_size: 28                 # Derived from verified tokenizer metadata
  embedding_dim: 64
  hidden_dim: 128
  num_layers: 1
  dropout: 0.0

task:
  name: causal_language_model

optimization:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1.0e-8

scheduler:
  name: none
  step_timing: null

training:
  epochs: 40
  device: auto
  amp: false
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  deterministic: false

checkpointing:
  monitor: validation/loss
  mode: min
  save_frequency: 10
  save_latest: true
  save_best: true

generation:
  prompt: cdefg
  max_new_tokens: 100
  temperature: null
  top_k: null
  sample_frequency: 10
```

### Configuration ownership

| Section | Required records |
|---|---|
| `dataset` | Dataset root, manifest, dataset identity/version/builder. |
| `loader` | Sequence/window length, batch size, workers, shuffle behavior. |
| `model` | Registry family/name and every architecture parameter. |
| `task` | Task/loss name and task-specific parameters. |
| `optimization` | Optimizer name and every optimizer parameter. |
| `scheduler` | Scheduler name, every parameter, and stepping timing. |
| `training` | Epochs, device policy, AMP, clipping, accumulation, seed/determinism. |
| `checkpointing` | Monitor metric, direction, and save policy. |
| task artifacts | Generation, forecast, retrieval, or other inference settings. |

The model config must include dataset-derived dimensions such as `vocab_size` after resolution. This ensures a checkpoint can be checked for compatibility without reconstructing implicit values.

### Optimization configuration

`optimization` defines the optimizer and every non-default optimizer parameter. The resolved config must never rely on an implicit PyTorch default that is absent from the recorded artifact.

```yaml
optimization:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1.0e-8
  # Optimizer-specific fields may be present only when applicable:
  momentum: null
  nesterov: false
```

Supported initial optimizer definitions:

| `optimization.name` | Required fields | Optional fields | Notes |
|---|---|---|---|
| `adamw` | `learning_rate`, `weight_decay`, `betas`, `eps` | `amsgrad`, `maximize`, `foreach`, `fused` | Default Goldfish optimizer for neural sequence models. |
| `adam` | `learning_rate`, `weight_decay`, `betas`, `eps` | `amsgrad`, `maximize`, `foreach`, `fused` | Same parameter meaning as PyTorch `Adam`. |
| `sgd` | `learning_rate`, `weight_decay`, `momentum` | `dampening`, `nesterov`, `maximize`, `foreach`, `fused` | Useful for explicit experiments; not the first LM default. |

Rules:

- `learning_rate` is the base optimizer learning rate before a scheduler changes it.
- `weight_decay` is an optimizer parameter; it is not a scheduler setting.
- `betas`, `eps`, and `momentum` must be recorded at their resolved values even if they are framework defaults.
- Multiple parameter groups are deferred from v1. When added, each group must record its selection rule and full effective optimizer settings.
- A new optimizer gets a named config schema and factory entry; it must not accept an unstructured bag of arbitrary kwargs.

### Scheduler configuration

A scheduler is optional. It controls the optimizer learning rate after optimizer construction.

```yaml
scheduler:
  name: cosine
  step_timing: epoch
  t_max: 40
  eta_min: 1.0e-6
```

`step_timing` is required whenever `scheduler.name` is not `none`:

| Timing | Meaning |
|---|---|
| `batch` | Call `scheduler.step()` after each successful optimizer update. Required for schedules defined in optimizer steps. |
| `epoch` | Call `scheduler.step()` after every completed training epoch. |
| `validation` | Call `scheduler.step(metric)` after validation. Used by metric-driven schedulers. |

Initial scheduler schemas:

| `scheduler.name` | `step_timing` | Required fields | Optional fields |
|---|---|---|---|
| `none` | `null` | none | none |
| `cosine` | `epoch` or `batch` | `t_max`, `eta_min` | `last_epoch` |
| `step` | `epoch` | `step_size`, `gamma` | `last_epoch` |
| `exponential` | `epoch` | `gamma` | `last_epoch` |
| `plateau` | `validation` | `mode`, `factor`, `patience` | `threshold`, `threshold_mode`, `cooldown`, `min_lr`, `eps` |

Examples:

```yaml
# No scheduler: the optimizer uses a constant learning rate.
scheduler:
  name: none
  step_timing: null

# Decay once at the end of each epoch.
scheduler:
  name: step
  step_timing: epoch
  step_size: 10
  gamma: 0.1

# Metric-driven decay after validation; monitor is defined by scheduler, not checkpointing.
scheduler:
  name: plateau
  step_timing: validation
  monitor: validation/loss
  mode: min
  factor: 0.5
  patience: 3
  threshold: 0.0001
  threshold_mode: rel
  cooldown: 0
  min_lr: 0.0
  eps: 1.0e-8
```

`checkpointing.monitor` and `scheduler.monitor` are independent:

- `checkpointing.monitor` chooses which model state becomes `best.pt`.
- `scheduler.monitor` supplies the metric to a metric-driven scheduler such as `plateau`.

They may use the same metric, but this must be stated independently in the resolved config.

### Why record the launch command?

`source.command` is a convenience audit record, not a configuration mechanism and not a resume input.

It helps answer practical questions such as:

- Which CLI overrides were typed by the user?
- Was a run launched from a shell, a script, a task runner, or an editor action?
- Which dataset path and run-name arguments were used before resolution?
- How can a human quickly attempt a similar run?

It is insufficient for reproducibility because it can depend on shell state, relative paths, environment variables, changing defaults, and code revisions. Therefore:

```text
resolved config.yaml + data.json + environment.json
    = authoritative reproducibility record

source.command
    = optional human-facing audit trail
```

The command must be stored as an argument vector or safely quoted string without credentials, secret values, or environment-variable contents. If the run is launched programmatically and no command is available, record `null` rather than inventing one.

## Data provenance

`data.json` records the verified data identity used by the run. It is written only after manifest, dataset lock, and preprocessing artifact locks have passed validation.

Example for a text dataset:

```json
{
  "dataset": {
    "root": "data/alphabet",
    "manifest": "data/alphabet/manifest.yaml",
    "name": "alphabet",
    "version": "1.0",
    "manifest_version": "1.0",
    "modality": "text",
    "builder": "text_files_lm"
  },
  "locking": {
    "dataset_lock": "data/alphabet/dataset-lock.json",
    "dataset_fingerprint": "<sha256>",
    "split_fingerprints": {
      "train": "<sha256>",
      "val": "<sha256>",
      "test": "<sha256>"
    }
  },
  "tokenizer": {
    "artifact": "data/alphabet/tokenizer/tokenizer.json",
    "lock": "data/alphabet/tokenizer/tokenizer-lock.json",
    "fingerprint": "<sha256>",
    "artifact_sha256": "<sha256>",
    "name": "character",
    "vocab_size": 28,
    "pad_token_id": 0,
    "eos_token_id": 1
  },
  "runtime": {
    "sequence_length": 13,
    "train_samples": 6,
    "val_samples": 2,
    "test_samples": 2
  }
}
```

A run must record at least:

- dataset name, dataset version, manifest path, manifest schema version, modality, and builder;
- dataset lock overall fingerprint;
- selected split fingerprints;
- preprocessing artifacts and fingerprints;
- derived runtime data settings and sample counts.

For numeric datasets, preprocessing records include the frozen normalizer/feature-schema artifact and its lock. For multimodal datasets, data provenance also records alignment configuration and any alignment index fingerprint.

## Environment provenance

`environment.json` captures runtime information that can affect results:

```json
{
  "goldfish_version": "0.1.0",
  "python": "3.13.0",
  "platform": "linux",
  "torch": "2.6.0",
  "cuda_available": true,
  "cuda_version": "12.6",
  "device": "cuda:0",
  "gpu_name": "...",
  "git_commit": "<commit-or-null>",
  "git_dirty": false
}
```

Git metadata is best effort. Missing Git information must be represented as `null`; it must not prevent training.

Future environment records may include dependency lock fingerprints, CPU information, distributed topology, and CUDA/cuDNN settings.

## Metrics journal

`metrics.jsonl` is append-only. Each line describes one completed epoch and must remain valid JSON independently.

```json
{"epoch":0,"global_step":6,"wall_time_seconds":0.48,"learning_rate":0.001,"train":{"loss":3.22,"perplexity":25.1},"validation":{"loss":3.18,"perplexity":24.0}}
{"epoch":1,"global_step":12,"wall_time_seconds":0.92,"learning_rate":0.001,"train":{"loss":2.80,"perplexity":16.4},"validation":{"loss":2.73,"perplexity":15.3}}
```

Every record contains:

- epoch number;
- global optimization step;
- elapsed wall-clock time;
- effective learning rate or learning rates;
- train metrics;
- validation metrics when validation ran.

Optional records may include:

- throughput (`tokens_per_second`, `samples_per_second`);
- memory use;
- gradient norm;
- scheduler state;
- model diagnostics such as MoE routing entropy.

JSONL is preferred over CSV because tasks and models may emit different nested metric structures, and it supports safe append during resume.

## Checkpoint format

All training checkpoints use an explicit versioned format:

```python
{
    "format": "goldfish-checkpoint-v1",

    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "amp_scaler": scaler.state_dict() if scaler else None,

    "epoch": epoch,
    "global_step": global_step,
    "metrics": {
        "train": {"loss": ...},
        "validation": {"loss": ...}
    },

    "provenance": {
        "run_id": "exp1-alphabet-gru",
        "config_fingerprint": "<sha256>",
        "dataset_fingerprint": "<sha256>",
        "tokenizer_fingerprint": "<sha256>",
        "model_family": "language",
        "model_name": "gru"
    },

    "rng_state": {
        "python": "...",
        "torch": "...",
        "cuda": "...",
        "numpy": "..."
    }
}
```

The exact serialization of RNG state is implementation-defined, but it must be sufficient to make a supported resume deterministic when deterministic mode is enabled.

### Checkpoint names

| File | Meaning |
|---|---|
| `latest.pt` | State after the most recently completed epoch; used for normal resume. |
| `best.pt` | Checkpoint with the best configured monitor metric. |
| `final.pt` | State after the final requested epoch or controlled early stop. |
| `epoch-XXXX.pt` | Optional periodic checkpoint according to `save_frequency`. |

`latest.pt`, `best.pt`, and `final.pt` may refer to the same underlying model state but are retained under distinct names for clear lifecycle semantics.

## Best-model selection

Best-model selection is configuration-driven; the checkpoint manager must not assume classification accuracy.

```yaml
checkpointing:
  monitor: validation/loss
  mode: min
```

Supported modes:

| Mode | Better value |
|---|---|
| `min` | Lower metric value is better. |
| `max` | Higher metric value is better. |

Examples:

| Task | Recommended monitor | Mode |
|---|---|---|
| Causal language modelling | `validation/loss` | `min` |
| Point forecasting | `validation/rmse` | `min` |
| Direction classification | `validation/f1` | `max` |
| Ranking/finance | `validation/information_coefficient` | `max` |

The checkpoint manager must fail clearly if the configured metric is absent for an epoch in which best-model selection is attempted.

## Task artifacts

Task artifacts are qualitative or task-specific outputs recorded independently of scalar metrics.

For causal language modelling, Goldfish writes deterministic generated samples using the configured prompt and decoding settings:

```text
artifacts/samples/
├── epoch-0001.txt
├── epoch-0010.txt
└── final.txt
```

Each text sample includes enough metadata to interpret it:

```text
run_id: exp1-alphabet-gru
epoch: 10
prompt: cdefg
temperature: null
top_k: null

cdefghijklmnopqrstuvwxyz...
```

Future tasks may write:

- timestamped numeric forecast tables;
- prediction/target plots;
- calibration reports;
- retrieval examples;
- MoE routing diagnostics.

## Summary

`summary.json` is written or updated on controlled completion, failure, and resume. It contains a compact run status:

```json
{
  "run_id": "exp1-alphabet-gru",
  "status": "completed",
  "started_at": "2026-07-26T12:00:00Z",
  "finished_at": "2026-07-26T12:01:00Z",
  "last_epoch": 39,
  "global_step": 240,
  "best": {
    "checkpoint": "checkpoints/best.pt",
    "epoch": 31,
    "metric": "validation/loss",
    "mode": "min",
    "value": 1.12
  },
  "final": {
    "checkpoint": "checkpoints/final.pt",
    "validation": {"loss": 1.15, "perplexity": 3.16}
  }
}
```

Possible statuses include:

```text
created
running
completed
failed
interrupted
```

A failure summary should include an error type/message and retain already-written metrics/checkpoints.

## New runs

A new run follows this lifecycle:

```text
validate dataset manifest and locks
-> resolve configuration
-> create run directory
-> write config/data/environment artifacts
-> create model, optimizer, scheduler, task, trainer
-> train and append metrics
-> save latest/best/periodic checkpoints and task artifacts
-> save final checkpoint
-> write completed summary
```

A run must not begin optimization if data lock verification fails.

## Strict resume

Resume continues the same run from `checkpoints/latest.pt`:

```sh
uv run python main.py train data/alphabet \
  --resume runs/exp1-alphabet-gru
```

Before loading optimizer state, Goldfish must verify compatibility:

- checkpoint format is supported;
- run `config.yaml` matches the requested immutable configuration;
- dataset fingerprint matches;
- tokenizer/preprocessing fingerprint matches;
- model family/name and architecture configuration match;
- optimizer and scheduler configurations are compatible;
- checkpoint provenance points to the same run ID.

A normal resume may increase the total epoch limit, but it must not silently change model architecture, tokenizer, data identity, task semantics, or optimizer/scheduler identity.

When resume succeeds:

- `metrics.jsonl` is appended rather than replaced;
- `latest.pt` is overwritten only after a later completed epoch;
- run ID, data provenance, and resolved immutable config remain unchanged;
- `summary.json` returns to `running` before the resumed loop begins.

Initial implementation should reject all mismatches. Any future escape hatch must be explicit and recorded, for example:

```text
--resume-allow-data-mismatch
```

Such a mode is intentionally out of scope for v1.

## Forking experiments

Forking creates a new run that records a parent run while preserving a distinct artifact directory:

```sh
uv run python main.py train data/alphabet \
  --fork runs/exp1-alphabet-gru \
  --name alphabet-gru-lr-3e-4 \
  --lr 0.0003
```

The new run records:

```yaml
experiment:
  parent_run: runs/exp1-alphabet-gru
  parent_checkpoint: checkpoints/best.pt
  fork_mode: compatible_weights
```

Two fork modes are anticipated:

| Mode | Meaning |
|---|---|
| `compatible_weights` | Creates a new run and initializes from a strictly compatible parent checkpoint. |
| `fresh_model` | Creates a new run that inherits data/config provenance but initializes model weights from scratch. |

A GRU checkpoint cannot initialize an LSTM with `compatible_weights`. Goldfish must fail clearly rather than partially or silently loading unrelated layers. `fresh_model` is the appropriate mode for architecture comparisons.

Forking is deferred until new-run and strict-resume behavior are stable.

## First implementation scope

The first experiment-management implementation includes:

1. YOLO-style automatic `runs/exp<N>-<name>` directory creation;
2. resolved config, data provenance, and environment artifacts;
3. append-only epoch metrics journal;
4. `latest.pt`, metric-configured `best.pt`, and `final.pt`;
5. generated text artifacts at a configured frequency;
6. strict resume with data/tokenizer/model/config checks.

The following are deferred:

- fork implementation;
- distributed training support;
- TensorBoard/W&B integration;
- plots/dashboard generation;
- self-contained deployment export bundles;
- unsafe partial checkpoint loading.
