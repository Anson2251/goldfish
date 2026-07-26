# Goldfish Dataset Layout and Manifest Specification

This document defines the on-disk contract for Goldfish datasets. It applies to text datasets now and is designed to extend to numeric and multimodal sequential datasets later.

## Goals

The dataset layout must make the following explicit and reproducible:

- which files belong to each split;
- document or shard boundaries;
- data modality and task intent;
- tokenizer, feature, target, and time semantics;
- provenance and optional integrity information.

> `manifest.yaml` is the authoritative definition of a dataset. Directory discovery is a convenience for authors, not the runtime source of truth.

The lock files are derived, machine-maintained snapshots. They must be verified during training and updated only by an explicit data-preparation/locking operation.

## Canonical layout

Every dataset lives under `data/<dataset-name>/`:

```text
data/
└── <dataset-name>/
    ├── manifest.yaml
    ├── dataset-lock.json
    ├── train/
    │   ├── 01-<name>.<extension>
    │   └── 02-<name>.<extension>
    ├── val/
    │   └── 01-<name>.<extension>
    └── test/
        └── 01-<name>.<extension>
```

`train`, `val`, and `test` are split directories. A dataset may omit `test` during early development, but training requires `train` and `val`.

The manifest explicitly lists the files used by each split. Files that exist on disk but are absent from the manifest are not used.

A text dataset additionally contains its frozen tokenizer and a tokenizer lock:

```text
data/<dataset-name>/
├── manifest.yaml
├── dataset-lock.json
├── train/
├── val/
├── test/
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer-lock.json
```

`tokenizer.json` is the actual token-to-ID mapping. `tokenizer-lock.json` binds that mapping to the locked training split and tokenizer configuration.

## Why multiple files

Files represent meaningful **documents** or **shards**, not arbitrary fragments. This provides:

- explicit document boundaries;
- stable, auditable train/validation/test membership;
- a place for per-file provenance, timestamps, groups, or checksums;
- incremental updates and tokenization caches;
- future group-aware splitting, temporal alignment, and parallel preprocessing.

Avoid excessive small-file layouts. A file should normally represent a meaningful unit such as an article, chapter, session, day, instrument partition, or a reasonably sized shard.

## Manifest format

All manifests use YAML and begin with dataset identity and builder information:

```yaml
name: alphabet
version: "1.0"
modality: text
builder: text_files_lm
task: causal_language_model
```

`name` identifies a dataset instance. `builder` identifies the framework implementation used to load it. For example, `alphabet` and `shakespeare` may both use the `text_files_lm` builder, while a market-bars dataset may use `numeric_files_forecast`.

### Split files

Paths are relative to the directory containing `manifest.yaml`.

The simplest form is an ordered list of paths:

```yaml
splits:
  train:
    files:
      - train/01-alphabet.txt
      - train/02-alphabet.txt
  val:
    files:
      - val/01-alphabet.txt
  test:
    files:
      - test/01-alphabet.txt
```

The listed order is significant for document-aware or streaming modes. The loader must not rely on filesystem/glob order.

For richer provenance, an entry may instead be a mapping:

```yaml
splits:
  train:
    files:
      - path: train/01-report.txt
        source: annual-report-2024
        language: en
        sha256: "<optional SHA-256>"
```

A loader must support the simple path form first. Rich file metadata is an optional extension.

## Text datasets

### Document semantics

For the initial `text_files_lm` builder:

```text
one file = one document
UTF-8 encoding
```

The DataModule reads files in manifest order. It appends an end-of-document (`EOS`) token after every document before constructing token windows:

```text
file A tokens + EOS + file B tokens + EOS
```

This prevents the model from learning artificial transitions between unrelated documents.

### Tokenizer and vocabulary

A text DataModule owns the tokenizer and vocabulary lifecycle:

```text
train files only
  -> fit tokenizer/vocabulary
  -> encode train, val, and test with that same frozen tokenizer
```

The validation and test splits must never be used to fit vocabulary/tokenizer state.

The lowest-level token-window `Dataset` only holds encoded token IDs. It does not fit or independently own a vocabulary.

The manifest declares tokenizer behavior and references the self-contained dataset artifacts:

```yaml
tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens:
    pad: "<pad>"
    eos: "<eos>"
    # Optional when the selected tokenizer supports it:
    # unk: "<unk>"

locking:
  dataset_lock: dataset-lock.json
```

The actual fitted token-to-ID mapping is stored in `tokenizer/tokenizer.json`. A tokenizer is fitted only from the training split, then frozen. Training loads and verifies it; it must never silently refit it.

Runs record the tokenizer artifact reference and fingerprint. They may copy the artifact for archival portability, but the canonical tokenizer belongs to the dataset. Model checkpoints must record compatible tokenizer metadata (at minimum vocabulary size and a tokenizer artifact reference or fingerprint), because embedding and vocabulary-head row IDs have no meaning without the same token-to-ID mapping.

### Example: alphabet dataset

```yaml
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
    files:
      - train/01-alphabet.txt
  val:
    files:
      - val/01-alphabet.txt
  test:
    files:
      - test/01-alphabet.txt

tokenizer:
  name: character
  fit_split: train
  special_tokens:
    pad: "<pad>"
    eos: "<eos>"
```

## File-pair text datasets

Goldfish v1 supports supervised text `input -> output` datasets with:

```yaml
format:
  encoding: utf-8
  document_unit: file-pair
```

A `file-pair` dataset uses the existing causal GRU/LSTM language models in **prefix language-model** mode. It is not yet a separate encoder-decoder/seq2seq architecture.

### File-pair manifest

Every split must be present and each entry explicitly names its input and output file. Paths are relative to the dataset root.

```yaml
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
      - input: train/01-input.txt
        output: train/01-output.txt
      - input: train/02-input.txt
        output: train/02-output.txt
  val:
    files:
      - input: val/01-input.txt
        output: val/01-output.txt
  test:
    files:
      - input: test/01-input.txt
        output: test/01-output.txt

tokenizer:
  name: character
  artifact: tokenizer/tokenizer.json
  lock: tokenizer/tokenizer-lock.json
  fit_split: train
  special_tokens:
    pad: "<pad>"
    eos: "<eos>"
    sep: "<sep>"

locking:
  dataset_lock: dataset-lock.json
```

For `document_unit: file-pair`:

- `train`, `val`, and `test` splits are required by manifest v1;
- every entry must be a mapping with non-empty `input` and `output` paths;
- both paths must be safe dataset-relative paths; absolute paths and traversal are rejected;
- `tokenizer.special_tokens.sep` is required;
- a manifest `input`/`output` entry is one paired **shard**; no pairing is inferred from file names.

### Multi-line paired shards

A paired shard may contain many supervised examples. Goldfish aligns it strictly by line number:

```text
train/inputs.txt             train/outputs.txt
----------------             -----------------
translate: cat               chat
translate: dog               chien
```

This produces two examples:

```text
"translate: cat" -> "chat"
"translate: dog" -> "chien"
```

Rules:

- line `N` in the input file pairs with line `N` in the output file;
- both files must contain exactly the same number of lines;
- empty lines are retained as valid empty input/output examples to preserve alignment;
- a mismatched line count is an error; Goldfish never silently truncates either side;
- manifest ordering first orders shards, then preserves line order inside each shard.

A shard may still contain one line. Thus `file-pair` supports both a single example and efficiently stored collections of short examples.

### Prefix-LM representation and loss

For one pair:

```text
input file:  abc
output file: cba
```

Goldfish encodes the sequence as:

```text
abc <sep> cba <eos>
```

The model consumes the normal shifted causal-LM inputs/targets. The dataset also emits `loss_mask`, so next-token cross-entropy is calculated only for predictions belonging to the output segment and its terminating EOS:

```text
context:        abc <sep>
loss targets:             c b a <eos>
```

Input and separator positions provide conditioning context but are excluded from optimization loss. The `PrefixLanguageModelTask` applies:

```text
effective_loss_mask = attention_mask AND loss_mask
```

This lets file-pair training reuse the current recurrent causal language-model architectures while making the supervised target boundary explicit.

### Tokenizer behavior

Tokenizer fitting reads both sides of **training pairs only**:

```text
all train inputs + all train outputs
-> fit tokenizer
-> encode train, val, and test with the frozen tokenizer
```

For the character tokenizer, a file-pair tokenizer reserves:

```text
PAD = 0
EOS = 1
SEP = 2
characters begin at ID 3
```

The existing non-paired character tokenizer remains compatible with legacy artifacts and uses only PAD/EOS special tokens.

### Preparation, training, and inference

Prepare a file-pair dataset through the normal workflow:

```sh
uv run goldfish prepare data/reverse-pairs
```

Preparation detects `document_unit: file-pair`, creates a SEP-enabled tokenizer from train pairs, then writes dataset and tokenizer locks.

Training selects the matching DataModule and task from the manifest:

| Document unit | Data module | Task |
|---|---|---|
| `file` | `TextFilesLanguageModelDataModule` | `CausalLanguageModelTask` |
| `file-pair` | `FilePairPrefixLanguageModelDataModule` | `PrefixLanguageModelTask` |

```sh
uv run goldfish train data/reverse-pairs --name reverse-gru --model gru
```

For a paired run, inference accepts the input side through `--prompt`; Goldfish appends the dataset SEP token internally before autoregressive output generation:

```sh
uv run goldfish infer runs/exp1-reverse-gru \
  --checkpoint best \
  --prompt "abc" \
  --max-new-tokens 20
```

Users must not include `<sep>` themselves in the prompt.

### Current v1 limitations

- This is prefix-LM conditioning, not an encoder-decoder model with cross-attention.
- Each tokenized line-level `input + SEP + output + EOS` example must fit into one configured `sequence_length`; oversized examples fail clearly rather than being silently truncated.
- The first implementation supports text file pairs only. Numeric/multimodal input-output pairs remain future dataset builders.

## Numeric sequential datasets

Numeric datasets use the same root/split/manifest layout, but files normally represent time partitions, entities, or shards instead of text documents.

```text
data/
└── aapl-5m-return/
    ├── manifest.yaml
    ├── dataset-lock.json
    ├── train/
    │   └── bars-2022.parquet
    ├── val/
    │   └── bars-2023-q1.parquet
    └── test/
        └── bars-2023-q2.parquet
```

A numeric manifest must eventually declare time and feature semantics, especially feature availability and normalization ownership:

```yaml
name: aapl_5m_return
version: "1.0"
modality: numeric
builder: numeric_files_forecast
task: point_forecast

splits:
  train:
    files: [train/bars-2022.parquet]
  val:
    files: [val/bars-2023-q1.parquet]
  test:
    files: [test/bars-2023-q2.parquet]

time:
  timezone: America/New_York
  frequency: 5m
  context_window: 78
  forecast_horizons: [1h]

features:
  - name: log_return
    availability: observed
    normalization: standard
  - name: volume
    availability: observed
    transform: log1p
    normalization: standard
  - name: minute_of_day_sin
    availability: known_future

target:
  name: future_log_return
  horizon: 1h
```

Numeric normalizers must be fit on the training split only. Chronological split boundaries, timezone, horizon, and any embargo are part of the dataset contract because they affect leakage safety.

## Multimodal datasets

Text-plus-numeric datasets use the same manifest-based split declaration. Their manifest additionally needs an alignment contract that states what text may be visible at a numeric prediction cutoff:

```yaml
alignment:
  text_timestamp_field: published_at
  rule: available_before_cutoff
  text_lookback: 24h
  no_text_behavior: empty_context
```

For a cutoff `t`, only text with `published_at <= t` may be included. This rule is a dataset invariant, not a model preference.

## DataModule responsibilities

A registered DataModule is responsible for:

1. reading and validating `manifest.yaml`;
2. loading only manifest-listed files;
3. enforcing the split and modality contracts;
4. fitting preprocessing state only on training data (tokenizer, vocabulary, normalizer, etc.);
5. constructing train/val/test datasets and DataLoaders;
6. exposing runtime metadata needed to build a model;
7. producing reproducibility metadata for run artifacts.

A DataModule should expose metadata comparable to:

```python
@dataclass(frozen=True)
class DataMetadata:
    name: str
    modality: str
    sequence_length: int | None
    train_samples: int
    validation_samples: int
    test_samples: int | None
    tokenizer: TokenizerMetadata | None
    feature_schema: FeatureSchema | None
    target_schema: TargetSchema | None
```

The generic Trainer does not interpret this metadata. The experiment runner uses it to construct compatible models and to save a data manifest for the run.

## Dataset locking and integrity

Goldfish separates declarative dataset definition from derived integrity snapshots:

| Artifact | Responsibility |
|---|---|
| `manifest.yaml` | Dataset semantics: split members and order, modality, builder, format, tokenizer configuration, and feature/target/time definitions. |
| `dataset-lock.json` | Immutable snapshot of manifest-listed raw split files and lock-relevant manifest semantics. |
| `tokenizer/tokenizer.json` | Actual text token-to-ID mapping. |
| `tokenizer/tokenizer-lock.json` | Binding from the tokenizer artifact to the locked training split and tokenizer configuration. |

### Dataset lock

`dataset-lock.json` stores SHA-256 hashes for each manifest-listed raw file, preserving manifest order and split membership. It also stores split fingerprints and one overall dataset fingerprint.

For `document_unit: file-pair`, a locked entry stores separate `input` and `output` snapshots. The pair role is part of its identity, so exchanging input/output files changes the lock even when their contents are unchanged.

A split fingerprint is computed over canonical ordered entries containing at least:

```text
split name + ordered entry identity + relative path(s) + file content SHA-256
```

The overall fingerprint additionally covers lock-relevant manifest semantics, such as builder, modality, format, and tokenizer configuration. A canonical JSON serialization (UTF-8, sorted object keys, fixed separators) is hashed with SHA-256. The lock payload never contains its own fingerprint.

This means the lock changes when file content, file order, split membership, or data-interpretation settings change.

Example shape:

```json
{
  "format": "goldfish-dataset-lock-v1",
  "algorithm": "sha256",
  "dataset": {"name": "alphabet", "version": "1.0"},
  "splits": {
    "train": {
      "fingerprint": "<sha256>",
      "files": [
        {"path": "train/01-alphabet.txt", "sha256": "<sha256>", "bytes": 624}
      ]
    },
    "val": {
      "fingerprint": "<sha256>",
      "files": [
        {"path": "val/01-alphabet.txt", "sha256": "<sha256>", "bytes": 617}
      ]
    }
  },
  "fingerprint": "<sha256>"
}
```

### Tokenizer lock

`tokenizer/tokenizer-lock.json` confirms that `tokenizer.json` is both unmodified and derived from the current locked training split using the declared tokenizer configuration.

```json
{
  "format": "goldfish-tokenizer-lock-v1",
  "algorithm": "sha256",
  "tokenizer": {
    "path": "tokenizer.json",
    "sha256": "<sha256>",
    "name": "character",
    "vocab_size": 28,
    "special_token_ids": {"pad": 0, "eos": 1}
  },
  "source": {
    "dataset_name": "alphabet",
    "dataset_version": "1.0",
    "train_fingerprint": "<dataset-lock train fingerprint>"
  },
  "config": {
    "fit_split": "train",
    "special_tokens": {"pad": "<pad>", "eos": "<eos>"}
  },
  "fingerprint": "<sha256>"
}
```

### Lock lifecycle

Dataset preparation is explicit:

```text
read manifest + raw train files
-> fit tokenizer from train only
-> write tokenizer.json
-> calculate dataset-lock.json
-> calculate tokenizer-lock.json
-> dataset is locked
```

Training verifies locks in this order:

```text
read manifest.yaml
-> verify dataset-lock.json against manifest-listed raw files
-> verify tokenizer-lock.json against tokenizer.json, train fingerprint, and manifest tokenizer config
-> load datasets and tokenizer
-> train
```

Training must fail on any mismatch and must not silently rewrite locks or refit tokenizer state. If raw training data or tokenizer configuration changes, the user explicitly rebuilds tokenizer artifacts, regenerates locks, and increments the dataset version as appropriate.

## Run artifacts and integrity

The dataset manifest and its locks describe the intended, verified dataset. Each experiment should also record what it actually consumed:

```text
runs/<run-id>/
├── config.resolved.yaml
├── data_manifest.json
└── tokenizer/
    └── tokenizer.json
```

`data_manifest.json` should record at least:

- the source dataset manifest path, dataset version, and dataset fingerprint;
- each selected split file in manifest order;
- verified file and split hashes;
- tokenizer/normalizer artifact references and fingerprints;
- sequence length and other derived preprocessing settings.

This lets a later run distinguish between a model change and an accidental data change.
