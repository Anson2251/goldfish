# Goldfish Task Specification

This document specifies the task layer currently implemented in Goldfish. A task translates a model's structured output and a typed batch into the differentiable optimization loss and scalar metrics used by the generic trainer.

For dataset construction and manifest-level task names, see [`DATASETS.md`](DATASETS.md). For model input/output contracts, see [`MODELS.md`](MODELS.md).

## Core contract

```python
@dataclass
class StepResult:
    loss: Tensor
    metrics: dict[str, Tensor | float]

class Task(Protocol[BatchT]):
    def compute(self, output: ModelOutput, batch: BatchT) -> StepResult: ...
```

`StepResult.loss` must remain connected to the computation graph. Metrics are batch-level scalar values; task implementations return detached tensors for metrics derived from the loss.

The trainer owns device transfer, backpropagation, gradient clipping, optimizer/scheduler steps, and aggregation. A task owns target interpretation, the primary objective, and task-specific metrics. The trainer may add configured weighted auxiliary losses from `ModelOutput.aux_losses` to the primary loss.

## Common conventions

### Structured predictions

Tasks retrieve their required tensors from `ModelOutput.predictions` by stable key. A missing key raises `KeyError` rather than producing a fallback result.

| Task family | Required prediction key |
|---|---|
| Causal language modelling | `token_logits` |
| Prefix language modelling | `token_logits` |
| Point forecasting | `forecast` |

### Empty supervised sets

The two language-model tasks return `logits.sum() * 0.0` when their effective loss mask selects no positions. This is a scalar zero that preserves a valid gradient path, with `loss = 0` and `perplexity = 1`. It prevents a reduction over an empty tensor while making no parameter update from the batch.

### Metric aggregation

The trainer records the total optimization loss under `loss`, then merges task metrics. Since implemented tasks also expose a `loss` metric, that task value replaces the same name and equals the primary objective for these tasks. Epoch metrics are unweighted means of the per-batch scalar metrics.

## Causal language modelling

**Implementation:** `goldfish.tasks.CausalLanguageModelTask`  
**Manifest task name:** `causal_language_model`

### Required batch fields

The task accepts any structural batch with:

```python
target_ids: Tensor       # torch.long, [B, T]
attention_mask: Tensor   # torch.bool, [B, T]
```

The standard `LanguageModelBatch` also includes `input_ids: Tensor [B, T]` for the model. `target_ids[b, t]` is the next token corresponding to `input_ids[b, t]`. `attention_mask[b, t]` is `True` exactly where that target contributes to loss and metrics.

### Required output

```python
output.predictions["token_logits"]  # [B, T, V]
```

`V` is the vocabulary size. The first two logits dimensions must equal `target_ids.shape`.

### Objective and metrics

Let \(M\) be the set of positions for which `attention_mask` is true. The task computes mean cross entropy over only those positions:

\[
L = -\frac{1}{|M|}\sum_{(b,t)\in M}\log \operatorname{softmax}(z_{b,t})[y_{b,t}]
\]

It reports:

- `loss`: detached cross entropy \(L\);
- `perplexity`: `exp(L)`.

### Validation rules

- `token_logits` must be rank 3 (`[B, T, V]`).
- `target_ids` and `attention_mask` must be rank 2.
- `target_ids.dtype` must be `torch.long`.
- `attention_mask.dtype` must be `torch.bool`.
- logits batch/time dimensions and mask shape must both match `target_ids`.

## Prefix language modelling

**Implementation:** `goldfish.tasks.PrefixLanguageModelTask`  
**Manifest task name:** `prefix_language_model`

Prefix LM uses a causal token model but restricts supervision to the completion portion of a paired input/output sequence.

### Required batch fields

```python
target_ids: Tensor       # torch.long, [B, T]
attention_mask: Tensor   # torch.bool, [B, T]
loss_mask: Tensor        # torch.bool, [B, T]
```

The standard `PrefixLanguageModelBatch` additionally has `input_ids`. `attention_mask` identifies non-padding positions. `loss_mask` identifies positions belonging to the completion; prefix targets are excluded even when they are valid tokens.

### Required output

```python
output.predictions["token_logits"]  # [B, T, V]
```

### Objective and metrics

The effective supervision mask is:

```python
valid_mask = attention_mask & loss_mask
```

Cross entropy and perplexity use only positions selected by `valid_mask`, with the same definitions as causal LM. Thus the model can consume prefix tokens but receives no direct loss for predicting them.

### Validation rules

The causal-LM validation rules apply, plus:

- `loss_mask` must be rank 2 and `torch.bool`;
- `loss_mask.shape` must equal `target_ids.shape`.

## Point forecasting

**Implementation:** `goldfish.tasks.PointForecastTask`  
**Manifest task name:** `point_forecast`

This task trains a direct multi-horizon point forecaster in normalized target space while exposing error metrics in the original target units.

### Construction

```python
PointForecastTask(normalizer: StandardNormalizer, targets: Sequence[str])
```

`targets` is the ordered target-column list. It must correspond to the final dimension of the forecast tensors and is used by the normalizer to select the correct train-fitted statistics.

### Required batch fields

```python
inputs: Tensor                         # [B, lookback, F]
targets: Tensor                        # [B, H, C], normalized
entity_ids: tuple[str, ...] = ()       # evaluation metadata
cutoff_timestamps: tuple[str, ...] = ()
```

Only `targets` is read by the task. `inputs` is consumed by the forecasting model; metadata remains available for ordered forecast export/evaluation.

### Required output

```python
output.predictions["forecast"]  # [B, H, C], normalized
```

The forecast shape must equal `batch.targets.shape` exactly.

### Objective and metrics

The differentiable objective is mean squared error in normalized units:

\[
L_{\mathrm{MSE}} = \operatorname{mean}((\hat y_{norm} - y_{norm})^2)
\]

For reporting, predictions and targets are inverse-transformed using the configured `StandardNormalizer` and selected target names. With raw error \(e = \hat y_{raw} - y_{raw}\), the task reports:

- `mse`: detached normalized MSE (equal to `loss`);
- `mae`: `mean(abs(e))` in raw target units;
- `rmse`: `sqrt(mean(e ** 2))` in raw target units.

No masking or variable-length sequence handling is implemented for forecast targets; each batch row is a fixed-shape window.

## Extension requirements

A new task should:

1. define the required prediction keys and structural batch fields;
2. validate shape and dtype assumptions before calculating reductions;
3. return a differentiable scalar primary loss and scalar metrics;
4. keep data preparation, decoding, and task-specific artifact export outside the generic trainer;
5. use stable metric names suitable for experiment logs and scheduler monitoring.
