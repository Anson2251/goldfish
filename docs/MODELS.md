# Goldfish Model Specification

This document specifies the model registry, recurrent components, and model compositions currently implemented in Goldfish. Models consume typed batches and return `ModelOutput`; objectives are defined separately in [`TASKS.md`](TASKS.md).

## Shared output contract

Every model returns:

```python
@dataclass
class ModelOutput:
    predictions: dict[str, Tensor]
    representations: Tensor | None = None
    aux_losses: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Tensor | float] = field(default_factory=dict)
```

`predictions` contains tensors required by a task. `representations` is optional contextual state for downstream inspection or extension. The current recurrent models do not emit auxiliary losses or diagnostics.

## Registry

**Implementation:** `goldfish.models.ModelRegistry`  
**Shared instance:** `goldfish.models.model_registry`

Models are registered under a `(family, name)` pair. Both values are normalized by trimming whitespace, lowercasing, and converting underscores to hyphens. Duplicate registration within a family is rejected. Lookup failure reports sorted available names in that family.

Current registrations:

| Family | Name | Factory |
|---|---|---|
| `language` | `gru` | `GRULanguageModel` |
| `language` | `lstm` | `LSTMLanguageModel` |
| `forecast` | `gru` | `GRUForecastModel` |
| `forecast` | `lstm` | `LSTMForecastModel` |
| `forecast` | `multihead-lstm` | `MultiHeadLSTMForecastModel` |

Create a configured model with:

```python
model = model_registry.create("language", "gru", **model_kwargs)
```

The training and inference entry points resolve the family from the dataset modality and use this registry rather than branching on individual model classes.

## Recurrent backbones

**Implementations:** `GRUBackbone`, `LSTMBackbone` in `goldfish.models.components.recurrent`

Both wrappers instantiate PyTorch batch-first recurrent layers and validate that input embeddings are rank 3:

```python
embedded: Tensor  # [B, T, D]
```

Constructor parameters:

| Parameter | Meaning |
|---|---|
| `input_dim` | Input feature dimension `D` |
| `hidden_dim` | Recurrent hidden dimension `H` |
| `num_layers` | Stacked recurrent layers, default `1` |
| `dropout` | PyTorch inter-layer recurrent dropout, default `0.0` |

Forward results are:

| Backbone | contextual states | final state |
|---|---|---|
| GRU | `[B, T, H]` | `[num_layers, B, H]` |
| LSTM | `[B, T, H]` | `(hidden, cell)`, each `[num_layers, B, H]` |

Both accept an optional compatible previous state. This enables incremental text generation without reprocessing the full generated prefix.

## Language models

**Implementations:** `GRULanguageModel`, `LSTMLanguageModel` in `goldfish.models.language.recurrent`

### Constructor

```python
GRULanguageModel(
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    *,
    num_layers: int = 1,
    dropout: float = 0.0,
)
```

`LSTMLanguageModel` has the same signature. `vocab_size` must be positive. The model consists of:

```text
input token IDs -> Embedding(V, E) -> GRU/LSTM(E, H) -> Linear(H, V)
```

### Batch and output

The normal forward path expects a structural token batch:

```python
input_ids: Tensor        # [B, T]
attention_mask: Tensor   # [B, T]
```

`input_ids` must be rank 2 and `attention_mask` must have exactly the same shape. The model validates shape compatibility but does not use the mask to pack, zero, or reset recurrent states. The task applies masking to loss calculation.

Forward output:

```python
ModelOutput(
    predictions={"token_logits": logits},  # [B, T, V]
    representations=states,                 # [B, T, H]
)
```

### Incremental token API

```python
output, final_state = model.forward_tokens(input_ids, hidden_state=None)
```

`input_ids` must have shape `[B, T]` with `T > 0`. Passing the final state from one call to the next produces the same sequence logits as evaluating the concatenated token sequence in a single call, provided the same model state is used.

`forward_tokens` does not take an attention mask and is the interface used by generation utilities.

## Multi-horizon forecast models

**Implementations:** `GRUForecastModel`, `LSTMForecastModel` in `goldfish.models.forecast.recurrent`

### Constructor

```python
GRUForecastModel(
    feature_count: int,
    target_count: int,
    horizon_count: int,
    hidden_dim: int,
    *,
    num_layers: int = 1,
    dropout: float = 0.0,
)
```

`LSTMForecastModel` has the same signature. All dimensions must be positive, and `num_layers` must be positive.

Architecture:

```text
normalized history [B, L, F]
-> GRU/LSTM input size F, hidden size H
-> final state at history position L - 1 [B, H]
-> Linear(H, horizon_count * target_count)
-> reshape [B, horizon_count, target_count]
```

The model uses `states[:, -1]`; it therefore requires a non-empty lookback dimension in practice. Inputs are fixed-length windows and are not packed or padded by the model.

### Batch and output

The forward path requires:

```python
inputs: Tensor  # [B, L, F]
```

`inputs` must be rank 3. The output is:

```python
ModelOutput(
    predictions={"forecast": forecast},    # [B, H, C]
    representations=states,                  # [B, L, hidden_dim]
)
```

Here `H` equals `horizon_count` and `C` equals `target_count`. `PointForecastTask` requires this forecast tensor to match the normalized batch target tensor exactly.

## Doubly stochastic channel mixer

**Implementation:** `DoublyStochasticMixer` in `goldfish.models.components.mixing`

`DoublyStochasticMixer` applies a learnable, static channel mixing matrix to an input shaped `[..., N, D]`:

```text
mixed[output_channel] = sum(input_channel, mixing[output_channel, input_channel] * input[input_channel])
```

The parameter logits are projected in log space through iterative Sinkhorn normalization. The resulting `mixing` matrix is non-negative with rows and columns summing to one, so each output channel is a convex combination of all source channels and each source channel contributes a total weight of one. It is initialized near the identity matrix, which makes initial mixing minimal while preserving a learnable communication path between channels.

## Multi-head LSTM forecast model

**Implementation:** `MultiHeadLSTMForecastModel` in `goldfish.models.forecast.recurrent`

The model runs `num_heads` independent LSTMs. Each head projects and LayerNorms the common numeric input, produces a `head_dim = hidden_dim / num_heads` state sequence, and then stacks the states as `[B, L, num_heads, head_dim]`. `DoublyStochasticMixer` mixes the head dimension before the mixed heads are concatenated and fused into `[B, L, hidden_dim]`. The final time position is projected to the standard `[B, horizon_count, target_count]` forecast.

`hidden_dim` must be divisible by `num_heads`. The `multihead-lstm` registry model accepts `num_heads` (CLI: `--lstm-heads`, default `4`) and `sinkhorn_iterations` (CLI: `--sinkhorn-iterations`, default `20`). Do not mean-pool the mixer output over heads: double stochasticity preserves this mean exactly, making such a readout independent of the mixer.

## Model profiles and configuration

Architecture settings are stored in version-controlled YAML profiles under the repository-root `model-profiles/` directory, for example `model-profiles/forecast/multihead-lstm-small.yaml`. New training runs require `--model-profile`:
```sh
uv run goldfish train data/fourier \
  --model-profile model-profiles/forecast/multihead-lstm-small.yaml
```

A profile identifies the registry model and supplies only architecture-owned parameters:

```yaml
model:
  family: forecast
  name: multihead-lstm
  parameters:
    hidden_dim: 32
    num_heads: 4
    num_layers: 1
    dropout: 0.0
    sinkhorn_iterations: 20
```

Goldfish injects dataset-derived dimensions into the resolved run configuration: `vocab_size` for language models, and `feature_count`, `target_count`, and `horizon_count` for forecast models. Profiles must not supply these fields. The fully resolved `model.parameters` mapping is saved to the run's `config.yaml`; training, resume, `infer`, and `forecast` construct models through that same mapping. A resumed run cannot take `--model-profile`, because its saved model configuration is authoritative.

## Extension requirements

A new model family or composition should:

1. accept an explicit typed or structural batch rather than an unstructured tensor tuple;
2. return `ModelOutput` with documented stable prediction keys;
3. register under a non-colliding family/name pair if it is configurable;
4. validate essential input ranks and configuration dimensions near the model boundary;
5. leave loss calculation, raw-unit metrics, and decoding to the corresponding task or inference component.
