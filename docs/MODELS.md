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
| `forecast` | `deltanet` | `DeltaNetForecastModel` |
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

`DeltaNetBackbone` (in `goldfish.models.components.deltanet`) is the third recurrent backbone; it is a fast-weight memory rather than a hidden-vector RNN and is described with its forecast model below.

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

## DeltaNet forecast model

**Implementation:** `DeltaNetForecastModel` in `goldfish.models.forecast.recurrent`, with `DeltaNetBackbone` and the `delta_rule_scan` primitive in `goldfish.models.components.deltanet`

DeltaNet (Schlag et al., 2021; Yang et al., 2024) is linear attention with an error-correcting delta-rule memory. Each head maintains a matrix-valued fast-weight memory `S` in `R^{head_dim x head_dim}` updated per position as:

```text
S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T
o_t = S_t q_t
```

`beta_t` is a learnable per-head scalar in `(0, 1)` (sigmoid of a logit, initialized near one) acting as the delta-rule learning rate: the update erases the old association for key `k_t` and writes a blended replacement. With unit-norm keys, the transition `I - k k^T` is a projection that removes only the direction of `k`, keeping interference between stored associations low.

### Constructor

```python
DeltaNetForecastModel(
    feature_count: int,
    target_count: int,
    horizon_count: int,
    hidden_dim: int,
    *,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.0,
    short_conv_kernel: int = 4,
    beta_initial_logit: float = 4.0,
)
```

`hidden_dim` must be divisible by `num_heads`. Each layer is a pre-norm residual block: LayerNorm, fused query/key/value projection, causal depthwise short convolution over the concatenated channels (`short_conv_kernel = 1` disables it), SiLU activation, L2 normalization of queries and keys, the delta-rule scan over the history, output RMSNorm, and an output projection. `dropout` applies between layers only, like the torch recurrent backbones. The first layer consumes `feature_count` inputs without a residual; subsequent layers consume and residual-connect `hidden_dim`.

### Batch and output

The forward path requires `inputs: Tensor  # [B, L, F]` and returns:

```python
ModelOutput(
    predictions={"forecast": forecast},    # [B, H, C]
    representations=states,                  # [B, L, hidden_dim]
)
```

The backbone additionally exposes the final fast-weight memory `S_L` with shape `[B, num_heads, head_dim, head_dim]` as its state; the forecast head reads the per-position output at history position `L - 1`. Profile example:

```sh
uv run goldfish train data/fourier \
  --model-profile model-profiles/forecast/deltanet-small.yaml
```

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
