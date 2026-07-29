# Layerwise Head Mixing Experiments

This document records the architectural change from output-only head mixing to layerwise head mixing, the resulting `exp79` observation, and the follow-up ablations needed to identify the cause of the change.

## Motivation

The original multi-head LSTM uses independent recurrent heads. Each head processes the full history independently, and the doubly stochastic mixer is applied only after the final recurrent layer:

```text
per-head projection and LayerNorm
→ per-head LSTM stack
→ doubly stochastic mixer
→ flatten → fusion → forecast
```

This makes the mixer a final fusion operation. It can combine completed head representations, but it cannot affect later recurrent gates or state updates. In contrast, a single LSTM allows all hidden dimensions to interact at every gate and time step.

`exp73` showed that a constrained output-only mixer was better than an unconstrained output-only mixer, but the single-LSTM reference was still more accurate. That result suggested that delayed cross-head communication, rather than the mixer constraint alone, could be the main limitation of the original multi-head architecture.

## Architectural change

The multi-head model now represents each head as a stack of single-layer LSTMs. After every recurrent layer, all head outputs are stacked, mixed, and returned to the corresponding heads as input to the next recurrent layer:

```text
per-head projection and LayerNorm
→ per-head LSTM layer 1
→ doubly stochastic mixer
→ optional inter-layer dropout
→ per-head LSTM layer 2
→ doubly stochastic mixer
→ flatten → LayerNorm → Linear → forecast
```

For `num_layers: 1`, this preserves the original behavior of mixing the final head outputs. For `num_layers >= 2`, it creates a constrained communication path between heads before later recurrent layers execute.

The mixer is still static: one learned `N × N` matrix is shared across examples, time positions, feature channels, and recurrent-layer boundaries. It is doubly stochastic, so each output head receives a convex combination of inputs and each source head retains unit total contribution across output heads.

## Experiment setup

`exp79` evaluates the layerwise mixer using the same dataset lock and training configuration as the constrained output-only baseline `exp73`:

```yaml
model:
  family: forecast
  name: multihead-lstm
  parameters:
    hidden_dim: 32
    num_heads: 4
    num_layers: 2
    dropout: 0.0
    sinkhorn_iterations: 20

optimization:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

scheduler:
  name: cosine
  t_max: 500

training:
  epochs: 1000
```

All compared runs use `data/fourier-lb256`: 256-step windows, horizons `[1, 5, 20]`, inputs `[signal, trend, phase_sin, phase_cos]`, target `signal`, and normalized MSE for training and checkpoint selection. The experiment is on one deterministic synthetic process with direct phase and trend features; it is a controlled architecture comparison, not a claim about general forecasting data.

The run environment records Git commit `a7f9eabaf3be0deb33e6ab13d74a7da0cfc30c34` and `git_dirty: true`. The working-tree change relevant to this experiment is the layerwise mixing implementation described above.

## Observed results

| Run | Architecture | Best validation loss | Best epoch | Final train loss | Final validation loss |
|---|---|---:|---:|---:|---:|
| `exp77` | Unconstrained, output-only mixer | `0.02544` | 993 | `0.01610` | `0.03228` |
| `exp73` | Doubly stochastic, output-only mixer | `0.01584` | 961 | `0.01646` | `0.01857` |
| `exp78` | Single LSTM, width 20 | `0.00772` | 958 | `0.00509` | `0.00929` |
| `exp79` | Doubly stochastic, layerwise mixer | **`0.00705`** | 913 | **`0.00469`** | **`0.00893`** |

Relative to the output-only constrained multi-head baseline, `exp79` reduced:

- best validation loss by approximately 55.5% (`0.01584 → 0.00705`); and
- final validation loss by approximately 51.9% (`0.01857 → 0.00893`).

`exp79` also slightly outperformed the parameter-near single-LSTM reference on this single trajectory.

### Training trajectory

The advantage appears before the second cosine cycle:

| Epoch | `exp73` output-only train / validation | `exp79` layerwise train / validation | `exp78` single LSTM train / validation |
|---:|---:|---:|---:|
| 50 | `0.19329 / 0.23532` | `0.17252 / 0.20267` | `0.08391 / 0.12482` |
| 100 | `0.07679 / 0.09971` | `0.05885 / 0.08763` | `0.03556 / 0.06664` |
| 250 | `0.03399 / 0.05075` | `0.01197 / 0.03041` | `0.01132 / 0.02451` |
| 500 | `0.02239 / 0.03279` | `0.00604 / 0.01812` | `0.00710 / 0.01671` |
| 750 | `0.01782 / 0.02658` | `0.00496 / 0.01850` | `0.00756 / 0.01246` |
| 1000 | `0.01646 / 0.01857` | `0.00469 / 0.00893` | `0.00509 / 0.00929` |

This supports the hypothesis that permitting head communication before the second recurrent layer removes an important bottleneck in the output-only architecture. The improvement is visible by epoch 50 and is not only a late effect of the second cosine cycle.

The second cycle still matters. With `T_max: 500`, the learning rate descends twice over the 1,000 epochs, and validation loss is not monotonic within the second cycle. Interpret trajectory shape as an observation of this optimizer/model path, rather than as a standalone stability measurement.

## Mixer inspection

Let `P` denote the final Sinkhorn-projected mixer and `I` the `4 × 4` identity matrix. The final checkpoints give the following distances:

| Run | `||P - I||_F` | `||P - I||_2` | `max(abs(P - I))` | Total off-diagonal mass | Diagonal range |
|---|---:|---:|---:|---:|---:|
| `exp73` output-only | `3.0497e-4` | `1.8742e-4` | `1.4293e-4` | `5.2643e-4` | `[0.999857, 0.999881]` |
| `exp79` layerwise | `2.6216e-4` | `1.7733e-4` | `1.3947e-4` | `4.4274e-4` | `[0.999861, 0.999927]` |

Here `||·||_F` is the Frobenius norm, `||·||_2` is the spectral norm, and total off-diagonal mass is `Σ_{i != j} P[i, j]`. Since `P` is doubly stochastic, this mass also equals the total diagonal deficit `Σ_i (1 - P[i, i])`.

The final `exp79` mixer is therefore not merely close to identity: it is slightly **closer** to identity than the final `exp73` mixer under all reported distance measures. Its complete matrix is:

```text
P_exp79 =
[[0.999889, 0.000026, 0.000043, 0.000042],
 [0.000029, 0.999927, 0.000044, 0.000047],
 [0.000041, 0.000023, 0.999880, 0.000050],
 [0.000036, 0.000024, 0.000037, 0.999861]]
```

At final time, at most about `1.4e-4` of any one output head's identity routing has been displaced in a single matrix entry, and only `4.4e-4` mass is routed off diagonal over the entire matrix. The observed `exp79` improvement therefore cannot be explained by a large, persistent final cross-head permutation or averaging operation.

This creates an apparent tension: `exp79` is much better than `exp73`, yet its final learned mixer does not visibly perform strong cross-head routing. Several explanations remain possible:

1. **Useful communication occurred earlier in training.** The matrix may have moved enough during optimization to change head specialization and gradient flow, even if it later returned near identity.
2. **Small mixing effects are amplified recurrently.** A small perturbation before layer 2 can alter gates and hidden-state trajectories over the remaining sequence.
3. **The benefit is not caused by learned routing.** Explicitly splitting the LSTM stack into one-layer modules can alter parameter initialization order, kernel execution, or the optimization path even with identity mixing.
4. **The result includes run-to-run variation.** All current comparisons contain one initialization/training trajectory per condition.

The first two explanations support layerwise constrained communication as the mechanism. The latter two are alternatives that require direct controls.

## Next ablations

The next experiments should use the `exp79` data lock, model dimensions, optimizer, scheduler, epoch budget, and preferably fixed seed sets. They should be run before changing other architectural details.

### 1. Fixed identity mixer at every layer

Use the same layerwise execution structure but fix:

```text
M = I
```

at every mixer application. It must have no learnable mixer parameters and must not apply even small off-diagonal routing.

This is the key control for explicit layer splitting. Interpretation:

| Result | Interpretation |
|---|---|
| Fixed identity approximately matches `exp79` | Layer splitting, initialization, or trajectory variation may explain most of the gain; learned routing is not shown to be necessary. |
| Fixed identity is near `exp73` and below `exp79` | Learned layerwise communication is a likely contributor. |
| Fixed identity lies between `exp73` and `exp79` | Both the layerwise structure and learned routing may contribute. |
| Fixed identity exceeds `exp79` | The learnable mixer may not be useful under this configuration. |

### 2. Randomly initialized learnable doubly stochastic mixer

Keep the layerwise, learnable Sinkhorn mixer but initialize logits from a specified random distribution rather than an identity-biased diagonal matrix. For example:

```text
logits ~ Normal(0, random_std)
M = Sinkhorn(logits)
```

Record `random_std`, the initial projected matrix, and the seed. This differs from a random permutation-like initialization: normal random logits followed by Sinkhorn usually create a diffuse matrix around `1 / N`, while a random permutation initialization tests sparse routing.

Interpretation relative to identity-initialized `exp79`:

| Result | Interpretation |
|---|---|
| Random initialization approximately matches `exp79` | Identity warm start is not essential; learnable layerwise constrained communication is more important. |
| Random initialization starts worse but converges near `exp79` | Identity initialization mainly helps the optimization path. |
| Random initialization stays worse | Near-independent heads at initialization are an important inductive bias. |
| Random initialization improves results | The identity bias may keep the mixer too close to identity and limit useful routing. |

### 3. Fixed random doubly stochastic mixer

This optional control separates learned routing from a fixed cross-head communication topology. Freeze a sampled, projected doubly stochastic matrix for the full run.

| Comparison | Question answered |
|---|---|
| Fixed random vs fixed identity | Does any fixed cross-head mixing help? |
| Fixed random vs learnable random-init | Does adaptation of routing matter after initialization? |
| Learnable random-init vs identity-init | Is the identity warm start important? |

## Decision criteria

Do not select a new default from one run. For each primary condition, use the same prespecified seed set and report:

- best validation loss and the epoch at which it occurs;
- final validation loss at the same total epoch budget;
- validation loss at matched cosine-cycle endpoints, especially epochs 500 and 1,000;
- test metrics from the selected validation checkpoint; and
- the initial, intermediate, and final mixer matrices.

A seed sweep is especially important because the currently observed advantage of `exp79` over the single LSTM is small, while its advantage over the output-only multi-head baseline is large.

## Current working conclusion

Layerwise mixing is the leading architectural direction for this multi-head LSTM. On the available `fourier-lb256` run, it substantially improves over output-only mixing and reaches the single-LSTM reference. The result is promising but does not yet isolate whether the gain comes from learnable cross-head routing, identity-preserving layerwise execution, or ordinary run-to-run variation.
