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

---

## UPDATE: Random-init ablation (exp80)

### Setup

`exp80` evaluates the layerwise doubly stochastic mixer with random initialization, using the same data lock, model dimensions, optimizer, scheduler, and epoch budget as `exp79`:

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
    mixer_initialization: random
    mixer_random_std: 1.0
```

The only difference from `exp79` is `mixer_initialization: random`.

The run environment records Git commit `36c31fbdb29f84148a5a7a151d4d8cb54b277384` and `git_dirty: true`. Data fingerprint, normalizer fingerprint, and split fingerprints match `exp73` and `exp79`.

### Results

| Run | Mixer init | Best validation loss | Best epoch | Final validation loss |
|---|---:|---:|---:|
| `exp77` | identity, output-only unconstrained | `0.02544` | 993 | `0.03228` |
| `exp73` | identity, output-only doubly stochastic | `0.01584` | 961 | `0.01857` |
| `exp78` | single LSTM, width 20 | `0.00772` | 958 | `0.00929` |
| `exp79` | identity, layerwise doubly stochastic | **`0.00705`** | 913 | **`0.00893`** |
| `exp80` | **random**, layerwise doubly stochastic | **`0.04832`** | 968 | **`0.05208`** |

Random initialization degraded best validation loss by a factor of 6.9× relative to identity-init (`0.00705 → 0.04832`). `exp80` is also worse than the output-only unconstrained baseline `exp77` (1.9×) and the output-only constrained baseline `exp73` (3.1×).

### Training trajectory

| Epoch | `exp79` identity-init train / validation | `exp80` random-init train / validation |
|---:|---:|---:|
| 1 | `1.01938 / 1.60837` | `0.96333 / 2.02788` |
| 10 | `0.35377 / 0.48669` | `0.35935 / 0.49358` |
| 20 | `0.23027 / 0.30023` | `0.24147 / 0.31370` |
| 30 | `0.20727 / 0.26507` | `0.22168 / 0.28185` |
| 50 | `0.17252 / 0.20267` | `0.19930 / 0.23910` |
| 100 | `0.05885 / 0.08763` | `0.09255 / 0.13421` |
| 250 | `0.01197 / 0.03041` | `0.05480 / 0.07405` |
| 500 | `0.00604 / 0.01812` | `0.04501 / 0.06538` |
| 750 | `0.00496 / 0.01850` | `0.04617 / 0.05664` |
| 1000 | `0.00469 / 0.00893` | `0.03605 / 0.05208` |

At epoch 1, random-init has slightly better training loss but worse validation loss—cross-head routing is immediately a worse prior for this dataset. The gap is visible by epoch 50, clear by epoch 250, and persists to epoch 1,000. The random-init run makes minimal progress after epoch 250 (`0.074 → 0.052`) while identity-init continues improving through the second cosine cycle (`0.030 → 0.009`).

This is not a slow-start problem that recovers given enough training.

### Final mixer

```text
P_exp80 =
[[0.525, 0.227, 0.095, 0.152],
 [0.333, 0.226, 0.080, 0.361],
 [0.064, 0.276, 0.407, 0.254],
 [0.079, 0.271, 0.418, 0.233]]
```

| Run | Mixer init | `||P - I||_F` | `||P - I||_2` | Total off-diagonal mass | Diagonal range |
|---|---:|---:|---:|---:|
| `exp73` | identity | `3.05e-4` | `1.87e-4` | `5.26e-4` | `[0.999857, 0.999881]` |
| `exp79` | identity | `2.62e-4` | `1.77e-4` | `4.43e-4` | `[0.999861, 0.999927]` |
| `exp80` | random | **`1.58`** | **`1.15`** | **`2.61`** | `[0.226, 0.525]` |

The random-init final mixer is far from identity: 65% of total routing mass sits off diagonal. The model learned a stable, non-trivial cross-head routing pattern under the doubly stochastic constraint, but that pattern has poor validation performance. The model did not recover identity-like routing during training.

### Interpretation

1. **Identity initialization is essential, not optional.** Changing only the mixer initialization degrades performance by 6.9×—the largest single-intervention effect observed across all ablations. This is not a slow-start problem; the gap persists and widens through epoch 1,000.

2. **For this dataset, near-independent head encoding is a strong inductive bias.** Forcing substantial cross-head communication from epoch 1—even under the doubly stochastic constraint—disrupts early head specialization. Each head likely needs to develop a useful temporal representation before constrained routing can improve it.

3. **The mechanism of identity-init gain remains undetermined.** The final mixer is close to identity, but the advantage is large and visible by epoch 50. The fixed-identity-mixer control (no learnable mixer parameters, `M = I` at every layer) is now the highest-priority next ablation to distinguish the contributions of explicit layer splitting from learned near-identity routing.

---

## UPDATE: Uniform-init ablation (exp81)

### Setup

`exp81` evaluates the layerwise doubly stochastic mixer with uniform initialization, using the same data lock, model dimensions, optimizer, scheduler, and epoch budget as `exp79`:

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
    mixer_initialization: uniform
    mixer_uniform_ratio: 0.3
```

The only difference from `exp79` is `mixer_initialization: uniform` with `mixer_uniform_ratio: 0.3`.

The uniform initialization constructs a doubly stochastic matrix with diagonal `1 - uniform_ratio` and uniform off-diagonal entries `uniform_ratio / (num_channels - 1)`. For `ratio: 0.3` and `num_heads: 4`, the initial projected matrix is:

```text
diag = 0.7,  off-diag = 0.1
```

This places the initial mixer between the near-identity warm start of `exp79` (`diag ≈ 1.0`) and the diffuse random start of `exp80` (`diag ≈ 0.25`), providing a controlled amount of initial cross-head communication.

The run environment records Git commit `36c31fbdb29f84148a5a7a151d4d8cb54b277384` and `git_dirty: true`. Data fingerprint, normalizer fingerprint, and split fingerprints match `exp73`, `exp79`, and `exp80`.

### Results

| Run | Architecture | Mixer init | Best validation loss | Best epoch | Final validation loss |
|---|---|---|---:|---:|---:|
| `exp77` | Output-only, unconstrained | identity-biased learned | `0.02544` | 993 | `0.03228` |
| `exp73` | Output-only, doubly stochastic | identity warm-start | `0.01584` | 961 | `0.01857` |
| `exp78` | Single LSTM, width 20 | — | `0.00772` | 958 | `0.00929` |
| `exp79` | Layerwise, doubly stochastic | identity warm-start | **`0.00705`** | 913 | **`0.00893`** |
| `exp80` | Layerwise, doubly stochastic | random | `0.04832` | 968 | `0.05208` |
| `exp81` | Layerwise, doubly stochastic | **uniform 0.3** | **`0.03702`** | 970 | **`0.03854`** |

Relative to the layerwise identity baseline (`exp79`), `exp81` degrades best validation loss by a factor of `5.2×` (`0.00705 → 0.03702`). Relative to the random-init ablation (`exp80`), `exp81` improves by `23%` (`0.04832 → 0.03702`).

`exp81` also falls short of the output-only doubly stochastic baseline (`exp73`, `0.01584`) by a factor of `2.3×`, and is approximately `4.1×` worse than the single-LSTM reference (`exp78`).

### Training trajectory

| Epoch | `exp79` identity-init | `exp81` uniform-init (0.3) | `exp80` random-init |
|---:|---:|---:|---:|
| 1 | `1.01938 / 1.60837` | `1.09158 / 2.00298` | `0.96333 / 2.02788` |
| 10 | `0.35377 / 0.48669` | `0.37528 / 0.54623` | `0.35935 / 0.49358` |
| 20 | `0.23027 / 0.30023` | `0.23747 / 0.31838` | `0.24147 / 0.31370` |
| 50 | `0.17252 / 0.20267` | **`0.14928 / 0.17467`** | `0.19930 / 0.23910` |
| 100 | `0.05885 / 0.08763` | `0.08820 / 0.11610` | `0.09255 / 0.13421` |
| 250 | `0.01197 / 0.03041` | `0.04134 / 0.08524` | `0.05480 / 0.07405` |
| 500 | `0.00604 / 0.01812` | `0.02919 / 0.05685` | `0.04501 / 0.06538` |
| 750 | `0.00496 / 0.01850` | `0.02724 / 0.04685` | `0.04617 / 0.05664` |
| 1000 | `0.00469 / 0.00893` | `0.02774 / 0.03854` | `0.03605 / 0.05208` |

`exp81` shows a distinct two-phase pattern:

1. **Early phase (epochs 1–50):** `exp81` outperforms both `exp79` and `exp80` on validation loss. The uniform initial coupling appears to act as a useful regularization or signal-sharing prior during the first specialization stage.
2. **Late phase (epochs 100+):** `exp79` continues to improve through the second cosine cycle, while `exp81` enters a plateau. By epoch 250 the gap is already large (`0.030` vs `0.085`), and `exp81` makes only marginal progress after epoch 500 (`0.057 → 0.039`).

### Mixer inspection

The initial uniform-projected matrix has `diag = 0.7` and `off-diag = 0.1` for every entry, yielding an initial off-diagonal mass of `1.20`.

The final `exp81` mixer (epoch 999):

```text
P_exp81 =
[[0.713501, 0.082055, 0.094082, 0.110362],
 [0.076272, 0.774637, 0.072041, 0.077050],
 [0.099441, 0.072734, 0.742409, 0.085416],
 [0.110786, 0.070573, 0.091469, 0.727172]]
```

| Run | Mixer init | `||P - I||_F` | `||P - I||_2` | Total off-diagonal mass | Diagonal range |
|---|---:|---:|---:|---:|---:|
| `exp73` | identity | `3.05e-4` | `1.87e-4` | `5.26e-4` | `[0.999857, 0.999881]` |
| `exp79` | identity | `2.62e-4` | `1.77e-4` | `4.43e-4` | `[0.999861, 0.999927]` |
| `exp80` | random | `1.58` | `1.15` | `2.61` | `[0.226, 0.525]` |
| `exp81` | uniform 0.3 | **`6.05e-1`** | **`3.92e-1`** | **`1.04`** | `[0.714, 0.775]` |

The final `exp81` mixer has moved only slightly from its initialization: diagonal entries rose from `0.700` to `0.713–0.775`, and total off-diagonal mass fell from `1.20` to `1.04` (a `13%` reduction). The movement is far smaller than would be needed to approach identity, and it appears to stabilize early: from epoch 219 onward the matrix changes only in the fourth decimal place.

This supports the hypothesis that once head specialization has occurred under a non-identity coupling, the mixer and the head parameters become mutually locked. Gradients that would move the mixer toward identity also require the heads to undo their learned coupled representations, making the joint escape costly.

### Interpretation

1. **Initial cross-head coupling is not monotonically harmful, but it has a ceiling.** `exp81` outperforms `exp79` before epoch 50, yet ends up `5.2×` worse. A moderate initial blend (`diag = 0.7`) accelerates early signal sharing but appears to prevent the heads from ever reaching the depth of independent temporal specialization that `exp79` achieves.

2. **The optimization landscape is likely continuous rather than a sharp two-basin problem.** `exp81` sits cleanly between `exp79` and `exp80` on both final loss and final mixer distance. There is no evidence of a sudden phase transition between "identity basin" and "random basin"; instead, performance degrades roughly monotonically with the initial off-diagonal mass.

3. **The mixer trajectory matters more than the final matrix.** Both `exp73` and `exp79` end near identity, yet `exp79` is dramatically better. The difference is that `exp79` starts at identity and explores only a tiny neighborhood, while `exp81` starts far from identity and cannot recover. This suggests that **the path taken during the first ~50 epochs**, not the final checkpoint, determines whether the model can reach the high-performance regime.

4. **"Delayed coupling" remains the leading explanation for the identity-init advantage.** `exp79` allows heads to specialize independently during the critical early phase, then introduces only subtle cross-head routing. `exp81` and `exp80` force coupling before useful head representations exist, and the resulting coupled representations are suboptimal even after 1,000 epochs of further training.

---

## UPDATE: Distinct per-layer mixers (exp84)

### Setup

`exp84` tests whether the shared mixer in `exp79` is a bottleneck. In `exp79`, the same doubly stochastic matrix is applied after every recurrent layer, including both the layer-1-to-layer-2 boundary and the final layer-to-fusion boundary. `exp84` instead gives every recurrent layer output its own independently learned mixer.

```yaml
model:
  family: forecast
  name: multihead-lstm-distinct
  parameters:
    hidden_dim: 32
    num_heads: 4
    num_layers: 2
    dropout: 0.0
    sinkhorn_iterations: 20
    mixer_initialization: identity
    mixer_random_std: 1.0
    mixer_uniform_ratio: 0.0
    use_distinct_mixers: true
```

For `num_layers: 2`, this creates:

```text
per-head LSTM layer 1
→ mixer 0 (layer 1 → layer 2)
→ per-head LSTM layer 2
→ mixer 1 (layer 2 → fusion)
→ flatten → LayerNorm → Linear → forecast
```

Both mixers are independent `4 × 4` doubly stochastic matrices with identity warm starts. The dataset lock, dimensions, optimizer, scheduler, and epoch budget match `exp79`. The run environment records Git commit `949fcdfb5c62b1302410927f36c2150090b28497` and `git_dirty: true`.

### Results

| Run | Mixer structure | Best validation loss | Best epoch | Final validation loss |
|---|---|---:|---:|---:|
| `exp73` | Output-only, one constrained mixer | `0.01584` | 961 | `0.01857` |
| `exp79` | Layerwise, one shared constrained mixer | **`0.00705`** | 913 | **`0.00893`** |
| `exp84` | Layerwise, two distinct constrained mixers | `0.01714` | 990 | `0.02773` |

`exp84` did not reproduce the shared-mixer result: its best validation loss is `2.4×` higher than `exp79` (`0.00705 → 0.01714`). It is, however, close to the output-only constrained baseline `exp73` (`0.01584`).

The final validation loss is materially above the selected best checkpoint (`0.02773` vs `0.01714`), so comparisons should use best validation loss for model selection and retain final loss as a trajectory observation.

### Training trajectory

| Epoch | `exp79` shared mixer train / validation | `exp84` distinct mixers train / validation |
|---:|---:|---:|
| 1 | `1.01938 / 1.60837` | `0.84076 / 1.26471` |
| 10 | `0.35377 / 0.48669` | `0.33349 / 0.46478` |
| 20 | `0.23027 / 0.30023` | `0.23313 / 0.30554` |
| 50 | `0.17252 / 0.20267` | `0.17893 / 0.20772` |
| 100 | `0.05885 / 0.08763` | `0.05278 / 0.10102` |
| 250 | `0.01197 / 0.03041` | `0.02463 / 0.03626` |
| 500 | `0.00604 / 0.01812` | `0.01615 / 0.02861` |
| 750 | `0.00496 / 0.01850` | `0.01340 / 0.02290` |
| 1000 | `0.00469 / 0.00893` | `0.01592 / 0.02773` |

The runs are similar through epoch 50. `exp84` then develops a worse validation trajectory: at epoch 100 it has lower training loss but higher validation loss, and it remains behind `exp79` through both cosine cycles. At its best epoch 990, `exp84` reaches train / validation `0.01567 / 0.01714`; its train-validation gap is small at that selected checkpoint, so the main difference from `exp79` is not a large late train/validation split but a higher achievable loss floor.

### Mixer inspection

The best checkpoint contains two independent near-identity matrices:

```text
P_exp84_layer_1_to_2 =
[[0.999890, 0.000046, 0.000062, 0.000018],
 [0.000038, 0.999876, 0.000044, 0.000042],
 [0.000034, 0.000038, 0.999860, 0.000036],
 [0.000039, 0.000040, 0.000034, 0.999903]]

P_exp84_layer_2_to_fusion =
[[0.999896, 0.000070, 0.000062, 0.000072],
 [0.000039, 0.999843, 0.000050, 0.000051],
 [0.000031, 0.000045, 0.999844, 0.000046],
 [0.000034, 0.000042, 0.000043, 0.999831]]
```

| Mixer | `||P - I||_F` | `||P - I||_2` | `max(abs(P - I))` | Total off-diagonal mass | Diagonal range |
|---|---:|---:|---:|---:|---:|
| `exp79` shared final | `2.62e-4` | `1.77e-4` | `1.39e-4` | `4.43e-4` | `[0.999861, 0.999927]` |
| `exp84` layer 1 → layer 2 | `2.76e-4` | `1.80e-4` | `1.40e-4` | `4.72e-4` | `[0.999860, 0.999903]` |
| `exp84` layer 2 → fusion | `3.45e-4` | `2.11e-4` | `1.69e-4` | `5.86e-4` | `[0.999831, 0.999896]` |

Both distinct mixers remain near identity. The final-output mixer has approximately 24% more off-diagonal mass than the inter-layer mixer (`5.86e-4` vs `4.72e-4`), which is directionally consistent with allowing slightly more final fusion than recurrent-state communication. The absolute difference is still very small.

The final checkpoint shows essentially the same matrices as the best checkpoint, so the late validation degradation is not accompanied by a material mixer change.

### Interpretation

1. **This run does not support the shared-mixer bottleneck hypothesis.** Giving the two positions independent matrices did not produce substantial layer-specific routing and did not improve validation performance. Both matrices selected near-identity routing.

2. **Shared parameters may be a useful regularizer or optimization aid.** `exp84` adds only one extra `4 × 4` logit matrix, but it has a higher loss floor than `exp79`. One interpretation is that requiring the same near-identity routing at both positions reduces degrees of freedom in a beneficial way. This is a single-run observation, not a general conclusion.

3. **Separate matrices alone do not solve the feature-space problem.** Each mixer still combines matching coordinates across independently parameterized heads. If those coordinates are not semantically aligned, making the matrices position-specific cannot create the missing cross-feature translation mechanism.

4. **The result motivates dense latent communication rather than further coordinate-wise mixer variants.** A layerwise `Linear(hidden_dim, hidden_dim)` communication block can map any source-head feature to any destination-head feature while preserving a strict identity initialization. This is implemented as the separate `multihead-lstm-communication` model and should be evaluated independently of `exp84`.

---

## UPDATE: Dense inter-layer communication (exp86)

### Setup

`exp86` replaces coordinate-wise head mixing with a dense identity-initialized communication transform at each true layer boundary. For the current `hidden_dim: 32`, `num_heads: 4`, and `head_dim: 8` configuration, it applies:

```text
per-head LSTM layer 1
→ stack: [B, T, 4, 8]
→ flatten: [B, T, 32]
→ Linear(32, 32), initialized as W = I and b = 0
→ reshape: [B, T, 4, 8]
→ per-head LSTM layer 2
→ flatten → existing LayerNorm + Linear fusion → forecast
```

There is no doubly stochastic mixer and no extra final-output communication transform. The dense transform exists only at real layer boundaries; a one-layer model would have none. This makes the communication block initially an exact identity while allowing any source-head feature to influence any destination-head feature during training.

```yaml
model:
  family: forecast
  name: multihead-lstm-communication
  parameters:
    hidden_dim: 32
    num_heads: 4
    num_layers: 2
    dropout: 0.0
```

The dataset and normalizer fingerprints, split fingerprints, optimizer, scheduler, and epoch budget match the preceding experiments. The run environment records Git commit `f5b72036d0c909d44c92fb52f5055fa27b52815e` and `git_dirty: true`.

### Results

| Run | Inter-layer mechanism | Best validation loss | Best epoch | Final validation loss |
|---|---|---:|---:|---:|
| `exp79` | Shared doubly stochastic coordinate mixer | **`0.00705`** | 913 | **`0.00893`** |
| `exp73` | No inter-layer communication; output-only constrained mixer | `0.01584` | 961 | `0.01857` |
| `exp84` | Distinct doubly stochastic coordinate mixers | `0.01714` | 990 | `0.02773` |
| `exp86` | Dense identity-init `Linear(32, 32)` | `0.02003` | 995 | `0.02349` |

`exp86` is stable and substantially better than the random/uniform coordinate-mixing ablations, but it does not exceed the constrained output-only baseline: its best validation loss is `1.26×` `exp73` (`0.01584 → 0.02003`).

### Training trajectory

| Epoch | `exp73` output-only | `exp79` shared mixer | `exp86` dense communication |
|---:|---:|---:|---:|
| 1 | `0.92322 / 1.88706` | `1.01938 / 1.60837` | `0.93266 / 1.68291` |
| 10 | `0.28866 / 0.42223` | `0.35377 / 0.48669` | `0.27066 / 0.37800` |
| 20 | `0.23186 / 0.30205` | `0.23027 / 0.30023` | `0.22059 / 0.26535` |
| 50 | `0.19329 / 0.23532` | `0.17252 / 0.20267` | `0.12169 / 0.18595` |
| 100 | `0.07679 / 0.09971` | `0.05885 / 0.08763` | `0.06846 / 0.09383` |
| 250 | `0.03399 / 0.05075` | `0.01197 / 0.03041` | `0.03125 / 0.04433` |
| 500 | `0.02239 / 0.03279` | `0.00604 / 0.01812` | `0.02282 / 0.03546` |
| 750 | `0.01782 / 0.02658` | `0.00496 / 0.01850` | `0.02119 / 0.02846` |
| 1000 | `0.01646 / 0.01857` | `0.00469 / 0.00893` | `0.01399 / 0.02349` |

Dense communication is initially competitive: through epoch 250 it has lower validation loss than `exp73`. It does not sustain that advantage through the cosine cycles, finishing with lower train loss but higher validation loss than `exp73`.

### Communication inspection

Unlike the previous mixers, the final communication matrix is strongly non-identity:

| Checkpoint | `||W - I||_F` | `||W - I||_2` | `max(abs(W - I))` | `||b||_2` |
|---|---:|---:|---:|---:|
| best (epoch 995) | `2.4334` | `1.9296` | `0.5379` | `0.2162` |
| final (epoch 1000) | `2.4360` | `1.9311` | `0.5389` | `0.2163` |

Partitioning the final `32 × 32` matrix into `4 × 4` blocks of size `8 × 8`, the following are Frobenius norms. Diagonal entries are `||W_ii - I||_F`; off-diagonal entries are `||W_ij||_F`, with rows indexing destination heads and columns source heads:

```text
[[0.524, 0.530, 0.646, 0.502],
 [0.460, 0.520, 0.646, 0.525],
 [0.526, 0.508, 0.627, 0.480],
 [0.702, 1.043, 0.756, 0.471]]
```

The mean cross-head block norm is `0.610`, larger than the mean head-local deviation from identity (`0.536`). Thus the model did not merely reparameterize individual heads: it learned substantial cross-head, cross-feature transforms. Best and final matrices are nearly unchanged, so this topology stabilizes before the end of training.

### Interpretation

1. **The model can and does learn strong cross-head communication when unconstrained dense feature mixing is available.** The near-identity outcome of the doubly stochastic `P` matrices therefore does not establish that the task rejects all communication.

2. **Strong communication alone is not sufficient for good validation performance.** `exp86` learns substantial non-identity cross-head blocks but has a worse final validation floor than `exp73` and is far from `exp79`. The dense transform is likely too unconstrained: it can mix, rescale, rotate, and bias local and cross-head features simultaneously.

3. **Feature translation remains a useful motivation, but dense mixing confounds translation with routing and local reparameterization.** The next model should separate source-head translation, head-level routing, destination-head decoding, and total communication strength.

---

## UPDATE: Gated latent head communication (exp87)

### Setup

`exp87` separates feature translation from head routing. At the layer-1-to-layer-2 boundary, each source head maps its local state to a communication latent through a head-specific MLP. Each destination head uses a static masked-softmax distribution over the *other* source heads, decodes the resulting latent message through its own MLP, and injects it as a small residual update.

```text
source head state h_j
→ source-specific encoder E_j
→ latent z_j
→ receiver-specific masked-softmax routing α[i ← j], with j != i
→ destination message m_i
→ destination-specific decoder D_i
→ h_i + sigmoid(g_i) ⊙ D_i(m_i)
→ LSTM layer 2
```

```yaml
model:
  family: forecast
  name: multihead-lstm-latent-communication
  parameters:
    hidden_dim: 32
    num_heads: 4
    num_layers: 2
    dropout: 0.0
    communication_dim: 8
    communication_gate_initial_logit: -5.0
```

Self-routes are masked. Initial routing is uniform over the other three heads, and `sigmoid(-5) ≈ 0.00669` makes initial message injection small but nonzero, preserving gradient flow to encoders, decoders, gates, and routing logits. No doubly stochastic matrix is used. Dataset, normalizer, split locks, optimizer, scheduler, epoch budget, and run environment match `exp86`.

### Results

| Run | Communication mechanism | Best validation loss | Best epoch | Final validation loss |
|---|---|---:|---:|---:|
| `exp79` | Shared doubly stochastic coordinate mixer | **`0.00705`** | 913 | **`0.00893`** |
| `exp78` | Single LSTM, width 20 | `0.00772` | 958 | `0.00929` |
| `exp87` | Latent translation + masked-softmax routing + gated residual | `0.01089` | 945 | `0.01223` |
| `exp73` | Output-only constrained mixer | `0.01584` | 961 | `0.01857` |
| `exp86` | Dense inter-layer communication | `0.02003` | 995 | `0.02349` |

Relative to `exp73`, `exp87` reduces best validation loss by `31.2%` (`0.01584 → 0.01089`). It also improves over dense communication `exp86` by `45.6%` (`0.02003 → 0.01089`). It remains `1.55×` above `exp79` and `1.41×` above the single-LSTM reference on this one trajectory.

### Training trajectory

| Epoch | `exp73` output-only | `exp79` shared mixer | `exp86` dense communication | `exp87` latent communication |
|---:|---:|---:|---:|---:|
| 1 | `0.92322 / 1.88706` | `1.01938 / 1.60837` | `0.93266 / 1.68291` | `0.94522 / 1.77485` |
| 10 | `0.28866 / 0.42223` | `0.35377 / 0.48669` | `0.27066 / 0.37800` | `0.36422 / 0.48531` |
| 20 | `0.23186 / 0.30205` | `0.23027 / 0.30023` | `0.22059 / 0.26535` | `0.23548 / 0.29458` |
| 50 | `0.19329 / 0.23532` | `0.17252 / 0.20267` | `0.12169 / 0.18595` | `0.19971 / 0.22616` |
| 100 | `0.07679 / 0.09971` | `0.05885 / 0.08763` | `0.06846 / 0.09383` | `0.09077 / 0.11932` |
| 250 | `0.03399 / 0.05075` | `0.01197 / 0.03041` | `0.03125 / 0.04433` | `0.02764 / 0.04636` |
| 500 | `0.02239 / 0.03279` | `0.00604 / 0.01812` | `0.02282 / 0.03546` | `0.01587 / 0.02759` |
| 750 | `0.01782 / 0.02658` | `0.00496 / 0.01850` | `0.02119 / 0.02846` | `0.01202 / 0.02114` |
| 1000 | `0.01646 / 0.01857` | `0.00469 / 0.00893` | `0.01399 / 0.02349` | `0.01267 / 0.01223` |

`exp87` does not win early: it remains behind `exp73` at epoch 100 and is close at epoch 250. Its main advantage emerges in the second cosine cycle, where it continues improving from `0.02759` at epoch 500 to `0.01223` at epoch 1000. This contrasts with `exp86`, whose early advantage does not translate to a lower final validation floor.

### Routing and gate inspection

The best checkpoint at epoch 945 learns the following receiver-by-source routing matrix; rows are destination heads, columns source heads, and self-routes are masked to zero:

```text
α_exp87 =
[[0.000000, 0.221056, 0.530484, 0.248459],
 [0.217126, 0.000000, 0.605853, 0.177021],
 [0.315508, 0.339333, 0.000000, 0.345159],
 [0.226404, 0.274482, 0.499114, 0.000000]]
```

The final checkpoint changes this only slightly:

```text
α_exp87_final =
[[0.000000, 0.212096, 0.544039, 0.243865],
 [0.210677, 0.000000, 0.616334, 0.172989],
 [0.314989, 0.339954, 0.000000, 0.345057],
 [0.220428, 0.273329, 0.506244, 0.000000]]
```

Three receivers (heads 0, 1, and 3) primarily route from head 2, with maximum source weights `0.53`, `0.61`, and `0.50`; head 2 remains near-uniform over the remaining heads. Head indices do not have prespecified semantic meanings, so this is evidence of a stable asymmetric communication topology, not an identification of a particular frequency role.

| Gate statistic | Initial | Best checkpoint | Final checkpoint |
|---|---:|---:|---:|
| Minimum | `0.00669` | `0.00614` | `0.00614` |
| Mean | `0.00669` | `0.00728` | `0.00733` |
| Maximum | `0.00669` | `0.00952` | `0.00950` |

The routing logits become clearly non-uniform, whereas gates remain small. This supports conservative, receiver-selective message injection. A small gate alone does not bound the actual message magnitude because decoders may rescale messages; direct measurement of `||gate ⊙ D(m)|| / ||h||` requires the planned probe system.

### Interpretation

1. **Head communication is useful when feature translation, routing, and message injection are separated.** `exp87` substantially improves over output-only mixing and dense unconstrained communication, while learning an explicit stable routing graph.

2. **Receiver-selective routing avoids a limitation of doubly stochastic head mixing.** A useful source head may broadcast to multiple destinations: head 2 is the dominant source for three receivers. Column-stochastic mass conservation would restrict this pattern.

3. **The small residual gate is a plausible regularizer.** In contrast to the strongly non-identity dense transform in `exp86`, `exp87` preserves a small communication injection while allowing encoders and decoders to learn feature translation. This is consistent with its better second-cycle validation trajectory, but does not establish causality.

4. **`exp79` remains unexplained.** The latent design is more expressive and produces directly observable communication, but it does not reproduce the shared near-identity mixer's `0.00705` result. One seed per condition remains a material confound.

---
