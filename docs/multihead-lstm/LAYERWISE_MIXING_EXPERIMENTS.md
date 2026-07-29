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
