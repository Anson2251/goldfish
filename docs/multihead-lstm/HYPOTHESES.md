# Multi-Head LSTM Mixer: Working Hypotheses

This document records the explanatory hypotheses generated during the multi-head LSTM communication ablation sequence (exp73–exp87). Each hypothesis is stated, linked to evidence that supports or contradicts it, and tagged with its current epistemic status and the experiments needed to validate or falsify it.

---

## 1. Gradient-channel hypothesis

**Statement:** The gain from the layerwise mixer comes primarily from *back-propagated gradient coupling* across heads at every layer boundary, not from the *forward* cross-head feature routing visible in the final checkpoint.

**Rationale:** The final `exp79` mixer is near-identity (`off-diag mass ≈ 4e-4`), yet the layerwise architecture outperforms the output-only architecture by 55%. Even tiny off-diagonal weights allow gradients to flow between heads during training, which may let each LSTM head learn complementary temporal frequencies without requiring large final routing weights.

**Evidence:**
- `exp79` final mixer ≈ identity but performance ≫ `exp73` output-only.
- `exp80` random init learns a stable non-identity mixer but performs poorly; the model does not spontaneously recover identity-like routing.
- `exp81` uniform 0.3 starts at `diag=0.7` and moves only to `diag≈0.74`; if forward routing were the only mechanism, one would expect stronger movement toward identity given the large performance penalty.
- `exp87` learns non-uniform routing while retaining small residual gates, showing that explicit forward routing can coexist with conservative state injection. It does not by itself establish whether the gain is forward-message, gradient-path, or joint-training driven.

**Contradictions / open questions:**
- If gradient coupling is the key mechanism, freezing the mixer or communication block after epoch N should cause performance to collapse when N is small but remain stable when N is large. No such experiment has been run.
- `exp104` (same architecture and configuration as `exp79`, only with probes) receives gradients on its mixer logits throughout training and moves them (`0 → 0.41` logits distance), yet performs `3.0×` worse than `exp79`. A live gradient path is therefore not sufficient to produce the `exp79` gain. If gradient coupling matters, it requires a specific head-initialization path to act on.

**Status:** Weakened by `exp104`; a live mixer gradient path exists without reproducing the gain.

**Validation experiments:**
1. Freeze mixer at epoch 50, 100, 250, 500; compare final performance. If early freeze ≈ bad and late freeze ≈ good, gradient coupling during early training is critical.
2. Compare `exp79` against a variant where mixer gradients are blocked (stop-gradient through mixer, or fix mixer while allowing LSTM heads to train). If fixed-mixer fails, the *learnability* of the coupling is part of the mechanism.

---

## 2. Premature information-bottleneck hypothesis

**Statement:** The problem with random and uniform initializations is not that cross-head communication is inherently bad, but that *too much communication before heads have learned useful representations* creates an irreversible information bottleneck. Each head leaks 75% (random) or 30% (uniform 0.3) of its signal to other heads before any useful specialization has occurred.

**Rationale:** `exp80` epoch-1 validation loss (`2.028`) is worse than `exp79` (`1.608`) despite similar training loss, suggesting cross-head averaging immediately hurts generalization. `exp81` epoch-1 validation (`2.003`) follows the same pattern, only slightly better than random.

**Evidence:**
- `exp80` and `exp81` both show worse validation than `exp79` from epoch 1.
- `exp81` outperforms `exp79` at epoch 50 (val `0.175` vs `0.203`), suggesting moderate early coupling can act as regularization, but the ceiling is lower.
- Final `exp81` mixer has only moved 13% toward identity from its initialization, suggesting the model cannot undo the early coupling structure.
- `exp87` begins with uniform routing over non-self sources, but its residual gate is only `sigmoid(-5) ≈ 0.0067`. Its eventual improvement over `exp73` shows that a uniform *routing topology* is not necessarily harmful when the actual injected message is initially small.

**Contradictions / open questions:**
- `exp81` early advantage contradicts the stronger version of this hypothesis ("any early coupling is harmful"). The evidence now distinguishes routing distribution from total injection magnitude: the harmful variable may be large early injected messages, not non-identity routing alone.

**Status:** Partially supported; revised to focus on early communication magnitude rather than routing topology alone.

**Validation experiments:**
1. Scheduled mixer initialization: start with `M=I` for first N epochs, then switch to learnable. If delayed coupling recovers `exp79`-level performance, the hypothesis is strongly supported.
2. Very small uniform ratios (`0.01`, `0.05`) to map the dose-response curve.

---

## 3. Optimization-landscape hypothesis (revised: continuous penalty)

**Original statement (three-basin):** The optimization landscape contains three attractor basins with distinct off-diagonal mass: near-0 (identity, optimal), ~1.0 (uniform 0.3, suboptimal), ~2.6 (random, worst).

**Revised statement:** The landscape is more likely a *continuous* function of initial off-diagonal mass, where performance degrades roughly monotonically as the initial mixer deviates from identity. There may be local minima, but no sharp phase transition between "good" and "bad" basins.

**Evidence:**
- `exp79` (`off-diag mass ≈ 0`) → best.
- `exp81` (`off-diag mass ≈ 1.2→1.0`) → intermediate (`0.037`).
- `exp80` (`off-diag mass ≈ 2.4→2.6`) → worst (`0.048`).
- `exp81` sits cleanly between the two extremes on both loss and mixer distance.

**Contradictions / open questions:**
- The gap from `exp81` (`0.037`) to `exp79` (`0.007`) is larger than the gap from `exp80` (`0.048`) to `exp81` (`0.037`). The penalty may be *non-linear* or there may be a sharper drop somewhere between `off-diag mass = 0` and `1.0`.

**Status:** Supported for the broad shape; finer resolution between 0 and 1.0 needed.

**Validation experiments:**
1. Uniform ratio sweep: `0.01`, `0.05`, `0.1`, `0.2`. These will reveal whether the drop is smooth or has a threshold.
2. Seed sweep for each ratio to confirm basin stability.

---

## 4. Representation-locking hypothesis

**Statement:** Once LSTM heads have specialized under a particular mixer configuration, the mixer and the head parameters become *mutually locked*. Gradients that would move the mixer toward a better configuration are resisted because the heads would simultaneously need to unlearn their existing coupled or decoupled representations.

**Rationale:** `exp81` mixer moves only 13% toward identity over 1,000 epochs despite a massive performance penalty. `exp80` mixer barely moves at all. This is not because the optimizer is lazy; training loss continues to decrease. The joint parameter space appears to have a flat or uphill direction for the mixer once the heads have settled.

**Evidence:**
- `exp81` off-diag mass: `1.20 → 1.04` over 1,000 epochs.
- `exp80` off-diag mass: `2.4 → 2.6` (slightly away from identity).
- Both models make progress on training loss, so the optimizer is not stuck globally; it is stuck in a mixer-head coupled subspace.

**Contradictions / open questions:**
- If locking is strong, how does `exp79` ever escape? Answer: `exp79` starts inside the good basin, so it never needs to escape.

**Status:** Strongly supported by `exp81` and `exp80` mixer trajectories.

**Validation experiments:**
1. Reset mixer to identity at epoch 250 or 500 while keeping LSTM head parameters. If performance improves, locking was the barrier.
2. Train with a very small learning rate on mixer logits only (or use a two-time-scale update) to see if the mixer can move when head parameters are nearly frozen.

---

## 5. Delayed-coupling hypothesis

**Statement:** The identity warm-start succeeds because it enforces a *temporal schedule*: heads specialize independently during the critical first ~50 epochs, and only then introduce subtle cross-head routing. The final near-identity checkpoint is the *result* of a successful schedule, not evidence that the mixer is unused.

**Rationale:** `exp79` outperforms `exp81` at epoch 50 despite `exp81` having better early validation loss. The second cosine cycle (epochs 500–1000) brings `exp79` from `0.018` to `0.009`, while `exp81` plateaus at `0.057→0.039`. The independent specialization achieved during the first cycle seems to enable deeper refinement in the second cycle.

**Evidence:**
- `exp79` epoch-50 val: `0.203` (worse than `exp81`'s `0.175`), but epoch-1000 val: `0.009` (far better than `exp81`'s `0.039`).
- `exp79` final mixer ≈ identity, but layerwise placement still matters. This is consistent with "the mixer was useful during training, then settled back."

**Contradictions / open questions:**
- We do not have intermediate mixer checkpoints for `exp79`. We cannot verify whether the mixer actually deviated from identity during epochs 1–250.
- `exp104` records the full mixer trajectory of the same architecture and configuration: the projected matrix never leaves the near-identity neighborhood (off-diagonal mass `5.5e-4 → 5.0e-4` over 1,000 epochs). No early routing excursion occurs. Since `exp104` does not reproduce the `exp79` gain, the delayed-coupling story is not supported by the only observed trajectory.

**Status:** Not observed on the `exp104` trajectory; substantially weakened as an explanation for the `exp79` advantage, though the `exp79` trajectory itself remains unrecorded.

**Validation experiments:**
1. **Highest priority:** Reconstruct mixer trajectory from saved intermediate checkpoints, or re-run `exp79` with periodic checkpoint saving. If the mixer off-diag mass was ever > 0.01 during training, this hypothesis is strongly supported.
2. Scheduled coupling: `M=I` for first N epochs, then switch to learnable. N=50, N=100, N=250.

---

## 6. Shared-mixer constraint hypothesis

**Statement:** The shared mixer may have forced a compromise between layer-boundary communication and final fusion, causing near-identity convergence.

**Evidence against:**
- `exp84` directly tested this by using distinct matrices for layer 1→2 and layer 2→fusion.
- Both matrices remained near identity; the final mixer had slightly more off-diagonal mass, but neither learned substantial routing.
- `exp84` best validation loss (`0.01714`) was materially worse than shared-mixer `exp79` (`0.00705`).

**Status:** Weakened by direct evidence. Shared-mixer tying is not established as a bottleneck; it may instead be a useful regularizer or a favorable optimization constraint. This conclusion remains limited to one trajectory.

---

## 7. Explicit single-layer stacking hypothesis

**Statement:** A portion of the `exp79` gain may come from the *implementation difference* between PyTorch's native `nn.LSTM(num_layers=2)` (`exp73` output-only) and explicit `ModuleList` stacking of two `nn.LSTM(num_layers=1)` modules (`exp79` layerwise), rather than from the mixer itself. Native multi-layer LSTM and stacked single-layer LSTMs differ in parameter initialization, gradient paths, and internal state handling.

**Rationale:** `exp73` and `exp79` differ in two ways simultaneously: (1) mixer position, and (2) LSTM implementation. We have not isolated which factor dominates.

**Evidence:**
- `exp73` uses `nn.LSTM(..., num_layers=2)` per head.
- `exp79` uses `nn.ModuleList([nn.LSTM(..., num_layers=1), nn.LSTM(..., num_layers=1)])` per head.
- PyTorch's native multi-layer LSTM applies dropout between layers internally and uses a fused backward path; the stacked version does not.
- `exp84`, `exp86`, and `exp87` all also use explicit stacked one-layer LSTMs but do not reproduce `exp79`. This weakens the claim that stacking alone is sufficient, but it does not isolate stacking because their communication mechanisms differ.

**Contradictions / open questions:**
- If the LSTM implementation change explains the gain, then a single-head layerwise model without any mixer should also perform well.

**Status:** Still unresolved, but stacking alone is unlikely to explain all of `exp79`.

**Validation experiments:**
1. `num_heads=1, head_dim=32` layerwise LSTM with no mixer (or fixed `M=I` acting as pure identity). If performance ≈ `exp79`, the gain is mostly from stacked-single-layer implementation.
2. Alternatively, modify `exp73` to use stacked single-layer LSTMs but keep the mixer output-only. If this closes the gap to `exp79`, the LSTM implementation is the primary factor.

---

## 8. Learnability-as-freedom hypothesis

**Statement:** Even if the final mixer is near-identity, the fact that it is *learnable* (not fixed) provides a useful optimization degree of freedom. The optimizer uses the mixer logits as a "soft anchor" around identity, allowing tiny deviations that shape the gradient landscape for the LSTM heads. The mixer does not need to learn large routing weights to be useful; its mere existence as a differentiable parameter may regularize or stabilize the training dynamics.

**Rationale:** `exp79` and `exp73` both end near-identity, but `exp79` is dramatically better. If the mixer were truly useless, one would expect `exp79` ≈ `exp73` ≈ "layerwise no-mixer." The gap suggests the learnable mixer matters even when it barely moves.

**Evidence:**
- `exp79` (learnable, layerwise) vs `exp73` (learnable, output-only): large gap.
- `exp81` (learnable, but starts far from identity): cannot recover. Learnability does not guarantee recovery from a bad basin.

**Contradictions / open questions:**
- If learnability alone is the key, then `M=I` fixed should perform similarly to `exp79`. If it does not, the *deviation* (however small) is also important.

**Status:** Needs direct comparison with fixed-identity control.

**Validation experiments:**
1. `M=I` fixed at every layer, no learnable mixer parameters. Compare to `exp79`.
2. If fixed-identity performs poorly, both the learnability and the small deviation matter.
3. If fixed-identity performs well, the gain may be from stacked-layer implementation (hypothesis 7).

---

## 9. Residual / implicit-regularization hypothesis

**Statement:** The mixer acts as a soft residual branch. Even when `P ≈ I`, the learnable off-diagonal terms provide a "correction channel" that the optimizer can use to cancel out or amplify specific head signals. This is analogous to how a ResNet skip-connection (`+ x`) is critical even though the residual branch often learns small weights.

**Rationale:** In ResNets, the final residual weights are often small, yet removing the skip connection destroys performance. Similarly, the mixer provides a structural path for gradient flow that is orthogonal to the main LSTM feedforward path.

**Evidence:**
- Indirect analogy only; no direct evidence in this architecture.

**Status:** Speculative.

**Validation experiments:**
1. Hard to test directly without a skip-connection ablation that would break the architecture.
2. Indirect test: measure the gradient magnitudes flowing through mixer logits vs LSTM weights during training. If mixer gradients are small but non-zero and correlate with head-gradient alignment, the residual analogy gains support.

---

## 10. Run-to-run variation hypothesis

**Statement:** The observed advantage of `exp79` over the single-LSTM reference (`exp78`) is small (`0.00705` vs `0.00772`, ~9%) and could be ordinary run-to-run initialization or training stochasticity. The larger gaps (`exp79` vs `exp73`, 55%; vs `exp80`, 6.9×) are more robust, but the claim that "multi-head layerwise matches or exceeds single LSTM" should be treated cautiously.

**Rationale:** All experiments use `seed: null` and `deterministic: false`. No seed sweep has been performed for any condition.

**Evidence:**
- `exp79` vs `exp78` is the only comparison where the gap is small enough to be within typical seed variation.
- All other comparisons are large enough to be robust to seed jitter.
- `exp104` is a direct same-architecture, same-configuration rerun of `exp79` (verified equivalent by checkpoint reproduction: both `exp79` checkpoints load strictly into HEAD code and reproduce reported losses to within `3e-5`). Best validation loss degrades `3.0×` (`0.00705 → 0.02124`). This is direct evidence that run-to-run variation is a material confound for the `exp79` result.

**Status:** Supported by the `exp104` replication; the `exp79` result must be treated as one trajectory until a seed sweep is run.

**Validation experiments:**
1. Seed sweep for `exp79`, `exp78`, and `exp73` (minimum 3–5 seeds each). Report mean ± std for best validation loss.

---

## 11. Translate-route-decode hypothesis

**Statement:** Cross-head communication is more useful when three responsibilities are separated: source-local feature translation, receiver-selective head routing, and destination-local decoding. Communication should be injected conservatively through a residual gate.

**Evidence:**
- `exp86` learns strong, directly mixed cross-feature communication but finishes at `0.02003` best validation loss.
- `exp87` uses head-specific encoders/decoders, masked-softmax receiver routing, and small residual gates; it improves to `0.01089`.
- `exp87` learns stable non-uniform routing, including a source head selected by three receivers, while maintaining small gates.

**Contradictions / open questions:**
- `exp87` is more expressive and uses many more parameters than the constrained coordinate mixer, so its improvement over `exp73` cannot be attributed solely to translation/routing separation.
- It still does not match `exp79`; the mechanism of the latter remains unresolved.

**Status:** Supported by one trajectory as a promising design direction; not a causal conclusion.

**Validation experiments:**
1. Fixed-uniform routing with learned encoders/decoders and gates, to isolate whether learned routing helps.
2. Linear encoder/decoder variant, to isolate whether MLP nonlinearity helps.
3. Gate-init sweep (`-6`, `-5`, `-4`) with a common seed set, to test delayed-message strength.

---

## Summary Table

| # | Hypothesis | Status | Key Missing Experiment |
|---|---|---|---|
| 1 | Gradient-channel | Weakened by `exp104` (gradient path active without gain) | Freeze communication at epoch N |
| 2 | Premature bottleneck | Partially supported, revised | Gate-init / scheduled-message sweep |
| 3 | Continuous coordinate-mixer landscape | Supported (coarse) | Uniform ratio sweep `0.01–0.2` |
| 4 | Representation-locking | Strongly supported | Mixer reset at mid-training |
| 5 | Delayed coupling | Not observed on `exp104` trajectory; weakened | `exp79` intermediate mixer trajectory |
| 6 | Shared-mixer constraint | Weakened by `exp84` | Multi-seed shared vs distinct comparison |
| 7 | Single-layer stacking | Unresolved; insufficient alone | `num_heads=1` layerwise baseline |
| 8 | Learnability-as-freedom | Needs test | Fixed-identity control |
| 9 | Residual regularization | Speculative | Gradient/message magnitude analysis |
| 10 | Seed variation | **Supported by `exp104` replication (`3.0×` gap)** | Multi-seed sweep |
| 11 | Translate-route-decode | Promising, one trajectory | Routing / gate / MLP ablations |

---

## Recommended priority order for next experiments

The `exp104` replication (same architecture/configuration as `exp79`, verified code-equivalent, `3.0×` worse) elevates seed variation from an acknowledged uncertainty to a supported confound. Until a seed sweep is run, every single-trajectory comparison in this document — including the layerwise gain itself — should be treated as an observation rather than a result.

1. **Seed sweep for `exp79`, `exp73`, `exp78`** — now the single most important experiment; establish the distribution of best/final validation loss under the layerwise identity-init configuration (expected median near `0.02`, not `0.007`, given `exp104`). Run with probes enabled to collect mixer trajectories across seeds.
2. **Implement the planned probe system for communication residual magnitude** — record `||gate ⊙ D(m)|| / ||h||`, routing, and gates through training.
3. **Fixed-uniform-routing latent communication** — isolate learned receiver routing from source/destination MLP capacity.
4. **`num_heads=1` layerwise baseline** — isolate explicit stacked-LSTM effects.
5. **Intermediate mixer trajectory for `exp79`** — now lower priority, since the observed `exp104` trajectory shows no excursion; useful only to rule out an `exp79`-specific path.
