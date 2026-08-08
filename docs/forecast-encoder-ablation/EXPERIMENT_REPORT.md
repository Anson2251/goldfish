# Forecast Encoder Ablation: Conv1d, No Front-End, and Linear Projection

## Summary

This report compares three forecasting encoders on `fourier-lb256`:

1. `exp106`: `Conv1d -> LSTM`.
2. `exp107`: raw-input `LSTM` with no front-end encoder.
3. `exp108`: per-time-step `Linear -> LSTM`.

The convolutional front-end is decisively better in this single-run ablation. Removing it increases the best validation loss by 127.2%; replacing it with a linear projection increases the best validation loss by 172.8%. On the held-out test split, the convolutional model has the lowest aggregate error: removing the front-end raises MAE by 61.2% and RMSE by 96.8%, while replacing it with a linear projection raises MAE by 112.4% and RMSE by 145.4%. The result supports the hypothesis that the gain comes from local temporal encoding, not merely from expanding the feature dimension or increasing the parameter count.

## Question

Does a local temporal convolution before an LSTM improve multi-horizon forecasting, relative to both a raw-input LSTM and a per-time-step linear projection?

## Experimental Setup

All three runs use the same data, split, preprocessing, task, optimizer, scheduler, and training budget:

| Setting | Value |
|---|---|
| Dataset | `fourier-lb256` |
| Dataset fingerprint | `ebe1ebc2b35f8dc8bfc1c3d06a3b1c26de451dd22c910fdee4a9218b4ce2e44c` |
| Features | `signal`, `trend`, `phase_sin`, `phase_cos` |
| Target | `signal` |
| Lookback | 256 |
| Forecast horizons | 1, 5, 20 |
| Train / validation / test windows | 19,725 / 3,980 / 3,980 |
| Loss | normalized MSE |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Scheduler | cosine, `t_max=500`, `eta_min=0` |
| Batch size | 2,048 |
| Training budget | 1,000 epochs, 10,000 optimization steps |
| Device | Tesla P100-PCIE-16GB |

The stored `data.json` files have identical dataset, split, and normalizer fingerprints. The `environment.json` files also match: PyTorch `2.6.0+cu124`, CUDA `12.4`, and git commit `ca90cc4683f8e9471be6c3b0c34aeda8223413b3`.

## Models

All models predict the three horizons from the final recurrent state using the same forecast head. Each LSTM has `hidden_dim=32`, one layer, and no dropout.

| Run | Architecture | Front-end behavior | Parameters |
|---|---|---|---:|
| exp106 | `Conv1d(4, 32, kernel_size=7, padding=same) -> LSTM(32, 32)` | Mixes input channels and neighboring time steps before recurrence | 9,475 |
| exp107 | `LSTM(4, 32)` | No front-end; recurrent layer receives raw features | 4,963 |
| exp108 | `Linear(4, 32) -> LSTM(32, 32)` | Mixes channels independently at each time step | 8,707 |

The convolution has 768 more parameters than the linear projection. It differs qualitatively as well: `Conv1d(kernel_size=7)` can encode local temporal patterns, while `Linear(4, 32)` has no access to neighboring time steps.

## Results

### Best and final validation metrics

| Run | Best epoch | Best validation loss | Final validation loss | Final MAE | Final RMSE | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| exp106, Conv1d -> LSTM | 960 | **0.005183** | **0.006640** | **0.07750** | **0.12115** | 446.6 s |
| exp107, LSTM | 985 | 0.011777 | 0.012569 | 0.09839 | 0.16670 | **334.4 s** |
| exp108, Linear -> LSTM | 996 | 0.014139 | 0.021251 | 0.12884 | 0.21664 | 375.0 s |

Relative to `exp106`:

| Run | Best-loss change | Final-loss change | Final-MAE change | Final-RMSE change |
|---|---:|---:|---:|---:|
| exp107, no front-end | +127.2% | +89.3% | +27.0% | +37.6% |
| exp108, linear projection | +172.8% | +220.1% | +66.3% | +78.8% |

### Held-Out Test Metrics

The following values evaluate each run's `best.pt` on the same locked test split of 3,980 windows. MAE and RMSE are in raw signal units.

| Run | Test MAE | Test RMSE | Test MAE change from exp106 | Test RMSE change from exp106 |
|---|---:|---:|---:|---:|
| exp106, Conv1d -> LSTM | **0.09312** | **0.13324** | baseline | baseline |
| exp107, LSTM | 0.15012 | 0.26225 | +61.2% | +96.8% |
| exp108, Linear -> LSTM | 0.19779 | 0.32697 | +112.4% | +145.4% |

#### Per-Horizon Test MAE

Each cell is raw-unit MAE followed by its change from `exp106`. Negative percentages indicate lower error.

| Run | Horizon 1 | Horizon 5 | Horizon 20 |
|---|---:|---:|---:|
| exp106, Conv1d -> LSTM | 0.067489 (baseline) | **0.048348 (baseline)** | **0.163520 (baseline)** |
| exp107, LSTM | **0.055305 (-18.1%)** | 0.064759 (+33.9%) | 0.330298 (+102.0%) |
| exp108, Linear -> LSTM | 0.069506 (+3.0%) | 0.096948 (+100.5%) | 0.426917 (+161.1%) |

The raw LSTM's horizon-1 MAE is lower than the convolutional model's, but it loses accuracy rapidly at longer horizons. Its horizon-20 MAE is approximately twice that of `exp106`. The linear-projection model is worse at every horizon. The convolutional front-end is therefore the best model for the aggregate multi-horizon objective and for both horizon 5 and horizon 20.

### Convergence

| Epoch | Conv1d -> LSTM | LSTM | Linear -> LSTM |
|---:|---:|---:|---:|
| 10 | 0.398926 | 0.460419 | **0.352378** |
| 50 | **0.064307** | 0.151561 | 0.148070 |
| 100 | **0.030727** | 0.082969 | 0.098246 |
| 200 | **0.013071** | 0.034689 | 0.043566 |
| 400 | **0.007351** | 0.026711 | 0.031243 |
| 800 | **0.006697** | 0.018716 | 0.018989 |
| 1,000 | **0.006640** | 0.012569 | 0.021251 |

The linear-projection model is slightly ahead at epoch 10, but the convolutional model is better by epoch 50 and retains a large margin for the rest of training.

| Validation-loss threshold | Conv1d -> LSTM | LSTM | Linear -> LSTM |
|---|---:|---:|---:|
| <= 0.10 | 40 | 79 | 76 |
| <= 0.05 | 60 | 142 | 159 |
| <= 0.02 | 136 | 739 | 762 |
| <= 0.01 | 256 | Not reached | Not reached |

The convolutional encoder reaches every threshold substantially earlier. Neither non-convolutional variant reaches validation loss `0.01` in the allotted training budget.

## Interpretation

### The front-end encoder matters

The raw LSTM baseline, `exp107`, performs materially worse than the convolutional model on the aggregate test objective despite being faster. It does achieve lower horizon-1 MAE, so the result is not that the raw LSTM cannot predict the immediate next step. Instead, its error grows sharply across the longer forecast horizons, indicating that it does not recover the local representation needed for robust multi-step forecasting in this capacity and optimization regime.

### The convolution, rather than width, explains the improvement

`exp108` has 75.4% more parameters than the raw LSTM and nearly as many as the convolutional model, yet it is the weakest run on both validation and test aggregate metrics. Its linear front-end mixes only the features at the current time step, so its added width does not provide the local temporal inductive bias supplied by the seven-step convolution. This is reflected at every horizon, with the largest relative regression at horizon 20.

### The convolutional model is the best quality-cost trade-off

Compared with the raw LSTM, `exp106` adds 4,512 parameters and about 112.2 seconds of training time, but lowers the best validation loss from `0.011777` to `0.005183`. Compared with the linear projection, it adds only 768 parameters and about 71.6 seconds, while cutting the best validation loss from `0.014139` to `0.005183`.

## Limitations

This is a single-run comparison. All configurations set `seed: null` and `deterministic: false`; random initialization, data order, and CUDA execution may contribute to the observed values. The effect size is large enough to motivate the conclusion, but it is not a statistical estimate.

The scheduler uses `t_max=500` for 1,000 epochs. Its learning rate reaches zero around epoch 500 and then rises again, which creates late-training oscillation. The final checkpoint is therefore less representative than the monitored best checkpoint. The last-20-epoch validation-loss standard deviations are `0.000892` for the convolutional model, `0.004334` for the raw LSTM, and `0.004863` for the linear-projection model.

The training summaries contain validation metrics only, but each original ablation run now has a held-out test forecast generated from its monitored `best.pt`. The test comparison remains a single-run estimate because all configurations use `seed: null` and `deterministic: false`.

## Conclusion

For this forecasting task, retain `Conv1d(4, 32, kernel_size=7, padding=same)` before the LSTM. The ablation shows that the convolutional encoder provides a useful local temporal representation that a raw LSTM and an equally wide per-time-step linear projection do not match.

## Follow-up Experiments

1. Repeat all three variants with a shared set of 3-5 fixed seeds and report mean plus standard deviation.
2. Repeat the held-out test comparison across the same fixed seeds and report mean plus standard deviation for each horizon.
3. Set `t_max=1000`, or use an explicit warm-restart scheduler if cyclic learning rates are intended.
4. Sweep convolution kernel sizes `1`, `3`, `7`, and `15` to identify whether the gain is specifically local context and to select the effective receptive field.

## Archived Runs

The complete canonical run directories are archived alongside this report:

- Original encoder ablation: `exp106.tar.xz`, `exp107.tar.xz`, `exp108.tar.xz`
- Direct stride compression: `exp109.tar.xz`, `exp110.tar.xz`, `exp111.tar.xz`
- Stacked encoders and early-downsampling counterexample: `exp113.tar.xz`, `exp114.tar.xz`, `exp115.tar.xz`, `exp116.tar.xz`

Each archive includes the resolved configuration, data and environment provenance, epoch metrics, training plot, lifecycle log, best/final/latest checkpoints, and the `best.pt` test forecast artifacts. The `exp106`, `exp107`, and `exp108` archives were rebuilt after their test forecasts were generated.

## UPDATE: Temporal Compression by Convolution Stride

Three follow-up runs vary only the stride of the existing `Conv1d(4, 32, kernel_size=7, padding=3)` front-end before the same one-layer `LSTM(32, 32)`. The stride compresses the 256-step input history before it reaches the LSTM. `exp106` is the no-compression baseline; all four runs have identical data provenance, optimization settings, training budget, and parameter count (9,475).

| Run | Conv stride | LSTM input length | Compression | Best validation loss | Test MAE | Test RMSE | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp106 | 1 | 256 | 1x | 0.005183 | 0.09312 | 0.13324 | 446.6 s |
| exp109 | 2 | 128 | 2x | 0.003736 | 0.10418 | 0.13251 | 275.4 s |
| exp110 | 4 | 64 | 4x | **0.002572** | **0.07588** | **0.10977** | 216.9 s |
| exp111 | 8 | 32 | 8x | 0.019813 | 0.17643 | 0.33695 | **179.1 s** |

Test metrics use each run's `best.pt` and the same locked test split of 3,980 windows. MAE and RMSE are reported in raw signal units.

### Per-Horizon Test MAE

Each cell gives raw-unit MAE followed by its change from the uncompressed `exp106` baseline. Negative percentages indicate lower error.

| Run | Horizon 1 MAE | Horizon 5 MAE | Horizon 20 MAE |
|---|---:|---:|---:|
| exp106, stride 1 | 0.067489 (baseline) | 0.048348 (baseline) | 0.163520 (baseline) |
| exp109, stride 2 | 0.092324 (+36.8%) | 0.086905 (+79.7%) | 0.133301 (-18.5%) |
| exp110, stride 4 | **0.045219 (-33.0%)** | 0.051621 (+6.8%) | **0.130795 (-20.0%)** |
| exp111, stride 8 | 0.052956 (-21.5%) | 0.072795 (+50.6%) | 0.403529 (+146.8%) |

### Interpretation

`exp110` is the best overall compression level. It cuts the LSTM sequence from 256 to 64 steps, reduces training time by 51.4%, and improves aggregate test MAE by 18.5% and RMSE by 17.6% relative to `exp106`. It provides the best horizon-1 and horizon-20 errors, while its horizon-5 MAE is only 6.8% above the baseline.

`exp109` has slightly lower aggregate test RMSE than `exp106` (-0.5%), but its aggregate MAE is worse (+11.9%) and its short-horizon errors regress substantially. Its horizon-20 improvement alone does not justify preferring it to the baseline or `exp110`.

`exp111` compresses too aggressively. Although horizon 1 remains better than the baseline, the 32-step representation loses information needed for long-range prediction: horizon-20 MAE rises by 146.8%, driving aggregate test MAE and RMSE up by 89.5% and 152.9%, respectively.

The recommended front-end for subsequent experiments is therefore:

```yaml
conv_kernel_size: 7
conv_stride: 4
conv_padding: 3
conv_channels: 32
```

This update remains a single-run result for each stride. Repeating the four configurations with matched fixed seeds is needed to quantify run-to-run variation.

## UPDATE: Stacked Encoders and Early Downsampling

The direct-stride sweep applies one `Conv1d(k=7)` directly to raw input and downsamples in that convolution. The next sweep separates local encoding from downsampling by adding a second convolution and `SiLU` activations:

```text
Conv1d(k=7, stride=1) -> SiLU -> Conv1d(k=3, stride=S) -> SiLU -> LSTM
```

The first convolution preserves all 256 time steps while constructing a 32-channel local representation. The second convolution then compresses that representation. `exp113`, `exp114`, and `exp115` use second-layer strides 2, 4, and 8, producing LSTM input lengths 128, 64, and 32. They contain 13,409 parameters, compared with 9,475 for the direct single-convolution models.

| Run | Encoder stride pattern | LSTM input length | Best validation loss | Test MAE | Test RMSE | Training time |
|---|---|---:|---:|---:|---:|---:|
| exp113 | `1 -> 2` | 128 | 0.003067 | 0.07455 | 0.10933 | 339.9 s |
| exp114 | `1 -> 4` | 64 | 0.002730 | 0.06192 | 0.09410 | 251.0 s |
| exp115 | `1 -> 8` | 32 | **0.001763** | **0.05841** | **0.09014** | 210.4 s |

All test metrics use the saved `best.pt`, the same locked test split, and raw signal units.

### Stacked Versus Direct at Equal Compression

| Total compression | Direct run | Stacked run | Test-MAE change | Test-RMSE change |
|---:|---|---|---:|---:|
| 2x | exp109: `2` | exp113: `1 -> 2` | -28.4% | -17.5% |
| 4x | exp110: `4` | exp114: `1 -> 4` | -18.4% | -14.3% |
| 8x | exp111: `8` | exp115: `1 -> 8` | **-66.9%** | **-73.2%** |

The stacked encoder improves every matched-compression comparison. The largest effect is at 8x: direct compression of raw features catastrophically harms horizon-20 prediction, while local encoding before compression recovers and surpasses the earlier best result.

### Per-Horizon Test MAE: Direct Versus Stacked

Each percentage is the stacked model's MAE change from its equal-compression direct counterpart. Negative values indicate lower error.

| Total compression | Horizon 1 | Horizon 5 | Horizon 20 |
|---:|---:|---:|---:|
| 2x: exp109 -> exp113 | 0.092324 -> 0.047447 (-48.6%) | 0.086905 -> 0.045126 (-48.1%) | 0.133301 -> 0.131073 (-1.7%) |
| 4x: exp110 -> exp114 | 0.045219 -> 0.032576 (-28.0%) | 0.051621 -> 0.035606 (-31.0%) | 0.130795 -> 0.117592 (-10.1%) |
| 8x: exp111 -> exp115 | 0.052956 -> 0.033311 (-37.1%) | 0.072795 -> 0.031557 (-56.7%) | 0.403529 -> 0.110373 (-72.6%) |

`exp115` is the strongest model observed so far. Relative to the previous direct best, `exp110`, it lowers aggregate test MAE from 0.07588 to 0.05841 (-23.0%) and RMSE from 0.10977 to 0.09014 (-17.9%), while reducing LSTM length from 64 to 32 and training time from 216.9 s to 210.4 s.

### Double-Compression Counterexample

`exp116` tests whether the same 8x compression can be distributed across both layers:

```text
Conv1d(k=7, stride=4) -> SiLU -> Conv1d(k=3, stride=2) -> SiLU -> LSTM
```

It has the same total stride (8), LSTM input length (32), channel count (32), and parameter count (13,409) as `exp115`. The difference is only where compression happens.

| Run | Encoder stride pattern | Best validation loss | Test MAE | Test RMSE | Training time |
|---|---|---:|---:|---:|---:|
| exp115 | `1 -> 8` | **0.001763** | **0.05841** | **0.09014** | 210.4 s |
| exp116 | `4 -> 2` | 0.003369 | 0.08957 | 0.14662 | **189.5 s** |

Relative to `exp115`, early downsampling in `exp116` increases test MAE by 53.3% and RMSE by 62.6%, for only a 9.9% training-time reduction.

| Horizon | exp115: `1 -> 8` MAE | exp116: `4 -> 2` MAE | exp116 change |
|---:|---:|---:|---:|
| 1 | 0.033311 | 0.048029 | +44.2% |
| 5 | 0.031557 | 0.037365 | +18.4% |
| 20 | 0.110373 | 0.183309 | +66.1% |

The largest regression remains horizon 20. This result supports preserving full temporal resolution through the first local encoder, then compressing once. A larger nominal receptive field in `exp116` (15 raw steps versus 9 for `exp115`) does not compensate for discarding intermediate raw-input time steps before they are encoded.

### Capacity Hypothesis and Next Controlled Test

The `exp115` versus `exp116` comparison rules out insufficient final LSTM input width as the immediate explanation: both models present the LSTM with exactly `[batch, 32 time steps, 32 channels]` and have the same parameter count. It does not rule out a narrower hypothesis: after the first `stride=4` operation, 32 encoder channels may be insufficient to retain both local details and multi-scale information for the second convolution.

To distinguish early-information loss from early-representation capacity, compare matched-width variants:

```text
A: Conv1d(4 -> 64, k=7, stride=4) -> SiLU -> Conv1d(64 -> 64, k=3, stride=2) -> SiLU -> LSTM(64 -> 32)
B: Conv1d(4 -> 64, k=7, stride=1) -> SiLU -> Conv1d(64 -> 64, k=3, stride=8) -> SiLU -> LSTM(64 -> 32)
```

If A remains worse than B, early temporal downsampling is the principal cause. If A closes the gap, the capacity of the early-downsampled encoder is a contributing factor. Both variants should be repeated with identical fixed seeds before drawing a statistical conclusion.
