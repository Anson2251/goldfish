# Recursive Self-Modifying Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small fully self-modifying numeric one-step forecasting model whose complete dynamic predictor parameter state is updated causally at every observed input and optimized with a weighted trajectory objective.

**Architecture:** `RecursiveSelfModifyingForecastModel` keeps a trainable prior parameter vector `theta_0` for a small MLP predictor. During each batch forward pass it expands `theta_0` to one distinct dynamic state per sequence, predicts with `theta_(t-1)`, obtains a complete bounded `Delta theta_t` from a fixed meta-network, and functionally creates `theta_t` without mutating `nn.Parameter` storage. A new task compares the model's interior one-step trajectory to shifted target features and combines a linearly time-weighted MAE with the direct one-step forecast after the last input.

**Tech Stack:** Python 3.13, PyTorch 2.6, PyYAML, pytest.

## Global Constraints

- Do not add dependencies.
- Preserve the `Batch -> ModelOutput -> Task -> Trainer` lifecycle; no model-specific branch belongs in `Trainer`.
- Dynamic state is a graph-connected per-batch `Tensor`; never assign to `nn.Parameter.data`, use `copy_`, or mutate registered model parameters inside `forward`.
- The model supports numeric manifests with exactly `window.horizons: [1]`; reject any other horizon layout before model construction.
- Every predictor layer's weights and biases belong to `theta_t` and are updated at every observation.
- Prediction at position `t` uses `theta_(t-1)`; only its subsequent update affects positions greater than `t`.
- Keep existing forecast models, `PointForecastTask`, checkpoint format, and forecast export behavior unchanged.
- Use ASCII source text unless editing an existing non-ASCII artifact.
- Do not commit unless the user explicitly requests a commit.

---

## File Structure

| File | Responsibility |
|---|---|
| `goldfish/models/forecast/self_modifying.py` | Functional dynamic MLP, fixed meta-network, causal parameter trajectory, and `ModelOutput` diagnostics. |
| `goldfish/models/forecast/__init__.py` | Public export for the new forecast model. |
| `goldfish/models/__init__.py` | Import and registry entry for `forecast/self-modifying`. |
| `goldfish/tasks/self_modifying_forecast.py` | Trajectory target extraction plus weighted MAE/final-MAE objective and metrics. |
| `goldfish/tasks/__init__.py` | Public task export. |
| `train.py` | Add model-specific numeric compatibility validation, resolve target feature positions, persist the specialized task configuration, and select the task. |
| `model-profiles/forecast/self-modifying-small.yaml` | Reproducible small model configuration. |
| `tests/models/test_self_modifying_forecast.py` | Model semantics, gradients, and configuration validation tests. |
| `tests/tasks/test_self_modifying_forecast.py` | Loss/metric and target-alignment unit tests. |
| `tests/integration/test_numeric_commands.py` | Profile-driven short numeric training and checkpoint restoration coverage. |
| `docs/MODELS.md` | Public model contract, state semantics, constructor, and diagnostics. |
| `docs/TASKS.md` | New task contract and objective equation. |
| `README.md` | One-step dataset/profile training invocation. |

## Dynamic Model Contract

The constructor uses runtime dimensions and profile-owned parameters:

```python
RecursiveSelfModifyingForecastModel(
    feature_count: int,
    target_count: int,
    horizon_count: int,
    target_feature_indices: tuple[int, ...],
    predictor_hidden_dims: tuple[int, ...],
    meta_hidden_dim: int,
    update_scale: float = 0.01,
)
```

Validation requirements:

```text
feature_count > 0
0 < target_count <= feature_count
horizon_count == 1
len(target_feature_indices) == target_count
all target feature indices are distinct and in [0, feature_count)
predictor_hidden_dims is non-empty and all widths are positive
meta_hidden_dim > 0
update_scale > 0
```

The dynamic predictor is an MLP with shape:

```text
feature_count -> predictor_hidden_dims[0] -> ... -> predictor_hidden_dims[-1] -> target_count
```

Its fully flattened state is the concatenation of each layer's weights and biases. It is stored by the module as one trainable prior:

```python
theta_initial: nn.Parameter  # [parameter_count]
```

The fixed meta-network has this shape:

```text
concat(x_t, theta_(t-1))
-> Linear(feature_count + parameter_count, meta_hidden_dim)
-> SiLU
-> Linear(meta_hidden_dim, parameter_count)
-> Delta theta_t
```

At every time `t`, with `dynamic_state` shaped `[B, parameter_count]`:

```python
prediction_t = functional_predict(batch.inputs[:, t], dynamic_state)
raw_update_t = self.meta_network(torch.cat((batch.inputs[:, t], dynamic_state), dim=-1))
delta_t = self.update_scale * torch.tanh(raw_update_t)
dynamic_state = dynamic_state + delta_t
```

For an input history `[B, L, F]`, this produces:

```python
trajectory_forecast: Tensor  # [B, L - 1, C], predictions for t = 0..L-2
forecast: Tensor             # [B, 1, C], prediction at t = L-1 before its update
representations: Tensor      # [B, L, parameter_count], state after each position's update
diagnostics: dict[str, Tensor]
```

`diagnostics` must provide graph-detached scalar tensors:

```python
{
    "parameter_count": float(parameter_count),
    "mean_update_norm": delta_trajectory.norm(dim=-1).mean().detach(),
    "mean_parameter_norm": state_trajectory.norm(dim=-1).mean().detach(),
    "final_parameter_shift": (state_trajectory[:, -1] - initial_state).norm(dim=-1).mean().detach(),
}
```

The `trajectory_forecast` deliberately excludes the final direct forecast so the task can weight only adjacent observations inside the input window. The final `forecast` remains `[B, 1, C]`, retaining compatibility with `forecast.py`.

## Task Contract

```python
SelfModifyingForecastTask(
    normalizer: StandardNormalizer,
    targets: Sequence[str],
    target_feature_indices: Sequence[int],
    final_loss_weight: float = 1.0,
)
```

Given `batch.inputs [B, L, F]`, compute the normalized trajectory target by selecting target feature columns from subsequent inputs:

```python
trajectory_targets = batch.inputs[:, 1:, target_feature_indices]  # [B, L - 1, C]
```

Given trajectory prediction `P` and final prediction `Q`:

```python
step_mae = (P - trajectory_targets).abs().mean(dim=-1)  # [B, L - 1]
weights = arange(1, L, device=P.device, dtype=P.dtype) / (L - 1)
trajectory_loss = (step_mae * weights).sum(dim=-1).mean()
final_loss = functional.l1_loss(Q, batch.targets)
loss = trajectory_loss + final_loss_weight * final_loss
```

The task returns graph-connected `loss` and detached metrics:

```python
{
    "trajectory_mae": trajectory_loss.detach(),
    "final_mae": final_loss.detach(),
    "mae": final_loss.detach(),
}
```

It validates prediction keys, ranks, matching dimensions, target-index bounds, `batch.inputs` target width, and `batch.targets` exact `[B, 1, C]` shape. It reports raw-unit MAE only for the direct final prediction through `StandardNormalizer`; internal trajectory MAE remains normalized because it is a representation-learning/adaptation diagnostic rather than an exported forecast metric.

### Task 1: Add the Fully Dynamic Functional Predictor

**Files:**
- Create: `goldfish/models/forecast/self_modifying.py`
- Test: `tests/models/test_self_modifying_forecast.py`

**Interfaces:**
- Consumes: `ForecastBatch.inputs: Tensor [B, L, F]` and `goldfish.core.ModelOutput`.
- Produces: `RecursiveSelfModifyingForecastModel` with constructor and output contract defined in **Dynamic Model Contract**.

- [ ] **Step 1: Write the failing shape and dynamic-state tests**

```python
import torch

from goldfish.data.numeric import ForecastBatch
from goldfish.models.forecast.self_modifying import RecursiveSelfModifyingForecastModel


def make_model() -> RecursiveSelfModifyingForecastModel:
    return RecursiveSelfModifyingForecastModel(
        feature_count=3,
        target_count=2,
        horizon_count=1,
        target_feature_indices=(0, 2),
        predictor_hidden_dims=(8, 8),
        meta_hidden_dim=16,
        update_scale=0.05,
    )


def test_self_modifying_forecast_returns_causal_trajectory_and_final_forecast() -> None:
    model = make_model()
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 1, 2))

    output = model(batch)

    assert output.predictions["trajectory_forecast"].shape == (2, 4, 2)
    assert output.predictions["forecast"].shape == (2, 1, 2)
    assert output.representations is not None
    assert output.representations.shape == (2, 5, model.parameter_count)
    assert output.diagnostics["parameter_count"] == float(model.parameter_count)
    assert output.diagnostics["mean_update_norm"].item() > 0
```

- [ ] **Step 2: Run the shape test to verify it fails**

Run:

```sh
uv run pytest tests/models/test_self_modifying_forecast.py::test_self_modifying_forecast_returns_causal_trajectory_and_final_forecast -v
```

Expected: FAIL because `goldfish.models.forecast.self_modifying` does not exist.

- [ ] **Step 3: Implement functional parameter layout and model forward**

Create a `_DynamicMLPLayout` value object that stores `(input_width, output_width)` for every layer and slices a `[B, parameter_count]` state into batched weights/biases without registering dynamic parameters. Its operations must be equivalent to:

```python
def apply_layer(inputs: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    # inputs: [B, in_features], weights: [B, out_features, in_features], bias: [B, out_features]
    return torch.bmm(weights, inputs.unsqueeze(-1)).squeeze(-1) + bias
```

Implement `functional_predict()` to use SiLU after every non-final layer. Initialize `theta_initial` with Xavier-uniform layer weights and zero biases packed into one parameter vector. Initialize the final meta-network projection to small values (for example Xavier weights multiplied by `0.01` and zero bias) so initial updates are bounded but nonzero.

Implement the forward loop exactly in causal order:

```python
state = self.theta_initial.unsqueeze(0).expand(batch_size, -1)
initial_state = state
states: list[Tensor] = []
predictions: list[Tensor] = []
updates: list[Tensor] = []
for time_index in range(sequence_length):
    observation = batch.inputs[:, time_index]
    predictions.append(self._predict(observation, state))
    update = self.update_scale * torch.tanh(self.meta_network(torch.cat((observation, state), dim=-1)))
    updates.append(update)
    state = state + update
    states.append(state)
```

Return the first `L - 1` predictions as `trajectory_forecast`, the last prediction after an inserted horizon dimension as `forecast`, and the stacked state/update diagnostics described above. Reject lookback shorter than two, invalid input rank, and wrong feature width.

- [ ] **Step 4: Run the shape test to verify it passes**

Run:

```sh
uv run pytest tests/models/test_self_modifying_forecast.py::test_self_modifying_forecast_returns_causal_trajectory_and_final_forecast -v
```

Expected: PASS.

- [ ] **Step 5: Write causal-order, zero-update, and gradient tests**

```python
def test_final_prediction_uses_pre_update_final_state() -> None:
    model = RecursiveSelfModifyingForecastModel(
        feature_count=1,
        target_count=1,
        horizon_count=1,
        target_feature_indices=(0,),
        predictor_hidden_dims=(1,),
        meta_hidden_dim=2,
        update_scale=1.0,
    )
    with torch.no_grad():
        model.theta_initial.zero_()
        for parameter in model.meta_network.parameters():
            parameter.zero_()
        model.meta_network[-1].bias.fill_(1.0)
    batch = ForecastBatch(inputs=torch.tensor([[[1.0], [2.0]]]), targets=torch.zeros(1, 1, 1))

    output = model(batch)

    torch.testing.assert_close(output.predictions["trajectory_forecast"], torch.zeros(1, 1, 1))
    torch.testing.assert_close(output.predictions["forecast"], torch.tensor([[[1.0]]]))


def test_zero_meta_network_keeps_the_dynamic_predictor_fixed() -> None:
    model = make_model()
    with torch.no_grad():
        for parameter in model.meta_network.parameters():
            parameter.zero_()
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 1, 2))

    output = model(batch)

    assert output.representations is not None
    torch.testing.assert_close(output.representations, output.representations[:, :1].expand_as(output.representations))
    assert output.diagnostics["mean_update_norm"].item() == 0


def test_self_modifying_forecast_backpropagates_through_every_parameter_update() -> None:
    model = make_model()
    batch = ForecastBatch(inputs=torch.randn(2, 5, 3), targets=torch.randn(2, 1, 2))

    loss = model(batch).predictions["trajectory_forecast"].square().mean()
    loss.backward()

    assert model.theta_initial.grad is not None
    assert torch.isfinite(model.theta_initial.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.meta_network.parameters())
```

- [ ] **Step 6: Run model tests to verify causal and BPTT behavior**

Run:

```sh
uv run pytest tests/models/test_self_modifying_forecast.py -v
```

Expected: PASS.

### Task 2: Register and Construct the Model from Profiles

**Files:**
- Modify: `goldfish/models/forecast/__init__.py`
- Modify: `goldfish/models/__init__.py`
- Create: `model-profiles/forecast/self-modifying-small.yaml`
- Modify: `tests/models/test_self_modifying_forecast.py`
- Modify: `tests/config/test_models.py`

**Interfaces:**
- Consumes: `RecursiveSelfModifyingForecastModel` from Task 1 and the existing `ModelRegistry`/profile resolution API.
- Produces: `model_registry.create("forecast", "self-modifying", ...)` and a profile whose dynamic predictor has a target size of approximately 6k to 10k parameters for the Fourier feature/target dimensions.

- [ ] **Step 1: Write the failing registry/profile test**

```python
from goldfish.config import create_model_from_config, load_model_profile, resolve_model_config
from goldfish.models import RecursiveSelfModifyingForecastModel, model_registry


def test_self_modifying_forecast_is_registered_and_profile_constructible() -> None:
    profile = load_model_profile(Path("model-profiles/forecast/self-modifying-small.yaml"))
    config = resolve_model_config(
        profile,
        family="forecast",
        runtime_parameters={
            "feature_count": 4,
            "target_count": 1,
            "horizon_count": 1,
            "target_feature_indices": (0,),
        },
    )

    model = create_model_from_config(config)

    assert isinstance(model, RecursiveSelfModifyingForecastModel)
    assert model_registry.get("forecast", "self-modifying") is RecursiveSelfModifyingForecastModel
    assert 6_000 <= model.parameter_count <= 10_000
```

- [ ] **Step 2: Run the registry/profile test to verify it fails**

Run:

```sh
uv run pytest tests/models/test_self_modifying_forecast.py::test_self_modifying_forecast_is_registered_and_profile_constructible -v
```

Expected: FAIL because the model is not exported/registered and the profile does not exist.

- [ ] **Step 3: Export, register, and configure the model**

Add the public import to `goldfish/models/forecast/__init__.py`, import it through `goldfish/models/__init__.py`, and register:

```python
model_registry.register("forecast", "self-modifying", RecursiveSelfModifyingForecastModel)
```

Create `model-profiles/forecast/self-modifying-small.yaml`:

```yaml
schema_version: 1
model:
  family: forecast
  name: self-modifying
  parameters:
    predictor_hidden_dims: [64, 64]
    meta_hidden_dim: 64
    update_scale: 0.01
```

Before finalizing the profile, instantiate it with `feature_count=4`, `target_count=1`, and compute `model.parameter_count`. Adjust only `predictor_hidden_dims` if the dynamic predictor is not in `[6000, 10000]`; `meta_hidden_dim` is fixed-model capacity and is not part of the dynamic predictor count.

- [ ] **Step 4: Run model and config tests to verify profile construction**

Run:

```sh
uv run pytest tests/models/test_self_modifying_forecast.py tests/config/test_models.py -v
```

Expected: PASS.

### Task 3: Add the Weighted Self-Modifying Forecast Objective

**Files:**
- Create: `goldfish/tasks/self_modifying_forecast.py`
- Modify: `goldfish/tasks/__init__.py`
- Create: `tests/tasks/test_self_modifying_forecast.py`

**Interfaces:**
- Consumes: `ModelOutput.predictions["trajectory_forecast"] [B, L-1, C]`, `ModelOutput.predictions["forecast"] [B, 1, C]`, `ForecastBatch`, `StandardNormalizer`, and ordered target feature indices.
- Produces: `SelfModifyingForecastTask.compute(output, batch) -> StepResult` with `loss`, `trajectory_mae`, `final_mae`, `mae`, and `rmse` metrics.

- [ ] **Step 1: Write the failing hand-computed objective test**

```python
import pytest
import torch

from goldfish.core import ModelOutput
from goldfish.data.numeric import ForecastBatch, StandardNormalizer
from goldfish.tasks import SelfModifyingForecastTask


def test_self_modifying_task_weights_later_trajectory_predictions_and_final_prediction() -> None:
    normalizer = StandardNormalizer(("a", "b"), torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    batch = ForecastBatch(
        inputs=torch.tensor([[[0.0, 10.0], [1.0, 20.0], [3.0, 30.0], [6.0, 40.0]]]),
        targets=torch.tensor([[[10.0]]]),
    )
    output = ModelOutput(
        predictions={
            "trajectory_forecast": torch.tensor([[[2.0], [5.0], [5.0]]], requires_grad=True),
            "forecast": torch.tensor([[[8.0]]], requires_grad=True),
        }
    )

    result = SelfModifyingForecastTask(normalizer, ["a"], [0], final_loss_weight=2.0).compute(output, batch)

    # Target trajectory is [1, 3, 6]; errors [1, 2, 1]; weights [1/3, 2/3, 1].
    assert result.metrics["trajectory_mae"] == pytest.approx((1 / 3 + 4 / 3 + 1) / 1)
    assert result.metrics["final_mae"] == pytest.approx(2.0)
    assert result.loss.item() == pytest.approx((1 / 3 + 4 / 3 + 1) + 4.0)
```

- [ ] **Step 2: Run the objective test to verify it fails**

Run:

```sh
uv run pytest tests/tasks/test_self_modifying_forecast.py::test_self_modifying_task_weights_later_trajectory_predictions_and_final_prediction -v
```

Expected: FAIL because `SelfModifyingForecastTask` does not exist.

- [ ] **Step 3: Implement task validation, loss, and raw metrics**

Implement `SelfModifyingForecastTask` with the exact contract above. Use `functional.l1_loss` for the final direct forecast. Derive raw direct-forecast MAE and RMSE in the same way as `PointForecastTask` and report them as `mae` and `rmse`; retain normalized `trajectory_mae` and `final_mae` explicitly. The returned `loss` must not be detached.

- [ ] **Step 4: Add failure and gradient tests**

```python
def test_self_modifying_task_rejects_incompatible_horizon_and_target_indices() -> None:
    normalizer = StandardNormalizer(("a",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="target_feature_indices"):
        SelfModifyingForecastTask(normalizer, ["a"], [1])


def test_self_modifying_task_rejects_non_one_step_final_forecast() -> None:
    task = SelfModifyingForecastTask(
        StandardNormalizer(("a",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
        ["a"],
        [0],
    )
    batch = ForecastBatch(inputs=torch.zeros(1, 3, 1), targets=torch.zeros(1, 2, 1))
    output = ModelOutput(predictions={"trajectory_forecast": torch.zeros(1, 2, 1), "forecast": torch.zeros(1, 2, 1)})

    with pytest.raises(ValueError, match="one-step"):
        task.compute(output, batch)


def test_self_modifying_task_loss_preserves_gradients() -> None:
    prediction = torch.zeros(1, 2, 1, requires_grad=True)
    final = torch.zeros(1, 1, 1, requires_grad=True)
    task = SelfModifyingForecastTask(
        StandardNormalizer(("a",), torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
        ["a"],
        [0],
    )
    result = task.compute(
        ModelOutput(predictions={"trajectory_forecast": prediction, "forecast": final}),
        ForecastBatch(inputs=torch.tensor([[[0.0], [1.0], [2.0]]]), targets=torch.tensor([[[3.0]]])),
    )

    result.loss.backward()
    assert prediction.grad is not None
    assert final.grad is not None
```

- [ ] **Step 5: Run task tests to verify they pass**

Run:

```sh
uv run pytest tests/tasks/test_self_modifying_forecast.py tests/tasks/test_point_forecast.py -v
```

Expected: PASS; existing point-forecast behavior remains unchanged.

### Task 4: Connect Dataset Semantics and Task Selection to Numeric Training

**Files:**
- Modify: `train.py`
- Modify: `tests/integration/test_numeric_commands.py`
- Modify: `tests/data/validation/test_manifest_and_lock.py`

**Interfaces:**
- Consumes: profile model name, numeric runtime metadata `features`, `targets`, and `horizons` plus `SelfModifyingForecastTask` from Task 3.
- Produces: a resolved model configuration with `target_feature_indices`, a persisted task block, and a trainer constructed with the specialized task only for `forecast/self-modifying`.

- [ ] **Step 1: Write the failing model/data compatibility tests**

```python
def test_self_modifying_profile_requires_one_step_numeric_horizon(tmp_path: Path) -> None:
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    write_numeric_bundle(dataset)
    manifest_path = dataset / "manifest.yaml"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace("horizons: [1]", "horizons: [1, 2]"), encoding="utf-8")
    assert load_entry("prepare")([str(dataset)]) == 0

    with pytest.raises(ValueError, match="horizons: \[1\]"):
        load_entry("train")([
            str(dataset), "--runs-dir", str(runs), "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
            "--model-profile", str(ROOT / "model-profiles" / "forecast" / "self-modifying-small.yaml"),
        ])


def test_self_modifying_profile_requires_targets_to_be_features(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    write_numeric_bundle(dataset)
    manifest_path = dataset / "manifest.yaml"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8").replace("targets: [value]", "targets: [missing]"), encoding="utf-8")

    with pytest.raises(ValueError, match="targets"):
        load_entry("prepare")([str(dataset)])
```

- [ ] **Step 2: Run compatibility tests to verify the missing self-modifying validation fails**

Run:

```sh
uv run pytest tests/integration/test_numeric_commands.py::test_self_modifying_profile_requires_one_step_numeric_horizon -v
```

Expected: FAIL because the model profile cannot yet be resolved with target positions or rejected with an explicit horizon error.

- [ ] **Step 3: Implement numeric resolution and task selection**

In `_train_numeric`, after loading the profile and before `resolve_model_config`, determine target positions from runtime metadata:

```python
features = cast(list[str], data_module.runtime_metadata["features"])
targets = cast(list[str], data_module.runtime_metadata["targets"])
horizons = cast(list[int], data_module.runtime_metadata["horizons"])
target_feature_indices = tuple(features.index(target) for target in targets)
```

For `profile["name"] == "self-modifying"`, require `horizons == [1]`, then include `target_feature_indices` in `runtime_parameters`. Keep all runtime parameters for all existing models exactly as they are today.

Add a helper with a narrow, testable boundary:

```python
def _numeric_task(model_config: Mapping[str, Any], data_module: Any) -> Task[Any]:
    if model_config.get("name") == "self-modifying":
        parameters = cast(Mapping[str, Any], model_config["parameters"])
        return SelfModifyingForecastTask(
            data_module.normalizer,
            cast(Sequence[str], data_module.runtime_metadata["targets"]),
            cast(Sequence[int], parameters["target_feature_indices"]),
        )
    return PointForecastTask(data_module.normalizer, cast(Sequence[str], data_module.runtime_metadata["targets"]))
```

Use this helper when constructing `Trainer`. In `_numeric_config`, derive task metadata from the model config:

```python
"task": {
    "name": manifest.get("task"),
    "loss": "weighted_trajectory_mae_plus_final_mae" if model["name"] == "self-modifying" else "normalized_mse",
    "metrics": ["trajectory_mae", "final_mae", "mae", "rmse"] if model["name"] == "self-modifying" else ["mae", "rmse"],
},
```

For strict and loose resume config reconstruction, preserve `target_feature_indices` by reading the saved model parameter rather than attempting to rediscover it from dimension-only placeholders. This avoids profile-specific loss of model constructor configuration.

- [ ] **Step 4: Run compatibility and existing numeric training tests**

Run:

```sh
uv run pytest tests/integration/test_numeric_commands.py tests/data/validation/test_manifest_and_lock.py -v
```

Expected: PASS.

### Task 5: Exercise End-to-End Training and Checkpoint Compatibility

**Files:**
- Modify: `tests/integration/test_numeric_commands.py`
- Modify: `tests/models/test_self_modifying_forecast.py`

**Interfaces:**
- Consumes: profile registration, model/task construction, and standard `Trainer` checkpoint behavior from Tasks 1-4.
- Produces: a proof that a real managed numeric run trains, persists trajectory metrics, and loads into the existing `forecast.py` export path.

- [ ] **Step 1: Write the failing end-to-end test**

```python
def test_self_modifying_numeric_profile_trains_and_exports_one_step_forecasts(tmp_path: Path) -> None:
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    write_numeric_bundle(dataset)
    assert load_entry("prepare")([str(dataset)]) == 0

    assert load_entry("train")([
        str(dataset), "--runs-dir", str(runs), "--name", "self-modifying", "--epochs", "1", "--batch-size", "2",
        "--device", "cpu", "--num-workers", "0", "--deterministic", "--seed", "7",
        "--model-profile", str(ROOT / "model-profiles" / "forecast" / "self-modifying-small.yaml"),
    ]) == 0

    run = runs / "exp1-self-modifying"
    metrics = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    checkpoint = torch.load(run / "checkpoints" / "final.pt", weights_only=False)
    assert "trajectory_mae" in metrics["train"]
    assert "final_mae" in metrics["validation"]
    assert checkpoint["model"]

    output = tmp_path / "forecasts.jsonl"
    assert load_entry("forecast")([str(run), "--checkpoint", "final", "--split", "test", "--output", str(output)]) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
```

- [ ] **Step 2: Run the end-to-end test to verify it fails before wiring is complete**

Run:

```sh
uv run pytest tests/integration/test_numeric_commands.py::test_self_modifying_numeric_profile_trains_and_exports_one_step_forecasts -v
```

Expected: FAIL until Task 4 is complete; after Task 4 it must PASS.

- [ ] **Step 3: Add a parameter-count regression test for the checked-in profile**

```python
def test_self_modifying_small_profile_dynamic_predictor_stays_in_small_model_budget() -> None:
    model = model_registry.create(
        "forecast", "self-modifying",
        feature_count=4,
        target_count=1,
        horizon_count=1,
        target_feature_indices=(0,),
        predictor_hidden_dims=(64, 64),
        meta_hidden_dim=64,
        update_scale=0.01,
    )

    assert 6_000 <= model.parameter_count <= 10_000
```

- [ ] **Step 4: Run focused model/task/integration verification**

Run:

```sh
uv run pytest tests/models/test_self_modifying_forecast.py tests/tasks/test_self_modifying_forecast.py tests/integration/test_numeric_commands.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run:

```sh
uv run pytest -q
```

Expected: PASS with no modifications to existing test expectations outside documented task metrics.

### Task 6: Document the New Model and Its Experiment Contract

**Files:**
- Modify: `docs/MODELS.md`
- Modify: `docs/TASKS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: final model and task contracts from Tasks 1-5.
- Produces: accurate developer-facing model, objective, and command documentation.

- [ ] **Step 1: Add the model specification**

Add `self-modifying` to the forecast registry table. Add a dedicated section explaining:

```text
- theta_initial is a trainable prior, not per-run mutable module state.
- Each batch row gets an independent theta_0 copy.
- The prediction at position t precedes the delta update at that position.
- theta_t is a graph Tensor, so loss gradients traverse the full state trajectory.
- Forecast export still consumes the final one-step forecast tensor.
- The current model accepts only horizon [1].
```

Include constructor parameters and output tensor shapes from **Dynamic Model Contract**.

- [ ] **Step 2: Add the task specification**

Add `SelfModifyingForecastTask` to `docs/TASKS.md`, documenting the shifted-input target construction, linear weights `t / (L - 1)`, final direct forecast term, raw direct forecast metrics, normalized trajectory metric, and validation restrictions.

- [ ] **Step 3: Add a runnable README example**

Add this concise command after the numeric forecasting example:

```sh
uv run goldfish train data/fourier --name fourier-self-modifying --epochs 10 --batch-size 32 \
  --model-profile model-profiles/forecast/self-modifying-small.yaml
```

State that the self-modifying profile requires a dataset manifest with `window.horizons: [1]` and targets present in `schema.features`.

- [ ] **Step 4: Validate docs references and final code state**

Run:

```sh
grep -RIn "self-modifying\|SelfModifying" README.md docs goldfish tests model-profiles
uv run pytest -q
```

Expected: documentation paths and names match the implementation, and the full test suite passes.

## Plan Self-Review

- **Coverage:** Tasks 1-2 define and register fully dynamic parameter state; Task 3 implements the prescribed weighted trajectory and final objective; Task 4 bridges existing numeric window semantics to target feature indices and safely restricts horizon usage; Task 5 validates managed training, checkpointing, and export; Task 6 documents the public contract.
- **No placeholder scan:** Completed. All planned files, interfaces, validation conditions, test bodies, and commands are explicit.
- **Type consistency:** `target_feature_indices`, `trajectory_forecast`, `forecast`, `SelfModifyingForecastTask`, and `parameter_count` use the same names and tensor contracts throughout.
- **Scope:** The plan intentionally excludes rollout beyond one direct horizon, variable-length sequences, and changes to `Trainer`; all are independent follow-up work after validating causal self-modification.
