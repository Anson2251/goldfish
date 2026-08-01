"""Tests for probe configuration resolution (profile declarations + run overrides)."""

from __future__ import annotations

from pathlib import Path

import pytest

from goldfish.config import load_model_profile, resolve_observability_config
from goldfish.config.observability import (
    ActivationPointConfig,
    ProbeConfig,
    ReferenceConfig,
    ResolvedObservabilityConfig,
    ScheduleConfig,
    TensorStatConfig,
)

MIXER_PROFILE = {
    "probes": [
        {"name": "mixer-state"},
    ]
}

LATENT_PROFILE = {
    "probes": [
        {
            "name": "communication-state",
            "include": ["latent_communications.*"],
            "include_grad_norms": True,
            "head_dim": 8,
        },
        {
            "name": "activation-stats",
            "points": [{"path": "latent_communications.*", "quantity": "message-magnitude"}],
        },
    ]
}


def test_resolve_without_observability_is_disabled() -> None:
    config = resolve_observability_config(None, None)

    assert config.reference is None
    assert config.probes == ()


def test_resolve_enabled_without_profile_declarations_is_empty() -> None:
    config = resolve_observability_config(None, {"enabled": True})

    assert config.reference is None
    assert config.probes == ()


def test_profile_mixer_state_applies_defaults() -> None:
    config = resolve_observability_config(MIXER_PROFILE, {"enabled": True})

    assert config.probes == (
        ProbeConfig(
            name="mixer-state",
            include=("mixer", "mixers.*"),
            schedule=ScheduleConfig(every_n_epochs=1),
        ),
    )


def test_profile_latent_declaration_resolves_points_and_options() -> None:
    config = resolve_observability_config(LATENT_PROFILE, {"enabled": True, "reference": {"split": "val", "batches": 8}})

    assert config.probes == (
        ProbeConfig(
            name="communication-state",
            include=("latent_communications.*",),
            include_grad_norms=True,
            head_dim=8,
            schedule=ScheduleConfig(every_n_epochs=1),
        ),
        ProbeConfig(
            name="activation-stats",
            points=(ActivationPointConfig(path="latent_communications.*", quantity="message-magnitude"),),
            schedule=ScheduleConfig(every_n_epochs=1),
        ),
    )


def test_declarative_tensor_point_resolves() -> None:
    profile = {
        "probes": [
            {
                "name": "activation-stats",
                "points": [
                    {
                        "path": "head_layers.*.1",
                        "tensors": [{"name": "output", "stats": ["norm", "p95"], "reduce": "per_head"}],
                    }
                ],
            }
        ]
    }

    config = resolve_observability_config(profile, {"enabled": True, "reference": {"split": "val", "batches": 8}})

    point = config.probes[0].points[0]
    assert point.quantity is None
    assert point.tensors == (TensorStatConfig(name="output", stats=("norm", "p95"), reduce="per_head"),)


def test_run_reference_is_resolved() -> None:
    config = resolve_observability_config(
        None,
        {"enabled": True, "reference": {"split": "val", "batches": 8}},
    )

    assert config.reference == ReferenceConfig(split="val", batches=8, selection="first")


def test_run_override_replaces_profile_declaration_entirely() -> None:
    profile = {
        "probes": [
            {
                "name": "activation-stats",
                "points": [{"path": "mixer", "quantity": "mixing-displacement"}],
                "every_n_epochs": 1,
            }
        ]
    }
    run = {
        "enabled": True,
        "reference": {"split": "val", "batches": 8},
        "probes": [
            {
                "name": "activation-stats",
                "points": [{"path": "head_layers.*.0", "tensors": [{"name": "output", "stats": ["norm"]}]}],
                "epochs": [1, 10, 100],
            }
        ],
    }

    config = resolve_observability_config(profile, run)

    probe = config.probes[0]
    assert probe.points == (
        ActivationPointConfig(path="head_layers.*.0", tensors=(TensorStatConfig(name="output", stats=("norm",)),)),
    )
    assert probe.schedule == ScheduleConfig(epochs=(1, 10, 100), every_n_epochs=None)
    assert config.reference == ReferenceConfig(split="val", batches=8)


def test_run_override_keeps_unoverridden_profile_probes() -> None:
    run = {
        "enabled": True,
        "reference": {"split": "val", "batches": 8},
        "probes": [
            {
                "name": "activation-stats",
                "points": [{"path": "latent_communications.*", "quantity": "message-magnitude"}],
                "epochs": [1, 2, 5],
            }
        ],
    }

    config = resolve_observability_config(LATENT_PROFILE, run)

    names = [probe.name for probe in config.probes]
    assert names == ["communication-state", "activation-stats"]
    assert config.probes[0].include == ("latent_communications.*",)
    assert config.probes[1].schedule.epochs == (1, 2, 5)


def test_load_model_profile_preserves_observability_block(tmp_path: Path) -> None:
    profile_path = tmp_path / "model.yaml"
    profile_path.write_text(
        """\
model:
  family: forecast
  name: multihead-lstm
  parameters:
    hidden_dim: 32
observability:
  probes:
    - name: mixer-state
""",
        encoding="utf-8",
    )

    profile = load_model_profile(profile_path)

    assert profile["observability"] == {"probes": [{"name": "mixer-state"}]}


def test_resolved_config_equality_is_value_based() -> None:
    assert ResolvedObservabilityConfig(reference=None, probes=()) == ResolvedObservabilityConfig(
        reference=None, probes=()
    )


@pytest.mark.parametrize(
    ("profile", "run", "message"),
    [
        # Run-level overrides must name a probe declared by the profile.
        (MIXER_PROFILE, {"enabled": True, "probes": [{"name": "activation-stats", "points": []}]}, "not declared"),
        # Run-level probes with no profile declarations at all.
        (None, {"enabled": True, "probes": [{"name": "mixer-state"}]}, "declares no probes"),
        # disabled runs must not carry reference or probes.
        (MIXER_PROFILE, {"enabled": False, "reference": {"split": "val", "batches": 8}}, "enabled"),
        (MIXER_PROFILE, {"enabled": False, "probes": [{"name": "mixer-state"}]}, "enabled"),
    ],
)
def test_run_level_conflicts_are_rejected(profile, run, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_observability_config(profile, run)


def test_unknown_probe_name_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        resolve_observability_config({"probes": [{"name": "telemetry"}]}, {"enabled": True})


def test_unknown_probe_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        resolve_observability_config({"probes": [{"name": "mixer-state", "foo": 1}]}, {"enabled": True})


def test_schedule_forms_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_observability_config(
            {"probes": [{"name": "mixer-state", "every_n_epochs": 1, "epochs": [1, 2]}]},
            {"enabled": True},
        )


def test_epochs_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="increasing"):
        resolve_observability_config({"probes": [{"name": "mixer-state", "epochs": [5, 3]}]}, {"enabled": True})


def test_every_n_epochs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_observability_config({"probes": [{"name": "mixer-state", "every_n_epochs": 0}]}, {"enabled": True})


def test_activation_stats_requires_points() -> None:
    with pytest.raises(ValueError, match="points"):
        resolve_observability_config({"probes": [{"name": "activation-stats"}]}, {"enabled": True})


@pytest.mark.parametrize(
    "point",
    [
        # Both quantity and tensors.
        {"path": "mixer", "quantity": "mixing-displacement", "tensors": [{"name": "output", "stats": ["norm"]}]},
        # Neither.
        {"path": "mixer"},
    ],
)
def test_point_declares_exactly_one_of_quantity_and_tensors(point) -> None:
    with pytest.raises(ValueError, match="quantity.*tensors|either"):
        resolve_observability_config(
            {"probes": [{"name": "activation-stats", "points": [point]}]},
            {"enabled": True},
        )


def test_tensor_stats_must_be_known() -> None:
    with pytest.raises(ValueError, match="stat"):
        resolve_observability_config(
            {
                "probes": [
                    {
                        "name": "activation-stats",
                        "points": [{"path": "mixer", "tensors": [{"name": "output", "stats": ["bogus"]}]}],
                    }
                ]
            },
            {"enabled": True},
        )


def test_tensor_stats_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resolve_observability_config(
            {
                "probes": [
                    {
                        "name": "activation-stats",
                        "points": [{"path": "mixer", "tensors": [{"name": "output", "stats": []}]}],
                    }
                ]
            },
            {"enabled": True},
        )


def test_tensor_reduce_must_be_known() -> None:
    with pytest.raises(ValueError, match="overall.*per_head"):
        resolve_observability_config(
            {
                "probes": [
                    {
                        "name": "activation-stats",
                        "points": [{"path": "mixer", "tensors": [{"name": "output", "stats": ["norm"], "reduce": "everything"}]}],
                    }
                ]
            },
            {"enabled": True},
        )


def test_communication_state_requires_include() -> None:
    with pytest.raises(ValueError, match="include"):
        resolve_observability_config({"probes": [{"name": "communication-state"}]}, {"enabled": True})


def test_head_dim_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_observability_config(
            {"probes": [{"name": "communication-state", "include": ["communications.*"], "head_dim": 0}]},
            {"enabled": True},
        )


def test_activation_probes_require_reference() -> None:
    with pytest.raises(ValueError, match="reference"):
        resolve_observability_config(LATENT_PROFILE, {"enabled": True})


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ({"split": "train", "batches": 8}, "val.*test"),
        ({"split": "val", "batches": 0}, "positive"),
        ({"split": "val", "batches": 8, "selection": "seeded"}, "first"),
        ({"split": "val", "batches": 8, "foo": 1}, "unknown field"),
    ],
)
def test_reference_configuration_is_validated(reference, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_observability_config(None, {"enabled": True, "reference": reference})


def test_run_observability_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        resolve_observability_config(MIXER_PROFILE, {"enabled": True, "foo": 1})


def test_profile_observability_may_only_declare_probes() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        resolve_observability_config({"reference": {"split": "val", "batches": 8}}, {"enabled": True})


def test_duplicate_probe_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        resolve_observability_config({"probes": [{"name": "mixer-state"}, {"name": "mixer-state"}]}, {"enabled": True})


def test_include_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="list"):
        resolve_observability_config(
            {"probes": [{"name": "communication-state", "include": "communications.*"}]},
            {"enabled": True},
        )


def test_points_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="list"):
        resolve_observability_config(
            {"probes": [{"name": "activation-stats", "points": {"path": "mixer", "quantity": "mixing-displacement"}}]},
            {"enabled": True},
        )


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "model-profiles" / "forecast"


def _resolve_profile(filename: str):
    profile = load_model_profile(PROFILE_DIR / filename)
    return resolve_observability_config(
        profile.get("observability"),
        {"enabled": True, "reference": {"split": "val", "batches": 8}},
    )


def test_real_mixer_profile_declares_mixer_state_probes() -> None:
    config = _resolve_profile("multihead-lstm-small.yaml")

    assert [probe.name for probe in config.probes] == ["mixer-state", "activation-stats"]
    assert config.probes[0].include == ("mixer", "mixers.*")
    assert config.probes[1].points == (ActivationPointConfig(path="mixer", quantity="mixing-displacement"),)


def test_real_dense_profile_declares_communication_probes() -> None:
    config = _resolve_profile("multihead-lstm-small-communication.yaml")

    assert [probe.name for probe in config.probes] == ["communication-state", "activation-stats"]
    assert config.probes[0].include == ("communications.*",)
    assert config.probes[1].points == (ActivationPointConfig(path="communications.*", quantity="dense-displacement"),)


def test_real_latent_profile_declares_message_magnitude_probe() -> None:
    config = _resolve_profile("multihead-lstm-small-latent-communication.yaml")

    assert [probe.name for probe in config.probes] == ["communication-state", "activation-stats"]
    assert config.probes[0].head_dim == 8
    assert config.probes[1].points == (
        ActivationPointConfig(path="latent_communications.*", quantity="message-magnitude"),
        ActivationPointConfig(path="fusion", quantity="io-stats"),
    )
