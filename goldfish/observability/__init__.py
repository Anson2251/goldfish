"""Training observability: lifecycle events, probes, hooks, and recorders."""

from .activation import ActivationStatsProbe
from .communication import CommunicationStateProbe
from .discovery import discover_modules
from .events import HookContext, ProbePhase, TrainingHook
from .hooks import ProbeHook, build_manifest, build_probe_hook
from .mixer import MixerStateProbe
from .probes import Probe, ProbeRegistry, probe_registry
from .recorder import JsonlRecorder, SCHEMA_VERSION
from .reference import take_first_batches

probe_registry.register("mixer-state", MixerStateProbe)
probe_registry.register("communication-state", CommunicationStateProbe)
probe_registry.register("activation-stats", ActivationStatsProbe)

__all__ = [
    "ActivationStatsProbe",
    "CommunicationStateProbe",
    "HookContext",
    "JsonlRecorder",
    "MixerStateProbe",
    "Probe",
    "ProbeHook",
    "ProbePhase",
    "ProbeRegistry",
    "SCHEMA_VERSION",
    "TrainingHook",
    "build_manifest",
    "build_probe_hook",
    "discover_modules",
    "probe_registry",
    "take_first_batches",
]
