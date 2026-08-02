"""Model components and modality-specific model compositions."""

from .components import DeltaNetBackbone, DoublyStochasticMixer, GRUBackbone, LSTMBackbone, RecurrentState, UnconstrainedMixer, delta_rule_scan
from .forecast import DeltaNetForecastModel, GRUForecastModel, InterLayerCommunicationMultiHeadLSTMForecastModel, LSTMForecastModel, LatentCommunicationMultiHeadLSTMForecastModel, MultiHeadLSTMForecastModel, UnconstrainedMultiHeadLSTMForecastModel
from .language import GRULanguageModel, LSTMLanguageModel, TokenBatch
from .registry import ModelRegistry, model_registry

model_registry.register("language", "gru", GRULanguageModel)
model_registry.register("language", "lstm", LSTMLanguageModel)
model_registry.register("forecast", "gru", GRUForecastModel)
model_registry.register("forecast", "lstm", LSTMForecastModel)
# Compatibility alias for managed runs created before model profiles.
model_registry.register("forecast", "lstm-128x2", LSTMForecastModel)
model_registry.register("forecast", "deltanet", DeltaNetForecastModel)
model_registry.register("forecast", "multihead-lstm", MultiHeadLSTMForecastModel)
model_registry.register("forecast", "multihead-lstm-unconstrained", UnconstrainedMultiHeadLSTMForecastModel)
model_registry.register("forecast", "multihead-lstm-distinct", MultiHeadLSTMForecastModel)
model_registry.register("forecast", "multihead-lstm-communication", InterLayerCommunicationMultiHeadLSTMForecastModel)
model_registry.register("forecast", "multihead-lstm-latent-communication", LatentCommunicationMultiHeadLSTMForecastModel)

__all__ = [
    "DeltaNetBackbone",
    "DeltaNetForecastModel",
    "DoublyStochasticMixer",
    "UnconstrainedMixer",
    "GRUBackbone",
    "GRULanguageModel",
    "GRUForecastModel",
    "InterLayerCommunicationMultiHeadLSTMForecastModel",
    "LatentCommunicationMultiHeadLSTMForecastModel",
    "LSTMBackbone",
    "LSTMForecastModel",
    "LSTMLanguageModel",
    "MultiHeadLSTMForecastModel",
    "UnconstrainedMultiHeadLSTMForecastModel",
    "ModelRegistry",
    "RecurrentState",
    "TokenBatch",
    "delta_rule_scan",
    "model_registry",
]
