"""Model components and modality-specific model compositions."""

from .components import DoublyStochasticMixer, GRUBackbone, LSTMBackbone, RecurrentState, UnconstrainedMixer
from .forecast import GRUForecastModel, LSTMForecastModel, MultiHeadLSTMForecastModel, UnconstrainedMultiHeadLSTMForecastModel
from .language import GRULanguageModel, LSTMLanguageModel, TokenBatch
from .registry import ModelRegistry, model_registry

model_registry.register("language", "gru", GRULanguageModel)
model_registry.register("language", "lstm", LSTMLanguageModel)
model_registry.register("forecast", "gru", GRUForecastModel)
model_registry.register("forecast", "lstm", LSTMForecastModel)
# Compatibility alias for managed runs created before model profiles.
model_registry.register("forecast", "lstm-128x2", LSTMForecastModel)
model_registry.register("forecast", "multihead-lstm", MultiHeadLSTMForecastModel)
model_registry.register("forecast", "multihead-lstm-unconstrained", UnconstrainedMultiHeadLSTMForecastModel)

__all__ = [
    "DoublyStochasticMixer",
    "UnconstrainedMixer",
    "GRUBackbone",
    "GRULanguageModel",
    "GRUForecastModel",
    "LSTMBackbone",
    "LSTMForecastModel",
    "LSTMLanguageModel",
    "MultiHeadLSTMForecastModel",
    "UnconstrainedMultiHeadLSTMForecastModel",
    "ModelRegistry",
    "RecurrentState",
    "TokenBatch",
    "model_registry",
]
