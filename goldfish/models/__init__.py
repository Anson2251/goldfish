"""Model components and modality-specific model compositions."""

from .components import GRUBackbone, LSTMBackbone, RecurrentState
from .forecast import GRUForecastModel, LSTMForecastModel
from .language import GRULanguageModel, LSTMLanguageModel, TokenBatch
from .registry import ModelRegistry, model_registry

model_registry.register("language", "gru", GRULanguageModel)
model_registry.register("language", "lstm", LSTMLanguageModel)
model_registry.register("forecast", "gru", GRUForecastModel)
model_registry.register("forecast", "lstm", LSTMForecastModel)

__all__ = [
    "GRUBackbone",
    "GRULanguageModel",
    "GRUForecastModel",
    "LSTMBackbone",
    "LSTMForecastModel",
    "LSTMLanguageModel",
    "ModelRegistry",
    "RecurrentState",
    "TokenBatch",
    "model_registry",
]
