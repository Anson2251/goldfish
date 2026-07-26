"""Model components and modality-specific model compositions."""

from .components import GRUBackbone, LSTMBackbone, RecurrentState
from .language import GRULanguageModel, LSTMLanguageModel, TokenBatch
from .registry import ModelRegistry, model_registry

model_registry.register("language", "gru", GRULanguageModel)
model_registry.register("language", "lstm", LSTMLanguageModel)

__all__ = [
    "GRUBackbone",
    "GRULanguageModel",
    "LSTMBackbone",
    "LSTMLanguageModel",
    "ModelRegistry",
    "RecurrentState",
    "TokenBatch",
    "model_registry",
]
