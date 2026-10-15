"""Task implementations for supported modelling objectives."""

from .causal_lm import CausalLanguageModelTask
from .point_forecast import PointForecastTask
from .self_modifying_forecast import SelfModifyingForecastTask
from .prefix_lm import PrefixLanguageModelTask

__all__ = ["CausalLanguageModelTask", "PointForecastTask", "PrefixLanguageModelTask", "SelfModifyingForecastTask"]
