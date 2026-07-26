"""Core, modality-agnostic Goldfish contracts."""

from .batch import Batch
from .output import ModelOutput
from .task import StepResult, Task

__all__ = ["Batch", "ModelOutput", "StepResult", "Task"]
