import pytest
from torch import nn

from goldfish.models.registry import ModelRegistry


class ExampleModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width


def test_registry_creates_registered_model_and_lists_sorted_names() -> None:
    registry = ModelRegistry()
    registry.register("language", "lstm", ExampleModel)
    registry.register("language", "gru", ExampleModel)

    model = registry.create("language", "GRU", width=8)

    assert isinstance(model, ExampleModel)
    assert model.width == 8
    assert registry.names("language") == ("gru", "lstm")


def test_registry_rejects_duplicate_or_unknown_models() -> None:
    registry = ModelRegistry()
    registry.register("language", "gru", ExampleModel)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("language", "gru", ExampleModel)
    with pytest.raises(ValueError, match="available: gru"):
        registry.create("language", "lstm", width=8)
