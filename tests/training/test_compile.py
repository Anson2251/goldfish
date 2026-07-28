import torch
from torch import nn

from goldfish.training import CompiledModel, compile_model


def test_compiled_model_preserves_uncompiled_checkpoint_keys() -> None:
    model = nn.Linear(2, 1)
    compiled = compile_model(model)

    assert isinstance(compiled, CompiledModel)
    assert set(compiled.state_dict()) == set(model.state_dict())

    restored = nn.Linear(2, 1)
    restored.load_state_dict(compiled.state_dict())
    torch.testing.assert_close(restored.weight, model.weight)
    torch.testing.assert_close(restored.bias, model.bias)
