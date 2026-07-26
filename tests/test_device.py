import pytest
import torch

from goldfish.device import best_available_device, resolve_device


def test_explicit_cpu_is_always_available() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_best_available_device_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert best_available_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert best_available_device() == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert best_available_device() == "cpu"


def test_explicit_unavailable_accelerator_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")
