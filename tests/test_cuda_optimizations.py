from types import SimpleNamespace

import pytest
import torch

import train


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA optimization tests")


def test_cuda_performance_optimizations_enable_benchmark_and_tf32(monkeypatch) -> None:
    cudnn = SimpleNamespace(benchmark=False, deterministic=False, allow_tf32=False)
    matmul = SimpleNamespace(allow_tf32=False)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.backends, "cudnn", cudnn)
    monkeypatch.setattr(train.torch.backends.cuda, "matmul", matmul)

    train._configure_reproducibility(deterministic=False, seed=None)

    assert cudnn.benchmark is True
    assert cudnn.deterministic is False
    assert cudnn.allow_tf32 is True
    assert matmul.allow_tf32 is True


def test_deterministic_mode_disables_cuda_performance_optimizations(monkeypatch) -> None:
    cudnn = SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)
    matmul = SimpleNamespace(allow_tf32=True)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.backends, "cudnn", cudnn)
    monkeypatch.setattr(train.torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(train.torch, "use_deterministic_algorithms", lambda enabled: None)

    train._configure_reproducibility(deterministic=True, seed=7)

    assert cudnn.benchmark is False
    assert cudnn.deterministic is True
    assert cudnn.allow_tf32 is False
    assert matmul.allow_tf32 is False
