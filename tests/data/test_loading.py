import torch

from goldfish.data.loading import resolve_loader_settings


def test_auto_loader_settings_reserve_twenty_percent_and_split_remaining_sixty_forty() -> None:
    settings = resolve_loader_settings(device="cuda", cpu_count=20)

    assert settings.train_workers == 9
    assert settings.validation_workers == 7
    assert settings.test_workers == 7
    assert settings.pin_memory is True
    assert settings.prefetch_factor == 2
    assert settings.persistent_workers is True


def test_explicit_loader_overrides_and_cpu_transfer_settings() -> None:
    settings = resolve_loader_settings(
        device=torch.device("cpu"), num_workers=10, train_workers=8, validation_workers=2, prefetch_factor=4,
    )

    assert settings.train_workers == 8
    assert settings.validation_workers == 2
    assert settings.pin_memory is False
    assert settings.prefetch_factor == 4
    assert settings.persistent_workers is True


def test_zero_workers_disables_prefetch_and_persistent_workers() -> None:
    settings = resolve_loader_settings(device="cpu", num_workers=0)

    assert settings.train_workers == 0
    assert settings.validation_workers == 0
    assert settings.prefetch_factor is None
    assert settings.persistent_workers is False
