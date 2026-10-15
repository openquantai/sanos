from __future__ import annotations

from dataclasses import replace

import torch

from sanos.models.dysanos.config import DailyFrontGeometryConfig, DirectModelConfig
from sanos.models.dysanos.dynamics import (
    DirectDynamicsModel,
    affine_svd_initialization,
    dynamics_negative_log_likelihood,
    factor_gauge_penalty,
    initialize_stable_dynamics,
)
from sanos.models.dysanos.surface import native_state_width


def _model_dimensions() -> tuple[DailyFrontGeometryConfig, DirectModelConfig]:
    geometry = replace(
        DailyFrontGeometryConfig(),
        model_business_dtes=(1.0, 5.0, 10.0),
        num_inner_strikes=10,
    )
    config = replace(
        DirectModelConfig(),
        state_size=2,
        decoded_state_size=native_state_width(geometry),
    )
    return geometry, config


def test_float64_svd_is_sign_deterministic_and_population_whitened() -> None:
    generator = torch.Generator().manual_seed(4)
    factors = torch.randn((12, 2), generator=generator, dtype=torch.float64)
    loading = torch.randn((2, 7), generator=generator, dtype=torch.float64)
    values = factors @ loading + torch.arange(7, dtype=torch.float64)
    first = affine_svd_initialization(values, 2)
    second = affine_svd_initialization(values, 2)
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.decoder_weight, second.decoder_weight)
    torch.testing.assert_close(first.states.mean(dim=0), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        first.states.T @ first.states / len(first.states),
        torch.eye(2, dtype=torch.float64),
    )
    assert first.reconstruction_max_abs < 1.0e-12


def test_stable_weekly_initialization_and_losses() -> None:
    geometry, config = _model_dimensions()
    generator = torch.Generator().manual_seed(7)
    native = torch.randn(
        (14, native_state_width(geometry)), generator=generator, dtype=torch.float64
    )
    initialization = affine_svd_initialization(native, config.state_size)
    model = DirectDynamicsModel(14, geometry, config, dtype=torch.float64)
    model.install_affine_initialization(initialization)
    origin = torch.arange(11)
    endpoint = origin + 1
    evidence = initialize_stable_dynamics(model, origin, endpoint)
    assert evidence.pair_count == 11
    assert evidence.projected_minimum_real_eigenvalue >= (
        config.minimum_mean_reversion_per_week - 1.0e-12
    )
    assert evidence.projected_spectral_radius < 1.0
    assert torch.isfinite(dynamics_negative_log_likelihood(model, origin, endpoint))
    assert factor_gauge_penalty(model.daily_states) < 1.0e-12


def test_gauge_uses_frobenius_sum_per_dimension() -> None:
    states = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    assert factor_gauge_penalty(states) == 1.0
