"""Canonical Direct-200 state decoder and stable weekly P dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from cdxcore.version import version

from sanos.models.dysanos.surface import (
    Geometry,
    NativeParameters,
    SanosSurface,
    SanosSurfaceDecoder,
    bound_native_state,
    native_state_width,
)


class ModelConfig(Protocol):
    @property
    def state_size(self) -> int: ...

    @property
    def decoded_state_size(self) -> int: ...

    @property
    def minimum_mean_reversion_per_week(self) -> float: ...

    @property
    def innovation_std_floor(self) -> float: ...


@version("1.0", dependencies=[native_state_width])
class NativeDailyModel(torch.nn.Module):
    """Independent native raw state for deterministic per-day initialization."""

    def __init__(
        self,
        num_days: int,
        geometry: Geometry,
        initial_raw: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if num_days <= 0:
            raise ValueError("num_days must be positive")
        width = native_state_width(geometry)
        initial = initial_raw.detach().to(dtype=dtype, device=device)
        if initial.shape != (num_days, width) or not bool(torch.all(torch.isfinite(initial))):
            raise ValueError("Native initial raw states have invalid shape or values")
        self.raw_embedding = torch.nn.Embedding(
            num_days, width, sparse=True, dtype=dtype, device=device
        )
        with torch.no_grad():
            self.raw_embedding.weight.copy_(initial)

    @property
    def raw_states(self) -> torch.Tensor:
        return self.raw_embedding.weight

    def forward(self, day_indices: torch.Tensor) -> torch.Tensor:
        return self.raw_embedding(day_indices)


@version("1.0")
@dataclass(frozen=True, slots=True)
class AffineInitialization:
    """Float64 SVD factors that exactly define one affine decoder."""

    states: torch.Tensor
    decoder_weight: torch.Tensor
    decoder_bias: torch.Tensor
    singular_values: torch.Tensor
    reconstruction_max_abs: float
    projected_raw_states: torch.Tensor


@version("1.0")
@dataclass(frozen=True, slots=True)
class DynamicsInitialization:
    """Auditable stable projection of the empirical weekly transition."""

    pair_count: int
    unrestricted_spectral_radius: float
    projected_minimum_real_eigenvalue: float
    projected_spectral_radius: float


@version("1.0", dependencies=[AffineInitialization])
def affine_svd_initialization(
    native_states: torch.Tensor,
    state_size: int,
) -> AffineInitialization:
    """Standardize an affine Direct state with deterministic float64 SVD."""

    if native_states.ndim != 2 or not 0 < state_size <= min(native_states.shape):
        raise ValueError("Invalid native-state matrix or Direct state size")
    values = native_states.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("Native states contain non-finite values")
    mean = values.mean(dim=0)
    centered = values - mean
    u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    u = u[:, :state_size]
    singular = singular[:state_size]
    vh = vh[:state_size]
    # Fix the otherwise arbitrary SVD sign using the largest absolute loading.
    pivots = torch.argmax(torch.abs(vh), dim=1)
    signs = torch.sign(vh[torch.arange(state_size), pivots])
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    u = u * signs
    vh = vh * signs[:, None]
    scale = singular / math.sqrt(len(values))
    if bool(scale[-1] <= math.sqrt(torch.finfo(torch.float64).eps)):
        raise ValueError("Native states are rank deficient at the requested state size")
    # Population whitening is the historical Direct gauge: E[z]=0, E[zz']=I.
    population_scale = float(len(values)) ** 0.5
    states = u * population_scale
    decoder_weight = scale[:, None] * vh
    reconstructed = states @ decoder_weight + mean
    residual = float(torch.max(torch.abs(reconstructed - values)))
    return AffineInitialization(
        states=states,
        decoder_weight=decoder_weight.T.contiguous(),
        decoder_bias=mean,
        singular_values=singular,
        reconstruction_max_abs=residual,
        projected_raw_states=reconstructed,
    )


@version(
    "1.0",
    dependencies=[
        SanosSurfaceDecoder,
        affine_svd_initialization,
        bound_native_state,
        native_state_width,
    ],
)
class DirectDynamicsModel(torch.nn.Module):
    """Affine Direct factors with stable continuous-time weekly dynamics."""

    def __init__(
        self,
        num_days: int,
        geometry: Geometry,
        config: ModelConfig,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if num_days <= 0 or config.state_size <= 0:
            raise ValueError("Direct model dimensions must be positive")
        expected_width = native_state_width(geometry)
        if config.decoded_state_size != expected_width:
            raise ValueError(
                f"Direct decoded width must be {expected_width}, "
                f"found {config.decoded_state_size}"
            )
        self.geometry = geometry
        self.config = config
        self.daily_states = torch.nn.Parameter(
            torch.zeros((num_days, config.state_size), dtype=dtype, device=device)
        )
        self.decoder = torch.nn.Linear(
            config.state_size,
            expected_width,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.surface_decoder = SanosSurfaceDecoder(geometry, dtype=dtype, device=device)
        self.damping_factor = torch.nn.Parameter(
            torch.zeros((config.state_size, config.state_size), dtype=dtype, device=device)
        )
        self.rotation_raw = torch.nn.Parameter(torch.zeros_like(self.damping_factor))
        self.equilibrium_mean = torch.nn.Parameter(
            torch.zeros(config.state_size, dtype=dtype, device=device)
        )
        self.innovation_cholesky_raw = torch.nn.Parameter(
            torch.zeros_like(self.damping_factor)
        )

    def install_affine_initialization(self, initialization: AffineInitialization) -> None:
        if initialization.states.shape != self.daily_states.shape:
            raise ValueError("SVD states do not match the Direct model")
        if initialization.decoder_weight.shape != self.decoder.weight.shape:
            raise ValueError("SVD decoder does not match the Direct model")
        with torch.no_grad():
            self.daily_states.copy_(initialization.states.to(self.daily_states))
            self.decoder.weight.copy_(initialization.decoder_weight.to(self.decoder.weight))
            self.decoder.bias.copy_(initialization.decoder_bias.to(self.decoder.bias))
            replay = self.decoder(self.daily_states)
            expected = initialization.projected_raw_states.to(replay)
            tolerance = 1.0e-6 * max(1.0, float(expected.abs().max()))
            if float(torch.max(torch.abs(replay - expected))) > tolerance:
                raise RuntimeError("Installing the affine gauge changed projected raw states")

    def mean_reversion_generator(self) -> torch.Tensor:
        lower = torch.tril(self.damping_factor)
        symmetric = lower @ lower.T
        rotation = 0.5 * (self.rotation_raw - self.rotation_raw.T)
        identity = torch.eye(
            self.config.state_size,
            dtype=self.daily_states.dtype,
            device=self.daily_states.device,
        )
        return (
            symmetric
            + rotation
            + self.config.minimum_mean_reversion_per_week * identity
        )

    def weekly_transition(self) -> torch.Tensor:
        return torch.matrix_exp(-self.mean_reversion_generator())

    def innovation_cholesky(self) -> torch.Tensor:
        diagonal = self.config.innovation_std_floor + torch.nn.functional.softplus(
            torch.diagonal(self.innovation_cholesky_raw)
        )
        return torch.tril(self.innovation_cholesky_raw, diagonal=-1) + torch.diag(diagonal)

    def raw_states(self, day_indices: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.daily_states[day_indices])

    def decode(
        self,
        day_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeParameters, SanosSurface]:
        raw = self.raw_states(day_indices)
        parameters = bound_native_state(raw, self.geometry)
        return raw, parameters, self.surface_decoder(parameters)

    def call_prices(
        self,
        day_indices: torch.Tensor,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
    ) -> torch.Tensor:
        _, _, surface = self.decode(day_indices)
        return self.surface_decoder.price_calls(surface, expiries_y, strikes)

    def predicted_weekly_states(self, origin_indices: torch.Tensor) -> torch.Tensor:
        centered = self.daily_states[origin_indices] - self.equilibrium_mean
        return self.equilibrium_mean + centered @ self.weekly_transition()


@version("1.0", dependencies=[DirectDynamicsModel, DynamicsInitialization])
def initialize_stable_dynamics(
    model: DirectDynamicsModel,
    origin_indices: torch.Tensor,
    endpoint_indices: torch.Tensor,
) -> DynamicsInitialization:
    """Project a Float64 empirical weekly transition into the stable model."""

    if origin_indices.shape != endpoint_indices.shape or origin_indices.ndim != 1:
        raise ValueError("Weekly pair indices must be equal one-dimensional vectors")
    if len(origin_indices) <= model.config.state_size:
        raise ValueError("Stable dynamics initialization needs more pairs than factors")
    states = model.daily_states.detach().cpu().to(torch.float64)
    origin = states[origin_indices.detach().cpu()]
    endpoint = states[endpoint_indices.detach().cpu()]
    design = torch.cat((torch.ones((len(origin), 1), dtype=torch.float64), origin), dim=1)
    coefficients = torch.linalg.lstsq(design, endpoint).solution
    unrestricted = coefficients[1:]
    approximate = torch.eye(model.config.state_size, dtype=torch.float64) - unrestricted
    symmetric = 0.5 * (approximate + approximate.T)
    skew = 0.5 * (approximate - approximate.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    minimum = model.config.minimum_mean_reversion_per_week
    projected_values = torch.clamp(eigenvalues, min=minimum + 1.0e-5)
    projected_symmetric = (eigenvectors * projected_values.unsqueeze(0)) @ eigenvectors.T
    damping = torch.linalg.cholesky(
        projected_symmetric - minimum * torch.eye(model.config.state_size, dtype=torch.float64)
    )
    generator = projected_symmetric + skew
    transition = torch.matrix_exp(-generator)
    residual = endpoint - origin @ transition
    covariance = residual.T @ residual / len(residual)
    covariance += 1.0e-6 * torch.eye(model.config.state_size, dtype=torch.float64)
    innovation = torch.linalg.cholesky(covariance)
    adjusted = torch.clamp(
        torch.diagonal(innovation) - model.config.innovation_std_floor,
        min=torch.finfo(torch.float64).eps,
    )
    inverse_softplus = adjusted + torch.log(-torch.expm1(-adjusted))
    raw_innovation = torch.tril(innovation, diagonal=-1) + torch.diag(inverse_softplus)
    with torch.no_grad():
        model.damping_factor.copy_(damping.to(model.damping_factor))
        model.rotation_raw.copy_(skew.to(model.rotation_raw))
        model.equilibrium_mean.zero_()
        model.innovation_cholesky_raw.copy_(
            raw_innovation.to(model.innovation_cholesky_raw)
        )
    return DynamicsInitialization(
        pair_count=len(origin),
        unrestricted_spectral_radius=float(
            torch.max(torch.abs(torch.linalg.eigvals(unrestricted)))
        ),
        projected_minimum_real_eigenvalue=float(
            torch.min(torch.linalg.eigvals(generator).real)
        ),
        projected_spectral_radius=float(
            torch.max(torch.abs(torch.linalg.eigvals(transition)))
        ),
    )


@version("1.0", dependencies=[DirectDynamicsModel])
def dynamics_negative_log_likelihood(
    model: DirectDynamicsModel,
    origin_indices: torch.Tensor,
    endpoint_indices: torch.Tensor,
) -> torch.Tensor:
    """Gaussian weekly transition loss with a nonsingular covariance."""

    if origin_indices.shape != endpoint_indices.shape or origin_indices.ndim != 1:
        raise ValueError("Weekly pair indices must be equal one-dimensional vectors")
    if len(origin_indices) == 0:
        raise ValueError("At least one weekly pair is required")
    residual = model.daily_states[endpoint_indices] - model.predicted_weekly_states(
        origin_indices
    )
    cholesky = model.innovation_cholesky()
    whitened = torch.linalg.solve_triangular(
        cholesky,
        residual.T,
        upper=False,
    ).T
    dimension = model.config.state_size
    return 0.5 * (
        whitened.square().sum(dim=1).mean()
        + 2.0 * torch.log(torch.diagonal(cholesky)).sum()
        + dimension * math.log(2.0 * math.pi)
    ) / dimension


@version("1.0")
def factor_gauge_penalty(states: torch.Tensor) -> torch.Tensor:
    """Keep Direct factors centered with approximately identity covariance."""

    if states.ndim != 2 or len(states) < 2:
        raise ValueError("Gauge penalty requires at least two state rows")
    centered = states - states.mean(dim=0)
    covariance = centered.T @ centered / len(states)
    identity = torch.eye(covariance.shape[0], dtype=states.dtype, device=states.device)
    dimension = covariance.shape[0]
    return states.mean(dim=0).square().mean() + (
        covariance - identity
    ).square().sum() / dimension


__all__ = [
    "AffineInitialization",
    "DynamicsInitialization",
    "DirectDynamicsModel",
    "NativeDailyModel",
    "affine_svd_initialization",
    "dynamics_negative_log_likelihood",
    "factor_gauge_penalty",
    "initialize_stable_dynamics",
]
