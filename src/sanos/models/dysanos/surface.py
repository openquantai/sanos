"""Differentiable canonical DySANOS 1.1 surface decoder."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from cdxcore.version import version


class Geometry(Protocol):
    """Structural subset of the canonical DySANOS geometry configuration."""

    @property
    def model_business_dtes(self) -> tuple[float, ...]: ...

    @property
    def support_epsilon_business_dte(self) -> float: ...

    @property
    def nms_range(self) -> tuple[float, float]: ...

    @property
    def num_inner_strikes(self) -> int: ...

    @property
    def min_dlvsq(self) -> float: ...

    @property
    def max_dlvsq(self) -> float: ...

    @property
    def min_forward_vol(self) -> float: ...

    @property
    def max_forward_vol(self) -> float: ...

    @property
    def min_eta(self) -> float: ...

    @property
    def max_eta(self) -> float: ...

    @property
    def lower_boundary_scale(self) -> float: ...

    @property
    def upper_boundary_scale(self) -> float: ...

    @property
    def minimum_log_spacing_fraction(self) -> float: ...

    @property
    def dlv_cap_barrier_log_width(self) -> float: ...

    @property
    def dlv_cap_barrier_log_std(self) -> float: ...


@version("1.1")
def nms_epsilon_year_fraction(business_dte: float = 0.125) -> float:
    """Convert the configured NMS support epsilon from business days to years."""

    if not math.isfinite(business_dte) or business_dte <= 0.0:
        raise ValueError("NMS support epsilon business DTE must be positive and finite")
    return business_dte / 252.0


@version("1.0")
def inner_nms_nodes(count: int, bounds: tuple[float, float]) -> np.ndarray:
    """Return increasing NMS nodes with exactly one ATM node."""

    if count < 5 or len(bounds) != 2 or not bounds[0] < 0.0 < bounds[1]:
        raise ValueError("Invalid canonical NMS geometry")
    distance = bounds[1] - bounds[0]
    below = min(max(2, int(count * abs(bounds[0]) // distance)), count - 3)
    above = count - below - 1
    nodes = np.concatenate(
        (
            np.linspace(bounds[0], 0.0, below, endpoint=False),
            np.linspace(0.0, bounds[1], above + 1),
        )
    )
    if nodes.shape != (count,) or np.count_nonzero(nodes == 0.0) != 1:
        raise RuntimeError("Canonical NMS construction failed")
    return nodes


@version("1.0", dependencies=[inner_nms_nodes])
def shared_nms_node_sets(
    count: int,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return front/final NMS candidates for one shared strike support."""

    if count % 2:
        raise ValueError("Canonical shared-strike geometry requires an even count")
    half = count // 2
    front = inner_nms_nodes(half, bounds)
    final = np.linspace(bounds[0], bounds[1], half)
    if np.any(np.isclose(final, 0.0, rtol=0.0, atol=1.0e-14)):
        edges = np.linspace(bounds[0], bounds[1], half + 1)
        final = 0.5 * (edges[:-1] + edges[1:])
    return front, final


@version("1.0")
def _space_shared_log_strikes(
    candidates: torch.Tensor,
    *,
    atm_index: int,
    minimum_spacing_fraction: float,
) -> torch.Tensor:
    ordered = torch.sort(candidates, dim=-1).values
    gap = (
        minimum_spacing_fraction
        * (ordered[..., -1] - ordered[..., 0])
        / (ordered.shape[-1] - 1)
    ).unsqueeze(-1)
    left_gaps = torch.maximum(torch.diff(ordered[..., : atm_index + 1]), gap)
    right_gaps = torch.maximum(torch.diff(ordered[..., atm_index:]), gap)
    left = -torch.flip(torch.cumsum(torch.flip(left_gaps, (-1,)), -1), (-1,))
    right = torch.cumsum(right_gaps, -1)
    return torch.cat((left, torch.zeros_like(ordered[..., :1]), right), dim=-1)


@version("1.0")
@dataclass(frozen=True, slots=True)
class NativeParameters:
    """Bounded canonical per-expiry SANOS coordinates."""

    dlv_squares: torch.Tensor
    forward_volatilities: torch.Tensor
    eta: torch.Tensor


@version("1.0")
def native_state_width(geometry: Geometry) -> int:
    return len(geometry.model_business_dtes) * (geometry.num_inner_strikes + 2)


@version("1.0", dependencies=[NativeParameters, native_state_width])
def bound_native_state(raw: torch.Tensor, geometry: Geometry) -> NativeParameters:
    """Map unconstrained expiry-major coordinates into canonical bounds."""

    width = native_state_width(geometry)
    if raw.ndim < 1 or raw.shape[-1] != width:
        raise ValueError(f"Expected native state width {width}, found {raw.shape}")
    expiry_count = len(geometry.model_business_dtes)
    node_count = geometry.num_inner_strikes
    blocks = raw.reshape(*raw.shape[:-1], expiry_count, node_count + 2)
    dlv_raw = blocks[..., :node_count]
    forward_raw = blocks[..., node_count]
    eta_raw = blocks[..., node_count + 1]
    maximum_log_distance = math.log(geometry.max_dlvsq - geometry.min_dlvsq)
    dlv_squares = geometry.min_dlvsq + torch.exp(
        torch.clamp(dlv_raw, max=maximum_log_distance)
    )
    forward = geometry.min_forward_vol + (
        geometry.max_forward_vol - geometry.min_forward_vol
    ) * torch.sigmoid(forward_raw)
    eta = geometry.min_eta + (geometry.max_eta - geometry.min_eta) * torch.sigmoid(
        eta_raw
    )
    return NativeParameters(dlv_squares, forward, eta)


@version("1.0", dependencies=[native_state_width])
def dlv_cap_penalty(raw: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """Return the mean restoring penalty for flat-gradient capped DLV nodes."""

    if raw.shape[-1] != native_state_width(geometry):
        raise ValueError("Invalid native state width for the DLV cap penalty")
    blocks = raw.reshape(
        *raw.shape[:-1],
        len(geometry.model_business_dtes),
        geometry.num_inner_strikes + 2,
    )
    maximum_log_distance = math.log(geometry.max_dlvsq - geometry.min_dlvsq)
    scaled = (
        blocks[..., : geometry.num_inner_strikes] - maximum_log_distance
    ) / geometry.dlv_cap_barrier_log_width
    barrier = (
        geometry.dlv_cap_barrier_log_width
        * torch.nn.functional.softplus(scaled)
        / geometry.dlv_cap_barrier_log_std
    )
    return 0.5 * barrier.square().sum(dim=(-2, -1)).mean()


@version("1.0", dependencies=[NativeParameters, native_state_width])
def raw_from_parameters(
    parameters: NativeParameters,
    geometry: Geometry,
    *,
    probability_floor: float = 1.0e-8,
) -> torch.Tensor:
    """Invert canonical bounds for deterministic native initialization."""

    if not 0.0 < probability_floor < 0.5:
        raise ValueError("probability_floor must lie in (0,0.5)")
    expiry_count = len(geometry.model_business_dtes)
    node_count = geometry.num_inner_strikes
    expected = parameters.dlv_squares.shape[:-2] + (expiry_count,)
    if parameters.dlv_squares.shape[-2:] != (expiry_count, node_count):
        raise ValueError("DLV parameter shape changed")
    if parameters.forward_volatilities.shape != expected or parameters.eta.shape != expected:
        raise ValueError("Forward-volatility or eta parameter shape changed")
    dlv_distance = torch.clamp(
        parameters.dlv_squares - geometry.min_dlvsq,
        min=probability_floor * (geometry.max_dlvsq - geometry.min_dlvsq),
    )
    forward_probability = torch.clamp(
        (parameters.forward_volatilities - geometry.min_forward_vol)
        / (geometry.max_forward_vol - geometry.min_forward_vol),
        probability_floor,
        1.0 - probability_floor,
    )
    eta_probability = torch.clamp(
        (parameters.eta - geometry.min_eta) / (geometry.max_eta - geometry.min_eta),
        probability_floor,
        1.0 - probability_floor,
    )
    blocks = torch.cat(
        (
            torch.log(dlv_distance),
            torch.logit(forward_probability).unsqueeze(-1),
            torch.logit(eta_probability).unsqueeze(-1),
        ),
        dim=-1,
    )
    result = blocks.flatten(start_dim=-2)
    if result.shape[-1] != native_state_width(geometry):
        raise RuntimeError("Native state packing failed")
    return result


@version("1.0")
@dataclass(frozen=True, slots=True)
class SanosSurface:
    """Decoded martingale densities and their bounded coordinates."""

    expiries_y: torch.Tensor
    support_strikes: torch.Tensor
    initial_density: torch.Tensor
    densities: torch.Tensor
    parameters: NativeParameters
    total_volatilities: torch.Tensor
    backbone_total_volatilities: torch.Tensor


@version(
    "1.0",
    dependencies=[
        NativeParameters,
        SanosSurface,
        _space_shared_log_strikes,
        nms_epsilon_year_fraction,
        shared_nms_node_sets,
    ],
)
class SanosSurfaceDecoder(torch.nn.Module):
    """Decode and price the canonical shared-strike DySANOS surface."""

    model_expiries_y: torch.Tensor
    model_time_steps_y: torch.Tensor
    front_nms: torch.Tensor
    final_nms: torch.Tensor
    _step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        geometry: Geometry,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.geometry = geometry
        expiries = torch.as_tensor(
            geometry.model_business_dtes, dtype=dtype, device=device
        ) / 252.0
        self.register_buffer("model_expiries_y", expiries)
        self.register_buffer(
            "model_time_steps_y",
            torch.diff(expiries, prepend=torch.zeros_like(expiries[:1])),
        )
        front, final = shared_nms_node_sets(
            geometry.num_inner_strikes, geometry.nms_range
        )
        self.register_buffer("front_nms", torch.as_tensor(front, dtype=dtype, device=device))
        self.register_buffer("final_nms", torch.as_tensor(final, dtype=dtype, device=device))
        self.atm_index = int(np.count_nonzero(front < 0.0) + np.count_nonzero(final < 0.0))
        self._step = self._solve_step

    def enable_solver_compile(self) -> None:
        """Compile the fixed-width differentiable recursion for production training."""

        self._step = torch.compile(self._solve_step, dynamic=True)

    def disable_solver_compile(self) -> None:
        """Restore eager arithmetic for canonical comparable evaluation."""

        self._step = self._solve_step

    @staticmethod
    def _density_from_calls(support: torch.Tensor, calls: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(calls[..., :1])
        zero = torch.zeros_like(calls[..., :1])
        slopes = torch.diff(calls, dim=-1, prepend=one) / torch.diff(
            support, dim=-1, prepend=zero
        )
        return torch.diff(slopes, dim=-1, append=zero)

    @staticmethod
    def _transfer_density(
        previous_support: torch.Tensor,
        previous_density: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        calls = torch.matmul(
            torch.relu(previous_support.unsqueeze(-2) - support.unsqueeze(-1)),
            previous_density.unsqueeze(-1),
        ).squeeze(-1)
        return SanosSurfaceDecoder._density_from_calls(support, calls)

    @staticmethod
    def _solve_step(
        w_minus: torch.Tensor,
        w_plus: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        n = rhs.shape[-1]
        if w_minus.shape != rhs.shape[:-1] + (n - 2,) or w_plus.shape != w_minus.shape:
            raise ValueError("Invalid SANOS tridiagonal system")
        cp: list[torch.Tensor] = [-w_minus[..., 0]]
        y: list[torch.Tensor] = [rhs[..., 0]]
        for index in range(1, n - 1):
            a = -w_plus[..., index - 2] if index > 1 else 0.0
            diagonal = 1.0 + w_minus[..., index - 1] + w_plus[..., index - 1]
            c = -w_minus[..., index] if index < n - 2 else 0.0
            denominator = diagonal - a * cp[index - 1]
            cp.append(c / denominator)
            y.append((rhs[..., index] - a * y[index - 1]) / denominator)
        a_last = -w_plus[..., -1]
        y.append((rhs[..., -1] - a_last * y[-1]) / (1.0 - a_last * cp[-1]))
        result: list[torch.Tensor] = [y[-1]]
        for index in range(n - 2, -1, -1):
            result.append(y[index] - cp[index] * result[-1])
        return torch.stack(tuple(reversed(result)), dim=-1)

    def _supports(self, forward_volatilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.cumsum(
            forward_volatilities * torch.sqrt(self.model_time_steps_y), dim=-1
        )
        epsilon_volatility = total[..., :1] / torch.sqrt(self.model_expiries_y[:1])
        epsilon = nms_epsilon_year_fraction(
            self.geometry.support_epsilon_business_dte
        )
        nms_scale = torch.sqrt(
            (epsilon_volatility.square() * epsilon + total.square()) / (1.0 + epsilon)
        )
        candidates = torch.cat(
            (nms_scale[..., :1] * self.front_nms, nms_scale[..., -1:] * self.final_nms),
            dim=-1,
        )
        log_strikes = _space_shared_log_strikes(
            candidates,
            atm_index=self.atm_index,
            minimum_spacing_fraction=self.geometry.minimum_log_spacing_fraction,
        )
        inner = torch.exp(log_strikes).unsqueeze(-2).expand(
            *log_strikes.shape[:-1], len(self.geometry.model_business_dtes), -1
        )
        support = torch.cat(
            (
                self.geometry.lower_boundary_scale * inner[..., :1],
                inner,
                self.geometry.upper_boundary_scale * inner[..., -1:],
            ),
            dim=-1,
        )
        return total, support

    def forward(self, parameters: NativeParameters) -> SanosSurface:
        if parameters.dlv_squares.ndim != 3:
            raise ValueError("Canonical decoder expects a batch of native parameters")
        batch, expiry_count, node_count = parameters.dlv_squares.shape
        if (expiry_count, node_count) != (
            len(self.geometry.model_business_dtes),
            self.geometry.num_inner_strikes,
        ):
            raise ValueError("Native parameter geometry changed")
        if parameters.forward_volatilities.shape != (batch, expiry_count):
            raise ValueError("Forward-volatility geometry changed")
        if parameters.eta.shape != (batch, expiry_count):
            raise ValueError("Eta geometry changed")
        total, supports = self._supports(parameters.forward_volatilities)
        intrinsic = torch.relu(1.0 - supports[:, 0])
        initial = self._density_from_calls(supports[:, 0], intrinsic)
        densities: list[torch.Tensor] = []
        previous_support = supports[:, 0]
        previous_density = initial
        for expiry_index in range(expiry_count):
            support = supports[:, expiry_index]
            rhs = (
                previous_density
                if expiry_index == 0
                else self._transfer_density(previous_support, previous_density, support)
            )
            box = (
                support[:, 1:-1].square()
                * self.model_time_steps_y[expiry_index]
                / (support[:, 2:] - support[:, :-2])
            )
            w_plus = (
                parameters.dlv_squares[:, expiry_index]
                * box
                / (support[:, 2:] - support[:, 1:-1])
            )
            w_minus = (
                parameters.dlv_squares[:, expiry_index]
                * box
                / (support[:, 1:-1] - support[:, :-2])
            )
            density = self._step(w_minus, w_plus, rhs)
            densities.append(density)
            previous_support, previous_density = support, density
        density_tensor = torch.stack(densities, dim=1)
        tolerance = max(1.0e-8, 256.0 * torch.finfo(density_tensor.dtype).eps)
        if not bool(torch.all(torch.isfinite(density_tensor))):
            raise ValueError("Decoded SANOS density is non-finite")
        if bool(torch.min(density_tensor) < -tolerance):
            raise ValueError("Decoded SANOS density is negative")
        if bool(torch.max(torch.abs(density_tensor.sum(-1) - 1.0)) > tolerance):
            raise ValueError("Decoded SANOS density does not have unit mass")
        if bool(torch.max(torch.abs((supports * density_tensor).sum(-1) - 1.0)) > tolerance):
            raise ValueError("Decoded SANOS density does not have unit mean")
        backbone = torch.cumsum(
            parameters.forward_volatilities
            * parameters.eta
            * torch.sqrt(self.model_time_steps_y),
            dim=-1,
        )
        return SanosSurface(
            self.model_expiries_y.unsqueeze(0).expand(batch, -1),
            supports,
            initial,
            density_tensor,
            parameters,
            total,
            backbone,
        )

    @staticmethod
    def _black_call(
        spot: torch.Tensor,
        strike: torch.Tensor,
        total_volatility: torch.Tensor,
    ) -> torch.Tensor:
        safe = torch.clamp(total_volatility, min=1.0e-10)
        d1 = (torch.log(spot / strike) + 0.5 * safe.square()) / safe
        d2 = d1 - safe
        value = spot * torch.special.ndtr(d1) - strike * torch.special.ndtr(d2)
        return torch.where(total_volatility > 0.0, value, torch.relu(spot - strike))

    def price_calls(
        self,
        surface: SanosSurface,
        expiries_y: torch.Tensor,
        strikes: torch.Tensor,
    ) -> torch.Tensor:
        """Price a padded batch using density/variance interpolation."""

        if expiries_y.shape != strikes.shape or strikes.ndim != 2:
            raise ValueError("Price queries must be equal-shaped batch matrices")
        if bool(torch.any(~torch.isfinite(expiries_y))) or bool(torch.any(expiries_y < 0.0)):
            raise ValueError("Price expiries must be finite and non-negative")
        if bool(torch.any(expiries_y > self.model_expiries_y[-1] + 1.0e-12)):
            raise ValueError("Price expiry exceeds the canonical model horizon")
        batch, quote_count = strikes.shape
        raw_upper = torch.searchsorted(self.model_expiries_y, expiries_y.contiguous())
        upper = torch.clamp(raw_upper, max=len(self.model_expiries_y) - 1)
        lower = torch.clamp(raw_upper - 1, min=0)
        before_first = raw_upper == 0
        previous_time = torch.where(
            before_first, torch.zeros_like(expiries_y), self.model_expiries_y[lower]
        )
        interval = torch.clamp(
            self.model_expiries_y[upper] - previous_time,
            min=torch.finfo(expiries_y.dtype).tiny,
        )
        alpha = torch.clamp((expiries_y - previous_time) / interval, 0.0, 1.0)
        batch_index = torch.arange(batch, device=strikes.device).unsqueeze(1)
        upper_support = surface.support_strikes[batch_index, upper]
        upper_density = surface.densities[batch_index, upper]
        lower_support = surface.support_strikes[batch_index, lower]
        lower_density = surface.densities[batch_index, lower]
        first_support = surface.support_strikes[:, :1].expand(-1, quote_count, -1)
        initial = surface.initial_density[:, None].expand(-1, quote_count, -1)
        lower_support = torch.where(before_first.unsqueeze(-1), first_support, lower_support)
        lower_density = torch.where(before_first.unsqueeze(-1), initial, lower_density)
        upper_variance = surface.backbone_total_volatilities[batch_index, upper].square()
        lower_variance = surface.backbone_total_volatilities[batch_index, lower].square()
        lower_variance = torch.where(before_first, torch.zeros_like(lower_variance), lower_variance)
        total_volatility = torch.sqrt(
            torch.clamp((1.0 - alpha) * lower_variance + alpha * upper_variance, min=0.0)
        )
        upper_calls = torch.sum(
            self._black_call(upper_support, strikes.unsqueeze(-1), total_volatility.unsqueeze(-1))
            * upper_density,
            dim=-1,
        )
        lower_calls = torch.sum(
            self._black_call(lower_support, strikes.unsqueeze(-1), total_volatility.unsqueeze(-1))
            * lower_density,
            dim=-1,
        )
        return (1.0 - alpha) * lower_calls + alpha * upper_calls


__all__ = [
    "NativeParameters",
    "SanosSurface",
    "SanosSurfaceDecoder",
    "bound_native_state",
    "dlv_cap_penalty",
    "inner_nms_nodes",
    "native_state_width",
    "nms_epsilon_year_fraction",
    "raw_from_parameters",
    "shared_nms_node_sets",
]
