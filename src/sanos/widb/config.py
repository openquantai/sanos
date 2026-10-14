"""Canonical compatibility configuration for daily WIDB processing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from cdxcore.version import version


@version("1.0")
def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class ClockConfig:
    """Business-time convention shared by every downstream daily stage."""

    overnight_variance_fraction: float = 0.2

    def __post_init__(self) -> None:
        _finite("overnight_variance_fraction", self.overnight_variance_fraction)
        if not 0.0 <= self.overnight_variance_fraction <= 1.0:
            raise ValueError("overnight_variance_fraction must lie in [0, 1]")

    @property
    def am_business_day_subtraction(self) -> float:
        """Return the business-day fraction removed from AM expiries."""

        return 1.0 - self.overnight_variance_fraction


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class RawDayConfig:
    """Reference-compatible raw quote cleaning inputs."""

    min_bid: float = 0.011
    drop_irregular_expiries: bool = True

    def __post_init__(self) -> None:
        _finite("min_bid", self.min_bid)
        if self.min_bid < 0.0:
            raise ValueError("min_bid must be non-negative")
        if not isinstance(self.drop_irregular_expiries, bool):
            raise ValueError("drop_irregular_expiries must be a boolean")


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class ExpiryEnrichmentConfig:
    """Reference thresholds used to infer forwards and enrich paired books."""

    num_min_either_traded: int = 10
    num_min_both_traded: int = 4
    min_pure_spread: float = 1.0e-6
    min_implied_vol: float = 0.01
    max_implied_vol: float = 5.0
    normalize: bool = False

    def __post_init__(self) -> None:
        for name in ("num_min_either_traded", "num_min_both_traded"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("min_pure_spread", "min_implied_vol", "max_implied_vol"):
            _finite(name, getattr(self, name))
        if self.min_pure_spread <= 0.0:
            raise ValueError("min_pure_spread must be positive")
        if self.min_implied_vol <= 0.0:
            raise ValueError("min_implied_vol must be positive")
        if self.max_implied_vol <= self.min_implied_vol:
            raise ValueError("max_implied_vol must exceed min_implied_vol")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be a boolean")


@version(
    "1.0",
    dependencies=[ClockConfig, RawDayConfig, ExpiryEnrichmentConfig],
)
@dataclass(frozen=True, slots=True)
class DailyWidbConfig:
    """Complete content configuration for Phase 5 daily transformations."""

    clock: ClockConfig = ClockConfig()
    raw: RawDayConfig = RawDayConfig()
    enrichment: ExpiryEnrichmentConfig = ExpiryEnrichmentConfig()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DailyWidbConfig:
        """Parse a strict portable mapping without accepting unknown fields."""

        allowed = {"clock", "raw", "enrichment"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown daily WIDB configuration keys: {sorted(unknown)}")
        clock_value = value.get("clock", {})
        raw_value = value.get("raw", {})
        enrichment_value = value.get("enrichment", {})
        if not all(
            isinstance(item, Mapping)
            for item in (clock_value, raw_value, enrichment_value)
        ):
            raise ValueError("clock, raw, and enrichment values must be mappings")
        try:
            clock = ClockConfig(**dict(clock_value))
            raw = RawDayConfig(**dict(raw_value))
            enrichment = ExpiryEnrichmentConfig(**dict(enrichment_value))
        except TypeError as exc:
            raise ValueError(f"Invalid daily WIDB configuration field: {exc}") from exc
        return cls(clock=clock, raw=raw, enrichment=enrichment)


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class OptionsFilterConfig:
    """Production option-population rules used by the 891-day reference."""

    normalize: bool = True
    max_num_options: int = 100_000
    max_business_days_to_expiry: int = 70
    max_nms: tuple[float, float] = (-100.0, 100.0)
    nms_use_atm: bool = True
    min_vega_sqrt_var: float = 0.001
    min_log_volume: float = 1.0
    min_log_open_interest: float = 1.0
    remove_am: bool = False
    min_options_per_expiry: int = 10

    def __post_init__(self) -> None:
        for name in ("max_num_options", "max_business_days_to_expiry", "min_options_per_expiry"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_num_options < 10:
            raise ValueError("max_num_options must be at least 10")
        if self.min_options_per_expiry > self.max_num_options:
            raise ValueError("min_options_per_expiry cannot exceed max_num_options")
        if not isinstance(self.max_nms, tuple) or len(self.max_nms) != 2:
            raise ValueError("max_nms must be a two-element tuple")
        lower, upper = self.max_nms
        _finite("max_nms[0]", lower)
        _finite("max_nms[1]", upper)
        if not lower < 0.0 < upper:
            raise ValueError("max_nms must straddle zero")
        for name in ("min_vega_sqrt_var", "min_log_volume", "min_log_open_interest"):
            value = getattr(self, name)
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("normalize", "nms_use_atm", "remove_am"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class NoArbitrageConfig:
    """Content-affecting settings shared by the linear and smooth surface fits.

    The reference solver timeout is deliberately fixed in code at 300 seconds.
    Exposing it as a content setting would resolve open decision CD-006 without
    approval; operational provenance records the fixed value instead.
    """

    min_half_spread: float = 1.0e-4
    mid_fit_weight: float = 0.1
    min_atm_volatility: float = 0.05
    min_dlv: float | None = None
    max_dlv: float | None = None
    min_strikes_per_expiry: int = 5
    min_density: float = 1.0e-4
    min_dc_epsilon: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in (
            "min_half_spread",
            "mid_fit_weight",
            "min_atm_volatility",
            "min_density",
            "min_dc_epsilon",
        ):
            _finite(name, getattr(self, name))
        if self.min_half_spread <= 0.0:
            raise ValueError("min_half_spread must be positive")
        if not 0.0 <= self.mid_fit_weight <= 1.0:
            raise ValueError("mid_fit_weight must lie in [0, 1]")
        if self.min_atm_volatility < 0.0:
            raise ValueError("min_atm_volatility must be non-negative")
        if self.min_density <= 0.0 or self.min_dc_epsilon <= 0.0:
            raise ValueError("min_density and min_dc_epsilon must be positive")
        if (
            isinstance(self.min_strikes_per_expiry, bool)
            or not isinstance(self.min_strikes_per_expiry, int)
            or self.min_strikes_per_expiry < 3
        ):
            raise ValueError("min_strikes_per_expiry must be an integer of at least 3")
        for name in ("min_dlv", "max_dlv"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative or null")
        if self.min_dlv is not None and self.max_dlv is not None:
            if self.max_dlv <= self.min_dlv:
                raise ValueError("max_dlv must exceed min_dlv")


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class SmoothSanosConfig:
    """Production smooth-SANOS settings applied after option filtering."""

    smoothness: float = 0.25

    def __post_init__(self) -> None:
        _finite("smoothness", self.smoothness)
        if self.smoothness <= 0.01:
            raise ValueError("smoothness must be strictly greater than 0.01")


@version("1.0", dependencies=[_finite])
@dataclass(frozen=True, slots=True)
class DlvProjectionTolerances:
    """Strict acceptance limits for the canonical discrete-DLV projection."""

    density_mass_abs: float = 1.0e-10
    density_moment_abs: float = 1.0e-10
    density_inverse_max_abs: float = 1.0e-12
    call_inverse_max_abs: float = 1.0e-12
    dlv_fixed_point_max_abs: float = 1.0e-9
    dlv_bound_abs: float = 1.0e-9
    projection_call_max_abs: float = 1.0e-4
    atm_annual_iv_max_abs: float = 1.0e-4
    conditioned_annual_iv_max_abs: float = 5.0e-3
    conditioned_annual_iv_p99_abs: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            _finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@version("1.4", dependencies=[_finite, DlvProjectionTolerances])
@dataclass(frozen=True, slots=True)
class DiscreteDlvConfig:
    """Canonical daily-front DLV projection of the smooth SANOS surface.

    The projection is deliberately a second representation of the completed
    smooth SANOS state. It is not the initial no-arbitrage audit surface and it
    is not an independent calibration to vendor books.
    """

    nms_range: tuple[float, float] = (-3.0, 2.0)
    inner_nodes: int = 200
    lower_support_multiplier: float = 0.5
    upper_support_multiplier: float = 1.5
    objective_weighting: Literal["uniform"] = "uniform"
    min_dlv: float = 0.01
    max_dlv: float = 3.0
    constraint_dlv_buffer: float = 1.0e-5
    density_mass_floor: float = 1.0e-8
    min_density_vol: float | None = 1.0e-4
    min_density_cap: float | None = 1.0e-5
    solver_timeout_seconds: float = 300.0
    solver_kkt_tolerance: float = 1.0e-10
    fit_call_max_abs: float = 9.0e-5
    allow_solver_epsilon_warnings: bool = True
    conditioned_iv_vega_sqrt_floor: float = 5.0e-2
    tolerances: DlvProjectionTolerances = DlvProjectionTolerances()

    def __post_init__(self) -> None:
        if not isinstance(self.nms_range, tuple) or len(self.nms_range) != 2:
            raise ValueError("nms_range must be a two-element tuple")
        lower, upper = self.nms_range
        _finite("nms_range[0]", lower)
        _finite("nms_range[1]", upper)
        if not lower < 0.0 < upper:
            raise ValueError("nms_range must straddle zero")
        if isinstance(self.inner_nodes, bool) or not isinstance(self.inner_nodes, int):
            raise ValueError("inner_nodes must be an integer")
        if self.inner_nodes < 5:
            raise ValueError("inner_nodes must be at least five")
        for name in (
            "lower_support_multiplier",
            "upper_support_multiplier",
            "min_dlv",
            "max_dlv",
            "constraint_dlv_buffer",
            "density_mass_floor",
            "solver_timeout_seconds",
            "solver_kkt_tolerance",
            "fit_call_max_abs",
            "conditioned_iv_vega_sqrt_floor",
        ):
            _finite(name, getattr(self, name))
        if not 0.0 < self.lower_support_multiplier < 1.0:
            raise ValueError("lower_support_multiplier must lie in (0, 1)")
        if self.upper_support_multiplier <= 1.0:
            raise ValueError("upper_support_multiplier must exceed one")
        if self.min_dlv <= 0.0 or self.max_dlv <= self.min_dlv:
            raise ValueError("DLV bounds must satisfy 0 < min_dlv < max_dlv")
        if (
            self.constraint_dlv_buffer <= 0.0
            or 2.0 * self.constraint_dlv_buffer >= self.max_dlv - self.min_dlv
        ):
            raise ValueError("constraint_dlv_buffer must fit strictly inside DLV bounds")
        if not 0.0 < self.density_mass_floor < 1.0 / (self.inner_nodes + 2):
            raise ValueError("density_mass_floor must leave free probability mass")
        if self.solver_timeout_seconds <= 0.0:
            raise ValueError("solver_timeout_seconds must be positive")
        if not 1.0e-10 <= self.solver_kkt_tolerance <= 1.0e-4:
            raise ValueError("solver_kkt_tolerance must lie in [1e-10, 1e-4]")
        if not isinstance(self.tolerances, DlvProjectionTolerances):
            raise TypeError("tolerances must be DlvProjectionTolerances")
        if not 0.0 < self.fit_call_max_abs <= self.tolerances.projection_call_max_abs:
            raise ValueError(
                "fit_call_max_abs must be positive and no greater than the "
                "projection-call acceptance tolerance"
            )
        if self.conditioned_iv_vega_sqrt_floor <= 0.0:
            raise ValueError("conditioned_iv_vega_sqrt_floor must be positive")
        for name in ("min_density_vol", "min_density_cap"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
                if value <= 0.0:
                    raise ValueError(f"{name} must be positive or null")
        if self.objective_weighting != "uniform":
            raise ValueError("Only the reference-faithful uniform objective is supported")
        if not isinstance(self.allow_solver_epsilon_warnings, bool):
            raise ValueError("allow_solver_epsilon_warnings must be a boolean")


@version(
    "2.0",
    dependencies=[
        OptionsFilterConfig,
        NoArbitrageConfig,
        SmoothSanosConfig,
        DiscreteDlvConfig,
    ],
)
@dataclass(frozen=True, slots=True)
class DailySurfaceConfig:
    """Complete Phase 6 compatibility configuration for a fitted day."""

    options_filter: OptionsFilterConfig = OptionsFilterConfig()
    no_arbitrage: NoArbitrageConfig = NoArbitrageConfig()
    smooth_sanos: SmoothSanosConfig = SmoothSanosConfig()
    discrete_dlv: DiscreteDlvConfig = DiscreteDlvConfig()

    def __post_init__(self) -> None:
        if not self.options_filter.normalize:
            raise ValueError("The smooth SANOS pipeline requires options_filter.normalize=true")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DailySurfaceConfig:
        """Parse a strict portable mapping without accepting unknown fields."""

        allowed = {"options_filter", "no_arbitrage", "smooth_sanos", "discrete_dlv"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown daily surface configuration keys: {sorted(unknown)}")
        filter_value = value.get("options_filter", {})
        no_arb_value = value.get("no_arbitrage", {})
        smooth_value = value.get("smooth_sanos", {})
        dlv_value = value.get("discrete_dlv", {})
        if not all(
            isinstance(item, Mapping)
            for item in (filter_value, no_arb_value, smooth_value, dlv_value)
        ):
            raise ValueError(
                "options_filter, no_arbitrage, smooth_sanos, and discrete_dlv values "
                "must be mappings"
            )
        filter_payload = dict(filter_value)
        if "max_nms" in filter_payload:
            max_nms = filter_payload["max_nms"]
            if not isinstance(max_nms, list | tuple) or len(max_nms) != 2:
                raise ValueError("options_filter.max_nms must contain exactly two values")
            filter_payload["max_nms"] = tuple(max_nms)
        dlv_payload = dict(dlv_value)
        if "nms_range" in dlv_payload:
            nms_range = dlv_payload["nms_range"]
            if not isinstance(nms_range, list | tuple) or len(nms_range) != 2:
                raise ValueError("discrete_dlv.nms_range must contain exactly two values")
            dlv_payload["nms_range"] = tuple(nms_range)
        if "tolerances" in dlv_payload:
            tolerance_payload = dlv_payload["tolerances"]
            if not isinstance(tolerance_payload, Mapping):
                raise ValueError("discrete_dlv.tolerances must be a mapping")
            dlv_payload["tolerances"] = DlvProjectionTolerances(**dict(tolerance_payload))
        try:
            return cls(
                options_filter=OptionsFilterConfig(**filter_payload),
                no_arbitrage=NoArbitrageConfig(**dict(no_arb_value)),
                smooth_sanos=SmoothSanosConfig(**dict(smooth_value)),
                discrete_dlv=DiscreteDlvConfig(**dlv_payload),
            )
        except TypeError as exc:
            raise ValueError(f"Invalid daily surface configuration field: {exc}") from exc


__all__ = [
    "ClockConfig",
    "DailySurfaceConfig",
    "DailyWidbConfig",
    "DiscreteDlvConfig",
    "DlvProjectionTolerances",
    "ExpiryEnrichmentConfig",
    "NoArbitrageConfig",
    "OptionsFilterConfig",
    "RawDayConfig",
    "SmoothSanosConfig",
]
