"""Canonical compatibility configuration for daily WIDB processing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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


@version(
    "3.0",
    dependencies=[
        OptionsFilterConfig,
        NoArbitrageConfig,
        SmoothSanosConfig,
    ],
)
@dataclass(frozen=True, slots=True)
class DailySurfaceConfig:
    """Complete Phase 6 compatibility configuration for a fitted day."""

    options_filter: OptionsFilterConfig = OptionsFilterConfig()
    no_arbitrage: NoArbitrageConfig = NoArbitrageConfig()
    smooth_sanos: SmoothSanosConfig = SmoothSanosConfig()

    def __post_init__(self) -> None:
        if not self.options_filter.normalize:
            raise ValueError("The smooth SANOS pipeline requires options_filter.normalize=true")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DailySurfaceConfig:
        """Parse a strict portable mapping without accepting unknown fields."""

        allowed = {"options_filter", "no_arbitrage", "smooth_sanos"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown daily surface configuration keys: {sorted(unknown)}")
        filter_value = value.get("options_filter", {})
        no_arb_value = value.get("no_arbitrage", {})
        smooth_value = value.get("smooth_sanos", {})
        if not all(
            isinstance(item, Mapping)
            for item in (filter_value, no_arb_value, smooth_value)
        ):
            raise ValueError(
                "options_filter, no_arbitrage, and smooth_sanos values must be mappings"
            )
        filter_payload = dict(filter_value)
        if "max_nms" in filter_payload:
            max_nms = filter_payload["max_nms"]
            if not isinstance(max_nms, list | tuple) or len(max_nms) != 2:
                raise ValueError("options_filter.max_nms must contain exactly two values")
            filter_payload["max_nms"] = tuple(max_nms)
        try:
            return cls(
                options_filter=OptionsFilterConfig(**filter_payload),
                no_arbitrage=NoArbitrageConfig(**dict(no_arb_value)),
                smooth_sanos=SmoothSanosConfig(**dict(smooth_value)),
            )
        except TypeError as exc:
            raise ValueError(f"Invalid daily surface configuration field: {exc}") from exc


__all__ = [
    "ClockConfig",
    "DailySurfaceConfig",
    "DailyWidbConfig",
    "ExpiryEnrichmentConfig",
    "NoArbitrageConfig",
    "OptionsFilterConfig",
    "RawDayConfig",
    "SmoothSanosConfig",
]
