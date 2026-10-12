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


__all__ = ["ClockConfig", "DailyWidbConfig", "ExpiryEnrichmentConfig", "RawDayConfig"]
