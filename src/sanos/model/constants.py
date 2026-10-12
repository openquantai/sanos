"""Time constants used by the standalone numerical model."""

from __future__ import annotations

from cdxcore.version import version


@version("1.0")
def business_days_per_year() -> int:
    """Return the frozen reference business-day annualization."""

    return 252


DAYS_PER_YEAR = business_days_per_year()


@version("1.0", dependencies=[business_days_per_year])
def one_minute_year_fraction() -> float:
    """Return one 6.5-hour trading-session minute in model years."""

    return 1.0 / (60.0 * 6.5 * DAYS_PER_YEAR)


ONE_MINUTE = one_minute_year_fraction()

__all__ = [
    "DAYS_PER_YEAR",
    "ONE_MINUTE",
    "business_days_per_year",
    "one_minute_year_fraction",
]
