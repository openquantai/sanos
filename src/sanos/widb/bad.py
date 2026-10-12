"""Pinned known-bad WIDB observations from the reference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from cdxcore.version import version


@version("1.0")
@dataclass(frozen=True, slots=True)
class BadDataPolicy:
    """Content-versioned copy of the pinned reference exclusion registries."""

    last_likely_valid_date: date
    bad_days_all: frozenset[date]
    bad_days_ticker: Mapping[str, frozenset[date]]
    bad_options: Mapping[str, frozenset[str]]
    bad_options_by_date: Mapping[str, Mapping[date, frozenset[str]]]
    irregular_expiries: frozenset[date]


@version("quantdata-de15be96-bad-registry-v1", dependencies=[BadDataPolicy])
def bad_data_policy() -> BadDataPolicy:
    """Return the exact exclusion policy whose version participates in artifacts."""

    return BadDataPolicy(
        last_likely_valid_date=date(2025, 9, 1),
        bad_days_all=frozenset(
            {
                date(2004, 10, 11),
                date(2004, 11, 11),
                date(2007, 11, 12),
                date(2018, 12, 4),
                date(2025, 1, 8),
            }
        ),
        bad_days_ticker={
            "SPX": frozenset({date(2006, 11, 17), date(2012, 11, 17), date(2015, 8, 21)}),
            "RUT": frozenset({date(2009, 2, 20)}),
            "NDX": frozenset(),
        },
        bad_options={
            "SPX": frozenset(
                {
                    "SPX 291221C400000",
                    "SPX 261218C200000",
                    "SPX 301220C400000",
                    "SPZ.OT",
                    "SPZ.CT",
                    "JXB.OT",
                    "JXB.CT",
                    "JXB.BT",
                    "JXB.NT",
                }
            )
        },
        bad_options_by_date={"SPX": {date(2024, 5, 6): frozenset({"SPX 291221C400000"})}},
        irregular_expiries=frozenset({date(2025, 1, 9)}),
    )


_POLICY = bad_data_policy()
LAST_LIKELY_VALID_DATE = _POLICY.last_likely_valid_date
BAD_DAYS_ALL = _POLICY.bad_days_all
BAD_DAYS_TICKER = _POLICY.bad_days_ticker
BAD_OPTIONS = _POLICY.bad_options
BAD_OPTIONS_BY_DATE = _POLICY.bad_options_by_date
IRREGULAR_EXPIRIES = _POLICY.irregular_expiries


@version("1.0", dependencies=[bad_data_policy])
def bad_day_reason(ticker: str, trading_date: date) -> str | None:
    """Return the pinned exclusion category for a known-bad day, if any."""

    policy = bad_data_policy()
    if trading_date in policy.bad_days_all:
        return "known_bad_day_all_tickers"
    if trading_date in policy.bad_days_ticker.get(ticker.upper(), frozenset()):
        return "known_bad_day_for_ticker"
    return None


@version("1.0", dependencies=[bad_data_policy])
def require_supported_date(trading_date: date) -> None:
    """Reject dates beyond the pinned OptionMetrics availability boundary."""

    boundary = bad_data_policy().last_likely_valid_date
    if trading_date >= boundary:
        raise ValueError(
            f"{trading_date} is on or after LAST_LIKELY_VALID_DATE "
            f"({boundary}); update the pinned boundary only after "
            "confirming OptionMetrics availability"
        )


@version("1.0", dependencies=[bad_data_policy])
def bad_option_symbols(ticker: str, trading_date: date) -> frozenset[str]:
    """Return the pinned union of global and date-specific bad symbols."""

    policy = bad_data_policy()
    normalized = ticker.upper()
    return policy.bad_options.get(normalized, frozenset()) | policy.bad_options_by_date.get(
        normalized, {}
    ).get(trading_date, frozenset())


@version("1.0", dependencies=[bad_data_policy])
def is_irregular_expiry(expiry: date) -> bool:
    """Return whether the expiry is a pinned late-announced irregular date."""

    return expiry in bad_data_policy().irregular_expiries


@version("1.0", dependencies=[bad_data_policy])
def irregular_expiries() -> tuple[date, ...]:
    """Return the pinned irregular expiries in deterministic order."""

    return tuple(sorted(bad_data_policy().irregular_expiries))


__all__ = [
    "BadDataPolicy",
    "bad_data_policy",
    "BAD_DAYS_ALL",
    "BAD_DAYS_TICKER",
    "BAD_OPTIONS",
    "BAD_OPTIONS_BY_DATE",
    "IRREGULAR_EXPIRIES",
    "LAST_LIKELY_VALID_DATE",
    "bad_day_reason",
    "bad_option_symbols",
    "is_irregular_expiry",
    "irregular_expiries",
    "require_supported_date",
]
