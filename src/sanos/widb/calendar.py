"""NYSE business-time conventions used by the WIDB daily pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas_market_calendars as mcal
from cdxcore.version import version


@version("widb-business-clock-252-sessions-6.5-hours-v1")
def business_clock_policy() -> tuple[float, float]:
    """Return version-owned annual-session and trading-day-hour constants."""

    return 252.0, 6.5


DAYS_PER_YEAR, _TRADING_DAY_HOURS = business_clock_policy()
ONE_MINUTE = 1.0 / (60.0 * _TRADING_DAY_HOURS * DAYS_PER_YEAR)


@version("1.0")
@dataclass(frozen=True, slots=True)
class NormalizedStrikes:
    """Reference log-moneyness scaling used by the WIDB filters."""

    epsilon_years: float
    volatility_scale: float = math.log(1.1)

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon_years) or self.epsilon_years <= 0.0:
            raise ValueError("epsilon_years must be finite and positive")
        if not math.isfinite(self.volatility_scale) or self.volatility_scale <= 0.0:
            raise ValueError("volatility_scale must be finite and positive")

    def to_nms(
        self,
        strike: float | np.ndarray[Any, Any],
        time_years: float | np.ndarray[Any, Any],
        *,
        volatility: float | np.ndarray[Any, Any] | None = None,
        epsilon_volatility: float | np.ndarray[Any, Any] | None = None,
    ) -> float | np.ndarray[Any, Any]:
        """Convert positive forward-relative strikes to normalized moneyness."""

        strike_array = np.asarray(strike)
        time_array = np.asarray(time_years)
        vol = self.volatility_scale if volatility is None else np.asarray(volatility)
        epsilon_vol = (
            self.volatility_scale if epsilon_volatility is None else np.asarray(epsilon_volatility)
        )
        if np.min(strike_array) <= 0.0:
            raise ValueError("strike must be positive")
        if np.min(time_array) < 0.0:
            raise ValueError("time_years must be non-negative")
        if np.min(vol) <= 0.0 or np.min(epsilon_vol) <= 0.0:
            raise ValueError("volatility inputs must be positive")
        result = np.log(strike_array) * np.sqrt(
            (self.epsilon_years + 1.0)
            / (epsilon_vol * epsilon_vol * self.epsilon_years + vol * vol * time_array)
        )
        return float(result) if result.ndim == 0 else result

    def from_nms(
        self,
        normalized: float | np.ndarray[Any, Any],
        time_years: float | np.ndarray[Any, Any],
        *,
        volatility: float | np.ndarray[Any, Any] | None = None,
        epsilon_volatility: float | np.ndarray[Any, Any] | None = None,
    ) -> float | np.ndarray[Any, Any]:
        """Convert normalized moneyness back to forward-relative strikes."""

        normalized_array = np.asarray(normalized)
        time_array = np.asarray(time_years)
        vol = self.volatility_scale if volatility is None else np.asarray(volatility)
        epsilon_vol = (
            self.volatility_scale if epsilon_volatility is None else np.asarray(epsilon_volatility)
        )
        if np.min(time_array) < 0.0:
            raise ValueError("time_years must be non-negative")
        if np.min(vol) <= 0.0 or np.min(epsilon_vol) <= 0.0:
            raise ValueError("volatility inputs must be positive")
        result = np.exp(
            normalized_array
            * np.sqrt(
                (epsilon_vol * epsilon_vol * self.epsilon_years + vol * vol * time_array)
                / (self.epsilon_years + 1.0)
            )
        )
        return float(result) if result.ndim == 0 else result


@version(
    "widb-nms-log1p0.1-epsilon-1over8-and-5-sessions-v1",
    dependencies=[
        NormalizedStrikes,
        business_clock_policy,
    ],
)
def normalized_strike_policy() -> tuple[NormalizedStrikes, NormalizedStrikes]:
    """Construct the exact normalized-strike geometries used by the reference."""

    scale = math.log(1.1)
    return (
        NormalizedStrikes(
            epsilon_years=1.0 / DAYS_PER_YEAR / 8.0,
            volatility_scale=scale,
        ),
        NormalizedStrikes(
            epsilon_years=5.0 / DAYS_PER_YEAR,
            volatility_scale=scale,
        ),
    )


NMS, NMS5 = normalized_strike_policy()


@version("1.0")
def business_dates(
    start: date,
    end: date,
    *,
    exclude_half_days: bool = False,
) -> tuple[date, ...]:
    """Return inclusive NYSE trading dates, optionally excluding early closes."""

    if end < start:
        raise ValueError("end cannot precede start")
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=start, end_date=end)
    if exclude_half_days and len(schedule):
        schedule = schedule.drop(calendar.early_closes(schedule).index, errors="ignore")
    return tuple(value.date() for value in schedule.index)


@version("1.0")
def half_days(start: date, end: date) -> tuple[date, ...]:
    """Return inclusive NYSE early-close sessions."""

    if end < start:
        raise ValueError("end cannot precede start")
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=start, end_date=end)
    return tuple(value.date() for value in calendar.early_closes(schedule).index)


@version("1.0", dependencies=[business_dates, half_days])
def is_half_day(trading_date: date) -> bool:
    """Return whether the NYSE session is an early close."""

    sessions = business_dates(trading_date, trading_date)
    if len(sessions) != 1:
        raise ValueError(f"{trading_date.isoformat()} is not an NYSE trading session")
    return bool(half_days(trading_date, trading_date))


@version("1.0", dependencies=[business_dates])
def count_business_days_between(
    start: date | np.ndarray[Any, Any],
    end: date | np.ndarray[Any, Any],
    *,
    exclude_half_days: bool = False,
) -> int | np.ndarray[Any, Any]:
    """Count NYSE sessions in ``(start, end]`` exactly as the reference does."""

    end_values = np.asarray(end, dtype="datetime64[D]")
    start_value = np.asarray(start, dtype="datetime64[D]")
    vector = start_value.ndim > 0 or end_values.ndim > 0
    lower = min(np.min(start_value), np.min(end_values))
    upper = max(np.max(start_value), np.max(end_values))
    calendar = np.asarray(
        business_dates(
            np.asarray(lower, dtype="datetime64[D]").item(),
            np.asarray(upper, dtype="datetime64[D]").item(),
            exclude_half_days=exclude_half_days,
        ),
        dtype="datetime64[D]",
    )
    if calendar.size == 0:
        raise ValueError(f"No NYSE sessions between {lower} and {upper}")
    if not vector:
        if start_value <= end_values:
            return int(np.sum((calendar > start_value) & (calendar <= end_values)))
        return -int(np.sum((calendar >= end_values) & (calendar < start_value)))
    left = np.searchsorted(calendar, start_value, side="right")
    right = np.searchsorted(calendar, end_values, side="right")
    result = (right - left).astype(np.int32)
    return result


@version("1.0", dependencies=[business_dates])
def add_business_days(
    start: date,
    number: int,
    *,
    exclude_half_days: bool = False,
    count_start: bool = False,
) -> date:
    """Move by NYSE sessions, with the reference's optional start counting."""

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError("number must be an integer")
    horizon = (abs(number) // 5) * 7 + 14
    if number >= 0:
        schedule = business_dates(
            start,
            start + timedelta(days=horizon),
            exclude_half_days=exclude_half_days,
        )
    else:
        schedule = tuple(
            reversed(
                business_dates(
                    start - timedelta(days=horizon),
                    start,
                    exclude_half_days=exclude_half_days,
                )
            )
        )
    if not schedule:
        raise ValueError(f"No NYSE sessions found around {start}")
    offset = abs(number)
    if schedule[0] == start and count_start:
        offset = max(0, offset - 1)
    try:
        return schedule[offset]
    except IndexError as exc:
        raise RuntimeError("Calendar planning horizon was unexpectedly too short") from exc


@version("1.0", dependencies=[add_business_days])
def previous_business_day(
    value: date,
    *,
    exclude_half_days: bool = False,
    count_start: bool = False,
) -> date:
    """Return the preceding eligible NYSE session."""

    return add_business_days(
        value,
        -1,
        exclude_half_days=exclude_half_days,
        count_start=count_start,
    )


@version("1.0", dependencies=[add_business_days])
def next_business_day(
    value: date,
    *,
    exclude_half_days: bool = False,
    count_start: bool = False,
) -> date:
    """Return the following eligible NYSE session."""

    return add_business_days(
        value,
        1,
        exclude_half_days=exclude_half_days,
        count_start=count_start,
    )


@version("1.0")
def session_open_close(value: date) -> tuple[datetime | None, datetime | None, bool]:
    """Return New York session bounds and early-close status for one date."""

    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=value, end_date=value)
    if schedule.empty:
        return None, None, False
    row = schedule.iloc[0]
    opening = row["market_open"].tz_convert("America/New_York").to_pydatetime()
    closing = row["market_close"].tz_convert("America/New_York").to_pydatetime()
    return opening, closing, len(calendar.early_closes(schedule)) == 1


__all__ = [
    "DAYS_PER_YEAR",
    "NMS",
    "NMS5",
    "ONE_MINUTE",
    "NormalizedStrikes",
    "add_business_days",
    "business_dates",
    "count_business_days_between",
    "half_days",
    "is_half_day",
    "next_business_day",
    "previous_business_day",
    "session_open_close",
]
