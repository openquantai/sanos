"""Deterministic raw OptionMetrics dependency planning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any, cast

import pandas_market_calendars as mcal
from cdxcore.version import version

_TICKER = re.compile(r"^[A-Za-z0-9.^-]{1,32}$")


@version("1.0")
class RawTable(StrEnum):
    """Logical OptionMetrics tables retained by the raw layer."""

    SECURITY = "security"
    OPTIONS = "options"
    SPOT = "spot"
    FORWARDS = "forwards"
    ZERO_CURVE = "zero_curve"


ALL_RAW_TABLES = tuple(RawTable)


@version("1.0", dependencies=[RawTable])
@dataclass(frozen=True, slots=True, order=True)
class RawPartition:
    """One immutable vendor-query partition."""

    table: RawTable
    ticker: str
    observation_date: date | None
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "table", RawTable(self.table))
        ticker = self.ticker.strip().upper()
        if _TICKER.fullmatch(ticker) is None:
            raise ValueError(f"Invalid ticker for an OptionMetrics query: {self.ticker!r}")
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "roles", tuple(sorted(set(self.roles))))
        if self.table is RawTable.SECURITY:
            if self.observation_date is not None:
                raise ValueError("The security-identity partition cannot have a date")
        elif self.observation_date is None:
            raise ValueError(f"Raw table {self.table.value!r} requires an observation date")

    @property
    def is_boundary_dependency(self) -> bool:
        """Whether this partition was added before the requested start date."""

        return "boundary" in self.roles


@version("1.1", dependencies=[RawTable])
@dataclass(frozen=True, slots=True)
class RawDownloadConfig:
    """Content and operational inputs for one raw download request."""

    ticker: str
    start_date: date
    end_date: date
    tables: tuple[RawTable, ...] = ALL_RAW_TABLES
    source: str = "optionmetrics"
    source_generation: str = "default-v1"
    max_connections: int = 2
    retry_attempts: int = 3
    retry_base_seconds: float = 1.0

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if _TICKER.fullmatch(ticker) is None:
            raise ValueError(f"Invalid ticker for an OptionMetrics query: {self.ticker!r}")
        object.__setattr__(self, "ticker", ticker)
        if self.source != "optionmetrics":
            raise ValueError(f"Unsupported raw source: {self.source!r}")
        if self.end_date < self.start_date:
            raise ValueError("Raw download end_date cannot precede start_date")
        tables = tuple(sorted({RawTable(value) for value in self.tables}, key=lambda x: x.value))
        if not tables:
            raise ValueError("At least one raw table must be requested")
        object.__setattr__(self, "tables", tables)
        if not self.source_generation.strip():
            raise ValueError("source_generation cannot be empty")
        if isinstance(self.max_connections, bool) or not isinstance(self.max_connections, int):
            raise TypeError("max_connections must be an integer")
        if not 1 <= self.max_connections <= 4:
            raise ValueError("max_connections must lie in [1, 4]")
        if isinstance(self.retry_attempts, bool) or not isinstance(self.retry_attempts, int):
            raise TypeError("retry_attempts must be an integer")
        if not 1 <= self.retry_attempts <= 3:
            raise ValueError("retry_attempts must lie in [1, 3]")
        if (
            isinstance(self.retry_base_seconds, bool)
            or not isinstance(self.retry_base_seconds, int | float)
            or not math.isfinite(self.retry_base_seconds)
            or self.retry_base_seconds < 0.0
        ):
            raise ValueError("retry_base_seconds must be finite and non-negative")


def _calendar_dates(values: Any) -> tuple[date, ...]:
    return tuple(cast(Any, value).date() for value in values)


@version("1.0")
def nyse_sessions(start_date: date, end_date: date) -> tuple[date, ...]:
    """Return NYSE sessions in the inclusive calendar-date interval."""

    calendar = mcal.get_calendar("NYSE")
    return _calendar_dates(calendar.valid_days(start_date=start_date, end_date=end_date))


@version("1.0", dependencies=[nyse_sessions])
def previous_nyse_session(value: date) -> date:
    """Return the fresh close dependency immediately before ``value``."""

    candidates = nyse_sessions(value - timedelta(days=14), value - timedelta(days=1))
    if not candidates:
        raise RuntimeError(f"Cannot find an NYSE session before {value}")
    return candidates[-1]


@version(
    "1.0",
    dependencies=[RawDownloadConfig, RawPartition, nyse_sessions, previous_nyse_session],
)
def plan_raw_partitions(config: RawDownloadConfig) -> tuple[RawPartition, ...]:
    """Plan requested partitions plus visible return-boundary dependencies."""

    sessions = nyse_sessions(config.start_date, config.end_date)
    if not sessions:
        raise ValueError(
            f"No NYSE sessions fall in {config.start_date.isoformat()} through "
            f"{config.end_date.isoformat()}"
        )
    planned: dict[tuple[RawTable, date | None], set[str]] = {}

    def add(table: RawTable, observation_date: date | None, role: str) -> None:
        planned.setdefault((table, observation_date), set()).add(role)

    add(RawTable.SECURITY, None, "identity")
    for table in config.tables:
        if table is RawTable.SECURITY:
            continue
        for session in sessions:
            add(table, session, "requested")

    boundary = previous_nyse_session(sessions[0])
    for table in (RawTable.SPOT, RawTable.FORWARDS, RawTable.ZERO_CURVE):
        if table in config.tables:
            add(table, boundary, "boundary")

    return tuple(
        RawPartition(
            table=table,
            ticker=config.ticker,
            observation_date=observation_date,
            roles=tuple(roles),
        )
        for (table, observation_date), roles in sorted(
            planned.items(),
            key=lambda item: (
                item[0][0].value,
                date.min if item[0][1] is None else item[0][1],
            ),
        )
    )


__all__ = [
    "ALL_RAW_TABLES",
    "RawDownloadConfig",
    "RawPartition",
    "RawTable",
    "nyse_sessions",
    "plan_raw_partitions",
    "previous_nyse_session",
]
