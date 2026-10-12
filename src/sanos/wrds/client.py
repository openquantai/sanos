"""Credential-safe WRDS connection helpers and failure classification."""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from cdxcore.version import version

from sanos.wrds.source import OptionMetricsCatalog, choose_optionmetrics_catalog


class WrdsError(RuntimeError):
    """Base class for guarded WRDS failures."""


class WrdsConfigurationError(WrdsError):
    """Raised before connecting when credentials or dependencies are unavailable."""


class WrdsAccessDisabledError(WrdsConfigurationError):
    """Raised when ``WRDS_OFF`` forbids live access."""


class WrdsAuthenticationError(WrdsError):
    """An authentication, 2FA, access, or permission failure that is never retried."""


class WrdsQueryError(WrdsError):
    """A non-retryable or exhausted WRDS query failure."""


class WrdsConnection(Protocol):
    """Small protocol implemented by ``wrds.Connection`` and offline fakes."""

    def list_libraries(self) -> list[str]: ...

    def list_tables(self, library: str) -> list[str]: ...

    def raw_sql(self, sql: str) -> Any: ...

    def close(self) -> None: ...


ConnectionFactory = Callable[[str], WrdsConnection]


@version("1.0")
@dataclass(frozen=True, slots=True)
class WrdsCredentials:
    """WRDS environment configuration whose representation redacts the username."""

    username: str = field(repr=False)
    access_disabled: bool = False

    def __post_init__(self) -> None:
        if not self.username.strip():
            raise ValueError("WRDS username cannot be empty")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> WrdsCredentials:
        """Read credentials only when a live cache miss has already been established."""

        source = os.environ if environment is None else environment
        username = source.get("WRDS_USERNAME", "").strip()
        if not username:
            raise WrdsConfigurationError(
                "Live WRDS access requires WRDS_USERNAME in the existing WRDS credential setup"
            )
        disabled = source.get("WRDS_OFF", "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disabled:
            raise WrdsAccessDisabledError(
                "Live WRDS access is disabled by WRDS_OFF; cache-only and dry-run remain available"
            )
        return cls(username=username)


@version("1.0")
class WrdsFailureKind(StrEnum):
    """Conservative retry classification."""

    AUTHENTICATION = "authentication"
    TRANSIENT_QUERY = "transient_query"
    PERMANENT_QUERY = "permanent_query"


_AUTH_MARKERS = (
    "2fa",
    "authentication",
    "authorization",
    "credential",
    "denied",
    "eoferror",
    "one-time password",
    "otp",
    "password",
    "permission",
    "pg_hba",
    "too many connection",
    "two-factor",
)
_TRANSIENT_MARKERS = (
    "connection reset",
    "connection timed out",
    "deadlock",
    "network is unreachable",
    "server closed the connection",
    "serialization failure",
    "temporarily unavailable",
    "timeout",
)


@version("1.0", dependencies=[WrdsFailureKind])
def classify_wrds_failure(error: BaseException) -> WrdsFailureKind:
    """Classify unknown failures conservatively: only known transients retry."""

    message = f"{type(error).__name__}: {error}".casefold()
    if any(marker in message for marker in _AUTH_MARKERS):
        return WrdsFailureKind.AUTHENTICATION
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return WrdsFailureKind.TRANSIENT_QUERY
    return WrdsFailureKind.PERMANENT_QUERY


@version("1.0")
def default_connection_factory(username: str) -> WrdsConnection:
    """Import the optional WRDS package lazily at the authorized network boundary."""

    try:
        wrds = importlib.import_module("wrds")
    except ImportError as exc:
        raise WrdsConfigurationError(
            "The optional WRDS client is not installed; install sanos[wrds]"
        ) from exc
    return wrds.Connection(wrds_username=username)


@version("1.0", dependencies=[OptionMetricsCatalog, choose_optionmetrics_catalog])
def inspect_optionmetrics(connection: WrdsConnection) -> OptionMetricsCatalog:
    """Verify account access and snapshot the selected library's table catalog."""

    libraries = tuple(connection.list_libraries())
    selected = next(
        (name for name in ("optionm_all", "optionm") if name in libraries),
        None,
    )
    tables = {} if selected is None else {selected: tuple(connection.list_tables(selected))}
    return choose_optionmetrics_catalog(list(libraries), tables)


@version("1.0", dependencies=[WrdsFailureKind, classify_wrds_failure])
def execute_query_with_retry[T](
    operation: Callable[[], T],
    *,
    attempts: int,
    base_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry only recognized transient query failures with bounded backoff."""

    if not 1 <= attempts <= 5:
        raise ValueError("WRDS query attempts must lie in [1, 5]")
    if base_seconds < 0.0:
        raise ValueError("WRDS retry base seconds cannot be negative")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            kind = classify_wrds_failure(exc)
            if kind is WrdsFailureKind.AUTHENTICATION:
                raise WrdsAuthenticationError(
                    "WRDS authentication, 2FA, connection-limit, or permission failure; "
                    "the operation was not retried"
                ) from exc
            if kind is not WrdsFailureKind.TRANSIENT_QUERY or attempt == attempts:
                raise WrdsQueryError(
                    f"WRDS query failed ({kind.value}) after {attempt} attempt(s)"
                ) from exc
            sleep(base_seconds * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def close_connection(connection: WrdsConnection) -> None:
    """Close a connection without masking the primary query failure."""

    connection.close()


__all__ = [
    "ConnectionFactory",
    "WrdsAccessDisabledError",
    "WrdsAuthenticationError",
    "WrdsConfigurationError",
    "WrdsConnection",
    "WrdsCredentials",
    "WrdsError",
    "WrdsFailureKind",
    "WrdsQueryError",
    "classify_wrds_failure",
    "close_connection",
    "default_connection_factory",
    "execute_query_with_retry",
    "inspect_optionmetrics",
]
