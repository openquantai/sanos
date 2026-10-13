"""Cross-process WRDS connection-attempt circuit breaker."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

from cdxcore.version import version
from filelock import FileLock, Timeout

from sanos.artifacts import ArtifactError, atomic_write_bytes
from sanos.config import canonical_json
from sanos.wrds.lease import account_key, default_lease_root


class WrdsAttemptMonitorError(RuntimeError):
    """Base class for fail-closed WRDS connection-attempt monitoring."""


class WrdsAttemptMonitorBusyError(WrdsAttemptMonitorError):
    """Raised when another process is currently opening a WRDS connection."""


class WrdsAttemptLimitError(WrdsAttemptMonitorError):
    """Raised before a fourth recent failed or interrupted connection attempt."""


class WrdsAttemptMonitorStateError(WrdsAttemptMonitorError):
    """Raised when the persistent monitor state cannot be trusted."""


@version(
    "1.0",
    dependencies=[
        account_key,
        default_lease_root,
        atomic_write_bytes,
        canonical_json,
    ],
)
class WrdsConnectionAttemptMonitor(AbstractContextManager["WrdsConnectionAttemptMonitor"]):
    """Serialize connection opens and stop after three recent failures.

    Entering reserves one attempt before network access. A successful connection
    clears the account's failure streak. A failed or interrupted attempt leaves
    its reservation in the rolling window, so a fourth attempt is rejected
    before it can contact WRDS. No username or credential is persisted.
    """

    MAX_FAILED_ATTEMPTS = 3
    FAILURE_WINDOW_SECONDS = 15 * 60.0
    LOCK_TIMEOUT_SECONDS = 180.0

    def __init__(
        self,
        username: str,
        *,
        root: Path | None = None,
        clock: Callable[[], float] = time.time,
        lock_timeout_seconds: float = LOCK_TIMEOUT_SECONDS,
    ) -> None:
        if lock_timeout_seconds < 0.0 or not math.isfinite(lock_timeout_seconds):
            raise ValueError("WRDS attempt-monitor lock timeout must be finite and non-negative")
        self.account = account_key(username)
        self.root = (default_lease_root() if root is None else root).expanduser().resolve(
            strict=False
        )
        self.clock = clock
        self.lock_timeout_seconds = lock_timeout_seconds
        self._lock: FileLock | None = None
        self._reservation: float | None = None
        self._prior_failures: tuple[float, ...] = ()
        self._resolved = False

    @property
    def account_root(self) -> Path:
        """Return the hashed account directory shared with connection leases."""

        return self.root / self.account

    @property
    def state_path(self) -> Path:
        """Return the non-secret persistent failure-window state path."""

        return self.account_root / "attempt-monitor.json"

    @property
    def failed_attempts_before_reservation(self) -> int:
        """Return active failures seen before this attempt was reserved."""

        if self._reservation is None:
            raise RuntimeError("WRDS connection attempt has not been reserved")
        return len(self._prior_failures)

    @property
    def remaining_attempts_after_reservation(self) -> int:
        """Return attempts left before the rolling failure limit opens."""

        return self.MAX_FAILED_ATTEMPTS - self.failed_attempts_before_reservation - 1

    def _read_failures(self, now: float) -> tuple[float, ...]:
        try:
            source = self.state_path.read_text(encoding="utf-8")
            payload = json.loads(source)
        except FileNotFoundError:
            return ()
        except (OSError, UnicodeError, ValueError) as exc:
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor state is unreadable; refusing network access"
            ) from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"failure_timestamps", "schema"}
            or payload.get("schema") != "sanos-wrds-attempt-monitor-v1"
            or not isinstance(payload.get("failure_timestamps"), list)
        ):
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor state is invalid; refusing network access"
            )
        if source != canonical_json(payload) + "\n":
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor state is not canonical; "
                "refusing network access"
            )
        failures: list[float] = []
        for value in payload["failure_timestamps"]:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0.0
                or value > now
            ):
                raise WrdsAttemptMonitorStateError(
                    "WRDS connection-attempt monitor timestamps are invalid; "
                    "refusing network access"
                )
            if now - float(value) < self.FAILURE_WINDOW_SECONDS:
                failures.append(float(value))
        if failures != sorted(failures):
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor timestamps are not ordered; "
                "refusing network access"
            )
        return tuple(failures)

    def _write_failures(self, failures: tuple[float, ...]) -> None:
        payload = {
            "schema": "sanos-wrds-attempt-monitor-v1",
            "failure_timestamps": failures,
        }
        try:
            atomic_write_bytes(
                self.state_path,
                (canonical_json(payload) + "\n").encode("utf-8"),
            )
        except (ArtifactError, OSError) as exc:
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor state cannot be updated; "
                "refusing network access"
            ) from exc

    def __enter__(self) -> WrdsConnectionAttemptMonitor:
        try:
            self.account_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor directory cannot be initialized; "
                "refusing network access"
            ) from exc
        lock = FileLock(str(self.account_root / "attempt-monitor.lock"))
        try:
            lock.acquire(timeout=self.lock_timeout_seconds)
        except Timeout as exc:
            raise WrdsAttemptMonitorBusyError(
                "Another WRDS connection attempt is still in progress; no new attempt was made"
            ) from exc
        except OSError as exc:
            if lock.is_locked:
                try:
                    lock.release()
                except OSError:
                    pass
            raise WrdsAttemptMonitorStateError(
                "WRDS connection-attempt monitor lock cannot be acquired; "
                "refusing network access"
            ) from exc
        self._lock = lock
        try:
            now = self.clock()
            if not math.isfinite(now) or now < 0.0:
                raise WrdsAttemptMonitorStateError(
                    "WRDS connection-attempt monitor clock is invalid; refusing network access"
                )
            failures = self._read_failures(now)
            if len(failures) >= self.MAX_FAILED_ATTEMPTS:
                retry_after = max(
                    1,
                    math.ceil(failures[0] + self.FAILURE_WINDOW_SECONDS - now),
                )
                raise WrdsAttemptLimitError(
                    "WRDS connection-attempt monitor blocked a fourth failed or interrupted "
                    f"attempt; retry after at least {retry_after} second(s)"
                )
            self._prior_failures = failures
            self._reservation = now
            self._write_failures((*failures, now))
            return self
        except BaseException:
            try:
                lock.release()
            except OSError as exc:
                raise WrdsAttemptMonitorStateError(
                    "WRDS connection-attempt monitor lock cannot be released; "
                    "refusing network access"
                ) from exc
            self._lock = None
            raise

    def succeed(self) -> None:
        """Clear the rolling failure window after a connection opens successfully."""

        if self._reservation is None or self._lock is None:
            raise RuntimeError("WRDS connection attempt has not been reserved")
        self._write_failures(())
        self._resolved = True

    def cancel(self) -> None:
        """Remove this reservation when no network attempt could be made."""

        if self._reservation is None or self._lock is None:
            raise RuntimeError("WRDS connection attempt has not been reserved")
        self._write_failures(self._prior_failures)
        self._resolved = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None and not self._resolved:
                raise RuntimeError("WRDS connection attempt completed without a monitor outcome")
        finally:
            try:
                if self._lock is not None and self._lock.is_locked:
                    try:
                        self._lock.release()
                    except OSError as release_error:
                        raise WrdsAttemptMonitorStateError(
                            "WRDS connection-attempt monitor lock cannot be released"
                        ) from release_error
            finally:
                self._lock = None
                self._reservation = None
                self._prior_failures = ()
                self._resolved = False


__all__ = [
    "WrdsAttemptLimitError",
    "WrdsAttemptMonitorBusyError",
    "WrdsAttemptMonitorError",
    "WrdsAttemptMonitorStateError",
    "WrdsConnectionAttemptMonitor",
]
