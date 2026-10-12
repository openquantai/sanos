"""Host-wide, account-scoped WRDS connection leases."""

from __future__ import annotations

import hashlib
import os
import socket
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from cdxcore.version import version
from filelock import FileLock, Timeout

from sanos.artifacts import atomic_write_bytes
from sanos.config import canonical_json


class WrdsLeaseUnavailableError(RuntimeError):
    """Raised when all four host-wide account slots are occupied."""


@version("1.0")
def default_lease_root() -> Path:
    """Return a host-local runtime path independent of configurable data roots."""

    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        base = Path(runtime).expanduser()
    else:
        base = Path("/tmp")
    return base / "sanos" / "wrds-leases"


@version("1.0")
def account_key(username: str) -> str:
    """Return a non-reversible account key suitable for paths and logs."""

    normalized = username.strip().casefold()
    if not normalized:
        raise ValueError("WRDS username cannot be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


@version("1.0", dependencies=[account_key, default_lease_root, atomic_write_bytes])
class WrdsConnectionLease(AbstractContextManager["WrdsConnectionLease"]):
    """One of four cross-process connection slots for a WRDS account."""

    SLOT_COUNT = 4

    def __init__(
        self,
        username: str,
        *,
        root: Path | None = None,
        timeout_seconds: float = 0.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if timeout_seconds < 0.0:
            raise ValueError("WRDS lease timeout cannot be negative")
        if poll_seconds <= 0.0:
            raise ValueError("WRDS lease polling interval must be positive")
        self.account = account_key(username)
        self.root = (default_lease_root() if root is None else root).expanduser().resolve(
            strict=False
        )
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.slot: int | None = None
        self._lock: FileLock | None = None

    @property
    def metadata_path(self) -> Path:
        """Return the active slot's metadata path without exposing the account."""

        if self.slot is None:
            raise RuntimeError("WRDS lease has not been acquired")
        return self.root / self.account / f"slot-{self.slot}.json"

    def __enter__(self) -> WrdsConnectionLease:
        account_root = self.root / self.account
        account_root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            for slot in range(self.SLOT_COUNT):
                lock = FileLock(str(account_root / f"slot-{slot}.lock"), timeout=0)
                try:
                    lock.acquire()
                except Timeout:
                    continue
                self.slot = slot
                self._lock = lock
                payload = {
                    "schema": "sanos-wrds-lease-v1",
                    "account_key": self.account,
                    "slot": slot,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_at": datetime.now(UTC).isoformat(),
                }
                try:
                    atomic_write_bytes(
                        self.metadata_path,
                        (canonical_json(payload) + "\n").encode("utf-8"),
                    )
                except BaseException:
                    lock.release()
                    self._lock = None
                    self.slot = None
                    raise
                return self
            if time.monotonic() >= deadline:
                raise WrdsLeaseUnavailableError(
                    "All four host-wide WRDS connection slots are in use for this account"
                )
            time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self.slot is not None:
                try:
                    self.metadata_path.unlink(missing_ok=True)
                except OSError:
                    # A stale record is harmless: the kernel-backed lock is authoritative,
                    # and the next successful holder replaces its metadata atomically.
                    pass
        finally:
            if self._lock is not None and self._lock.is_locked:
                self._lock.release()
            self._lock = None
            self.slot = None


__all__ = [
    "WrdsConnectionLease",
    "WrdsLeaseUnavailableError",
    "account_key",
    "default_lease_root",
]
