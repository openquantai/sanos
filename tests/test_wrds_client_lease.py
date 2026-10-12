from __future__ import annotations

import json
from pathlib import Path

import pytest

import sanos.wrds.lease as lease_module
from sanos.wrds.client import (
    WrdsAccessDisabledError,
    WrdsAuthenticationError,
    WrdsCredentials,
    WrdsQueryError,
    execute_query_with_retry,
)
from sanos.wrds.lease import (
    WrdsConnectionLease,
    WrdsLeaseUnavailableError,
    account_key,
)


def test_credentials_are_lazy_redacted_and_honor_wrds_off() -> None:
    credentials = WrdsCredentials.from_environment({"WRDS_USERNAME": "private-account"})

    assert "private-account" not in repr(credentials)
    assert len(account_key(credentials.username)) == 24
    assert "private-account" not in account_key(credentials.username)

    with pytest.raises(WrdsAccessDisabledError, match="WRDS_OFF"):
        WrdsCredentials.from_environment(
            {"WRDS_USERNAME": "private-account", "WRDS_OFF": "true"}
        )


def test_query_retry_is_limited_to_recognized_transients() -> None:
    attempts = 0
    waits: list[float] = []

    def transient_then_success() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("connection reset by peer")
        return "ok"

    assert execute_query_with_retry(
        transient_then_success,
        attempts=3,
        base_seconds=0.5,
        sleep=waits.append,
    ) == "ok"
    assert attempts == 3
    assert waits == [0.5, 1.0]

    auth_attempts = 0

    def authentication_failure() -> None:
        nonlocal auth_attempts
        auth_attempts += 1
        raise RuntimeError("two-factor authentication failed")

    with pytest.raises(WrdsAuthenticationError, match="not retried"):
        execute_query_with_retry(
            authentication_failure,
            attempts=5,
            base_seconds=0.0,
        )
    assert auth_attempts == 1

    with pytest.raises(WrdsQueryError, match="permanent_query"):
        execute_query_with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("invalid SQL")),
            attempts=5,
            base_seconds=0.0,
        )


def test_account_scoped_lease_hard_limits_four_and_records_no_username(
    tmp_path: Path,
) -> None:
    root = tmp_path / "leases"
    leases = [WrdsConnectionLease("private-account", root=root) for _ in range(4)]
    for lease in leases:
        lease.__enter__()
    try:
        metadata = [json.loads(lease.metadata_path.read_text()) for lease in leases]
        assert {item["slot"] for item in metadata} == {0, 1, 2, 3}
        assert all(item["pid"] > 0 and item["host"] for item in metadata)
        assert "private-account" not in str(root) + json.dumps(metadata)
        with pytest.raises(WrdsLeaseUnavailableError, match="four"):
            with WrdsConnectionLease("private-account", root=root):
                pass
    finally:
        for lease in reversed(leases):
            lease.__exit__(None, None, None)

    with WrdsConnectionLease("private-account", root=root) as reclaimed:
        assert reclaimed.slot in {0, 1, 2, 3}


def test_lease_releases_slot_when_metadata_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = lease_module.atomic_write_bytes
    calls = 0

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated metadata failure")
        real_write(path, data)

    monkeypatch.setattr(lease_module, "atomic_write_bytes", fail_once)
    failed = WrdsConnectionLease("private-account", root=tmp_path / "leases")
    with pytest.raises(OSError, match="metadata failure"):
        failed.__enter__()
    assert failed.slot is None

    with WrdsConnectionLease("private-account", root=tmp_path / "leases") as recovered:
        assert recovered.slot == 0
