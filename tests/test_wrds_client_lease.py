from __future__ import annotations

import builtins
import getpass
import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import sanos.wrds.client as client_module
import sanos.wrds.lease as lease_module
from sanos.wrds.client import (
    WrdsAccessDisabledError,
    WrdsAuthenticationError,
    WrdsConfigurationError,
    WrdsCredentials,
    WrdsQueryError,
    default_connection_factory,
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


def test_default_connection_factory_uses_fail_closed_wrds_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[object] = []

    class FakeConnection:
        def __init__(self, *, autoconnect: bool, wrds_username: str) -> None:
            calls.append((autoconnect, wrds_username))
            self.engine: object | None = None

        def _Connection__make_sa_engine_conn(self, *, raise_err: bool) -> None:
            calls.append(("connect", raise_err))
            self.engine = object()

        def raw_sql(self, sql: str) -> dict[str, list[str]]:
            if "AS library" in sql:
                calls.append("libraries")
                return {"library": ["optionm_all"]}
            if "AS table_name" in sql:
                calls.append("tables")
                return {"table_name": ["securd", "opprcd2025"]}
            raise AssertionError("unexpected SQL")

        def close(self) -> None:
            calls.append("close")

    fake_wrds = types.SimpleNamespace(Connection=FakeConnection)
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: fake_wrds)
    connection = default_connection_factory("private-account")

    assert connection.list_libraries() == ["optionm_all"]
    assert connection.list_tables("optionm_all") == ["securd", "opprcd2025"]
    assert calls == [
        (False, "private-account"),
        ("connect", True),
        "libraries",
        "tables",
    ]
    assert capsys.readouterr() == ("", "")


def test_default_connection_factory_has_resolved_version() -> None:
    assert str(default_connection_factory.version).startswith("1.1")


def test_default_connection_factory_never_prompts_or_leaks_authentication_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingConnection:
        engine = None

        def __init__(self, *, autoconnect: bool, wrds_username: str) -> None:
            assert autoconnect is False
            assert wrds_username == "private-account"

        def _Connection__make_sa_engine_conn(self, *, raise_err: bool) -> None:
            assert raise_err is True
            raise RuntimeError(
                'password authentication failed for user "private-account"; secret-value'
            )

    def forbidden_prompt(*args: object, **kwargs: object) -> str:
        raise AssertionError("credential prompt must not be called")

    fake_wrds = types.SimpleNamespace(Connection=FailingConnection)
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: fake_wrds)
    monkeypatch.setattr(builtins, "input", forbidden_prompt)
    monkeypatch.setattr(getpass, "getpass", forbidden_prompt)

    with pytest.raises(WrdsAuthenticationError) as caught:
        default_connection_factory("private-account")

    rendered = f"{caught.value!r} {caught.value}"
    assert "private-account" not in rendered
    assert "secret-value" not in rendered
    assert caught.value.__cause__ is None
    assert capsys.readouterr() == ("", "")


def test_default_connection_factory_rejects_incompatible_wrds_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompatibleConnection:
        def __init__(self, *, autoconnect: bool, wrds_username: str) -> None:
            assert autoconnect is False
            assert wrds_username == "private-account"

    fake_wrds = types.SimpleNamespace(Connection=IncompatibleConnection)
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: fake_wrds)

    with pytest.raises(WrdsConfigurationError, match="fail-closed"):
        default_connection_factory("private-account")


def test_default_connection_factory_closes_after_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    class PartiallyConnectedFailure:
        def __init__(self, *, autoconnect: bool, wrds_username: str) -> None:
            assert autoconnect is False
            assert wrds_username == "private-account"
            self.engine: object | None = None

        def _Connection__make_sa_engine_conn(self, *, raise_err: bool) -> None:
            assert raise_err is True
            self.engine = object()
            raise RuntimeError("password authentication failed; secret-value")

        def close(self) -> None:
            calls.append("close")
            self.engine = None

    fake_wrds = types.SimpleNamespace(Connection=PartiallyConnectedFailure)
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: fake_wrds)

    with pytest.raises(WrdsAuthenticationError) as caught:
        default_connection_factory("private-account")

    assert calls == ["close"]
    assert "secret-value" not in f"{caught.value!r} {caught.value}"
    assert caught.value.__cause__ is None
    assert capsys.readouterr() == ("", "")


def test_default_connection_factory_does_not_mutate_streams_during_fanout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    barrier = threading.Barrier(2)

    class ConcurrentConnection:
        def __init__(self, *, autoconnect: bool, wrds_username: str) -> None:
            assert autoconnect is False
            assert wrds_username == "private-account"
            self.engine: object | None = None

        def _Connection__make_sa_engine_conn(self, *, raise_err: bool) -> None:
            assert raise_err is True
            barrier.wait(timeout=2.0)
            self.engine = object()

    fake_wrds = types.SimpleNamespace(Connection=ConcurrentConnection)
    monkeypatch.setattr(client_module.importlib, "import_module", lambda name: fake_wrds)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def open_and_report(index: int) -> None:
        default_connection_factory("private-account")
        print(f'{{"event":"progress","worker":{index}}}')

    with ThreadPoolExecutor(max_workers=2) as pool:
        tuple(pool.map(open_and_report, range(2)))

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    captured = capsys.readouterr()
    assert captured.err == ""
    assert sorted(captured.out.splitlines()) == [
        '{"event":"progress","worker":0}',
        '{"event":"progress","worker":1}',
    ]


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
