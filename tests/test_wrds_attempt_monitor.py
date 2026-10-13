from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from filelock import FileLock

import sanos.wrds.monitor as monitor_module
from sanos.artifacts import ArtifactStore
from sanos.config import StorageLocations
from sanos.wrds import (
    RawDownloadConfig,
    WrdsAttemptLimitError,
    WrdsAttemptMonitorBusyError,
    WrdsAttemptMonitorStateError,
    WrdsAuthenticationError,
    WrdsConfigurationError,
    WrdsConnectionAttemptMonitor,
    WrdsCredentials,
    download_raw,
)
from sanos.wrds.download import _open_connection_monitored


class _FailedAttempt(RuntimeError):
    pass


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        StorageLocations(
            raw=tmp_path / "raw",
            intermediate=tmp_path / "intermediate",
            datasets=tmp_path / "datasets",
            experiments=tmp_path / "experiments",
        )
    )


def _config() -> RawDownloadConfig:
    return RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        max_connections=1,
        retry_attempts=1,
        retry_base_seconds=0.0,
    )


def _fail(monitor: WrdsConnectionAttemptMonitor) -> None:
    with pytest.raises(_FailedAttempt):
        with monitor:
            raise _FailedAttempt("synthetic connection failure")


def _assert_lock_is_free(monitor: WrdsConnectionAttemptMonitor) -> None:
    with FileLock(
        str(monitor.account_root / "attempt-monitor.lock"),
        timeout=0.0,
    ):
        pass


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else source_root + os.pathsep + inherited
    )
    return environment


def test_monitor_blocks_fourth_failure_and_expires_rolling_window(tmp_path: Path) -> None:
    now = [1_000.0]
    root = tmp_path / "runtime"

    for expected_prior in range(3):
        monitor = WrdsConnectionAttemptMonitor(
            "private-account",
            root=root,
            clock=lambda: now[0],
            lock_timeout_seconds=0.0,
        )
        with pytest.raises(_FailedAttempt):
            with monitor as active:
                assert active.failed_attempts_before_reservation == expected_prior
                assert active.remaining_attempts_after_reservation == 2 - expected_prior
                raise _FailedAttempt("synthetic connection failure")
        now[0] += 1.0

    blocked = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: now[0],
        lock_timeout_seconds=0.0,
    )
    with pytest.raises(WrdsAttemptLimitError, match="fourth"):
        blocked.__enter__()

    now[0] += WrdsConnectionAttemptMonitor.FAILURE_WINDOW_SECONDS
    recovered = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: now[0],
        lock_timeout_seconds=0.0,
    )
    with recovered as active:
        assert active.failed_attempts_before_reservation == 0
        active.succeed()

    state = json.loads(recovered.state_path.read_text(encoding="utf-8"))
    assert state == {
        "failure_timestamps": [],
        "schema": "sanos-wrds-attempt-monitor-v1",
    }


def test_monitor_success_clears_prior_failures_and_never_persists_username(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    first = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: 100.0,
        lock_timeout_seconds=0.0,
    )
    _fail(first)

    second = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: 101.0,
        lock_timeout_seconds=0.0,
    )
    with second as active:
        assert active.failed_attempts_before_reservation == 1
        active.succeed()

    third = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: 102.0,
        lock_timeout_seconds=0.0,
    )
    with third as active:
        assert active.failed_attempts_before_reservation == 0
        active.cancel()

    state_text = third.state_path.read_text(encoding="utf-8")
    assert "private-account" not in str(root) + state_text
    assert json.loads(state_text)["failure_timestamps"] == []


def test_cancelled_pre_network_attempt_preserves_earlier_failure(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    failed = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: 100.0,
        lock_timeout_seconds=0.0,
    )
    _fail(failed)

    cancelled = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        clock=lambda: 101.0,
        lock_timeout_seconds=0.0,
    )
    with cancelled as active:
        assert active.failed_attempts_before_reservation == 1
        active.cancel()

    state = json.loads(cancelled.state_path.read_text(encoding="utf-8"))
    assert state["failure_timestamps"] == [100.0]


def test_monitor_is_busy_while_another_connection_open_is_in_progress(
    tmp_path: Path,
) -> None:
    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=tmp_path / "runtime",
        clock=lambda: 100.0,
        lock_timeout_seconds=0.0,
    )
    monitor.account_root.mkdir(parents=True)
    held = FileLock(str(monitor.account_root / "attempt-monitor.lock"))
    with held:
        with pytest.raises(WrdsAttemptMonitorBusyError, match="in progress"):
            monitor.__enter__()
    assert not monitor.state_path.exists()


def test_corrupt_or_future_monitor_state_fails_closed(tmp_path: Path) -> None:
    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=tmp_path / "runtime",
        clock=lambda: 100.0,
        lock_timeout_seconds=0.0,
    )
    monitor.account_root.mkdir(parents=True)
    monitor.state_path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(WrdsAttemptMonitorStateError, match="unreadable"):
        monitor.__enter__()

    monitor.state_path.write_text(
        '{ "failure_timestamps": [], "schema": "sanos-wrds-attempt-monitor-v1" }\n',
        encoding="utf-8",
    )
    with pytest.raises(WrdsAttemptMonitorStateError, match="not canonical"):
        monitor.__enter__()

    monitor.state_path.write_text(
        '{"failure_timestamps":[101.0],"schema":"sanos-wrds-attempt-monitor-v1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(WrdsAttemptMonitorStateError, match="timestamps"):
        monitor.__enter__()


def test_monitor_normalizes_directory_initialization_failure(tmp_path: Path) -> None:
    blocked_root = tmp_path / "not-a-directory"
    blocked_root.write_text("file", encoding="utf-8")
    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=blocked_root,
        lock_timeout_seconds=0.0,
    )

    with pytest.raises(WrdsAttemptMonitorStateError, match="directory"):
        monitor.__enter__()


def test_monitor_normalizes_lock_acquisition_failure_and_releases_partial_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenLock:
        is_locked = False
        released = False

        def acquire(self, *, timeout: float) -> None:
            assert timeout == 0.0
            self.is_locked = True
            raise PermissionError("synthetic lock failure")

        def release(self) -> None:
            self.is_locked = False
            self.released = True

    broken = BrokenLock()
    monkeypatch.setattr(monitor_module, "FileLock", lambda _path: broken)
    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=tmp_path / "runtime",
        lock_timeout_seconds=0.0,
    )

    with pytest.raises(WrdsAttemptMonitorStateError, match="lock cannot be acquired"):
        monitor.__enter__()
    assert broken.released


def test_monitor_normalizes_reservation_write_failure_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=tmp_path / "runtime",
        lock_timeout_seconds=0.0,
    )
    monkeypatch.setattr(
        monitor_module,
        "atomic_write_bytes",
        lambda _path, _data: (_ for _ in ()).throw(
            PermissionError("synthetic reservation failure")
        ),
    )

    with pytest.raises(WrdsAttemptMonitorStateError, match="cannot be updated"):
        monitor.__enter__()
    _assert_lock_is_free(monitor)


@pytest.mark.parametrize("outcome", ["succeed", "cancel"])
def test_monitor_normalizes_outcome_write_failure_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    real_write = monitor_module.atomic_write_bytes
    write_count = 0

    def fail_outcome(path: Path, data: bytes) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise PermissionError("synthetic outcome failure")
        real_write(path, data)

    monkeypatch.setattr(monitor_module, "atomic_write_bytes", fail_outcome)
    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=tmp_path / outcome,
        lock_timeout_seconds=0.0,
    )

    with pytest.raises(WrdsAttemptMonitorStateError, match="cannot be updated"):
        with monitor as active:
            getattr(active, outcome)()
    _assert_lock_is_free(monitor)


def test_monitor_serializes_opens_across_independent_processes(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    script = """
import sys
from pathlib import Path
from sanos.wrds.monitor import WrdsConnectionAttemptMonitor

with WrdsConnectionAttemptMonitor(
    "private-account", root=Path(sys.argv[1]), lock_timeout_seconds=0.0
) as monitor:
    print("READY", flush=True)
    sys.stdin.readline()
    monitor.cancel()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_subprocess_environment(),
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        competing = WrdsConnectionAttemptMonitor(
            "private-account",
            root=root,
            lock_timeout_seconds=0.0,
        )
        with pytest.raises(WrdsAttemptMonitorBusyError, match="in progress"):
            competing.__enter__()
        stdout, stderr = process.communicate(input="\n", timeout=10.0)
        assert process.returncode == 0, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10.0)


def test_hard_process_exit_retains_reserved_attempt(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    script = """
import os
import sys
from pathlib import Path
from sanos.wrds.monitor import WrdsConnectionAttemptMonitor

with WrdsConnectionAttemptMonitor(
    "private-account", root=Path(sys.argv[1]), lock_timeout_seconds=0.0
):
    print("READY", flush=True)
    os._exit(23)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
        timeout=10.0,
        check=False,
    )
    assert process.returncode == 23, process.stdout + process.stderr
    assert process.stdout.strip() == "READY"

    recovered = WrdsConnectionAttemptMonitor(
        "private-account",
        root=root,
        lock_timeout_seconds=0.0,
    )
    with recovered as active:
        assert active.failed_attempts_before_reservation == 1
        active.cancel()
    state = json.loads(recovered.state_path.read_text(encoding="utf-8"))
    assert len(state["failure_timestamps"]) == 1


def test_attempt_monitor_version_dependencies_are_exact() -> None:
    monitor_dependencies = set(
        dict(cast(Any, WrdsConnectionAttemptMonitor.version).dependencies[1])
    )
    open_dependencies = set(
        dict(cast(Any, _open_connection_monitored.version).dependencies[1])
    )

    assert monitor_dependencies == {
        "account_key",
        "atomic_write_bytes",
        "canonical_json",
        "default_lease_root",
    }
    assert open_dependencies == {
        "WrdsConnectionAttemptMonitor",
        "_close_without_masking",
        "_open_connection",
        "canonical_value",
    }


def test_monitored_open_closes_connection_once_when_lock_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReleaseFailureLock:
        def __init__(self, _path: str) -> None:
            self.is_locked = False

        def acquire(self, *, timeout: float) -> None:
            assert timeout == WrdsConnectionAttemptMonitor.LOCK_TIMEOUT_SECONDS
            self.is_locked = True

        def release(self) -> None:
            raise PermissionError("synthetic release failure")

    class Connection:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    factory_calls = 0
    connection = Connection()

    def factory(_username: str) -> Connection:
        nonlocal factory_calls
        factory_calls += 1
        return connection

    monkeypatch.setattr(monitor_module, "FileLock", ReleaseFailureLock)

    with pytest.raises(WrdsAttemptMonitorStateError, match="lock cannot be released"):
        _open_connection_monitored(
            factory,
            WrdsCredentials(username="private-account"),
            monitor_root=tmp_path / "runtime",
            emit=lambda _payload: None,
        )
    assert factory_calls == 1
    assert connection.close_count == 1


def test_download_monitor_counts_connection_failures_and_blocks_fourth(
    tmp_path: Path,
) -> None:
    calls = 0

    def fail(_username: str) -> None:
        nonlocal calls
        calls += 1
        raise WrdsAuthenticationError("authentication failed without a retry")

    for _ in range(3):
        with pytest.raises(WrdsAuthenticationError):
            download_raw(
                _store(tmp_path / "data"),
                _config(),
                connection_factory=fail,
                environment={"WRDS_USERNAME": "private-account"},
                lease_root=tmp_path / "runtime",
            )

    with pytest.raises(WrdsAttemptLimitError, match="fourth"):
        download_raw(
            _store(tmp_path / "data"),
            _config(),
            connection_factory=fail,
            environment={"WRDS_USERNAME": "private-account"},
            lease_root=tmp_path / "runtime",
        )
    assert calls == 3


def test_pre_network_configuration_failure_does_not_consume_attempt(
    tmp_path: Path,
) -> None:
    def misconfigured(_username: str) -> None:
        raise WrdsConfigurationError("optional WRDS runtime is unavailable")

    with pytest.raises(WrdsConfigurationError):
        download_raw(
            _store(tmp_path / "data"),
            _config(),
            connection_factory=misconfigured,
            environment={"WRDS_USERNAME": "private-account"},
            lease_root=tmp_path / "runtime",
        )

    monitor = WrdsConnectionAttemptMonitor(
        "private-account",
        root=tmp_path / "runtime",
        lock_timeout_seconds=0.0,
    )
    state = json.loads(monitor.state_path.read_text(encoding="utf-8"))
    assert state["failure_timestamps"] == []


def test_query_attempt_configuration_is_hard_capped_at_three() -> None:
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        RawDownloadConfig(
            ticker="SPX",
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 3),
            retry_attempts=4,
        )
