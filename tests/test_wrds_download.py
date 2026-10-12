from __future__ import annotations

import json
import re
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sanos.artifacts import ArtifactStore
from sanos.config import StorageLocations
from sanos.wrds import (
    RawCacheMissError,
    RawDownloadConfig,
    RawDownloadError,
    RawTable,
    WrdsNetworkAuthorizationError,
    download_raw,
    normalize_raw_frame,
    raw_content_producer_version,
)
from sanos.wrds.plan import RawPartition


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        StorageLocations(
            raw=tmp_path / "raw",
            intermediate=tmp_path / "intermediate",
            datasets=tmp_path / "datasets",
            experiments=tmp_path / "experiments",
        )
    )


def _option_frame(
    observation_date: str = "2024-01-03",
    *,
    secid: int = 108105,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "secid": [secid],
            "date": [observation_date],
            "symbol": ["SPX240119C04500000"],
            "optionid": [123],
            "am_settlement": [0],
            "cp_flag": ["C"],
            "strike_price": [4_500_000],
            "exdate": ["2024-01-19"],
            "best_bid": [20.0],
            "best_offer": [21.0],
            "volume": [100],
            "open_interest": [1000],
            "impl_volatility": [0.2],
            "ss_flag": [0],
            "delta": [0.5],
            "gamma": [0.01],
            "vega": [1.0],
            "theta": [-0.5],
            "expiry_indicator": ["w"],
            "contract_size": [100],
        }
    )


class _FakeWrdsState:
    def __init__(
        self,
        *,
        query_delay_seconds: float = 0.0,
        empty_options: bool = False,
        security_secid: int = 108105,
        security_barrier: threading.Barrier | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self.created = 0
        self.closed = 0
        self.active = 0
        self.max_active = 0
        self.queries: list[str] = []
        self.query_delay_seconds = query_delay_seconds
        self.empty_options = empty_options
        self.security_secid = security_secid
        self.security_barrier = security_barrier

    def factory(self, username: str) -> _FakeConnection:
        assert username == "test-account"
        return _FakeConnection(self)


class _FakeConnection:
    def __init__(self, state: _FakeWrdsState) -> None:
        self.state = state
        self.is_closed = False
        with state.lock:
            state.created += 1
            state.active += 1
            state.max_active = max(state.max_active, state.active)

    def list_libraries(self) -> list[str]:
        return ["other", "optionm_all", "optionm"]

    def list_tables(self, library: str) -> list[str]:
        assert library == "optionm_all"
        return [
            "securd",
            "opprcd2023",
            "opprcd2024",
            "secprd",
            "fwdprd2023",
            "fwdprd2024",
            "zerocd",
        ]

    def raw_sql(self, sql: str) -> pl.DataFrame:
        with self.state.lock:
            self.state.queries.append(sql)
        time.sleep(self.state.query_delay_seconds)
        if "FROM optionm_all.securd " in sql:
            if self.state.security_barrier is not None:
                self.state.security_barrier.wait(timeout=5.0)
            return pl.DataFrame(
                {"ticker": ["SPX"], "secid": [self.state.security_secid]}
            )
        date_match = re.search(r"date = '(\d{4}-\d{2}-\d{2})'", sql)
        assert date_match is not None
        observation_date = date_match.group(1)
        secid_match = re.search(r"secid = (\d+)", sql)
        requested_secid = int(secid_match.group(1)) if secid_match is not None else None
        if ".opprcd" in sql:
            assert requested_secid is not None
            frame = _option_frame(observation_date, secid=requested_secid)
            return frame.head(0) if self.state.empty_options else frame
        if ".secprd" in sql:
            assert requested_secid is not None
            return pl.DataFrame(
                {
                    "secid": [requested_secid],
                    "date": [observation_date],
                    "close": [4700.0],
                }
            )
        if ".fwdprd" in sql:
            assert requested_secid is not None
            return pl.DataFrame(
                {
                    "secid": [requested_secid],
                    "date": [observation_date],
                    "expiration": ["2024-01-19"],
                    "amsettlement": [0],
                    "forwardprice": [4710.0],
                }
            )
        if ".zerocd" in sql:
            return pl.DataFrame(
                {"date": [observation_date], "days": [30], "rate": [5.0]}
            )
        raise AssertionError(f"Unexpected SQL: {sql}")

    def close(self) -> None:
        assert not self.is_closed
        self.is_closed = True
        with self.state.lock:
            self.state.closed += 1
            self.state.active -= 1


def _config(*, generation: str = "default-v1") -> RawDownloadConfig:
    return RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        source_generation=generation,
        max_connections=2,
        retry_base_seconds=0.0,
    )


def test_dry_run_and_cache_only_never_authorize_connect_or_create_roots(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    called = False

    def forbidden() -> None:
        nonlocal called
        called = True
        raise AssertionError("network authorization must not run")

    dry = download_raw(
        store,
        _config(),
        dry_run=True,
        authorize_network=forbidden,
        environment={},
        connection_factory=lambda username: (_ for _ in ()).throw(AssertionError(username)),
    )
    assert dry.missing == 8
    assert dry.network_accessed is False
    assert not called
    assert not (tmp_path / "raw").exists()

    with pytest.raises(RawCacheMissError, match="8 missing"):
        download_raw(
            store,
            _config(),
            cache_only=True,
            authorize_network=forbidden,
            environment={},
        )
    assert not called


def test_empty_dated_vendor_result_is_accepted_for_durable_publication() -> None:
    columns = _option_frame().head(0)
    partition = RawPartition(RawTable.OPTIONS, "SPX", date(2024, 1, 3))

    normalized = normalize_raw_frame(columns, partition, expected_secid=108105)

    assert normalized.is_empty()
    assert normalized.columns == _option_frame().columns


@pytest.mark.parametrize(
    ("frame", "partition", "expected_secid", "message"),
    [
        (
            _option_frame("2024-01-02"),
            RawPartition(RawTable.OPTIONS, "SPX", date(2024, 1, 3)),
            108105,
            "date partition",
        ),
        (
            _option_frame().with_columns(pl.lit(999999).alias("secid")),
            RawPartition(RawTable.OPTIONS, "SPX", date(2024, 1, 3)),
            108105,
            "secid partition",
        ),
        (
            pl.DataFrame({"ticker": ["NDX"], "secid": [108105]}),
            RawPartition(RawTable.SECURITY, "SPX", None),
            None,
            "ticker partition",
        ),
    ],
)
def test_raw_rows_must_match_their_immutable_partition(
    frame: pl.DataFrame,
    partition: RawPartition,
    expected_secid: int | None,
    message: str,
) -> None:
    with pytest.raises(RawDownloadError, match=message):
        normalize_raw_frame(frame, partition, expected_secid=expected_secid)


def test_empty_dated_vendor_result_is_published_and_reusable(tmp_path: Path) -> None:
    config = RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        tables=(RawTable.OPTIONS,),
        retry_base_seconds=0.0,
    )
    store = _store(tmp_path)
    state = _FakeWrdsState(empty_options=True)

    result = download_raw(
        store,
        config,
        authorize_network=lambda: None,
        connection_factory=state.factory,
        environment={"WRDS_USERNAME": "test-account"},
        lease_root=tmp_path / "leases",
    )
    option = next(
        artifact
        for artifact in result.artifacts
        if artifact.manifest["configuration"]["table"] == "options"
    )
    assert pl.read_parquet(option.path / "data.parquet").is_empty()
    assert option.manifest["metadata"]["row_count"] == 0
    assert option.manifest["metadata"]["empty_vendor_result"] is True
    provenance = json.loads((option.path / "query.json").read_text())
    assert provenance["empty_vendor_result"] is True

    reused = download_raw(store, config, environment={})
    assert reused.reused == 2 and reused.network_accessed is False


def test_live_download_is_guarded_bounded_documented_and_output_first(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    state = _FakeWrdsState()
    authorization_calls = 0
    events: list[dict[str, Any]] = []

    def authorize() -> None:
        nonlocal authorization_calls
        authorization_calls += 1

    result = download_raw(
        store,
        _config(),
        authorize_network=authorize,
        progress=lambda payload: events.append(dict(payload)),
        connection_factory=state.factory,
        environment={"WRDS_USERNAME": "test-account"},
        lease_root=tmp_path / "leases",
    )

    assert authorization_calls == 1
    assert result.planned == result.downloaded == 8
    assert result.reused == result.missing == 0
    assert result.network_accessed is True
    assert state.created == state.closed == 3  # one closed probe, then two workers
    assert state.max_active <= 2
    assert events[0]["event"] == "raw_plan"
    assert events[0]["boundary_partitions"] == 3
    assert events[-1]["event"] == "raw_download_complete"

    option_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.manifest["configuration"]["table"] == "options"
    )
    frame = pl.read_parquet(option_artifact.path / "data.parquet")
    assert frame.item(0, "strike_price") == 4_500_000
    readme = (option_artifact.path / "README.md").read_text(encoding="utf-8")
    provenance = json.loads((option_artifact.path / "query.json").read_text())
    assert "thousandths of one currency/index unit" in readme
    assert "`strike_price` (`Int64`)" in readme
    assert provenance["persisted_schema"]["strike_price"] == "Int64"
    assert provenance["physical_table"] == "optionm_all.opprcd2024"
    assert "test-account" not in readme + json.dumps(provenance)

    def forbidden_factory(username: str) -> _FakeConnection:
        raise AssertionError(f"cache hit opened WRDS for {username}")

    reused = download_raw(
        store,
        _config(),
        environment={},
        connection_factory=forbidden_factory,
        authorize_network=lambda: (_ for _ in ()).throw(AssertionError("authorization")),
        lease_root=tmp_path / "leases",
    )
    assert reused.reused == 8
    assert reused.downloaded == 0
    assert reused.network_accessed is False


def test_security_publication_race_uses_the_winning_secid_downstream(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    barrier = threading.Barrier(2)
    states = (
        _FakeWrdsState(security_secid=108105, security_barrier=barrier),
        _FakeWrdsState(security_secid=999999, security_barrier=barrier),
    )
    config = RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        tables=(RawTable.OPTIONS,),
        max_connections=1,
        retry_base_seconds=0.0,
    )
    errors: list[BaseException] = []

    def run(state: _FakeWrdsState) -> None:
        try:
            download_raw(
                store,
                config,
                authorize_network=lambda: None,
                connection_factory=state.factory,
                environment={"WRDS_USERNAME": "test-account"},
                lease_root=tmp_path / "leases",
            )
        except BaseException as exc:
            errors.append(exc)

    threads = tuple(threading.Thread(target=run, args=(state,)) for state in states)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    security_artifacts = tuple((tmp_path / "raw").glob("wrds/**/security/**/data.parquet"))
    assert len(security_artifacts) == 1
    winning_secid = pl.read_parquet(security_artifacts[0]).item(0, "secid")
    downstream_sql = [
        sql
        for state in states
        for sql in state.queries
        if ".opprcd" in sql
    ]
    assert len(downstream_sql) == 2
    assert all(f"secid = {winning_secid}" in sql for sql in downstream_sql)


def test_live_miss_requires_immediate_authorization_before_credentials(
    tmp_path: Path,
) -> None:
    with pytest.raises(WrdsNetworkAuthorizationError, match="2FA"):
        download_raw(_store(tmp_path), _config(), environment={})


def test_refresh_resumes_generation_and_never_replaces_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_state = _FakeWrdsState()
    first_config = _config(generation="2026-10-12-refresh-1")
    first = download_raw(
        store,
        first_config,
        authorize_network=lambda: None,
        connection_factory=first_state.factory,
        environment={"WRDS_USERNAME": "test-account"},
        lease_root=tmp_path / "leases",
    )
    old_paths = {artifact.path for artifact in first.artifacts}

    resumed = download_raw(
        store,
        first_config,
        refresh=True,
        authorize_network=lambda: (_ for _ in ()).throw(AssertionError("authorization")),
        connection_factory=lambda username: (_ for _ in ()).throw(
            AssertionError(username)
        ),
        environment={},
    )
    assert resumed.reused == 8
    assert resumed.downloaded == 0
    assert {artifact.path for artifact in resumed.artifacts} == old_paths

    second_state = _FakeWrdsState()
    refreshed = download_raw(
        store,
        _config(generation="2026-10-12-refresh-2"),
        refresh=True,
        authorize_network=lambda: None,
        connection_factory=second_state.factory,
        environment={"WRDS_USERNAME": "test-account"},
        lease_root=tmp_path / "leases",
    )
    assert refreshed.downloaded == 8
    assert old_paths.isdisjoint({artifact.path for artifact in refreshed.artifacts})
    assert all(path.exists() for path in old_paths)


def test_wildcard_reuse_can_never_build_raw_data(tmp_path: Path) -> None:
    with pytest.raises(RawCacheMissError, match=r"version='\*' cannot build"):
        download_raw(
            _store(tmp_path),
            _config(),
            dry_run=True,
            version_override="*",
            environment={},
        )


def test_table_subset_downloads_only_security_and_requested_table(tmp_path: Path) -> None:
    config = RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        tables=(RawTable.OPTIONS,),
        retry_base_seconds=0.0,
    )
    state = _FakeWrdsState()
    result = download_raw(
        _store(tmp_path),
        config,
        authorize_network=lambda: None,
        connection_factory=state.factory,
        environment={"WRDS_USERNAME": "test-account"},
        lease_root=tmp_path / "leases",
    )
    assert result.planned == 2
    assert len(state.queries) == 2


def test_blocking_queries_emit_progress_heartbeats(tmp_path: Path) -> None:
    config = RawDownloadConfig(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        tables=(RawTable.OPTIONS,),
        retry_base_seconds=0.0,
    )
    events: list[dict[str, Any]] = []
    state = _FakeWrdsState(query_delay_seconds=0.03)

    download_raw(
        _store(tmp_path),
        config,
        authorize_network=lambda: None,
        progress=lambda payload: events.append(dict(payload)),
        connection_factory=state.factory,
        environment={"WRDS_USERNAME": "test-account"},
        lease_root=tmp_path / "leases",
        heartbeat_interval_seconds=0.01,
    )

    heartbeats = [event for event in events if event["event"] == "raw_progress"]
    assert heartbeats
    assert all(event["total"] == 2 and event["failures"] == 0 for event in heartbeats)


def test_raw_producer_version_closes_over_content_and_safety_contracts() -> None:
    resolved = str(download_raw.version)

    for dependency in (
        "RawDownloadConfig",
        "WrdsConnectionLease",
        "_publish_raw_partition",
        "_write_parquet_durable",
        "build_raw_query",
        "normalize_raw_frame",
        "plan_raw_partitions",
        "raw_artifact_request",
        "security_secid",
    ):
        assert dependency in resolved

    content = raw_content_producer_version()
    assert "_publish_raw_partition" in content
    assert "build_raw_query" in content
    assert "normalize_raw_frame" in content
    assert "inspect_optionmetrics" in content
    assert "security_secid" in content
    assert "_safe_ticker_label" in content
    assert "_date_literal" in content
    assert "WrdsConnectionLease" not in content
