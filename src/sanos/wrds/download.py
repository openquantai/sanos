"""Output-first, resumable raw WRDS OptionMetrics extraction."""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import polars as pl
from cdxcore.version import version

from sanos.artifacts import (
    ArtifactExistsError,
    ArtifactSpec,
    ArtifactStore,
    StorageClass,
    ValidatedArtifact,
    WildcardReuseMissError,
)
from sanos.config import canonical_value
from sanos.wrds.client import (
    ConnectionFactory,
    WrdsAuthenticationError,
    WrdsCredentials,
    WrdsFailureKind,
    WrdsQueryError,
    classify_wrds_failure,
    close_connection,
    default_connection_factory,
    execute_query_with_retry,
    inspect_optionmetrics,
)
from sanos.wrds.lease import WrdsConnectionLease
from sanos.wrds.plan import RawDownloadConfig, RawPartition, RawTable, plan_raw_partitions
from sanos.wrds.source import (
    RawQuery,
    build_raw_query,
    raw_table_spec,
    render_source_readme,
)

RAW_SCHEMA_VERSION = "wrds-optionmetrics-raw-v1"
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


class RawDownloadError(RuntimeError):
    """Base class for raw extraction failures."""


class RawCacheMissError(RawDownloadError):
    """Raised when cache-only validation finds missing partitions."""


class WrdsNetworkAuthorizationError(RawDownloadError):
    """Raised when a live miss has not received immediate 2FA confirmation."""


RawProgress = Callable[[Mapping[str, Any]], None]


@version("1.0", dependencies=[RawPartition])
@dataclass(frozen=True, slots=True)
class RawArtifactRequest:
    """One planned partition and its immutable artifact identity."""

    partition: RawPartition
    spec: ArtifactSpec


@version("1.0", dependencies=[ValidatedArtifact])
@dataclass(frozen=True, slots=True)
class RawDownloadResult:
    """Summary returned to the CLI or another non-agent caller."""

    planned: int
    reused: int
    downloaded: int
    missing: int
    network_accessed: bool
    dry_run: bool
    artifacts: tuple[ValidatedArtifact, ...]


@version("1.0")
def _safe_ticker_label(ticker: str) -> str:
    label = _SAFE_LABEL.sub("-", ticker).strip("-") or "ticker"
    return f"ticker-{label}"


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        RawArtifactRequest,
        StorageClass,
        _safe_ticker_label,
        raw_table_spec,
    ],
)
def raw_artifact_request(
    partition: RawPartition,
    *,
    source: str,
    source_generation: str,
    producer_version: str,
) -> RawArtifactRequest:
    """Create a content identity without needing credentials or a source catalog."""

    ticker = _safe_ticker_label(partition.ticker)
    if partition.observation_date is None:
        suffix = ticker
    else:
        suffix = (
            f"{ticker}/year-{partition.observation_date.year}/"
            f"date-{partition.observation_date.isoformat()}"
        )
    namespace = f"wrds/{source}/{partition.table.value}/{suffix}"
    configuration = {
        "source": source,
        "table": partition.table.value,
        "ticker": partition.ticker,
        "observation_date": partition.observation_date,
    }
    spec = ArtifactSpec.create(
        storage_class=StorageClass.RAW,
        namespace=namespace,
        configuration=configuration,
        schema_version=f"{RAW_SCHEMA_VERSION}-{partition.table.value}",
        producer_version=producer_version,
        source_generation=source_generation,
    )
    return RawArtifactRequest(partition=partition, spec=spec)


@version("1.0", dependencies=[RawPartition, raw_table_spec])
def normalize_raw_frame(
    value: Any,
    partition: RawPartition,
    *,
    expected_secid: int | None,
) -> pl.DataFrame:
    """Convert to Polars and prove rows belong to their immutable partition."""

    table = partition.table
    if isinstance(value, pl.DataFrame):
        frame = value.clone()
    else:
        try:
            frame = pl.from_pandas(value, include_index=False)
        except Exception as exc:
            raise RawDownloadError(
                f"WRDS {table.value} result cannot be converted to a Polars frame"
            ) from exc
    expected = raw_table_spec(table).columns
    actual = tuple(frame.columns)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RawDownloadError(
            f"WRDS {table.value} schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    frame = frame.select(expected)
    if frame.is_empty():
        return frame
    if table is RawTable.SECURITY:
        tickers = (
            frame.get_column("ticker")
            .cast(pl.String, strict=False)
            .str.to_uppercase()
            .to_list()
        )
        if any(ticker != partition.ticker for ticker in tickers):
            raise RawDownloadError(
                f"WRDS security rows do not match ticker partition {partition.ticker!r}"
            )
        return frame

    observation_date = partition.observation_date
    if observation_date is None:
        raise AssertionError("A dated raw partition has no observation date")
    date_values = frame.get_column("date")
    if date_values.dtype == pl.String:
        dates = date_values.str.to_date(strict=False)
    elif date_values.dtype == pl.Date:
        dates = date_values
    elif date_values.dtype.base_type() == pl.Datetime:
        dates = date_values.cast(pl.Date)
    else:
        raise RawDownloadError(
            f"WRDS {table.value} date has unsupported dtype {date_values.dtype}"
        )
    if dates.null_count() or set(dates.to_list()) != {observation_date}:
        raise RawDownloadError(
            f"WRDS {table.value} rows do not match date partition "
            f"{observation_date.isoformat()}"
        )
    if table is not RawTable.ZERO_CURVE:
        if isinstance(expected_secid, bool) or not isinstance(expected_secid, int):
            raise ValueError(f"A secid is required to validate {table.value} rows")
        secids = frame.get_column("secid").cast(pl.Int64, strict=False)
        if secids.null_count() or set(secids.to_list()) != {expected_secid}:
            raise RawDownloadError(
                f"WRDS {table.value} rows do not match secid partition query"
            )
    return frame


@version("1.0")
def security_secid(frame: pl.DataFrame, ticker: str) -> int:
    """Resolve one unique positive secid from a raw security partition."""

    normalized = frame.filter(pl.col("ticker").cast(pl.String).str.to_uppercase() == ticker.upper())
    values = normalized.get_column("secid").drop_nulls().unique().to_list()
    if len(values) != 1:
        raise RawDownloadError(
            f"OptionMetrics ticker {ticker!r} resolves to {len(values)} distinct secids"
        )
    secid = int(values[0])
    if secid <= 0:
        raise RawDownloadError(f"OptionMetrics ticker {ticker!r} has an invalid secid")
    return secid


@version("1.0")
def _write_parquet_durable(frame: pl.DataFrame, path: Path) -> None:
    """Write one private Parquet payload and flush it before directory publication."""

    frame.write_parquet(path, compression="zstd")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


@version(
    "1.0",
    dependencies=[
        ArtifactStore,
        RawArtifactRequest,
        RawQuery,
        _write_parquet_durable,
        render_source_readme,
        raw_table_spec,
    ],
)
def _publish_raw_partition(
    store: ArtifactStore,
    request: RawArtifactRequest,
    query: RawQuery,
    frame: pl.DataFrame,
) -> ValidatedArtifact:
    persisted_schema = {name: str(dtype) for name, dtype in frame.schema.items()}
    metadata = {
        "row_count": frame.height,
        "column_count": frame.width,
        "empty_vendor_result": frame.is_empty(),
        "persisted_schema": persisted_schema,
        "query_sha256": query.sha256,
        "physical_table": query.physical_table,
    }
    builder = store.writer(request.spec, metadata=metadata)
    try:
        with builder as staging:
            _write_parquet_durable(frame, staging.staging_path / "data.parquet")
            staging.write_text(
                "README.md",
                render_source_readme(
                    query,
                    row_count=frame.height,
                    persisted_schema=persisted_schema,
                ),
            )
            staging.write_json(
                "query.json",
                {
                    "source": "wrds",
                    "dataset": "optionmetrics",
                    "logical_table": request.partition.table.value,
                    "physical_table": query.physical_table,
                    "ticker": request.partition.ticker,
                    "observation_date": request.partition.observation_date,
                    "columns": raw_table_spec(request.partition.table).columns,
                    "persisted_schema": persisted_schema,
                    "row_count": frame.height,
                    "empty_vendor_result": frame.is_empty(),
                    "sql": query.sql,
                    "sql_sha256": query.sha256,
                },
            )
    except ArtifactExistsError:
        # Another SANOS process may have published the exact immutable request
        # while this process was querying. Validate and reuse its completed output.
        return store.validate(request.spec.pointer, expected=request.spec)
    return builder.artifact


def _read_security_artifact(artifact: ValidatedArtifact, ticker: str) -> int:
    frame = pl.read_parquet(artifact.path / "data.parquet")
    return security_secid(frame, ticker)


def _open_connection(
    factory: ConnectionFactory,
    credentials: WrdsCredentials,
) -> Any:
    try:
        return factory(credentials.username)
    except Exception as exc:
        kind = classify_wrds_failure(exc)
        if kind is WrdsFailureKind.AUTHENTICATION:
            raise WrdsAuthenticationError(
                "WRDS authentication or 2FA failed; the connection was not retried"
            ) from exc
        raise WrdsQueryError("WRDS connection failed and was not retried") from exc


def _inspect_optionmetrics_guarded(connection: Any) -> Any:
    try:
        return inspect_optionmetrics(connection)
    except Exception as exc:
        kind = classify_wrds_failure(exc)
        if kind is WrdsFailureKind.AUTHENTICATION:
            raise WrdsAuthenticationError(
                "WRDS OptionMetrics access or permission check failed; it was not retried"
            ) from exc
        raise WrdsQueryError("WRDS OptionMetrics catalog probe failed and was not retried") from exc


def _close_without_masking(connection: Any, primary: BaseException | None) -> None:
    try:
        close_connection(connection)
    except Exception as close_error:
        if primary is None:
            raise WrdsQueryError("WRDS connection close failed") from close_error
        primary.add_note(f"WRDS connection close also failed: {close_error}")


def _partition_label(partition: RawPartition) -> str:
    when = (
        "identity"
        if partition.observation_date is None
        else partition.observation_date.isoformat()
    )
    return f"{partition.table.value}/{partition.ticker}/{when}"


@version(
    "1.0",
    dependencies=[
        _publish_raw_partition,
        build_raw_query,
        inspect_optionmetrics,
        normalize_raw_frame,
        raw_artifact_request,
        security_secid,
    ],
)
def raw_content_producer_version() -> str:
    """Return only the version closure capable of changing raw artifact content."""

    return str(raw_content_producer_version.version)


@version(
    "1.0",
    dependencies=[
        RawDownloadConfig,
        RawDownloadResult,
        WrdsConnectionLease,
        build_raw_query,
        execute_query_with_retry,
        inspect_optionmetrics,
        normalize_raw_frame,
        plan_raw_partitions,
        raw_artifact_request,
        raw_content_producer_version,
        security_secid,
        _publish_raw_partition,
    ],
)
def download_raw(
    store: ArtifactStore,
    config: RawDownloadConfig,
    *,
    dry_run: bool = False,
    cache_only: bool = False,
    refresh: bool = False,
    version_override: str | None = None,
    authorize_network: Callable[[], None] | None = None,
    progress: RawProgress | None = None,
    connection_factory: ConnectionFactory = default_connection_factory,
    environment: Mapping[str, str] | None = None,
    lease_root: Path | None = None,
    retry_sleep: Callable[[float], None] = sleep,
    heartbeat_interval_seconds: float = 30.0,
) -> RawDownloadResult:
    """Resolve complete outputs first, then perform one guarded WRDS fan-out."""

    if dry_run and cache_only:
        raise ValueError("dry_run and cache_only are mutually exclusive")
    if version_override not in {None, "*"}:
        raise ValueError("version_override must be None or '*'")
    if not 0.0 < heartbeat_interval_seconds <= 300.0:
        raise ValueError("heartbeat_interval_seconds must lie in (0, 300]")
    if refresh and config.source_generation == "default-v1":
        raise ValueError("--refresh requires an explicit non-default source generation")
    emit = (lambda payload: None) if progress is None else progress
    producer_version = raw_content_producer_version()
    requests = tuple(
        raw_artifact_request(
            partition,
            source=config.source,
            source_generation=config.source_generation,
            producer_version=producer_version,
        )
        for partition in plan_raw_partitions(config)
    )
    resolved: dict[RawArtifactRequest, ValidatedArtifact] = {}
    missing: list[RawArtifactRequest] = []
    for request in requests:
        try:
            reuse = store.resolve(request.spec, version=version_override)
        except WildcardReuseMissError:
            missing.append(request)
            continue
        if reuse is None:
            missing.append(request)
            continue
        resolved[request] = reuse.artifact
        emit(
            canonical_value(
                {
                    "event": "raw_partition_reused",
                    "partition": _partition_label(request.partition),
                    "roles": request.partition.roles,
                    "path": reuse.artifact.path,
                    "version_override": reuse.used_wildcard,
                }
            )
        )

    emit(
        canonical_value(
            {
                "event": "raw_plan",
                "ticker": config.ticker,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "planned": len(requests),
                "reused": len(resolved),
                "missing": len(missing),
                "boundary_partitions": sum(
                    request.partition.is_boundary_dependency for request in requests
                ),
                "boundary_dependencies": [
                    {
                        "table": request.partition.table.value,
                        "ticker": request.partition.ticker,
                        "observation_date": request.partition.observation_date,
                        "roles": request.partition.roles,
                    }
                    for request in requests
                    if request.partition.is_boundary_dependency
                ],
                "network_accessed": False,
                "dry_run": dry_run,
            }
        )
    )
    if version_override == "*" and missing:
        raise RawCacheMissError(
            f"version='*' cannot build {len(missing)} missing raw partition(s)"
        )
    if dry_run:
        return RawDownloadResult(
            planned=len(requests),
            reused=len(resolved),
            downloaded=0,
            missing=len(missing),
            network_accessed=False,
            dry_run=True,
            artifacts=tuple(resolved.values()),
        )
    if not missing:
        return RawDownloadResult(
            planned=len(requests),
            reused=len(resolved),
            downloaded=0,
            missing=0,
            network_accessed=False,
            dry_run=False,
            artifacts=tuple(resolved[request] for request in requests),
        )
    if cache_only:
        labels = ", ".join(_partition_label(item.partition) for item in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise RawCacheMissError(
            f"Cache-only validation found {len(missing)} missing raw partition(s): "
            f"{labels}{suffix}"
        )
    if authorize_network is None:
        raise WrdsNetworkAuthorizationError(
            "A live raw cache miss requires immediate WRDS 2FA readiness confirmation"
        )
    authorize_network()
    credentials = WrdsCredentials.from_environment(environment)

    started = monotonic()
    progress_lock = threading.Lock()
    completed = len(resolved)
    downloaded = 0
    failures = 0

    def report(request: RawArtifactRequest, artifact: ValidatedArtifact) -> None:
        nonlocal completed, downloaded
        with progress_lock:
            completed += 1
            downloaded += 1
            elapsed = monotonic() - started
            rate = downloaded / elapsed if elapsed > 0.0 else 0.0
            remaining = len(requests) - completed
            eta = remaining / rate if rate > 0.0 else None
            emit(
                canonical_value(
                    {
                        "event": "raw_partition_downloaded",
                        "partition": _partition_label(request.partition),
                        "roles": request.partition.roles,
                        "path": artifact.path,
                        "completed": completed,
                        "total": len(requests),
                        "failures": 0,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    }
                )
            )

    def report_failure(request: RawArtifactRequest, error: BaseException) -> None:
        nonlocal failures
        with progress_lock:
            failures += 1
            emit(
                canonical_value(
                    {
                        "event": "raw_partition_failed",
                        "partition": _partition_label(request.partition),
                        "error_type": type(error).__name__,
                        "completed": completed,
                        "total": len(requests),
                        "failures": failures,
                        "elapsed_seconds": monotonic() - started,
                    }
                )
            )

    def with_heartbeat[T](label: str, operation: Callable[[], T]) -> T:
        finished = threading.Event()

        def heartbeat() -> None:
            while not finished.wait(heartbeat_interval_seconds):
                with progress_lock:
                    elapsed = monotonic() - started
                    rate = downloaded / elapsed if elapsed > 0.0 else 0.0
                    remaining = len(requests) - completed
                    eta = remaining / rate if rate > 0.0 else None
                    emit(
                        canonical_value(
                            {
                                "event": "raw_progress",
                                "operation": label,
                                "completed": completed,
                                "total": len(requests),
                                "failures": failures,
                                "elapsed_seconds": elapsed,
                                "eta_seconds": eta,
                            }
                        )
                    )

        thread = threading.Thread(
            target=heartbeat,
            name="sanos-wrds-progress",
            daemon=True,
        )
        thread.start()
        try:
            return operation()
        finally:
            finished.set()
            thread.join()

    security_request = next(
        request for request in requests if request.partition.table is RawTable.SECURITY
    )
    security_missing = security_request in missing
    connection: Any | None = None
    probe_error: BaseException | None = None
    with WrdsConnectionLease(credentials.username, root=lease_root):
        try:
            connection = with_heartbeat(
                "authentication_probe",
                lambda: _open_connection(connection_factory, credentials),
            )
            catalog = with_heartbeat(
                "optionmetrics_catalog_probe",
                lambda: _inspect_optionmetrics_guarded(connection),
            )
            if security_missing:
                query = build_raw_query(security_request.partition, catalog, secid=None)
                value = with_heartbeat(
                    _partition_label(security_request.partition),
                    lambda: execute_query_with_retry(
                        lambda: connection.raw_sql(query.sql),
                        attempts=config.retry_attempts,
                        base_seconds=config.retry_base_seconds,
                        sleep=retry_sleep,
                    ),
                )
                frame = normalize_raw_frame(
                    value,
                    security_request.partition,
                    expected_secid=None,
                )
                # Reject an unusable result before publication. If another process
                # wins the immutable publication race, its artifact becomes the
                # authoritative identity for every downstream query in this run.
                security_secid(frame, config.ticker)
                artifact = _publish_raw_partition(store, security_request, query, frame)
                secid = _read_security_artifact(artifact, config.ticker)
                resolved[security_request] = artifact
                report(security_request, artifact)
            else:
                secid = _read_security_artifact(resolved[security_request], config.ticker)
        except BaseException as exc:
            probe_error = exc
            with progress_lock:
                failures += 1
                emit(
                    canonical_value(
                        {
                            "event": "raw_probe_failed",
                            "error_type": type(exc).__name__,
                            "completed": completed,
                            "total": len(requests),
                            "failures": failures,
                            "elapsed_seconds": monotonic() - started,
                        }
                    )
                )
            raise
        finally:
            if connection is not None:
                with_heartbeat(
                    "authentication_probe_close",
                    lambda: _close_without_masking(connection, probe_error),
                )

    work = [request for request in missing if request is not security_request]
    if work:
        worker_count = min(config.max_connections, len(work))
        groups: list[list[RawArtifactRequest]] = [[] for _ in range(worker_count)]
        for index, request in enumerate(work):
            groups[index % worker_count].append(request)

        def run_group(group: Sequence[RawArtifactRequest]) -> tuple[ValidatedArtifact, ...]:
            worker_connection: Any | None = None
            worker_error: BaseException | None = None
            artifacts: list[ValidatedArtifact] = []
            with WrdsConnectionLease(credentials.username, root=lease_root):
                try:
                    worker_connection = with_heartbeat(
                        "worker_connection",
                        lambda: _open_connection(connection_factory, credentials),
                    )
                    for request in group:
                        query = build_raw_query(request.partition, catalog, secid=secid)
                        try:
                            value = with_heartbeat(
                                _partition_label(request.partition),
                                lambda: execute_query_with_retry(
                                    lambda: worker_connection.raw_sql(query.sql),
                                    attempts=config.retry_attempts,
                                    base_seconds=config.retry_base_seconds,
                                    sleep=retry_sleep,
                                ),
                            )
                            frame = normalize_raw_frame(
                                value,
                                request.partition,
                                expected_secid=secid,
                            )
                            artifact = _publish_raw_partition(store, request, query, frame)
                        except BaseException as exc:
                            report_failure(request, exc)
                            raise
                        resolved[request] = artifact
                        artifacts.append(artifact)
                        report(request, artifact)
                    return tuple(artifacts)
                except BaseException as exc:
                    worker_error = exc
                    raise
                finally:
                    if worker_connection is not None:
                        with_heartbeat(
                            "worker_connection_close",
                            lambda: _close_without_masking(worker_connection, worker_error),
                        )

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="sanos-wrds") as pool:
            futures = tuple(pool.submit(run_group, group) for group in groups)
            for future in futures:
                future.result()

    artifacts = tuple(resolved[request] for request in requests)
    emit(
        canonical_value(
            {
                "event": "raw_download_complete",
                "planned": len(requests),
                "reused": len(requests) - downloaded,
                "downloaded": downloaded,
                "failures": 0,
                "elapsed_seconds": monotonic() - started,
                "network_accessed": True,
            }
        )
    )
    return RawDownloadResult(
        planned=len(requests),
        reused=len(requests) - downloaded,
        downloaded=downloaded,
        missing=0,
        network_accessed=True,
        dry_run=False,
        artifacts=artifacts,
    )


__all__ = [
    "RAW_SCHEMA_VERSION",
    "RawArtifactRequest",
    "RawCacheMissError",
    "RawDownloadError",
    "RawDownloadResult",
    "WrdsNetworkAuthorizationError",
    "download_raw",
    "normalize_raw_frame",
    "raw_artifact_request",
    "raw_content_producer_version",
    "security_secid",
]
