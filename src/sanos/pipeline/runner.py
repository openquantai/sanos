"""Deterministic bounded local execution with append-only progress journals."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic
from typing import Any, cast

from cdxcore.version import version
from filelock import FileLock

from sanos.artifacts import ArtifactPointer
from sanos.config import canonical_json, canonical_sha256, canonical_value

_WORK_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_INTERRUPT_GRACE_SECONDS = 1.0
_INTERRUPT_KILL_SECONDS = 1.0


@version("1.0")
def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@version("1.0", dependencies=[_fsync_directory])
def _ensure_journal_parent(path: Path) -> None:
    """Create and durably link journal parents without accepting symlinks."""

    parent = path.parent
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"Run journal parent is unsafe: {current}")
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)
    current = parent
    while current != current.parent:
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"Run journal parent is unsafe: {current}")
        current = current.parent


@version("1.0")
def _stop_process_pool(
    executor: ProcessPoolExecutor,
    futures: tuple[Future[ArtifactPointer | None], ...],
) -> None:
    """Bound interruption cleanup and ensure no submitted worker survives return."""

    for future in futures:
        future.cancel()
    process_map = getattr(executor, "_processes", None)
    processes = () if process_map is None else tuple(process_map.values())
    _, unfinished = wait(futures, timeout=_INTERRUPT_GRACE_SECONDS)
    if unfinished:
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = monotonic() + _INTERRUPT_KILL_SECONDS
        for process in processes:
            process.join(max(0.0, deadline - monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        deadline = monotonic() + _INTERRUPT_KILL_SECONDS
        for process in processes:
            process.join(max(0.0, deadline - monotonic()))
    executor.shutdown(wait=True, cancel_futures=True)


class DailyExecutionError(RuntimeError):
    """Raised after a daily run records one or more failed work units."""

    def __init__(self, summary: DailyRunSummary) -> None:
        self.summary = summary
        super().__init__(
            f"Daily run failed for {summary.failed} of {summary.total} work units; "
            f"journal={summary.journal_path}"
        )


@version("1.0", dependencies=[canonical_json])
@dataclass(frozen=True, slots=True)
class DailyWorkItem:
    """One deterministic per-day unit passed to a process worker."""

    key: str
    ticker: str
    trading_date: date
    stage: str
    schema_version: str
    producer_version: str
    payload_json: str

    def __post_init__(self) -> None:
        if _WORK_LABEL.fullmatch(self.ticker) is None:
            raise ValueError(f"Invalid ticker work label: {self.ticker!r}")
        if _WORK_LABEL.fullmatch(self.stage) is None:
            raise ValueError(f"Invalid stage work label: {self.stage!r}")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("Daily work schema_version cannot be empty")
        if (
            not isinstance(self.producer_version, str)
            or not self.producer_version.strip()
            or self.producer_version == "*"
        ):
            raise ValueError("Daily work producer_version must be concrete and non-empty")
        expected_key = f"{self.ticker}/{self.trading_date.isoformat()}"
        if self.key != expected_key:
            raise ValueError(f"Daily work key must be {expected_key!r}, found {self.key!r}")
        try:
            value = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Daily work payload_json is invalid") from exc
        if canonical_json(value) != self.payload_json:
            raise ValueError("Daily work payload_json must already be canonical")

    @property
    def payload(self) -> Any:
        """Decode and return a fresh work payload."""

        return json.loads(self.payload_json)

    def identity_payload(self) -> dict[str, Any]:
        """Return all output-relevant inputs for deterministic plan identity."""

        return {
            "key": self.key,
            "ticker": self.ticker,
            "trading_date": self.trading_date,
            "stage": self.stage,
            "schema_version": self.schema_version,
            "producer_version": self.producer_version,
            "payload": self.payload,
        }


@version("1.0", dependencies=[DailyWorkItem, canonical_sha256])
@dataclass(frozen=True, slots=True)
class DailyPlan:
    """Stable ordered set of daily work and bounded executor settings."""

    items: tuple[DailyWorkItem, ...]
    max_workers: int = 1
    progress_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("A daily plan must contain at least one work item")
        keys = tuple(item.key for item in self.items)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("Daily plan work keys must be unique and sorted")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        if (
            isinstance(self.progress_interval_seconds, bool)
            or not isinstance(self.progress_interval_seconds, int | float)
            or not 0.1 <= self.progress_interval_seconds <= 300.0
        ):
            raise ValueError("progress_interval_seconds must lie in [0.1, 300]")
        homogeneous = {
            (
                item.ticker,
                item.stage,
                item.schema_version,
                item.producer_version,
                item.payload_json,
            )
            for item in self.items
        }
        if len(homogeneous) != 1:
            raise ValueError(
                "A daily plan must use one ticker, stage, schema, producer, and payload"
            )

    @property
    def plan_id(self) -> str:
        """Hash ordered output-relevant work while excluding executor tuning."""

        return canonical_sha256(
            {
                "schema": "sanos-daily-plan-v1",
                "items": [item.identity_payload() for item in self.items],
            }
        )

    @property
    def payload_id(self) -> str:
        """Hash the shared canonical stage payload."""

        return canonical_sha256(self.items[0].payload)


@version("1.0", dependencies=[DailyPlan, canonical_json, canonical_sha256])
def plan_daily_dates(
    *,
    ticker: str,
    dates: Iterable[date],
    stage: str,
    payload: Any,
    schema_version: str,
    producer_version: str,
    max_workers: int = 1,
    progress_interval_seconds: float = 30.0,
) -> DailyPlan:
    """Create a sorted daily plan and reject duplicate date requests."""

    requested = tuple(dates)
    if len(requested) != len(set(requested)):
        raise ValueError("Daily plan dates contain duplicates")
    payload_json = canonical_json(payload)
    items = tuple(
        DailyWorkItem(
            key=f"{ticker}/{trading_date.isoformat()}",
            ticker=ticker,
            trading_date=trading_date,
            stage=stage,
            schema_version=schema_version,
            producer_version=producer_version,
            payload_json=payload_json,
        )
        for trading_date in sorted(requested)
    )
    return DailyPlan(
        items=items,
        max_workers=max_workers,
        progress_interval_seconds=progress_interval_seconds,
    )


@version(
    "1.0",
    dependencies=[canonical_json, canonical_value, _ensure_journal_parent, _fsync_directory],
)
class RunJournal:
    """Durable append-only JSONL journal written by the parent process."""

    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        self.path = Path(os.path.abspath(os.fspath(path.expanduser())))
        self.run_id = uuid.uuid4().hex if run_id is None else run_id
        if not self.run_id or "\n" in self.run_id:
            raise ValueError("run_id must be a non-empty single-line value")
        _ensure_journal_parent(self.path)
        if self.path.is_symlink():
            raise ValueError(f"Run journal may not be a symlink: {self.path}")
        lock_path = Path(str(self.path) + ".lock")
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise ValueError(f"Run journal lock path is unsafe: {lock_path}")
        self._lock = FileLock(str(self.path) + ".lock", timeout=30.0)
        with self._lock:
            self._sequence, recovered = self._scan_locked(repair_tail=True)
            self._known_size = self.path.stat().st_size if self.path.exists() else 0
        if recovered:
            self.emit("journal_recovered", truncated_bytes=recovered)

    def _scan_locked(self, *, repair_tail: bool) -> tuple[int, int]:
        if not self.path.exists():
            return 0, 0
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError(f"Run journal must be a regular file: {self.path}")
        last = 0
        try:
            source = self.path.read_bytes()
            complete_end = source.rfind(b"\n") + 1
            tail = source[complete_end:]
            complete = source[:complete_end]
            if tail:
                if not repair_tail:
                    raise ValueError("journal has an incomplete final record")
                descriptor = os.open(self.path, os.O_WRONLY)
                try:
                    os.ftruncate(descriptor, complete_end)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            for raw_line in complete.splitlines():
                if not raw_line:
                    raise ValueError("journal contains a blank record")
                line = raw_line.decode("utf-8")
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("sequence"), int):
                    raise ValueError("journal record has no integer sequence")
                if line != canonical_json(value):
                    raise ValueError("journal record is not canonical JSON")
                sequence = int(value["sequence"])
                if sequence != last + 1:
                    raise ValueError(
                        f"journal sequence must be contiguous; expected {last + 1}, "
                        f"found {sequence}"
                    )
                last = sequence
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Cannot resume malformed run journal {self.path}: {exc}") from exc
        return last, len(tail)

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return a validated immutable snapshot of complete journal records."""

        with self._lock:
            self._scan_locked(repair_tail=False)
            if not self.path.exists():
                return ()
            return tuple(
                cast(dict[str, Any], json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines()
            )

    def require_plan(self, plan_id: str) -> None:
        """Reject reuse of a journal path for a different deterministic plan."""

        prior = {
            record.get("plan_id")
            for record in self.records()
            if record.get("event") == "run_started"
        }
        if prior and prior != {plan_id}:
            raise ValueError(
                f"Run journal belongs to plan(s) {sorted(str(value) for value in prior)}, "
                f"not {plan_id}"
            )

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append and fsync one canonical event record."""

        if _WORK_LABEL.fullmatch(event) is None:
            raise ValueError(f"Invalid journal event label: {event!r}")
        reserved = {"sequence", "timestamp", "run_id", "event"}.intersection(fields)
        if reserved:
            raise ValueError(f"Journal fields may not replace reserved keys: {sorted(reserved)}")
        prepared_fields = cast(dict[str, Any], canonical_value(fields))
        with self._lock:
            journal_existed = self.path.exists()
            current_size = self.path.stat().st_size if journal_existed else 0
            if current_size != self._known_size:
                self._sequence, recovered = self._scan_locked(repair_tail=True)
                self._known_size = self.path.stat().st_size if self.path.exists() else 0
                if recovered:
                    prepared_fields["journal_recovered_tail_bytes"] = recovered
            next_sequence = self._sequence + 1
            record = cast(
                dict[str, Any],
                canonical_value(
                    {
                        "sequence": next_sequence,
                        "timestamp": datetime.now(UTC),
                        "run_id": self.run_id,
                        "event": event,
                        **prepared_fields,
                    }
                ),
            )
            encoded = (canonical_json(record) + "\n").encode("utf-8")
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written == 0:
                        raise OSError("A run journal append wrote zero bytes")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if not journal_existed:
                _fsync_directory(self.path.parent)
            self._sequence = next_sequence
            self._known_size += len(encoded)
        return record


@version("1.0")
@dataclass(frozen=True, slots=True)
class DailyFailure:
    """Small serializable failure record; no arrays or tracebacks are retained."""

    key: str
    error_type: str
    message: str


@version("1.0", dependencies=[ArtifactPointer, DailyFailure])
@dataclass(frozen=True, slots=True)
class DailyRunSummary:
    """Completed local-run result and journal locator."""

    run_id: str
    plan_id: str
    journal_path: Path
    total: int
    completed: int
    reused: int
    failed: int
    elapsed_seconds: float
    results: tuple[tuple[str, ArtifactPointer | None], ...]
    failures: tuple[DailyFailure, ...]

    @property
    def success(self) -> bool:
        """Whether every requested unit either completed or was reused."""

        return self.failed == 0 and self.completed + self.reused == self.total


DailyWorker = Callable[[DailyWorkItem], ArtifactPointer | None]
CompletionProbe = Callable[[DailyWorkItem], ArtifactPointer | None]
ProgressCallback = Callable[[Mapping[str, Any]], None]


@version("1.0")
def _run_worker(worker: DailyWorker, item: DailyWorkItem) -> ArtifactPointer | None:
    result = worker(item)
    if result is not None and not isinstance(result, ArtifactPointer):
        raise TypeError(
            f"A daily worker must return an ArtifactPointer or None, found {type(result).__name__}"
        )
    return result


@version("1.0")
def _progress_record(
    *,
    elapsed_seconds: float,
    completed: int,
    reused: int,
    failed: int,
    total: int,
) -> dict[str, Any]:
    finished = completed + reused + failed
    rate = finished / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    eta = (total - finished) / rate if rate > 0.0 else None
    return {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "completed": finished,
        "total": total,
        "failures": failed,
        "reused": reused,
        "eta_seconds": None if eta is None else round(eta, 6),
    }


@version(
    "1.0",
    dependencies=[
        DailyPlan,
        RunJournal,
        DailyRunSummary,
        _run_worker,
        _progress_record,
        _stop_process_pool,
    ],
)
def run_daily_plan(
    plan: DailyPlan,
    worker: DailyWorker,
    *,
    journal_path: Path,
    is_complete: CompletionProbe | None = None,
    progress: ProgressCallback | None = None,
) -> DailyRunSummary:
    """Run pending daily work in bounded processes and journal all state changes."""

    journal = RunJournal(journal_path)
    journal.require_plan(plan.plan_id)
    started = monotonic()
    journal.emit(
        "run_started",
        plan_id=plan.plan_id,
        payload_id=plan.payload_id,
        stage=plan.items[0].stage,
        schema_version=plan.items[0].schema_version,
        producer_version=plan.items[0].producer_version,
        total=len(plan.items),
        max_workers=plan.max_workers,
        progress_interval_seconds=plan.progress_interval_seconds,
    )
    pending_items: list[DailyWorkItem] = []
    reused = 0
    completed = 0
    failures: list[DailyFailure] = []
    results: list[tuple[str, ArtifactPointer | None]] = []
    total = len(plan.items)
    executor: ProcessPoolExecutor | None = None
    future_items: dict[Future[ArtifactPointer | None], DailyWorkItem] = {}

    def emit_progress() -> None:
        record = _progress_record(
            elapsed_seconds=monotonic() - started,
            completed=completed,
            reused=reused,
            failed=len(failures),
            total=total,
        )
        journal.emit("progress", **record)
        if progress is not None:
            progress(record)

    try:
        for item in plan.items:
            reused_artifact = None if is_complete is None else is_complete(item)
            if reused_artifact is not None:
                if not isinstance(reused_artifact, ArtifactPointer):
                    raise TypeError("A completion probe must return an ArtifactPointer or None")
                reused += 1
                results.append((item.key, reused_artifact))
                journal.emit(
                    "unit_reused",
                    key=item.key,
                    artifact=reused_artifact.as_dict(),
                )
            else:
                pending_items.append(item)

        if pending_items:
            workers = min(plan.max_workers, len(pending_items))
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            for item in pending_items:
                journal.emit("unit_started", key=item.key)
                future_items[executor.submit(_run_worker, worker, item)] = item
            remaining = set(future_items)
            last_progress = monotonic()
            while remaining:
                timeout = max(
                    0.0,
                    plan.progress_interval_seconds - (monotonic() - last_progress),
                )
                done, remaining = wait(
                    remaining,
                    timeout=timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    emit_progress()
                    last_progress = monotonic()
                    continue
                for future in done:
                    item = future_items[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - worker boundary
                        failure = DailyFailure(
                            key=item.key,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                        failures.append(failure)
                        journal.emit(
                            "unit_failed",
                            key=failure.key,
                            error_type=failure.error_type,
                            message=failure.message,
                        )
                    else:
                        completed += 1
                        results.append((item.key, result))
                        journal.emit(
                            "unit_completed",
                            key=item.key,
                            artifact=None if result is None else result.as_dict(),
                        )
                emit_progress()
                last_progress = monotonic()
            executor.shutdown(wait=True)
            executor = None
        else:
            emit_progress()
    except BaseException as exc:
        try:
            journal.emit(
                "run_interrupted",
                plan_id=plan.plan_id,
                error_type=type(exc).__name__,
                message=str(exc),
                **_progress_record(
                    elapsed_seconds=monotonic() - started,
                    completed=completed,
                    reused=reused,
                    failed=len(failures),
                    total=total,
                ),
            )
        except BaseException as journal_error:
            exc.add_note(f"Could not journal interruption: {journal_error}")
        if executor is not None:
            _stop_process_pool(executor, tuple(future_items))
        raise

    elapsed = monotonic() - started
    summary = DailyRunSummary(
        run_id=journal.run_id,
        plan_id=plan.plan_id,
        journal_path=journal.path,
        total=total,
        completed=completed,
        reused=reused,
        failed=len(failures),
        elapsed_seconds=elapsed,
        results=tuple(sorted(results)),
        failures=tuple(sorted(failures, key=lambda failure: failure.key)),
    )
    journal.emit(
        "run_completed",
        plan_id=plan.plan_id,
        status="ok" if summary.success else "error",
        **_progress_record(
            elapsed_seconds=elapsed,
            completed=completed,
            reused=reused,
            failed=len(failures),
            total=total,
        ),
    )
    if failures:
        raise DailyExecutionError(summary)
    return summary
