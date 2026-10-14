from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import pytest

from sanos.artifacts import ArtifactPointer, StorageClass
from sanos.pipeline import (
    DailyExecutionError,
    DailyWorkItem,
    RunJournal,
    plan_daily_dates,
    run_daily_plan,
)


def _pointer(item: DailyWorkItem) -> ArtifactPointer:
    artifact_id = hashlib.sha256(item.key.encode("utf-8")).hexdigest()
    return ArtifactPointer(StorageClass.INTERMEDIATE, "daily/fit", artifact_id)


def _identity_worker(item: DailyWorkItem) -> ArtifactPointer:
    return _pointer(item)


def _selective_failure_worker(item: DailyWorkItem) -> ArtifactPointer:
    if item.trading_date.day == 3:
        raise RuntimeError("expected worker failure")
    return _pointer(item)


def _invalid_worker(item: DailyWorkItem) -> object:
    return {"large": item.key}


def _slow_worker(item: DailyWorkItem) -> ArtifactPointer:
    time.sleep(0.5)
    return _pointer(item)


def _post_interrupt_side_effect_worker(item: DailyWorkItem) -> ArtifactPointer:
    payload = item.payload
    Path(payload["started_path"]).write_text("started", encoding="utf-8")
    time.sleep(2.0)
    Path(payload["finished_path"]).write_text("finished", encoding="utf-8")
    return _pointer(item)


def _journal_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_plan_is_sorted_canonical_and_rejects_duplicates() -> None:
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 3), date(2024, 1, 2)],
        stage="fit",
        payload={"b": 2, "a": 1},
        schema_version="daily-1",
        producer_version="fit-1",
        max_workers=2,
    )
    assert [item.key for item in plan.items] == ["SPX/2024-01-02", "SPX/2024-01-03"]
    assert plan.items[0].payload_json == '{"a":1,"b":2}'

    with pytest.raises(ValueError, match="duplicates"):
        plan_daily_dates(
            ticker="SPX",
            dates=[date(2024, 1, 2), date(2024, 1, 2)],
            stage="fit",
            payload={},
            schema_version="daily-1",
            producer_version="fit-1",
        )


def test_bounded_process_run_journals_progress_and_results(tmp_path: Path) -> None:
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        stage="fit",
        payload={},
        schema_version="daily-1",
        producer_version="fit-1",
        max_workers=2,
        progress_interval_seconds=0.1,
    )
    progress: list[dict[str, object]] = []
    journal = tmp_path / "run.jsonl"
    summary = run_daily_plan(plan, _identity_worker, journal_path=journal, progress=progress.append)

    assert summary.success
    assert (summary.completed, summary.reused, summary.failed) == (3, 0, 0)
    assert len(summary.results) == 3
    records = _journal_records(journal)
    assert records[0]["event"] == "run_started"
    assert records[0]["plan_id"] == plan.plan_id
    assert records[-1]["event"] == "run_completed"
    assert records[-1]["completed"] == 3
    assert progress


def test_completion_probe_reuses_units_without_starting_workers(tmp_path: Path) -> None:
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2), date(2024, 1, 3)],
        stage="fit",
        payload={},
        schema_version="daily-1",
        producer_version="fit-1",
    )
    summary = run_daily_plan(
        plan,
        _identity_worker,
        journal_path=tmp_path / "reuse.jsonl",
        is_complete=lambda item: _pointer(item) if item.trading_date.day == 2 else None,
    )
    assert (summary.completed, summary.reused) == (1, 1)
    events = [record["event"] for record in _journal_records(summary.journal_path)]
    assert events.count("unit_reused") == 1
    assert events.count("unit_started") == 1


def test_failures_are_small_durable_and_actionable(tmp_path: Path) -> None:
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2), date(2024, 1, 3)],
        stage="fit",
        payload={},
        schema_version="daily-1",
        producer_version="fit-1",
        max_workers=2,
    )
    with pytest.raises(DailyExecutionError) as caught:
        run_daily_plan(plan, _selective_failure_worker, journal_path=tmp_path / "failure.jsonl")

    summary = caught.value.summary
    assert (summary.completed, summary.failed) == (1, 1)
    assert summary.failures[0].key == "SPX/2024-01-03"
    assert summary.failures[0].error_type == "RuntimeError"
    assert "expected worker failure" in summary.failures[0].message
    assert _journal_records(summary.journal_path)[-1]["status"] == "error"


def test_workers_may_not_return_large_objects(tmp_path: Path) -> None:
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2)],
        stage="fit",
        payload={},
        schema_version="daily-1",
        producer_version="fit-1",
    )
    with pytest.raises(DailyExecutionError) as caught:
        run_daily_plan(plan, _invalid_worker, journal_path=tmp_path / "invalid.jsonl")
    assert caught.value.summary.failures[0].error_type == "TypeError"


def test_journal_sequence_is_safe_across_instances_and_malformed_input_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    first = RunJournal(path, run_id="first")
    second = RunJournal(path, run_id="second")
    assert first.emit("started")["sequence"] == 1
    assert second.emit("started")["sequence"] == 2
    with pytest.raises(ValueError, match="reserved"):
        first.emit("started", sequence=99)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        RunJournal(malformed)

    non_contiguous = tmp_path / "non-contiguous.jsonl"
    non_contiguous.write_text('{"sequence":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        RunJournal(non_contiguous)


def test_runner_foundation_is_versioned() -> None:
    assert str(plan_daily_dates.version).startswith("1.0")
    assert str(run_daily_plan.version).startswith("1.0")


def test_plan_identity_changes_with_payload_schema_and_producer() -> None:
    arguments = {
        "ticker": "SPX",
        "dates": [date(2024, 1, 2)],
        "stage": "fit",
        "schema_version": "daily-1",
        "producer_version": "fit-1",
    }
    first = plan_daily_dates(payload={"smoothness": 0.25}, **arguments)
    changed_payload = plan_daily_dates(payload={"smoothness": 0.5}, **arguments)
    changed_schema = plan_daily_dates(
        payload={"smoothness": 0.25}, **{**arguments, "schema_version": "daily-2"}
    )
    changed_producer = plan_daily_dates(
        payload={"smoothness": 0.25}, **{**arguments, "producer_version": "fit-2"}
    )
    assert (
        len(
            {
                first.plan_id,
                changed_payload.plan_id,
                changed_schema.plan_id,
                changed_producer.plan_id,
            }
        )
        == 4
    )

    with pytest.raises(ValueError, match="positive integer"):
        plan_daily_dates(payload={}, max_workers=1.5, **arguments)


def test_changed_plan_cannot_append_to_existing_journal(tmp_path: Path) -> None:
    first = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2)],
        stage="fit",
        payload={"smoothness": 0.25},
        schema_version="daily-1",
        producer_version="fit-1",
    )
    changed = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2)],
        stage="fit",
        payload={"smoothness": 0.5},
        schema_version="daily-1",
        producer_version="fit-1",
    )
    journal = tmp_path / "plan.jsonl"
    run_daily_plan(first, _identity_worker, journal_path=journal)
    with pytest.raises(ValueError, match="belongs to plan"):
        run_daily_plan(changed, _identity_worker, journal_path=journal)


def test_journal_canonicalization_failure_does_not_consume_sequence(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path / "canonical.jsonl")
    with pytest.raises(TypeError):
        journal.emit("bad", unsupported={1, 2})
    assert journal.emit("started")["sequence"] == 1
    assert RunJournal(journal.path).emit("continued")["sequence"] == 2


def test_journal_repairs_only_an_incomplete_final_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sanos.pipeline.runner as runner_module

    path = tmp_path / "partial.jsonl"
    journal = RunJournal(path)
    original_write = runner_module.os.write
    calls = 0

    def partial_then_fail(descriptor: int, data: memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, bytes(data)[:5])
        raise OSError("injected partial append")

    with monkeypatch.context() as patcher:
        patcher.setattr(runner_module.os, "write", partial_then_fail)
        with pytest.raises(OSError, match="partial"):
            journal.emit("started")

    recovered = RunJournal(path)
    records = recovered.records()
    assert records[0]["event"] == "journal_recovered"
    assert records[0]["sequence"] == 1
    assert records[0]["truncated_bytes"] == 5

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"sequence":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        RunJournal(corrupt)


def test_probe_interrupt_is_journaled_before_propagation(tmp_path: Path) -> None:
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2)],
        stage="fit",
        payload={},
        schema_version="daily-1",
        producer_version="fit-1",
    )
    journal = tmp_path / "interrupt.jsonl"

    def interrupt_probe(item: DailyWorkItem) -> ArtifactPointer | None:
        raise KeyboardInterrupt(item.key)

    with pytest.raises(KeyboardInterrupt):
        run_daily_plan(
            plan,
            _identity_worker,
            journal_path=journal,
            is_complete=interrupt_probe,
        )
    assert _journal_records(journal)[-1]["event"] == "run_interrupted"


def test_new_nested_journal_fsyncs_its_containing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sanos.pipeline.runner as runner_module

    calls: list[Path] = []
    original = runner_module._journal_fsync_directory

    def record(path: Path) -> None:
        calls.append(path)
        original(path)

    monkeypatch.setattr(runner_module, "_journal_fsync_directory", record)
    path = tmp_path / "nested" / "journal" / "run.jsonl"
    RunJournal(path).emit("started")
    assert path.parent in calls


def test_active_pool_interrupt_is_journaled_and_plan_can_restart(tmp_path: Path) -> None:
    started = tmp_path / "worker-started"
    finished = tmp_path / "worker-finished"
    plan = plan_daily_dates(
        ticker="SPX",
        dates=[date(2024, 1, 2)],
        stage="fit",
        payload={"started_path": str(started), "finished_path": str(finished)},
        schema_version="daily-1",
        producer_version="fit-1",
        progress_interval_seconds=0.1,
    )
    journal = tmp_path / "active-interrupt.jsonl"

    def interrupt_progress(record: object) -> None:
        if started.exists():
            raise KeyboardInterrupt(record)

    with pytest.raises(KeyboardInterrupt):
        run_daily_plan(
            plan,
            _post_interrupt_side_effect_worker,
            journal_path=journal,
            progress=interrupt_progress,
    )
    assert _journal_records(journal)[-1]["event"] == "run_interrupted"
    assert started.is_file()
    assert not finished.exists()
    time.sleep(1.2)
    assert not finished.exists()

    restarted = run_daily_plan(
        plan,
        _identity_worker,
        journal_path=journal,
    )
    assert restarted.success
    assert restarted.completed == 1
