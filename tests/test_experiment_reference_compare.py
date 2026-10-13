from __future__ import annotations

import hashlib
import importlib
import io
import json
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from math import ceil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import sanos.reference.experiment_compare as compare_module
from sanos.artifacts import ArtifactPointer, ValidatedArtifact
from sanos.cli import main
from sanos.experiments import ExperimentConfig
from sanos.experiments.daily_front import DailyFrontDay, EligibilityAudit
from sanos.reference import (
    ExperimentComparisonRequest,
    ExperimentReferenceTolerances,
    load_reference_catalog,
)
from sanos.reference._experiment_oracle_worker import (
    PINNED_SOURCE_SHA256,
    _compare_array,
    _compare_logical_day,
    _load_pinned_consumer,
)
from sanos.reference.compare import (
    PHASE11_PRODUCTION_ORACLE_GAPS,
    PINNED_CATALOG_SHA256,
)
from sanos.reference.experiment_compare import (
    PHASE11_DATASET_ARTIFACT_ID,
    PHASE11_DATASET_NAMESPACE,
    PHASE11_EXPERIMENT_ARTIFACT_ID,
)


class _FakeOracleProcess:
    def __init__(self, *, timeout_on_terminate: bool) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO('{"kind":"progress","completed":1}\n')
        self.timeout_on_terminate = timeout_on_terminate
        self.terminated = False
        self.killed = False
        self.reaped = False

    def poll(self) -> int | None:
        return 0 if self.reaped else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        if self.timeout_on_terminate and self.terminated and not self.killed and timeout:
            raise subprocess.TimeoutExpired("fake-oracle", timeout)
        self.reaped = True
        return 0


@pytest.mark.parametrize(
    ("failure", "timeout_on_terminate"),
    [(RuntimeError("callback failed"), False), (KeyboardInterrupt(), True)],
)
def test_oracle_worker_is_stopped_and_reaped_on_parent_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    timeout_on_terminate: bool,
) -> None:
    process = _FakeOracleProcess(timeout_on_terminate=timeout_on_terminate)
    monkeypatch.setattr(
        compare_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def fail_progress(_value: object) -> None:
        raise failure

    with pytest.raises(type(failure), match=str(failure) or None):
        compare_module._run_oracle_worker(
            {"request": "fixture"},
            legacy_code_root=tmp_path,
            progress=fail_progress,
        )
    assert process.terminated is True
    assert process.killed is timeout_on_terminate
    assert process.reaped is True
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_oracle_consumer_pins_and_rejects_changed_transitive_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "quantdata/dysanos/flex/metrics.py"
    assert relative in PINNED_SOURCE_SHA256
    assert "quantdata/dysanos/flex/model.py" in PINNED_SOURCE_SHA256
    assert "quantdata/torch/ffw.py" in PINNED_SOURCE_SHA256
    changed = tmp_path / relative
    changed.parent.mkdir(parents=True)
    changed.write_text("# changed transitive source\n", encoding="utf-8")
    monkeypatch.setattr(
        "sanos.reference._experiment_oracle_worker.PINNED_SOURCE_SHA256",
        {relative: PINNED_SOURCE_SHA256[relative]},
    )
    with pytest.raises(RuntimeError, match="source changed"):
        _load_pinned_consumer(tmp_path)


@dataclass(frozen=True)
class _LegacyAudit:
    active_european_books: int
    horizon_eligible_books: int
    nms_eligible_books: int
    input_expiries: int
    eligible_expiries: int


@dataclass(frozen=True)
class _LegacyDay:
    ticker: str
    trading_date: date
    expiries_y: np.ndarray
    expiry_dates: tuple[str, ...]
    expiry_ids: np.ndarray
    expiry_is_am: np.ndarray
    expiry_index: np.ndarray
    strikes: np.ndarray
    is_call: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    fit_prices: np.ndarray
    log_volumes: np.ndarray
    log_open_interests: np.ndarray
    quote_ids: tuple[str, ...]
    atm_annual_vols: np.ndarray
    model_atm_total_vols: np.ndarray
    strike_lower_bounds: np.ndarray
    strike_upper_bounds: np.ndarray
    market_uid: str
    audit: _LegacyAudit
    spot_ref: float

    @property
    def num_quotes(self) -> int:
        return len(self.strikes)

    @property
    def training_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.ticker,
            self.trading_date.isoformat(),
            self.market_uid,
            *self.expiry_dates,
            *self.quote_ids,
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        for array in (
            self.expiries_y,
            self.expiry_ids,
            self.expiry_is_am,
            self.expiry_index,
            self.strikes,
            self.is_call,
            self.bids,
            self.asks,
            self.fit_prices,
            self.atm_annual_vols,
            self.model_atm_total_vols,
            self.strike_lower_bounds,
            self.strike_upper_bounds,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


def _dual_day() -> tuple[DailyFrontDay, _LegacyDay]:
    trading_date = date(2024, 1, 3)
    expiry_date = date(2024, 1, 4)
    expiry_days = int(
        (np.datetime64(expiry_date, "D") - np.datetime64("1970-01-01", "D")).astype(
            np.int64
        )
    )
    audit = EligibilityAudit(2, 2, 2, 1, 1)
    fresh = DailyFrontDay(
        ticker="SPX",
        trading_date=trading_date,
        is_half_day=False,
        spot_reference=4_800.0,
        market_uid="a" * 64,
        market_fingerprint="b" * 64,
        expiry_time_y=np.asarray([1.0 / 252.0], dtype=np.float64),
        expiry_date_days=np.asarray([expiry_days], dtype=np.int32),
        expiry_id=np.asarray([123], dtype=np.int64),
        expiry_is_am=np.asarray([False], dtype=np.bool_),
        expiry_atm_annual_vol=np.asarray([0.2], dtype=np.float64),
        expiry_strike_lower=np.asarray([0.8], dtype=np.float64),
        expiry_strike_upper=np.asarray([1.2], dtype=np.float64),
        model_atm_total_vol=np.asarray([0.2 / np.sqrt(252.0)], dtype=np.float64),
        quote_expiry_index=np.asarray([0, 0], dtype=np.int64),
        quote_price_index=np.asarray([0, 0], dtype=np.int64),
        quote_strike=np.asarray([1.0, 1.0], dtype=np.float64),
        quote_is_call=np.asarray([True, False], dtype=np.bool_),
        quote_bid=np.asarray([0.01, 0.01], dtype=np.float64),
        quote_ask=np.asarray([0.02, 0.02], dtype=np.float64),
        quote_fit=np.asarray([0.015, 0.015], dtype=np.float64),
        quote_log_volume=np.asarray([1.0, 1.0], dtype=np.float64),
        quote_log_open_interest=np.asarray([2.0, 2.0], dtype=np.float64),
        price_expiry_y=np.asarray([1.0 / 252.0], dtype=np.float64),
        price_strike=np.asarray([1.0], dtype=np.float64),
        audit=audit,
    )
    legacy = _LegacyDay(
        ticker=fresh.ticker,
        trading_date=fresh.trading_date,
        expiries_y=fresh.expiry_time_y.copy(),
        expiry_dates=fresh.expiry_dates,
        expiry_ids=fresh.expiry_id.copy(),
        expiry_is_am=fresh.expiry_is_am.copy(),
        expiry_index=fresh.quote_expiry_index.copy(),
        strikes=fresh.quote_strike.copy(),
        is_call=fresh.quote_is_call.copy(),
        bids=fresh.quote_bid.copy(),
        asks=fresh.quote_ask.copy(),
        fit_prices=fresh.quote_fit.copy(),
        log_volumes=fresh.quote_log_volume.copy(),
        log_open_interests=fresh.quote_log_open_interest.copy(),
        quote_ids=fresh.quote_ids,
        atm_annual_vols=fresh.expiry_atm_annual_vol.copy(),
        model_atm_total_vols=fresh.model_atm_total_vol.copy(),
        strike_lower_bounds=fresh.expiry_strike_lower.copy(),
        strike_upper_bounds=fresh.expiry_strike_upper.copy(),
        market_uid="legacy-uid",
        audit=_LegacyAudit(2, 2, 2, 1, 1),
        spot_ref=fresh.spot_reference,
    )
    return fresh, legacy


def test_logical_experiment_oracle_rebases_only_source_identity() -> None:
    fresh, legacy = _dual_day()
    comparison = _compare_logical_day(fresh, legacy, float_atol=1.0e-12)
    assert comparison["matched"] is True
    assert comparison["mismatch_fields"] == []
    assert comparison["identity"]["legacy_market_uid"] == "legacy-uid"
    assert comparison["identity"]["fresh_market_uid"] == "a" * 64
    assert comparison["identity"]["rebased_training_fingerprint_matched"] is True

    changed = replace(legacy, bids=legacy.bids + np.asarray([0.0, 1.0e-6]))
    mismatch = _compare_logical_day(fresh, changed, float_atol=1.0e-12)
    assert mismatch["matched"] is False
    assert "quote/bid" in mismatch["mismatch_fields"]


def test_experiment_array_comparison_distinguishes_nonfinite_values() -> None:
    matched = _compare_array(
        "test",
        np.asarray([np.nan, np.inf, -np.inf]),
        np.asarray([np.nan, np.inf, -np.inf]),
        float_atol=0.0,
    )
    assert matched["matched"] is True
    mismatched = _compare_array(
        "test",
        np.asarray([np.nan, np.inf, -np.inf]),
        np.asarray([np.inf, -np.inf, np.nan]),
        float_atol=0.0,
    )
    assert mismatched["matched"] is False
    assert mismatched["mismatch_count"] == 3


def _phase11_worker_evidence(catalog: dict[str, object]) -> dict[str, object]:
    dates = [
        str(record["trading_date"])
        for record in catalog["files"]
        if record["populated"] is True
    ]
    quotes = [7_898, *([3_999] * 140), *([3_998] * 750)]
    assert len(quotes) == 891 and sum(quotes) == 3_566_258
    days = []
    for index, (trading_date, count) in enumerate(zip(dates, quotes, strict=True)):
        calls = count // 2 + (1 if 1 <= index <= 70 else 0)
        days.append(
            {
                "date": trading_date,
                "classification": "match",
                "logical": {
                    "population": {
                        "quotes": count,
                        "calls": calls,
                        "puts": count - calls,
                        "prices": 3_949 if index == 0 else min(count, 2_000),
                        "expiries": 36 if index == 0 else 20,
                    }
                },
            }
        )
    assert sum(day["logical"]["population"]["calls"] for day in days) == 1_783_129
    batches = [
        {"batch_index": index, "matched": True}
        for index in range(ceil(891 / 32))
    ]
    return {
        "source_hashes": PINNED_SOURCE_SHA256,
        "runtime": {"python": "3.12.test", "numpy": "2.test", "torch": "2.test"},
        "dates": days,
        "consumer_batches": batches,
        "summary": {
            "fresh_dates": 917,
            "common_dates": 891,
            "fresh_only_dates": list(PHASE11_PRODUCTION_ORACLE_GAPS),
            "mismatch_dates": [],
            "legacy_books": 3_566_258,
            "legacy_calls": 1_783_129,
            "legacy_puts": 1_783_129,
            "legacy_qmax": 7_898,
            "legacy_pmax": 3_949,
            "legacy_emax": 36,
            "legacy_pre_2025_dates": 729,
            "legacy_from_2025_dates": 162,
            "legacy_weekly_pairs": 861,
            "consumer_batches": len(batches),
            "consumer_batch_mismatches": 0,
        },
    }


def test_phase11_experiment_contract_is_exact_and_tamper_evident(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_document(
        json.loads(
            (root / "configs" / "dysanos-daily-front-reference.json").read_text(
                encoding="utf-8"
            )
        )
    )
    pointer = ArtifactPointer(
        "experiment",
        f"dysanos/daily-front-market/name-{config.name}",
        PHASE11_EXPERIMENT_ARTIFACT_ID,
    )
    artifact = ValidatedArtifact(pointer=pointer, path=tmp_path, manifest={})
    request = ExperimentComparisonRequest(
        experiment=pointer,
        enforce_phase11_production=True,
    )
    catalog = load_reference_catalog(
        root / "reviews" / "phase-1" / "reference-market-catalog.json"
    )
    worker = _phase11_worker_evidence(catalog)
    validation = {
        "dates": {"first": "2022-01-03", "last": "2025-08-29", "count": 917},
        "books": 3_566_259,
    }
    contract = compare_module._phase11_contract(
        request,
        artifact=artifact,
        config=config,
        validation=validation,
        worker=worker,
        catalog_sha256=PINNED_CATALOG_SHA256,
        catalog=catalog,
    )
    assert contract["passed"] is True
    assert config.source_dataset.pointer == ArtifactPointer(
        "dataset", PHASE11_DATASET_NAMESPACE, PHASE11_DATASET_ARTIFACT_ID
    )

    changed = json.loads(json.dumps(worker))
    changed["summary"]["legacy_qmax"] = 7_897
    blocked = compare_module._phase11_contract(
        request,
        artifact=artifact,
        config=config,
        validation=validation,
        worker=changed,
        catalog_sha256=PINNED_CATALOG_SHA256,
        catalog=catalog,
    )
    assert blocked["passed"] is False
    assert any("legacy_qmax" in failure for failure in blocked["failures"])


def test_experiment_reference_request_and_tolerances_are_frozen() -> None:
    pointer = ArtifactPointer(
        "experiment",
        "dysanos/daily-front-market/name-test",
        "a" * 64,
    )
    assert ExperimentComparisonRequest(experiment=pointer).consumer_batch_size == 32
    with pytest.raises(ValueError, match="frozen"):
        ExperimentReferenceTolerances(float64_atol=1.0e-8)
    with pytest.raises(ValueError, match="Experiment pointer"):
        ExperimentComparisonRequest(
            experiment=ArtifactPointer("dataset", "widb/internal/name-test", "b" * 64)
        )


def test_experiment_reference_cli_is_structured_and_dry_run_is_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    legacy = tmp_path / "legacy"
    code = tmp_path / "code"
    legacy.mkdir()
    code.mkdir()
    output = tmp_path / "evidence" / "experiment-report.json"
    cli_module = importlib.import_module("sanos.cli.main")
    monkeypatch.setattr(cli_module, "repository_source_revision", lambda: "c" * 40)
    monkeypatch.setattr(
        cli_module,
        "compare_experiment_artifact",
        lambda *args, **kwargs: {
            "schema": "sanos-legacy-experiment-comparison-v1",
            "outcome": "passed",
            "report_id": "d" * 64,
            "comparison": {
                "summary": {
                    "common_dates": 891,
                    "fresh_only_dates": list(PHASE11_PRODUCTION_ORACLE_GAPS),
                    "mismatch_dates": [],
                }
            },
        },
    )

    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--dry-run",
            "--log-format",
            "json",
            "reference",
            "compare-experiment",
            "--config",
            str(root / "configs" / "dysanos-daily-front-reference.json"),
            "--legacy-root",
            str(legacy),
            "--legacy-code-root",
            str(code),
            "--phase11-production-contract",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "experiment_reference_comparison_result"
    assert payload["status"] == "passed"
    assert payload["common_dates"] == 891
    assert payload["network_accessed"] is False
    assert payload["artifacts_built"] is False
    assert payload["filesystem_modified"] is False
    assert not output.exists()
