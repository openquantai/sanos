from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from threading import Barrier
from typing import Any

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs

import sanos.artifacts.store as artifact_store_module
import sanos.datasets.internal as internal_dataset_module
from sanos.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    WildcardReuseMissError,
)
from sanos.config import StorageLocations, canonical_json, canonical_sha256, canonical_value
from sanos.datasets import (
    DatasetBuildOptions,
    DatasetValidationError,
    InternalDatasetConfig,
    build_internal_dataset,
    open_internal_dataset,
    plan_internal_dataset,
    validate_internal_dataset,
)
from sanos.datasets.internal import internal_dataset_producer_version
from sanos.pipeline import DailyExecutionError
from sanos.widb import (
    ENRICHED_SCHEMA,
    FILTERED_SCHEMA,
    FITTED_SCHEMA,
    PAIRED_SCHEMA,
    ClockConfig,
    DailySurfaceConfig,
    DailyWidbConfig,
    OptionsFilterConfig,
    SmoothSanosConfig,
    business_dates,
    count_business_days_between,
    enrich_daily_frame,
    previous_business_day,
)
from sanos.widb.build import (
    IntermediateBuildRequest,
    build_daily_target,
    build_intermediates,
    enriched_day_producer_version,
    enriched_day_spec,
    filtered_day_producer_version,
    filtered_day_spec,
    fitted_day_producer_version,
    fitted_day_spec,
    no_arb_day_producer_version,
    no_arb_day_spec,
    paired_day_producer_version,
    paired_day_spec,
    plan_intermediate_build,
    plan_raw_day_pointers,
    probe_daily_target,
    publish_daily_frame,
    read_daily_frame,
)
from sanos.wrds import RawPartition, RawTable, raw_artifact_request, raw_content_producer_version


def _locations(root: Path) -> StorageLocations:
    return StorageLocations(
        raw=root / "raw",
        intermediate=root / "intermediate",
        datasets=root / "datasets",
        experiments=root / "experiments",
    )


def _raw_frames(
    trading_date: date,
    expiry: date,
    *,
    ticker: str = "SPX",
    force_enrichment_rejection: bool = False,
) -> dict[RawTable | str, pl.DataFrame]:
    strikes = np.arange(90.0, 117.0, 2.0)
    spot = 100.0
    forward = 101.0
    calendar_dte = (expiry - trading_date).days
    discount = float(np.exp(-0.05 * calendar_dte / 365.0))
    time_years = count_business_days_between(trading_date, expiry) / 252.0
    calls = (
        np.asarray(bs(k=strikes / forward, vol=0.4, sqrtT=np.sqrt(time_years))) * discount * forward
    )
    puts = calls - discount * (forward - strikes)
    rows: list[dict[str, object]] = []
    for index, (strike, call, put) in enumerate(zip(strikes, calls, puts, strict=True)):
        for side, price in (("C", call), ("P", put)):
            rows.append(
                {
                    "secid": 108105,
                    "date": trading_date.isoformat(),
                    "symbol": f"SPX-{expiry}-{side}-{index}",
                    "optionid": index * 2 + int(side == "P"),
                    "am_settlement": 0,
                    "cp_flag": side,
                    "strike_price": int(strike * 1000),
                    "exdate": expiry.isoformat(),
                    "best_bid": float(
                        0.02
                        if side == "P"
                        and (
                            index == len(strikes) - 1
                            or (force_enrichment_rejection and index == 9)
                        )
                        else price - 0.05
                    ),
                    "best_offer": float(price + 0.05),
                    "volume": 10,
                    "open_interest": 100,
                    "impl_volatility": 0.4,
                    "ss_flag": "0",
                    "delta": 0.5,
                    "gamma": 0.01,
                    "vega": 1.0,
                    "theta": -0.5,
                    "expiry_indicator": "w",
                    "contract_size": 100,
                }
            )
    return {
        RawTable.SECURITY: pl.DataFrame({"ticker": [ticker], "secid": [108105]}),
        RawTable.OPTIONS: pl.DataFrame(rows),
        RawTable.SPOT: pl.DataFrame(
            {"secid": [108105], "date": [trading_date.isoformat()], "close": [spot]}
        ),
        "previous_spot": pl.DataFrame(
            {
                "secid": [108105],
                "date": [previous_business_day(trading_date).isoformat()],
                "close": [99.0],
            }
        ),
        RawTable.FORWARDS: pl.DataFrame(
            {
                "secid": [108105],
                "date": [trading_date.isoformat()],
                "expiration": [expiry.isoformat()],
                "amsettlement": [0],
                "forwardprice": [forward],
            }
        ),
        RawTable.ZERO_CURVE: pl.DataFrame(
            {
                "date": [trading_date.isoformat(), trading_date.isoformat()],
                "days": [1, 30],
                "rate": [5.0, 5.0],
            }
        ),
    }


def _publish_raw_inputs(
    store: ArtifactStore,
    trading_date: date,
    *,
    ticker: str = "SPX",
    expiry: date = date(2024, 1, 19),
    force_enrichment_rejection: bool = False,
) -> None:
    frames = _raw_frames(
        trading_date,
        expiry,
        ticker=ticker,
        force_enrichment_rejection=force_enrichment_rejection,
    )
    producer = raw_content_producer_version()
    partitions = (
        (RawTable.SECURITY, None, frames[RawTable.SECURITY]),
        (RawTable.OPTIONS, trading_date, frames[RawTable.OPTIONS]),
        (RawTable.SPOT, trading_date, frames[RawTable.SPOT]),
        (RawTable.SPOT, previous_business_day(trading_date), frames["previous_spot"]),
        (RawTable.FORWARDS, trading_date, frames[RawTable.FORWARDS]),
        (RawTable.ZERO_CURVE, trading_date, frames[RawTable.ZERO_CURVE]),
    )
    for table, observation_date, frame in partitions:
        spec = raw_artifact_request(
            RawPartition(table, ticker, observation_date),
            source="optionmetrics",
            source_generation="default-v1",
            producer_version=producer,
        ).spec
        if store.resolve(spec) is not None:
            continue
        with store.writer(spec, metadata={"test_fixture": True}) as builder:
            frame.write_parquet(builder.staging_path / "data.parquet")


def _dataset_config(trading_date: date = date(2024, 1, 3)) -> InternalDatasetConfig:
    return InternalDatasetConfig(
        name="test-reference",
        ticker="SPX",
        start_date=trading_date,
        end_date=trading_date,
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )


def test_pointer_and_stage_specs_pin_exact_lineage(tmp_path: Path) -> None:
    config = DailyWidbConfig()
    pointers = plan_raw_day_pointers(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        source="optionmetrics",
        source_generation="default-v1",
        raw_producer_version=raw_content_producer_version(),
    )
    assert len(pointers.all()) == 6
    assert pointers.previous_spot.namespace.endswith("date-2024-01-02")
    paired = paired_day_spec(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        config=config,
        raw=pointers,
    )
    enriched = enriched_day_spec(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        config=config,
        paired=paired.pointer,
    )
    assert paired.upstream == pointers.all()
    assert enriched.upstream == (paired.pointer,)
    assert "/paired-day/" in paired.namespace
    assert "/enriched-day/" in enriched.namespace


def test_overnight_fraction_changes_paired_and_enriched_artifact_identities() -> None:
    trading_date = date(2024, 1, 3)
    raw = plan_raw_day_pointers(
        ticker="SPX",
        trading_date=trading_date,
        source="optionmetrics",
        source_generation="default-v1",
        raw_producer_version=raw_content_producer_version(),
    )
    paired = tuple(
        paired_day_spec(
            ticker="SPX",
            trading_date=trading_date,
            config=DailyWidbConfig(clock=ClockConfig(overnight_variance_fraction=fraction)),
            raw=raw,
        )
        for fraction in (0.1, 0.2, 0.5)
    )
    enriched = tuple(
        enriched_day_spec(
            ticker="SPX",
            trading_date=trading_date,
            config=DailyWidbConfig(clock=ClockConfig(overnight_variance_fraction=fraction)),
            paired=paired_spec.pointer,
        )
        for fraction, paired_spec in zip((0.1, 0.2, 0.5), paired, strict=True)
    )
    assert len({spec.artifact_id for spec in paired}) == 3
    assert len({spec.artifact_id for spec in enriched}) == 3


def test_daily_io_and_producer_version_closures_resolve() -> None:
    assert str(read_daily_frame.version)
    assert str(publish_daily_frame.version)
    assert paired_day_producer_version()
    assert enriched_day_producer_version()
    assert no_arb_day_producer_version()
    assert filtered_day_producer_version()
    assert fitted_day_producer_version()


def test_surface_stage_identities_use_projected_configuration() -> None:
    trading_date = date(2024, 1, 3)
    daily = DailyWidbConfig()
    raw = plan_raw_day_pointers(
        ticker="SPX",
        trading_date=trading_date,
        source="optionmetrics",
        source_generation="default-v1",
        raw_producer_version=raw_content_producer_version(),
    )
    paired = paired_day_spec(
        ticker="SPX", trading_date=trading_date, config=daily, raw=raw
    )
    enriched = enriched_day_spec(
        ticker="SPX", trading_date=trading_date, config=daily, paired=paired.pointer
    )
    base = DailySurfaceConfig()
    changed_smooth = replace(
        base,
        smooth_sanos=SmoothSanosConfig(smoothness=0.5),
    )
    no_arb_base = no_arb_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=base,
        enriched=enriched.pointer,
    )
    no_arb_changed = no_arb_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=changed_smooth,
        enriched=enriched.pointer,
    )
    assert no_arb_base.artifact_id == no_arb_changed.artifact_id
    filtered_base = filtered_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=base,
        no_arb=no_arb_base.pointer,
    )
    filtered_changed = filtered_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=changed_smooth,
        no_arb=no_arb_changed.pointer,
    )
    assert filtered_base.artifact_id == filtered_changed.artifact_id
    fitted_base = fitted_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=base,
        filtered=filtered_base.pointer,
    )
    fitted_changed = fitted_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=changed_smooth,
        filtered=filtered_changed.pointer,
    )
    assert fitted_base.artifact_id != fitted_changed.artifact_id
    with pytest.raises(ValueError, match="at most four"):
        IntermediateBuildRequest(
            ticker="SPX",
            start_date=trading_date,
            end_date=trading_date,
            through="fitted-day",
            max_workers=5,
        )


def test_fitted_build_reuses_terminal_and_rebuilds_from_filtered_only(tmp_path: Path) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    daily = DailyWidbConfig()
    surface = DailySurfaceConfig()
    _publish_raw_inputs(store, trading_date)
    request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=trading_date,
        end_date=trading_date,
        through="fitted-day",
        max_workers=1,
    )
    first = build_intermediates(locations, request, daily, surface)
    assert first.summary is not None
    assert first.summary.completed == 1

    raw = plan_raw_day_pointers(
        ticker="SPX",
        trading_date=trading_date,
        source="optionmetrics",
        source_generation="default-v1",
        raw_producer_version=raw_content_producer_version(),
    )
    paired = paired_day_spec(
        ticker="SPX", trading_date=trading_date, config=daily, raw=raw
    )
    enriched = enriched_day_spec(
        ticker="SPX", trading_date=trading_date, config=daily, paired=paired.pointer
    )
    no_arb = no_arb_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=surface,
        enriched=enriched.pointer,
    )
    filtered = filtered_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=surface,
        no_arb=no_arb.pointer,
    )
    fitted = fitted_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=surface,
        filtered=filtered.pointer,
    )
    assert first.terminal == (fitted.pointer,)
    frame = read_daily_frame(
        store,
        store.validate(fitted.pointer, expected=fitted),
        schema=FITTED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )
    assert frame.schema == pl.Schema(FITTED_SCHEMA)
    assert frame.get_column("options_filter_eligible").sum() > 0

    shutil.rmtree(locations.raw)
    reused = build_intermediates(locations, request, daily, surface)
    assert reused.summary is not None
    assert reused.summary.reused == 1

    shutil.rmtree(store.path_for(fitted.pointer))
    for pointer in (paired.pointer, enriched.pointer, no_arb.pointer):
        shutil.rmtree(store.path_for(pointer))
    assert store.validate(filtered.pointer, expected=filtered)
    rebuilt = build_intermediates(locations, request, daily, surface)
    assert rebuilt.summary is not None
    assert rebuilt.summary.completed == 1
    rebuilt_frame = read_daily_frame(
        store,
        store.validate(fitted.pointer, expected=fitted),
        schema=FITTED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )
    assert rebuilt_frame.equals(frame)


def test_failed_fitted_worker_journals_error_publishes_nothing_and_restarts(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    daily = DailyWidbConfig()
    base_surface = DailySurfaceConfig()
    _publish_raw_inputs(store, trading_date)
    filtered_request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=trading_date,
        end_date=trading_date,
        through="filtered-day",
        max_workers=1,
    )
    filtered_result = build_intermediates(
        locations, filtered_request, daily, base_surface
    )
    valid_filtered = read_daily_frame(
        store,
        store.validate(filtered_result.terminal[0]),
        schema=FILTERED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )

    surface = DailySurfaceConfig(
        options_filter=OptionsFilterConfig(max_nms=(-99.0, 99.0))
    )
    raw = plan_raw_day_pointers(
        ticker="SPX",
        trading_date=trading_date,
        source="optionmetrics",
        source_generation="default-v1",
        raw_producer_version=raw_content_producer_version(),
    )
    paired = paired_day_spec(
        ticker="SPX", trading_date=trading_date, config=daily, raw=raw
    )
    enriched = enriched_day_spec(
        ticker="SPX", trading_date=trading_date, config=daily, paired=paired.pointer
    )
    no_arb = no_arb_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=surface,
        enriched=enriched.pointer,
    )
    filtered = filtered_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=surface,
        no_arb=no_arb.pointer,
    )
    fitted = fitted_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=surface,
        filtered=filtered.pointer,
    )
    invalid_filtered = valid_filtered.with_columns(
        pl.when(pl.col("options_filter_eligible"))
        .then(pl.lit(0.0, dtype=pl.Float32))
        .otherwise(pl.col("spot_normalized_implied_forward"))
        .alias("spot_normalized_implied_forward"),
        pl.lit(canonical_sha256({"options_filter": surface.options_filter})).alias(
            "options_filter_configuration_id"
        ),
    ).select(*FILTERED_SCHEMA)
    publish_daily_frame(
        store,
        filtered,
        invalid_filtered,
        schema=FILTERED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )

    fitted_request = replace(filtered_request, through="fitted-day")
    with pytest.raises(DailyExecutionError) as caught:
        build_intermediates(locations, fitted_request, daily, surface)
    assert caught.value.summary.failed == 1
    assert caught.value.summary.failures[0].error_type == "DailyDataError"
    assert "finite positive discounted forward" in caught.value.summary.failures[0].message
    records = [
        json.loads(line)
        for line in caught.value.summary.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    failed = [record for record in records if record["event"] == "unit_failed"]
    assert len(failed) == 1
    assert "finite positive discounted forward" in str(failed[0]["message"])
    assert not store.path_for(fitted.pointer).exists()

    shutil.rmtree(store.path_for(filtered.pointer))
    restarted = build_intermediates(locations, fitted_request, daily, surface)
    assert restarted.summary is not None
    assert restarted.summary.completed == 1
    assert store.validate(fitted.pointer, expected=fitted)


def test_multi_day_fitted_build_reuses_completed_day_after_failure(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    first_date = date(2024, 1, 3)
    second_date = date(2024, 1, 4)
    daily = DailyWidbConfig()
    surface = DailySurfaceConfig()
    _publish_raw_inputs(store, first_date)
    request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=first_date,
        end_date=second_date,
        through="fitted-day",
        max_workers=1,
    )
    inspection = build_intermediates(
        locations, request, daily, surface, dry_run=True
    )
    with pytest.raises(DailyExecutionError) as caught:
        build_intermediates(locations, request, daily, surface)
    assert caught.value.summary.completed == 1
    assert caught.value.summary.failed == 1
    assert store.validate(inspection.expected[0])
    assert not store.path_for(inspection.expected[1]).exists()

    _publish_raw_inputs(store, second_date)
    restarted = build_intermediates(locations, request, daily, surface)
    assert restarted.summary is not None
    assert restarted.summary.reused == 1
    assert restarted.summary.completed == 1
    assert restarted.summary.failed == 0
    assert all(store.validate(pointer) for pointer in inspection.expected)


def test_known_bad_typed_empty_manifest_records_reason_and_policy(tmp_path: Path) -> None:
    store = ArtifactStore(_locations(tmp_path))
    trading_date = date(2025, 1, 8)
    raw = plan_raw_day_pointers(
        ticker="SPX",
        trading_date=trading_date,
        source="optionmetrics",
        source_generation="default-v1",
        raw_producer_version=raw_content_producer_version(),
    )
    partitions = (
        (RawTable.SECURITY, None),
        (RawTable.OPTIONS, trading_date),
        (RawTable.SPOT, trading_date),
        (RawTable.SPOT, date(2025, 1, 7)),
        (RawTable.FORWARDS, trading_date),
        (RawTable.ZERO_CURVE, trading_date),
    )
    for table, observation_date in partitions:
        raw_spec = raw_artifact_request(
            RawPartition(table, "SPX", observation_date),
            source="optionmetrics",
            source_generation="default-v1",
            producer_version=raw_content_producer_version(),
        ).spec
        with store.writer(raw_spec) as builder:
            pl.DataFrame({"placeholder": []}).write_parquet(builder.staging_path / "data.parquet")
    spec = paired_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=DailyWidbConfig(),
        raw=raw,
    )
    artifact = publish_daily_frame(
        store,
        spec,
        pl.DataFrame(schema=PAIRED_SCHEMA),
        schema=PAIRED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )
    metadata = artifact.manifest["metadata"]
    assert metadata["typed_empty"] is True
    assert metadata["typed_empty_reason"] == "known_bad_day_all_tickers"
    assert "bad-registry" in metadata["bad_data_policy_version"]


def test_vendor_ticker_uses_safe_work_label_without_changing_raw_identity(
    tmp_path: Path,
) -> None:
    request = IntermediateBuildRequest(
        ticker="^SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
    )
    plan = plan_intermediate_build(_locations(tmp_path), request, DailyWidbConfig())
    item = plan.items[0]
    assert "^" not in item.ticker
    assert item.payload["ticker"] == "^SPX"
    result = build_intermediates(_locations(tmp_path), request, DailyWidbConfig(), dry_run=True)
    assert "ticker-SPX-" in result.expected[0].namespace
    assert result.raw_dependencies[0].options.namespace.endswith(
        "ticker-SPX/year-2024/date-2024-01-03"
    )


def test_dry_run_is_read_only_and_wildcard_miss_is_reuse_only(tmp_path: Path) -> None:
    locations = _locations(tmp_path / "dry")
    request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
    )
    result = build_intermediates(locations, request, DailyWidbConfig(), dry_run=True)
    assert len(result.expected) == 1
    assert result.existing == ()
    assert not (tmp_path / "dry").exists()
    wildcard = IntermediateBuildRequest(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        version_override="*",
    )
    with pytest.raises(WildcardReuseMissError, match="reuse only"):
        build_intermediates(locations, wildcard, DailyWidbConfig(), dry_run=True)


def test_build_enriched_day_from_six_raw_artifacts_and_reuse(tmp_path: Path) -> None:
    locations = _locations(tmp_path / "material")
    store = ArtifactStore(locations)
    _publish_raw_inputs(store, date(2024, 1, 3))
    request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        max_workers=1,
        progress_interval_seconds=0.1,
    )
    first = build_intermediates(locations, request, DailyWidbConfig())
    assert first.summary is not None
    assert first.summary.completed == 1
    target = store.validate(first.expected[0])
    frame = pl.read_parquet(target.path / "data.parquet")
    assert frame.height == 13
    assert (target.path / "config.json").is_file()
    assert (target.path / "README.md").is_file()
    assert len(target.manifest["upstream"]) == 1

    second = build_intermediates(locations, request, DailyWidbConfig())
    assert second.summary is not None
    assert second.summary.reused == 1
    assert second.summary.completed == 0

    relocated = StorageLocations(
        raw=tmp_path / "raw-no-longer-present",
        intermediate=locations.intermediate,
        datasets=locations.datasets,
        experiments=locations.experiments,
    )
    output_first = build_intermediates(relocated, request, DailyWidbConfig())
    assert output_first.summary is not None
    assert output_first.summary.reused == 1


def test_wildcard_reports_actual_reused_artifact_and_stored_version(tmp_path: Path) -> None:
    locations = _locations(tmp_path / "wildcard")
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    config = DailyWidbConfig()
    _publish_raw_inputs(store, trading_date)
    paired_request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=trading_date,
        end_date=trading_date,
        through="paired-day",
        progress_interval_seconds=0.1,
    )
    paired_result = build_intermediates(locations, paired_request, config)
    paired_artifact = store.validate(paired_result.terminal[0])
    paired_frame = read_daily_frame(
        store,
        paired_artifact,
        schema=PAIRED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )
    current_spec = enriched_day_spec(
        ticker="SPX",
        trading_date=trading_date,
        config=config,
        paired=paired_artifact.pointer,
    )
    legacy_spec = current_spec.with_producer_version("legacy-test-version")
    legacy = publish_daily_frame(
        store,
        legacy_spec,
        enrich_daily_frame(paired_frame, config=config),
        schema=ENRICHED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )

    wildcard = build_intermediates(
        locations,
        IntermediateBuildRequest(
            ticker="SPX",
            start_date=trading_date,
            end_date=trading_date,
            version_override="*",
        ),
        config,
        dry_run=True,
    )
    assert wildcard.expected[0] != legacy.pointer
    assert wildcard.terminal == (legacy.pointer,)
    assert wildcard.existing == (legacy.pointer,)
    assert wildcard.reuses[0].used_wildcard
    assert wildcard.reuses[0].reused_producer_version == "legacy-test-version"
    assert wildcard.reuses[0].warning is not None


def test_plan_excludes_half_days_without_changing_daily_content_config(tmp_path: Path) -> None:
    locations = _locations(tmp_path)
    included = plan_intermediate_build(
        locations,
        IntermediateBuildRequest(
            ticker="SPX",
            start_date=date(2023, 11, 24),
            end_date=date(2023, 11, 24),
            include_half_days=True,
        ),
        DailyWidbConfig(),
    )
    assert len(included.items) == 1
    with pytest.raises(ValueError, match="no eligible"):
        plan_intermediate_build(
            locations,
            IntermediateBuildRequest(
                ticker="SPX",
                start_date=date(2023, 11, 24),
                end_date=date(2023, 11, 24),
                include_half_days=False,
            ),
            DailyWidbConfig(),
        )
    after_half_day = plan_internal_dataset(
        replace(
            _dataset_config(date(2023, 11, 27)),
            include_half_days=False,
        )
    )
    assert after_half_day.daily_artifacts[0].raw.previous_spot.namespace.endswith(
        "date-2023-11-24"
    )


def test_internal_dataset_plan_commits_dates_raw_roles_and_clock() -> None:
    base = _dataset_config()
    changed = replace(
        base,
        daily=DailyWidbConfig(clock=ClockConfig(overnight_variance_fraction=0.1)),
    )
    first = plan_internal_dataset(base)
    second = plan_internal_dataset(changed)

    assert first.expected_dates == (date(2024, 1, 3),)
    assert len(first.raw_catalog_spec.upstream) == 6
    assert first.dataset_spec.upstream == (first.raw_catalog_spec.pointer,)
    assert first.configuration["planned_dates"] == ["2024-01-03"]
    assert first.configuration["raw_mapping_digest"] == first.raw_mapping_digest
    assert first.dataset_spec.artifact_id != second.dataset_spec.artifact_id


def test_reference_profile_freezes_full_calendar_and_policy_structure() -> None:
    document = json.loads(
        (Path(__file__).parents[1] / "configs" / "widb-spx-reference.json").read_text(
            encoding="utf-8"
        )
    )
    included_config = InternalDatasetConfig.from_document(document)
    excluded_config = replace(included_config, include_half_days=False)
    included = plan_internal_dataset(included_config)
    excluded = plan_internal_dataset(excluded_config)
    assert len(included.expected_dates) == 918
    assert len(excluded.expected_dates) == 911
    half_day_dates = tuple(
        sorted(set(included.expected_dates) - set(excluded.expected_dates))
    )
    assert half_day_dates == (
        date(2022, 11, 25),
        date(2023, 7, 3),
        date(2023, 11, 24),
        date(2024, 7, 3),
        date(2024, 11, 29),
        date(2024, 12, 24),
        date(2025, 7, 3),
    )
    assert tuple(
        record["trading_date"]
        for record in internal_dataset_module._scheduled_exclusions(excluded_config)
    ) == half_day_dates
    typed_empty_dates = tuple(
        trading_date
        for trading_date in included.expected_dates
        if internal_dataset_module.bad_day_reason(
            included_config.ticker, trading_date
        )
        == "known_bad_day_all_tickers"
    )
    assert typed_empty_dates == (date(2025, 1, 8),)
    assert len(included.raw_catalog_spec.upstream) == 3_674


def test_all_stage_and_dataset_versions_are_deterministic_in_fresh_process() -> None:
    expected = {
        "paired": paired_day_producer_version(),
        "enriched": enriched_day_producer_version(),
        "no_arb": no_arb_day_producer_version(),
        "filtered": filtered_day_producer_version(),
        "fitted": fitted_day_producer_version(),
        "dataset": internal_dataset_producer_version(),
    }
    script = """
import json
from sanos.datasets.internal import internal_dataset_producer_version
from sanos.widb.build import (
    enriched_day_producer_version,
    filtered_day_producer_version,
    fitted_day_producer_version,
    no_arb_day_producer_version,
    paired_day_producer_version,
)
print(json.dumps({
    "paired": paired_day_producer_version(),
    "enriched": enriched_day_producer_version(),
    "no_arb": no_arb_day_producer_version(),
    "filtered": filtered_day_producer_version(),
    "fitted": fitted_day_producer_version(),
    "dataset": internal_dataset_producer_version(),
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WRDS_OFF": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == expected


def test_restart_policy_versions_resolve_in_a_fresh_process() -> None:
    functions = {
        "probe": probe_daily_target,
        "worker": build_daily_target,
        "intermediates": build_intermediates,
        "dataset_build": build_internal_dataset,
    }
    expected = {
        name: hashlib.sha256(str(function.version).encode("utf-8")).hexdigest()
        for name, function in functions.items()
    }
    script = """
import hashlib
import json
from sanos.datasets.internal import build_internal_dataset
from sanos.widb.build import build_daily_target, build_intermediates, probe_daily_target
functions = {
    "probe": probe_daily_target,
    "worker": build_daily_target,
    "intermediates": build_intermediates,
    "dataset_build": build_internal_dataset,
}
print(json.dumps({
    name: hashlib.sha256(str(function.version).encode("utf-8")).hexdigest()
    for name, function in functions.items()
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WRDS_OFF": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == expected


def test_restart_orchestration_versions_own_direct_dependencies() -> None:
    probe_dependencies = dict(probe_daily_target.version.dependencies[1])
    assert {"_specs_from_item", "_target_spec"} <= set(probe_dependencies)

    worker_dependencies = dict(build_daily_target.version.dependencies[1])
    assert {
        "_build_enriched",
        "_build_filtered",
        "_build_fitted",
        "_build_no_arb",
        "_build_paired",
        "_specs_from_item",
        "_target_spec",
    } <= set(worker_dependencies)

    intermediate_dependencies = dict(build_intermediates.version.dependencies[1])
    assert {
        "IntermediateBuildRequest",
        "IntermediateBuildResult",
        "StorageLocations",
        "build_daily_target",
        "plan_intermediate_build",
        "probe_daily_target",
        "run_daily_plan",
    } <= set(intermediate_dependencies)


def test_internal_dataset_validator_version_owns_all_semantic_helpers() -> None:
    dependencies = dict(
        internal_dataset_module.validate_internal_dataset.version.dependencies[1]
    )
    assert {
        "_fail",
        "_legacy_comparison_document",
        "_open_validated_internal_dataset",
        "_partition_record_statistics",
        "_pointer_from_value",
        "_read_json_file",
        "_safe_partition_ticker_label",
        "_scheduled_exclusions",
        "_schema_document",
        "_dataset_sha256_file",
        "aggregate_statistics",
        "bad_data_policy",
        "canonical_sha256",
        "canonical_value",
        "is_half_day",
        "validate_fitted_partition",
    } <= set(dependencies)


def test_legacy_comparison_document_schema_change_is_version_owned() -> None:
    document = internal_dataset_module._legacy_comparison_document(_dataset_config())
    assert document["schema"] == internal_dataset_module.LEGACY_COMPARISON_SCHEMA
    assert document["schema"] == "sanos-legacy-comparison-v4"
    assert str(internal_dataset_module._legacy_comparison_document.version).startswith(
        "1.2"
    )


def test_internal_dataset_document_is_strict() -> None:
    config = _dataset_config()
    document = {
        "dataset": {
            "name": config.name,
            "ticker": config.ticker,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "include_half_days": config.include_half_days,
            "source": config.source,
            "source_generation": config.source_generation,
        },
        "daily": canonical_value(config.daily),
        "surface": canonical_value(config.surface),
    }
    assert InternalDatasetConfig.from_document(document) == config
    with pytest.raises(ValueError, match="sections changed"):
        InternalDatasetConfig.from_document({**document, "path": "/private"})
    bad_type = json.loads(json.dumps(document))
    bad_type["dataset"]["include_half_days"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        InternalDatasetConfig.from_document(bad_type)


def test_internal_dataset_publication_is_self_contained_and_output_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path / "self-contained")
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    _publish_raw_inputs(store, trading_date)
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "a" * 40)

    result = build_internal_dataset(
        locations,
        _dataset_config(trading_date),
        DatasetBuildOptions(max_workers=1),
    )
    assert result.built
    assert result.validation is not None
    assert result.validation.populated_dates == 1
    assert result.validation.typed_empty_dates == 0
    published = store.validate(result.artifact)
    assert published.manifest["metadata"]["canonical_promotion_ready"] is False
    dataset = open_internal_dataset(store, result.artifact)
    loaded_day = dataset.load_day(trading_date)
    assert loaded_day.schema == pl.Schema(FITTED_SCHEMA)
    fitted_pointer = plan_internal_dataset(
        _dataset_config(trading_date)
    ).daily_artifacts[0].fitted.pointer
    fitted_source = read_daily_frame(
        store,
        store.validate(fitted_pointer),
        schema=FITTED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )
    logical_keys = fitted_source.select("expiry_id", "normalized_strike")
    assert not logical_keys.is_duplicated().any()
    repeated_geometry = fitted_source.select(
        "trading_date", "expiry_id", "time_to_expiry_years"
    )
    assert repeated_geometry.is_duplicated().any()
    assert loaded_day.equals(fitted_source)
    assert dataset.scan().collect().height == result.validation.row_count
    deep = validate_internal_dataset(store, result.artifact, deep_lineage=True)
    assert deep is not None and deep.row_count == result.validation.row_count

    interrupted_config = replace(_dataset_config(trading_date), name="interrupted-test")
    expected_interrupted = plan_internal_dataset(interrupted_config).dataset_spec.pointer
    copy_file = artifact_store_module.ArtifactBuilder.copy_file

    def interrupt_copy(
        builder: artifact_store_module.ArtifactBuilder,
        relative_path: str,
        source: Path,
    ) -> Path:
        copy_file(builder, relative_path, source)
        raise KeyboardInterrupt("simulated mid-copy interruption")

    monkeypatch.setattr(
        artifact_store_module.ArtifactBuilder,
        "copy_file",
        interrupt_copy,
    )
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        build_internal_dataset(locations, interrupted_config)
    assert not store.path_for(expected_interrupted).exists()
    interrupted_plan = plan_internal_dataset(interrupted_config)
    journal_path = (
        locations.datasets / ".runs" / "widb" / f"{interrupted_plan.plan_id}.jsonl"
    )
    assert '"event":"run_interrupted"' in journal_path.read_text(encoding="utf-8")
    monkeypatch.setattr(artifact_store_module.ArtifactBuilder, "copy_file", copy_file)
    recovered = build_internal_dataset(locations, interrupted_config)
    assert recovered.built and recovered.artifact == expected_interrupted
    recovered_dataset = open_internal_dataset(store, recovered.artifact)
    tampered_partition = recovered_dataset.artifact.path / str(
        recovered_dataset.partitions[0]["path"]
    )
    tampered_partition.write_bytes(tampered_partition.read_bytes() + b"tamper")
    with pytest.raises(ArtifactIntegrityError, match="byte size changed"):
        validate_internal_dataset(store, recovered.artifact)

    shutil.rmtree(locations.raw)
    shutil.rmtree(locations.intermediate)
    shallow = validate_internal_dataset(store, result.artifact)
    assert shallow is not None and shallow.populated_dates == 1
    with pytest.raises(ArtifactNotFoundError, match="does not exist"):
        validate_internal_dataset(store, result.artifact, deep_lineage=True)

    current = build_internal_dataset(locations, _dataset_config(trading_date))
    assert not current.built
    assert current.reuse is not None and not current.reuse.used_wildcard

    with pytest.warns(RuntimeWarning, match="version='\\*'"):
        reused = build_internal_dataset(
            locations,
            _dataset_config(trading_date),
            DatasetBuildOptions(version_override="*"),
        )
    assert not reused.built
    assert reused.reuse is not None and reused.reuse.used_wildcard
    assert reused.artifact == result.artifact


def test_internal_dataset_round_trips_caret_optionmetrics_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path / "caret-ticker")
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    _publish_raw_inputs(store, trading_date, ticker="^SPX")
    config = replace(_dataset_config(trading_date), name="caret-spx", ticker="^SPX")
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "b" * 40)

    result = build_internal_dataset(locations, config)
    dataset = open_internal_dataset(store, result.artifact)
    assert dataset.config.ticker == "^SPX"
    assert dataset.load_day(trading_date).get_column("ticker").unique().to_list() == ["^SPX"]
    assert "ticker=SPX-e32f668a" in str(dataset.partitions[0]["path"])
    assert validate_internal_dataset(store, result.artifact) == result.validation


def test_internal_dataset_concurrent_identical_publishers_converge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path / "concurrent-publishers")
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    _publish_raw_inputs(store, trading_date)
    config = replace(_dataset_config(trading_date), name="concurrent-publishers")
    build_intermediates(
        locations,
        IntermediateBuildRequest(
            ticker=config.ticker,
            start_date=trading_date,
            end_date=trading_date,
            through="fitted-day",
        ),
        config.daily,
        config.surface,
    )
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "e" * 40)
    plan = plan_internal_dataset(config)
    fitted_artifacts, statistics = internal_dataset_module._validate_fitted_sources(
        store,
        config,
        plan,
        progress=None,
        progress_interval_seconds=30.0,
    )
    internal_dataset_module.publish_raw_lineage_catalog(store, plan)
    publisher = internal_dataset_module._publish_internal_dataset
    rendezvous = Barrier(2, timeout=20.0)

    def synchronized_publish(*args: Any, **kwargs: Any) -> Any:
        rendezvous.wait()
        return publisher(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _: synchronized_publish(
                    store,
                    config,
                    plan,
                    fitted_artifacts,
                    statistics,
                    source_revision="e" * 40,
                    progress=None,
                    progress_interval_seconds=30.0,
                ),
                range(2),
            )
        )
    expected = plan.dataset_spec.pointer
    assert {result.pointer for result in results} == {expected}
    assert validate_internal_dataset(store, expected) is not None
    namespace = store.namespace_path(expected.storage_class, expected.namespace)
    assert not tuple(namespace.glob(f".staging-{expected.artifact_id}-*"))


def test_internal_dataset_publishes_half_day_ordinary_and_typed_empty_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path / "multi-outcome")
    store = ArtifactStore(locations)
    start = date(2024, 12, 24)
    end = date(2025, 1, 8)
    expected_dates = business_dates(start, end)
    for trading_date in expected_dates:
        _publish_raw_inputs(
            store,
            trading_date,
            expiry=date(2025, 1, 17),
            force_enrichment_rejection=True,
        )
    config = replace(
        _dataset_config(),
        name="multi-outcome-reference-candidate",
        start_date=start,
        end_date=end,
    )
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "c" * 40)
    progress_events: list[dict[str, object]] = []

    result = build_internal_dataset(
        locations,
        config,
        DatasetBuildOptions(max_workers=4, progress_interval_seconds=1.0),
        progress=progress_events.append,
    )
    assert result.validation is not None
    assert result.validation.requested_dates == len(expected_dates)
    assert result.validation.populated_dates == len(expected_dates) - 1
    assert result.validation.typed_empty_dates == 1
    dataset = open_internal_dataset(store, result.artifact)
    assert tuple(
        date.fromisoformat(str(record["trading_date"])) for record in dataset.partitions
    ) == expected_dates
    by_date = {str(record["trading_date"]): record for record in dataset.partitions}
    assert by_date["2024-12-24"]["is_half_day"] is True
    assert by_date["2025-01-08"]["typed_empty"] is True
    assert by_date["2025-01-08"]["typed_empty_reason"] == "known_bad_day_all_tickers"
    assert dataset.load_day(date(2025, 1, 8)).is_empty()
    assert dataset.load_day(date(2024, 12, 24))["is_half_day"].all()
    event_names = [event.get("event") for event in progress_events]
    assert "dataset_validation_started" in event_names
    assert "dataset_copy_progress" in event_names
    assert "dataset_validation_completed" in event_names
    assert event_names[-1] == "dataset_build_completed"
    completed_event = next(
        event
        for event in reversed(progress_events)
        if event.get("event") == "dataset_validation_completed"
    )
    assert completed_event["completed"] == len(expected_dates)
    assert completed_event["total"] == len(expected_dates)

    lineage = json.loads(
        (dataset.artifact.path / "lineage.json").read_text(encoding="utf-8")
    )
    chains = {item["trading_date"]: item for item in lineage["daily_chains"]}
    assert (
        chains["2024-12-26"]["raw_roles"]["previous_spot"]
        == chains["2024-12-24"]["raw_roles"]["spot"]
    )


def test_internal_dataset_rejects_resigned_catalog_lineage_and_evidence_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path / "resigned-mutations")
    store = ArtifactStore(locations)
    start = date(2024, 1, 3)
    end = date(2024, 1, 4)
    for trading_date in business_dates(start, end):
        _publish_raw_inputs(store, trading_date)
    config = replace(
        _dataset_config(),
        name="resigned-mutations",
        start_date=start,
        end_date=end,
    )
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    result = build_internal_dataset(locations, config)
    source = store.validate(result.artifact)
    plan = plan_internal_dataset(config)

    def resign_dataset(
        producer: str,
        *,
        document_mutations: dict[str, Callable[[dict[str, Any]], None]] | None = None,
        metadata_mutation: Callable[[dict[str, Any]], None] | None = None,
        spec: ArtifactSpec | None = None,
    ) -> ArtifactPointer:
        target = plan.dataset_spec.with_producer_version(producer) if spec is None else spec
        documents: dict[str, bytes] = {}
        for record in source.manifest["files"]:
            relative = str(record["path"])
            payload = (source.path / relative).read_bytes()
            if document_mutations is not None and relative in document_mutations:
                document = json.loads(payload)
                document_mutations[relative](document)
                payload = (canonical_json(document) + "\n").encode("utf-8")
            elif relative == "README.md":
                payload = payload.replace(
                    result.artifact.artifact_id.encode(),
                    target.artifact_id.encode(),
                )
            documents[relative] = payload
        metadata = json.loads(canonical_json(source.manifest["metadata"]))
        metadata["producer_version"] = target.producer_version
        if metadata_mutation is not None:
            metadata_mutation(metadata)
        for evidence_name in metadata["evidence_checksums"]:
            metadata["evidence_checksums"][evidence_name] = hashlib.sha256(
                documents[evidence_name]
            ).hexdigest()
        with store.writer(target, metadata=metadata) as builder:
            for relative, payload in documents.items():
                builder.write_bytes(relative, payload)
        return target.pointer

    def remove_last_partition(document: dict[str, Any]) -> None:
        document["partitions"].pop()

    def duplicate_partition(document: dict[str, Any]) -> None:
        document["partitions"].append(dict(document["partitions"][0]))

    def reverse_partitions(document: dict[str, Any]) -> None:
        document["partitions"].reverse()

    for producer, mutation in (
        ("missing-date-v0", remove_last_partition),
        ("duplicate-date-v0", duplicate_partition),
        ("reordered-date-v0", reverse_partitions),
    ):
        pointer = resign_dataset(
            producer,
            document_mutations={"partitions.json": mutation},
        )
        with pytest.raises(DatasetValidationError, match="Partition dates"):
            validate_internal_dataset(store, pointer)

    def corrupt_schema(document: dict[str, Any]) -> None:
        document["fields"][0]["units"] = "corrupt"

    schema_pointer = resign_dataset(
        "schema-evidence-v0",
        document_mutations={"schema.json": corrupt_schema},
    )
    with pytest.raises(DatasetValidationError, match="schema.json"):
        validate_internal_dataset(store, schema_pointer)

    def corrupt_validation(document: dict[str, Any]) -> None:
        document["summary"]["row_count"] += 1

    validation_pointer = resign_dataset(
        "validation-evidence-v0",
        document_mutations={"validation.json": corrupt_validation},
    )
    with pytest.raises(DatasetValidationError, match="validation.json"):
        validate_internal_dataset(store, validation_pointer)

    lineage_document = json.loads(
        (source.path / "lineage.json").read_text(encoding="utf-8")
    )

    def corrupt_stage_ancestry(document: dict[str, Any]) -> None:
        document["daily_chains"][0]["paired"] = document["daily_chains"][0][
            "enriched"
        ]

    ancestry_pointer = resign_dataset(
        "ancestry-evidence-v0",
        document_mutations={"lineage.json": corrupt_stage_ancestry},
    )
    with pytest.raises(DatasetValidationError, match="ancestry differs"):
        validate_internal_dataset(store, ancestry_pointer, deep_lineage=True)

    def corrupt_fitted_manifest_digest(document: dict[str, Any]) -> None:
        document["partitions"][0]["fitted_manifest_sha256"] = "0" * 64

    digest_pointer = resign_dataset(
        "fitted-digest-v0",
        document_mutations={"partitions.json": corrupt_fitted_manifest_digest},
    )
    with pytest.raises(DatasetValidationError, match="manifest checksum evidence"):
        validate_internal_dataset(store, digest_pointer, deep_lineage=True)

    def corrupt_fitted_source_digest(document: dict[str, Any]) -> None:
        document["partitions"][0]["fitted_source_sha256"] = "0" * 64

    source_digest_pointer = resign_dataset(
        "fitted-source-digest-v0",
        document_mutations={"partitions.json": corrupt_fitted_source_digest},
    )
    with pytest.raises(DatasetValidationError, match="source/copy checksum evidence"):
        validate_internal_dataset(store, source_digest_pointer, deep_lineage=True)

    def corrupt_flattened_fitted(metadata: dict[str, Any]) -> None:
        metadata["fitted_artifacts"][0] = lineage_document["daily_chains"][0][
            "filtered"
        ]

    flattened_pointer = resign_dataset(
        "flattened-fitted-v0",
        metadata_mutation=corrupt_flattened_fitted,
    )
    with pytest.raises(DatasetValidationError, match="fitted identities"):
        validate_internal_dataset(store, flattened_pointer)

    def corrupt_flattened_raw(metadata: dict[str, Any]) -> None:
        metadata["raw_artifacts"][0] = metadata["raw_artifacts"][1]

    flattened_raw_pointer = resign_dataset(
        "flattened-raw-v0",
        metadata_mutation=corrupt_flattened_raw,
    )
    with pytest.raises(DatasetValidationError, match="flattened Raw identities"):
        validate_internal_dataset(store, flattened_raw_pointer)

    catalog_source = store.validate(plan.raw_catalog_spec.pointer)
    catalog_spec = plan.raw_catalog_spec.with_producer_version("catalog-date-role-v0")
    catalog_document = json.loads(
        (catalog_source.path / "raw-catalog.json").read_text(encoding="utf-8")
    )
    catalog_document["date_roles"].pop()
    with store.writer(catalog_spec, metadata=catalog_source.manifest["metadata"]) as builder:
        builder.write_json("raw-catalog.json", catalog_document)
        builder.copy_file("README.md", catalog_source.path / "README.md")
    dataset_spec = ArtifactSpec.create(
        storage_class=plan.dataset_spec.storage_class,
        namespace=plan.dataset_spec.namespace,
        configuration=plan.dataset_spec.configuration,
        schema_version=plan.dataset_spec.schema_version,
        producer_version="dataset-catalog-date-role-v0",
        upstream=(catalog_spec.pointer,),
        distribution=plan.dataset_spec.distribution,
        source_generation=plan.dataset_spec.source_generation,
    )

    def point_lineage_at_mutated_catalog(document: dict[str, Any]) -> None:
        document["raw_catalog"] = catalog_spec.pointer.as_dict()

    def point_metadata_at_mutated_catalog(metadata: dict[str, Any]) -> None:
        metadata["raw_catalog"] = catalog_spec.pointer.as_dict()

    catalog_pointer = resign_dataset(
        "unused",
        spec=dataset_spec,
        document_mutations={"lineage.json": point_lineage_at_mutated_catalog},
        metadata_mutation=point_metadata_at_mutated_catalog,
    )
    with pytest.raises(DatasetValidationError, match="date/role mapping"):
        validate_internal_dataset(store, catalog_pointer, deep_lineage=True)


def test_internal_dataset_wildcard_miss_cannot_build(tmp_path: Path) -> None:
    locations = _locations(tmp_path / "wildcard-dataset-miss")
    with pytest.raises(WildcardReuseMissError, match="may reuse only"):
        build_internal_dataset(
            locations,
            _dataset_config(),
            DatasetBuildOptions(version_override="*"),
            dry_run=True,
        )


def test_internal_dataset_wildcard_reuses_historical_producer_and_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path / "historical-wildcard")
    store = ArtifactStore(locations)
    trading_date = date(2024, 1, 3)
    _publish_raw_inputs(store, trading_date)
    config = replace(_dataset_config(trading_date), name="historical-wildcard")
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "d" * 40)
    current = build_internal_dataset(locations, config)
    source = store.validate(current.artifact)
    historical_spec = plan_internal_dataset(config).dataset_spec.with_producer_version(
        "historical-dataset-producer-v0"
    )
    validation = json.loads(
        (source.path / "validation.json").read_text(encoding="utf-8")
    )
    validation["validator_version"] = "historical-validator-v0"
    validation_payload = (canonical_json(validation) + "\n").encode("utf-8")
    metadata = dict(source.manifest["metadata"])
    metadata["producer_version"] = historical_spec.producer_version
    metadata["evidence_checksums"] = dict(metadata["evidence_checksums"])
    metadata["evidence_checksums"]["validation.json"] = hashlib.sha256(
        validation_payload
    ).hexdigest()
    with store.writer(historical_spec, metadata=metadata) as builder:
        for record in source.manifest["files"]:
            relative = str(record["path"])
            if relative == "validation.json":
                builder.write_bytes(relative, validation_payload)
            elif relative == "README.md":
                readme = (source.path / relative).read_text(encoding="utf-8").replace(
                    current.artifact.artifact_id,
                    historical_spec.artifact_id,
                )
                builder.write_text(relative, readme)
            else:
                builder.copy_file(relative, source.path / relative)
    shutil.rmtree(source.path)

    with pytest.warns(RuntimeWarning, match="version='\\*'"):
        reused = build_internal_dataset(
            locations,
            config,
            DatasetBuildOptions(version_override="*"),
        )
    assert not reused.built
    assert reused.artifact == historical_spec.pointer
    assert reused.reuse is not None and reused.reuse.used_wildcard
    assert reused.reuse.reused_producer_version == historical_spec.producer_version
    assert reused.validation is not None and reused.validation.outcome == "passed"


def test_internal_dataset_accepts_only_policy_owned_typed_empty_days() -> None:
    bad_date = date(2025, 1, 8)
    config = replace(_dataset_config(), start_date=bad_date, end_date=bad_date)
    statistics = internal_dataset_module.validate_fitted_partition(
        pl.DataFrame(schema=FITTED_SCHEMA),
        config=config,
        trading_date=bad_date,
        typed_empty_reason="known_bad_day_all_tickers",
    )
    assert statistics.typed_empty
    with pytest.raises(
        internal_dataset_module.DatasetValidationError,
        match="not justified",
    ):
        internal_dataset_module.validate_fitted_partition(
            pl.DataFrame(schema=FITTED_SCHEMA),
            config=_dataset_config(),
            trading_date=date(2024, 1, 3),
            typed_empty_reason=None,
        )
