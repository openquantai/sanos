from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs

from sanos.artifacts import ArtifactStore, WildcardReuseMissError
from sanos.config import StorageLocations, canonical_sha256
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
    count_business_days_between,
    enrich_daily_frame,
    previous_business_day,
)
from sanos.widb.build import (
    IntermediateBuildRequest,
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


def _raw_frames(trading_date: date, expiry: date) -> dict[RawTable | str, pl.DataFrame]:
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
                        0.02 if side == "P" and index == len(strikes) - 1 else price - 0.05
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
        RawTable.SECURITY: pl.DataFrame({"ticker": ["SPX"], "secid": [108105]}),
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


def _publish_raw_inputs(store: ArtifactStore, trading_date: date) -> None:
    frames = _raw_frames(trading_date, date(2024, 1, 19))
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
            RawPartition(table, "SPX", observation_date),
            source="optionmetrics",
            source_generation="default-v1",
            producer_version=producer,
        ).spec
        if store.resolve(spec) is not None:
            continue
        with store.writer(spec, metadata={"test_fixture": True}) as builder:
            frame.write_parquet(builder.staging_path / "data.parquet")


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
