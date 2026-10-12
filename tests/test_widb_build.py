from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs

from sanos.artifacts import ArtifactStore, WildcardReuseMissError
from sanos.config import StorageLocations
from sanos.widb import (
    ENRICHED_SCHEMA,
    PAIRED_SCHEMA,
    ClockConfig,
    DailyWidbConfig,
    count_business_days_between,
    enrich_daily_frame,
)
from sanos.widb.build import (
    IntermediateBuildRequest,
    build_intermediates,
    enriched_day_producer_version,
    enriched_day_spec,
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
        "previous_spot": pl.DataFrame({"secid": [108105], "date": ["2024-01-02"], "close": [99.0]}),
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
        (RawTable.SPOT, date(2024, 1, 2), frames["previous_spot"]),
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
