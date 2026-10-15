"""Public-Dataset-only DySANOS materialization tests."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sanos.artifacts import ArtifactStore
from sanos.datasets.public import (
    PublicBuildOptions,
    build_public_candidate,
    open_public_dataset,
)
from sanos.models.dysanos import (
    ExperimentBuildOptions,
    ExperimentConfig,
    SourceDatasetConfig,
    build_experiment,
    exact_weekly_pairs,
    open_experiment,
    plan_experiment,
    project_daily_front,
)
from sanos.models.dysanos.artifacts import _prepare_shards
from sanos.models.dysanos.hdf5 import (
    ExperimentValidationError,
    _h5py,
    read_day_shard,
    validate_experiment_hdf5,
)
from sanos.pipeline import RunJournal
from tests.test_public_dataset import _config as _public_config
from tests.test_public_dataset import _publish_source


def _config(source) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-dysanos-1.1",
        source_dataset=SourceDatasetConfig(
            namespace=source.namespace,
            artifact_id=source.artifact_id,
        ),
    )


def _publish_public(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    locations, internal = _publish_source(tmp_path, monkeypatch)
    public = build_public_candidate(
        locations,
        _public_config(internal),
        PublicBuildOptions(),
    )
    return locations, internal, public


def _first_day_frame(locations, public) -> pl.DataFrame:
    frame = open_public_dataset(ArtifactStore(locations), public.artifact).load()
    first = frame.get_column("trading_date")[0]
    return frame.filter(pl.col("trading_date") == first)


def _with_second_expiry(frame: pl.DataFrame) -> pl.DataFrame:
    """Make a scalar-schema two-expiry projection fixture with per-expiry book indices."""

    expiry = frame.filter(pl.col("record_kind") == "expiry").with_columns(
        pl.lit(1, dtype=pl.Int32).alias("expiry_index"),
        (pl.col("expiry_date") + pl.duration(days=7)).alias("expiry_date"),
        (2.0 * pl.col("business_dte")).alias("business_dte"),
        (pl.col("calendar_dte") + 7).alias("calendar_dte"),
        (2.0 * pl.col("time_to_expiry_years")).alias("time_to_expiry_years"),
    )
    books = frame.filter(pl.col("record_kind") == "book").with_columns(
        pl.lit(1, dtype=pl.Int32).alias("expiry_index")
    )
    smooth = frame.filter(pl.col("record_kind") == "smooth_node").with_columns(
        pl.lit(1, dtype=pl.Int32).alias("expiry_index"),
        (2.0 * pl.col("kernel_input_total_variance")).alias(
            "kernel_input_total_variance"
        ),
    )
    dlv = frame.filter(pl.col("record_kind") == "dlv_node").with_columns(
        pl.lit(1, dtype=pl.Int32).alias("expiry_index"),
        (2.0 * pl.col("input_total_variance")).alias("input_total_variance"),
    )
    return pl.concat((frame, expiry, books, smooth, dlv), how="vertical")


def test_public_fit_and_half_spread_are_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, _, public = _publish_public(tmp_path, monkeypatch)
    config = _config(public.artifact)
    day = project_daily_front(
        _with_second_expiry(_first_day_frame(locations, public)),
        ticker=config.ticker,
        source_dataset=public.artifact,
        market=config.market,
        geometry=config.geometry,
    )
    np.testing.assert_array_equal(day.quote_bid, day.quote_fit - day.quote_half_spread)
    np.testing.assert_array_equal(day.quote_ask, day.quote_fit + day.quote_half_spread)
    assert day.expiry_count >= 2
    assert np.all(np.diff(day.quote_expiry_index) >= 0)
    for expiry_index in range(day.expiry_count):
        selected = day.quote_expiry_index == expiry_index
        sides = day.quote_is_call[selected]
        strikes = day.quote_strike[selected]
        assert np.all(sides[:-1] | ~sides[1:])
        for side in (True, False):
            side_strikes = strikes[sides == side]
            assert np.all(np.diff(side_strikes) > 0.0)
    np.testing.assert_allclose(
        day.quote_fit[~day.quote_is_call],
        day.call_equivalent_fit_prices[~day.quote_is_call] - 1.0
        + day.quote_strike[~day.quote_is_call],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_negative_synthetic_bid_is_not_clipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, _, public = _publish_public(tmp_path, monkeypatch)
    config = _config(public.artifact)
    frame = _first_day_frame(locations, public).with_row_index("_row")
    book_row = int(frame.filter(pl.col("record_kind") == "book")["_row"][0])
    frame = frame.with_columns(
        pl.when(pl.col("_row") == book_row)
        .then(pl.lit(10.0))
        .otherwise(pl.col("half_spread"))
        .alias("half_spread")
    ).drop("_row")
    day = project_daily_front(
        frame,
        ticker=config.ticker,
        source_dataset=public.artifact,
        market=config.market,
        geometry=config.geometry,
    )
    assert np.any(day.quote_bid < 0.0)


def test_hdf5_omits_private_fields_and_masks_first_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, _, public = _publish_public(tmp_path, monkeypatch)
    config = _config(public.artifact)
    public_days = (
        open_public_dataset(ArtifactStore(locations), public.artifact)
        .load()
        .filter(pl.col("record_kind") == "day")
        .sort("trading_date")
    )
    public_returns = public_days.get_column("driftless_spot_return").to_list()
    assert len(public_returns) == 2 and public_returns[0] is None
    result = build_experiment(locations, config, ExperimentBuildOptions(max_workers=1))
    experiment = open_experiment(ArtifactStore(locations), result.artifact)
    first_day = experiment.load_day(0)
    second_day = experiment.load_day(1)
    assert not first_day.driftless_spot_return_valid
    assert np.isnan(first_day.driftless_spot_return)
    assert second_day.driftless_spot_return_valid
    assert second_day.driftless_spot_return == public_returns[1]
    stored_returns, return_valid = experiment.driftless_spot_returns()
    np.testing.assert_array_equal(return_valid, np.asarray([False, True]))
    assert stored_returns[0] == 0.0
    assert stored_returns[1] == public_returns[1]
    np.testing.assert_array_equal(
        experiment.driftless_spot_path(),
        np.asarray([1.0, 1.0 + float(public_returns[1])]),
    )
    with _h5py().File(result.path / "data.h5", "r") as handle:
        paths: set[str] = set()
        handle.visit(paths.add)
        assert {
            "quote/fit",
            "quote/half_spread",
            "day/driftless_spot_return",
            "day/driftless_spot_return_valid",
        } <= paths
        assert not {
            "day/spot_ref",
            "expiry/id",
            "quote/bid",
            "quote/ask",
            "quote/log_volume",
            "quote/log_open_interest",
        }.intersection(paths)
    with _h5py().File(result.path / "data.h5", "r+") as handle:
        original_return = float(handle["day/driftless_spot_return"][1])
    for invalid_return in (-1.0, -1.01):
        with _h5py().File(result.path / "data.h5", "r+") as handle:
            handle["day/driftless_spot_return"][1] = invalid_return
        with pytest.raises(
            ExperimentValidationError,
            match="Driftless spot returns",
        ):
            validate_experiment_hdf5(
                result.path / "data.h5",
                config=config,
                producer_version=result.plan.spec.producer_version,
            )
    with _h5py().File(result.path / "data.h5", "r+") as handle:
        handle["day/driftless_spot_return"][1] = original_return
    with _h5py().File(result.path / "data.h5", "r+") as handle:
        handle["quote/half_spread"][0, 0] += 1.0e-6
    with pytest.raises(
        ExperimentValidationError,
        match="Market fingerprint|logical content digest",
    ):
        validate_experiment_hdf5(
            result.path / "data.h5",
            config=config,
            producer_version=result.plan.spec.producer_version,
        )


def test_public_restart_shards_reuse_and_rebuild_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, _, public = _publish_public(tmp_path, monkeypatch)
    config = _config(public.artifact)
    plan = plan_experiment(config)
    dataset = open_public_dataset(ArtifactStore(locations), public.artifact)
    workspace = tmp_path / "restart-workspace"
    journal = RunJournal(tmp_path / "restart.jsonl")
    journal.require_plan(plan.plan_id)
    options = ExperimentBuildOptions(max_workers=1, progress_interval_seconds=0.1)

    shards, _ = _prepare_shards(
        dataset,
        plan=plan,
        workspace=workspace,
        config=config,
        options=options,
        progress=None,
        journal=journal,
    )
    fingerprints = tuple(read_day_shard(path).market_fingerprint for path in shards)
    shards[0].write_bytes(b"PK\x03\x04garbage")
    rebuilt, _ = _prepare_shards(
        dataset,
        plan=plan,
        workspace=workspace,
        config=config,
        options=options,
        progress=None,
        journal=journal,
    )
    assert tuple(read_day_shard(path).market_fingerprint for path in rebuilt) == fingerprints


def test_build_needs_only_completed_public_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, internal, public = _publish_public(tmp_path, monkeypatch)
    config = _config(public.artifact)
    shutil.rmtree(internal.path)
    shutil.rmtree(locations.raw)
    shutil.rmtree(locations.intermediate)
    monkeypatch.setenv("WRDS_OFF", "1")
    result = build_experiment(locations, config, ExperimentBuildOptions(max_workers=1))
    assert result.built and result.source_accessed and result.validation is not None


def test_completed_experiment_is_output_first_without_public_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, _, public = _publish_public(tmp_path, monkeypatch)
    config = _config(public.artifact)
    first = build_experiment(locations, config, ExperimentBuildOptions(max_workers=1))
    shutil.rmtree(public.path)
    second = build_experiment(locations, config, ExperimentBuildOptions(max_workers=1))
    assert second.artifact == first.artifact
    assert not second.source_accessed


def test_exact_weekly_pairs_remain_five_nyse_sessions() -> None:
    origin, endpoint = exact_weekly_pairs(
        (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 9), date(2024, 1, 10))
    )
    np.testing.assert_array_equal(origin, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(endpoint, np.asarray([2, 3], dtype=np.int64))
