from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from sanos.artifacts import ArtifactStore
from sanos.cli import main
from sanos.config import canonical_json
from sanos.reference import (
    APPROVAL_SCHEMA,
    DailyComparisonRequest,
    DailyLegacyComparisonTolerances,
    ReferenceComparisonError,
    catalog_legacy_market,
    compare_daily_artifacts,
    compare_populated_day,
    load_comparison_approvals,
    load_oracle_gap_declaration,
    load_reference_catalog,
    write_comparison_report,
)
from sanos.widb import DailySurfaceConfig, DailyWidbConfig, smooth_sanos_pricer
from sanos.widb.build import (
    IntermediateBuildRequest,
    build_intermediates,
    plan_daily_artifacts,
    read_daily_frame,
)
from sanos.widb.surface import FITTED_SCHEMA, selected_reference_view
from sanos.wrds import (
    RawPartition,
    RawTable,
    raw_artifact_request,
    raw_content_producer_version,
)
from tests.test_reference_legacy_market import _write_legacy
from tests.test_widb_build import _locations, _publish_raw_inputs


def _built_fitted_day(tmp_path: Path) -> tuple[Any, pl.DataFrame]:
    return _built_fitted_day_at(tmp_path, date(2024, 1, 3))


def _built_fitted_day_at(
    tmp_path: Path,
    trading_date: date,
    *,
    expiry: date = date(2024, 1, 19),
) -> tuple[Any, pl.DataFrame]:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    _publish_raw_inputs(store, trading_date, expiry=expiry)
    request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=trading_date,
        end_date=trading_date,
        through="fitted-day",
    )
    result = build_intermediates(
        locations,
        request,
        DailyWidbConfig(),
        DailySurfaceConfig(),
    )
    artifact = store.validate(result.terminal[0])
    frame = read_daily_frame(
        store,
        artifact,
        schema=FITTED_SCHEMA,
        ticker="SPX",
        trading_date=trading_date,
    )
    return locations, frame


def _constant(group: pl.DataFrame, name: str, dtype: Any) -> np.ndarray[Any, Any]:
    values = group.get_column(name).to_list()
    assert values and values[0] is not None
    assert all(value == values[0] for value in values)
    return np.asarray(values[0], dtype=dtype)


def _legacy_payload(frame: pl.DataFrame) -> dict[str, object]:
    view = selected_reference_view(frame)
    groups = view.partition_by("expiry_id", maintain_order=True)
    retained = [
        group
        for group in groups
        if group.get_column("smooth_input_expiry_index").null_count() == 0
    ]
    fresh_pricer = smooth_sanos_pricer(frame)
    mappings: list[np.ndarray[Any, Any] | str] = []
    for group in retained:
        values = group.get_column("smooth_x_to_surface_indices").to_list()
        assert values and all(value == values[0] for value in values)
        mappings.append(
            "__ARB_FREE_PRICER_NONE__"
            if values[0] is None
            else np.asarray(values[0], dtype=np.int32)
        )
    af_pricer = {
        "smoothness": float(frame.get_column("smoothness").item(0)),
        "expiry_linear": True,
        "min_half_spread": float(frame.get_column("surface_min_half_spread").item(0)),
        "expiries": np.asarray(fresh_pricer.expiries, dtype=np.float64),
        "variances": np.asarray(fresh_pricer.variances, dtype=np.float64),
        "input_index": np.asarray(
            [group.get_column("smooth_input_expiry_index").item(0) for group in retained],
            dtype=np.int32,
        ),
        "masks": [group.get_column("smooth_fit_mask").to_numpy() for group in retained],
        "strikes": [group.get_column("normalized_strike").to_numpy() for group in retained],
        "bids": [group.get_column("pure_bid_price").to_numpy() for group in retained],
        "asks": [group.get_column("pure_ask_price").to_numpy() for group in retained],
        "fits1": [
            group.get_column("smooth_expiry_fit_price").to_numpy() for group in retained
        ],
        "fitsX": [group.get_column("smooth_call_mid").to_numpy() for group in retained],
        "xstrikes": [
            _constant(group, "smooth_expiry_model_strikes", np.float32)
            for group in retained
        ],
        "xdensities": [
            _constant(group, "smooth_expiry_model_density", np.float64)
            for group in retained
        ],
        "xix_to_mix": mappings,
        "mstrikes": [
            _constant(group, "smooth_surface_model_strikes", np.float32)
            for group in retained
        ],
        "mdensities": [
            _constant(group, "smooth_surface_model_density", np.float64)
            for group in retained
        ],
        "run_time_expiry": 0.0,
        "run_time_surface": 0.0,
    }
    market_map = {
        "strikes": "normalized_strike",
        "bids": "pure_bid_price",
        "asks": "pure_ask_price",
        "is_american": "is_american",
        "pre_af_masks": "mask_arb_free",
        "model_masks": "smooth_model_mask",
        "call_bids": "pure_market_call_bid",
        "call_asks": "pure_market_call_ask",
        "put_bids": "pure_market_put_bid",
        "put_asks": "pure_market_put_ask",
        "call_log_volumes": "market_call_log_volume",
        "call_log_open_interests": "market_call_log_open_interest",
        "put_log_volumes": "market_put_log_volume",
        "put_log_open_interests": "market_put_log_open_interest",
        "call_active": "call_active",
        "put_active": "put_active",
        "af_call_mids": "smooth_call_mid",
        "af_put_mids": "smooth_put_mid",
        "af_mid_available": "smooth_mid_available",
    }
    market: dict[str, object] = {
        "uid": "synthetic-reference",
        "expiries": np.asarray(
            [group.get_column("time_to_expiry_years").item(0) for group in groups],
            dtype=np.float32,
        ),
        "expiry_dates": [group.get_column("expiry_date").item(0).isoformat() for group in groups],
        "expiry_ids": np.asarray(
            [group.get_column("expiry_id").item(0) for group in groups], dtype=np.int64
        ),
        "is_am": np.asarray(
            [group.get_column("expiry_is_am").item(0) for group in groups], dtype=bool
        ),
    }
    for legacy_name, fresh_name in market_map.items():
        market[legacy_name] = [group.get_column(fresh_name).to_numpy() for group in groups]
    return {
        "schema_version": 4,
        "ticker": "SPX",
        "d": frame.get_column("trading_date").item(0),
        "spot_ref": float(np.float32(view.get_column("spot_reference").item(0))),
        "spot_ret": float(np.float32(view.get_column("spot_return").item(0))),
        "market": market,
        "af_pricer": af_pricer,
        "fit_result": {
            "fits": [group.get_column("smooth_call_mid").to_numpy() for group in groups],
            "fit_errs": [
                group.get_column("smooth_fit_error_half_spreads").to_numpy()
                for group in groups
            ],
            "pcnt_fit": float(view.get_column("smooth_pcnt_fit").item(0)),
            "median_err": float(view.get_column("smooth_median_error").item(0)),
            "mean_err": float(view.get_column("smooth_mean_error").item(0)),
            "median_total_err": float(
                view.get_column("smooth_median_total_error").item(0)
            ),
            "mean_total_err": float(view.get_column("smooth_mean_total_error").item(0)),
        },
    }


@pytest.fixture(scope="module")
def comparison_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pl.DataFrame, dict[str, object]]:
    _, frame = _built_fitted_day(tmp_path_factory.mktemp("comparison-pair"))
    return frame, _legacy_payload(frame)


def test_populated_comparison_exact_match_and_fresh_only_dlv_gap(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, payload = comparison_pair
    result = compare_populated_day(frame, payload)

    assert result["matched"] is True
    assert result["mismatch_fields"] == []
    assert result["population"]["fresh_selected_rows"] == frame.height
    assert result["item_oracle_gaps"] == [
        {
            "item": "canonical_discrete_dlv_projection",
            "reason": "legacy daily HDF5 stores only the smooth SANOS state",
            "fresh_invariants_passed": True,
            "dlv_fixed_point_maximum_absolute_error": pytest.approx(0.0, abs=1e-9),
            "canonical_density_dlv_inverse_maximum_absolute_error": pytest.approx(
                0.0, abs=1e-12
            ),
            "density_dlv_inverse_maximum_absolute_error": pytest.approx(0.0, abs=1e-12),
        }
    ]
    fields = {
        item["field"]: item for item in result["numerical_comparisons"]
    }
    assert all("smooth_dlv" not in field for field in fields)
    assert fields["reconstruction.atm_implied_volatility"]["count"] > len(
        cast(
            np.ndarray[Any, Any],
            cast(dict[str, object], payload["af_pricer"])["expiries"],
        )
    )


def test_population_distinguishes_input_and_selected_expiries(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, original = comparison_pair
    ineligible = frame.with_columns(
        (pl.col("expiry_id") + 1).cast(pl.Int32),
        (pl.col("expiry_date") + timedelta(days=7)).cast(pl.Date),
        (pl.col("calendar_dte") + 7).cast(pl.Int32),
        (pl.col("business_dte") + 5.0).cast(pl.Float32),
        (pl.col("time_to_expiry_years") + np.float32(5.0 / 252.0)).cast(
            pl.Float32
        ),
        pl.lit(False).alias("options_filter_eligible"),
        pl.lit(None, dtype=pl.Int32).alias("initial_no_arb_input_expiry_index"),
        pl.lit(None, dtype=pl.Int32).alias("smooth_input_expiry_index"),
        pl.lit(False).alias("discrete_dlv_anchor"),
        pl.lit(None, dtype=pl.Int32).alias("discrete_dlv_input_expiry_index"),
        pl.lit(None, dtype=pl.Float64).alias("discrete_dlv_time_increment_years"),
        pl.lit(None, dtype=pl.Float64).alias("discrete_dlv_atm_annual_volatility"),
        pl.lit(None, dtype=pl.Float64).alias("discrete_dlv_input_total_variance"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_nms"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_model_strikes"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_model_density"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_squares"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_target_calls"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_fitted_calls"),
    )
    with_ineligible_expiry = pl.concat([frame, ineligible]).cast(
        FITTED_SCHEMA, strict=False
    )

    result = compare_populated_day(with_ineligible_expiry, copy.deepcopy(original))

    assert result["matched"] is True
    assert result["population"]["input_expiries"] == 2
    assert result["population"]["selected_expiries"] == 1


def test_populated_comparison_explicitly_rejects_zero_selected_fresh_day(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, payload = comparison_pair
    zero_selected = frame.with_columns(
        pl.lit(False).alias("options_filter_eligible")
    )

    with pytest.raises(ReferenceComparisonError, match="zero filter-selected rows"):
        compare_populated_day(zero_selected, payload)


def test_comparison_accepts_selected_expiry_dropped_from_smooth_state(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, _ = comparison_pair
    dropped = frame.with_columns(
        (pl.col("expiry_id") + 1).cast(pl.Int32),
        (pl.col("expiry_date") + timedelta(days=7)).cast(pl.Date),
        (pl.col("calendar_dte") + 7).cast(pl.Int32),
        (pl.col("business_dte") + 5.0).cast(pl.Float32),
        (pl.col("time_to_expiry_years") + np.float32(5.0 / 252.0)).cast(
            pl.Float32
        ),
        pl.lit(None, dtype=pl.Int32).alias("initial_no_arb_input_expiry_index"),
        pl.lit(None, dtype=pl.Int32).alias("smooth_input_expiry_index"),
        pl.lit(False).alias("discrete_dlv_anchor"),
        pl.lit(None, dtype=pl.Int32).alias("discrete_dlv_input_expiry_index"),
        pl.lit(None, dtype=pl.Float64).alias("discrete_dlv_time_increment_years"),
        pl.lit(None, dtype=pl.Float64).alias("discrete_dlv_atm_annual_volatility"),
        pl.lit(None, dtype=pl.Float64).alias("discrete_dlv_input_total_variance"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_nms"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_model_strikes"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_model_density"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_squares"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_target_calls"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("discrete_dlv_fitted_calls"),
    )
    with_dropped_expiry = pl.concat([frame, dropped]).cast(
        FITTED_SCHEMA, strict=False
    )
    payload = _legacy_payload(with_dropped_expiry)

    result = compare_populated_day(with_dropped_expiry, payload)

    assert result["matched"] is True
    assert result["population"]["input_expiries"] == 2
    assert result["population"]["selected_expiries"] == 2
    assert result["population"]["smooth_retained_expiries"] == 1
    assert result["population"]["dropped_expiry_ids"] == [
        int(frame.get_column("expiry_id").item(0)) + 1
    ]


def test_comparison_accepts_smooth_state_with_no_active_books(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, _ = comparison_pair
    inactive = frame.with_columns(
        pl.lit(False).alias("call_active"),
        pl.lit(False).alias("put_active"),
    ).cast(FITTED_SCHEMA, strict=False)

    result = compare_populated_day(inactive, _legacy_payload(inactive))

    assert result["matched"] is True
    assert result["population"]["active_call_books"] == 0
    assert result["population"]["active_put_books"] == 0


def test_float64_difference_inside_tolerance_remains_visible(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, original = comparison_pair
    payload = copy.deepcopy(original)
    fit_result = cast(dict[str, object], payload["fit_result"])
    fit_result["mean_err"] = float(fit_result["mean_err"]) + 5.0e-13

    result = compare_populated_day(frame, payload)

    assert result["matched"] is True
    metric = next(
        item
        for item in result["numerical_comparisons"]
        if item["field"] == "fit_result.mean_err"
    )
    assert metric["nonzero_difference_count"] == 1
    assert metric["above_tolerance_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected_family"),
    [
        ("date", "identity"),
        ("order", "structure"),
        ("legacy_order", "structure"),
        ("legacy_key", "structure"),
        ("dtype", "market"),
        ("state_dtype", "smooth_state"),
        ("mask_dtype", "mask"),
        ("mapping_dtype", "smooth_state"),
        ("shape", "market"),
        ("population", "population"),
        ("mask", "mask"),
        ("null", "market"),
        ("numeric", "fit_diagnostics"),
    ],
)
def test_structural_and_numerical_families_block_match(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
    mutation: str,
    expected_family: str,
) -> None:
    frame, original = comparison_pair
    payload = copy.deepcopy(original)
    mutated_frame = frame
    market = cast(dict[str, object], payload["market"])
    if mutation == "date":
        payload["d"] = date(2024, 1, 4)
    elif mutation == "order":
        mutated_frame = frame.reverse().cast(FITTED_SCHEMA, strict=False)
    elif mutation == "legacy_order":
        values = cast(list[np.ndarray[Any, Any]], market["strikes"])
        values[0][[0, 1]] = values[0][[1, 0]]
    elif mutation == "legacy_key":
        values = cast(list[np.ndarray[Any, Any]], market["strikes"])
        values[0][1] = values[0][0]
    elif mutation == "dtype":
        values = cast(list[np.ndarray[Any, Any]], market["strikes"])
        values[0] = values[0].astype(np.float64)
    elif mutation == "state_dtype":
        state = cast(dict[str, object], payload["af_pricer"])
        state["input_index"] = cast(np.ndarray[Any, Any], state["input_index"]).astype(
            np.int64
        )
    elif mutation == "mask_dtype":
        values = cast(list[np.ndarray[Any, Any]], market["call_active"])
        values[0] = values[0].astype(np.int8)
    elif mutation == "mapping_dtype":
        state = cast(dict[str, object], payload["af_pricer"])
        mappings = cast(list[np.ndarray[Any, Any] | str], state["xix_to_mix"])
        mappings[0] = np.asarray([], dtype=np.int64)
    elif mutation == "shape":
        values = cast(list[np.ndarray[Any, Any]], market["call_bids"])
        values[0] = values[0][:-1]
    elif mutation == "population":
        values = cast(list[np.ndarray[Any, Any]], market["call_bids"])
        values.clear()
    elif mutation == "mask":
        values = cast(list[np.ndarray[Any, Any]], market["call_active"])
        values[0][0] = ~values[0][0]
    elif mutation == "null":
        values = cast(list[np.ndarray[Any, Any]], market["call_bids"])
        values[0][0] = np.nan
    else:
        fit_result = cast(dict[str, object], payload["fit_result"])
        fit_result["mean_err"] = float(fit_result["mean_err"]) + 2.0e-12

    result = compare_populated_day(mutated_frame, payload)

    assert result["matched"] is False
    assert any(item["family"] == expected_family for item in result["mismatches"])


def test_tolerance_contract_is_versioned_and_frozen() -> None:
    tolerance = DailyLegacyComparisonTolerances()
    assert str(DailyLegacyComparisonTolerances.version).startswith(
        "phase10-legacy-daily-comparison-tolerances-v3"
    )
    assert tolerance.to_dict()["float32_atol"] == 0.0
    assert tolerance.to_dict()["float64_atol"] == 1.0e-12
    assert tolerance.to_dict()["reconstructed_call_atol"] == 5.0e-8
    assert tolerance.to_dict()["density_dlv_inverse_atol"] == 1.0e-12
    assert tolerance.to_dict()["dlv_fixed_point_atol"] == 1.0e-9
    with pytest.raises(ValueError, match="frozen"):
        DailyLegacyComparisonTolerances(float32_atol=1.0)
    with pytest.raises(ValueError, match="frozen"):
        DailyLegacyComparisonTolerances(float64_atol=float("nan"))


@pytest.mark.parametrize(
    ("label", "dtype", "atol"),
    [
        ("persisted_float32", np.dtype("float32"), 0.0),
        ("persisted_float64", np.dtype("float64"), 1.0e-12),
        ("reconstructed_call", np.dtype("float64"), 5.0e-8),
        ("density_dlv_inverse", np.dtype("float64"), 1.0e-12),
    ],
)
def test_numerical_contract_at_and_beyond_each_tolerance(
    label: str,
    dtype: np.dtype[Any],
    atol: float,
) -> None:
    compare_module = importlib.import_module("sanos.reference.compare")
    at_value = dtype.type(atol)
    at_mismatches: list[dict[str, object]] = []
    at_numerical: list[dict[str, object]] = []
    compare_module._compare_numeric(
        label,
        np.asarray([at_value], dtype=dtype),
        np.asarray([dtype.type(0.0)], dtype=dtype),
        fresh_dtype=dtype,
        legacy_dtype=dtype,
        comparison_dtype=dtype,
        atol=float(at_value),
        keys=["one"],
        family="test",
        mismatches=at_mismatches,
        numerical=at_numerical,
    )
    assert at_mismatches == []
    assert at_numerical[0]["above_tolerance_count"] == 0
    assert at_numerical[0]["nonzero_difference_count"] == int(atol > 0.0)

    beyond_value = np.nextafter(at_value, dtype.type(np.inf), dtype=dtype)
    beyond_mismatches: list[dict[str, object]] = []
    beyond_numerical: list[dict[str, object]] = []
    compare_module._compare_numeric(
        label,
        np.asarray([beyond_value], dtype=dtype),
        np.asarray([dtype.type(0.0)], dtype=dtype),
        fresh_dtype=dtype,
        legacy_dtype=dtype,
        comparison_dtype=dtype,
        atol=float(at_value),
        keys=["one"],
        family="test",
        mismatches=beyond_mismatches,
        numerical=beyond_numerical,
    )
    assert beyond_mismatches[0]["kind"] == "numerical"
    assert beyond_numerical[0]["above_tolerance_count"] == 1


def _catalog(root: Path, output: Path) -> None:
    catalog = catalog_legacy_market(
        root,
        source_revision="legacy-test-revision",
        relative_path="synthetic/SPX",
    )
    output.write_text(catalog.to_json(), encoding="utf-8")


def test_existing_artifact_orchestrator_matches_and_binds_checksums(tmp_path: Path) -> None:
    locations, frame = _built_fitted_day(tmp_path / "data")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    _write_legacy(legacy_root / "2024-01-03.h5", _legacy_payload(frame))
    catalog_path = tmp_path / "catalog.json"
    _catalog(legacy_root, catalog_path)

    report = compare_daily_artifacts(
        locations,
        DailyComparisonRequest(
            ticker="SPX",
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 3),
            source="optionmetrics",
            source_generation="default-v1",
        ),
        daily_config=DailyWidbConfig(),
        surface_config=DailySurfaceConfig(),
        legacy_root=legacy_root,
        catalog_path=catalog_path,
        source_revision="a" * 40,
        require_pinned_catalog=False,
    )

    assert report["outcome"] == "passed"
    assert report["classification_counts"] == {"match": 1}
    assert len(report["legacy_catalog"]["selected_files"]) == 1
    assert report["dates"][0]["fresh"]["data_sha256"]
    assert report["dates"][0]["legacy"]["sha256"]
    assert report["dates"][0]["observations"]["populated"] is True
    assert report["pilot_contract"] == {
        "daily_checks_passed": True,
        "enabled": False,
        "passed": True,
    }

    repeated = compare_daily_artifacts(
        locations,
        DailyComparisonRequest(
            ticker="SPX",
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 3),
            source="optionmetrics",
            source_generation="default-v1",
        ),
        daily_config=DailyWidbConfig(),
        surface_config=DailySurfaceConfig(),
        legacy_root=legacy_root,
        catalog_path=catalog_path,
        source_revision="a" * 40,
        require_pinned_catalog=False,
    )
    assert repeated["report_id"] == report["report_id"]


def test_expected_and_unexpected_oracle_gap_classification(tmp_path: Path) -> None:
    locations, _ = _built_fitted_day(tmp_path / "data")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    _write_legacy(legacy_root / "2024-01-04.h5", {})
    catalog_path = tmp_path / "catalog.json"
    _catalog(legacy_root, catalog_path)
    base = dict(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        source="optionmetrics",
        source_generation="default-v1",
    )

    expected = compare_daily_artifacts(
        locations,
        DailyComparisonRequest(**base, declared_oracle_gaps=(date(2024, 1, 3),)),
        daily_config=DailyWidbConfig(),
        surface_config=DailySurfaceConfig(),
        legacy_root=legacy_root,
        catalog_path=catalog_path,
        source_revision="b" * 40,
        require_pinned_catalog=False,
    )
    unexpected = compare_daily_artifacts(
        locations,
        DailyComparisonRequest(**base),
        daily_config=DailyWidbConfig(),
        surface_config=DailySurfaceConfig(),
        legacy_root=legacy_root,
        catalog_path=catalog_path,
        source_revision="b" * 40,
        require_pinned_catalog=False,
    )

    assert expected["outcome"] == "passed"
    assert expected["classification_counts"] == {"oracle_gap": 1}
    assert unexpected["outcome"] == "blocked"
    assert unexpected["classification_counts"] == {"unexplained_discrepancy": 1}


def test_approval_document_rejects_broadened_and_tampered_records(tmp_path: Path) -> None:
    approval = {
        "date": "2024-01-03",
        "fields": ["market.bids[0]"],
        "fresh_artifact_id": "a" * 64,
        "fresh_data_sha256": "b" * 64,
        "legacy_file_sha256": "c" * 64,
        "explanation": "Reviewed vendor revision evidence",
        "approver": "owner",
        "approval_date": "2026-10-13",
        "approval_reference": "decision://phase10/example",
    }
    path = tmp_path / "approvals.json"
    path.write_text(
        json.dumps({"schema": APPROVAL_SCHEMA, "approvals": [approval]}),
        encoding="utf-8",
    )
    assert load_comparison_approvals(path)["2024-01-03"] == approval

    broadened = copy.deepcopy(approval)
    broadened["fields"] = ["market.bids[0]", "market.strikes[0]"]
    broadened["fields"].reverse()
    path.write_text(
        json.dumps({"schema": APPROVAL_SCHEMA, "approvals": [broadened]}),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceComparisonError, match="sorted unique"):
        load_comparison_approvals(path)


def test_approval_overlay_requires_exact_evidence_binding(tmp_path: Path) -> None:
    locations, frame = _built_fitted_day(tmp_path / "data")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    payload = _legacy_payload(frame)
    fit_result = cast(dict[str, object], payload["fit_result"])
    fit_result["mean_err"] = float(fit_result["mean_err"]) + 2.0e-12
    legacy_path = legacy_root / "2024-01-03.h5"
    _write_legacy(legacy_path, payload)
    catalog_path = tmp_path / "catalog.json"
    _catalog(legacy_root, catalog_path)
    request = DailyComparisonRequest(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        source="optionmetrics",
        source_generation="default-v1",
    )
    arguments = {
        "daily_config": DailyWidbConfig(),
        "surface_config": DailySurfaceConfig(),
        "legacy_root": legacy_root,
        "catalog_path": catalog_path,
        "source_revision": "e" * 40,
        "require_pinned_catalog": False,
    }

    blocked = compare_daily_artifacts(locations, request, **arguments)
    assert blocked["outcome"] == "blocked"
    day = blocked["dates"][0]
    approval = {
        "date": "2024-01-03",
        "fields": day["comparison"]["mismatch_fields"],
        "fresh_artifact_id": day["fresh"]["artifact"]["artifact_id"],
        "fresh_data_sha256": day["fresh"]["data_sha256"],
        "legacy_file_sha256": day["legacy"]["sha256"],
        "explanation": "Owner-reviewed exact synthetic discrepancy",
        "approver": "owner",
        "approval_date": "2026-10-13",
        "approval_reference": "decision://phase10/synthetic",
    }
    approvals_path = tmp_path / "approvals.json"
    approvals_path.write_text(
        json.dumps({"schema": APPROVAL_SCHEMA, "approvals": [approval]}),
        encoding="utf-8",
    )

    approved = compare_daily_artifacts(
        locations,
        request,
        **arguments,
        approvals_path=approvals_path,
    )
    assert approved["outcome"] == "passed"
    assert approved["classification_counts"] == {"approved_discrepancy": 1}

    approval["fresh_data_sha256"] = "f" * 64
    approvals_path.write_text(
        json.dumps({"schema": APPROVAL_SCHEMA, "approvals": [approval]}),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceComparisonError, match="stale, broadened"):
        compare_daily_artifacts(
            locations,
            request,
            **arguments,
            approvals_path=approvals_path,
        )


def test_policy_empty_and_population_mismatch_classifications(tmp_path: Path) -> None:
    empty_locations, empty_frame = _built_fitted_day_at(
        tmp_path / "empty-data",
        date(2025, 1, 8),
        expiry=date(2025, 1, 17),
    )
    assert empty_frame.is_empty()
    empty_legacy = tmp_path / "empty-legacy"
    empty_legacy.mkdir()
    _write_legacy(empty_legacy / "2025-01-08.h5", {})
    empty_catalog = tmp_path / "empty-catalog.json"
    _catalog(empty_legacy, empty_catalog)
    empty_request = DailyComparisonRequest(
        ticker="SPX",
        start_date=date(2025, 1, 8),
        end_date=date(2025, 1, 8),
        source="optionmetrics",
        source_generation="default-v1",
    )
    arguments = {
        "daily_config": DailyWidbConfig(),
        "surface_config": DailySurfaceConfig(),
        "legacy_root": empty_legacy,
        "catalog_path": empty_catalog,
        "source_revision": "1" * 40,
        "require_pinned_catalog": False,
    }

    policy_empty = compare_daily_artifacts(empty_locations, empty_request, **arguments)
    assert policy_empty["classification_counts"] == {"policy_empty": 1}
    assert policy_empty["dates"][0]["fresh"]["statistics"]["typed_empty_reason"] == (
        "known_bad_day_all_tickers"
    )

    populated_payload = _legacy_payload(comparison_pair_from_frame(tmp_path / "populated"))
    populated_payload["d"] = date(2025, 1, 8)
    _write_legacy(empty_legacy / "2025-01-08.h5", populated_payload)
    _catalog(empty_legacy, empty_catalog)
    fresh_empty_legacy_populated = compare_daily_artifacts(
        empty_locations,
        empty_request,
        **arguments,
    )
    assert fresh_empty_legacy_populated["classification_counts"] == {
        "unexplained_discrepancy": 1
    }


def comparison_pair_from_frame(tmp_path: Path) -> pl.DataFrame:
    """Return a populated fitted fixture for population-mismatch coverage."""

    _, frame = _built_fitted_day(tmp_path)
    return frame


def test_empty_legacy_against_populated_fresh_is_unexplained(tmp_path: Path) -> None:
    locations, _ = _built_fitted_day(tmp_path / "data")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    _write_legacy(legacy_root / "2024-01-03.h5", {})
    catalog = tmp_path / "catalog.json"
    _catalog(legacy_root, catalog)

    report = compare_daily_artifacts(
        locations,
        DailyComparisonRequest(
            ticker="SPX",
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 3),
            source="optionmetrics",
            source_generation="default-v1",
        ),
        daily_config=DailyWidbConfig(),
        surface_config=DailySurfaceConfig(),
        legacy_root=legacy_root,
        catalog_path=catalog,
        source_revision="2" * 40,
        require_pinned_catalog=False,
    )
    assert report["classification_counts"] == {"unexplained_discrepancy": 1}


def test_catalog_file_tamper_is_rejected_before_decode(tmp_path: Path) -> None:
    locations, frame = _built_fitted_day(tmp_path / "data")
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_path = legacy_root / "2024-01-03.h5"
    _write_legacy(legacy_path, _legacy_payload(frame))
    catalog = tmp_path / "catalog.json"
    _catalog(legacy_root, catalog)
    legacy_path.write_bytes(legacy_path.read_bytes() + b"tamper")

    with pytest.raises(ReferenceComparisonError, match="size changed"):
        compare_daily_artifacts(
            locations,
            DailyComparisonRequest(
                ticker="SPX",
                start_date=date(2024, 1, 3),
                end_date=date(2024, 1, 3),
                source="optionmetrics",
                source_generation="default-v1",
            ),
            daily_config=DailyWidbConfig(),
            surface_config=DailySurfaceConfig(),
            legacy_root=legacy_root,
            catalog_path=catalog,
            source_revision="3" * 40,
            require_pinned_catalog=False,
        )


def test_phase10_case_checks_fail_closed() -> None:
    compare_module = importlib.import_module("sanos.reference.compare")
    half_day = compare_module._phase10_pilot_day_checks(
        "2024-12-24",
        observations={
            "populated": True,
            "is_half_day": True,
            "irregular_expiry_rows_removed": 0,
            "point_two_am_active_side_books": 0,
        },
        typed_empty_reason=None,
        record=None,
    )
    assert all(check["passed"] for check in half_day)

    irregular = compare_module._phase10_pilot_day_checks(
        "2025-01-07",
        observations={
            "populated": True,
            "is_half_day": False,
            "irregular_expiry_rows_removed": 0,
            "point_two_am_active_side_books": 0,
        },
        typed_empty_reason=None,
        record={"populated": True},
        raw_case_evidence={"required_expiry_row_count": 2},
    )
    assert irregular[0]["check"] == "irregular_expiry_raw_case_present"
    assert irregular[0]["passed"] is True
    assert irregular[1]["check"] == "irregular_expiry_case_exercised"
    assert irregular[1]["passed"] is False

    am = compare_module._phase10_pilot_day_checks(
        "2025-01-16",
        observations={
            "populated": True,
            "is_half_day": False,
            "irregular_expiry_rows_removed": 0,
            "point_two_am_active_side_books": 186,
        },
        typed_empty_reason=None,
        record={"populated": True, "active_point_two_day_books": 186},
    )
    assert all(check["passed"] for check in am)


def test_daily_observations_capture_irregular_and_am_cases(
    comparison_pair: tuple[pl.DataFrame, dict[str, object]],
) -> None:
    frame, _ = comparison_pair
    compare_module = importlib.import_module("sanos.reference.compare")
    diagnostic = json.dumps(
        {"filter_counts": {"irregular_expiry": {"removed": 4}}},
        sort_keys=True,
    )
    exercised = frame.with_columns(
        pl.lit(diagnostic).alias("normalization_diagnostics_json"),
        pl.lit(True).alias("expiry_is_am"),
        pl.lit(0.2, dtype=pl.Float32).alias("business_dte"),
    ).cast(FITTED_SCHEMA, strict=False)

    observed = compare_module._daily_observations(exercised)

    assert observed["irregular_expiry_rows_removed"] == 4
    assert observed["point_two_am_active_side_books"] == int(
        exercised.get_column("call_active").sum()
        + exercised.get_column("put_active").sum()
    )


def test_phase10_contract_pins_dates_configuration_and_catalog() -> None:
    compare_module = importlib.import_module("sanos.reference.compare")
    catalog_path = (
        Path(__file__).resolve().parents[1]
        / "reviews"
        / "phase-1"
        / "reference-market-catalog.json"
    )
    catalog = compare_module.load_reference_catalog(catalog_path)
    records = {record["trading_date"]: record for record in catalog["files"]}
    request = DailyComparisonRequest(
        ticker="SPX",
        start_date=date(2024, 12, 23),
        end_date=date(2025, 1, 16),
        source="optionmetrics",
        source_generation="phase10-test",
        declared_oracle_gaps=(date(2024, 12, 24),),
        enforce_phase10_pilot=True,
    )
    daily = DailyWidbConfig()
    surface = DailySurfaceConfig()
    plans = compare_module.plan_daily_artifacts(
        compare_module.IntermediateBuildRequest(
            ticker="SPX",
            start_date=request.start_date,
            end_date=request.end_date,
            through="fitted-day",
            include_half_days=True,
            source="optionmetrics",
            source_generation=request.source_generation,
        ),
        daily,
        surface,
    )

    contract = compare_module._validate_phase10_pilot_contract(
        request,
        daily_config=daily,
        surface_config=surface,
        plans=plans,
        catalog_records=records,
        require_pinned_catalog=True,
    )

    assert contract["catalog_populated_files"] == 14
    assert contract["catalog_empty_markers"] == 1
    assert contract["catalog_aggregates"]["active_books"] == 76_044

    changed_clock = DailyWidbConfig.from_mapping(
        {"clock": {"overnight_variance_fraction": 0.1}}
    )
    with pytest.raises(ReferenceComparisonError, match="overnight_variance_fraction"):
        compare_module._validate_phase10_pilot_contract(
            request,
            daily_config=changed_clock,
            surface_config=surface,
            plans=plans,
            catalog_records=records,
            require_pinned_catalog=True,
        )

    changed_bid = DailyWidbConfig.from_mapping({"raw": {"min_bid": 0.5}})
    with pytest.raises(ReferenceComparisonError, match="frozen reference"):
        compare_module._validate_phase10_pilot_contract(
            request,
            daily_config=changed_bid,
            surface_config=surface,
            plans=plans,
            catalog_records=records,
            require_pinned_catalog=True,
        )
    changed_surface = DailySurfaceConfig.from_mapping(
        {"smooth_sanos": {"smoothness": 0.5}}
    )
    with pytest.raises(ReferenceComparisonError, match="frozen reference"):
        compare_module._validate_phase10_pilot_contract(
            request,
            daily_config=daily,
            surface_config=changed_surface,
            plans=plans,
            catalog_records=records,
            require_pinned_catalog=True,
        )


def test_phase11_contract_pins_full_range_raw_lineage_and_oracle() -> None:
    root = Path(__file__).resolve().parents[1]
    compare_module = importlib.import_module("sanos.reference.compare")
    daily = DailyWidbConfig()
    surface = DailySurfaceConfig()
    gaps = load_oracle_gap_declaration(
        root / "configs" / "spx-2022-2025-oracle-gaps.json",
        ticker="SPX",
        start_date=date(2022, 1, 3),
        end_date=date(2025, 8, 29),
    )
    request = DailyComparisonRequest(
        ticker="SPX",
        start_date=date(2022, 1, 3),
        end_date=date(2025, 8, 29),
        source="optionmetrics",
        source_generation="default-v1",
        declared_oracle_gaps=gaps,
        enforce_phase11_production=True,
    )
    plans = plan_daily_artifacts(
        IntermediateBuildRequest(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            through="fitted-day",
            include_half_days=True,
            source=request.source,
            source_generation=request.source_generation,
        ),
        daily,
        surface,
    )
    catalog = load_reference_catalog(
        root / "reviews" / "phase-1" / "reference-market-catalog.json"
    )
    records = {
        str(record["trading_date"]): record
        for record in cast(list[dict[str, object]], catalog["files"])
    }

    contract = compare_module._validate_phase11_production_contract(
        request,
        daily_config=daily,
        surface_config=surface,
        plans=plans,
        catalog_records=records,
        require_pinned_catalog=True,
    )

    assert contract["planned_sessions"] == 918
    assert contract["acquired_raw_artifacts"] == 3676
    assert contract["boundary_raw_artifacts"] == 3
    assert contract["daily_lineage_raw_artifacts"] == 3674
    assert {
        (record["table"], record["observation_date"])
        for record in contract["acquisition_only_raw_artifacts"]
    } == {
        ("forwards", "2021-12-31"),
        ("zero_curve", "2021-12-31"),
    }
    assert [
        (record["table"], record["observation_date"])
        for record in contract["consumed_boundary_raw_artifacts"]
    ] == [("spot", "2021-12-31")]
    assert contract["catalog_files"] == 892
    assert contract["catalog_populated_files"] == 891
    assert contract["catalog_empty_markers"] == 1
    assert contract["catalog_aggregates"]["active_books"] == 4_651_416

    with pytest.raises(ReferenceComparisonError, match="source_generation"):
        compare_module._validate_phase11_production_contract(
            replace(request, source_generation="changed-v1"),
            daily_config=daily,
            surface_config=surface,
            plans=plans,
            catalog_records=records,
            require_pinned_catalog=True,
        )
    with pytest.raises(ReferenceComparisonError, match="26-date"):
        compare_module._validate_phase11_production_contract(
            replace(request, declared_oracle_gaps=gaps[:-1]),
            daily_config=daily,
            surface_config=surface,
            plans=plans,
            catalog_records=records,
            require_pinned_catalog=True,
        )

    boundary_forward = raw_artifact_request(
        RawPartition(RawTable.FORWARDS, "SPX", date(2021, 12, 31)),
        source=request.source,
        source_generation=request.source_generation,
        producer_version=raw_content_producer_version(),
    ).spec.pointer
    wrong_first = replace(
        plans[0],
        raw=replace(plans[0].raw, previous_spot=boundary_forward),
    )
    with pytest.raises(ReferenceComparisonError, match="acquisition-only|spot partition"):
        compare_module._validate_phase11_production_contract(
            request,
            daily_config=daily,
            surface_config=surface,
            plans=(wrong_first, *plans[1:]),
            catalog_records=records,
            require_pinned_catalog=True,
        )

    empty_record = records["2025-01-08"]
    empty_checks = compare_module._phase11_production_day_checks(
        "2025-01-08",
        observations={"populated": False},
        typed_empty_reason="known_bad_day_all_tickers",
        record=empty_record,
        classification="policy_empty",
    )
    assert all(check["passed"] for check in empty_checks)
    stale_reason = compare_module._phase11_production_day_checks(
        "2025-01-08",
        observations={"populated": False},
        typed_empty_reason="different_reason",
        record=empty_record,
        classification="policy_empty",
    )
    assert [check["check"] for check in stale_reason if not check["passed"]] == [
        "policy_empty_reason"
    ]
    gap_checks = compare_module._phase11_production_day_checks(
        gaps[0].isoformat(),
        observations={"populated": True},
        typed_empty_reason=None,
        record=None,
        classification="oracle_gap",
    )
    assert all(check["passed"] for check in gap_checks)
    converted_gap = compare_module._phase11_production_day_checks(
        gaps[0].isoformat(),
        observations={"populated": True},
        typed_empty_reason=None,
        record=None,
        classification="approved_discrepancy",
    )
    assert not all(check["passed"] for check in converted_gap)


def test_comparison_contract_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        DailyComparisonRequest(
            ticker="SPX",
            start_date=date(2024, 12, 23),
            end_date=date(2025, 1, 16),
            source="optionmetrics",
            source_generation="default-v1",
            enforce_phase10_pilot=True,
            enforce_phase11_production=True,
        )


def test_comparison_rejects_invalid_source_revision_before_io(tmp_path: Path) -> None:
    with pytest.raises(ReferenceComparisonError, match="source_revision"):
        compare_daily_artifacts(
            _locations(tmp_path / "data"),
            DailyComparisonRequest(
                ticker="SPX",
                start_date=date(2024, 1, 3),
                end_date=date(2024, 1, 3),
                source="optionmetrics",
                source_generation="default-v1",
            ),
            daily_config=DailyWidbConfig(),
            surface_config=DailySurfaceConfig(),
            legacy_root=tmp_path / "absent",
            catalog_path=tmp_path / "absent.json",
            source_revision="not-a-commit",
        )


def test_comparison_version_resolves_in_clean_process() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not inherited_path
        else os.pathsep.join((source_root, inherited_path))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sanos.reference.compare import compare_daily_artifacts; "
            "print(str(compare_daily_artifacts.version))",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("1.1 {")


def test_checked_oracle_gap_declaration_is_exact_catalog_complement() -> None:
    root = Path(__file__).resolve().parents[1]
    declaration_path = root / "configs" / "spx-2022-2025-oracle-gaps.json"
    catalog = load_reference_catalog(
        root / "reviews" / "phase-1" / "reference-market-catalog.json"
    )
    gaps = load_oracle_gap_declaration(
        declaration_path,
        ticker="SPX",
        start_date=date(2022, 1, 3),
        end_date=date(2025, 8, 29),
    )
    plans = plan_daily_artifacts(
        IntermediateBuildRequest(
            ticker="SPX",
            start_date=date(2022, 1, 3),
            end_date=date(2025, 8, 29),
            through="fitted-day",
            include_half_days=True,
            source="optionmetrics",
            source_generation="default-v1",
        ),
        DailyWidbConfig(),
        DailySurfaceConfig(),
    )
    planned = {item.trading_date for item in plans}
    catalog_dates = {
        date.fromisoformat(cast(dict[str, object], record)["trading_date"])
        for record in cast(list[dict[str, object]], catalog["files"])
    }

    assert len(gaps) == 26
    assert set(gaps) == planned - catalog_dates


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("ticker", "NDX", "ticker differs"),
        ("start_date", "2022-01-04", "start_date differs"),
        ("end_date", "2025-08-28", "end_date differs"),
        ("catalog_sha256", "0" * 64, "pinned legacy catalog"),
        ("dates", ["2024-12-24", "2024-01-03"], "sorted and unique"),
    ],
)
def test_oracle_gap_declaration_rejects_identity_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "spx-2022-2025-oracle-gaps.json"
    )
    document = json.loads(source.read_text(encoding="utf-8"))
    document[field] = replacement
    changed = tmp_path / "gaps.json"
    changed.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReferenceComparisonError, match=message):
        load_oracle_gap_declaration(
            changed,
            ticker="SPX",
            start_date=date(2022, 1, 3),
            end_date=date(2025, 8, 29),
        )


def test_report_write_is_atomic_immutable_and_outside_oracle(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    output = tmp_path / "evidence" / "report.json"
    report = {"schema": "test", "outcome": "passed"}

    write_comparison_report(output, report, legacy_root=legacy)

    assert output.read_text(encoding="utf-8") == canonical_json(report) + "\n"
    with pytest.raises(ReferenceComparisonError, match="already exists"):
        write_comparison_report(output, report, legacy_root=legacy)
    with pytest.raises(ReferenceComparisonError, match="outside the read-only"):
        write_comparison_report(legacy / "report.json", report, legacy_root=legacy)


def test_report_interruption_removes_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    output = tmp_path / "evidence" / "report.json"
    compare_module = importlib.import_module("sanos.reference.compare")

    def interrupt_link(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic publish interruption")

    monkeypatch.setattr(compare_module.os, "link", interrupt_link)
    with pytest.raises(OSError, match="synthetic publish interruption"):
        write_comparison_report(
            output,
            {"schema": "test", "outcome": "passed"},
            legacy_root=legacy,
        )

    assert not output.exists()
    assert output.parent.is_dir()
    assert list(output.parent.iterdir()) == []


def test_cli_comparison_is_read_only_and_emits_structured_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, frame = _built_fitted_day(tmp_path / "data")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_legacy(legacy / "2024-01-03.h5", _legacy_payload(frame))
    catalog = tmp_path / "catalog.json"
    _catalog(legacy, catalog)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "daily": {
                    "clock": {"overnight_variance_fraction": 0.2},
                    "raw": {"min_bid": 0.011, "drop_irregular_expiries": True},
                    "enrichment": {},
                },
                "surface": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evidence" / "report.json"
    monkeypatch.setenv("SANOS_SOURCE_REVISION", "d" * 40)
    cli_module = importlib.import_module("sanos.cli.main")
    monkeypatch.setattr(
        cli_module,
        "compare_daily_artifacts",
        lambda *args, **kwargs: compare_daily_artifacts(
            *args,
            **{**kwargs, "require_pinned_catalog": False},
        ),
    )

    completed = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "-v",
            "reference",
            "compare-daily",
            "--config",
            str(config),
            "--ticker",
            "SPX",
            "--start-date",
            "2024-01-03",
            "--end-date",
            "2024-01-03",
            "--source-generation",
            "default-v1",
            "--legacy-root",
            str(legacy),
            "--legacy-catalog",
            str(catalog),
            "--output",
            str(output),
            "--progress-interval-seconds",
            "0.001",
        ],
    )

    assert completed.exit_code == 0, completed.output
    events = [json.loads(line) for line in completed.output.splitlines()]
    assert events[-1]["event"] == "reference_comparison_result"
    assert events[-1]["status"] == "passed"
    assert events[-1]["network_accessed"] is False
    assert events[-1]["artifacts_built"] is False
    assert output.is_file()
    assert locations.intermediate.is_dir()
