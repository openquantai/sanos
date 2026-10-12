from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs

import sanos.datasets.internal as internal_dataset_module
import sanos.widb.surface as surface_module
from sanos.config import canonical_json, canonical_sha256
from sanos.datasets import InternalDatasetConfig
from sanos.datasets.internal import DatasetValidationError, validate_fitted_partition
from sanos.widb import (
    ENRICHED_SCHEMA,
    FILTERED_SCHEMA,
    FITTED_SCHEMA,
    NO_ARB_SCHEMA,
    ClockConfig,
    DailyDataError,
    DailyRawInputs,
    DailySurfaceConfig,
    DailyWidbConfig,
    OptionsFilterConfig,
    SmoothSanosConfig,
    apply_options_filter,
    compatibility_dlv_pricer,
    enrich_daily_frame,
    fit_initial_no_arbitrage,
    fit_smooth_sanos,
    pair_daily_frame,
    selected_reference_view,
    smooth_sanos_pricer,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = REPOSITORY_ROOT / "reference" / "quant_data"


def _reference_surface(
    frame: pl.DataFrame,
    *,
    selected: np.ndarray[Any, Any],
    smoothness: float | None,
    input_dtype: str,
) -> dict[str, Any]:
    expiry_ids = frame.get_column("expiry_id").to_numpy()
    expiries: list[float] = []
    strikes: list[list[float]] = []
    bids: list[list[float]] = []
    asks: list[list[float]] = []
    for expiry_id in np.unique(expiry_ids[selected]):
        positions = np.flatnonzero(selected & (expiry_ids == expiry_id))
        local_strikes = frame.get_column("normalized_strike").to_numpy()[positions]
        positions = positions[np.argsort(local_strikes, kind="stable")]
        expiries.append(float(frame.get_column("time_to_expiry_years").item(int(positions[0]))))
        strikes.append(frame.get_column("normalized_strike").to_numpy()[positions].tolist())
        bids.append(frame.get_column("pure_bid_price").to_numpy()[positions].tolist())
        asks.append(frame.get_column("pure_ask_price").to_numpy()[positions].tolist())
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REFERENCE_SOURCE),
            "QUANTDATA_ROOT": "/tmp/sanos/phase-6-reference/data",
            "QUANTDATA_TEMP_DATA": "/tmp/sanos/phase-6-reference/temp",
            "WRDS_OFF": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tests" / "reference_surface_probe.py")],
        input=json.dumps(
            {
                "expiries": expiries,
                "strikes": strikes,
                "bids": bids,
                "asks": asks,
                "smoothness": smoothness,
                "input_dtype": input_dtype,
            }
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def _reference_filter(
    frame: pl.DataFrame,
    config: OptionsFilterConfig,
) -> dict[str, Any]:
    mapped = {
        "expiry_id": "expiry_id",
        "business_dte": "bdte",
        "normalized_moneyness": "nms",
        "atm_normalized_moneyness": "nms_atm",
        "combined_log_volume": "log_volume",
        "combined_log_open_interest": "log_open_interest",
        "expiry_is_am": "is_am",
        "spot": "spot",
        "previous_spot": "spot_prev",
        "strike": "strike",
        "implied_forward": "fwd",
        "market_call_bid": "mkt_call_bid",
        "market_call_ask": "mkt_call_ask",
        "market_put_bid": "mkt_put_bid",
        "market_put_ask": "mkt_put_ask",
    }
    rows = []
    for row in frame.iter_rows(named=True):
        rows.append({target: row[source] for source, target in mapped.items()})
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REFERENCE_SOURCE),
            "QUANTDATA_ROOT": "/tmp/sanos/phase-6-reference/data",
            "QUANTDATA_TEMP_DATA": "/tmp/sanos/phase-6-reference/temp",
            "WRDS_OFF": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tests" / "reference_filter_probe.py")],
        input=json.dumps(
            {
                "rows": rows,
                "config": {
                    "normalize": config.normalize,
                    "max_num_options": config.max_num_options,
                    "max_business_days_to_expiry": config.max_business_days_to_expiry,
                    "max_nms": config.max_nms,
                    "nms_use_atm": config.nms_use_atm,
                    "min_vegaSqrtVar": config.min_vega_sqrt_var,
                    "min_log_volume": config.min_log_volume,
                    "min_log_open_interest": config.min_log_open_interest,
                    "remove_am": config.remove_am,
                    "min_opt_per_expiry": config.min_options_per_expiry,
                },
            }
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def _two_expiry_raw_inputs(*, first_is_am: bool = False) -> DailyRawInputs:
    trading_date = date(2024, 1, 3)
    spot = 100.0
    expiry_specs = (
        (date(2024, 1, 19), 101.0, np.arange(90.0, 113.0, 2.0), 0.40),
        (date(2024, 2, 16), 102.0, np.arange(88.0, 117.0, 2.0), 0.25),
    )
    rows: list[dict[str, object]] = []
    forwards: list[dict[str, object]] = []
    option_id = 0
    for expiry_index, (expiry, forward, strikes, volatility) in enumerate(expiry_specs):
        is_am = first_is_am and expiry_index == 0
        calendar_dte = (expiry - trading_date).days
        discount = float(np.exp(-0.04 * calendar_dte / 365.0))
        business_dte = 11.0 if expiry_index == 0 else 31.0
        calls = (
            np.asarray(
                bs(
                    k=strikes / forward,
                    vol=volatility,
                    sqrtT=np.sqrt(business_dte / 252.0),
                )
            )
            * discount
            * forward
        )
        puts = calls - discount * (forward - strikes)
        for strike_index, (strike, call, put) in enumerate(
            zip(strikes, calls, puts, strict=True)
        ):
            for side, price in (("C", call), ("P", put)):
                rows.append(
                    {
                        "secid": 108105,
                        "date": trading_date.isoformat(),
                        "symbol": f"SPX-{expiry}-{side}-{strike_index}",
                        "optionid": option_id,
                        "am_settlement": int(is_am),
                        "cp_flag": side,
                        "strike_price": int(strike * 1000),
                        "exdate": expiry.isoformat(),
                        "best_bid": float(
                            0.02
                            if expiry_index == 1
                            and strike_index == len(strikes) - 1
                            and side == "P"
                            else price - 0.06
                        ),
                        "best_offer": float(price + 0.07),
                        "volume": 12,
                        "open_interest": 120,
                        "impl_volatility": volatility,
                        "ss_flag": "0",
                        "delta": 0.0,
                        "gamma": 0.0,
                        "vega": 0.0,
                        "theta": 0.0,
                        "expiry_indicator": "w",
                        "contract_size": 100,
                    }
                )
                option_id += 1
        forwards.append(
            {
                "secid": 108105,
                "date": trading_date.isoformat(),
                "expiration": expiry.isoformat(),
                "amsettlement": int(is_am),
                "forwardprice": forward,
            }
        )
    return DailyRawInputs(
        security=pl.DataFrame({"ticker": ["SPX"], "secid": [108105]}),
        options=pl.DataFrame(rows),
        spot=pl.DataFrame(
            {"secid": [108105], "date": [trading_date.isoformat()], "close": [spot]}
        ),
        previous_spot=pl.DataFrame(
            {"secid": [108105], "date": ["2024-01-02"], "close": [99.0]}
        ),
        forwards=pl.DataFrame(forwards),
        zero_curve=pl.DataFrame(
            {
                "date": [trading_date.isoformat()] * 3,
                "days": [1, 30, 90],
                "rate": [4.0, 4.0, 4.0],
            }
        ),
    )


@pytest.fixture(scope="module")
def phase6_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    enriched = enrich_daily_frame(
        pair_daily_frame(
            ticker="SPX",
            trading_date=date(2024, 1, 3),
            raw=_two_expiry_raw_inputs(),
        )
    )
    config = DailySurfaceConfig()
    no_arb = fit_initial_no_arbitrage(enriched, config=config)
    filtered = apply_options_filter(no_arb, config=config)
    fitted = fit_smooth_sanos(filtered, config=config)
    return enriched, no_arb, filtered, fitted


def test_phase6_frames_preserve_auditable_population_and_complete_states(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    enriched, no_arb, filtered, fitted = phase6_frames
    assert no_arb.schema == pl.Schema(NO_ARB_SCHEMA)
    assert filtered.schema == pl.Schema(FILTERED_SCHEMA)
    assert fitted.schema == pl.Schema(FITTED_SCHEMA)
    assert no_arb.height == filtered.height == fitted.height == enriched.height
    assert fitted.get_column("options_filter_eligible").sum() == enriched.height

    one_per_expiry = fitted.unique("expiry_id", keep="first", maintain_order=True)
    for row in one_per_expiry.iter_rows(named=True):
        initial_strikes = np.asarray(row["initial_surface_model_strikes"])
        initial_density = np.asarray(row["initial_surface_model_density"])
        initial_dlv = np.asarray(row["initial_surface_dlv_squares"])
        assert np.all(np.diff(initial_strikes) > 0.0)
        assert np.min(initial_density) >= -1.0e-7
        assert np.sum(initial_density) == pytest.approx(1.0, abs=1.0e-4)
        assert initial_strikes @ initial_density == pytest.approx(1.0, abs=1.0e-4)
        assert initial_dlv.shape == initial_strikes.shape
        assert initial_dlv[0] == initial_dlv[-1] == 0.0
        assert np.all(np.isfinite(initial_dlv))

        smooth_strikes = np.asarray(row["smooth_surface_model_strikes"])
        smooth_density = np.asarray(row["smooth_surface_model_density"])
        assert np.all(np.diff(smooth_strikes) > 0.0)
        assert np.min(smooth_density) >= -1.0e-7
        assert np.sum(smooth_density) == pytest.approx(1.0, abs=1.0e-4)
        assert smooth_strikes @ smooth_density == pytest.approx(1.0, abs=1.0e-4)

    eligible = fitted.get_column("options_filter_eligible").to_numpy().astype(bool)
    discount_forward = (
        fitted.get_column("discount_factor").to_numpy()
        * fitted.get_column("implied_forward").to_numpy()
    )
    expected_call_bid = fitted.get_column("market_call_bid").to_numpy() / discount_forward
    np.testing.assert_allclose(
        fitted.get_column("pure_market_call_bid").to_numpy()[eligible],
        expected_call_bid[eligible],
        rtol=2.0e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        fitted.get_column("smooth_put_mid").to_numpy()[eligible],
        fitted.get_column("smooth_call_mid").to_numpy()[eligible]
        - 1.0
        + fitted.get_column("normalized_strike").to_numpy()[eligible],
        rtol=0.0,
        atol=0.0,
    )
    assert selected_reference_view(fitted).height == int(np.sum(eligible))
    diagnostics = json.loads(fitted.get_column("smooth_diagnostics_json").item(0))
    assert diagnostics["selected_rows"] == int(np.sum(eligible))
    assert diagnostics["solver_timeout_seconds"] == 300.0
    assert diagnostics["solver_events"]
    assert {event["stage"] for event in diagnostics["solver_events"]} >= {
        "expiry_soft",
        "surface",
    }
    assert all(
        event["status"] in {"optimal", "optimal_inaccurate"}
        for event in diagnostics["solver_events"]
    )
    no_arb_diagnostics = json.loads(no_arb.get_column("no_arb_diagnostics_json").item(0))
    assert no_arb_diagnostics["solver_events"]
    config = DailySurfaceConfig()
    assert no_arb.get_column("no_arb_configuration_id").unique().to_list() == [
        canonical_sha256({"no_arbitrage": config.no_arbitrage})
    ]
    assert filtered.get_column("options_filter_configuration_id").unique().to_list() == [
        canonical_sha256({"options_filter": config.options_filter})
    ]
    assert fitted.get_column("smooth_sanos_configuration_id").unique().to_list() == [
        canonical_sha256(
            {
                "no_arbitrage": config.no_arbitrage,
                "smooth_sanos": config.smooth_sanos,
            }
        )
    ]

    dlv_pricer = compatibility_dlv_pricer(fitted)
    smooth_pricer = smooth_sanos_pricer(fitted)
    decoded_densities = dlv_pricer.densities_from_dlvsqs(
        dlv_pricer.dlvsqs(on_all_strikes=True)
    )
    for observed, expected in zip(
        decoded_densities,
        dlv_pricer.xdensities,
        strict=True,
    ):
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-14)
    decoded_pricer = replace(dlv_pricer, xdensities=decoded_densities)
    irregular_query = np.asarray([0.713, 0.947, 1.037, 1.289], dtype=np.float64)
    for expiry in dlv_pricer.expiries:
        np.testing.assert_allclose(
            decoded_pricer(expiry, irregular_query),
            dlv_pricer(expiry, irregular_query),
            rtol=0.0,
            atol=2.0e-14,
        )
    for expiry_index, expiry in enumerate(dlv_pricer.expiries):
        expiry_rows = fitted.filter(
            pl.col("initial_no_arb_input_expiry_index") == expiry_index
        )
        observed = dlv_pricer(expiry, expiry_rows.get_column("normalized_strike").to_numpy())
        np.testing.assert_allclose(
            observed,
            expiry_rows.get_column("pure_fitX_price").to_numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )
    for expiry_index, expiry in enumerate(smooth_pricer.expiries):
        expiry_rows = fitted.filter(pl.col("smooth_input_expiry_index") == expiry_index)
        observed = smooth_pricer(
            expiry,
            expiry_rows.get_column("normalized_strike").to_numpy(),
        )
        np.testing.assert_allclose(
            observed,
            expiry_rows.get_column("smooth_call_mid").to_numpy(),
            rtol=0.0,
            # The stable SanosPricer API promotes query arrays to Float64,
            # whereas the compatibility producer prices its persisted
            # Float32 query arrays directly.
            atol=5.0e-8,
        )


def test_initial_and_smooth_states_match_pinned_reference(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    enriched, no_arb, filtered, fitted = phase6_frames
    initial_reference = _reference_surface(
        enriched,
        selected=np.ones(enriched.height, dtype=bool),
        smoothness=None,
        input_dtype="float64",
    )
    assert initial_reference["input_index"] == [0, 1]
    for source_index, expiry_id in enumerate(
        no_arb.get_column("expiry_id").unique(maintain_order=True).to_list()
    ):
        observed = no_arb.filter(pl.col("expiry_id") == expiry_id)
        np.testing.assert_array_equal(
            observed.get_column("mask_arb_free").to_numpy(),
            np.asarray(initial_reference["masks"][source_index], dtype=bool),
        )
        for output_prefix in ("fit1", "fitX"):
            reference_fields = initial_reference["quote_fields"][source_index]
            for greek in ("price", "delta", "gamma", "vega", "theta", "dk"):
                np.testing.assert_array_equal(
                    observed.get_column(f"pure_{output_prefix}_{greek}").to_numpy(),
                    np.asarray(
                        reference_fields[f"pure_{output_prefix}_{greek}"],
                        dtype=np.float32,
                    ),
                )
            np.testing.assert_array_equal(
                observed.get_column(f"{output_prefix}_volatility").to_numpy(),
                np.asarray(
                    reference_fields[f"{output_prefix}_volatility"],
                    dtype=np.float32,
                ),
            )
        first = observed.row(0, named=True)
        for column, reference_key in (
            ("initial_expiry_model_strikes", "xstrikes"),
            ("initial_expiry_model_density", "xdensities"),
            ("initial_surface_model_strikes", "mstrikes"),
            ("initial_surface_model_density", "mdensities"),
        ):
            np.testing.assert_allclose(
                first[column],
                initial_reference[reference_key][source_index],
                rtol=0.0,
                atol=1.0e-12,
            )

    eligible = filtered.get_column("options_filter_eligible").to_numpy().astype(bool)
    smooth_reference = _reference_surface(
        filtered,
        selected=eligible,
        smoothness=0.25,
        input_dtype="float32",
    )
    reference_expiry_ids = (
        filtered.filter(pl.col("options_filter_eligible"))
        .get_column("expiry_id")
        .unique(maintain_order=True)
        .to_list()
    )
    assert smooth_reference["input_index"] == list(range(len(reference_expiry_ids)))
    for source_index, expiry_id in enumerate(reference_expiry_ids):
        observed = fitted.filter(
            pl.col("options_filter_eligible") & (pl.col("expiry_id") == expiry_id)
        )
        np.testing.assert_allclose(
            observed.get_column("smooth_call_mid").to_numpy(),
            smooth_reference["fit"]["fits"][source_index],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            observed.get_column("smooth_fit_error_half_spreads").to_numpy(),
            smooth_reference["fit"]["fit_errs"][source_index],
            rtol=0.0,
            atol=1.0e-12,
        )
        first = observed.row(0, named=True)
        for column, reference_key in (
            ("smooth_expiry_model_strikes", "xstrikes"),
            ("smooth_expiry_model_density", "xdensities"),
            ("smooth_surface_model_strikes", "mstrikes"),
            ("smooth_surface_model_density", "mdensities"),
        ):
            np.testing.assert_allclose(
                first[column],
                smooth_reference[reference_key][source_index],
                rtol=0.0,
                atol=1.0e-12,
            )
    assert fitted.get_column("smooth_pcnt_fit").item(0) == pytest.approx(
        smooth_reference["fit"]["pcnt_fit"], abs=0.0
    )


def test_two_expiry_fitted_frame_passes_dataset_quantitative_boundary(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    fitted = phase6_frames[-1]
    config = InternalDatasetConfig(
        name="two-expiry-validation",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )
    statistics = validate_fitted_partition(
        fitted,
        config=config,
        trading_date=date(2024, 1, 3),
        typed_empty_reason=None,
    )
    assert statistics.input_expiries == 2
    assert statistics.smooth_retained_expiries == 2
    assert statistics.residual_maxima["dlv_density_roundtrip"] <= 2.0e-14


def test_dataset_quantitative_boundary_rejects_state_config_and_clock_corruption(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    fitted = phase6_frames[-1]
    config = InternalDatasetConfig(
        name="two-expiry-adversarial-validation",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )

    def inconsistent_last(column: str, value: str) -> pl.DataFrame:
        return (
            fitted.with_row_index("__row")
            .with_columns(
                pl.when(pl.col("__row") == fitted.height - 1)
                .then(pl.lit(value))
                .otherwise(pl.col(column))
                .alias(column)
            )
            .drop("__row")
        )

    def clear_first_true(column: str) -> pl.DataFrame:
        row = int(
            np.flatnonzero(fitted.get_column(column).to_numpy().astype(bool))[0]
        )
        return (
            fitted.with_row_index("__row")
            .with_columns(
                pl.when(pl.col("__row") == row)
                .then(pl.lit(False))
                .otherwise(pl.col(column))
                .alias(column)
            )
            .drop("__row")
        )

    corruptions = {
        "duplicate_expiry_strike_key": pl.concat(
            (fitted.slice(0, 1), fitted), how="vertical"
        ),
        "configuration_id": fitted.with_columns(
            pl.lit("0" * 64).alias("configuration_id")
        ),
        "stage_version": fitted.with_columns(pl.lit("corrupt").alias("stage_version")),
        "raw_options_artifact_id": fitted.with_columns(
            pl.lit("1" * 64).alias("raw_options_artifact_id")
        ),
        "surface_min_half_spread": fitted.with_columns(
            pl.lit(0.123).alias("surface_min_half_spread")
        ),
        "pure_fit1_price": fitted.with_columns(
            (pl.col("pure_fit1_price") + 0.01).alias("pure_fit1_price")
        ),
        "mask_arb_free": clear_first_true("mask_arb_free"),
        "smooth_fit_mask": clear_first_true("smooth_fit_mask"),
        "pure_bid_delta": fitted.with_columns(
            (pl.col("pure_bid_delta") + 0.25).alias("pure_bid_delta")
        ),
        "pure_bid_delta_nan": fitted.with_columns(
            pl.lit(float("nan"), dtype=pl.Float32).alias("pure_bid_delta")
        ),
        "pure_fitX_gamma_positive_infinity": fitted.with_columns(
            pl.lit(float("inf"), dtype=pl.Float32).alias("pure_fitX_gamma")
        ),
        "smooth_pcnt_fit_negative_infinity": fitted.with_columns(
            pl.lit(float("-inf"), dtype=pl.Float64).alias("smooth_pcnt_fit")
        ),
        "fit1_volatility": fitted.with_columns(
            (pl.col("fit1_volatility") + 0.25).alias("fit1_volatility")
        ),
        "pure_fit1_delta": fitted.with_columns(
            (pl.col("pure_fit1_delta") + 0.25).alias("pure_fit1_delta")
        ),
        "fitX_volatility": fitted.with_columns(
            (pl.col("fitX_volatility") + 0.25).alias("fitX_volatility")
        ),
        "pure_fitX_gamma": fitted.with_columns(
            (pl.col("pure_fitX_gamma") + 0.25).alias("pure_fitX_gamma")
        ),
        "initial_expiry_density": fitted.with_columns(
            pl.col("initial_expiry_model_density")
            .list.eval(pl.element() * 0.0)
            .alias("initial_expiry_model_density")
        ),
        "smooth_expiry_density": fitted.with_columns(
            pl.col("smooth_expiry_model_density")
            .list.eval(pl.element() * 0.0)
            .alias("smooth_expiry_model_density")
        ),
        "smooth_surface_strikes_nan": fitted.with_columns(
            pl.col("smooth_surface_model_strikes")
            .list.eval(pl.element() * float("nan"))
            .alias("smooth_surface_model_strikes")
        ),
        "initial_mapping": fitted.with_columns(
            pl.col("initial_x_to_surface_indices")
            .list.eval(pl.element() * 0)
            .alias("initial_x_to_surface_indices")
        ),
        "smooth_mapping": fitted.with_columns(
            pl.col("smooth_x_to_surface_indices")
            .list.eval(pl.element() * 0)
            .alias("smooth_x_to_surface_indices")
        ),
        "spot_return": fitted.with_columns(
            pl.lit(999.0, dtype=pl.Float32).alias("spot_return")
        ),
        "calendar_dte": fitted.with_columns(
            (pl.col("calendar_dte") + pl.lit(1, dtype=pl.Int32)).alias("calendar_dte")
        ),
        "business_dte": fitted.with_columns(
            (pl.col("business_dte") + pl.lit(1.0, dtype=pl.Float32)).alias("business_dte")
        ),
        "smooth_expiry_fit_price": fitted.with_columns(
            (pl.col("smooth_expiry_fit_price") + 0.01).alias("smooth_expiry_fit_price")
        ),
        "smooth_fit_summary": fitted.with_columns(
            (pl.col("smooth_pcnt_fit") + 0.01).alias("smooth_pcnt_fit")
        ),
        "no_arb_diagnostics": inconsistent_last("no_arb_diagnostics_json", "{}"),
        "filter_diagnostics": inconsistent_last("filter_diagnostics_json", "{}"),
        "smooth_diagnostics": inconsistent_last("smooth_diagnostics_json", "{}"),
    }
    for corrupted in corruptions.values():
        with pytest.raises(DatasetValidationError, match=".+"):
            validate_fitted_partition(
                corrupted,
                config=config,
                trading_date=date(2024, 1, 3),
                typed_empty_reason=None,
            )


def test_dataset_finiteness_gate_covers_every_present_float_field(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    fitted = phase6_frames[-1]
    bad_values = (float("nan"), float("inf"), float("-inf"))
    checked: list[str] = []
    for field_index, (column, dtype) in enumerate(FITTED_SCHEMA.items()):
        bad_value = bad_values[field_index % len(bad_values)]
        series = fitted.get_column(column)
        if dtype in (pl.Float32, pl.Float64):
            present = np.flatnonzero(series.is_not_null().to_numpy())
            if not len(present):
                continue
            row_index = int(present[0])
            corrupted = (
                fitted.with_row_index("__row")
                .with_columns(
                    pl.when(pl.col("__row") == row_index)
                    .then(pl.lit(bad_value, dtype=dtype))
                    .otherwise(pl.col(column))
                    .alias(column)
                )
                .drop("__row")
            )
        elif dtype in (pl.List(pl.Float32), pl.List(pl.Float64)):
            values = series.to_list()
            row_index = next(
                (
                    index
                    for index, items in enumerate(values)
                    if items is not None and len(items) > 0
                ),
                None,
            )
            if row_index is None:
                continue
            values[row_index][0] = bad_value
            corrupted = fitted.with_columns(pl.Series(column, values, dtype=dtype))
        else:
            continue
        with pytest.raises(DatasetValidationError, match=column):
            internal_dataset_module._validate_fitted_numeric_finiteness(corrupted)
        checked.append(column)

    expected = {
        name
        for name, dtype in FITTED_SCHEMA.items()
        if dtype in (
            pl.Float32,
            pl.Float64,
            pl.List(pl.Float32),
            pl.List(pl.Float64),
        )
        and fitted.get_column(name).drop_nulls().len()
    }
    assert set(checked) == expected


def test_initial_daily_mapping_matches_reference_with_boundary_mask(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    enriched, _, _, _ = phase6_frames
    expiry_id = enriched.get_column("expiry_id").item(0)
    boundary = enriched.filter(pl.col("expiry_id") == expiry_id).head(10)
    strikes = np.arange(0.6, 1.51, 0.1, dtype=np.float32)
    sqrt_time = np.sqrt(float(boundary.get_column("time_to_expiry_years").item(0)))
    mids = np.asarray(bs(k=strikes, vol=0.3, sqrtT=sqrt_time), dtype=np.float32)
    mids[-2:] = mids[-3] - np.arange(1, 3, dtype=np.float32) * np.float32(1.0e-5)
    boundary = boundary.with_columns(
        pl.Series("normalized_strike", strikes),
        pl.Series("pure_bid_price", np.maximum(0.0, mids - np.float32(2.0e-5))),
        pl.Series("pure_ask_price", mids + np.float32(2.0e-5)),
    ).select(*ENRICHED_SCHEMA)
    reference = _reference_surface(
        boundary,
        selected=np.ones(boundary.height, dtype=bool),
        smoothness=None,
        input_dtype="float64",
    )
    observed = fit_initial_no_arbitrage(boundary)
    expected_mask = np.asarray(reference["masks"][0], dtype=bool)
    np.testing.assert_array_equal(
        observed.get_column("mask_arb_free").to_numpy(), expected_mask
    )
    assert np.any(~expected_mask)
    assert observed.filter(~pl.col("mask_arb_free")).get_column(
        "initial_no_arb_rejection_reason"
    ).unique().to_list() == ["initial_no_arb_strike_masked"]


def test_late_expiry_drop_and_smooth_alignment_match_reference(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    enriched, _, filtered, _ = phase6_frames
    expiry_ids = enriched.get_column("expiry_id").unique(maintain_order=True).to_list()
    sparse_enriched = pl.concat(
        (
            enriched.filter(pl.col("expiry_id") == expiry_ids[0]),
            enriched.filter(pl.col("expiry_id") == expiry_ids[1]).head(4),
        )
    ).select(*ENRICHED_SCHEMA)
    initial_reference = _reference_surface(
        sparse_enriched,
        selected=np.ones(sparse_enriched.height, dtype=bool),
        smoothness=None,
        input_dtype="float64",
    )
    no_arb = fit_initial_no_arbitrage(sparse_enriched)
    assert initial_reference["input_index"] == [0]
    assert no_arb.filter(pl.col("expiry_id") == expiry_ids[1]).get_column(
        "initial_no_arb_rejection_reason"
    ).unique().to_list() == ["initial_no_arb_expiry_dropped"]
    assert no_arb.filter(pl.col("expiry_id") == expiry_ids[1]).get_column(
        "initial_no_arb_input_expiry_index"
    ).null_count() == 4

    sparse_filtered = pl.concat(
        (
            filtered.filter(pl.col("expiry_id") == expiry_ids[0]),
            filtered.filter(pl.col("expiry_id") == expiry_ids[1]).head(4),
        )
    ).select(*FILTERED_SCHEMA)
    selected = sparse_filtered.get_column("options_filter_eligible").to_numpy().astype(bool)
    smooth_reference = _reference_surface(
        sparse_filtered,
        selected=selected,
        smoothness=0.25,
        input_dtype="float32",
    )
    fitted = fit_smooth_sanos(sparse_filtered)
    assert smooth_reference["input_index"] == [0]
    late = fitted.filter(pl.col("expiry_id") == expiry_ids[1])
    assert late.get_column("smooth_rejection_reason").unique().to_list() == [
        "smooth_expiry_dropped"
    ]
    assert late.get_column("smooth_input_expiry_index").null_count() == 4


def test_filter_order_reasons_cap_and_vega_compatibility_noop(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    _, no_arb, _, _ = phase6_frames
    expiry_ids = no_arb.get_column("expiry_id").unique(maintain_order=True).to_list()
    modified = no_arb.with_row_index("test_row").with_columns(
        pl.when(pl.col("test_row") == 0)
        .then(pl.lit(1.0e6))
        .otherwise(pl.col("atm_normalized_moneyness"))
        .alias("atm_normalized_moneyness"),
        pl.when(pl.col("test_row") == 1)
        .then(pl.lit(0.0))
        .otherwise(pl.col("combined_log_volume"))
        .alias("combined_log_volume"),
        pl.when(pl.col("test_row") == 2)
        .then(pl.lit(0.0))
        .otherwise(pl.col("combined_log_open_interest"))
        .alias("combined_log_open_interest"),
    ).drop("test_row")
    rules = OptionsFilterConfig(
        max_num_options=10,
        max_business_days_to_expiry=70,
        max_nms=(-100.0, 100.0),
        min_options_per_expiry=5,
    )
    filtered = apply_options_filter(
        modified,
        config=DailySurfaceConfig(options_filter=rules),
    )
    assert filtered.get_column("options_filter_rejection_reason").head(3).to_list() == [
        "normalized_moneyness_outside_range",
        "log_volume_below_minimum",
        "log_open_interest_below_minimum",
    ]
    selected = filtered.filter(pl.col("options_filter_eligible"))
    assert selected.height == 10
    counts = selected.group_by("expiry_id").len().sort("expiry_id")
    assert counts.get_column("expiry_id").to_list() == expiry_ids
    assert counts.get_column("len").to_list() == [5, 5]

    reference = _reference_filter(modified, rules)
    reference_ids = np.asarray(reference["row_ids"], dtype=np.int64)
    np.testing.assert_array_equal(
        np.sort(reference_ids),
        np.flatnonzero(filtered.get_column("options_filter_eligible").to_numpy()),
    )
    selected_in_reference_order = filtered[reference_ids]
    for local_name, reference_name in (
        ("spot_reference", "spot_ref"),
        ("normalized_previous_spot", "spot_prev"),
        ("spot_normalized_strike", "strike"),
        ("spot_normalized_implied_forward", "fwd"),
        ("spot_normalized_market_call_bid", "mkt_call_bid"),
        ("spot_normalized_market_call_ask", "mkt_call_ask"),
        ("spot_normalized_market_put_bid", "mkt_put_bid"),
        ("spot_normalized_market_put_ask", "mkt_put_ask"),
    ):
        np.testing.assert_array_equal(
            selected_in_reference_order.get_column(local_name).to_numpy(),
            np.asarray(reference[reference_name], dtype=np.float32),
        )

    no_vega = apply_options_filter(
        no_arb,
        config=DailySurfaceConfig(
            options_filter=replace(rules, max_num_options=100_000, min_vega_sqrt_var=0.0)
        ),
    )
    huge_vega = apply_options_filter(
        no_arb,
        config=DailySurfaceConfig(
            options_filter=replace(rules, max_num_options=100_000, min_vega_sqrt_var=1.0e9)
        ),
    )
    assert no_vega.get_column("options_filter_eligible").equals(
        huge_vega.get_column("options_filter_eligible")
    )
    assert json.loads(huge_vega.get_column("filter_diagnostics_json").item(0))[
        "min_vega_sqrt_var_applied"
    ] is False

    late_modified = no_arb.with_row_index("test_row").with_columns(
        pl.when(pl.col("test_row") == no_arb.height - 1)
        .then(pl.lit(1.0e6))
        .otherwise(pl.col("atm_normalized_moneyness"))
        .alias("atm_normalized_moneyness")
    ).drop("test_row")
    late_filtered = apply_options_filter(late_modified)
    assert late_filtered.get_column("options_filter_rejection_reason").item(-1) == (
        "normalized_moneyness_outside_range"
    )


def test_surface_config_is_strict_and_uses_reference_profile() -> None:
    config = DailySurfaceConfig.from_mapping(
        {
            "options_filter": {"max_nms": [-100.0, 100.0]},
            "smooth_sanos": {"smoothness": 0.25},
        }
    )
    assert config.options_filter.max_num_options == 100_000
    assert config.options_filter.max_business_days_to_expiry == 70
    assert config.options_filter.max_nms == (-100.0, 100.0)
    assert config.smooth_sanos.smoothness == 0.25
    with pytest.raises(ValueError, match="Unknown daily surface"):
        DailySurfaceConfig.from_mapping({"unknown": {}})
    with pytest.raises(ValueError, match="straddle zero"):
        OptionsFilterConfig(max_nms=(0.0, 1.0))
    with pytest.raises(ValueError, match="requires options_filter.normalize"):
        DailySurfaceConfig(options_filter=OptionsFilterConfig(normalize=False))


def test_typed_empty_day_flows_without_solver_access() -> None:
    enriched = pl.DataFrame(schema=ENRICHED_SCHEMA)
    no_arb = fit_initial_no_arbitrage(enriched)
    filtered = apply_options_filter(no_arb)
    fitted = fit_smooth_sanos(filtered)
    assert no_arb.schema == pl.Schema(NO_ARB_SCHEMA)
    assert filtered.schema == pl.Schema(FILTERED_SCHEMA)
    assert fitted.schema == pl.Schema(FITTED_SCHEMA)
    assert no_arb.is_empty() and filtered.is_empty() and fitted.is_empty()


def test_no_selected_options_still_records_smooth_configuration_identity(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    _, _, filtered, _ = phase6_frames
    config = DailySurfaceConfig(
        smooth_sanos=SmoothSanosConfig(smoothness=0.5),
    )
    none_selected = filtered.with_columns(
        pl.lit(False).alias("options_filter_eligible")
    )
    fitted = fit_smooth_sanos(none_selected, config=config)
    assert fitted.get_column("smooth_sanos_configuration_id").unique().to_list() == [
        canonical_sha256(
            {
                "no_arbitrage": config.no_arbitrage,
                "smooth_sanos": config.smooth_sanos,
            }
        )
    ]


def test_natural_zero_selected_day_passes_dataset_boundary(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    no_arb = phase6_frames[1]
    surface = DailySurfaceConfig(
        options_filter=replace(
            DailySurfaceConfig().options_filter,
            min_log_volume=1.0e6,
        )
    )
    fitted = fit_smooth_sanos(apply_options_filter(no_arb, config=surface), config=surface)
    config = InternalDatasetConfig(
        name="zero-selected-validation",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=surface,
    )
    statistics = validate_fitted_partition(
        fitted,
        config=config,
        trading_date=date(2024, 1, 3),
        typed_empty_reason=None,
    )
    assert statistics.eligible_rows == 0
    assert statistics.smooth_retained_expiries == 0

    filter_diagnostics = json.loads(
        fitted.get_column("filter_diagnostics_json").item(0)
    )
    candidate = int(
        np.flatnonzero(
            fitted.get_column("filter_horizon_eligible").to_numpy().astype(bool)
            & fitted.get_column("filter_nms_eligible").to_numpy().astype(bool)
            & ~fitted.get_column("filter_volume_eligible").to_numpy().astype(bool)
        )[0]
    )
    filter_diagnostics["counts"]["horizon"] -= 1
    filter_diagnostics["counts"]["nms"] -= 1
    corrupted_filter = (
        fitted.with_row_index("__row")
        .with_columns(
            pl.when(pl.col("__row") == candidate)
            .then(pl.lit(False))
            .otherwise(pl.col("filter_horizon_eligible"))
            .alias("filter_horizon_eligible"),
            pl.when(pl.col("__row") == candidate)
            .then(pl.lit(False))
            .otherwise(pl.col("filter_nms_eligible"))
            .alias("filter_nms_eligible"),
            pl.lit(canonical_json(filter_diagnostics)).alias(
                "filter_diagnostics_json"
            ),
        )
        .drop("__row")
    )
    with pytest.raises(DatasetValidationError, match="canonical OptionsFilter"):
        validate_fitted_partition(
            corrupted_filter,
            config=config,
            trading_date=date(2024, 1, 3),
            typed_empty_reason=None,
        )


def test_solver_failure_retains_structured_events_in_error(
    monkeypatch: pytest.MonkeyPatch,
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    _, _, filtered, _ = phase6_frames
    event = {
        "stage": "surface",
        "solver": "CLARABEL",
        "algorithm": "simplex",
        "status": "solver_error",
        "is_fallback": True,
        "for_expiry": False,
        "objective": None,
    }

    def failing_factory(
        config: Any,
        *,
        smoothness: float | None,
        diagnostic_sink: Any = None,
    ) -> Any:
        del config, smoothness

        def fail(**kwargs: Any) -> Any:
            del kwargs
            diagnostic_sink(event)
            raise RuntimeError("forced fallback failure")

        return fail

    monkeypatch.setattr(surface_module, "_make_no_arb_fitter", failing_factory)
    with pytest.raises(DailyDataError, match="forced fallback failure") as caught:
        fit_smooth_sanos(filtered)
    message = str(caught.value)
    assert '"outcome":"failed"' in message
    assert '"is_fallback":true' in message
    assert '"stage":"surface"' in message


def test_smooth_fit_preserves_float32_ingress(
    monkeypatch: pytest.MonkeyPatch,
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    _, _, filtered, _ = phase6_frames
    original_factory = surface_module._make_no_arb_fitter
    captured: dict[str, object] = {}

    def capturing_factory(*args: Any, **kwargs: Any) -> Any:
        fitter = original_factory(*args, **kwargs)

        def capture(**fit_kwargs: Any) -> Any:
            captured["expiry_dtype"] = fit_kwargs["expiries"].dtype
            captured["strike_dtypes"] = tuple(
                value.dtype for value in fit_kwargs["strikes"]
            )
            captured["bid_dtypes"] = tuple(value.dtype for value in fit_kwargs["bids"])
            captured["ask_dtypes"] = tuple(value.dtype for value in fit_kwargs["asks"])
            return fitter(**fit_kwargs)

        return capture

    monkeypatch.setattr(surface_module, "_make_no_arb_fitter", capturing_factory)
    fit_smooth_sanos(filtered)
    assert captured["expiry_dtype"] == np.dtype(np.float32)
    for key in ("strike_dtypes", "bid_dtypes", "ask_dtypes"):
        assert set(cast(tuple[np.dtype[Any], ...], captured[key])) == {
            np.dtype(np.float32)
        }


def test_surface_version_closures_include_content_affecting_helpers() -> None:
    def direct_dependency_names(component: Any) -> set[str]:
        dependencies = cast(Any, component.version).dependencies
        assert isinstance(dependencies, tuple)
        return set(cast(dict[str, Any], dependencies[1]))

    smooth_dependencies = direct_dependency_names(fit_smooth_sanos)
    assert {"_float_column", "_float32_column", "_require_schema"} <= (
        smooth_dependencies
    )
    assert direct_dependency_names(selected_reference_view) == {"_require_schema"}


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5])
def test_overnight_fraction_reaches_filter_and_both_model_states(fraction: float) -> None:
    daily = DailyWidbConfig(clock=ClockConfig(overnight_variance_fraction=fraction))
    enriched = enrich_daily_frame(
        pair_daily_frame(
            ticker="SPX",
            trading_date=date(2024, 1, 3),
            raw=_two_expiry_raw_inputs(first_is_am=True),
            config=daily,
        ),
        config=daily,
    )
    no_arb = fit_initial_no_arbitrage(enriched)
    filtered = apply_options_filter(no_arb)
    fitted = fit_smooth_sanos(filtered)
    first_expiry = fitted.sort("time_to_expiry_years").row(0, named=True)
    assert first_expiry["expiry_is_am"] is True
    assert first_expiry["business_dte"] == pytest.approx(
        np.float32(10.0 + fraction)
    )
    assert first_expiry["time_to_expiry_years"] == pytest.approx(
        np.float32((10.0 + fraction) / 252.0)
    )
    assert first_expiry["initial_model_variance"] is not None
    assert first_expiry["initial_surface_dlv_squares"] is not None
    assert first_expiry["smooth_model_variance"] is not None
    assert first_expiry["smooth_surface_model_density"] is not None
    assert fitted.get_column("options_filter_eligible").sum() == fitted.height
