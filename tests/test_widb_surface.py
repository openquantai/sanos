from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs
from cdxcore.verbose import Context

import sanos.datasets.internal as internal_dataset_module
import sanos.widb.surface as surface_module
from sanos.config import canonical_json, canonical_sha256
from sanos.datasets import InternalDatasetConfig
from sanos.datasets.internal import DatasetValidationError, validate_fitted_partition
from sanos.model import MkArbFree
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
    discrete_dlv_pricer,
    enrich_daily_frame,
    fit_initial_no_arbitrage,
    fit_smooth_sanos,
    initial_no_arb_pricer,
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


def _two_expiry_raw_inputs(
    *,
    first_is_am: bool = False,
    same_expiry_date: bool = False,
) -> DailyRawInputs:
    trading_date = date(2024, 1, 3)
    spot = 100.0
    second_expiry = date(2024, 1, 19) if same_expiry_date else date(2024, 2, 16)
    second_volatility = 0.42 if same_expiry_date else 0.25
    expiry_specs = (
        (date(2024, 1, 19), 101.0, np.arange(90.0, 113.0, 2.0), 0.40),
        (second_expiry, 102.0, np.arange(88.0, 117.0, 2.0), second_volatility),
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
        assert np.all(np.diff(initial_strikes) > 0.0)
        assert np.min(initial_density) >= -1.0e-7
        assert np.sum(initial_density) == pytest.approx(1.0, abs=1.0e-4)
        assert initial_strikes @ initial_density == pytest.approx(1.0, abs=1.0e-4)
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
        set(event) >= {"input_expiry_index", "expiry_attempt"}
        for event in diagnostics["solver_events"]
        if event["stage"] in {"expiry_soft", "expiry_hard"}
    )
    assert all(
        event["status"] in {"optimal", "optimal_inaccurate"}
        for event in diagnostics["solver_events"]
        if event["stage"] != "expiry_decision"
    )
    dlv_diagnostics = json.loads(
        fitted.get_column("discrete_dlv_diagnostics_json").item(0)
    )
    assert dlv_diagnostics["schema"] == "sanos-discrete-dlv-projection-diagnostics-v3"
    assert dlv_diagnostics["source"] == "smooth_sanos_surface"
    assert dlv_diagnostics["objective_weighting"] == "uniform"
    assert dlv_diagnostics["fit_call_max_abs"] == 9.0e-5
    assert dlv_diagnostics["solver_kkt_tolerance"] == 1.0e-10
    assert dlv_diagnostics["constraint_scaling"] == "geometric_box_and_bound"
    assert dlv_diagnostics["constraint_dlv_bounds"] == [0.01001, 2.99999]
    assert dlv_diagnostics["density_mass_floor"] == 1.0e-8
    assert dlv_diagnostics["stabilization_cycles"] == 1
    assert dlv_diagnostics["projection_domain_nms"] == [-3.0, 2.0]
    assert dlv_diagnostics["inner_nodes"] == 200
    assert dlv_diagnostics["singular_nodes"] == 0
    assert dlv_diagnostics["canonical_density_inverse_max_abs"] <= 1.0e-12
    assert dlv_diagnostics["generic_density_inverse_max_abs"] <= 1.0e-12
    assert dlv_diagnostics["canonical_dlv_fixed_point_max_abs"] <= 1.0e-9
    assert dlv_diagnostics["canonical_call_inverse"]["max"] <= 1.0e-12
    assert dlv_diagnostics["generic_call_inverse"]["max"] <= 1.0e-12
    target_band = dlv_diagnostics["target_fit_price_band"]
    assert target_band["conditioned_nodes"] > 0
    assert len(target_band["conditioned_nodes_by_expiry"]) == dlv_diagnostics["expiries"]
    assert target_band["maximum_absolute_iv_error"] <= 5.0e-3
    assert target_band["maximum_violation"] <= 1.0e-10
    assert target_band["minimum_lower_slack"] >= -1.0e-10
    assert target_band["minimum_upper_slack"] >= -1.0e-10
    assert set(dlv_diagnostics["conditioned_annual_iv_by_source"]) == {
        "atm",
        "listed",
        "midpoints",
        "smooth_support",
        "targets",
    }
    regularization = dlv_diagnostics["numerical_regularization"]
    assert regularization["density_mass_floor"] == 1.0e-8
    assert regularization["constraint_dlv_buffer"] == 1.0e-5
    assert regularization["required_density_floor_mass_per_expiry"] == pytest.approx(
        202.0e-8
    )
    assert regularization["required_density_floor_mass_all_expiries"] == pytest.approx(
        202.0e-8 * dlv_diagnostics["expiries"]
    )
    assert regularization["dlv_epsilon_warning_count"] == 0
    for stage in ("pre_stabilization_state", "post_stabilization_state"):
        assert regularization[stage]["density_floor_shortfall"] <= 1.0e-10
        assert regularization[stage]["published_minimum_dlv_square_distance"] >= (
            -1.0e-9
        )
        assert regularization[stage]["published_maximum_dlv_square_distance"] >= (
            -1.0e-9
        )
    assert dlv_diagnostics["projection_call"]["max"] <= 1.0e-4
    assert dlv_diagnostics["conditioned_annual_iv"]["max"] <= 5.0e-3
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
    assert fitted.get_column("discrete_dlv_configuration_id").unique().to_list() == [
        canonical_sha256({"discrete_dlv": config.discrete_dlv})
    ]
    anchors = fitted.filter(pl.col("discrete_dlv_anchor")).sort(
        "discrete_dlv_input_expiry_index"
    )
    assert anchors.height == fitted.get_column("smooth_input_expiry_index").n_unique()
    for row in anchors.iter_rows(named=True):
        support = np.asarray(row["discrete_dlv_model_strikes"])
        density = np.asarray(row["discrete_dlv_model_density"])
        squared_dlv = np.asarray(row["discrete_dlv_squares"])
        assert len(support) == config.discrete_dlv.inner_nodes + 2
        assert support.shape == density.shape == squared_dlv.shape
        assert squared_dlv[0] == squared_dlv[-1] == 0.0
        assert np.min(density) >= 0.0
        assert np.sum(density) == pytest.approx(1.0, abs=1.0e-10)
        assert support @ density == pytest.approx(1.0, abs=1.0e-10)

    initial_pricer = initial_no_arb_pricer(fitted)
    dlv_pricer = discrete_dlv_pricer(fitted)
    smooth_pricer = smooth_sanos_pricer(fitted)
    stored_dlvs = [
        np.asarray(value, dtype=np.float64)
        for value in anchors.get_column("discrete_dlv_squares").to_list()
    ]
    decoded_densities = dlv_pricer.densities_from_dlvsqs(stored_dlvs)
    canonical_densities = surface_module._canonical_discrete_dlv_densities(
        dlv_pricer,
        stored_dlvs,
    )
    for generic, canonical, expected in zip(
        decoded_densities, canonical_densities, dlv_pricer.xdensities, strict=True
    ):
        np.testing.assert_allclose(generic, expected, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(canonical, expected, rtol=0.0, atol=1.0e-12)
    decoded_pricer = replace(dlv_pricer, xdensities=decoded_densities)
    irregular_query = np.asarray([0.713, 0.947, 1.037, 1.289], dtype=np.float64)
    for expiry in dlv_pricer.expiries:
        np.testing.assert_allclose(
            decoded_pricer(expiry, irregular_query),
            dlv_pricer(expiry, irregular_query),
            rtol=0.0,
            atol=2.0e-14,
        )
    for expiry_index, expiry in enumerate(initial_pricer.expiries):
        expiry_rows = fitted.filter(
            pl.col("initial_no_arb_input_expiry_index") == expiry_index
        )
        observed = initial_pricer(
            expiry, expiry_rows.get_column("normalized_strike").to_numpy()
        )
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
        np.testing.assert_allclose(
            observed,
            expiry_rows.get_column("smooth_surface_fit_price").to_numpy(),
            rtol=0.0,
            atol=5.0e-8,
        )


def test_canonical_dlv_serialization_is_stable_with_vanishing_tail_mass(
    tmp_path: Path,
) -> None:
    expiries = np.arange(1, 21, dtype=np.float64) / 252.0
    support = np.linspace(0.8, 1.2, 22, dtype=np.float64)
    template_density = np.full(22, 1.0 / 22.0, dtype=np.float64)
    template = surface_module.SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.2**2,
        xstrikes=[support.copy() for _ in expiries],
        xdensities=[template_density.copy() for _ in expiries],
    )
    input_dlvs = [
        np.concatenate(([0.0], np.full(20, 0.2**2), [0.0]))
        for _ in expiries
    ]
    densities = surface_module._canonical_discrete_dlv_densities(
        template,
        input_dlvs,
    )
    pricer = replace(template, xdensities=densities)
    stored_dlvs = surface_module._canonical_discrete_dlv_squares(pricer)
    assert min(float(np.min(value[1:-1])) for value in densities) < 1.0e-8

    path = tmp_path / "canonical-dlv.parquet"
    pl.DataFrame(
        {
            "expiry": expiries,
            "variance": pricer.variances,
            "support": pricer.xstrikes,
            "density": pricer.xdensities,
            "dlv_squares": stored_dlvs,
        }
    ).write_parquet(path)
    reopened = pl.read_parquet(path)
    restored = surface_module.SanosPricer(
        smoothness=None,
        expiries=reopened.get_column("expiry").to_numpy(),
        variances=reopened.get_column("variance").to_numpy(),
        xstrikes=[np.asarray(value) for value in reopened["support"].to_list()],
        xdensities=[np.asarray(value) for value in reopened["density"].to_list()],
    )
    restored_dlvs = [
        np.asarray(value) for value in reopened["dlv_squares"].to_list()
    ]

    canonical_encoded = surface_module._canonical_discrete_dlv_squares(restored)
    canonical_decoded = surface_module._canonical_discrete_dlv_densities(
        restored,
        restored_dlvs,
    )
    generic_decoded = restored.densities_from_dlvsqs(restored_dlvs)
    assert max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(canonical_encoded, restored_dlvs, strict=True)
    ) <= 1.0e-9
    assert max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(canonical_decoded, restored.xdensities, strict=True)
    ) <= 1.0e-12
    assert max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(generic_decoded, restored.xdensities, strict=True)
    ) <= 1.0e-12

    generic_pricer = replace(restored, xdensities=generic_decoded)
    queries = np.asarray([0.83, 0.947, 1.037, 1.17], dtype=np.float64)
    assert max(
        float(
            np.max(
                np.abs(
                    generic_pricer(float(expiry), queries)
                    - restored(float(expiry), queries)
                )
            )
        )
        for expiry in expiries
    ) <= 1.0e-12

    mixed_reencoded = surface_module._canonical_discrete_dlv_squares(
        generic_pricer
    )
    assert max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(mixed_reencoded, restored_dlvs, strict=True)
    ) > 1.0e-9

    perturbed_dlv = [value.copy() for value in restored_dlvs]
    perturbed_dlv[-1][1] += 1.0e-8
    assert max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(canonical_encoded, perturbed_dlv, strict=True)
    ) > 1.0e-9

    perturbed_density = [value.copy() for value in restored.xdensities]
    perturbed_density[-1][1] += 2.0e-10
    corrupted = replace(restored, xdensities=perturbed_density)
    assert max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(canonical_decoded, corrupted.xdensities, strict=True)
    ) > 1.0e-12


def test_projection_target_iv_price_bands_are_exact_and_threshold_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expiry = np.asarray([2.0 / 252.0], dtype=np.float64)
    inner = np.asarray([[0.9999, 1.0, 1.0001]], dtype=np.float64)
    target_iv = np.asarray([0.003, 0.003, 0.003], dtype=np.float64)
    target_calls = np.asarray(
        [surface_module._bs_call_price(inner[0], target_iv, float(expiry[0]))]
    )

    class SmoothTarget:
        def __call__(
            self,
            query_expiry: float,
            strikes: np.ndarray[Any, Any],
            *,
            and_vols: bool = False,
        ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | np.ndarray[Any, Any]:
            del query_expiry
            calls = surface_module._bs_call_price(
                np.asarray(strikes),
                np.full_like(strikes, target_iv[0]),
                float(expiry[0]),
            )
            return (calls, np.full_like(strikes, target_iv[0])) if and_vols else calls

    threshold = 5.0e-2
    monkeypatch.setattr(
        surface_module,
        "_bs_vega_sqrt_time",
        lambda *args, **kwargs: np.asarray(
            [
                np.nextafter(threshold, 0.0),
                threshold,
                np.nextafter(threshold, np.inf),
            ],
            dtype=np.float64,
        ),
    )
    config = DailySurfaceConfig().discrete_dlv
    bounds, diagnostics = surface_module._projection_fit_price_bounds(
        smooth=cast(Any, SmoothTarget()),
        expiries=expiry,
        inner=inner,
        target_calls=target_calls,
        config=config,
    )
    lower, upper = bounds[0]
    assert diagnostics["conditioned_nodes"] == 2
    assert diagnostics["conditioned_nodes_by_expiry"] == [2]
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= target_calls[0])
    assert np.all(target_calls[0] <= upper)
    # A sub-tolerance target IV uses zero volatility for the lower Black band.
    expected_zero_vol = surface_module._bs_call_price(
        inner[0][1:],
        np.zeros(2, dtype=np.float64),
        float(expiry[0]),
    )
    np.testing.assert_allclose(
        lower[1:],
        np.maximum(
            target_calls[0][1:] - config.fit_call_max_abs,
            expected_zero_vol + config.solver_kkt_tolerance,
        ),
        rtol=0.0,
        atol=0.0,
    )
    at_bounds = target_calls.copy()
    at_bounds[0, 1] = lower[1]
    at_bounds[0, 2] = upper[2]
    at_metrics = surface_module._projection_target_band_metrics(
        smooth=cast(Any, SmoothTarget()),
        expiries=expiry,
        inner=inner,
        fitted_calls=at_bounds,
        bounds=bounds,
        config=config,
    )
    assert at_metrics["maximum_violation"] == 0.0

    below = at_bounds.copy()
    below[0, 1] = np.nextafter(lower[1], -np.inf)
    below_metrics = surface_module._projection_target_band_metrics(
        smooth=cast(Any, SmoothTarget()),
        expiries=expiry,
        inner=inner,
        fitted_calls=below,
        bounds=bounds,
        config=config,
    )
    assert below_metrics["maximum_violation"] > 0.0

    above = at_bounds.copy()
    above[0, 2] = np.nextafter(upper[2], np.inf)
    above_metrics = surface_module._projection_target_band_metrics(
        smooth=cast(Any, SmoothTarget()),
        expiries=expiry,
        inner=inner,
        fitted_calls=above,
        bounds=bounds,
        config=config,
    )
    assert above_metrics["maximum_violation"] > 0.0

    regression_strike = np.asarray([1.0156393019564227], dtype=np.float64)
    regression_expiry = 0.007936508394777775
    regression_iv = np.asarray([0.09748547049469465], dtype=np.float64)
    lower_regression = surface_module._bs_call_price(
        regression_strike,
        regression_iv - 5.0e-3,
        regression_expiry,
    )[0]
    upper_regression = surface_module._bs_call_price(
        regression_strike,
        regression_iv + 5.0e-3,
        regression_expiry,
    )[0]
    assert lower_regression == pytest.approx(9.580920912291915e-5, abs=1.0e-16)
    assert upper_regression == pytest.approx(1.684019810493792e-4, abs=1.0e-16)
    assert 7.637951934297273e-5 < lower_regression


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
            observed.get_column("smooth_surface_fit_price").to_numpy(),
            smooth_reference["fitsX"][source_index],
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


def _solver_event(
    stage: str,
    status: str,
    *,
    fallback: bool = False,
    input_expiry_index: int | None = None,
    expiry_attempt: int | None = None,
    objective: float | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "stage": stage,
        "solver": "CLARABEL" if fallback else "HIGHS",
        "algorithm": "simplex",
        "status": status,
        "is_fallback": fallback,
        "for_expiry": stage.startswith("expiry_"),
        "objective": (
            1.25 if objective is None and status.startswith("optimal") else objective
        ),
    }
    if stage.startswith("expiry_"):
        event["input_expiry_index"] = input_expiry_index
        event["expiry_attempt"] = expiry_attempt
    return event


def _expiry_decision(
    input_expiry_index: int,
    *,
    retained: bool,
    support_attempt: int | None,
) -> dict[str, object]:
    return {
        "stage": "expiry_decision",
        "input_expiry_index": input_expiry_index,
        "retained": retained,
        "support_attempt": support_attempt,
    }


def test_solver_event_contract_accepts_documented_hard_fit_fallbacks() -> None:
    events = [
        _solver_event(
            "expiry_soft",
            "user_limit",
            input_expiry_index=0,
            expiry_attempt=1,
            objective=2.0,
        ),
        _solver_event(
            "expiry_soft",
            "optimal",
            fallback=True,
            input_expiry_index=0,
            expiry_attempt=1,
        ),
        _solver_event(
            "expiry_hard",
            "solver_error",
            input_expiry_index=0,
            expiry_attempt=1,
        ),
        _solver_event(
            "expiry_hard",
            "infeasible_inaccurate",
            fallback=True,
            input_expiry_index=0,
            expiry_attempt=1,
        ),
        _solver_event(
            "expiry_soft",
            "optimal_inaccurate",
            input_expiry_index=0,
            expiry_attempt=2,
        ),
        _expiry_decision(0, retained=True, support_attempt=2),
        _solver_event("surface", "optimal"),
    ]
    internal_dataset_module._terminal_solver_events(
        {"outcome": "completed", "solver_events": events},
        "test",
        expected_expiry_indices=(0,),
        retained_expiry_indices=(0,),
    )
    internal_dataset_module._terminal_solver_events(
        {
            "outcome": "completed",
            "solver_events": [_solver_event("discrete_dlv_surface", "optimal")],
        },
        "test DLV",
        expected_expiry_indices=None,
    )
    dropped = [
        _solver_event(
            "expiry_soft",
            "optimal",
            input_expiry_index=0,
            expiry_attempt=1,
        ),
        _solver_event(
            "expiry_hard",
            "infeasible",
            input_expiry_index=0,
            expiry_attempt=1,
        ),
        _expiry_decision(0, retained=False, support_attempt=None),
        _solver_event("surface", "optimal"),
    ]
    assert (
        internal_dataset_module._terminal_solver_events(
            {"outcome": "completed", "solver_events": dropped},
            "dropped test",
            expected_expiry_indices=(0,),
            retained_expiry_indices=(),
        )
        == frozenset()
    )


def test_support_mask_contract_allows_superset_only_after_second_hard_failure() -> None:
    support = [0.7, 0.9, 1.0, 1.1, 1.3]
    frame = pl.DataFrame(
        {
            "expiry_id": [202401191, 202401191, 202401191],
            "normalized_strike": [0.9, 1.0, 1.1],
            "smooth_fit_mask": [True, False, True],
            "smooth_input_expiry_index": pl.Series([0, 0, 0], dtype=pl.Int32),
            "smooth_expiry_model_strikes": [support, support, support],
        }
    )
    internal_dataset_module._validate_support_mask_geometry(
        frame,
        mask_column="smooth_fit_mask",
        state_index_column="smooth_input_expiry_index",
        expiry_strikes_column="smooth_expiry_model_strikes",
        allow_masked_support_superset=frozenset({0}),
        label="test",
    )
    with pytest.raises(DatasetValidationError, match="exact retained expiry support"):
        internal_dataset_module._validate_support_mask_geometry(
            frame,
            mask_column="smooth_fit_mask",
            state_index_column="smooth_input_expiry_index",
            expiry_strikes_column="smooth_expiry_model_strikes",
            allow_masked_support_superset=frozenset(),
            label="test",
        )
    exact = frame.with_columns(pl.lit(True).alias("smooth_fit_mask"))
    with pytest.raises(DatasetValidationError, match="strict ordered subset"):
        internal_dataset_module._validate_support_mask_geometry(
            exact,
            mask_column="smooth_fit_mask",
            state_index_column="smooth_input_expiry_index",
            expiry_strikes_column="smooth_expiry_model_strikes",
            allow_masked_support_superset=frozenset({0}),
            label="test",
        )


@pytest.mark.parametrize(
    ("branch", "retained_indices", "relaxed_support"),
    [
        ("drop_after_hard_one", (1,), frozenset()),
        ("hard_one_boundary_retained", (0, 1), frozenset()),
        ("attempt_two_success", (0, 1), frozenset()),
        ("attempt_two_boundary_retained", (0, 1), frozenset()),
        ("attempt_two_hard_failure", (0, 1), frozenset({0})),
    ],
)
def test_mkarbfree_hard_fit_branches_match_diagnostic_population_contract(
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    retained_indices: tuple[int, ...],
    relaxed_support: frozenset[int],
) -> None:
    expiries = np.asarray([10.0 / 252.0, 20.0 / 252.0], dtype=np.float64)
    strikes = [
        np.asarray([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64)
        for _ in expiries
    ]
    mids = [
        np.asarray(bs.price(k=value, vol=0.25, sqrtT=np.sqrt(expiry)), dtype=np.float64)
        for expiry, value in zip(expiries, strikes, strict=True)
    ]
    bids = [value - 0.01 for value in mids]
    asks = [value + 0.01 for value in mids]
    events: list[dict[str, object]] = []

    def density(support: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        result = np.zeros(len(support), dtype=np.float64)
        result[int(np.flatnonzero(support == 1.0)[0])] = 1.0
        return result

    def emit(
        self: MkArbFree,
        stage: str,
        status: str,
        context: dict[str, object] | None,
    ) -> None:
        if self._solver_diagnostic_sink is None:
            return
        self._solver_diagnostic_sink(
            {
                **_solver_event(
                    stage,
                    status,
                    objective=None if status == "infeasible" else 1.0,
                ),
                **({} if context is None else context),
            }
        )

    def scripted_surface(
        self: MkArbFree,
        *,
        strikes: list[np.ndarray[Any, Any]],
        bids: list[np.ndarray[Any, Any]],
        asks: list[np.ndarray[Any, Any]],
        masks: list[np.ndarray[Any, Any]] | None,
        xstrikes: list[np.ndarray[Any, Any]],
        diagnostic_stage: str = "surface",
        diagnostic_context: dict[str, object] | None = None,
        hard_fit: bool = False,
        **kwargs: Any,
    ) -> Any:
        del masks, kwargs
        if diagnostic_stage == "surface":
            emit(self, diagnostic_stage, "optimal", diagnostic_context)
            return SimpleNamespace(
                fits=[0.5 * (bid + ask) for bid, ask in zip(bids, asks, strict=True)],
                xdensities=[density(value) for value in xstrikes],
            )
        assert diagnostic_context is not None
        input_index = int(diagnostic_context["input_expiry_index"])
        attempt = int(diagnostic_context["expiry_attempt"])
        if hard_fit:
            emit(self, diagnostic_stage, "infeasible", diagnostic_context)
            return None
        emit(self, diagnostic_stage, "optimal", diagnostic_context)
        fit = 0.5 * (bids[0] + asks[0])
        if input_index == 1:
            return SimpleNamespace(
                fits=[fit],
                xdensities=[density(xstrikes[0])],
            )
        if branch == "hard_one_boundary_retained":
            outside = bids[0] - self._eps_price
            inside: list[int] = []
        else:
            outside = fit + 0.1
        if branch == "hard_one_boundary_retained":
            pass
        elif attempt == 1:
            inside = [2] if branch == "drop_after_hard_one" else [1, 2, 3]
        elif branch == "attempt_two_success":
            inside = [0, 1, 2, 3, 4]
        elif branch == "attempt_two_boundary_retained":
            outside = bids[0] - self._eps_price
            inside = []
        else:
            inside = [1, 2]
        outside[inside] = fit[inside]
        return SimpleNamespace(
            fits=[outside],
            xdensities=[density(xstrikes[0])],
        )

    monkeypatch.setattr(MkArbFree, "surface_densities", scripted_surface)
    result = MkArbFree(
        "surface",
        min_strikes_per_expiry=3,
        _solver_diagnostic_sink=lambda event: events.append(dict(event)),
    )(
        expiries,
        strikes,
        bids,
        asks,
        vols=0.25,
        verbose=cast(Any, Context).quiet,
    )
    assert tuple(int(value) for value in result.input_index) == retained_indices
    observed_relaxed = internal_dataset_module._terminal_solver_events(
        {"outcome": "completed", "solver_events": events},
        "scripted MkArbFree",
        expected_expiry_indices=(0, 1),
        retained_expiry_indices=retained_indices,
    )
    assert observed_relaxed == relaxed_support
    if 0 not in retained_indices:
        return
    output_index = retained_indices.index(0)
    interior = np.asarray(result.xstrikes[output_index])[1:-1]
    masked = strikes[0][np.asarray(result.masks[output_index], dtype=bool)]
    if branch in {"attempt_two_success", "attempt_two_boundary_retained"}:
        np.testing.assert_array_equal(interior, masked)
        np.testing.assert_array_equal(masked, strikes[0][[1, 2, 3]])
    elif branch == "hard_one_boundary_retained":
        np.testing.assert_array_equal(interior, masked)
        np.testing.assert_array_equal(masked, strikes[0])
    else:
        assert len(interior) > len(masked)
        np.testing.assert_array_equal(interior, strikes[0])
        np.testing.assert_array_equal(masked, strikes[0][[1, 2, 3]])


def test_solver_call_parser_rejects_malformed_sequences() -> None:
    soft = _solver_event(
        "expiry_soft",
        "optimal",
        input_expiry_index=0,
        expiry_attempt=1,
    )
    unknown = dict(soft)
    unknown["stage"] = "mystery"
    wrong_bool = dict(soft)
    wrong_bool["is_fallback"] = 0
    success_without_objective = dict(soft)
    success_without_objective["objective"] = None
    orphan = _solver_event(
        "expiry_soft",
        "optimal",
        fallback=True,
        input_expiry_index=0,
        expiry_attempt=1,
    )
    dangling = _solver_event(
        "expiry_soft",
        "solver_error",
        input_expiry_index=0,
        expiry_attempt=1,
    )
    trigger = _solver_event(
        "expiry_soft",
        "user_limit",
        input_expiry_index=0,
        expiry_attempt=1,
    )
    mismatched_fallback = _solver_event(
        "expiry_soft",
        "optimal",
        fallback=True,
        input_expiry_index=1,
        expiry_attempt=1,
    )
    cases = [
        ("unknown solver stage", [unknown]),
        ("invalid boolean fields", [wrong_bool]),
        ("success has no finite objective", [success_without_objective]),
        ("orphan fallback event", [orphan]),
        ("dangling fallback trigger", [dangling]),
        ("mismatched fallback event", [trigger, mismatched_fallback]),
    ]
    for expected, events in cases:
        with pytest.raises(DatasetValidationError, match=expected):
            internal_dataset_module._solver_calls(events, "test")


def test_solver_event_contract_rejects_failed_or_misordered_solves() -> None:
    soft = _solver_event(
        "expiry_soft",
        "optimal",
        input_expiry_index=0,
        expiry_attempt=1,
    )
    hard = _solver_event(
        "expiry_hard",
        "infeasible",
        input_expiry_index=0,
        expiry_attempt=1,
    )
    decision = _expiry_decision(0, retained=True, support_attempt=1)
    surface = _solver_event("surface", "optimal")
    valid = [soft, decision, surface]
    failed_soft = dict(soft)
    failed_soft.update(status="infeasible", objective=None)
    failed_hard = dict(hard)
    failed_hard["status"] = "unbounded"
    failed_surface = dict(surface)
    failed_surface.update(status="infeasible", objective=None)
    retry_without_failed_hard = [
        soft,
        _solver_event(
            "expiry_hard",
            "optimal",
            input_expiry_index=0,
            expiry_attempt=1,
        ),
        _solver_event(
            "expiry_soft",
            "optimal",
            input_expiry_index=0,
            expiry_attempt=2,
        ),
    ]
    cases = [
        ("does not declare a completed outcome", {"outcome": "failed", "solver_events": valid}),
        ("has no non-empty solver_events list", {"outcome": "completed", "solver_events": []}),
        (
            "has invalid solve ordering",
            {"outcome": "completed", "solver_events": [hard, decision, surface]},
        ),
        (
            "expiry_soft ends in unexpected state",
            {"outcome": "completed", "solver_events": [failed_soft, decision, surface]},
        ),
        (
            "expiry_hard ends in unexpected state",
            {"outcome": "completed", "solver_events": [soft, failed_hard, decision, surface]},
        ),
        (
            "does not end in one successful surface solve",
            {"outcome": "completed", "solver_events": [soft, decision, failed_surface]},
        ),
        (
            "does not end in one successful surface solve",
            {"outcome": "completed", "solver_events": [soft, decision, surface, surface]},
        ),
        (
            "expiry solve order/count changed",
            {"outcome": "completed", "solver_events": [surface, soft, decision, surface]},
        ),
        (
            "retries without a failed hard fit",
            {
                "outcome": "completed",
                "solver_events": [
                    *retry_without_failed_hard,
                    _expiry_decision(0, retained=True, support_attempt=2),
                    surface,
                ],
            },
        ),
    ]
    for expected, diagnostics in cases:
        with pytest.raises(DatasetValidationError, match=expected):
            internal_dataset_module._terminal_solver_events(
                diagnostics,
                "test",
                expected_expiry_indices=(0,),
                retained_expiry_indices=(0,),
            )


def test_solver_event_contract_rejects_malformed_decisions() -> None:
    soft = _solver_event(
        "expiry_soft",
        "optimal",
        input_expiry_index=0,
        expiry_attempt=1,
    )
    surface = _solver_event("surface", "optimal")
    valid_decision = _expiry_decision(0, retained=True, support_attempt=1)

    missing_field = dict(valid_decision)
    missing_field.pop("support_attempt")
    extra_field = {**valid_decision, "unexpected": None}
    wrong_index = {**valid_decision, "input_expiry_index": 1}
    bool_index = {**valid_decision, "input_expiry_index": True}
    non_boolean_retained = {**valid_decision, "retained": 1}
    invalid_support = {**valid_decision, "support_attempt": 3}
    boolean_support = {**valid_decision, "support_attempt": True}
    conflicting_drop = _expiry_decision(0, retained=False, support_attempt=1)
    conflicting_retained = _expiry_decision(0, retained=True, support_attempt=None)
    cases = [
        ("decision fields changed", [soft, missing_field, surface], (0,)),
        ("decision fields changed", [soft, extra_field, surface], (0,)),
        ("decision is malformed", [soft, wrong_index, surface], (0,)),
        ("decision is malformed", [soft, bool_index, surface], (0,)),
        ("decision is malformed", [soft, non_boolean_retained, surface], (0,)),
        ("decision is malformed", [soft, invalid_support, surface], (0,)),
        ("decision is malformed", [soft, boolean_support, surface], (0,)),
        ("decision conflicts with its solve history", [soft, conflicting_drop, surface], ()),
        (
            "decision conflicts with its solve history",
            [soft, conflicting_retained, surface],
            (0,),
        ),
        ("has invalid solve ordering", [valid_decision, soft, surface], (0,)),
        (
            "unknown solver stage 'expiry_decision'",
            [soft, valid_decision, valid_decision, surface],
            (0,),
        ),
        ("retained-expiry population changed", [soft, valid_decision, surface], ()),
    ]
    for expected, events, retained in cases:
        with pytest.raises(DatasetValidationError, match=expected):
            internal_dataset_module._terminal_solver_events(
                {"outcome": "completed", "solver_events": events},
                "test",
                expected_expiry_indices=(0,),
                retained_expiry_indices=retained,
            )

    fallback_dlv = [
        _solver_event("discrete_dlv_surface", "solver_error"),
        _solver_event("discrete_dlv_surface", "optimal", fallback=True),
    ]
    with pytest.raises(DatasetValidationError):
        internal_dataset_module._terminal_solver_events(
            {"outcome": "completed", "solver_events": fallback_dlv},
            "test DLV",
            expected_expiry_indices=None,
        )


def test_partition_validator_couples_hard_fit_failure_to_retained_state(
    phase6_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> None:
    fitted = phase6_frames[-1]
    config = InternalDatasetConfig(
        name="solver-event-validation",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )
    diagnostics = json.loads(fitted.get_column("smooth_diagnostics_json").item(0))
    events = diagnostics["solver_events"]
    soft_position = next(
        index
        for index, event in enumerate(events)
        if event["stage"] == "expiry_soft"
        and event["input_expiry_index"] == 0
        and event["expiry_attempt"] == 1
    )
    assert events[soft_position]["is_fallback"] is False
    hard_failure = dict(events[soft_position])
    hard_failure.update(stage="expiry_hard", status="infeasible", objective=None)
    retry_soft = dict(events[soft_position])
    retry_soft["expiry_attempt"] = 2
    diagnostics["solver_events"] = [
        *events[: soft_position + 1],
        hard_failure,
        retry_soft,
        *events[soft_position + 1 :],
    ]
    decision = next(
        event
        for event in diagnostics["solver_events"]
        if event["stage"] == "expiry_decision"
        and event["input_expiry_index"] == 0
    )
    decision["support_attempt"] = 2
    accepted = fitted.with_columns(
        pl.lit(canonical_json(diagnostics)).alias("smooth_diagnostics_json")
    )
    validate_fitted_partition(
        accepted,
        config=config,
        trading_date=date(2024, 1, 3),
        typed_empty_reason=None,
    )

    wrong_population = json.loads(json.dumps(diagnostics))
    wrong_decision = next(
        event
        for event in wrong_population["solver_events"]
        if event["stage"] == "expiry_decision"
        and event["input_expiry_index"] == 0
    )
    wrong_decision.update(retained=False, support_attempt=None)
    failed_surface = json.loads(json.dumps(diagnostics))
    failed_surface["solver_events"][-1].update(
        status="infeasible",
        objective=None,
    )
    for corrupted_diagnostics in (wrong_population, failed_surface):
        corrupted = fitted.with_columns(
            pl.lit(canonical_json(corrupted_diagnostics)).alias(
                "smooth_diagnostics_json"
            )
        )
        with pytest.raises(DatasetValidationError):
            validate_fitted_partition(
                corrupted,
                config=config,
                trading_date=date(2024, 1, 3),
                typed_empty_reason=None,
            )


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
        "smooth_surface_fit_price": fitted.with_columns(
            (pl.col("smooth_surface_fit_price") + 0.01).alias(
                "smooth_surface_fit_price"
            )
        ),
        "smooth_fit_summary": fitted.with_columns(
            (pl.col("smooth_pcnt_fit") + 0.01).alias("smooth_pcnt_fit")
        ),
        "discrete_dlv_time_increment_years": fitted.with_columns(
            pl.when(pl.col("discrete_dlv_anchor"))
            .then(pl.col("discrete_dlv_time_increment_years") + 0.001)
            .otherwise(pl.col("discrete_dlv_time_increment_years"))
            .alias("discrete_dlv_time_increment_years")
        ),
        "discrete_dlv_atm_annual_volatility": fitted.with_columns(
            pl.when(pl.col("discrete_dlv_anchor"))
            .then(pl.col("discrete_dlv_atm_annual_volatility") + 0.001)
            .otherwise(pl.col("discrete_dlv_atm_annual_volatility"))
            .alias("discrete_dlv_atm_annual_volatility")
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
    assert fitted.get_column("discrete_dlv_configuration_id").unique().to_list() == [
        canonical_sha256({"discrete_dlv": config.discrete_dlv})
    ]
    assert not fitted.get_column("discrete_dlv_anchor").any()


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
    assert {
        "_canonical_fitted_rows",
        "_float_column",
        "_float32_column",
        "_require_schema",
    } <= smooth_dependencies
    assert direct_dependency_names(selected_reference_view) == {"_require_schema"}


def test_fitted_output_canonicalizes_same_date_am_before_pm() -> None:
    paired = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_two_expiry_raw_inputs(first_is_am=True, same_expiry_date=True),
    )
    paired_expiry_ids = paired.get_column("expiry_id").unique(maintain_order=True).to_list()
    assert paired_expiry_ids == sorted(paired_expiry_ids, reverse=True)

    enriched = enrich_daily_frame(paired)
    no_arb = fit_initial_no_arbitrage(enriched)
    filtered = apply_options_filter(no_arb)
    fitted = fit_smooth_sanos(filtered)

    fitted_expiry_ids = fitted.get_column("expiry_id").unique(maintain_order=True).to_list()
    assert fitted_expiry_ids == sorted(fitted_expiry_ids)
    expected = fitted.sort("expiry_id", "normalized_strike", maintain_order=True)
    assert fitted.equals(expected)
    assert fitted.select("expiry_id", "normalized_strike").equals(
        filtered.select("expiry_id", "normalized_strike").sort(
            "expiry_id", "normalized_strike", maintain_order=True
        )
    )

    none_selected = filtered.with_columns(
        pl.lit(False).alias("options_filter_eligible")
    )
    empty_smooth = fit_smooth_sanos(none_selected)
    assert empty_smooth.equals(
        empty_smooth.sort("expiry_id", "normalized_strike", maintain_order=True)
    )
    assert empty_smooth.select("expiry_id", "normalized_strike").equals(
        filtered.select("expiry_id", "normalized_strike").sort(
            "expiry_id", "normalized_strike", maintain_order=True
        )
    )


def test_black_vector_validator_uses_price_not_ill_conditioned_inverse() -> None:
    frame = pl.DataFrame(
        {
            "normalized_strike": pl.Series(
                [0.7527551054954529], dtype=pl.Float32
            ),
            "time_to_expiry_years": pl.Series(
                [0.011904762126505375], dtype=pl.Float32
            ),
            "pure_bid_price": pl.Series([0.24725332856178284], dtype=pl.Float32),
            "bid_implied_volatility": pl.Series(
                [0.7817830443382263], dtype=pl.Float32
            ),
            "pure_bid_delta": pl.Series([0.9996272325515747], dtype=pl.Float32),
            "pure_bid_gamma": pl.Series([0.015867725014686584], dtype=pl.Float32),
            "pure_bid_vega": pl.Series(
                [0.00014767998072784394], dtype=pl.Float32
            ),
            "pure_bid_theta": pl.Series([0.004849055781960487], dtype=pl.Float32),
            "pure_bid_dk": pl.Series([-0.9994936585426331], dtype=pl.Float32),
        }
    )
    residuals = internal_dataset_module._black_vector_residual(
        frame,
        price_column="pure_bid_price",
        volatility_column="bid_implied_volatility",
        greek_prefix="pure_bid",
        strike_was_persisted_input=False,
        volatility_minimum=0.01,
        volatility_maximum=5.0,
    )
    direct_inverse = np.asarray(
        bs.implied(
            frame.get_column("normalized_strike").to_numpy().astype(np.float64),
            frame.get_column("pure_bid_price").to_numpy().astype(np.float64),
            sqrtT=np.sqrt(
                frame.get_column("time_to_expiry_years").to_numpy().astype(np.float64)
            ),
            eps=1.0e-8,
        )
    )
    assert abs(
        float(direct_inverse[0])
        - float(frame.get_column("bid_implied_volatility").item(0))
    ) > 0.006
    assert residuals.price_repricing < 1.0e-7
    assert residuals.maximum_excess == 0.0

    price_corrupted = frame.with_columns(
        (pl.col("pure_bid_price") + 5.0e-7).alias("pure_bid_price")
    )
    price_residuals = internal_dataset_module._black_vector_residual(
        price_corrupted,
        price_column="pure_bid_price",
        volatility_column="bid_implied_volatility",
        greek_prefix="pure_bid",
        strike_was_persisted_input=False,
        volatility_minimum=0.01,
        volatility_maximum=5.0,
    )
    assert price_residuals.price_rounding_excess > 4.0e-7

    greek_corrupted = frame.with_columns(
        (pl.col("pure_bid_gamma") + 0.01).alias("pure_bid_gamma")
    )
    greek_residuals = internal_dataset_module._black_vector_residual(
        greek_corrupted,
        price_column="pure_bid_price",
        volatility_column="bid_implied_volatility",
        greek_prefix="pure_bid",
        strike_was_persisted_input=False,
        volatility_minimum=0.01,
        volatility_maximum=5.0,
    )
    assert greek_residuals.greek_rounding_excess > 0.009


@pytest.mark.parametrize("strike_was_persisted_input", [False, True])
def test_black_vector_envelope_contains_random_float32_preimages(
    strike_was_persisted_input: bool,
) -> None:
    rng = np.random.default_rng(8471)
    count = 512
    time = rng.uniform(0.2 / 252.0, 2.0, size=count).astype(np.float32)
    time[:3] = np.asarray([0.2 / 252.0, 1.0 / 252.0, 2.0], dtype=np.float32)
    sqrt_time = np.sqrt(time.astype(np.float64))
    latent_strike = np.exp(rng.uniform(np.log(0.45), np.log(2.5), size=count))
    strike32 = latent_strike.astype(np.float32)
    if strike_was_persisted_input:
        latent_strike = strike32.astype(np.float64)
    latent_volatility = rng.uniform(0.01, 4.99, size=count)
    latent_volatility[:2] = (0.01, 5.0)
    volatility32 = latent_volatility.astype(np.float32)
    values = bs(
        k=latent_strike,
        vol=latent_volatility,
        sqrtT=sqrt_time,
        eps=1.0e-8,
        what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
    )
    frame = pl.DataFrame(
        {
            "normalized_strike": pl.Series(strike32, dtype=pl.Float32),
            "time_to_expiry_years": pl.Series(time, dtype=pl.Float32),
            "test_volatility": pl.Series(volatility32, dtype=pl.Float32),
            **{
                f"test_{name}": pl.Series(
                    np.asarray(getattr(values, name), dtype=np.float32),
                    dtype=pl.Float32,
                )
                for name in ("price", "delta", "gamma", "vega", "theta", "dk")
            },
        }
    )
    residuals = internal_dataset_module._black_vector_residual(
        frame,
        price_column="test_price",
        volatility_column="test_volatility",
        greek_prefix="test",
        strike_was_persisted_input=strike_was_persisted_input,
        volatility_minimum=0.01,
        volatility_maximum=5.0,
    )
    assert residuals.maximum_excess == 0.0


def test_black_vector_envelope_allows_one_output_ulp_only() -> None:
    strike = np.asarray([1.05], dtype=np.float32)
    volatility = np.asarray([0.37], dtype=np.float32)
    time = np.asarray([0.2 / 252.0], dtype=np.float32)
    sqrt_time = np.sqrt(time.astype(np.float64))
    values = bs(
        k=strike.astype(np.float64),
        vol=volatility.astype(np.float64),
        sqrtT=sqrt_time,
        eps=1.0e-8,
        what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
    )
    names = ("price", "delta", "gamma", "vega", "theta", "dk")
    frame = pl.DataFrame(
        {
            "normalized_strike": pl.Series(strike, dtype=pl.Float32),
            "time_to_expiry_years": pl.Series(time, dtype=pl.Float32),
            "test_volatility": pl.Series(volatility, dtype=pl.Float32),
            **{
                f"test_{name}": pl.Series(
                    np.asarray(getattr(values, name), dtype=np.float32),
                    dtype=pl.Float32,
                )
                for name in names
            },
        }
    )
    envelope = internal_dataset_module._black_value_envelope(
        strikes32=strike,
        volatilities32=volatility,
        sqrt_time=sqrt_time,
        strike_was_persisted_input=True,
        volatility_minimum=0.01,
        volatility_maximum=5.0,
        epsilon=1.0e-8,
    )

    for name in names:
        upper = float(envelope[name][1][0])
        candidate = np.float32(upper)
        accepted = candidate
        for _ in range(8):
            radius = float(
                internal_dataset_module._float32_ulp(np.asarray([candidate]))[0]
            )
            if float(candidate) - upper > radius:
                break
            accepted = candidate
            candidate = np.nextafter(
                candidate, np.float32(np.inf), dtype=np.float32
            )
        else:
            raise AssertionError(f"Could not step beyond the {name} output allowance")

        column = "test_price" if name == "price" else f"test_{name}"
        accepted_frame = frame.with_columns(
            pl.lit(float(accepted), dtype=pl.Float32).alias(column)
        )
        accepted_residuals = internal_dataset_module._black_vector_residual(
            accepted_frame,
            price_column="test_price",
            volatility_column="test_volatility",
            greek_prefix="test",
            strike_was_persisted_input=True,
            volatility_minimum=0.01,
            volatility_maximum=5.0,
        )
        assert accepted_residuals.maximum_excess == 0.0

        rejected_frame = frame.with_columns(
            pl.lit(float(candidate), dtype=pl.Float32).alias(column)
        )
        rejected_residuals = internal_dataset_module._black_vector_residual(
            rejected_frame,
            price_column="test_price",
            volatility_column="test_volatility",
            greek_prefix="test",
            strike_was_persisted_input=True,
            volatility_minimum=0.01,
            volatility_maximum=5.0,
        )
        assert rejected_residuals.maximum_excess > 0.0


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
    assert first_expiry["smooth_model_variance"] is not None
    assert first_expiry["smooth_surface_model_density"] is not None
    assert first_expiry["discrete_dlv_squares"] is not None
    assert fitted.get_column("options_filter_eligible").sum() == fitted.height


def test_dlv_bound_solver_allowance_is_inclusive_and_does_not_clip() -> None:
    minimum_square = 0.01**2
    maximum_square = 3.0**2
    tolerance = 1.0e-9
    lower_limit = minimum_square - tolerance
    upper_limit = maximum_square + tolerance
    values = np.asarray([lower_limit, minimum_square, maximum_square, upper_limit])

    diagnostics = surface_module._validate_dlv_bounds(
        values,
        minimum_square=minimum_square,
        maximum_square=maximum_square,
        absolute_tolerance=tolerance,
        label="test",
    )

    assert diagnostics["minimum"] == lower_limit
    assert diagnostics["maximum"] == upper_limit
    np.testing.assert_array_equal(
        values,
        np.asarray([lower_limit, minimum_square, maximum_square, upper_limit]),
    )
    with pytest.raises(DailyDataError, match="violates configured DLV bounds"):
        surface_module._validate_dlv_bounds(
            np.asarray([np.nextafter(lower_limit, -np.inf), maximum_square]),
            minimum_square=minimum_square,
            maximum_square=maximum_square,
            absolute_tolerance=tolerance,
            label="test lower",
        )
    with pytest.raises(DailyDataError, match="violates configured DLV bounds"):
        surface_module._validate_dlv_bounds(
            np.asarray([minimum_square, np.nextafter(upper_limit, np.inf)]),
            minimum_square=minimum_square,
            maximum_square=maximum_square,
            absolute_tolerance=tolerance,
            label="test upper",
        )
