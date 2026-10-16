"""Reference-compatible daily no-arbitrage, filter, and smooth-SANOS stages."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.verbose import Context
from cdxcore.version import version

from sanos.config import canonical_json, canonical_sha256
from sanos.model.arbfree import ArbFreePricer, MkArbFree
from sanos.model.tools import SanosPricer
from sanos.widb.config import (
    DailySurfaceConfig,
    NoArbitrageConfig,
    OptionsFilterConfig,
)
from sanos.widb.daily import (
    ENRICHED_SCHEMA,
    DailyDataError,
    _black_implied,
    _black_values,
    _numerical_tolerances,
)

SOLVER_TIMEOUT_SECONDS = 300.0
_IMPLIED_VOL_CALC_EPS, _GREEK_CALC_EPS = _numerical_tolerances()


@version("widb-no-arb-day-v2")
def _no_arb_only_schema() -> dict[str, pl.DataType]:
    return {
        "no_arb_configuration_id": pl.String,
        "surface_min_half_spread": pl.Float64,
        "initial_no_arb_expiry_retained": pl.Boolean,
        "initial_no_arb_input_expiry_index": pl.Int32,
        "mask_arb_free": pl.Boolean,
        "initial_no_arb_rejection_reason": pl.String,
        "fit1_volatility": pl.Float32,
        "pure_fit1_price": pl.Float32,
        "pure_fit1_delta": pl.Float32,
        "pure_fit1_gamma": pl.Float32,
        "pure_fit1_vega": pl.Float32,
        "pure_fit1_theta": pl.Float32,
        "pure_fit1_dk": pl.Float32,
        "fitX_volatility": pl.Float32,
        "pure_fitX_price": pl.Float32,
        "pure_fitX_delta": pl.Float32,
        "pure_fitX_gamma": pl.Float32,
        "pure_fitX_vega": pl.Float32,
        "pure_fitX_theta": pl.Float32,
        "pure_fitX_dk": pl.Float32,
        "initial_model_variance": pl.Float64,
        "initial_expiry_model_strikes": pl.List(pl.Float64),
        "initial_expiry_model_density": pl.List(pl.Float64),
        "initial_surface_model_strikes": pl.List(pl.Float64),
        "initial_surface_model_density": pl.List(pl.Float64),
        "initial_x_to_surface_indices": pl.List(pl.Int32),
        "no_arb_diagnostics_json": pl.String,
    }


@version("widb-filtered-day-v2")
def _filter_only_schema() -> dict[str, pl.DataType]:
    return {
        "filter_initial_eligible": pl.Boolean,
        "filter_horizon_eligible": pl.Boolean,
        "filter_nms_eligible": pl.Boolean,
        "filter_volume_eligible": pl.Boolean,
        "filter_open_interest_eligible": pl.Boolean,
        "filter_am_eligible": pl.Boolean,
        "filter_expiry_population_eligible": pl.Boolean,
        "filter_cap_eligible": pl.Boolean,
        "options_filter_eligible": pl.Boolean,
        "options_filter_rejection_reason": pl.String,
        "spot_reference": pl.Float32,
        "normalized_previous_spot": pl.Float32,
        "spot_normalized_strike": pl.Float32,
        "spot_normalized_implied_forward": pl.Float32,
        "spot_normalized_market_call_bid": pl.Float32,
        "spot_normalized_market_call_ask": pl.Float32,
        "spot_normalized_market_put_bid": pl.Float32,
        "spot_normalized_market_put_ask": pl.Float32,
        "filter_diagnostics_json": pl.String,
        "options_filter_configuration_id": pl.String,
    }


@version("widb-fitted-day-v4")
def _fitted_only_schema() -> dict[str, pl.DataType]:
    return {
        "pure_market_call_bid": pl.Float32,
        "pure_market_call_ask": pl.Float32,
        "pure_market_put_bid": pl.Float32,
        "pure_market_put_ask": pl.Float32,
        "call_active": pl.Boolean,
        "put_active": pl.Boolean,
        "smooth_fit_mask": pl.Boolean,
        "smooth_model_mask": pl.Boolean,
        "smoothness": pl.Float64,
        "smooth_input_expiry_index": pl.Int32,
        "smooth_expiry_fit_price": pl.Float64,
        "smooth_surface_fit_price": pl.Float64,
        "smooth_call_mid": pl.Float64,
        "smooth_put_mid": pl.Float64,
        "smooth_mid_available": pl.Boolean,
        "smooth_fit_error_half_spreads": pl.Float64,
        "smooth_rejection_reason": pl.String,
        "smooth_model_variance": pl.Float64,
        "smooth_expiry_model_strikes": pl.List(pl.Float64),
        "smooth_expiry_model_density": pl.List(pl.Float64),
        "smooth_surface_model_strikes": pl.List(pl.Float64),
        "smooth_surface_model_density": pl.List(pl.Float64),
        "smooth_x_to_surface_indices": pl.List(pl.Int32),
        "smooth_pcnt_fit": pl.Float64,
        "smooth_median_error": pl.Float64,
        "smooth_mean_error": pl.Float64,
        "smooth_median_total_error": pl.Float64,
        "smooth_mean_total_error": pl.Float64,
        "smooth_diagnostics_json": pl.String,
        "smooth_sanos_configuration_id": pl.String,
    }


NO_ARB_ONLY_SCHEMA = _no_arb_only_schema()
FILTER_ONLY_SCHEMA = _filter_only_schema()
FITTED_ONLY_SCHEMA = _fitted_only_schema()
NO_ARB_SCHEMA = ENRICHED_SCHEMA | NO_ARB_ONLY_SCHEMA
FILTERED_SCHEMA = NO_ARB_SCHEMA | FILTER_ONLY_SCHEMA
FITTED_SCHEMA = FILTERED_SCHEMA | FITTED_ONLY_SCHEMA
NO_ARB_SCHEMA_VERSION = str(_no_arb_only_schema.version)
FILTERED_SCHEMA_VERSION = str(_filter_only_schema.version)
FITTED_SCHEMA_VERSION = str(_fitted_only_schema.version)


@version("1.0")
def _empty_surface_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


@version("1.0")
def _require_schema(frame: pl.DataFrame, schema: Mapping[str, pl.DataType], stage: str) -> None:
    expected = pl.Schema(schema)
    if frame.schema != expected:
        raise DailyDataError(
            f"{stage} schema mismatch: expected={expected}, observed={frame.schema}"
        )


@version("1.0")
def _float_column(frame: pl.DataFrame, name: str) -> np.ndarray[Any, Any]:
    return frame.get_column(name).to_numpy().astype(np.float64, copy=False)


@version("1.0")
def _float32_column(frame: pl.DataFrame, name: str) -> np.ndarray[Any, Any]:
    """Return a persisted Float32 column without silently widening it."""

    column = frame.get_column(name)
    if column.dtype != pl.Float32:
        raise DailyDataError(
            f"Column {name!r} must be Float32 at the smooth SANOS boundary; "
            f"observed {column.dtype}"
        )
    values = column.to_numpy()
    if values.dtype != np.dtype(np.float32):
        raise DailyDataError(
            f"Column {name!r} produced NumPy dtype {values.dtype}; expected float32"
        )
    return values


@version("1.0")
def _nullable_float32(name: str, values: np.ndarray[Any, Any]) -> pl.Series:
    return pl.Series(name, values.astype(np.float32, copy=False)).fill_nan(None)


@version("1.0")
def _nullable_float64(name: str, values: np.ndarray[Any, Any]) -> pl.Series:
    return pl.Series(name, values.astype(np.float64, copy=False)).fill_nan(None)


@version("1.1", dependencies=[NoArbitrageConfig, MkArbFree])
def _make_no_arb_fitter(
    config: NoArbitrageConfig,
    *,
    smoothness: float | None,
    diagnostic_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> MkArbFree:
    """Construct the reviewed numerical adapter with the frozen timeout."""

    return MkArbFree(
        no_arb_mode="surface",
        smoothness=smoothness,
        surface_expiry_linear=True,
        min_half_spread=config.min_half_spread,
        mid_fit_weight=config.mid_fit_weight,
        min_atm_vol=config.min_atm_volatility,
        min_dlv=config.min_dlv,
        max_dlv=config.max_dlv,
        timeout_seconds=SOLVER_TIMEOUT_SECONDS,
        min_strikes_per_expiry=config.min_strikes_per_expiry,
        min_density=config.min_density,
        min_dC_eps=config.min_dc_epsilon,
        _solver_fallback_algorithms=(
            ("ipm", "clarabel") if smoothness is not None else ("clarabel",)
        ),
        _solver_diagnostic_sink=diagnostic_sink,
    )


@version("1.0")
def _groups(
    frame: pl.DataFrame,
    row_mask: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]]]:
    expiry_ids = frame.get_column("expiry_id").to_numpy().astype(np.int32, copy=False)
    unique_ids = np.unique(expiry_ids[row_mask])
    positions: list[np.ndarray[Any, Any]] = []
    for expiry_id in unique_ids:
        local = np.flatnonzero(row_mask & (expiry_ids == expiry_id))
        strikes = frame.get_column("normalized_strike").to_numpy()[local]
        local = local[np.argsort(strikes, kind="stable")]
        positions.append(local)
    return unique_ids, positions, [frame[index] for index in positions]


@version("1.0")
def _state_lists(
    pricer: ArbFreePricer,
    output_index: int,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[int] | None,
]:
    if pricer.mstrikes is None or pricer.mdensities is None:
        raise DailyDataError("Surface fit returned no surface model state")
    mapping = None
    if pricer.xix_to_mix is not None and pricer.xix_to_mix[output_index] is not None:
        mapping = np.asarray(pricer.xix_to_mix[output_index], dtype=np.int32).tolist()
    return (
        np.asarray(pricer.xstrikes[output_index], dtype=np.float64).tolist(),
        np.asarray(pricer.xdensities[output_index], dtype=np.float64).tolist(),
        np.asarray(pricer.mstrikes[output_index], dtype=np.float64).tolist(),
        np.asarray(pricer.mdensities[output_index], dtype=np.float64).tolist(),
        mapping,
    )


@version(
    "1.0",
    dependencies=[
        _make_no_arb_fitter,
        _no_arb_only_schema,
        _empty_surface_frame,
        _require_schema,
        _float_column,
        _nullable_float32,
        _nullable_float64,
        _groups,
        _state_lists,
        _black_implied,
        _black_values,
        _numerical_tolerances,
        DailySurfaceConfig,
        canonical_json,
        canonical_sha256,
    ],
)
def fit_initial_no_arbitrage(
    frame: pl.DataFrame,
    *,
    config: DailySurfaceConfig = DailySurfaceConfig(),
) -> pl.DataFrame:
    """Append the pre-filter linear surface state without discarding audit rows."""

    if frame.is_empty():
        return _empty_surface_frame(NO_ARB_SCHEMA)
    _require_schema(frame, ENRICHED_SCHEMA, "enriched-day")
    row_count = frame.height
    all_rows = np.ones(row_count, dtype=bool)
    expiry_ids, positions, groups = _groups(frame, all_rows)
    expiries = np.asarray(
        [group.get_column("time_to_expiry_years").item(0) for group in groups],
        dtype=np.float64,
    )
    strikes = [_float_column(group, "normalized_strike") for group in groups]
    bids = [_float_column(group, "pure_bid_price") for group in groups]
    asks = [_float_column(group, "pure_ask_price") for group in groups]
    solver_events: list[dict[str, object]] = []
    fitter = _make_no_arb_fitter(
        config.no_arbitrage,
        smoothness=None,
        diagnostic_sink=lambda event: solver_events.append(dict(event)),
    )
    try:
        pricer = fitter(
            expiries=expiries,
            strikes=strikes,
            bids=bids,
            asks=asks,
            message_context=lambda: (
                f"{frame.get_column('ticker').item(0)} "
                f"{frame.get_column('trading_date').item(0)} initial no-arbitrage"
            ),
            verbose=Context.quiet,
        )
    except Exception as exc:
        failure = canonical_json(
            {
                "outcome": "failed",
                "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
                "solver_events": solver_events,
            }
        )
        raise DailyDataError(
            f"Initial no-arbitrage solver failed: {exc}; diagnostics={failure}"
        ) from exc

    retained = np.zeros(row_count, dtype=bool)
    masks = np.zeros(row_count, dtype=bool)
    input_indices = np.full(row_count, -1, dtype=np.int32)
    float_fields = {
        name: np.full(row_count, np.nan, dtype=np.float64)
        for name in (
            "fit1_volatility",
            "pure_fit1_price",
            "pure_fit1_delta",
            "pure_fit1_gamma",
            "pure_fit1_vega",
            "pure_fit1_theta",
            "pure_fit1_dk",
            "fitX_volatility",
            "pure_fitX_price",
            "pure_fitX_delta",
            "pure_fitX_gamma",
            "pure_fitX_vega",
            "pure_fitX_theta",
            "pure_fitX_dk",
        )
    }
    variances = np.full(row_count, np.nan, dtype=np.float64)
    xstrikes: list[list[float] | None] = [None] * row_count
    xdensities: list[list[float] | None] = [None] * row_count
    mstrikes: list[list[float] | None] = [None] * row_count
    mdensities: list[list[float] | None] = [None] * row_count
    mappings: list[list[int] | None] = [None] * row_count
    if pricer.fitsX is None:
        raise DailyDataError("Initial surface fit returned no surface prices")

    for output_index, source_index_value in enumerate(pricer.input_index):
        source_index = int(source_index_value)
        rows = positions[source_index]
        pure_strike = _float_column(groups[source_index], "normalized_strike")
        expiry = float(pricer.expiries[output_index])
        sqrt_time = math.sqrt(expiry)
        fit1 = np.asarray(pricer.fits1[output_index], dtype=np.float64)
        fitx = np.asarray(pricer.fitsX[output_index], dtype=np.float64)
        fit1_vol = _black_implied(
            pure_strike,
            fit1,
            sqrtT=sqrt_time,
            eps=_IMPLIED_VOL_CALC_EPS,
        )
        fitx_vol = _black_implied(
            pure_strike,
            fitx,
            sqrtT=sqrt_time,
            eps=_IMPLIED_VOL_CALC_EPS,
        )
        fit1_greeks = _black_values(
            k=pure_strike,
            vol=fit1_vol,
            sqrtT=sqrt_time,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        fitx_greeks = _black_values(
            k=pure_strike,
            vol=fitx_vol,
            sqrtT=sqrt_time,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        retained[rows] = True
        masks[rows] = np.asarray(pricer.masks[output_index], dtype=bool)
        input_indices[rows] = source_index
        float_fields["fit1_volatility"][rows] = fit1_vol
        float_fields["fitX_volatility"][rows] = fitx_vol
        for name, value in fit1_greeks.items():
            float_fields[f"pure_fit1_{name}"][rows] = np.asarray(value, dtype=np.float64)
        for name, value in fitx_greeks.items():
            float_fields[f"pure_fitX_{name}"][rows] = np.asarray(value, dtype=np.float64)
        state = _state_lists(pricer, output_index)
        variances[rows] = float(pricer.variances[output_index])
        for row in rows:
            xstrikes[int(row)], xdensities[int(row)], mstrikes[int(row)], mdensities[int(row)] = (
                state[0],
                state[1],
                state[2],
                state[3],
            )
            mappings[int(row)] = state[4]

    rejection = np.full(row_count, None, dtype=object)
    rejection[~retained] = "initial_no_arb_expiry_dropped"
    rejection[retained & ~masks] = "initial_no_arb_strike_masked"
    retained_ids = {int(expiry_ids[int(index)]) for index in pricer.input_index}
    diagnostics = canonical_json(
        {
            "input_rows": row_count,
            "input_expiries": len(expiry_ids),
            "retained_expiries": len(pricer.expiries),
            "dropped_expiry_ids": [
                int(value) for value in expiry_ids if int(value) not in retained_ids
            ],
            "masked_strikes": int(np.sum(retained & ~masks)),
            "outcome": "completed",
            "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
            "solver_events": solver_events,
        }
    )
    result = frame.with_columns(
        pl.lit(canonical_sha256({"no_arbitrage": config.no_arbitrage})).alias(
            "no_arb_configuration_id"
        ),
        pl.lit(config.no_arbitrage.min_half_spread).alias("surface_min_half_spread"),
        pl.Series("initial_no_arb_expiry_retained", retained),
        pl.Series(
            "initial_no_arb_input_expiry_index",
            [None if value < 0 else int(value) for value in input_indices],
            dtype=pl.Int32,
        ),
        pl.Series("mask_arb_free", masks),
        pl.Series(
            "initial_no_arb_rejection_reason", rejection.tolist(), dtype=pl.String
        ),
        *(_nullable_float32(name, values) for name, values in float_fields.items()),
        _nullable_float64("initial_model_variance", variances),
        pl.Series("initial_expiry_model_strikes", xstrikes, dtype=pl.List(pl.Float64)),
        pl.Series("initial_expiry_model_density", xdensities, dtype=pl.List(pl.Float64)),
        pl.Series("initial_surface_model_strikes", mstrikes, dtype=pl.List(pl.Float64)),
        pl.Series("initial_surface_model_density", mdensities, dtype=pl.List(pl.Float64)),
        pl.Series("initial_x_to_surface_indices", mappings, dtype=pl.List(pl.Int32)),
        pl.lit(diagnostics).alias("no_arb_diagnostics_json"),
        pl.lit(str(fit_initial_no_arbitrage.version)).alias("stage_version"),
    ).select(*NO_ARB_SCHEMA)
    return result.cast(NO_ARB_SCHEMA, strict=True)


@version("1.0")
def _set_first_reason(
    reasons: np.ndarray[Any, Any],
    before: np.ndarray[Any, Any],
    after: np.ndarray[Any, Any],
    label: str,
) -> None:
    rejected = before & ~after
    unset = np.fromiter((value is None for value in reasons), dtype=bool, count=len(reasons))
    reasons[rejected & unset] = label


@version("1.0")
def _cap_positions(
    expiry_ids: np.ndarray[Any, Any],
    nms: np.ndarray[Any, Any],
    eligible: np.ndarray[Any, Any],
    config: OptionsFilterConfig,
) -> np.ndarray[Any, Any]:
    """Return the exact stable legacy cap selection on original row positions."""

    selected = eligible.copy()
    live_positions = np.flatnonzero(eligible)
    if len(live_positions) <= config.max_num_options:
        return selected
    reserved: list[np.ndarray[Any, Any]] = []
    remaining: list[np.ndarray[Any, Any]] = []
    for expiry_id in np.unique(expiry_ids[live_positions]):
        expiry_positions = live_positions[expiry_ids[live_positions] == expiry_id]
        if len(expiry_positions) < config.min_options_per_expiry:
            raise DailyDataError("Cap input contains an expiry below the minimum population")
        if len(expiry_positions) == config.min_options_per_expiry:
            reserved.append(expiry_positions)
            continue
        local_nms = nms[expiry_positions]
        chosen = np.zeros(len(expiry_positions), dtype=bool)
        positive = np.flatnonzero(local_nms > 0.0)
        if len(positive):
            chosen[positive[np.argmin(local_nms[positive])]] = True
        negative = np.flatnonzero(local_nms < 0.0)
        if len(negative):
            chosen[negative[np.argmax(local_nms[negative])]] = True
        needed = config.min_options_per_expiry - int(np.sum(chosen))
        if needed > 0:
            candidates = np.flatnonzero(~chosen)
            order = candidates[np.argsort(np.abs(local_nms[candidates]), kind="stable")]
            chosen[order[:needed]] = True
        reserved.append(expiry_positions[chosen])
        remaining.append(expiry_positions[~chosen])
    reserved_positions = np.concatenate(reserved) if reserved else np.asarray([], dtype=np.int64)
    if len(reserved_positions) > config.max_num_options:
        raise DailyDataError(
            "Reference cap invariant failed: per-expiry reservations exceed max_num_options"
        )
    remaining_positions = (
        np.concatenate(remaining) if remaining else np.asarray([], dtype=np.int64)
    )
    slots = config.max_num_options - len(reserved_positions)
    if slots:
        order = np.argsort(np.abs(nms[remaining_positions]), kind="stable")[:slots]
        kept = np.concatenate((reserved_positions, remaining_positions[order]))
    else:
        kept = reserved_positions
    selected[:] = False
    selected[kept] = True
    return selected


@version(
    "1.0",
    dependencies=[
        _cap_positions,
        _filter_only_schema,
        _empty_surface_frame,
        _require_schema,
        _set_first_reason,
        OptionsFilterConfig,
        DailySurfaceConfig,
        canonical_json,
        canonical_sha256,
    ],
)
def apply_options_filter(
    frame: pl.DataFrame,
    *,
    config: DailySurfaceConfig = DailySurfaceConfig(),
) -> pl.DataFrame:
    """Apply the exact legacy filter order while retaining rejection evidence."""

    if frame.is_empty():
        return _empty_surface_frame(FILTERED_SCHEMA)
    _require_schema(frame, NO_ARB_SCHEMA, "no-arb-day")
    rules = config.options_filter
    count = frame.height
    reasons = np.full(count, None, dtype=object)
    expiry_ids = frame.get_column("expiry_id").to_numpy().astype(np.int32, copy=False)
    initial = frame.get_column("initial_no_arb_expiry_retained").to_numpy().astype(bool)
    reasons[~initial] = "initial_no_arb_expiry_dropped"
    horizon = initial & (
        frame.get_column("business_dte").to_numpy() <= rules.max_business_days_to_expiry
    )
    _set_first_reason(reasons, initial, horizon, "business_dte_above_maximum")
    nms_values = frame.get_column(
        "atm_normalized_moneyness" if rules.nms_use_atm else "normalized_moneyness"
    ).to_numpy()
    nms_ok = horizon & (nms_values >= rules.max_nms[0]) & (nms_values <= rules.max_nms[1])
    _set_first_reason(reasons, horizon, nms_ok, "normalized_moneyness_outside_range")
    volume = nms_ok
    if rules.min_log_volume > 0.0:
        volume = nms_ok & (
            frame.get_column("combined_log_volume").to_numpy() >= rules.min_log_volume
        )
    _set_first_reason(reasons, nms_ok, volume, "log_volume_below_minimum")
    open_interest = volume
    if rules.min_log_open_interest > 0.0:
        open_interest = volume & (
            frame.get_column("combined_log_open_interest").to_numpy()
            >= rules.min_log_open_interest
        )
    _set_first_reason(reasons, volume, open_interest, "log_open_interest_below_minimum")
    am = open_interest
    if rules.remove_am:
        am = open_interest & ~frame.get_column("expiry_is_am").to_numpy().astype(bool)
    _set_first_reason(reasons, open_interest, am, "am_expiry_removed")
    population = am.copy()
    for expiry_id in np.unique(expiry_ids[am]):
        positions = np.flatnonzero(am & (expiry_ids == expiry_id))
        if len(positions) < rules.min_options_per_expiry:
            population[positions] = False
    _set_first_reason(reasons, am, population, "expiry_below_minimum_population")
    normalized_moneyness = frame.get_column("normalized_moneyness").to_numpy()
    capped = _cap_positions(expiry_ids, normalized_moneyness, population, rules)
    _set_first_reason(reasons, population, capped, "global_option_cap")

    spot = frame.get_column("spot").to_numpy().astype(np.float32, copy=False)
    previous_spot = frame.get_column("previous_spot").to_numpy().astype(np.float32, copy=False)
    if np.min(spot) <= 0.0 or np.min(previous_spot) <= 0.0:
        raise DailyDataError("Spot and previous spot must be positive before normalization")

    def normalized(name: str) -> np.ndarray[Any, Any]:
        return (
            frame.get_column(name).to_numpy().astype(np.float32, copy=False) / spot
        ).astype(np.float32, copy=False)

    counts = {
        "initial": int(np.sum(initial)),
        "horizon": int(np.sum(horizon)),
        "nms": int(np.sum(nms_ok)),
        "volume": int(np.sum(volume)),
        "open_interest": int(np.sum(open_interest)),
        "am": int(np.sum(am)),
        "expiry_population": int(np.sum(population)),
        "cap": int(np.sum(capped)),
    }
    diagnostics = canonical_json(
        {
            "counts": counts,
            "min_vega_sqrt_var": rules.min_vega_sqrt_var,
            "min_vega_sqrt_var_applied": False,
            "rejections": {
                str(reason): int(np.sum(reasons == reason))
                for reason in sorted({value for value in reasons if value is not None})
            },
        }
    )
    result = frame.with_columns(
        pl.Series("filter_initial_eligible", initial),
        pl.Series("filter_horizon_eligible", horizon),
        pl.Series("filter_nms_eligible", nms_ok),
        pl.Series("filter_volume_eligible", volume),
        pl.Series("filter_open_interest_eligible", open_interest),
        pl.Series("filter_am_eligible", am),
        pl.Series("filter_expiry_population_eligible", population),
        pl.Series("filter_cap_eligible", capped),
        pl.Series("options_filter_eligible", capped),
        pl.Series(
            "options_filter_rejection_reason", reasons.tolist(), dtype=pl.String
        ),
        pl.Series("spot_reference", spot, dtype=pl.Float32),
        pl.Series(
            "normalized_previous_spot",
            (previous_spot / spot).astype(np.float32, copy=False),
            dtype=pl.Float32,
        ),
        pl.Series("spot_normalized_strike", normalized("strike"), dtype=pl.Float32),
        pl.Series(
            "spot_normalized_implied_forward",
            normalized("implied_forward"),
            dtype=pl.Float32,
        ),
        pl.Series(
            "spot_normalized_market_call_bid",
            normalized("market_call_bid"),
            dtype=pl.Float32,
        ),
        pl.Series(
            "spot_normalized_market_call_ask",
            normalized("market_call_ask"),
            dtype=pl.Float32,
        ),
        pl.Series(
            "spot_normalized_market_put_bid",
            normalized("market_put_bid"),
            dtype=pl.Float32,
        ),
        pl.Series(
            "spot_normalized_market_put_ask",
            normalized("market_put_ask"),
            dtype=pl.Float32,
        ),
        pl.lit(diagnostics).alias("filter_diagnostics_json"),
        pl.lit(canonical_sha256({"options_filter": rules})).alias(
            "options_filter_configuration_id"
        ),
        pl.lit(str(apply_options_filter.version)).alias("stage_version"),
    ).select(*FILTERED_SCHEMA)
    return result.cast(FILTERED_SCHEMA, strict=True)


@version("1.0")
def _canonical_fitted_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Order fitted rows by the immutable daily key used by every consumer."""

    return frame.sort("expiry_id", "normalized_strike", maintain_order=True)


@version(
    "2.0",
    dependencies=[
        _canonical_fitted_rows,
        _nullable_float32,
        _nullable_float64,
        _fitted_only_schema,
    ],
)
def _finish_empty_smooth(
    frame: pl.DataFrame,
    *,
    direct: Mapping[str, np.ndarray[Any, Any]],
    diagnostics: str,
    smoothness: float,
    smooth_config_id: str,
) -> pl.DataFrame:
    count = frame.height
    false = np.zeros(count, dtype=bool)
    nan = np.full(count, np.nan, dtype=np.float64)
    null_lists: list[list[float] | None] = [None] * count
    null_maps: list[list[int] | None] = [None] * count
    result = frame.with_columns(
        *(_nullable_float32(name, values) for name, values in direct.items()),
        pl.Series("call_active", false),
        pl.Series("put_active", false),
        pl.Series("smooth_fit_mask", false),
        pl.Series("smooth_model_mask", false),
        pl.lit(smoothness).alias("smoothness"),
        pl.lit(None, dtype=pl.Int32).alias("smooth_input_expiry_index"),
        pl.lit(None, dtype=pl.Float64).alias("smooth_expiry_fit_price"),
        pl.lit(None, dtype=pl.Float64).alias("smooth_surface_fit_price"),
        _nullable_float64("smooth_call_mid", nan),
        _nullable_float64("smooth_put_mid", nan),
        pl.Series("smooth_mid_available", false),
        _nullable_float64("smooth_fit_error_half_spreads", nan),
        pl.lit("not_options_filter_eligible").alias("smooth_rejection_reason"),
        _nullable_float64("smooth_model_variance", nan),
        pl.Series("smooth_expiry_model_strikes", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_expiry_model_density", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_surface_model_strikes", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_surface_model_density", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_x_to_surface_indices", null_maps, dtype=pl.List(pl.Int32)),
        pl.lit(None, dtype=pl.Float64).alias("smooth_pcnt_fit"),
        pl.lit(None, dtype=pl.Float64).alias("smooth_median_error"),
        pl.lit(None, dtype=pl.Float64).alias("smooth_mean_error"),
        pl.lit(None, dtype=pl.Float64).alias("smooth_median_total_error"),
        pl.lit(None, dtype=pl.Float64).alias("smooth_mean_total_error"),
        pl.lit(diagnostics).alias("smooth_diagnostics_json"),
        pl.lit(smooth_config_id).alias("smooth_sanos_configuration_id"),
        pl.lit(str(fit_smooth_sanos.version)).alias("stage_version"),
    ).select(*FITTED_SCHEMA)
    return _canonical_fitted_rows(result.cast(FITTED_SCHEMA, strict=True))


@version(
    "3.0",
    dependencies=[
        _canonical_fitted_rows,
        _fitted_only_schema,
        _make_no_arb_fitter,
        _empty_surface_frame,
        _require_schema,
        _float_column,
        _float32_column,
        _nullable_float32,
        _nullable_float64,
        _groups,
        _state_lists,
        _finish_empty_smooth,
        DailySurfaceConfig,
        canonical_json,
        canonical_sha256,
    ],
)
def fit_smooth_sanos(
    frame: pl.DataFrame,
    *,
    config: DailySurfaceConfig = DailySurfaceConfig(),
) -> pl.DataFrame:
    """Fit the production smooth surface to the exact filtered population."""

    if frame.is_empty():
        return _empty_surface_frame(FITTED_SCHEMA)
    _require_schema(frame, FILTERED_SCHEMA, "filtered-day")
    selected = frame.get_column("options_filter_eligible").to_numpy().astype(bool)
    count = frame.height
    direct_names = (
        "pure_market_call_bid",
        "pure_market_call_ask",
        "pure_market_put_bid",
        "pure_market_put_ask",
    )
    direct = {name: np.full(count, np.nan, dtype=np.float32) for name in direct_names}
    smooth_config_id = canonical_sha256(
        {
            "no_arbitrage": config.no_arbitrage,
            "smooth_sanos": config.smooth_sanos,
        }
    )
    discount = _float32_column(frame, "discount_factor")
    normalized_forward = _float32_column(frame, "spot_normalized_implied_forward")
    discounted_forward = (discount * normalized_forward).astype(np.float32, copy=False)
    if np.any(selected & (~np.isfinite(discounted_forward) | (discounted_forward <= 0.0))):
        raise DailyDataError("Selected books require a finite positive discounted forward")
    for target, source in (
        ("pure_market_call_bid", "spot_normalized_market_call_bid"),
        ("pure_market_call_ask", "spot_normalized_market_call_ask"),
        ("pure_market_put_bid", "spot_normalized_market_put_bid"),
        ("pure_market_put_ask", "spot_normalized_market_put_ask"),
    ):
        source_values = _float32_column(frame, source)
        direct[target][selected] = source_values[selected] / discounted_forward[selected]

    if not np.any(selected):
        diagnostics = canonical_json(
            {
                "outcome": "no_eligible_options",
                "selected_rows": 0,
                "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
            }
        )
        return _finish_empty_smooth(
            frame,
            direct=direct,
            diagnostics=diagnostics,
            smoothness=config.smooth_sanos.smoothness,
            smooth_config_id=smooth_config_id,
        )

    expiry_ids, positions, groups = _groups(frame, selected)
    expiries = np.asarray(
        [_float32_column(group, "time_to_expiry_years")[0] for group in groups],
        dtype=np.float32,
    )
    strikes = [_float32_column(group, "normalized_strike") for group in groups]
    bids = [_float32_column(group, "pure_bid_price") for group in groups]
    asks = [_float32_column(group, "pure_ask_price") for group in groups]
    solver_events: list[dict[str, object]] = []
    fitter = _make_no_arb_fitter(
        config.no_arbitrage,
        smoothness=config.smooth_sanos.smoothness,
        diagnostic_sink=lambda event: solver_events.append(dict(event)),
    )
    try:
        pricer = fitter(
            expiries=expiries,
            strikes=strikes,
            bids=bids,
            asks=asks,
            message_context=lambda: (
                f"{frame.get_column('ticker').item(0)} "
                f"{frame.get_column('trading_date').item(0)} smooth SANOS"
            ),
            verbose=Context.quiet,
        )
        if not len(pricer.expiries):
            raise DailyDataError("Smooth SANOS dropped every selected expiry")
        if pricer.fitsX is None:
            raise DailyDataError("Smooth SANOS did not produce fitted surface prices")
        fit_result = pricer.compute_market_fit(
            expiries=expiries,
            strikes=strikes,
            bids=bids,
            asks=asks,
            and_vols=False,
        )
    except Exception as exc:
        failure = canonical_json(
            {
                "outcome": "failed",
                "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
                "solver_events": solver_events,
            }
        )
        raise DailyDataError(f"Smooth SANOS solver failed: {exc}; diagnostics={failure}") from exc

    smooth_mask = np.zeros(count, dtype=bool)
    smooth_input_index = np.full(count, -1, dtype=np.int32)
    expiry_fit = np.full(count, np.nan, dtype=np.float64)
    surface_fit = np.full(count, np.nan, dtype=np.float64)
    call_mid = np.full(count, np.nan, dtype=np.float64)
    fit_error = np.full(count, np.nan, dtype=np.float64)
    variances = np.full(count, np.nan, dtype=np.float64)
    xstrikes: list[list[float] | None] = [None] * count
    xdensities: list[list[float] | None] = [None] * count
    mstrikes: list[list[float] | None] = [None] * count
    mdensities: list[list[float] | None] = [None] * count
    mappings: list[list[int] | None] = [None] * count
    for source_index, rows in enumerate(positions):
        call_mid[rows] = np.asarray(fit_result.fits[source_index], dtype=np.float64)
        fit_error[rows] = np.asarray(fit_result.fit_errs[source_index], dtype=np.float64)
    for output_index, source_index_value in enumerate(pricer.input_index):
        source_index = int(source_index_value)
        rows = positions[source_index]
        smooth_mask[rows] = np.asarray(pricer.masks[output_index], dtype=bool)
        smooth_input_index[rows] = source_index
        expiry_fit[rows] = np.asarray(pricer.fits1[output_index], dtype=np.float64)
        surface_fit[rows] = np.asarray(pricer.fitsX[output_index], dtype=np.float64)
        variances[rows] = float(pricer.variances[output_index])
        state = _state_lists(pricer, output_index)
        for row in rows:
            xstrikes[int(row)], xdensities[int(row)], mstrikes[int(row)], mdensities[int(row)] = (
                state[0],
                state[1],
                state[2],
                state[3],
            )
            mappings[int(row)] = state[4]

    pre_mask = frame.get_column("mask_arb_free").to_numpy().astype(bool)
    model_mask = selected & pre_mask & smooth_mask
    call_active = (
        selected
        & pre_mask
        & np.isfinite(direct["pure_market_call_bid"])
        & np.isfinite(direct["pure_market_call_ask"])
        & (direct["pure_market_call_bid"] >= 0.0)
        & (direct["pure_market_call_ask"] > direct["pure_market_call_bid"])
    )
    put_active = (
        selected
        & pre_mask
        & np.isfinite(direct["pure_market_put_bid"])
        & np.isfinite(direct["pure_market_put_ask"])
        & (direct["pure_market_put_bid"] >= 0.0)
        & (direct["pure_market_put_ask"] > direct["pure_market_put_bid"])
    )
    normalized_strikes = _float_column(frame, "normalized_strike")
    put_mid = call_mid - 1.0 + normalized_strikes
    available = selected & np.isfinite(call_mid) & np.isfinite(put_mid)
    rejection = np.full(count, None, dtype=object)
    rejection[~selected] = "not_options_filter_eligible"
    selected_expiry_ids = set(int(value) for value in expiry_ids)
    smooth_retained_ids = {
        int(expiry_ids[int(source_index)]) for source_index in pricer.input_index
    }
    dropped_ids = selected_expiry_ids - smooth_retained_ids
    raw_expiry_ids = frame.get_column("expiry_id").to_numpy()
    for dropped in dropped_ids:
        rejection[selected & (raw_expiry_ids == dropped)] = "smooth_expiry_dropped"
    rejection[selected & np.isin(raw_expiry_ids, list(smooth_retained_ids)) & ~smooth_mask] = (
        "smooth_strike_masked"
    )
    diagnostics = canonical_json(
        {
            "input_rows": count,
            "selected_rows": int(np.sum(selected)),
            "selected_expiries": len(expiry_ids),
            "retained_expiries": len(pricer.expiries),
            "dropped_expiry_ids": sorted(dropped_ids),
            "smooth_masked_strikes": int(np.sum(selected & ~smooth_mask)),
            "model_mask_rows": int(np.sum(model_mask)),
            "call_active_rows": int(np.sum(call_active)),
            "put_active_rows": int(np.sum(put_active)),
            "pcnt_fit": float(fit_result.pcnt_fit),
            "median_error": float(fit_result.median_err),
            "mean_error": float(fit_result.mean_err),
            "median_total_error": float(fit_result.median_total_err),
            "mean_total_error": float(fit_result.mean_total_err),
            "outcome": "completed",
            "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
            "solver_events": solver_events,
        }
    )
    result = frame.with_columns(
        *(_nullable_float32(name, values) for name, values in direct.items()),
        pl.Series("call_active", call_active),
        pl.Series("put_active", put_active),
        pl.Series("smooth_fit_mask", smooth_mask),
        pl.Series("smooth_model_mask", model_mask),
        pl.lit(config.smooth_sanos.smoothness).alias("smoothness"),
        pl.Series(
            "smooth_input_expiry_index",
            [None if value < 0 else int(value) for value in smooth_input_index],
            dtype=pl.Int32,
        ),
        _nullable_float64("smooth_expiry_fit_price", expiry_fit),
        _nullable_float64("smooth_surface_fit_price", surface_fit),
        _nullable_float64("smooth_call_mid", call_mid),
        _nullable_float64("smooth_put_mid", put_mid),
        pl.Series("smooth_mid_available", available),
        _nullable_float64("smooth_fit_error_half_spreads", fit_error),
        pl.Series("smooth_rejection_reason", rejection.tolist(), dtype=pl.String),
        _nullable_float64("smooth_model_variance", variances),
        pl.Series("smooth_expiry_model_strikes", xstrikes, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_expiry_model_density", xdensities, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_surface_model_strikes", mstrikes, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_surface_model_density", mdensities, dtype=pl.List(pl.Float64)),
        pl.Series("smooth_x_to_surface_indices", mappings, dtype=pl.List(pl.Int32)),
        pl.lit(float(fit_result.pcnt_fit)).alias("smooth_pcnt_fit"),
        pl.lit(float(fit_result.median_err)).alias("smooth_median_error"),
        pl.lit(float(fit_result.mean_err)).alias("smooth_mean_error"),
        pl.lit(float(fit_result.median_total_err)).alias("smooth_median_total_error"),
        pl.lit(float(fit_result.mean_total_err)).alias("smooth_mean_total_error"),
        pl.lit(diagnostics).alias("smooth_diagnostics_json"),
        pl.lit(smooth_config_id).alias("smooth_sanos_configuration_id"),
        pl.lit(str(fit_smooth_sanos.version)).alias("stage_version"),
    ).select(*FITTED_SCHEMA)
    return _canonical_fitted_rows(result.cast(FITTED_SCHEMA, strict=True))


@version("1.0", dependencies=[SanosPricer])
def _reconstruct_pricer(
    frame: pl.DataFrame,
    *,
    index_column: str,
    strikes_column: str,
    density_column: str,
    variance_column: str,
    smoothness: float | None,
    expiry_dtype: Any,
) -> SanosPricer:
    required = {
        "time_to_expiry_years",
        "surface_min_half_spread",
        index_column,
        strikes_column,
        density_column,
        variance_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DailyDataError(f"Cannot reconstruct model state; missing columns: {missing}")
    state = frame.filter(pl.col(index_column).is_not_null()).sort(index_column)
    if state.is_empty():
        raise DailyDataError("Cannot reconstruct an empty model state")
    indices = state.get_column(index_column).unique(maintain_order=True).to_list()
    if int(indices[0]) < 0 or any(
        int(right) <= int(left) for left, right in zip(indices, indices[1:], strict=False)
    ):
        raise DailyDataError(f"Model input indices must be non-negative and increasing: {indices}")
    expiries: list[float] = []
    variances: list[float] = []
    strikes: list[np.ndarray[Any, Any]] = []
    densities: list[np.ndarray[Any, Any]] = []
    for index in indices:
        group = state.filter(pl.col(index_column) == index)
        expiry_values = group.get_column("time_to_expiry_years").unique().to_list()
        variance_values = group.get_column(variance_column).unique().to_list()
        strike_values = group.get_column(strikes_column).to_list()
        density_values = group.get_column(density_column).to_list()
        if len(expiry_values) != 1 or len(variance_values) != 1:
            raise DailyDataError(f"Model state is not constant for input expiry index {index}")
        if any(value != strike_values[0] for value in strike_values[1:]) or any(
            value != density_values[0] for value in density_values[1:]
        ):
            raise DailyDataError(f"Model list state is not constant for input expiry index {index}")
        expiries.append(float(expiry_values[0]))
        variances.append(float(variance_values[0]))
        strikes.append(np.asarray(strike_values[0], dtype=np.float64))
        densities.append(np.asarray(density_values[0], dtype=np.float64))
    spread_values = state.get_column("surface_min_half_spread").unique().to_list()
    if len(spread_values) != 1:
        raise DailyDataError("surface_min_half_spread must be constant within a day")
    return SanosPricer(
        smoothness=smoothness,
        expiries=np.asarray(expiries, dtype=expiry_dtype),
        variances=np.asarray(variances, dtype=np.float64),
        xstrikes=strikes,
        xdensities=densities,
        min_half_spread=float(spread_values[0]),
    )


@version("1.0", dependencies=[SanosPricer, _reconstruct_pricer])
def initial_no_arb_pricer(frame: pl.DataFrame) -> SanosPricer:
    """Reconstruct the pre-filter linear audit surface from one daily frame."""

    return _reconstruct_pricer(
        frame,
        index_column="initial_no_arb_input_expiry_index",
        strikes_column="initial_surface_model_strikes",
        density_column="initial_surface_model_density",
        variance_column="initial_model_variance",
        smoothness=None,
        expiry_dtype=np.float64,
    )


@version("1.0", dependencies=[SanosPricer, _reconstruct_pricer])
def smooth_sanos_pricer(frame: pl.DataFrame) -> SanosPricer:
    """Reconstruct the production smooth surface from one fitted daily frame."""

    if "smoothness" not in frame.columns:
        raise DailyDataError("Cannot reconstruct smooth SANOS state; missing smoothness")
    values = frame.get_column("smoothness").unique().to_list()
    if len(values) != 1 or values[0] is None:
        raise DailyDataError("smoothness must contain one non-null daily value")
    return _reconstruct_pricer(
        frame,
        index_column="smooth_input_expiry_index",
        strikes_column="smooth_surface_model_strikes",
        density_column="smooth_surface_model_density",
        variance_column="smooth_model_variance",
        smoothness=float(values[0]),
        expiry_dtype=np.float32,
    )


@version("1.0", dependencies=[_require_schema])
def selected_reference_view(frame: pl.DataFrame) -> pl.DataFrame:
    """Flatten the retained reference population in deterministic legacy order."""

    _require_schema(frame, FITTED_SCHEMA, "fitted-day")
    return frame.filter(pl.col("options_filter_eligible")).sort(
        "expiry_id", "normalized_strike", maintain_order=True
    )


__all__ = [
    "FILTERED_SCHEMA",
    "FILTERED_SCHEMA_VERSION",
    "FITTED_SCHEMA",
    "FITTED_SCHEMA_VERSION",
    "NO_ARB_SCHEMA",
    "NO_ARB_SCHEMA_VERSION",
    "SOLVER_TIMEOUT_SECONDS",
    "apply_options_filter",
    "fit_initial_no_arbitrage",
    "fit_smooth_sanos",
    "initial_no_arb_pricer",
    "selected_reference_view",
    "smooth_sanos_pricer",
]
