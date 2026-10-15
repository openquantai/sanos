"""Reference-compatible daily no-arbitrage, filter, and smooth-SANOS stages."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.verbose import Context
from cdxcore.version import version

from sanos.config import canonical_json, canonical_sha256
from sanos.model.arbfree import ArbFreePricer, MkArbFree
from sanos.model.fitter import SanosFitter
from sanos.model.tools import SanosPricer
from sanos.model.tridiag import solve_sanos_tridiagonal
from sanos.widb.calendar import NMS
from sanos.widb.config import (
    DailySurfaceConfig,
    DiscreteDlvConfig,
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


class _StrictDlvFitQualityError(DailyDataError):
    """A nominally solved strict DLV fit that violates its imposed bands."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "Strict discrete-DLV fit violated its feasibility bands: "
            + ", ".join(failures)
        )


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


@version("widb-fitted-day-v3")
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
        "discrete_dlv_anchor": pl.Boolean,
        "discrete_dlv_input_expiry_index": pl.Int32,
        "discrete_dlv_time_increment_years": pl.Float64,
        "discrete_dlv_atm_annual_volatility": pl.Float64,
        "discrete_dlv_input_total_variance": pl.Float64,
        "discrete_dlv_nms": pl.List(pl.Float64),
        "discrete_dlv_model_strikes": pl.List(pl.Float64),
        "discrete_dlv_model_density": pl.List(pl.Float64),
        "discrete_dlv_squares": pl.List(pl.Float64),
        "discrete_dlv_target_calls": pl.List(pl.Float64),
        "discrete_dlv_fitted_calls": pl.List(pl.Float64),
        "discrete_dlv_diagnostics_json": pl.String,
        "discrete_dlv_configuration_id": pl.String,
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


@version("1.0", dependencies=[NoArbitrageConfig, MkArbFree])
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


@version("1.0", dependencies=[NMS, DiscreteDlvConfig])
def _discrete_dlv_grid(
    expiries: np.ndarray[Any, Any],
    atm_volatilities: np.ndarray[Any, Any],
    config: DiscreteDlvConfig,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build the canonical nested NMS target grid and its support boundaries."""

    lower, upper = config.nms_range
    distance = upper - lower
    lower_count = min(
        max(2, int(config.inner_nodes * abs(lower) // distance)),
        config.inner_nodes - 3,
    )
    upper_count = config.inner_nodes - lower_count - 1
    if lower_count < 2 or upper_count < 2:
        raise DailyDataError("The discrete-DLV NMS grid needs at least two nodes per wing")
    raw_nms = np.concatenate(
        (
            np.linspace(lower, 0.0, lower_count, endpoint=False, dtype=np.float64),
            np.linspace(0.0, upper, upper_count + 1, dtype=np.float64),
        )
    )
    if raw_nms.shape != (config.inner_nodes,) or np.count_nonzero(raw_nms == 0.0) != 1:
        raise DailyDataError("The discrete-DLV target grid must contain one exact ATM node")
    inner = np.asarray(
        NMS.from_nms(
            normalized=raw_nms[None, :],
            time_years=expiries[:, None],
            volatility=atm_volatilities[:, None],
            epsilon_volatility=float(atm_volatilities[0]),
        ),
        dtype=np.float64,
    )
    supports = np.empty((len(expiries), config.inner_nodes + 2), dtype=np.float64)
    supports[:, 1:-1] = inner
    supports[:, 0] = inner[:, 0] * config.lower_support_multiplier
    supports[:, -1] = inner[:, -1] * config.upper_support_multiplier
    if (
        np.any(~np.isfinite(inner))
        or np.any(inner <= 0.0)
        or np.any(np.diff(inner, axis=1) <= 0.0)
        or np.any(np.diff(inner[:, 0]) > 0.0)
        or np.any(np.diff(inner[:, -1]) < 0.0)
        or np.any(np.diff(supports, axis=1) <= 0.0)
    ):
        raise DailyDataError("The discrete-DLV strike grid is not finite, positive, and nested")
    return raw_nms, inner, supports


@version("1.0")
def _error_metrics(values: np.ndarray[Any, Any]) -> dict[str, float | int]:
    """Return deterministic absolute-error summaries used by projection validation."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "max": 0.0, "p99": 0.0, "median": 0.0, "rms": 0.0}
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise DailyDataError("Projection error metrics require finite non-negative values")
    return {
        "count": int(values.size),
        "max": float(np.max(values)),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


@version("1.0")
def _bs_vega_sqrt_time(
    strikes: np.ndarray[Any, Any],
    volatilities: np.ndarray[Any, Any],
    expiry: float,
) -> np.ndarray[Any, Any]:
    """Return Black vega divided by sqrt(time) for projection conditioning."""

    return np.asarray(
        bs(
            k=strikes,
            vol=volatilities,
            sqrtT=math.sqrt(expiry),
            what=bs.VEGA,
        ),
        dtype=np.float64,
    ) / math.sqrt(expiry)


@version("1.0")
def _bs_call_price(
    strikes: np.ndarray[Any, Any],
    volatilities: np.ndarray[Any, Any],
    expiry: float,
) -> np.ndarray[Any, Any]:
    """Return normalized Black calls for the canonical projection contract."""

    return np.asarray(
        bs(
            k=strikes,
            vol=volatilities,
            sqrtT=math.sqrt(expiry),
            eps=_IMPLIED_VOL_CALC_EPS,
        ),
        dtype=np.float64,
    )


@version(
    "1.0",
    dependencies=[
        SanosPricer,
        DiscreteDlvConfig,
        _bs_call_price,
        _bs_vega_sqrt_time,
    ],
)
def _projection_fit_price_bounds(
    *,
    smooth: SanosPricer,
    expiries: np.ndarray[Any, Any],
    inner: np.ndarray[Any, Any],
    target_calls: np.ndarray[Any, Any],
    config: DiscreteDlvConfig,
) -> tuple[
    list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    dict[str, object],
]:
    """Build target-node Black-price bands that imply the configured IV bound."""

    if inner.shape != target_calls.shape or inner.shape[0] != len(expiries):
        raise DailyDataError("Discrete-DLV fit-cap geometry is inconsistent")
    bounds: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    conditioned_counts: list[int] = []
    for index, expiry in enumerate(expiries):
        _, target_iv = smooth(float(expiry), inner[index], and_vols=True)
        target_iv = np.asarray(target_iv, dtype=np.float64)
        vega_sqrt_time = _bs_vega_sqrt_time(
            inner[index],
            target_iv,
            float(expiry),
        )
        conditioned = vega_sqrt_time >= config.conditioned_iv_vega_sqrt_floor
        lower = target_calls[index] - config.fit_call_max_abs
        upper = target_calls[index] + config.fit_call_max_abs
        if np.any(conditioned):
            iv_tolerance = config.tolerances.conditioned_annual_iv_max_abs
            lower_iv = np.maximum(target_iv[conditioned] - iv_tolerance, 0.0)
            lower_calls = _bs_call_price(
                inner[index][conditioned],
                lower_iv,
                float(expiry),
            )
            upper_calls = _bs_call_price(
                inner[index][conditioned],
                target_iv[conditioned] + iv_tolerance,
                float(expiry),
            )
            # Inset the exact Black-price band by the solver feasibility tolerance
            # so a permitted LP residual cannot cross the annual-IV acceptance edge.
            lower[conditioned] = np.maximum(
                lower[conditioned],
                lower_calls + config.solver_kkt_tolerance,
            )
            upper[conditioned] = np.minimum(
                upper[conditioned],
                upper_calls - config.solver_kkt_tolerance,
            )
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower > target_calls[index])
            or np.any(target_calls[index] > upper)
            or np.any(lower > upper)
        ):
            raise DailyDataError(
                f"Discrete-DLV target price bands are invalid at expiry {index}"
            )
        bounds.append((lower, upper))
        conditioned_counts.append(int(np.sum(conditioned)))
    lower_slacks = np.concatenate(
        [target - lower for target, (lower, _) in zip(target_calls, bounds, strict=True)]
    )
    upper_slacks = np.concatenate(
        [upper - target for target, (_, upper) in zip(target_calls, bounds, strict=True)]
    )
    return bounds, {
        "conditioned_nodes": int(sum(conditioned_counts)),
        "conditioned_nodes_by_expiry": conditioned_counts,
        "minimum_target_lower_slack": float(np.min(lower_slacks)),
        "minimum_target_upper_slack": float(np.min(upper_slacks)),
        "solver_feasibility_inset": config.solver_kkt_tolerance,
    }


@version(
    "1.0",
    dependencies=[
        SanosPricer,
        DiscreteDlvConfig,
        _black_implied,
        _bs_vega_sqrt_time,
    ],
)
def _projection_target_band_metrics(
    *,
    smooth: SanosPricer,
    expiries: np.ndarray[Any, Any],
    inner: np.ndarray[Any, Any],
    fitted_calls: np.ndarray[Any, Any],
    bounds: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    config: DiscreteDlvConfig,
) -> dict[str, object]:
    """Measure target-node IV errors and price-band slacks."""

    if inner.shape != fitted_calls.shape or len(bounds) != len(expiries):
        raise DailyDataError("Discrete-DLV target-band metric geometry is inconsistent")
    iv_errors: list[np.ndarray[Any, Any]] = []
    lower_slacks: list[np.ndarray[Any, Any]] = []
    upper_slacks: list[np.ndarray[Any, Any]] = []
    conditioned_counts: list[int] = []
    maximum_iv_by_expiry: list[float] = []
    for index, expiry in enumerate(expiries):
        _, target_iv = smooth(float(expiry), inner[index], and_vols=True)
        target_iv = np.asarray(target_iv, dtype=np.float64)
        source_vega = _bs_vega_sqrt_time(
            inner[index], target_iv, float(expiry)
        )
        conditioned = source_vega >= config.conditioned_iv_vega_sqrt_floor
        projected_iv = _black_implied(
            k=inner[index],
            prices=fitted_calls[index],
            sqrtT=math.sqrt(float(expiry)),
            eps=_IMPLIED_VOL_CALC_EPS,
        )
        local_errors = np.abs(
            np.asarray(projected_iv, dtype=np.float64) - target_iv
        )[conditioned]
        if np.any(~np.isfinite(local_errors)):
            raise DailyDataError(
                f"Discrete-DLV target IV metrics are non-finite at expiry {index}"
            )
        iv_errors.append(local_errors)
        conditioned_counts.append(int(np.sum(conditioned)))
        maximum_iv_by_expiry.append(
            0.0 if local_errors.size == 0 else float(np.max(local_errors))
        )
        lower, upper = bounds[index]
        lower_slacks.append(fitted_calls[index] - lower)
        upper_slacks.append(upper - fitted_calls[index])
    all_iv_errors = np.concatenate(iv_errors)
    all_lower_slacks = np.concatenate(lower_slacks)
    all_upper_slacks = np.concatenate(upper_slacks)
    return {
        "conditioned_nodes": int(sum(conditioned_counts)),
        "conditioned_nodes_by_expiry": conditioned_counts,
        "maximum_absolute_iv_error": (
            0.0 if all_iv_errors.size == 0 else float(np.max(all_iv_errors))
        ),
        "maximum_absolute_iv_error_by_expiry": maximum_iv_by_expiry,
        "minimum_lower_slack": float(np.min(all_lower_slacks)),
        "minimum_upper_slack": float(np.min(all_upper_slacks)),
        "lower_binding_count": int(
            np.sum(all_lower_slacks <= config.solver_kkt_tolerance)
        ),
        "upper_binding_count": int(
            np.sum(all_upper_slacks <= config.solver_kkt_tolerance)
        ),
        "maximum_violation": max(
            -float(np.min(all_lower_slacks)),
            -float(np.min(all_upper_slacks)),
            0.0,
        ),
    }


@version("1.0")
def _validate_dlv_bounds(
    values: np.ndarray[Any, Any],
    *,
    minimum_square: float,
    maximum_square: float,
    absolute_tolerance: float,
    label: str,
) -> dict[str, float | int]:
    """Validate solver-scale DLV bounds without modifying fitted values."""

    observed = np.asarray(values, dtype=np.float64)
    if observed.size == 0 or np.any(~np.isfinite(observed)):
        raise DailyDataError(f"{label} has empty or non-finite DLV values")
    minimum = float(np.min(observed))
    maximum = float(np.max(observed))
    if (
        minimum < minimum_square - absolute_tolerance
        or maximum > maximum_square + absolute_tolerance
    ):
        raise DailyDataError(
            f"{label} violates configured DLV bounds: observed=[{minimum}, {maximum}], "
            f"required=[{minimum_square}, {maximum_square}] within "
            f"{absolute_tolerance}"
        )
    return {
        "minimum": minimum,
        "maximum": maximum,
        "minimum_saturation_count": int(
            np.sum(observed <= minimum_square + absolute_tolerance)
        ),
        "maximum_saturation_count": int(
            np.sum(observed >= maximum_square - absolute_tolerance)
        ),
    }


@version("1.0", dependencies=[DiscreteDlvConfig])
def _discrete_dlv_state_metrics(
    pricer: SanosPricer,
    dlv_squares: list[np.ndarray[Any, Any]],
    config: DiscreteDlvConfig,
) -> dict[str, float | int]:
    """Summarize density-floor and DLV-bound numerical margins."""

    if len(dlv_squares) != len(pricer.xdensities):
        raise DailyDataError("Discrete-DLV numerical state has inconsistent lengths")
    densities = [np.asarray(value, dtype=np.float64) for value in pricer.xdensities]
    interiors = [
        np.asarray(value, dtype=np.float64)[1:-1] for value in dlv_squares
    ]
    if (
        not densities
        or any(np.any(~np.isfinite(value)) for value in densities)
        or any(np.any(~np.isfinite(value)) for value in interiors)
    ):
        raise DailyDataError("Discrete-DLV numerical state is empty or non-finite")
    all_density = np.concatenate(densities)
    all_dlv = np.concatenate(interiors)
    minimum_density = float(np.min(all_density))
    minimum_dlv_square = float(np.min(all_dlv))
    maximum_dlv_square = float(np.max(all_dlv))
    solver_minimum_square = (config.min_dlv + config.constraint_dlv_buffer) ** 2
    solver_maximum_square = (config.max_dlv - config.constraint_dlv_buffer) ** 2
    support_nodes = sum(len(value) for value in densities)
    return {
        "support_nodes": support_nodes,
        "interior_dlv_nodes": int(sum(len(value) for value in interiors)),
        "minimum_density": minimum_density,
        "density_floor_near_count": int(
            np.sum(
                all_density
                <= config.density_mass_floor + config.solver_kkt_tolerance
            )
        ),
        "density_floor_minimum_slack": minimum_density - config.density_mass_floor,
        "density_floor_shortfall": max(
            config.density_mass_floor - minimum_density,
            0.0,
        ),
        "minimum_dlv_square": minimum_dlv_square,
        "maximum_dlv_square": maximum_dlv_square,
        "solver_minimum_dlv_square_distance": (
            minimum_dlv_square - solver_minimum_square
        ),
        "solver_maximum_dlv_square_distance": (
            solver_maximum_square - maximum_dlv_square
        ),
        "published_minimum_dlv_square_distance": (
            minimum_dlv_square - config.min_dlv**2
        ),
        "published_maximum_dlv_square_distance": (
            config.max_dlv**2 - maximum_dlv_square
        ),
        "required_density_floor_mass_per_expiry": (
            config.density_mass_floor * len(densities[0])
        ),
        "required_density_floor_mass_all_expiries": (
            config.density_mass_floor * support_nodes
        ),
    }


@version("1.0", dependencies=[SanosPricer])
def _canonical_discrete_dlv_squares(
    pricer: SanosPricer,
) -> list[np.ndarray[Any, Any]]:
    """Encode the canonical finite-DLV state with guarded extra precision.

    This is deliberately a dataset-boundary adapter rather than a change to
    :meth:`SanosPricer.dlvsqs`.  Tiny tail masses make the subtraction of two
    nearly equal calls ill-conditioned in binary64.  Evaluating only that
    serialization formula in ``longdouble`` avoids turning a valid fitted state
    into a platform-scale zero or a noisy DLV fixed point; persisted arrays
    remain ordinary Float64.
    """

    precision = np.longdouble
    if np.finfo(precision).eps >= np.finfo(np.float64).eps:
        raise DailyDataError(
            "Canonical discrete-DLV encoding requires NumPy extended precision"
        )
    expiries = np.asarray(pricer.expiries, dtype=precision)
    result: list[np.ndarray[Any, Any]] = []
    for index, expiry in enumerate(expiries):
        support = np.asarray(pricer.xstrikes[index], dtype=precision)
        density = np.asarray(pricer.xdensities[index], dtype=precision)
        if support.ndim != 1 or density.shape != support.shape or len(support) < 3:
            raise DailyDataError(
                f"Canonical discrete-DLV expiry {index} has invalid state geometry"
            )
        if index:
            previous_support = np.asarray(
                pricer.xstrikes[index - 1], dtype=precision
            )
            previous_density = np.asarray(
                pricer.xdensities[index - 1], dtype=precision
            )
            previous_calls = (
                np.maximum(
                    precision(0.0),
                    previous_support[None, :] - support[:, None],
                )
                @ previous_density
            )
        else:
            previous_calls = np.maximum(
                precision(0.0), precision(1.0) - support
            )
        previous_calls[0] = precision(1.0) - support[0]
        previous_calls[-1] = precision(0.0)
        calls = (
            np.maximum(
                precision(0.0),
                support[None, :] - support[1:-1, None],
            )
            @ density
        )
        dt = expiry - (expiries[index - 1] if index else precision(0.0))
        density_box = (
            support[1:-1] ** 2
            * dt
            / (support[2:] - support[:-2])
            * density[1:-1]
        )
        delta_calls = calls - previous_calls[1:-1]
        if (
            np.any(~np.isfinite(delta_calls))
            or np.any(~np.isfinite(density_box))
            or np.any(delta_calls <= 0.0)
            or np.any(density_box <= 0.0)
        ):
            raise DailyDataError(
                f"Canonical discrete-DLV expiry {index} has a singular transition"
            )
        interior = delta_calls / density_box
        if np.any(~np.isfinite(interior)) or np.any(interior <= 0.0):
            raise DailyDataError(
                f"Canonical discrete-DLV expiry {index} has invalid coefficients"
            )
        result.append(
            np.concatenate(
                (
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray(interior, dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                )
            )
        )
    return result


@version("1.0", dependencies=[SanosPricer, solve_sanos_tridiagonal])
def _canonical_discrete_dlv_densities(
    pricer: SanosPricer,
    dlv_squares: list[np.ndarray[Any, Any]],
) -> list[np.ndarray[Any, Any]]:
    """Decode canonical finite DLVs in extended precision for stabilization."""

    precision = np.longdouble
    if np.finfo(precision).eps >= np.finfo(np.float64).eps:
        raise DailyDataError(
            "Canonical discrete-DLV decoding requires NumPy extended precision"
        )
    if len(dlv_squares) != len(pricer.expiries):
        raise DailyDataError("Canonical discrete-DLV state has inconsistent lengths")
    expiries = np.asarray(pricer.expiries, dtype=precision)
    previous_support = np.ones(1, dtype=precision)
    previous_density = np.ones(1, dtype=precision)
    result: list[np.ndarray[Any, Any]] = []
    for index, expiry in enumerate(expiries):
        support = np.asarray(pricer.xstrikes[index], dtype=precision)
        coefficients = np.asarray(dlv_squares[index], dtype=precision)
        if (
            coefficients.shape != support.shape
            or coefficients[0] != 0.0
            or coefficients[-1] != 0.0
            or np.any(~np.isfinite(coefficients))
            or np.any(coefficients < 0.0)
        ):
            raise DailyDataError(
                f"Canonical discrete-DLV expiry {index} has invalid coefficients"
            )
        previous_calls = (
            np.maximum(
                precision(0.0),
                previous_support[None, :] - support[:, None],
            )
            @ previous_density
        )
        previous_calls[0] = precision(1.0) - support[0]
        previous_calls[-1] = precision(0.0)
        rhs = np.diff(
            np.diff(previous_calls, prepend=precision(1.0))
            / np.diff(support, prepend=precision(0.0)),
            append=precision(0.0),
        )
        dt = expiry - (expiries[index - 1] if index else precision(0.0))
        interior = coefficients[1:-1]
        w_plus = (
            interior
            * support[1:-1] ** 2
            * dt
            / ((support[2:] - support[1:-1]) * (support[2:] - support[:-2]))
        )
        w_minus = (
            interior
            * support[1:-1] ** 2
            * dt
            / ((support[1:-1] - support[:-2]) * (support[2:] - support[:-2]))
        )
        previous_density = solve_sanos_tridiagonal(
            w_minus=w_minus,
            w_plus=w_plus,
            rhs=rhs,
        )
        if np.any(~np.isfinite(previous_density)) or np.any(previous_density < 0.0):
            raise DailyDataError(
                f"Canonical discrete-DLV expiry {index} decoded an invalid density"
            )
        previous_support = support
        result.append(np.asarray(previous_density, dtype=np.float64))
    return result


@version(
    "1.1",
    dependencies=[
        SanosPricer,
        _canonical_discrete_dlv_squares,
        _canonical_discrete_dlv_densities,
    ],
)
def _stabilize_discrete_dlv_state(
    pricer: SanosPricer,
) -> tuple[SanosPricer, list[np.ndarray[Any, Any]], dict[str, float]]:
    """Apply one encode/decode cycle to make the persisted DLV pair stable."""

    initial_dlvs = _canonical_discrete_dlv_squares(pricer)
    stabilized_densities = _canonical_discrete_dlv_densities(pricer, initial_dlvs)
    stabilized = SanosPricer(
        smoothness=None,
        expiries=np.asarray(pricer.expiries, dtype=np.float64),
        variances=np.asarray(pricer.variances, dtype=np.float64),
        xstrikes=[np.asarray(value, dtype=np.float64) for value in pricer.xstrikes],
        xdensities=[np.asarray(value, dtype=np.float64) for value in stabilized_densities],
        min_half_spread=pricer.min_half_spread,
        min_dlv=pricer.min_dlv,
        max_dlv=pricer.max_dlv,
    )
    stabilized_dlvs = _canonical_discrete_dlv_squares(stabilized)
    return stabilized, stabilized_dlvs, {
        "density_max_abs": max(
            float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            for left, right in zip(
                stabilized.xdensities,
                pricer.xdensities,
                strict=True,
            )
        ),
        "dlv_max_abs": max(
            float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
            for left, right in zip(stabilized_dlvs, initial_dlvs, strict=True)
        ),
    }


@version("1.0")
def _is_discrete_dlv_solver_event(
    event: object,
    *,
    stage: str,
    statuses: frozenset[str],
) -> bool:
    """Authenticate one primary HiGHS event from the discrete-DLV solve."""

    fields = {
        "stage",
        "solver",
        "algorithm",
        "status",
        "is_fallback",
        "for_expiry",
        "objective",
    }
    if not isinstance(event, Mapping) or set(event) != fields:
        return False
    status = event.get("status")
    objective = event.get("objective")
    successful = status in {"optimal", "optimal_inaccurate"}
    valid_objective = (
        not isinstance(objective, bool)
        and isinstance(objective, (int, float))
        and math.isfinite(float(objective))
        if successful
        else objective is None
    )
    return (
        event.get("stage") == stage
        and event.get("solver") == "HIGHS"
        and event.get("algorithm") == "simplex"
        and isinstance(status, str)
        and status in statuses
        and event.get("is_fallback") is False
        and event.get("for_expiry") is False
        and valid_objective
    )


@version("1.0", dependencies=[_is_discrete_dlv_solver_event])
def _strict_dlv_fit_is_infeasible(solver_events: list[dict[str, object]]) -> bool:
    """Return whether HiGHS reported the one retry-eligible strict outcome."""

    return len(solver_events) == 1 and _is_discrete_dlv_solver_event(
        solver_events[0],
        stage="discrete_dlv_strict",
        statuses=frozenset({"infeasible", "infeasible_inaccurate"}),
    )


@version("1.0")
def _strict_dlv_fit_rejections_are_valid(values: tuple[str, ...]) -> bool:
    """Authenticate the closed, canonical strict-band rejection vocabulary."""

    allowed = {
        "post_stabilization_fit_cap",
        "post_stabilization_target_band",
        "post_stabilization_target_iv",
        "pre_stabilization_fit_cap",
        "pre_stabilization_target_band",
        "pre_stabilization_target_iv",
    }
    return (
        tuple(sorted(values)) == values
        and len(set(values)) == len(values)
        and set(values) <= allowed
    )


@version(
    "1.2",
    dependencies=[
        _is_discrete_dlv_solver_event,
        _strict_dlv_fit_rejections_are_valid,
    ],
)
def _discrete_dlv_solver_path_is_valid(
    solver_events: list[dict[str, object]],
    fit_mode: str,
    fallback_reason: str | None,
    strict_fit_rejections: tuple[str, ...],
) -> bool:
    """Validate the closed strict-or-L1 discrete-DLV solver grammar."""

    successful = frozenset({"optimal", "optimal_inaccurate"})
    if fit_mode == "strict_feasible":
        return (
            fallback_reason is None
            and not strict_fit_rejections
            and len(solver_events) == 1
            and _is_discrete_dlv_solver_event(
                solver_events[0],
                stage="discrete_dlv_strict",
                statuses=successful,
            )
        )
    if fit_mode == "uniform_l1_fallback":
        authenticated_rejections = _strict_dlv_fit_rejections_are_valid(
            strict_fit_rejections
        )
        strict_statuses = (
            frozenset({"infeasible", "infeasible_inaccurate"})
            if fallback_reason == "solver_infeasible"
            else successful
        )
        return (
            fallback_reason in {"solver_infeasible", "strict_fit_quality"}
            and authenticated_rejections
            and bool(strict_fit_rejections)
            == (fallback_reason == "strict_fit_quality")
            and len(solver_events) == 2
            and _is_discrete_dlv_solver_event(
                solver_events[0],
                stage="discrete_dlv_strict",
                statuses=strict_statuses,
            )
            and _is_discrete_dlv_solver_event(
                solver_events[1],
                stage="discrete_dlv_l1",
                statuses=successful,
            )
        )
    return False


@version(
    "1.9",
    dependencies=[
        SanosPricer,
        DiscreteDlvConfig,
        _canonical_discrete_dlv_densities,
        _canonical_discrete_dlv_squares,
        _bs_vega_sqrt_time,
        _discrete_dlv_state_metrics,
        _error_metrics,
        _projection_fit_price_bounds,
        _projection_target_band_metrics,
        _discrete_dlv_solver_path_is_valid,
        _validate_dlv_bounds,
    ],
)
def _validate_discrete_dlv_projection(
    *,
    smooth: SanosPricer,
    discrete: SanosPricer,
    raw_nms: np.ndarray[Any, Any],
    inner: np.ndarray[Any, Any],
    supports: np.ndarray[Any, Any],
    target_calls: np.ndarray[Any, Any],
    fitted_calls: np.ndarray[Any, Any],
    listed_strikes: list[np.ndarray[Any, Any]],
    dlv_squares: list[np.ndarray[Any, Any]],
    solver_events: list[dict[str, object]],
    solver_warnings: list[str],
    fit_mode: str,
    fallback_reason: str | None,
    strict_fit_rejections: tuple[str, ...],
    stabilization: Mapping[str, float],
    pre_state_metrics: Mapping[str, float | int],
    config: DiscreteDlvConfig,
) -> dict[str, object]:
    """Apply correctness-critical structural, inverse, and projection gates."""

    tolerance = config.tolerances
    if raw_nms.shape != (config.inner_nodes,) or inner.shape != (
        len(discrete.expiries),
        config.inner_nodes,
    ):
        raise DailyDataError("Discrete-DLV target geometry has an unexpected shape")
    expected_support_shape = (len(discrete.expiries), config.inner_nodes + 2)
    if supports.shape != expected_support_shape:
        raise DailyDataError("Discrete-DLV support geometry has an unexpected shape")
    if target_calls.shape != inner.shape or fitted_calls.shape != inner.shape:
        raise DailyDataError("Discrete-DLV target and fitted call arrays changed shape")
    if len(listed_strikes) != len(discrete.expiries) or len(dlv_squares) != len(
        discrete.expiries
    ):
        raise DailyDataError("Discrete-DLV expiry state has inconsistent lengths")
    if set(stabilization) != {
        "density_max_abs",
        "dlv_max_abs",
        "pre_target_call_max_abs",
        "post_target_call_max_abs",
        "pre_target_iv_max_abs",
        "post_target_iv_max_abs",
        "pre_target_band_max_violation",
        "post_target_band_max_violation",
    } or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in stabilization.values()
    ):
        raise DailyDataError("Discrete-DLV stabilization diagnostics are invalid")
    if not _discrete_dlv_solver_path_is_valid(
        solver_events,
        fit_mode,
        fallback_reason,
        strict_fit_rejections,
    ):
        raise DailyDataError(
            "Discrete-DLV solver path is invalid: "
            f"fit_mode={fit_mode!r}, events={solver_events!r}"
        )

    fit_price_bounds, fit_price_band_contract = _projection_fit_price_bounds(
        smooth=smooth,
        expiries=np.asarray(discrete.expiries, dtype=np.float64),
        inner=inner,
        target_calls=target_calls,
        config=config,
    )
    post_target_band_metrics = _projection_target_band_metrics(
        smooth=smooth,
        expiries=np.asarray(discrete.expiries, dtype=np.float64),
        inner=inner,
        fitted_calls=fitted_calls,
        bounds=fit_price_bounds,
        config=config,
    )
    if not math.isclose(
        float(post_target_band_metrics["maximum_absolute_iv_error"]),
        stabilization["post_target_iv_max_abs"],
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or not math.isclose(
        float(post_target_band_metrics["maximum_violation"]),
        stabilization["post_target_band_max_violation"],
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise DailyDataError("Discrete-DLV target-band diagnostics changed")

    canonical_decoded = _canonical_discrete_dlv_densities(discrete, dlv_squares)
    generic_decoded = discrete.densities_from_dlvsqs(dlv_squares)
    generic_decoded_pricer = SanosPricer(
        smoothness=None,
        expiries=np.asarray(discrete.expiries, dtype=np.float64),
        variances=np.asarray(discrete.variances, dtype=np.float64),
        xstrikes=[np.asarray(value, dtype=np.float64) for value in discrete.xstrikes],
        xdensities=generic_decoded,
        min_half_spread=discrete.min_half_spread,
        min_dlv=discrete.min_dlv,
        max_dlv=discrete.max_dlv,
    )
    canonical_decoded_pricer = SanosPricer(
        smoothness=None,
        expiries=np.asarray(discrete.expiries, dtype=np.float64),
        variances=np.asarray(discrete.variances, dtype=np.float64),
        xstrikes=[np.asarray(value, dtype=np.float64) for value in discrete.xstrikes],
        xdensities=canonical_decoded,
        min_half_spread=discrete.min_half_spread,
        min_dlv=discrete.min_dlv,
        max_dlv=discrete.max_dlv,
    )
    canonical_reencoded = _canonical_discrete_dlv_squares(discrete)
    mixed_reencoded = _canonical_discrete_dlv_squares(generic_decoded_pricer)
    post_state_metrics = _discrete_dlv_state_metrics(
        discrete,
        dlv_squares,
        config,
    )
    if set(pre_state_metrics) != set(post_state_metrics) or any(
        not math.isfinite(float(value)) for value in pre_state_metrics.values()
    ):
        raise DailyDataError("Discrete-DLV pre-stabilization metrics are invalid")

    density_mass_error = 0.0
    density_moment_error = 0.0
    canonical_density_inverse_error = 0.0
    generic_density_inverse_error = 0.0
    canonical_call_inverse_errors: list[float] = []
    generic_call_inverse_errors: list[float] = []
    canonical_fixed_point_error = 0.0
    mixed_reencode_error = 0.0
    projection_errors: list[float] = []
    conditioned_iv_errors: list[float] = []
    conditioned_iv_worst: dict[str, object] | None = None
    atm_iv_errors: list[float] = []
    singular_count = 0
    minimum_density = float("inf")
    minimum_dlv_square = float("inf")
    maximum_dlv_square = 0.0
    minimum_bound_count = 0
    maximum_bound_count = 0
    in_domain_by_source: dict[str, list[float]] = {
        name: [] for name in ("targets", "midpoints", "atm", "smooth_support", "listed")
    }
    conditioned_iv_by_source: dict[str, list[float]] = {
        name: [] for name in ("targets", "midpoints", "atm", "smooth_support", "listed")
    }
    out_of_domain: dict[str, list[float]] = {"left": [], "right": []}
    previous_strikes = np.asarray([1.0], dtype=np.float64)
    previous_density = np.asarray([1.0], dtype=np.float64)

    for index, expiry in enumerate(discrete.expiries):
        support = np.asarray(discrete.xstrikes[index], dtype=np.float64)
        density = np.asarray(discrete.xdensities[index], dtype=np.float64)
        dlv = np.asarray(dlv_squares[index], dtype=np.float64)
        if (
            support.shape != (config.inner_nodes + 2,)
            or density.shape != support.shape
            or dlv.shape != support.shape
            or np.any(~np.isfinite(support))
            or np.any(~np.isfinite(density))
            or np.any(~np.isfinite(dlv))
            or np.any(support <= 0.0)
            or np.any(np.diff(support) <= 0.0)
            or np.any(density < 0.0)
            or dlv[0] != 0.0
            or dlv[-1] != 0.0
        ):
            raise DailyDataError(f"Discrete-DLV expiry {index} has invalid state geometry")
        density_mass_error = max(density_mass_error, abs(float(np.sum(density)) - 1.0))
        density_moment_error = max(
            density_moment_error, abs(float(support @ density) - 1.0)
        )
        minimum_density = min(minimum_density, float(np.min(density)))
        canonical_density_inverse_error = max(
            canonical_density_inverse_error,
            float(np.max(np.abs(np.asarray(canonical_decoded[index]) - density))),
        )
        generic_density_inverse_error = max(
            generic_density_inverse_error,
            float(np.max(np.abs(np.asarray(generic_decoded[index]) - density))),
        )
        canonical_fixed_point_error = max(
            canonical_fixed_point_error,
            float(np.max(np.abs(np.asarray(canonical_reencoded[index]) - dlv))),
        )
        mixed_reencode_error = max(
            mixed_reencode_error,
            float(np.max(np.abs(np.asarray(mixed_reencoded[index]) - dlv))),
        )
        interior_dlv = dlv[1:-1]
        min_square = config.min_dlv**2
        max_square = config.max_dlv**2
        bound_diagnostics = _validate_dlv_bounds(
            interior_dlv,
            minimum_square=min_square,
            maximum_square=max_square,
            absolute_tolerance=tolerance.dlv_bound_abs,
            label=f"Discrete-DLV expiry {index}",
        )
        minimum_dlv_square = min(
            minimum_dlv_square, float(bound_diagnostics["minimum"])
        )
        maximum_dlv_square = max(
            maximum_dlv_square, float(bound_diagnostics["maximum"])
        )
        minimum_bound_count += int(bound_diagnostics["minimum_saturation_count"])
        maximum_bound_count += int(bound_diagnostics["maximum_saturation_count"])

        current_calls = (
            np.maximum(0.0, support[None, :] - support[1:-1, None]) @ density
        )
        previous_calls = (
            np.maximum(0.0, previous_strikes[None, :] - support[:, None])
            @ previous_density
        )
        previous_calls[0] = 1.0 - support[0]
        previous_calls[-1] = 0.0
        delta_calls = current_calls - previous_calls[1:-1]
        dt = float(discrete.expiries[index]) - (
            float(discrete.expiries[index - 1]) if index else 0.0
        )
        density_box = (
            support[1:-1] ** 2
            * dt
            / (support[2:] - support[:-2])
            * density[1:-1]
        )
        singular_count += int(np.sum((delta_calls <= 0.0) | (density_box <= 0.0)))

        midpoints = 0.5 * (inner[index, 1:] + inner[index, :-1])
        sources = {
            "targets": inner[index],
            "midpoints": midpoints,
            "atm": np.asarray([1.0], dtype=np.float64),
            "smooth_support": np.asarray(smooth.xstrikes[index], dtype=np.float64),
            "listed": np.asarray(listed_strikes[index], dtype=np.float64),
        }
        lower = float(inner[index, 0])
        upper = float(inner[index, -1])
        in_domain_parts: list[np.ndarray[Any, Any]] = []
        for name, values in sources.items():
            in_domain = np.unique(values[(values >= lower) & (values <= upper)])
            if in_domain.size:
                expected, expected_iv = smooth(
                    float(expiry), in_domain, and_vols=True
                )
                observed, observed_iv = discrete(
                    float(expiry), in_domain, and_vols=True
                )
                expected = np.asarray(expected, dtype=np.float64)
                observed = np.asarray(observed, dtype=np.float64)
                errors = np.abs(observed - expected)
                in_domain_by_source[name].extend(errors.tolist())
                source_vega_by_group = _bs_vega_sqrt_time(
                    in_domain,
                    np.asarray(expected_iv, dtype=np.float64),
                    float(expiry),
                )
                source_conditioned = (
                    source_vega_by_group >= config.conditioned_iv_vega_sqrt_floor
                )
                conditioned_iv_by_source[name].extend(
                    np.abs(
                        np.asarray(observed_iv, dtype=np.float64)
                        - np.asarray(expected_iv, dtype=np.float64)
                    )[source_conditioned].tolist()
                )
                in_domain_parts.append(in_domain)
            if name in {"smooth_support", "listed"}:
                for side, mask in (
                    ("left", values < lower),
                    ("right", values > upper),
                ):
                    outside = np.unique(values[mask])
                    if outside.size:
                        expected = np.asarray(smooth(float(expiry), outside), dtype=np.float64)
                        observed = np.asarray(discrete(float(expiry), outside), dtype=np.float64)
                        out_of_domain[side].extend(np.abs(observed - expected).tolist())
        validation = np.unique(np.concatenate(in_domain_parts))
        target_price, target_iv = smooth(float(expiry), validation, and_vols=True)
        projected_price, projected_iv = discrete(float(expiry), validation, and_vols=True)
        projection_errors.extend(np.abs(projected_price - target_price).tolist())
        source_vega = _bs_vega_sqrt_time(
            validation,
            target_iv,
            float(expiry),
        )
        conditioned = source_vega >= config.conditioned_iv_vega_sqrt_floor
        iv_errors = np.abs(projected_iv - target_iv)
        conditioned_iv_errors.extend(iv_errors[conditioned].tolist())
        if np.any(conditioned):
            conditioned_positions = np.flatnonzero(conditioned)
            worst_position = int(
                conditioned_positions[
                    np.argmax(iv_errors[conditioned_positions])
                ]
            )
            worst_error = float(iv_errors[worst_position])
            if (
                conditioned_iv_worst is None
                or worst_error > float(conditioned_iv_worst["absolute_iv_error"])
            ):
                strike = float(validation[worst_position])
                atm_volatility = math.sqrt(
                    float(discrete.variances[index]) / float(expiry)
                )
                epsilon_volatility = math.sqrt(
                    float(discrete.variances[0]) / float(discrete.expiries[0])
                )
                conditioned_iv_worst = {
                    "expiry_index": index,
                    "expiry_years": float(expiry),
                    "strike": strike,
                    "nms": float(
                        NMS.to_nms(
                            strike,
                            float(expiry),
                            volatility=atm_volatility,
                            epsilon_volatility=epsilon_volatility,
                        )
                    ),
                    "sources": sorted(
                        name
                        for name, values in sources.items()
                        if np.any(values == validation[worst_position])
                    ),
                    "target_call": float(target_price[worst_position]),
                    "projected_call": float(projected_price[worst_position]),
                    "absolute_call_error": float(
                        abs(projected_price[worst_position] - target_price[worst_position])
                    ),
                    "target_annual_iv": float(target_iv[worst_position]),
                    "projected_annual_iv": float(projected_iv[worst_position]),
                    "absolute_iv_error": worst_error,
                    "source_vega_sqrt_time": float(source_vega[worst_position]),
                    "vega_sqrt_time_distance_to_floor": float(
                        source_vega[worst_position]
                        - config.conditioned_iv_vega_sqrt_floor
                    ),
                }
        _, target_atm_iv = smooth(float(expiry), np.asarray([1.0]), and_vols=True)
        _, projected_atm_iv = discrete(float(expiry), np.asarray([1.0]), and_vols=True)
        atm_iv_errors.append(float(abs(projected_atm_iv[0] - target_atm_iv[0])))

        inverse_grid = np.unique(
            np.concatenate(
                (
                    support,
                    0.5 * (support[1:] + support[:-1]),
                    np.asarray([1.0]),
                    np.asarray(smooth.xstrikes[index], dtype=np.float64),
                    np.asarray(listed_strikes[index], dtype=np.float64),
                )
            )
        )
        canonical_call_inverse_errors.extend(
            np.abs(
                np.asarray(
                    canonical_decoded_pricer(float(expiry), inverse_grid),
                    dtype=np.float64,
                )
                - np.asarray(discrete(float(expiry), inverse_grid), dtype=np.float64)
            ).tolist()
        )
        generic_call_inverse_errors.extend(
            np.abs(
                np.asarray(
                    generic_decoded_pricer(float(expiry), inverse_grid),
                    dtype=np.float64,
                )
                - np.asarray(discrete(float(expiry), inverse_grid), dtype=np.float64)
            ).tolist()
        )
        previous_strikes = support
        previous_density = density

    projection_metrics = _error_metrics(np.asarray(projection_errors))
    iv_metrics = _error_metrics(np.asarray(conditioned_iv_errors))
    atm_metrics = _error_metrics(np.asarray(atm_iv_errors))
    canonical_inverse_call_metrics = _error_metrics(
        np.asarray(canonical_call_inverse_errors)
    )
    generic_inverse_call_metrics = _error_metrics(np.asarray(generic_call_inverse_errors))
    structural_failures = {
        "stabilization_density": (
            stabilization["density_max_abs"] > tolerance.density_inverse_max_abs
        ),
        "stabilization_dlv": (
            stabilization["dlv_max_abs"] > tolerance.dlv_fixed_point_max_abs
        ),
        "density_mass": density_mass_error > tolerance.density_mass_abs,
        "density_moment": density_moment_error > tolerance.density_moment_abs,
        "density_floor": (
            float(post_state_metrics["density_floor_shortfall"])
            > config.solver_kkt_tolerance
        ),
        "canonical_density_inverse": (
            canonical_density_inverse_error > tolerance.density_inverse_max_abs
        ),
        "generic_density_inverse": (
            generic_density_inverse_error > tolerance.density_inverse_max_abs
        ),
        "canonical_call_inverse": (
            canonical_inverse_call_metrics["max"] > tolerance.call_inverse_max_abs
        ),
        "generic_call_inverse": (
            generic_inverse_call_metrics["max"] > tolerance.call_inverse_max_abs
        ),
        "canonical_dlv_fixed_point": (
            canonical_fixed_point_error > tolerance.dlv_fixed_point_max_abs
        ),
        "singular_nodes": singular_count != 0,
    }
    fit_quality_failures = {
        "pre_stabilization_fit_cap": (
            stabilization["pre_target_call_max_abs"]
            > config.fit_call_max_abs + config.solver_kkt_tolerance
        ),
        "post_stabilization_fit_cap": (
            stabilization["post_target_call_max_abs"]
            > config.fit_call_max_abs + config.solver_kkt_tolerance
        ),
        "pre_stabilization_target_iv": (
            stabilization["pre_target_iv_max_abs"]
            > tolerance.conditioned_annual_iv_max_abs
        ),
        "post_stabilization_target_iv": (
            stabilization["post_target_iv_max_abs"]
            > tolerance.conditioned_annual_iv_max_abs
        ),
        "pre_stabilization_target_band": (
            stabilization["pre_target_band_max_violation"]
            > config.solver_kkt_tolerance
        ),
        "post_stabilization_target_band": (
            stabilization["post_target_band_max_violation"]
            > config.solver_kkt_tolerance
        ),
        "projection_call": projection_metrics["max"] > tolerance.projection_call_max_abs,
        "atm_iv": atm_metrics["max"] > tolerance.atm_annual_iv_max_abs,
        "conditioned_iv_max": (
            iv_metrics["max"] > tolerance.conditioned_annual_iv_max_abs
        ),
        "conditioned_iv_p99": (
            iv_metrics["p99"] > tolerance.conditioned_annual_iv_p99_abs
        ),
    }
    rejected = sorted(name for name, failed in structural_failures.items() if failed)
    if rejected:
        raise DailyDataError(
            "Discrete-DLV projection failed strict validation: " + ", ".join(rejected)
        )
    fit_quality_rejections = sorted(
        name for name, failed in fit_quality_failures.items() if failed
    )
    strict_band_rejections = sorted(
        name
        for name in (
            "pre_stabilization_fit_cap",
            "post_stabilization_fit_cap",
            "pre_stabilization_target_iv",
            "post_stabilization_target_iv",
            "pre_stabilization_target_band",
            "post_stabilization_target_band",
        )
        if fit_quality_failures[name]
    )
    if fit_mode == "strict_feasible" and strict_band_rejections:
        raise _StrictDlvFitQualityError(strict_band_rejections)
    return {
        "schema": "sanos-discrete-dlv-projection-diagnostics-v5",
        "outcome": "completed",
        "source": "smooth_sanos_surface",
        "fit_mode": fit_mode,
        "fallback_reason": fallback_reason,
        "strict_fit_rejections": list(strict_fit_rejections),
        "fit_quality": {
            "accepted": not fit_quality_rejections,
            "failures": fit_quality_rejections,
        },
        "objective_weighting": config.objective_weighting,
        "fit_call_max_abs": config.fit_call_max_abs,
        "target_fit_price_band": {
            **fit_price_band_contract,
            **post_target_band_metrics,
            "pre_stabilization_maximum_absolute_iv_error": stabilization[
                "pre_target_iv_max_abs"
            ],
            "pre_stabilization_maximum_violation": stabilization[
                "pre_target_band_max_violation"
            ],
        },
        "solver_kkt_tolerance": config.solver_kkt_tolerance,
        "constraint_scaling": "geometric_box_and_bound",
        "constraint_dlv_bounds": [
            config.min_dlv + config.constraint_dlv_buffer,
            config.max_dlv - config.constraint_dlv_buffer,
        ],
        "density_mass_floor": config.density_mass_floor,
        "numerical_regularization": {
            "density_mass_floor": config.density_mass_floor,
            "constraint_dlv_buffer": config.constraint_dlv_buffer,
            "required_density_floor_mass_per_expiry": post_state_metrics[
                "required_density_floor_mass_per_expiry"
            ],
            "required_density_floor_mass_all_expiries": post_state_metrics[
                "required_density_floor_mass_all_expiries"
            ],
            "pre_stabilization_state": dict(pre_state_metrics),
            "post_stabilization_state": post_state_metrics,
            "dlv_epsilon_warning_count": sum(
                "below min dlv constraint" in value
                or "above max dlv constraint" in value
                for value in solver_warnings
            ),
        },
        "stabilization_cycles": 1,
        "projection_domain_nms": list(config.nms_range),
        "inner_nodes": config.inner_nodes,
        "support_nodes": config.inner_nodes + 2,
        "expiries": len(discrete.expiries),
        "density_mass_max_abs": density_mass_error,
        "density_moment_max_abs": density_moment_error,
        "canonical_density_inverse_max_abs": canonical_density_inverse_error,
        "generic_density_inverse_max_abs": generic_density_inverse_error,
        "canonical_call_inverse": canonical_inverse_call_metrics,
        "generic_call_inverse": generic_inverse_call_metrics,
        "canonical_dlv_fixed_point_max_abs": canonical_fixed_point_error,
        "generic_decode_canonical_reencode_max_abs": mixed_reencode_error,
        "singular_nodes": singular_count,
        "minimum_density": minimum_density,
        "minimum_dlv_square": minimum_dlv_square,
        "maximum_dlv_square": maximum_dlv_square,
        "minimum_bound_shortfall": max(config.min_dlv**2 - minimum_dlv_square, 0.0),
        "maximum_bound_excess": max(maximum_dlv_square - config.max_dlv**2, 0.0),
        "minimum_bound_saturation_count": minimum_bound_count,
        "maximum_bound_saturation_count": maximum_bound_count,
        "projection_call": projection_metrics,
        "projection_call_by_source": {
            name: _error_metrics(np.asarray(values))
            for name, values in in_domain_by_source.items()
        },
        "out_of_domain_call_report_only": {
            side: _error_metrics(np.asarray(values)) for side, values in out_of_domain.items()
        },
        "atm_annual_iv": atm_metrics,
        "conditioned_annual_iv": iv_metrics,
        "conditioned_annual_iv_by_source": {
            name: _error_metrics(np.asarray(values))
            for name, values in conditioned_iv_by_source.items()
        },
        "conditioned_annual_iv_worst": conditioned_iv_worst,
        "conditioned_iv_vega_sqrt_floor": config.conditioned_iv_vega_sqrt_floor,
        "conditioned_iv_excluded_points": int(projection_metrics["count"])
        - int(iv_metrics["count"]),
        "solver_events": solver_events,
        "solver_warnings": solver_warnings,
        "stabilization_density_max_abs": stabilization["density_max_abs"],
        "stabilization_dlv_max_abs": stabilization["dlv_max_abs"],
        "pre_stabilization_target_call_max_abs": stabilization[
            "pre_target_call_max_abs"
        ],
        "post_stabilization_target_call_max_abs": stabilization[
            "post_target_call_max_abs"
        ],
        "post_stabilization_target_call_slack": (
            config.fit_call_max_abs - stabilization["post_target_call_max_abs"]
        ),
    }


@version(
    "1.8",
    dependencies=[
        SanosFitter,
        SanosPricer,
        _canonical_discrete_dlv_squares,
        _discrete_dlv_state_metrics,
        _discrete_dlv_grid,
        _projection_fit_price_bounds,
        _projection_target_band_metrics,
        _strict_dlv_fit_is_infeasible,
        _stabilize_discrete_dlv_state,
        _validate_discrete_dlv_projection,
        _float_column,
        _nullable_float64,
        DiscreteDlvConfig,
        canonical_json,
        canonical_sha256,
    ],
)
def _fit_discrete_dlv_projection(
    frame: pl.DataFrame,
    *,
    smooth_pricer: ArbFreePricer,
    selected_positions: list[np.ndarray[Any, Any]],
    config: DiscreteDlvConfig,
) -> list[pl.Series]:
    """Fit a finite DLV representation, relaxing only infeasible fit bands."""

    if smooth_pricer.mstrikes is None or smooth_pricer.mdensities is None:
        raise DailyDataError("Smooth SANOS returned no common-surface state for DLV projection")
    expiries = np.asarray(smooth_pricer.expiries, dtype=np.float64)
    variances = np.asarray(smooth_pricer.variances, dtype=np.float64)
    if (
        len(expiries) == 0
        or np.any(~np.isfinite(variances))
        or np.any(variances <= 0.0)
        or (len(variances) > 1 and np.any(np.diff(variances) <= 0.0))
    ):
        raise DailyDataError("Smooth SANOS variances are not positive and strictly increasing")
    smooth = SanosPricer(
        smoothness=float(smooth_pricer.smoothness),
        expiries=expiries,
        variances=variances,
        xstrikes=[np.asarray(value, dtype=np.float64) for value in smooth_pricer.mstrikes],
        xdensities=[np.asarray(value, dtype=np.float64) for value in smooth_pricer.mdensities],
        min_half_spread=float(smooth_pricer.min_half_spread),
    )
    atm_volatilities = np.sqrt(variances / expiries)
    raw_nms, inner, supports = _discrete_dlv_grid(expiries, atm_volatilities, config)
    target_calls = np.asarray(smooth(expiries[:, None], inner), dtype=np.float64)
    fit_price_bounds, _ = _projection_fit_price_bounds(
        smooth=smooth,
        expiries=expiries,
        inner=inner,
        target_calls=target_calls,
        config=config,
    )
    source_indices = [int(value) for value in smooth_pricer.input_index]
    listed_strikes = [
        _float_column(frame[selected_positions[source_index]], "normalized_strike")
        for source_index in source_indices
    ]
    solver_events: list[dict[str, object]] = []

    def stabilize_and_measure(
        candidate: SanosPricer,
    ) -> tuple[
        SanosPricer,
        list[np.ndarray[Any, Any]],
        np.ndarray[Any, Any],
        Mapping[str, float | int],
        dict[str, float],
    ]:
        pre_fitted = np.asarray(candidate(expiries[:, None], inner), dtype=np.float64)
        pre_metrics = _discrete_dlv_state_metrics(
            candidate,
            _canonical_discrete_dlv_squares(candidate),
            config,
        )
        stabilized, squares, stabilization_base = _stabilize_discrete_dlv_state(
            candidate
        )
        post_fitted = np.asarray(
            stabilized(expiries[:, None], inner), dtype=np.float64
        )
        pre_band_metrics = _projection_target_band_metrics(
            smooth=smooth,
            expiries=expiries,
            inner=inner,
            fitted_calls=pre_fitted,
            bounds=fit_price_bounds,
            config=config,
        )
        post_band_metrics = _projection_target_band_metrics(
            smooth=smooth,
            expiries=expiries,
            inner=inner,
            fitted_calls=post_fitted,
            bounds=fit_price_bounds,
            config=config,
        )
        metrics = {
            **stabilization_base,
            "pre_target_call_max_abs": float(
                np.max(np.abs(pre_fitted - target_calls))
            ),
            "post_target_call_max_abs": float(
                np.max(np.abs(post_fitted - target_calls))
            ),
            "pre_target_iv_max_abs": float(
                pre_band_metrics["maximum_absolute_iv_error"]
            ),
            "post_target_iv_max_abs": float(
                post_band_metrics["maximum_absolute_iv_error"]
            ),
            "pre_target_band_max_violation": float(
                pre_band_metrics["maximum_violation"]
            ),
            "post_target_band_max_violation": float(
                post_band_metrics["maximum_violation"]
            ),
        }
        return stabilized, squares, post_fitted, pre_metrics, metrics

    fallback_reason: str | None = None
    strict_fit_rejections: tuple[str, ...] = ()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            def solve(*, stage: str, impose_fit_bounds: bool) -> SanosPricer:
                return SanosFitter(
                    smoothness=None,
                    timeout_seconds=config.solver_timeout_seconds,
                    min_dlv=config.min_dlv + config.constraint_dlv_buffer,
                    max_dlv=config.max_dlv - config.constraint_dlv_buffer,
                    min_density_vol=config.min_density_vol,
                    min_density_cap=config.min_density_cap,
                    density_mass_floor=config.density_mass_floor,
                    scale_dlv_constraints=True,
                    solver_kkt_tolerance=config.solver_kkt_tolerance,
                    max_abs_fit_error=(
                        config.fit_call_max_abs if impose_fit_bounds else None
                    ),
                    allow_solver_fallback=False,
                    _raise_on_eps=not config.allow_solver_epsilon_warnings,
                )(
                    expiries=expiries,
                    strikes=[value for value in inner],
                    mids=[value for value in target_calls],
                    xstrikes=[value for value in supports],
                    weights=None,
                    fit_price_bounds=(fit_price_bounds if impose_fit_bounds else None),
                    vols=atm_volatilities,
                    message_context=lambda: (
                        f"{frame.get_column('ticker').item(0)} "
                        f"{frame.get_column('trading_date').item(0)} "
                        f"discrete DLV projection ({stage})"
                    ),
                    diagnostic_sink=lambda event: solver_events.append(
                        {"stage": stage, **dict(event)}
                    ),
                    verbose=Context.quiet,
                )

            fit_mode = "strict_feasible"
            strict_start = len(solver_events)
            try:
                discrete = solve(
                    stage="discrete_dlv_strict",
                    impose_fit_bounds=True,
                )
            except RuntimeError:
                strict_events = solver_events[strict_start:]
                if not _strict_dlv_fit_is_infeasible(strict_events):
                    raise
                fit_mode = "uniform_l1_fallback"
                fallback_reason = "solver_infeasible"
                discrete = solve(
                    stage="discrete_dlv_l1",
                    impose_fit_bounds=False,
                )
            (
                discrete,
                dlv_squares,
                fitted_calls,
                pre_state_metrics,
                stabilization,
            ) = stabilize_and_measure(discrete)
        except Exception as exc:
            failure = canonical_json(
                {
                    "outcome": "failed",
                    "solver_events": solver_events,
                    "solver_timeout_seconds": config.solver_timeout_seconds,
                }
            )
            raise DailyDataError(
                f"Discrete-DLV projection solver failed: {exc}; diagnostics={failure}"
            ) from exc
    solver_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]

    def validate() -> dict[str, object]:
        return _validate_discrete_dlv_projection(
            smooth=smooth,
            discrete=discrete,
            raw_nms=raw_nms,
            inner=inner,
            supports=supports,
            target_calls=target_calls,
            fitted_calls=fitted_calls,
            listed_strikes=listed_strikes,
            dlv_squares=dlv_squares,
            solver_events=solver_events,
            solver_warnings=solver_warnings,
            fit_mode=fit_mode,
            fallback_reason=fallback_reason,
            strict_fit_rejections=strict_fit_rejections,
            stabilization=stabilization,
            pre_state_metrics=pre_state_metrics,
            config=config,
        )

    try:
        diagnostics = validate()
    except _StrictDlvFitQualityError as exc:
        strict_fit_rejections = exc.failures
        fit_mode = "uniform_l1_fallback"
        fallback_reason = "strict_fit_quality"
        with warnings.catch_warnings(record=True) as fallback_warnings:
            warnings.simplefilter("always")
            try:
                discrete = solve(
                    stage="discrete_dlv_l1",
                    impose_fit_bounds=False,
                )
                (
                    discrete,
                    dlv_squares,
                    fitted_calls,
                    pre_state_metrics,
                    stabilization,
                ) = stabilize_and_measure(discrete)
            except Exception as fallback_exc:
                failure = canonical_json(
                    {
                        "outcome": "failed",
                        "fallback_reason": fallback_reason,
                        "solver_events": solver_events,
                        "solver_timeout_seconds": config.solver_timeout_seconds,
                        "strict_fit_rejections": list(strict_fit_rejections),
                    }
                )
                raise DailyDataError(
                    "Discrete-DLV projection solver failed after strict fit-quality "
                    f"rejection: {fallback_exc}; diagnostics={failure}"
                ) from fallback_exc
        solver_warnings.extend(
            f"{item.category.__name__}: {item.message}" for item in fallback_warnings
        )
        diagnostics = validate()
    diagnostics_json = canonical_json(diagnostics)
    configuration_id = canonical_sha256({"discrete_dlv": config})
    count = frame.height
    anchors = np.zeros(count, dtype=bool)
    input_indices: list[int | None] = [None] * count
    time_increments = np.full(count, np.nan, dtype=np.float64)
    atm_values = np.full(count, np.nan, dtype=np.float64)
    variance_values = np.full(count, np.nan, dtype=np.float64)
    nms_values: list[list[float] | None] = [None] * count
    strike_values: list[list[float] | None] = [None] * count
    density_values: list[list[float] | None] = [None] * count
    dlv_values: list[list[float] | None] = [None] * count
    target_values: list[list[float] | None] = [None] * count
    fitted_values: list[list[float] | None] = [None] * count
    increments = np.diff(expiries, prepend=0.0)
    for index, source_index in enumerate(source_indices):
        rows = selected_positions[source_index]
        if len(rows) == 0:
            raise DailyDataError("A retained smooth expiry has no deterministic anchor row")
        anchor = int(rows[0])
        anchors[anchor] = True
        input_indices[anchor] = index
        time_increments[anchor] = float(increments[index])
        atm_values[anchor] = float(atm_volatilities[index])
        variance_values[anchor] = float(variances[index])
        nms_values[anchor] = raw_nms.tolist()
        strike_values[anchor] = np.asarray(discrete.xstrikes[index], dtype=np.float64).tolist()
        density_values[anchor] = np.asarray(
            discrete.xdensities[index], dtype=np.float64
        ).tolist()
        dlv_values[anchor] = np.asarray(dlv_squares[index], dtype=np.float64).tolist()
        target_values[anchor] = target_calls[index].tolist()
        fitted_values[anchor] = fitted_calls[index].tolist()
    return [
        pl.Series("discrete_dlv_anchor", anchors),
        pl.Series("discrete_dlv_input_expiry_index", input_indices, dtype=pl.Int32),
        _nullable_float64("discrete_dlv_time_increment_years", time_increments),
        _nullable_float64("discrete_dlv_atm_annual_volatility", atm_values),
        _nullable_float64("discrete_dlv_input_total_variance", variance_values),
        pl.Series("discrete_dlv_nms", nms_values, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_model_strikes", strike_values, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_model_density", density_values, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_squares", dlv_values, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_target_calls", target_values, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_fitted_calls", fitted_values, dtype=pl.List(pl.Float64)),
        pl.Series(
            "discrete_dlv_diagnostics_json", [diagnostics_json] * count, dtype=pl.String
        ),
        pl.Series(
            "discrete_dlv_configuration_id", [configuration_id] * count, dtype=pl.String
        ),
    ]


@version("1.3", dependencies=[DiscreteDlvConfig, canonical_json, canonical_sha256])
def _empty_discrete_dlv_columns(
    count: int,
    *,
    config: DiscreteDlvConfig,
    outcome: str,
) -> list[pl.Series]:
    """Return typed null DLV state for a day with no smooth model state."""

    null_floats = [None] * count
    null_lists: list[list[float] | None] = [None] * count
    diagnostics = canonical_json(
        {
            "schema": "sanos-discrete-dlv-projection-diagnostics-v5",
            "outcome": outcome,
            "source": "smooth_sanos_surface",
        }
    )
    configuration_id = canonical_sha256({"discrete_dlv": config})
    return [
        pl.Series("discrete_dlv_anchor", [False] * count, dtype=pl.Boolean),
        pl.Series("discrete_dlv_input_expiry_index", [None] * count, dtype=pl.Int32),
        pl.Series("discrete_dlv_time_increment_years", null_floats, dtype=pl.Float64),
        pl.Series("discrete_dlv_atm_annual_volatility", null_floats, dtype=pl.Float64),
        pl.Series("discrete_dlv_input_total_variance", null_floats, dtype=pl.Float64),
        pl.Series("discrete_dlv_nms", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_model_strikes", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_model_density", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_squares", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_target_calls", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series("discrete_dlv_fitted_calls", null_lists, dtype=pl.List(pl.Float64)),
        pl.Series(
            "discrete_dlv_diagnostics_json", [diagnostics] * count, dtype=pl.String
        ),
        pl.Series(
            "discrete_dlv_configuration_id", [configuration_id] * count, dtype=pl.String
        ),
    ]


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
    "1.3",
    dependencies=[
        _canonical_fitted_rows,
        _nullable_float32,
        _nullable_float64,
        _empty_discrete_dlv_columns,
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
    discrete_dlv_config: DiscreteDlvConfig,
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
        *_empty_discrete_dlv_columns(
            count,
            config=discrete_dlv_config,
            outcome="no_smooth_state",
        ),
        pl.lit(str(fit_smooth_sanos.version)).alias("stage_version"),
    ).select(*FITTED_SCHEMA)
    return _canonical_fitted_rows(result.cast(FITTED_SCHEMA, strict=True))


@version(
    "2.3",
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
        _fit_discrete_dlv_projection,
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
            discrete_dlv_config=config.discrete_dlv,
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
    dlv_columns = _fit_discrete_dlv_projection(
        frame,
        smooth_pricer=pricer,
        selected_positions=positions,
        config=config.discrete_dlv,
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
        *dlv_columns,
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
def discrete_dlv_pricer(frame: pl.DataFrame) -> SanosPricer:
    """Reconstruct the canonical finite DLV projection from its anchor rows."""

    return _reconstruct_pricer(
        frame,
        index_column="discrete_dlv_input_expiry_index",
        strikes_column="discrete_dlv_model_strikes",
        density_column="discrete_dlv_model_density",
        variance_column="discrete_dlv_input_total_variance",
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
    "discrete_dlv_pricer",
    "fit_initial_no_arbitrage",
    "fit_smooth_sanos",
    "initial_no_arb_pricer",
    "selected_reference_view",
    "smooth_sanos_pricer",
]
