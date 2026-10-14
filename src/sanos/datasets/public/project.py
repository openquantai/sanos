"""Pure projection from one fitted internal day to the public daily contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from cdxcore.version import version

from sanos.datasets.public.config import PublicProjectionConfig
from sanos.datasets.public.schema import public_polars_schema
from sanos.model.tools import SanosPricer
from sanos.widb.calendar import (
    NMS,
    business_clock_policy,
    count_business_days_between,
    normalized_strike_policy,
)
from sanos.widb.daily import DailyDataError
from sanos.widb.surface import (
    FITTED_SCHEMA,
    FITTED_SCHEMA_VERSION,
    _canonical_discrete_dlv_densities,
    discrete_dlv_pricer,
    smooth_sanos_pricer,
)


class PublicProjectionError(DailyDataError):
    """Raised when an internal fitted day cannot satisfy the public contract."""


@version("1.0")
@dataclass(frozen=True, slots=True)
class CarryCurve:
    """Private build evidence used only to compute the following public return."""

    trading_date: date
    spot: float
    business_dtes: tuple[float, ...]
    forward_over_spot: tuple[float, ...]
    discount_factors: tuple[float, ...]


@version("1.0", dependencies=[CarryCurve])
@dataclass(frozen=True, slots=True)
class ProjectedPublicDay:
    """One public row plus private evidence that is never serialized as row data."""

    row: dict[str, Any]
    carry: CarryCurve
    book_count: int
    expiry_count: int


def _fail(message: str) -> None:
    raise PublicProjectionError(message)


@version(FITTED_SCHEMA_VERSION)
def _require_fitted_schema(frame: pl.DataFrame) -> None:
    if frame.schema != pl.Schema(FITTED_SCHEMA):
        _fail("Public projection source does not have the exact fitted-day schema")


@version("1.0")
def _one(frame: pl.DataFrame, column: str) -> Any:
    values = frame.get_column(column).unique().to_list()
    if len(values) != 1 or values[0] is None:
        _fail(f"{column} must contain exactly one non-null value within a day")
    return values[0]


@version("1.0")
def _expiry_record(frame: pl.DataFrame, index_column: str, index: int) -> dict[str, Any]:
    group = frame.filter(pl.col(index_column) == index)
    if group.is_empty():
        _fail(f"Missing source rows for {index_column}={index}")
    columns = (
        "expiry_id",
        "expiry_date",
        "business_dte",
        "calendar_dte",
        "expiry_is_am",
        "time_to_expiry_years",
        "discount_factor",
        "implied_forward",
        "spot_reference",
        "spot_normalized_implied_forward",
    )
    record: dict[str, Any] = {}
    for column in columns:
        values = group.get_column(column).unique().to_list()
        if len(values) != 1 or values[0] is None:
            _fail(f"{index_column}={index} has inconsistent {column}")
        record[column] = values[0]
    return record


@version("1.0", dependencies=[_expiry_record])
def _state_records(frame: pl.DataFrame, index_column: str) -> tuple[dict[str, Any], ...]:
    indices = sorted(
        int(value)
        for value in frame.get_column(index_column).drop_nulls().unique().to_list()
    )
    if not indices or indices != list(range(len(indices))):
        _fail(f"{index_column} must be non-empty and contiguous from zero")
    return tuple(_expiry_record(frame, index_column, index) for index in indices)


@version("1.0")
def _atm_volatility(
    time_years: np.ndarray | float,
    pricer: SanosPricer,
) -> np.ndarray:
    requested = np.asarray(time_years, dtype=np.float64)
    clipped = np.clip(requested, float(pricer.expiries[0]), float(pricer.expiries[-1]))
    variance = np.interp(clipped, pricer.expiries, pricer.variances)
    result = np.sqrt(variance / clipped)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        _fail("Model-state ATM volatility interpolation produced invalid values")
    return np.asarray(result, dtype=np.float64)


@version("1.0")
def _consistent_curve_node(
    nodes: dict[float, tuple[float, float]],
    business_dte: float,
    forward_over_spot: float,
    discount_factor: float,
) -> None:
    if (
        not math.isfinite(business_dte)
        or business_dte <= 0.0
        or not math.isfinite(forward_over_spot)
        or forward_over_spot <= 0.0
        or not math.isfinite(discount_factor)
        or discount_factor <= 0.0
    ):
        _fail("Carry curves require positive finite DTE, forward ratio, and discount")
    previous = nodes.get(business_dte)
    candidate = (forward_over_spot, discount_factor)
    if previous is not None and previous != candidate:
        _fail(f"Carry curve has inconsistent duplicate node at DTE {business_dte}")
    nodes[business_dte] = candidate


@version("1.0", dependencies=[CarryCurve, _consistent_curve_node])
def _source_carry_curve(
    trading_date: date,
    spot: float,
    frame: pl.DataFrame,
) -> CarryCurve:
    if not math.isfinite(spot) or spot <= 0.0:
        _fail("Internal spot_reference must be positive and finite")
    nodes: dict[float, tuple[float, float]] = {0.0: (1.0, 1.0)}
    # CD-004 uses every distinct positive source-day forward/discount node,
    # including expiries which fitting legitimately omitted from model state.
    records = frame.select(
        "business_dte", "implied_forward", "discount_factor"
    ).unique(maintain_order=True)
    for record in records.iter_rows(named=True):
        _consistent_curve_node(
            nodes,
            float(record["business_dte"]),
            float(record["implied_forward"]) / spot,
            float(record["discount_factor"]),
        )
    ordered = sorted(nodes.items())
    return CarryCurve(
        trading_date=trading_date,
        spot=spot,
        business_dtes=tuple(item[0] for item in ordered),
        forward_over_spot=tuple(item[1][0] for item in ordered),
        discount_factors=tuple(item[1][1] for item in ordered),
    )


@version("2.0", dependencies=[SanosPricer])
def _state_payload(
    pricer: SanosPricer,
    source_records: tuple[dict[str, Any], ...],
    public_index_by_expiry_id: dict[int, int],
    *,
    smooth: bool,
    dlv_squares: tuple[np.ndarray, ...] | None = None,
) -> dict[str, Any]:
    if len(source_records) != len(pricer.expiries):
        _fail("Model state and source expiry geometry have different lengths")
    if smooth and dlv_squares is not None:
        _fail("Smooth SANOS payload cannot contain a discrete-DLV vector")
    if not smooth and (dlv_squares is None or len(dlv_squares) != len(pricer.expiries)):
        _fail("Discrete-DLV payload differs from the stored expiry state")
    increments = np.diff(np.concatenate(([0.0], np.asarray(pricer.expiries, dtype=np.float64))))
    model_expiries: list[dict[str, Any]] = []
    for index, record in enumerate(source_records):
        expiry_id = int(record["expiry_id"])
        try:
            public_index = public_index_by_expiry_id[expiry_id]
        except KeyError as exc:
            raise PublicProjectionError(
                "Model expiry is absent from public union geometry"
            ) from exc
        item: dict[str, Any] = {
            "expiry_index": public_index,
            "time_increment_years": float(increments[index]),
            (
                "kernel_input_total_variance" if smooth else "input_total_variance"
            ): float(pricer.variances[index]),
            "normalized_support": np.asarray(
                pricer.xstrikes[index], dtype=np.float64
            ).tolist(),
            "density_mass": np.asarray(
                pricer.xdensities[index], dtype=np.float64
            ).tolist(),
        }
        if dlv_squares is not None:
            item["squared_dlv"] = np.asarray(
                dlv_squares[index], dtype=np.float64
            ).tolist()
        model_expiries.append(item)
    result: dict[str, Any] = {"model_expiries": model_expiries}
    if smooth:
        if pricer.smoothness is None:
            _fail("Smooth SANOS state has no smoothness")
        result["smoothness"] = float(pricer.smoothness)
        return {"smoothness": result["smoothness"], "model_expiries": model_expiries}
    return result


@version("1.1", dependencies=[SanosPricer, _canonical_discrete_dlv_densities])
def _stored_dlv_squares(
    frame: pl.DataFrame,
    pricer: SanosPricer,
) -> tuple[np.ndarray, ...]:
    """Read and independently verify the persisted canonical DLV vectors."""

    anchors = frame.filter(pl.col("discrete_dlv_anchor")).sort(
        "discrete_dlv_input_expiry_index"
    )
    indices = anchors.get_column("discrete_dlv_input_expiry_index").to_list()
    if indices != list(range(len(pricer.expiries))):
        _fail("Stored discrete-DLV anchors are not contiguous from zero")
    values = tuple(
        np.asarray(value, dtype=np.float64)
        for value in anchors.get_column("discrete_dlv_squares").to_list()
    )
    if any(
        value.shape != np.asarray(pricer.xstrikes[index]).shape
        or np.any(~np.isfinite(value))
        for index, value in enumerate(values)
    ):
        _fail("Stored discrete-DLV vectors have invalid shape or values")
    for label, decoded in (
        ("canonical", _canonical_discrete_dlv_densities(pricer, list(values))),
        ("generic", pricer.densities_from_dlvsqs(list(values))),
    ):
        residual = max(
            float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
            for observed, expected in zip(decoded, pricer.xdensities, strict=True)
        )
        if residual > 1.0e-12:
            _fail(
                f"Stored discrete-DLV {label} density round trip residual is {residual}"
            )
    return values


@version("1.0", dependencies=[normalized_strike_policy])
def _daily_front_mask(
    *,
    business_dte: float,
    time_years: float,
    strikes: np.ndarray,
    active: np.ndarray,
    is_american: np.ndarray,
    smooth_available: np.ndarray,
    smooth_fit: np.ndarray,
    atm_volatility: float,
    epsilon_atm_volatility: float,
    config: PublicProjectionConfig,
) -> np.ndarray:
    """Apply the exact side-specific daily-front public population rule."""

    excluded = any(
        abs(business_dte - value) <= config.exclusion_tolerance
        for value in config.excluded_business_dtes
    )
    if excluded or business_dte < 0.0 or business_dte > config.maximum_business_dte:
        return np.zeros(strikes.shape, dtype=bool)
    bounds = np.asarray(
        NMS.from_nms(
            np.asarray(config.nms_range, dtype=np.float64),
            time_years,
            volatility=atm_volatility,
            epsilon_volatility=epsilon_atm_volatility,
        ),
        dtype=np.float64,
    )
    strike64 = strikes.astype(np.float64)
    return (
        active.astype(bool)
        & ~is_american.astype(bool)
        & smooth_available.astype(bool)
        & np.isfinite(smooth_fit.astype(np.float64))
        & (strike64 >= bounds[0])
        & (strike64 <= bounds[1])
    )


@version(
    "1.0",
    dependencies=[_atm_volatility, _daily_front_mask, normalized_strike_policy],
)
def _book_payload(
    frame: pl.DataFrame,
    *,
    expiry_records: tuple[dict[str, Any], ...],
    public_index_by_expiry_id: dict[int, int],
    surface_pricer: SanosPricer,
    epsilon_atm_vol: float,
    config: PublicProjectionConfig,
) -> list[dict[str, Any]]:
    books: list[dict[str, Any]] = []
    eligible = frame.filter(pl.col("options_filter_eligible")).sort(
        "time_to_expiry_years", "expiry_id", "normalized_strike", maintain_order=True
    )
    for record in expiry_records:
        expiry_id = int(record["expiry_id"])
        group = eligible.filter(pl.col("expiry_id") == expiry_id).sort(
            "normalized_strike", maintain_order=True
        )
        if group.is_empty():
            continue
        business_dte = float(record["business_dte"])
        time_years = float(record["time_to_expiry_years"])
        atm_vol = float(_atm_volatility(time_years, surface_pricer))
        strikes = group.get_column("normalized_strike").to_numpy()
        for is_call, active_column, fit_column, bid_column, ask_column in (
            (
                True,
                "call_active",
                "smooth_call_mid",
                "spot_normalized_market_call_bid",
                "spot_normalized_market_call_ask",
            ),
            (
                False,
                "put_active",
                "smooth_put_mid",
                "spot_normalized_market_put_bid",
                "spot_normalized_market_put_ask",
            ),
        ):
            fit = group.get_column(fit_column).cast(pl.Float64).to_numpy()
            selected = _daily_front_mask(
                business_dte=business_dte,
                time_years=time_years,
                strikes=strikes,
                active=group.get_column(active_column).to_numpy(),
                is_american=group.get_column("is_american").to_numpy(),
                smooth_available=group.get_column("smooth_mid_available").to_numpy(),
                smooth_fit=fit,
                atm_volatility=atm_vol,
                epsilon_atm_volatility=epsilon_atm_vol,
                config=config,
            )
            bids = group.get_column(bid_column).to_numpy().astype(np.float64)
            asks = group.get_column(ask_column).to_numpy().astype(np.float64)
            pure_bid_column = (
                "pure_market_call_bid" if is_call else "pure_market_put_bid"
            )
            pure_ask_column = (
                "pure_market_call_ask" if is_call else "pure_market_put_ask"
            )
            pure_bids = group.get_column(pure_bid_column).to_numpy().astype(np.float64)
            pure_asks = group.get_column(pure_ask_column).to_numpy().astype(np.float64)
            bridge = float(record["discount_factor"]) * float(
                record["spot_normalized_implied_forward"]
            )
            for strike, bid, ask, pure_bid, pure_ask in zip(
                strikes[selected],
                bids[selected],
                asks[selected],
                pure_bids[selected],
                pure_asks[selected],
                strict=True,
            ):
                half_spread = 0.5 * (float(ask) - float(bid))
                if not math.isfinite(half_spread) or half_spread <= 0.0:
                    _fail("Selected public book has a non-positive spot-normalized half-spread")
                pure_half_spread = 0.5 * (float(pure_ask) - float(pure_bid))
                if not math.isclose(
                    half_spread / bridge,
                    pure_half_spread,
                    rel_tol=2.0e-5,
                    abs_tol=2.0e-8,
                ):
                    _fail("Selected public book fails the approved spread unit bridge")
                books.append(
                    {
                        "expiry_index": public_index_by_expiry_id[expiry_id],
                        "is_call": is_call,
                        "normalized_strike": float(strike),
                        "half_spread_spot": half_spread,
                    }
                )
    if not books:
        _fail("A populated internal day has no public daily-front books")
    return books


@version(
    "1.0",
    dependencies=[
        ProjectedPublicDay,
        PublicProjectionConfig,
        _atm_volatility,
        _book_payload,
        _one,
        _require_fitted_schema,
        _source_carry_curve,
        _state_payload,
        _state_records,
        _stored_dlv_squares,
        business_clock_policy,
        discrete_dlv_pricer,
        public_polars_schema,
        smooth_sanos_pricer,
    ],
)
def project_public_day(
    frame: pl.DataFrame,
    config: PublicProjectionConfig,
) -> ProjectedPublicDay:
    """Project one exact populated fitted day without serializing private inputs."""

    _require_fitted_schema(frame)
    if frame.is_empty():
        _fail("A typed-empty internal day has no public row")
    if not bool(frame.get_column("options_filter_eligible").any()):
        _fail(
            "A populated internal day with zero filter-selected rows has no public "
            "model state; it requires an explicit retained-date policy decision"
        )
    trading_date = _one(frame, "trading_date")
    if not isinstance(trading_date, date):
        _fail("trading_date is not a calendar date")
    spot = float(_one(frame, "spot_reference"))
    dlv_pricer = discrete_dlv_pricer(frame)
    smooth_pricer = smooth_sanos_pricer(frame)
    stored_dlv_squares = _stored_dlv_squares(frame, dlv_pricer)
    if smooth_pricer.smoothness != config.smoothness:
        _fail(
            "Internal smoothness differs from public projection configuration "
            f"({smooth_pricer.smoothness} != {config.smoothness})"
        )
    dlv_records = _state_records(frame, "discrete_dlv_input_expiry_index")
    smooth_records = _state_records(frame, "smooth_input_expiry_index")

    record_by_expiry_id: dict[int, dict[str, Any]] = {}
    for record in (*dlv_records, *smooth_records):
        expiry_id = int(record["expiry_id"])
        previous = record_by_expiry_id.get(expiry_id)
        if previous is not None and previous != record:
            _fail(f"Expiry {expiry_id} has inconsistent DLV/SANOS geometry")
        record_by_expiry_id[expiry_id] = record
    ordered_records = tuple(
        sorted(
            record_by_expiry_id.values(),
            key=lambda item: (float(item["time_to_expiry_years"]), int(item["expiry_id"])),
        )
    )
    if not ordered_records:
        _fail("Public state has no expiry geometry")
    public_index_by_expiry_id = {
        int(record["expiry_id"]): index for index, record in enumerate(ordered_records)
    }
    times = np.asarray(
        [float(record["time_to_expiry_years"]) for record in ordered_records],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
        _fail("Public expiry geometry must have positive strictly increasing model times")
    expiry_atm_vols = _atm_volatility(times, smooth_pricer)
    epsilon_atm_vol = float(
        _atm_volatility(
            config.nms_epsilon_business_dte / business_clock_policy()[0], smooth_pricer
        )
    )
    expiries: list[dict[str, Any]] = []
    for index, (record, atm_vol) in enumerate(
        zip(ordered_records, expiry_atm_vols, strict=True)
    ):
        expiry_date = record["expiry_date"]
        if not isinstance(expiry_date, date):
            _fail("expiry_date is not a calendar date")
        bridge = float(record["discount_factor"]) * float(
            record["spot_normalized_implied_forward"]
        )
        if not math.isfinite(bridge) or bridge <= 0.0:
            _fail("discounted_forward_over_spot must be positive and finite")
        expiries.append(
            {
                "expiry_index": index,
                "expiry_date": expiry_date,
                "business_dte": float(record["business_dte"]),
                "calendar_dte": int(record["calendar_dte"]),
                "is_am": bool(record["expiry_is_am"]),
                "time_to_expiry_years": float(record["time_to_expiry_years"]),
                "discounted_forward_over_spot": bridge,
                "nms_atm_annual_volatility": float(atm_vol),
            }
        )
    books = _book_payload(
        frame,
        expiry_records=ordered_records,
        public_index_by_expiry_id=public_index_by_expiry_id,
        surface_pricer=smooth_pricer,
        epsilon_atm_vol=epsilon_atm_vol,
        config=config,
    )
    row = {
        "trading_date": trading_date,
        "is_half_day": bool(_one(frame, "is_half_day")),
        "forward_adjusted_spot_return": None,
        "nms_epsilon_atm_annual_volatility": epsilon_atm_vol,
        "expiries": expiries,
        "books": books,
        "smooth_sanos": _state_payload(
            smooth_pricer,
            smooth_records,
            public_index_by_expiry_id,
            smooth=True,
        ),
        "discrete_dlv": _state_payload(
            dlv_pricer,
            dlv_records,
            public_index_by_expiry_id,
            smooth=False,
            dlv_squares=stored_dlv_squares,
        ),
    }
    # Force the recursive Polars cast here so a projection cannot accumulate an
    # untyped Python field which publication would later pass through.
    pl.DataFrame([row], schema=public_polars_schema(), strict=True)
    return ProjectedPublicDay(
        row=row,
        carry=_source_carry_curve(trading_date, spot, frame),
        book_count=len(books),
        expiry_count=len(expiries),
    )


@version("1.0")
def _log_curve_value(
    business_dtes: tuple[float, ...],
    values: tuple[float, ...],
    horizon: float,
    label: str,
) -> float:
    nodes = np.asarray(business_dtes, dtype=np.float64)
    curve = np.asarray(values, dtype=np.float64)
    if (
        len(nodes) != len(curve)
        or len(nodes) < 2
        or nodes[0] != 0.0
        or curve[0] != 1.0
        or np.any(np.diff(nodes) <= 0.0)
        or np.any(~np.isfinite(curve))
        or np.any(curve <= 0.0)
    ):
        _fail(f"{label} curve is invalid")
    if horizon < 0.0 or horizon > nodes[-1]:
        _fail(f"{label} curve does not span return horizon {horizon}")
    result = float(np.exp(np.interp(horizon, nodes, np.log(curve))))
    if not math.isfinite(result) or result <= 0.0:
        _fail(f"{label} curve interpolation is invalid")
    return result


@version(
    "1.0",
    dependencies=[
        ProjectedPublicDay,
        _log_curve_value,
        count_business_days_between,
        public_polars_schema,
    ],
)
def assemble_public_frame(
    days: tuple[ProjectedPublicDay, ...],
) -> tuple[pl.DataFrame, tuple[dict[str, Any], ...]]:
    """Add sequential approved returns and assemble the one-frame time series."""

    if not days:
        _fail("A public Dataset requires at least one populated day")
    ordered_dates = tuple(day.carry.trading_date for day in days)
    if ordered_dates != tuple(sorted(set(ordered_dates))):
        _fail("Projected public dates must be unique and strictly increasing")
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, day in enumerate(days):
        row = dict(day.row)
        if index == 0:
            row["forward_adjusted_spot_return"] = None
        else:
            previous = days[index - 1]
            sessions = int(
                count_business_days_between(
                    previous.carry.trading_date,
                    day.carry.trading_date,
                )
            )
            if sessions <= 0:
                _fail("Consecutive public dates have no positive session horizon")
            carry = _log_curve_value(
                previous.carry.business_dtes,
                previous.carry.forward_over_spot,
                float(sessions),
                "forward-over-spot",
            )
            discount = _log_curve_value(
                previous.carry.business_dtes,
                previous.carry.discount_factors,
                float(sessions),
                "discount-factor",
            )
            value = (day.carry.spot / previous.carry.spot) / carry - 1.0
            if not math.isfinite(value):
                _fail("forward_adjusted_spot_return is non-finite")
            row["forward_adjusted_spot_return"] = value
            evidence.append(
                {
                    "trading_date": day.carry.trading_date,
                    "previous_trading_date": previous.carry.trading_date,
                    "business_sessions": sessions,
                    "interpolated_discount_factor": discount,
                }
            )
        rows.append(row)
    return (
        pl.DataFrame(rows, schema=public_polars_schema(), strict=True),
        tuple(evidence),
    )


__all__ = [
    "CarryCurve",
    "ProjectedPublicDay",
    "PublicProjectionError",
    "assemble_public_frame",
    "project_public_day",
]
