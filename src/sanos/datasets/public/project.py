"""Pure projection from one fitted internal day to the public daily contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import date
from typing import Any, NoReturn

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
from sanos.widb.daily import DailyDataError, _black_implied
from sanos.widb.surface import (
    FITTED_SCHEMA,
    FITTED_SCHEMA_VERSION,
    initial_no_arb_pricer,
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


def _fail(message: str) -> NoReturn:
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


@version("3.0", dependencies=[SanosPricer])
def _state_payload(
    pricer: SanosPricer,
    source_records: tuple[dict[str, Any], ...],
    public_index_by_expiry_id: dict[int, int],
    *,
    smooth: bool,
) -> dict[str, Any]:
    if len(source_records) != len(pricer.expiries):
        _fail("Model state and source expiry geometry have different lengths")
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
        model_expiries.append(item)
    result: dict[str, Any] = {"model_expiries": model_expiries}
    if smooth:
        if pricer.smoothness is None:
            _fail("Smooth SANOS state has no smoothness")
        result["smoothness"] = float(pricer.smoothness)
        return {"smoothness": result["smoothness"], "model_expiries": model_expiries}
    return result


@version("1.0", dependencies=[public_polars_schema])
def _flat_row(
    trading_date: date,
    record_kind: str,
    **values: Any,
) -> dict[str, Any]:
    """Create one exact scalar public row with all inapplicable fields null."""

    row = dict.fromkeys(public_polars_schema().names())
    row.update({"trading_date": trading_date, "record_kind": record_kind})
    unknown = set(values) - set(row)
    if unknown:
        _fail(f"Flat public row contains unknown fields: {sorted(unknown)}")
    row.update(values)
    return row


@version("2.0", dependencies=[_flat_row, public_polars_schema])
def _flatten_public_day(row: Mapping[str, Any]) -> pl.DataFrame:
    """Encode one private nested semantic day as canonical scalar public rows."""

    required = {
        "trading_date",
        "is_half_day",
        "driftless_spot_return",
        "nms_epsilon_atm_annual_volatility",
        "sanos_market_percent_books_outside",
        "sanos_market_conditional_mean_excess_halfspreads",
        "sanos_market_conditional_median_excess_halfspreads",
        "dlv_market_percent_books_outside",
        "dlv_market_conditional_mean_excess_halfspreads",
        "dlv_market_conditional_median_excess_halfspreads",
        "expiries",
        "books",
        "smooth_sanos",
        "dlv",
    }
    if set(row) != required or not isinstance(row["trading_date"], date):
        _fail("Private public-day fields changed")
    trading_date = row["trading_date"]
    smooth = row["smooth_sanos"]
    dlv = row["dlv"]
    if not isinstance(smooth, Mapping) or not isinstance(dlv, Mapping):
        _fail("Private public model state is malformed")
    expiries = row["expiries"]
    books = row["books"]
    smooth_expiries = smooth.get("model_expiries")
    dlv_expiries = dlv.get("model_expiries")
    if not isinstance(expiries, list):
        _fail("Private public-day expiries must be a list")
    if not isinstance(books, list):
        _fail("Private public-day books must be a list")
    if not isinstance(smooth_expiries, list):
        _fail("Private smooth SANOS expiries must be a list")
    if not isinstance(dlv_expiries, list):
        _fail("Private discrete DLV expiries must be a list")
    smoothness = smooth.get("smoothness")
    if smoothness is None:
        _fail("Private smooth SANOS state has no smoothness")
    smooth_by_expiry = {
        int(item["expiry_index"]): item for item in smooth_expiries
    }
    dlv_by_expiry = {int(item["expiry_index"]): item for item in dlv_expiries}
    flat: list[dict[str, Any]] = [
        _flat_row(
            trading_date,
            "day",
            is_half_day=row["is_half_day"],
            driftless_spot_return=row["driftless_spot_return"],
            nms_epsilon_atm_annual_volatility=row[
                "nms_epsilon_atm_annual_volatility"
            ],
            smoothness=smoothness,
            sanos_market_percent_books_outside=row[
                "sanos_market_percent_books_outside"
            ],
            sanos_market_conditional_mean_excess_halfspreads=row[
                "sanos_market_conditional_mean_excess_halfspreads"
            ],
            sanos_market_conditional_median_excess_halfspreads=row[
                "sanos_market_conditional_median_excess_halfspreads"
            ],
            dlv_market_percent_books_outside=row[
                "dlv_market_percent_books_outside"
            ],
            dlv_market_conditional_mean_excess_halfspreads=row[
                "dlv_market_conditional_mean_excess_halfspreads"
            ],
            dlv_market_conditional_median_excess_halfspreads=row[
                "dlv_market_conditional_median_excess_halfspreads"
            ],
        )
    ]
    for expiry in expiries:
        if not isinstance(expiry, Mapping):
            _fail("Private public expiry is malformed")
        expiry_index = int(expiry["expiry_index"])
        flat.append(
            _flat_row(
                trading_date,
                "expiry",
                expiry_index=expiry_index,
                expiry_date=expiry["expiry_date"],
                business_dte=expiry["business_dte"],
                calendar_dte=expiry["calendar_dte"],
                is_am=expiry["is_am"],
                time_to_expiry_years=expiry["time_to_expiry_years"],
                nms_atm_annual_volatility=expiry[
                    "nms_atm_annual_volatility"
                ],
            )
        )
        expiry_books = [
            item
            for item in books
            if isinstance(item, Mapping)
            and int(item["expiry_index"]) == expiry_index
        ]
        for book_index, book in enumerate(expiry_books):
            flat.append(
                _flat_row(
                    trading_date,
                    "book",
                    expiry_index=expiry_index,
                    book_index=book_index,
                    is_call=book["is_call"],
                    normalized_strike=book["normalized_strike"],
                    half_spread=book["half_spread"],
                    sanos_fit_annual_iv=book["sanos_fit_annual_iv"],
                    sanos_synthetic_bid_annual_iv=book[
                        "sanos_synthetic_bid_annual_iv"
                    ],
                    sanos_synthetic_ask_annual_iv=book[
                        "sanos_synthetic_ask_annual_iv"
                    ],
                    dlv_fit_annual_iv=book["dlv_fit_annual_iv"],
                    dlv_synthetic_bid_annual_iv=book[
                        "dlv_synthetic_bid_annual_iv"
                    ],
                    dlv_synthetic_ask_annual_iv=book[
                        "dlv_synthetic_ask_annual_iv"
                    ],
                )
            )
        for kind, state, variance_name in (
            (
                "smooth_node",
                smooth_by_expiry.get(expiry_index),
                "kernel_input_total_variance",
            ),
            ("dlv_node", dlv_by_expiry.get(expiry_index), "input_total_variance"),
        ):
            if state is None:
                continue
            if not isinstance(state, Mapping):
                _fail(f"Private {kind} state is malformed at expiry {expiry_index}")
            support = state.get("normalized_support")
            density = state.get("density_mass")
            if not isinstance(support, list) or not isinstance(density, list):
                _fail(f"Private {kind} vectors are malformed")
            if len(support) != len(density):
                _fail(f"Private {kind} vectors have different lengths")
            for node_index, (strike, mass) in enumerate(
                zip(support, density, strict=True)
            ):
                values = {
                    "expiry_index": expiry_index,
                    "node_index": node_index,
                    "time_increment_years": state["time_increment_years"],
                    variance_name: state[variance_name],
                    "normalized_support": strike,
                    "density_mass": mass,
                }
                flat.append(_flat_row(trading_date, kind, **values))
    return pl.DataFrame(flat, schema=public_polars_schema(), strict=True)


@version("1.0")
def _require_flat_population(
    frame: pl.DataFrame,
    kind: str,
    required: set[str],
    *,
    nullable: Set[str] = frozenset(),
) -> None:
    """Enforce the exact populated/null column matrix for one row kind."""

    common = {"trading_date", "record_kind"}
    for column in frame.columns:
        if column in common:
            if frame.get_column(column).null_count():
                _fail(f"Flat {kind} rows contain a null {column}")
        elif column in required:
            if column not in nullable and frame.get_column(column).null_count():
                _fail(f"Flat {kind} rows contain a null {column}")
        elif frame.get_column(column).null_count() != frame.height:
            _fail(f"Flat {kind} rows populate inapplicable field {column}")


@version("1.0")
def _one_repeated(frame: pl.DataFrame, column: str, label: str) -> Any:
    values = frame.get_column(column).unique().to_list()
    if len(values) != 1 or values[0] is None:
        _fail(f"{label} has inconsistent {column}")
    return values[0]


@version(
    "2.0",
    dependencies=[
        _flatten_public_day,
        _one_repeated,
        _require_flat_population,
        public_polars_schema,
    ],
)
def _inflate_public_day(frame: pl.DataFrame) -> dict[str, Any]:
    """Decode and authenticate one canonical scalar day for numerical validation."""

    if frame.schema != public_polars_schema() or frame.is_empty():
        _fail("Flat public day does not have the exact scalar schema")
    dates = frame.get_column("trading_date").unique().to_list()
    if len(dates) != 1 or not isinstance(dates[0], date):
        _fail("Flat public day must contain exactly one trading date")
    trading_date = dates[0]
    observed_kinds = set(frame.get_column("record_kind").to_list())
    expected_kinds = {"day", "expiry", "book", "smooth_node", "dlv_node"}
    if observed_kinds != expected_kinds:
        _fail("Flat public day record kinds changed")

    day = frame.filter(pl.col("record_kind") == "day")
    if day.height != 1:
        _fail("Flat public day requires exactly one day row")
    _require_flat_population(
        day,
        "day",
        {
            "is_half_day",
            "driftless_spot_return",
            "nms_epsilon_atm_annual_volatility",
            "smoothness",
            "sanos_market_percent_books_outside",
            "sanos_market_conditional_mean_excess_halfspreads",
            "sanos_market_conditional_median_excess_halfspreads",
            "dlv_market_percent_books_outside",
            "dlv_market_conditional_mean_excess_halfspreads",
            "dlv_market_conditional_median_excess_halfspreads",
        },
        nullable={"driftless_spot_return"},
    )

    expiry_rows = frame.filter(pl.col("record_kind") == "expiry")
    _require_flat_population(
        expiry_rows,
        "expiry",
        {
            "expiry_index",
            "expiry_date",
            "business_dte",
            "calendar_dte",
            "is_am",
            "time_to_expiry_years",
            "nms_atm_annual_volatility",
        },
    )
    expiry_indices = expiry_rows.get_column("expiry_index").to_list()
    if expiry_indices != list(range(len(expiry_indices))):
        _fail("Flat public expiry indices must be contiguous from zero")
    expiries = [
        {
            "expiry_index": int(item["expiry_index"]),
            "expiry_date": item["expiry_date"],
            "business_dte": item["business_dte"],
            "calendar_dte": item["calendar_dte"],
            "is_am": item["is_am"],
            "time_to_expiry_years": item["time_to_expiry_years"],
            "nms_atm_annual_volatility": item["nms_atm_annual_volatility"],
        }
        for item in expiry_rows.iter_rows(named=True)
    ]

    book_rows = frame.filter(pl.col("record_kind") == "book")
    _require_flat_population(
        book_rows,
        "book",
        {
            "expiry_index",
            "is_call",
            "normalized_strike",
            "half_spread",
            "sanos_fit_annual_iv",
            "sanos_synthetic_bid_annual_iv",
            "sanos_synthetic_ask_annual_iv",
            "dlv_fit_annual_iv",
            "dlv_synthetic_bid_annual_iv",
            "dlv_synthetic_ask_annual_iv",
            "book_index",
        },
        nullable={
            "sanos_fit_annual_iv",
            "sanos_synthetic_bid_annual_iv",
            "sanos_synthetic_ask_annual_iv",
            "dlv_fit_annual_iv",
            "dlv_synthetic_bid_annual_iv",
            "dlv_synthetic_ask_annual_iv",
        },
    )
    books: list[dict[str, Any]] = []
    for expiry_index in expiry_indices:
        group = book_rows.filter(pl.col("expiry_index") == expiry_index)
        if group.get_column("book_index").to_list() != list(range(group.height)):
            _fail("Flat public book indices must be contiguous within expiry")
        books.extend(
            {
                "expiry_index": int(item["expiry_index"]),
                "is_call": item["is_call"],
                "normalized_strike": item["normalized_strike"],
                "half_spread": item["half_spread"],
                "sanos_fit_annual_iv": item["sanos_fit_annual_iv"],
                "sanos_synthetic_bid_annual_iv": item[
                    "sanos_synthetic_bid_annual_iv"
                ],
                "sanos_synthetic_ask_annual_iv": item[
                    "sanos_synthetic_ask_annual_iv"
                ],
                "dlv_fit_annual_iv": item["dlv_fit_annual_iv"],
                "dlv_synthetic_bid_annual_iv": item[
                    "dlv_synthetic_bid_annual_iv"
                ],
                "dlv_synthetic_ask_annual_iv": item[
                    "dlv_synthetic_ask_annual_iv"
                ],
            }
            for item in group.iter_rows(named=True)
        )

    states: dict[str, list[dict[str, Any]]] = {
        "smooth_node": [],
        "dlv_node": [],
    }
    for kind, variance_name in (
        ("smooth_node", "kernel_input_total_variance"),
        ("dlv_node", "input_total_variance"),
    ):
        rows = frame.filter(pl.col("record_kind") == kind)
        required = {
            "expiry_index",
            "node_index",
            "time_increment_years",
            variance_name,
            "normalized_support",
            "density_mass",
        }
        _require_flat_population(rows, kind, required)
        model_expiry_indices = rows.get_column("expiry_index").unique(
            maintain_order=True
        ).to_list()
        if not model_expiry_indices or any(
            right <= left
            for left, right in zip(
                model_expiry_indices, model_expiry_indices[1:], strict=False
            )
        ):
            _fail(f"Flat {kind} expiry indices must be non-empty and increasing")
        if any(index not in expiry_indices for index in model_expiry_indices):
            _fail(f"Flat {kind} expiry index is outside union geometry")
        for expiry_index in model_expiry_indices:
            group = rows.filter(pl.col("expiry_index") == expiry_index)
            if group.is_empty() or group.get_column("node_index").to_list() != list(
                range(group.height)
            ):
                _fail(f"Flat {kind} node indices must be non-empty and contiguous")
            item = {
                "expiry_index": int(expiry_index),
                "time_increment_years": _one_repeated(
                    group, "time_increment_years", kind
                ),
                variance_name: _one_repeated(group, variance_name, kind),
                "normalized_support": group.get_column("normalized_support").to_list(),
                "density_mass": group.get_column("density_mass").to_list(),
            }
            states[kind].append(item)

    nested = {
        "trading_date": trading_date,
        "is_half_day": day.item(0, "is_half_day"),
        "driftless_spot_return": day.item(0, "driftless_spot_return"),
        "nms_epsilon_atm_annual_volatility": day.item(
            0, "nms_epsilon_atm_annual_volatility"
        ),
        "sanos_market_percent_books_outside": day.item(
            0, "sanos_market_percent_books_outside"
        ),
        "sanos_market_conditional_mean_excess_halfspreads": day.item(
            0, "sanos_market_conditional_mean_excess_halfspreads"
        ),
        "sanos_market_conditional_median_excess_halfspreads": day.item(
            0, "sanos_market_conditional_median_excess_halfspreads"
        ),
        "dlv_market_percent_books_outside": day.item(
            0, "dlv_market_percent_books_outside"
        ),
        "dlv_market_conditional_mean_excess_halfspreads": day.item(
            0, "dlv_market_conditional_mean_excess_halfspreads"
        ),
        "dlv_market_conditional_median_excess_halfspreads": day.item(
            0, "dlv_market_conditional_median_excess_halfspreads"
        ),
        "expiries": expiries,
        "books": books,
        "smooth_sanos": {
            "smoothness": day.item(0, "smoothness"),
            "model_expiries": states["smooth_node"],
        },
        "dlv": {"model_expiries": states["dlv_node"]},
    }
    if not _flatten_public_day(nested).equals(frame):
        _fail("Flat public rows are not in exact canonical order or form")
    return nested


@version("1.0")
def _excluded_public_business_dte(
    business_dte: float,
    config: PublicProjectionConfig,
) -> bool:
    """Return whether one expiry is outside the configured public book population."""

    return (
        business_dte < 0.0
        or business_dte > config.maximum_business_dte
        or any(
            abs(business_dte - value) <= config.exclusion_tolerance
            for value in config.excluded_business_dtes
        )
    )


@version("1.1", dependencies=[_excluded_public_business_dte, normalized_strike_policy])
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

    if _excluded_public_business_dte(business_dte, config):
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


@version("1.0", dependencies=[_black_implied])
def _public_implied_volatility(
    strike: float,
    call_price: float,
    time_years: float,
    *,
    required: bool,
) -> float | None:
    """Invert an identifiable normalized call price without clipping it."""

    intrinsic = max(1.0 - strike, 0.0)
    if (
        not math.isfinite(call_price)
        or not math.isfinite(strike)
        or not math.isfinite(time_years)
        or time_years <= 0.0
        or call_price <= intrinsic
        or call_price >= 1.0
    ):
        if required:
            _fail("Public model fit has no identifiable finite Black volatility")
        return None
    try:
        value = float(
            _black_implied(
                k=np.asarray([strike], dtype=np.float64),
                prices=np.asarray([call_price], dtype=np.float64),
                sqrtT=math.sqrt(time_years),
                eps=1.0e-8,
            )[0]
        )
    except (ArithmeticError, RuntimeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value <= 0.0:
        if required:
            _fail("Public model fit has no identifiable finite Black volatility")
        return None
    return value


@version("1.0", dependencies=[_public_implied_volatility])
def _model_iv_fields(
    label: str,
    strike: float,
    call_price: float,
    half_spread: float,
    time_years: float,
) -> dict[str, float | None]:
    """Return one model's exact fit and synthetic-bound IV fields."""

    if time_years <= 0.0:
        return {
            f"{label}_fit_annual_iv": None,
            f"{label}_synthetic_bid_annual_iv": None,
            f"{label}_synthetic_ask_annual_iv": None,
        }
    return {
        f"{label}_fit_annual_iv": _public_implied_volatility(
            strike, call_price, time_years, required=True
        ),
        f"{label}_synthetic_bid_annual_iv": _public_implied_volatility(
            strike, call_price - half_spread, time_years, required=False
        ),
        f"{label}_synthetic_ask_annual_iv": _public_implied_volatility(
            strike, call_price + half_spread, time_years, required=False
        ),
    }


@version("1.0")
def _market_quality(excesses: list[float], label: str) -> dict[str, float]:
    if not excesses or any(not math.isfinite(value) or value < 0.0 for value in excesses):
        _fail(f"{label} market-quality inputs are invalid")
    positive = np.asarray([value for value in excesses if value > 0.0], dtype=np.float64)
    prefix = f"{label}_market_"
    return {
        prefix + "percent_books_outside": 100.0 * len(positive) / len(excesses),
        prefix + "conditional_mean_excess_halfspreads": (
            float(np.mean(positive)) if len(positive) else 0.0
        ),
        prefix + "conditional_median_excess_halfspreads": (
            float(np.median(positive)) if len(positive) else 0.0
        ),
    }


@version(
    "3.1",
    dependencies=[
        _atm_volatility,
        _daily_front_mask,
        _excluded_public_business_dte,
        _market_quality,
        _model_iv_fields,
        normalized_strike_policy,
    ],
)
def _book_payload(
    frame: pl.DataFrame,
    *,
    expiry_records: tuple[dict[str, Any], ...],
    public_index_by_expiry_id: dict[int, int],
    surface_pricer: SanosPricer,
    dlv_pricer: SanosPricer,
    epsilon_atm_vol: float,
    config: PublicProjectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    books: list[dict[str, Any]] = []
    excesses: dict[str, list[float]] = {"sanos": [], "dlv": []}
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
        # Book exclusion never removes this expiry from the complete published
        # state/geometry.  In particular, do not try to infer an ATM IV at T=0.
        if _excluded_public_business_dte(business_dte, config):
            continue
        atm_vol = float(_atm_volatility(time_years, surface_pricer))
        strikes = group.get_column("normalized_strike").to_numpy()
        for is_call, active_column, fit_column in (
            (
                True,
                "call_active",
                "smooth_call_mid",
            ),
            (
                False,
                "put_active",
                "smooth_put_mid",
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
            pure_bid_column = (
                "pure_market_call_bid" if is_call else "pure_market_put_bid"
            )
            pure_ask_column = (
                "pure_market_call_ask" if is_call else "pure_market_put_ask"
            )
            pure_bids = group.get_column(pure_bid_column).to_numpy().astype(np.float64)
            pure_asks = group.get_column(pure_ask_column).to_numpy().astype(np.float64)
            for strike, pure_bid, pure_ask in zip(
                strikes[selected],
                pure_bids[selected],
                pure_asks[selected],
                strict=True,
            ):
                half_spread = 0.5 * (float(pure_ask) - float(pure_bid))
                if not math.isfinite(half_spread) or half_spread <= 0.0:
                    _fail("Selected public book has a non-positive driftless half-spread")
                strike_value = float(strike)
                model_fields: dict[str, float | None] = {}
                for label, pricer in (("sanos", surface_pricer), ("dlv", dlv_pricer)):
                    call_price = float(pricer(time_years, np.asarray([strike_value]))[0])
                    put_price = call_price - 1.0 + strike_value
                    model_price = call_price if is_call else put_price
                    excesses[label].append(
                        (
                            max(model_price - float(pure_ask), 0.0)
                            + max(float(pure_bid) - model_price, 0.0)
                        )
                        / half_spread
                    )
                    model_fields.update(
                        _model_iv_fields(
                            label,
                            strike_value,
                            call_price,
                            half_spread,
                            time_years,
                        )
                    )
                books.append(
                    {
                        "expiry_index": public_index_by_expiry_id[expiry_id],
                        "is_call": is_call,
                        "normalized_strike": strike_value,
                        "half_spread": half_spread,
                        **model_fields,
                    }
                )
    if not books:
        _fail("A populated internal day has no public daily-front books")
    return books, {
        **_market_quality(excesses["sanos"], "sanos"),
        **_market_quality(excesses["dlv"], "dlv"),
    }


@version(
    "4.0",
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
        business_clock_policy,
        initial_no_arb_pricer,
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
    dlv_pricer = initial_no_arb_pricer(frame)
    smooth_pricer = smooth_sanos_pricer(frame)
    if smooth_pricer.smoothness != config.smoothness:
        _fail(
            "Internal smoothness differs from public projection configuration "
            f"({smooth_pricer.smoothness} != {config.smoothness})"
        )
    dlv_records = _state_records(frame, "initial_no_arb_input_expiry_index")
    smooth_records = _state_records(frame, "smooth_input_expiry_index")

    dlv_expiry_ids = {int(record["expiry_id"]) for record in dlv_records}
    smooth_expiry_ids = {int(record["expiry_id"]) for record in smooth_records}
    if not smooth_expiry_ids.issubset(dlv_expiry_ids):
        _fail("Smooth SANOS expiries must be a subset of the initial DLV chain")

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
        expiries.append(
            {
                "expiry_index": index,
                "expiry_date": expiry_date,
                "business_dte": float(record["business_dte"]),
                "calendar_dte": int(record["calendar_dte"]),
                "is_am": bool(record["expiry_is_am"]),
                "time_to_expiry_years": float(record["time_to_expiry_years"]),
                "nms_atm_annual_volatility": float(atm_vol),
            }
        )
    books, market_quality = _book_payload(
        frame,
        expiry_records=ordered_records,
        public_index_by_expiry_id=public_index_by_expiry_id,
        surface_pricer=smooth_pricer,
        dlv_pricer=dlv_pricer,
        epsilon_atm_vol=epsilon_atm_vol,
        config=config,
    )
    row = {
        "trading_date": trading_date,
        "is_half_day": bool(_one(frame, "is_half_day")),
        "driftless_spot_return": None,
        "nms_epsilon_atm_annual_volatility": epsilon_atm_vol,
        **market_quality,
        "expiries": expiries,
        "books": books,
        "smooth_sanos": _state_payload(
            smooth_pricer,
            smooth_records,
            public_index_by_expiry_id,
            smooth=True,
        ),
        "dlv": _state_payload(
            dlv_pricer,
            dlv_records,
            public_index_by_expiry_id,
            smooth=False,
        ),
    }
    # Force the public scalar cast here so projection cannot accumulate an
    # untyped private field which publication would later pass through.
    _flatten_public_day(row)
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
        _flatten_public_day,
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
    frames: list[pl.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for index, day in enumerate(days):
        row = dict(day.row)
        if index == 0:
            row["driftless_spot_return"] = None
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
            value = discount * (day.carry.spot / previous.carry.spot) / carry - 1.0
            if not math.isfinite(value):
                _fail("driftless_spot_return is non-finite")
            row["driftless_spot_return"] = value
            evidence.append(
                {
                    "trading_date": day.carry.trading_date,
                    "previous_trading_date": previous.carry.trading_date,
                    "business_sessions": sessions,
                    "interpolated_discount_factor": discount,
                }
            )
        frames.append(_flatten_public_day(row))
    return (
        pl.concat(frames, how="vertical", rechunk=True),
        tuple(evidence),
    )


__all__ = [
    "CarryCurve",
    "ProjectedPublicDay",
    "PublicProjectionError",
    "assemble_public_frame",
    "project_public_day",
]
