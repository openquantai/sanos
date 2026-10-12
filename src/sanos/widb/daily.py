"""Reference-compatible daily WIDB normalization and enrichment.

The public stage boundaries are Polars frames. Numerical vector operations use
NumPy and the same cdxcore Black-Scholes implementation as the pinned source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.version import version

from sanos.config import canonical_json, canonical_sha256
from sanos.widb.bad import (
    bad_data_policy,
    bad_day_reason,
    bad_option_symbols,
    irregular_expiries,
    require_supported_date,
)
from sanos.widb.calendar import (
    DAYS_PER_YEAR,
    NMS5,
    business_clock_policy,
    count_business_days_between,
    is_half_day,
    normalized_strike_policy,
)
from sanos.widb.config import (
    ClockConfig,
    DailyWidbConfig,
    ExpiryEnrichmentConfig,
    RawDayConfig,
)
from sanos.wrds import RawTable
from sanos.wrds.source import raw_table_spec

PAIRED_SCHEMA_VERSION = "widb-daily-paired-v1"
ENRICHED_SCHEMA_VERSION = "widb-daily-enriched-v1"


@version("cdxcore-bs-kernel-2026-10-v1")
def _black_implied(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
    """Version-owned adapter around the cdxcore Black implied-volatility kernel."""

    return np.asarray(bs.implied(*args, **kwargs))


@version("cdxcore-bs-kernel-2026-10-v1")
def _black_values(*args: Any, **kwargs: Any) -> Any:
    """Version-owned adapter around the cdxcore Black value/Greek kernel."""

    return bs(*args, **kwargs)


@version("widb-iv-and-greek-epsilon-1e-8-v1")
def _numerical_tolerances() -> tuple[float, float]:
    """Return version-owned numerical tolerances that can alter daily content."""

    return 1.0e-8, 1.0e-8


_IMPLIED_VOL_CALC_EPS, _GREEK_CALC_EPS = _numerical_tolerances()


class DailyDataError(RuntimeError):
    """Raised when exact raw evidence cannot form a valid daily frame."""


@version("1.0")
@dataclass(frozen=True, slots=True)
class DailyRawInputs:
    """The six immutable raw partitions consumed by one normalized day."""

    security: pl.DataFrame
    options: pl.DataFrame
    spot: pl.DataFrame
    previous_spot: pl.DataFrame
    forwards: pl.DataFrame
    zero_curve: pl.DataFrame

    def __post_init__(self) -> None:
        for name in (
            "security",
            "options",
            "spot",
            "previous_spot",
            "forwards",
            "zero_curve",
        ):
            if not isinstance(getattr(self, name), pl.DataFrame):
                raise TypeError(f"{name} must be a Polars DataFrame")


@version("1.0")
@dataclass(frozen=True, slots=True)
class DailyProvenance:
    """Non-secret source identity repeated in a self-describing daily frame."""

    raw_options_artifact_id: str | None = None

    def __post_init__(self) -> None:
        if self.raw_options_artifact_id is not None and (
            len(self.raw_options_artifact_id) != 64
            or any(
                character not in "0123456789abcdef" for character in self.raw_options_artifact_id
            )
        ):
            raise ValueError("raw_options_artifact_id must be a lowercase SHA-256 identity")


@version("widb-daily-paired-v1")
def _paired_schema() -> dict[str, pl.DataType]:
    """Return the version-owned paired-day schema."""

    return {
        "ticker": pl.String,
        "trading_date": pl.Date,
        "spot": pl.Float32,
        "previous_spot": pl.Float32,
        "spot_return": pl.Float32,
        "is_half_day": pl.Boolean,
        "source_secid": pl.Int64,
        "stage_version": pl.String,
        "configuration_id": pl.String,
        "raw_options_artifact_id": pl.String,
        "normalization_diagnostics_json": pl.String,
        "call_symbol": pl.String,
        "put_symbol": pl.String,
        "call_option_id": pl.Int32,
        "put_option_id": pl.Int32,
        "expiry_id": pl.Int32,
        "is_american": pl.Boolean,
        "expiry_is_am": pl.Boolean,
        "strike": pl.Float32,
        "vendor_forward": pl.Float32,
        "discount_factor": pl.Float32,
        "expiry_date": pl.Date,
        "business_dte": pl.Float32,
        "calendar_dte": pl.Int32,
        "time_to_expiry_years": pl.Float32,
        "market_call_ask": pl.Float32,
        "market_call_bid": pl.Float32,
        "market_call_log_volume": pl.Float32,
        "market_call_log_open_interest": pl.Float32,
        "market_put_ask": pl.Float32,
        "market_put_bid": pl.Float32,
        "market_put_log_volume": pl.Float32,
        "market_put_log_open_interest": pl.Float32,
    }


@version("widb-daily-enriched-v1")
def _enriched_only_schema() -> dict[str, pl.DataType]:
    """Return the version-owned enriched-only schema."""

    return {
        "implied_forward": pl.Float32,
        "normalized_moneyness": pl.Float32,
        "atm_normalized_moneyness": pl.Float32,
        "synthetic_call_ask": pl.Float32,
        "synthetic_call_bid": pl.Float32,
        "ask_implied_volatility": pl.Float32,
        "bid_implied_volatility": pl.Float32,
        "combined_log_volume": pl.Float32,
        "combined_log_open_interest": pl.Float32,
        "normalized_strike": pl.Float32,
        "pure_ask_price": pl.Float32,
        "pure_bid_price": pl.Float32,
        "pure_ask_delta": pl.Float32,
        "pure_bid_delta": pl.Float32,
        "pure_ask_gamma": pl.Float32,
        "pure_bid_gamma": pl.Float32,
        "pure_ask_vega": pl.Float32,
        "pure_bid_vega": pl.Float32,
        "pure_ask_theta": pl.Float32,
        "pure_bid_theta": pl.Float32,
        "pure_ask_dk": pl.Float32,
        "pure_bid_dk": pl.Float32,
        "enrichment_diagnostics_json": pl.String,
    }


PAIRED_SCHEMA = _paired_schema()
ENRICHED_ONLY_SCHEMA = _enriched_only_schema()
ENRICHED_SCHEMA = PAIRED_SCHEMA | ENRICHED_ONLY_SCHEMA


@version("1.0")
def _empty_daily_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


@version("1.0", dependencies=[raw_table_spec])
def _require_raw_schema(frame: pl.DataFrame, table: RawTable) -> pl.DataFrame:
    expected = raw_table_spec(table).columns
    actual = tuple(frame.columns)
    if set(actual) != set(expected):
        raise DailyDataError(
            f"Raw {table.value} schema mismatch: missing="
            f"{sorted(set(expected) - set(actual))}, unexpected="
            f"{sorted(set(actual) - set(expected))}"
        )
    return frame.select(expected)


@version("1.0", dependencies=[_require_raw_schema])
def interpolate_zero_curve(
    zero_curve: pl.DataFrame,
    calendar_days: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Interpolate continuous ACT/365 discount factors from rates in percent."""

    curve = _require_raw_schema(zero_curve, RawTable.ZERO_CURVE)
    if curve.is_empty():
        raise DailyDataError("The daily zero curve is empty")
    days = curve.get_column("days").cast(pl.Float64).to_numpy()
    rates_percent = curve.get_column("rate").cast(pl.Float64).to_numpy()
    requested = np.asarray(calendar_days, dtype=np.float64)
    if not np.all(np.isfinite(days)) or not np.all(np.isfinite(rates_percent)):
        raise DailyDataError("The daily zero curve contains non-finite values")
    rates = np.interp(requested, days, rates_percent)
    discount = np.exp(-rates * requested / 36500.0)
    if not np.all(np.isfinite(discount)) or np.min(discount) <= 0.0:
        raise DailyDataError("Interpolated discount factors must be finite and positive")
    return discount


@version("1.0")
def _one_positive_value(
    frame: pl.DataFrame,
    column: str,
    *,
    label: str,
) -> float:
    if frame.height == 0:
        raise DailyDataError(f"No {label} row is available")
    values = frame.get_column(column).cast(pl.Float64).to_numpy()
    value = float(values[0])  # Reference semantics select the first returned close.
    if not math.isfinite(value) or value <= 0.0:
        raise DailyDataError(f"{label} must be finite and positive")
    return value


@version("1.0")
def _series(frame: pl.DataFrame, name: str, dtype: Any = np.float64) -> np.ndarray[Any, Any]:
    return frame.get_column(name).to_numpy().astype(dtype, copy=False)


@version("1.0")
def _validate_constant_fields(frame: pl.DataFrame) -> None:
    """Prove the per-day and per-expiry constants required downstream."""

    for name in (
        "ticker",
        "trading_date",
        "spot",
        "previous_spot",
        "spot_return",
        "is_half_day",
        "source_secid",
        "stage_version",
        "configuration_id",
        "normalization_diagnostics_json",
    ):
        if frame.get_column(name).n_unique() != 1:
            raise DailyDataError(f"Daily field {name!r} is not constant")
    expiry_fields = [
        "expiry_date",
        "expiry_is_am",
        "business_dte",
        "calendar_dte",
        "time_to_expiry_years",
        "discount_factor",
        "vendor_forward",
    ]
    if "implied_forward" in frame.columns:
        expiry_fields.append("implied_forward")
    for name in expiry_fields:
        varying = (
            frame.group_by("expiry_id")
            .agg(pl.col(name).n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )
        if varying.height:
            raise DailyDataError(f"Expiry field {name!r} is not constant within expiry_id")


@version(
    "1.2",
    dependencies=[
        _empty_daily_frame,
        _one_positive_value,
        _require_raw_schema,
        _validate_constant_fields,
        DailyProvenance,
        DailyRawInputs,
        DailyWidbConfig,
        ClockConfig,
        RawDayConfig,
        canonical_json,
        canonical_sha256,
        bad_day_reason,
        bad_data_policy,
        bad_option_symbols,
        irregular_expiries,
        count_business_days_between,
        business_clock_policy,
        interpolate_zero_curve,
        is_half_day,
        _paired_schema,
        raw_table_spec,
        require_supported_date,
    ],
)
def pair_daily_frame(
    *,
    ticker: str,
    trading_date: date,
    raw: DailyRawInputs,
    config: DailyWidbConfig = DailyWidbConfig(),
    provenance: DailyProvenance = DailyProvenance(),
) -> pl.DataFrame:
    """Clean, pair, clock, and normalize one day's raw OptionMetrics evidence."""

    ticker = ticker.strip().upper()
    require_supported_date(trading_date)
    configuration_id = canonical_sha256(config)
    reason = bad_day_reason(ticker, trading_date)
    if reason is not None:
        return _empty_daily_frame(PAIRED_SCHEMA)

    security = _require_raw_schema(raw.security, RawTable.SECURITY)
    options = _require_raw_schema(raw.options, RawTable.OPTIONS)
    spot_frame = _require_raw_schema(raw.spot, RawTable.SPOT)
    previous_spot_frame = _require_raw_schema(raw.previous_spot, RawTable.SPOT)
    forwards = _require_raw_schema(raw.forwards, RawTable.FORWARDS)
    zero_curve = _require_raw_schema(raw.zero_curve, RawTable.ZERO_CURVE)
    if security.is_empty():
        raise DailyDataError("The ticker identity raw artifact is empty")
    matching = security.filter(pl.col("ticker").cast(pl.String).str.to_uppercase() == ticker)
    secids = matching.get_column("secid").drop_nulls().unique().cast(pl.Int64).to_list()
    if len(secids) != 1 or int(secids[0]) <= 0:
        raise DailyDataError(f"Ticker {ticker!r} does not resolve to one positive secid")
    secid = int(secids[0])
    if options.is_empty():
        raise DailyDataError(f"No option rows are available for {ticker} {trading_date}")
    if forwards.is_empty():
        raise DailyDataError(f"No forward rows are available for {ticker} {trading_date}")

    spot = _one_positive_value(spot_frame, "close", label="spot close")
    previous_spot = _one_positive_value(
        previous_spot_frame,
        "close",
        label="previous-session spot close",
    )
    diagnostics: dict[str, Any] = {
        "input_option_rows": options.height,
        "filter_counts": {},
        "removed": {},
    }

    def remove(frame: pl.DataFrame, mask: pl.Expr, label: str) -> pl.DataFrame:
        before = frame.height
        kept = frame.filter(~mask)
        removed = before - kept.height
        diagnostics["filter_counts"][label] = {
            "before": before,
            "removed": removed,
            "after": kept.height,
        }
        if removed:
            diagnostics["removed"][label] = removed
        if kept.is_empty():
            raise DailyDataError(
                f"All {before} option rows were removed at normalization stage {label!r}"
            )
        return kept

    options = options.with_columns(
        pl.col("exdate").cast(pl.String).str.to_date(strict=False).alias("exdate"),
        pl.col("symbol").cast(pl.String),
        pl.col("ss_flag").cast(pl.String),
    )
    bad_symbols = bad_option_symbols(ticker, trading_date)
    if bad_symbols:
        options = remove(options, pl.col("symbol").is_in(bad_symbols), "known_bad_symbol")
    else:
        diagnostics["filter_counts"]["known_bad_symbol"] = {
            "before": options.height,
            "removed": 0,
            "after": options.height,
        }
    options = remove(options, pl.col("exdate") == trading_date, "same_day_expiry")
    if config.raw.drop_irregular_expiries:
        options = remove(
            options,
            pl.col("exdate").is_in(irregular_expiries()),
            "irregular_expiry",
        )
    options = remove(
        options,
        pl.col("best_bid").is_null() | pl.col("best_offer").is_null(),
        "missing_quote",
    )
    options = remove(
        options,
        pl.col("best_bid").cast(pl.Float64) >= pl.col("best_offer").cast(pl.Float64),
        "crossed_quote",
    )
    options = remove(
        options,
        pl.col("best_bid").cast(pl.Float64) <= config.raw.min_bid,
        "bid_at_or_below_minimum",
    )
    options = remove(options, pl.col("ss_flag") != "0", "bad_settlement")

    calendar_dte = (
        options.get_column("exdate").cast(pl.Date).to_numpy().astype("datetime64[D]")
        - np.datetime64(trading_date, "D")
    ).astype(np.int32)
    discount = interpolate_zero_curve(zero_curve, calendar_dte.astype(np.float64))
    volume = options.get_column("volume").fill_null(0).cast(pl.Float64).to_numpy()
    open_interest = options.get_column("open_interest").fill_null(0).cast(pl.Float64).to_numpy()
    options = options.with_columns(
        pl.Series("discount_factor", discount),
        (pl.col("am_settlement").fill_null(0).cast(pl.Float64) == 1.0).alias("expiry_is_am"),
        (pl.col("strike_price").cast(pl.Float64) / 1000.0).alias("strike"),
        (pl.col("cp_flag").cast(pl.String).str.to_uppercase() == "C").alias("is_call"),
        pl.lit(False).alias("is_american"),
        pl.Series("log_volume", np.log10(1.0 + volume)),
        pl.Series("log_open_interest", np.log10(1.0 + open_interest)),
    )

    forwards = forwards.with_columns(
        pl.col("expiration").cast(pl.String).str.to_date(strict=False).alias("expiry_date"),
        (pl.col("amsettlement").fill_null(0).cast(pl.Float64) == 1.0).alias("expiry_is_am"),
        pl.col("forwardprice").cast(pl.Float64).alias("vendor_forward"),
    ).filter(pl.col("expiry_date") != trading_date)
    broken = pl.col("vendor_forward") == spot / 1_000_000.0
    diagnostics["repaired_vendor_forwards"] = forwards.filter(broken).height
    forwards = forwards.with_columns(
        pl.when(broken)
        .then(pl.lit(spot))
        .otherwise(pl.col("vendor_forward"))
        .alias("vendor_forward")
    )
    if forwards.height:
        minimum = forwards.get_column("vendor_forward").min()
        maximum = forwards.get_column("vendor_forward").max()
        if (
            minimum is not None
            and maximum is not None
            and (float(minimum) < spot / 2.0 or float(maximum) > spot * 2.0)
        ):
            raise DailyDataError(
                f"Suspicious vendor forward range [{minimum}, {maximum}] for spot {spot}"
            )
    duplicate = forwards.group_by(["expiry_date", "expiry_is_am"]).len().filter(pl.col("len") > 1)
    if duplicate.height:
        raise DailyDataError("Vendor forward table has duplicate expiry/settlement keys")
    options = options.join(
        forwards.select("expiry_date", "expiry_is_am", "vendor_forward"),
        left_on=["exdate", "expiry_is_am"],
        right_on=["expiry_date", "expiry_is_am"],
        how="left",
        maintain_order="left",
    )
    diagnostics["unmatched_forward_rows"] = options.filter(
        pl.col("vendor_forward").is_null()
    ).height

    keys = ["exdate", "expiry_is_am", "strike", "is_american"]
    calls = (
        options.filter(pl.col("is_call"))
        .sort(keys, maintain_order=True)
        .select(
            *keys,
            pl.col("symbol").alias("call_symbol"),
            pl.col("optionid").alias("call_option_id"),
            pl.col("best_bid").alias("market_call_bid"),
            pl.col("best_offer").alias("market_call_ask"),
            pl.col("log_volume").alias("market_call_log_volume"),
            pl.col("log_open_interest").alias("market_call_log_open_interest"),
            "discount_factor",
            "vendor_forward",
        )
    )
    puts = (
        options.filter(~pl.col("is_call"))
        .sort(keys, maintain_order=True)
        .select(
            *keys,
            pl.col("symbol").alias("put_symbol"),
            pl.col("optionid").alias("put_option_id"),
            pl.col("best_bid").alias("market_put_bid"),
            pl.col("best_offer").alias("market_put_ask"),
            pl.col("log_volume").alias("market_put_log_volume"),
            pl.col("log_open_interest").alias("market_put_log_open_interest"),
        )
    )
    diagnostics["call_rows_before_pairing"] = calls.height
    diagnostics["put_rows_before_pairing"] = puts.height
    diagnostics["unpaired_call_rows"] = calls.join(
        puts.select(keys).unique(), on=keys, how="anti"
    ).height
    diagnostics["unpaired_put_rows"] = puts.join(
        calls.select(keys).unique(), on=keys, how="anti"
    ).height
    paired = calls.join(puts, on=keys, how="inner", maintain_order="left")
    if paired.is_empty():
        raise DailyDataError("No call/put books share expiry, settlement, style, and strike")
    diagnostics["paired_rows"] = paired.height

    expiry_dates = paired.get_column("exdate").to_numpy().astype("datetime64[D]")
    expiry_is_am = paired.get_column("expiry_is_am").to_numpy().astype(bool)
    integer_bdte = np.asarray(
        count_business_days_between(trading_date, expiry_dates), dtype=np.float32
    )
    if np.min(integer_bdte) < 0.0:
        raise DailyDataError("An option expiry precedes its trading date")
    zero_bdte = integer_bdte == 0.0
    if np.any(zero_bdte):
        expiry_weekdays = np.asarray(
            [value.weekday() for value in paired.get_column("exdate").to_list()]
        )
        historical_saturday_am = expiry_is_am & (expiry_weekdays == 5)
        if not np.all(historical_saturday_am[zero_bdte]):
            raise DailyDataError(
                "Zero-business-day expiries are permitted only for historical Saturday AM books"
            )
    business_dte = np.clip(
        integer_bdte - np.where(expiry_is_am, config.clock.am_business_day_subtraction, 0.0),
        a_min=0.0,
        a_max=None,
    ).astype(np.float32)
    calendar_dte_paired = (expiry_dates - np.datetime64(trading_date, "D")).astype(np.int32)
    expiry_objects = paired.get_column("exdate").to_list()
    expiry_id = np.asarray(
        [
            value.year * 100000 + value.month * 1000 + value.day * 10 + (0 if am else 1)
            for value, am in zip(expiry_objects, expiry_is_am, strict=True)
        ],
        dtype=np.int32,
    )
    vendor_forward_values = paired.get_column("vendor_forward").drop_nulls()
    vendor_forward_range = (
        [None, None]
        if vendor_forward_values.is_empty()
        else [
            float(vendor_forward_values.min()),
            float(vendor_forward_values.max()),
        ]
    )
    diagnostics["ranges"] = {
        "spot": [float(spot), float(spot)],
        "previous_spot": [float(previous_spot), float(previous_spot)],
        "strike": [
            float(paired.get_column("strike").min()),
            float(paired.get_column("strike").max()),
        ],
        "discount_factor": [
            float(paired.get_column("discount_factor").min()),
            float(paired.get_column("discount_factor").max()),
        ],
        "vendor_forward": vendor_forward_range,
        "business_dte": [float(np.min(business_dte)), float(np.max(business_dte))],
        "calendar_dte": [
            int(np.min(calendar_dte_paired)),
            int(np.max(calendar_dte_paired)),
        ],
    }
    diagnostics_json = canonical_json(diagnostics)
    count = paired.height
    spot_float32 = np.float32(spot)
    previous_spot_float32 = np.float32(previous_spot)
    spot_return = np.float32(spot_float32 / previous_spot_float32 - np.float32(1.0))
    result = (
        paired.rename({"exdate": "expiry_date"})
        .with_columns(
            pl.lit(ticker).alias("ticker"),
            pl.lit(trading_date).cast(pl.Date).alias("trading_date"),
            pl.lit(spot_float32).cast(pl.Float32).alias("spot"),
            pl.lit(previous_spot_float32).cast(pl.Float32).alias("previous_spot"),
            pl.lit(spot_return).cast(pl.Float32).alias("spot_return"),
            pl.lit(is_half_day(trading_date)).alias("is_half_day"),
            pl.lit(secid).cast(pl.Int64).alias("source_secid"),
            pl.lit(str(pair_daily_frame.version)).alias("stage_version"),
            pl.lit(configuration_id).alias("configuration_id"),
            pl.Series(
                "raw_options_artifact_id",
                [provenance.raw_options_artifact_id] * count,
                dtype=pl.String,
            ),
            pl.lit(diagnostics_json).alias("normalization_diagnostics_json"),
            pl.Series("expiry_id", expiry_id),
            pl.Series("business_dte", business_dte),
            pl.Series("calendar_dte", calendar_dte_paired),
            pl.Series(
                "time_to_expiry_years",
                (business_dte / np.float32(DAYS_PER_YEAR)).astype(np.float32),
            ),
        )
        .select(*PAIRED_SCHEMA)
    )
    result = result.cast(PAIRED_SCHEMA, strict=True)
    _validate_constant_fields(result)
    return result


@version(
    "1.2",
    dependencies=[
        _empty_daily_frame,
        _series,
        _validate_constant_fields,
        DailyWidbConfig,
        ExpiryEnrichmentConfig,
        _black_implied,
        _black_values,
        _enriched_only_schema,
        _numerical_tolerances,
        _paired_schema,
        canonical_json,
        canonical_sha256,
        normalized_strike_policy,
    ],
)
def enrich_daily_frame(
    frame: pl.DataFrame,
    *,
    config: DailyWidbConfig = DailyWidbConfig(),
) -> pl.DataFrame:
    """Infer forwards, construct synthetic calls, and calculate pure Greeks."""

    # CD-003 preserves the reference's observable typed-empty behavior: an empty
    # input is not schema-validated before returning a typed empty enriched frame.
    if frame.is_empty():
        return _empty_daily_frame(ENRICHED_SCHEMA)
    if frame.schema != pl.Schema(PAIRED_SCHEMA):
        missing = sorted(set(PAIRED_SCHEMA) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(PAIRED_SCHEMA))
        raise DailyDataError(
            f"Normalized daily schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    data = frame.select(*PAIRED_SCHEMA)
    _validate_constant_fields(data)
    expiry_ids = _series(data, "expiry_id", np.int32)
    unique_expiry_ids = np.unique(expiry_ids)
    row_count = data.height
    put_log_volume = _series(data, "market_put_log_volume")
    call_log_volume = _series(data, "market_call_log_volume")
    put_log_oi = _series(data, "market_put_log_open_interest")
    call_log_oi = _series(data, "market_call_log_open_interest")
    either_traded = (put_log_volume > 0.0) | (call_log_volume > 0.0)
    both_traded = (put_log_volume > 0.0) & (call_log_volume > 0.0)
    discount = _series(data, "discount_factor")
    strike = _series(data, "strike")
    market_call_ask = _series(data, "market_call_ask")
    market_call_bid = _series(data, "market_call_bid")
    market_put_ask = _series(data, "market_put_ask")
    market_put_bid = _series(data, "market_put_bid")
    dffwd_ask = market_call_ask - market_put_bid + discount * strike
    dffwd_bid = market_call_bid - market_put_ask + discount * strike
    dffwd_mid = 0.5 * (dffwd_ask + dffwd_bid)
    dffwd_weight = np.where(
        both_traded,
        1.0 / np.maximum(dffwd_ask - dffwd_bid, 0.001),
        0.0,
    )

    arrays = {
        name: np.zeros(row_count, dtype=np.float32)
        for name, dtype in ENRICHED_ONLY_SCHEMA.items()
        if dtype == pl.Float32
    }
    arrays["combined_log_volume"] = np.log10(
        np.power(10.0, call_log_volume) + np.power(10.0, put_log_volume) - 1.0
    ).astype(np.float32)
    arrays["combined_log_open_interest"] = np.log10(
        np.power(10.0, call_log_oi) + np.power(10.0, put_log_oi) - 1.0
    ).astype(np.float32)
    valid = np.zeros(row_count, dtype=bool)
    diagnostics: dict[str, Any] = {
        "input_rows": row_count,
        "input_expiries": len(unique_expiry_ids),
        "dropped_expiries": [],
        "row_rejections": {
            "synthetic_price": 0,
            "pure_spread": 0,
            "ask_inversion_boundary": 0,
            "bid_inversion_boundary": 0,
        },
    }
    time_years = _series(data, "time_to_expiry_years", np.float32)
    expiry_dates = data.get_column("expiry_date").to_list()
    expiry_is_am = data.get_column("expiry_is_am").to_numpy().astype(bool)

    for expiry_id in unique_expiry_ids:
        mask = expiry_ids == expiry_id
        local_valid = np.zeros(np.sum(mask), dtype=bool)
        first_index = int(np.flatnonzero(mask)[0])
        label = f"{expiry_dates[first_index]}{'am' if expiry_is_am[first_index] else 'pm'}"
        if np.sum(both_traded[mask]) < config.enrichment.num_min_both_traded:
            diagnostics["dropped_expiries"].append(
                {"expiry": label, "reason": "insufficient_both_traded"}
            )
            continue
        if np.sum(either_traded[mask]) < config.enrichment.num_min_either_traded:
            diagnostics["dropped_expiries"].append(
                {"expiry": label, "reason": "insufficient_either_traded"}
            )
            continue
        weight_sum = np.sum(dffwd_weight[mask])
        dffwd = np.sum(dffwd_mid[mask] * dffwd_weight[mask]) / weight_sum
        if np.min(dffwd) < 1.0e-6:
            diagnostics["dropped_expiries"].append(
                {"expiry": label, "reason": "invalid_implied_forward"}
            )
            continue
        df_t = discount[mask][0]
        forward = dffwd / df_t
        tte = time_years[mask][0]
        sqrt_time = math.sqrt(float(tte))
        arrays["implied_forward"][mask] = forward
        local_valid[:] = True

        synthetic_ask = market_put_ask[mask] + dffwd - df_t * strike[mask]
        synthetic_bid = market_put_bid[mask] + dffwd - df_t * strike[mask]
        synthetic_spread = synthetic_ask - synthetic_bid
        bad = (
            (synthetic_ask <= 0.0)
            | (synthetic_bid <= 0.0)
            | (synthetic_spread < config.enrichment.min_pure_spread * dffwd)
        )
        rejected = local_valid & bad
        local_valid &= ~bad
        diagnostics["row_rejections"]["synthetic_price"] += int(np.sum(rejected))

        call_mid = 0.5 * (
            np.minimum(synthetic_ask, market_call_ask[mask])
            + np.maximum(synthetic_bid, market_call_bid[mask])
        )
        call_spread = np.minimum(
            market_call_ask[mask] - market_call_bid[mask],
            synthetic_ask - synthetic_bid,
        )
        new_call_ask = call_mid + 0.5 * call_spread
        new_call_bid = call_mid - 0.5 * call_spread
        normalized_strike = strike[mask] / forward
        pure_call_ask = new_call_ask / dffwd
        pure_call_bid = new_call_bid / dffwd
        bad = (pure_call_ask - pure_call_bid) < config.enrichment.min_pure_spread
        rejected = local_valid & bad
        local_valid &= ~bad
        diagnostics["row_rejections"]["pure_spread"] += int(np.sum(rejected))
        if not np.any(local_valid):
            diagnostics["dropped_expiries"].append({"expiry": label, "reason": "no_valid_prices"})
            continue
        if np.sum(both_traded[mask][local_valid]) < config.enrichment.num_min_both_traded:
            diagnostics["dropped_expiries"].append(
                {"expiry": label, "reason": "insufficient_both_after_prices"}
            )
            continue
        if np.sum(either_traded[mask][local_valid]) < config.enrichment.num_min_either_traded:
            diagnostics["dropped_expiries"].append(
                {"expiry": label, "reason": "insufficient_either_after_prices"}
            )
            continue

        spread = pure_call_ask - pure_call_bid
        price_tolerance = np.minimum(spread * 0.1, 1.0e-6)
        ask_vols = _black_implied(
            normalized_strike,
            pure_call_ask,
            sqrtT=sqrt_time,
            price_tol=price_tolerance,
            mask=local_valid,
            eps=_IMPLIED_VOL_CALC_EPS,
            vol_min=config.enrichment.min_implied_vol,
            vol_max=config.enrichment.max_implied_vol,
        )
        bid_vols = _black_implied(
            normalized_strike,
            pure_call_bid,
            sqrtT=sqrt_time,
            price_tol=price_tolerance,
            mask=local_valid,
            eps=_IMPLIED_VOL_CALC_EPS,
            vol_min=config.enrichment.min_implied_vol,
            vol_max=config.enrichment.max_implied_vol,
        )
        atm_vol = np.interp(1.0, normalized_strike, (ask_vols + bid_vols) * 0.5)
        ask_vols[~local_valid] = config.enrichment.min_implied_vol * 1.1
        bid_vols[~local_valid] = config.enrichment.min_implied_vol
        bid_greeks = _black_values(
            k=normalized_strike,
            vol=bid_vols,
            sqrtT=sqrt_time,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        ask_greeks = _black_values(
            k=normalized_strike,
            vol=ask_vols,
            sqrtT=sqrt_time,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        for side, calculated, target in (
            ("ask", ask_greeks, pure_call_ask),
            ("bid", bid_greeks, pure_call_bid),
        ):
            difference = calculated.price - target
            invalid_inversion = (np.abs(difference) > price_tolerance * 1.1) & local_valid
            if np.any(invalid_inversion):
                inversion_vols = (ask_vols if calculated is ask_greeks else bid_vols)[
                    invalid_inversion
                ]
                at_boundary = (inversion_vols == config.enrichment.min_implied_vol) | (
                    inversion_vols == config.enrichment.max_implied_vol
                )
                if not np.all(at_boundary):
                    raise DailyDataError(
                        f"Implied-volatility inversion exceeded tolerance away from a "
                        f"configured boundary for expiry {label}"
                    )
                rejected = local_valid & invalid_inversion
                local_valid &= ~invalid_inversion
                diagnostics["row_rejections"][f"{side}_inversion_boundary"] += int(np.sum(rejected))
        if not np.any(local_valid):
            raise DailyDataError(
                f"All otherwise eligible rows failed implied-volatility inversion for {label}"
            )

        arrays["normalized_strike"][mask] = normalized_strike
        arrays["ask_implied_volatility"][mask] = ask_vols
        arrays["bid_implied_volatility"][mask] = bid_vols
        for side, greeks in (("bid", bid_greeks), ("ask", ask_greeks)):
            for name, values in greeks.items():
                copied = np.asarray(values).copy()
                copied[~local_valid] = 0.0
                arrays[f"pure_{side}_{name}"][mask] = copied
        arrays["synthetic_call_ask"][mask] = ask_greeks.price * dffwd
        arrays["synthetic_call_bid"][mask] = bid_greeks.price * dffwd
        arrays["normalized_moneyness"][mask] = NMS5.to_nms(normalized_strike, tte)
        arrays["atm_normalized_moneyness"][mask] = NMS5.to_nms(
            normalized_strike,
            tte,
            volatility=atm_vol,
        )
        if not np.all(arrays["synthetic_call_bid"][mask][local_valid] > 0.0):
            raise DailyDataError(f"Non-positive synthetic call bid for expiry {label}")
        if not np.all(
            arrays["synthetic_call_ask"][mask][local_valid]
            > arrays["synthetic_call_bid"][mask][local_valid]
        ):
            raise DailyDataError(f"Non-positive synthetic call spread for expiry {label}")
        valid[mask] = local_valid

    # CD-020: preserve the pinned reference's observable all-valid failure.
    # The legacy implementation retained its internal ``is_valid`` column when
    # every row survived and then failed schema validation. Removing this guard
    # is the proposed repair, but requires explicit user approval.
    if np.all(valid):
        raise DailyDataError(
            "Reference compatibility: an all-valid enriched day fails because "
            "the legacy is_valid control column reaches schema validation (CD-020)"
        )

    result = data.with_columns(
        *(pl.Series(name, values) for name, values in arrays.items()),
        pl.lit(str(enrich_daily_frame.version)).alias("stage_version"),
        pl.lit(canonical_sha256(config)).alias("configuration_id"),
    ).filter(pl.Series("_valid", valid))
    if result.is_empty():
        raise DailyDataError(
            "No valid option rows remain after daily enrichment; "
            f"diagnostics={canonical_json(diagnostics)}"
        )
    diagnostics["output_rows"] = result.height
    diagnostics["dropped_rows"] = row_count - result.height
    diagnostics["output_expiries"] = result.get_column("expiry_id").n_unique()
    diagnostics["output_rows_by_expiry"] = {
        str(row["expiry_id"]): int(row["len"])
        for row in result.group_by("expiry_id").len().sort("expiry_id").to_dicts()
    }
    range_columns = (
        "implied_forward",
        "normalized_strike",
        "normalized_moneyness",
        "atm_normalized_moneyness",
        "bid_implied_volatility",
        "ask_implied_volatility",
        "pure_bid_price",
        "pure_ask_price",
    )
    diagnostics["ranges"] = {
        name: [
            float(result.get_column(name).min()),
            float(result.get_column(name).max()),
        ]
        for name in range_columns
    }
    result = result.with_columns(
        pl.lit(canonical_json(diagnostics)).alias("enrichment_diagnostics_json")
    ).select(*ENRICHED_SCHEMA)
    result = result.cast(ENRICHED_SCHEMA, strict=True)
    _validate_constant_fields(result)
    return result


__all__ = [
    "DailyDataError",
    "DailyProvenance",
    "DailyRawInputs",
    "ENRICHED_SCHEMA",
    "ENRICHED_SCHEMA_VERSION",
    "PAIRED_SCHEMA",
    "PAIRED_SCHEMA_VERSION",
    "enrich_daily_frame",
    "interpolate_zero_curve",
    "pair_daily_frame",
]
