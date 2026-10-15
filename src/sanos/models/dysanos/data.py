"""Public-derived driftless data contracts for DySANOS 1.1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from cdxcore.version import version

from sanos.artifacts import ArtifactPointer
from sanos.config import canonical_json, canonical_sha256
from sanos.datasets.public.schema import public_polars_schema
from sanos.model.tools import SanosPricer
from sanos.models.dysanos.config import (
    DailyFrontGeometryConfig,
    DailyFrontMarketConfig,
)
from sanos.widb.calendar import DAYS_PER_YEAR, NMS, business_dates
from sanos.widb.daily import DailyDataError


@version("1.0")
@dataclass(frozen=True, slots=True)
class EligibilityAudit:
    active_european_books: int
    horizon_eligible_books: int
    nms_eligible_books: int
    input_expiries: int
    eligible_expiries: int


@version("1.0")
def _ordered_fingerprint(
    scalars: tuple[str, ...],
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    for value in scalars:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@version("2.0", dependencies=[EligibilityAudit, _ordered_fingerprint])
@dataclass(frozen=True, slots=True)
class DailyFrontDay:
    """One immutable logical day before global HDF5 padding."""

    ticker: str
    trading_date: date
    is_half_day: bool
    driftless_spot_return: float
    driftless_spot_return_valid: bool
    market_uid: str
    market_fingerprint: str
    expiry_time_y: np.ndarray
    expiry_date_days: np.ndarray
    expiry_is_am: np.ndarray
    expiry_atm_annual_vol: np.ndarray
    expiry_strike_lower: np.ndarray
    expiry_strike_upper: np.ndarray
    model_atm_total_vol: np.ndarray
    quote_expiry_index: np.ndarray
    quote_price_index: np.ndarray
    quote_strike: np.ndarray
    quote_is_call: np.ndarray
    quote_fit: np.ndarray
    quote_half_spread: np.ndarray
    price_expiry_y: np.ndarray
    price_strike: np.ndarray
    audit: EligibilityAudit

    def __post_init__(self) -> None:
        if not self.ticker or not isinstance(self.is_half_day, bool):
            raise ValueError("A daily-front day requires ticker and half-day state")
        if not isinstance(self.driftless_spot_return_valid, bool):
            raise ValueError("driftless_spot_return_valid must be boolean")
        if self.driftless_spot_return_valid:
            if not np.isfinite(self.driftless_spot_return):
                raise ValueError("A valid driftless spot return must be finite")
        elif not np.isnan(self.driftless_spot_return):
            raise ValueError("An unavailable driftless spot return must be NaN")
        if (
            len(self.market_uid) != 64
            or len(self.market_fingerprint) != 64
            or any(value not in "0123456789abcdef" for value in self.market_uid)
            or any(value not in "0123456789abcdef" for value in self.market_fingerprint)
        ):
            raise ValueError("Market identities must be SHA-256 digests")
        expiry_count = len(self.expiry_time_y)
        expiry_arrays = (
            self.expiry_date_days,
            self.expiry_is_am,
            self.expiry_atm_annual_vol,
            self.expiry_strike_lower,
            self.expiry_strike_upper,
        )
        if expiry_count == 0 or any(array.shape != (expiry_count,) for array in expiry_arrays):
            raise ValueError("Expiry arrays must be non-empty and aligned")
        if (
            np.any(~np.isfinite(self.expiry_time_y))
            or np.any(self.expiry_time_y < 0.0)
            or np.any(np.diff(self.expiry_time_y) <= 0.0)
            or np.any(~np.isfinite(self.expiry_atm_annual_vol))
            or np.any(self.expiry_atm_annual_vol <= 0.0)
            or np.any(self.expiry_strike_lower <= 0.0)
            or np.any(self.expiry_strike_lower >= 1.0)
            or np.any(self.expiry_strike_upper <= 1.0)
        ):
            raise ValueError("Expiry geometry is invalid")
        quote_count = len(self.quote_strike)
        quote_arrays = (
            self.quote_expiry_index,
            self.quote_price_index,
            self.quote_is_call,
            self.quote_fit,
            self.quote_half_spread,
        )
        if quote_count == 0 or any(array.shape != (quote_count,) for array in quote_arrays):
            raise ValueError("Quote arrays must be non-empty and aligned")
        if (
            np.any(self.quote_expiry_index < 0)
            or np.any(self.quote_expiry_index >= expiry_count)
            or np.any(~np.isfinite(self.quote_strike))
            or np.any(self.quote_strike <= 0.0)
            or np.any(~np.isfinite(self.quote_fit))
            or np.any(~np.isfinite(self.quote_half_spread))
            or np.any(self.quote_half_spread <= 0.0)
        ):
            raise ValueError("Quote objective arrays are invalid")
        quote_lower = self.expiry_strike_lower[self.quote_expiry_index]
        quote_upper = self.expiry_strike_upper[self.quote_expiry_index]
        if np.any(self.quote_strike < quote_lower) or np.any(self.quote_strike > quote_upper):
            raise ValueError("Quote strikes lie outside their inclusive NMS bounds")
        price_count = len(self.price_strike)
        if self.price_expiry_y.shape != (price_count,) or price_count == 0:
            raise ValueError("Price-location arrays must be non-empty and aligned")
        if (
            np.any(self.quote_price_index < 0)
            or np.any(self.quote_price_index >= price_count)
            or np.any(~np.isfinite(self.price_expiry_y))
            or np.any(self.price_expiry_y < 0.0)
            or np.any(~np.isfinite(self.price_strike))
            or np.any(self.price_strike <= 0.0)
        ):
            raise ValueError("Price-location geometry is invalid")
        if (
            self.model_atm_total_vol.ndim != 1
            or np.any(~np.isfinite(self.model_atm_total_vol))
            or np.any(self.model_atm_total_vol <= 0.0)
        ):
            raise ValueError("Model ATM total volatilities must be positive")
        expected_price_indices: list[int] = []
        prices: dict[tuple[int, float], int] = {}
        for expiry_index, strike in zip(
            self.quote_expiry_index,
            self.quote_strike,
            strict=True,
        ):
            key = (int(expiry_index), float(strike))
            if key not in prices:
                prices[key] = len(prices)
            expected_price_indices.append(prices[key])
        if len(prices) != price_count or not np.array_equal(
            self.quote_price_index,
            np.asarray(expected_price_indices, dtype=np.int64),
        ):
            raise ValueError("Quote-to-price mapping is not first-occurrence canonical")
        ordered_price_keys = tuple(prices)
        expected_price_expiry = np.asarray(
            [self.expiry_time_y[key[0]] for key in ordered_price_keys], dtype=np.float64
        )
        expected_price_strike = np.asarray([key[1] for key in ordered_price_keys], dtype=np.float64)
        if not np.array_equal(self.price_expiry_y, expected_price_expiry) or not np.array_equal(
            self.price_strike, expected_price_strike
        ):
            raise ValueError("Price-location geometry does not match quote keys")
        if np.any(np.diff(self.quote_expiry_index) < 0):
            raise ValueError("Quotes must be ordered by expiry")
        for expiry_index in range(expiry_count):
            indices = np.flatnonzero(self.quote_expiry_index == expiry_index)
            if len(indices) == 0:
                raise ValueError("Every retained expiry must own at least one quote")
            sides = self.quote_is_call[indices]
            if np.any((~sides[:-1]) & sides[1:]):
                raise ValueError("Calls must precede puts within each expiry")
            for side in (True, False):
                side_strikes = self.quote_strike[indices][sides == side]
                if len(side_strikes) > 1 and np.any(np.diff(side_strikes) <= 0.0):
                    raise ValueError("Quote strikes must increase within expiry and side")
        if not (
            self.audit.active_european_books
            >= self.audit.horizon_eligible_books
            >= self.audit.nms_eligible_books
            == quote_count
            and self.audit.input_expiries >= self.audit.eligible_expiries == expiry_count
        ):
            raise ValueError("Eligibility audit counts do not match projected arrays")

    @property
    def quote_count(self) -> int:
        return len(self.quote_strike)

    @property
    def price_count(self) -> int:
        return len(self.price_strike)

    @property
    def expiry_count(self) -> int:
        return len(self.expiry_time_y)

    # ``flex-market-day-v1`` compatibility names.  Keep these as a thin,
    # zero-copy view so the pinned reference consumer can ingest a day loaded
    # from the SANOS Experiment without importing quant_data in production.
    @property
    def expiries_y(self) -> np.ndarray:
        return self.expiry_time_y

    @property
    def expiry_dates(self) -> tuple[str, ...]:
        epoch = np.datetime64("1970-01-01", "D")
        return tuple(
            str(epoch + np.timedelta64(int(value), "D")) for value in self.expiry_date_days
        )

    @property
    def expiry_ids(self) -> np.ndarray:
        """Dense public expiry indices retained for legacy adapters only."""

        return np.arange(self.expiry_count, dtype=np.int64)

    @property
    def expiry_index(self) -> np.ndarray:
        return self.quote_expiry_index

    @property
    def strikes(self) -> np.ndarray:
        return self.quote_strike

    @property
    def is_call(self) -> np.ndarray:
        return self.quote_is_call

    @property
    def bids(self) -> np.ndarray:
        return self.quote_fit - self.quote_half_spread

    @property
    def asks(self) -> np.ndarray:
        return self.quote_fit + self.quote_half_spread

    @property
    def quote_bid(self) -> np.ndarray:
        """Synthetic driftless bid reconstructed from fit minus half-spread."""

        return self.bids

    @property
    def quote_ask(self) -> np.ndarray:
        """Synthetic driftless ask reconstructed from fit plus half-spread."""

        return self.asks

    @property
    def fit_prices(self) -> np.ndarray:
        return self.quote_fit

    @property
    def quote_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.trading_date.isoformat()}:{int(expiry_index)}:"
            f"{strike:.12g}:{'C' if is_call else 'P'}"
            for expiry_index, strike, is_call in zip(
                self.quote_expiry_index,
                self.quote_strike,
                self.quote_is_call,
                strict=True,
            )
        )

    @property
    def atm_annual_vols(self) -> np.ndarray:
        return self.expiry_atm_annual_vol

    @property
    def model_atm_total_vols(self) -> np.ndarray:
        return self.model_atm_total_vol

    @property
    def strike_lower_bounds(self) -> np.ndarray:
        return self.expiry_strike_lower

    @property
    def strike_upper_bounds(self) -> np.ndarray:
        return self.expiry_strike_upper

    @property
    def num_quotes(self) -> int:
        return self.quote_count

    @property
    def training_fingerprint(self) -> str:
        """Exact pinned FLEX objective-only fingerprint for consumer parity."""

        return _ordered_fingerprint(
            (
                self.ticker,
                self.trading_date.isoformat(),
                self.market_uid,
                *self.expiry_dates,
                *self.quote_ids,
            ),
            (
                self.expiries_y,
                self.expiry_is_am,
                self.expiry_index,
                self.strikes,
                self.is_call,
                self.fit_prices,
                self.quote_half_spread,
                self.atm_annual_vols,
                self.model_atm_total_vols,
                self.strike_lower_bounds,
                self.strike_upper_bounds,
            ),
        )

    @property
    def business_dtes(self) -> np.ndarray:
        return self.expiry_time_y * DAYS_PER_YEAR

    @property
    def quote_business_dtes(self) -> np.ndarray:
        return self.business_dtes[self.quote_expiry_index]

    @property
    def quote_expiry_y(self) -> np.ndarray:
        """Observed business-year expiry for every separate side book."""

        return self.expiry_time_y[self.quote_expiry_index]

    @property
    def half_spreads(self) -> np.ndarray:
        """One normalized half-spread per retained separate side book."""

        return self.quote_half_spread

    @property
    def call_equivalent_fit_prices(self) -> np.ndarray:
        """Smooth targets represented as calls without merging observations."""

        return self.quote_fit + np.where(
            self.quote_is_call,
            0.0,
            1.0 - self.quote_strike,
        )

    @property
    def call_equivalent_book_mids(self) -> np.ndarray:
        """Observed mids represented as calls without merging observations."""

        return self.quote_fit + np.where(
            self.quote_is_call,
            0.0,
            1.0 - self.quote_strike,
        )


@version("1.0", dependencies=[DailyFrontDay])
def daily_quote_objective(
    prices: np.ndarray,
    day: DailyFrontDay,
    *,
    fit_weight: float,
) -> float:
    """Evaluate the recovered within-date quote objective in NumPy."""

    values = np.asarray(prices, dtype=np.float64)
    if values.shape != (day.quote_count,) or np.any(~np.isfinite(values)):
        raise ValueError("prices must be one finite value per retained quote")
    if isinstance(fit_weight, bool) or not np.isfinite(fit_weight) or fit_weight < 0.0:
        raise ValueError("fit_weight must be finite and non-negative")
    loss = (
        np.maximum(values - day.quote_ask, 0.0)
        + np.maximum(day.quote_bid - values, 0.0)
        + float(fit_weight) * np.abs(values - day.quote_fit)
    ) / day.half_spreads
    return float(np.mean(loss))


@version("1.0")
def _one_value(frame: pl.DataFrame, column: str) -> Any:
    values = frame.get_column(column).unique().to_list()
    if len(values) != 1 or values[0] is None:
        raise DailyDataError(f"{column} must contain one non-null daily value")
    return values[0]


@version("1.0")
def _days_since_epoch(values: list[date]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[D]").astype(np.int64).astype(np.int32)


@version("1.0")
def _interpolate_atm_vols(
    expiries: np.ndarray,
    model_expiries: np.ndarray,
    model_variances: np.ndarray,
) -> np.ndarray:
    """Reproduce the recovered ArbFreePricer total-variance interpolation."""

    requested = np.asarray(expiries, dtype=np.float64)
    clipped = np.clip(requested, model_expiries[0], model_expiries[-1])
    variances = np.interp(clipped, model_expiries, model_variances)
    result = np.sqrt(variances / clipped)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise DailyDataError("ATM interpolation produced invalid volatility")
    return result


@version("1.2", dependencies=[canonical_json, _ordered_fingerprint])
def _market_fingerprint(
    *,
    ticker: str,
    trading_date: date,
    market_uid: str,
    source: ArtifactPointer,
    expiry_dates: tuple[str, ...],
    quote_ids: tuple[str, ...],
    arrays: tuple[np.ndarray, ...],
) -> str:
    return _ordered_fingerprint(
        (
            ticker,
            trading_date.isoformat(),
            market_uid,
            canonical_json(source.as_dict()),
            *expiry_dates,
            *quote_ids,
        ),
        arrays,
    )


@version("2.0", dependencies=[DailyFrontDay, _market_fingerprint])
def recompute_market_fingerprint(
    day: DailyFrontDay,
    source_dataset: ArtifactPointer,
) -> str:
    """Reconstruct the exact source-day training fingerprint from one projection."""

    epoch = np.datetime64("1970-01-01", "D")
    expiry_dates = tuple(
        str(epoch + np.timedelta64(int(value), "D")) for value in day.expiry_date_days
    )
    quote_ids = tuple(
        f"{day.trading_date.isoformat()}:{int(expiry_index)}:"
        f"{strike:.12g}:{'C' if is_call else 'P'}"
        for expiry_index, strike, is_call in zip(
            day.quote_expiry_index,
            day.quote_strike,
            day.quote_is_call,
            strict=True,
        )
    )
    return _market_fingerprint(
        ticker=day.ticker,
        trading_date=day.trading_date,
        market_uid=day.market_uid,
        source=source_dataset,
        expiry_dates=expiry_dates,
        quote_ids=quote_ids,
        arrays=(
            day.expiry_time_y,
            day.expiry_is_am,
            day.quote_expiry_index,
            day.quote_strike,
            day.quote_is_call,
            day.quote_fit,
            day.quote_half_spread,
            day.expiry_atm_annual_vol,
            day.model_atm_total_vol,
            day.expiry_strike_lower,
            day.expiry_strike_upper,
        ),
    )


@version(
    "2.0",
    dependencies=[
        DailyFrontDay,
        DailyFrontMarketConfig,
        DailyFrontGeometryConfig,
        _interpolate_atm_vols,
        _market_fingerprint,
        _one_value,
        _days_since_epoch,
        canonical_sha256,
        SanosPricer,
        public_polars_schema,
        NMS,
    ],
)
def project_daily_front(
    frame: pl.DataFrame,
    *,
    ticker: str,
    source_dataset: ArtifactPointer,
    market: DailyFrontMarketConfig,
    geometry: DailyFrontGeometryConfig,
) -> DailyFrontDay:
    """Project one flat public day into the normalized DySANOS objective."""

    if frame.schema != public_polars_schema():
        raise DailyDataError("daily-front source does not have the flat public schema")
    if frame.is_empty():
        raise DailyDataError("A public day cannot be empty")
    trading_date = _one_value(frame, "trading_date")
    if not isinstance(trading_date, date):
        raise DailyDataError("trading_date must be a date")
    day_rows = frame.filter(pl.col("record_kind") == "day")
    if day_rows.height != 1:
        raise DailyDataError("A public day requires exactly one day row")
    is_half_day = bool(_one_value(day_rows, "is_half_day"))
    smoothness = float(_one_value(day_rows, "smoothness"))
    if smoothness != market.smoothness:
        raise DailyDataError(
            f"Public smoothness {smoothness} differs from Experiment {market.smoothness}"
        )
    return_value = day_rows.get_column("driftless_spot_return")[0]
    return_valid = return_value is not None
    driftless_return = float(return_value) if return_valid else float("nan")

    expiry_rows = frame.filter(pl.col("record_kind") == "expiry").sort("expiry_index")
    book_rows = frame.filter(pl.col("record_kind") == "book").sort(
        "expiry_index", "book_index"
    )
    smooth_rows = frame.filter(pl.col("record_kind") == "smooth_node").sort(
        "expiry_index", "node_index"
    )
    if expiry_rows.is_empty() or book_rows.is_empty() or smooth_rows.is_empty():
        raise DailyDataError("A public day requires expiry, book, and smooth-state rows")

    pricer_times: list[float] = []
    pricer_variances: list[float] = []
    pricer_supports: list[np.ndarray] = []
    pricer_densities: list[np.ndarray] = []
    for public_index in smooth_rows.get_column("expiry_index").unique(maintain_order=True):
        nodes = smooth_rows.filter(pl.col("expiry_index") == public_index).sort("node_index")
        expiry = expiry_rows.filter(pl.col("expiry_index") == public_index)
        if expiry.height != 1:
            raise DailyDataError("Smooth state refers to missing public expiry geometry")
        pricer_times.append(float(expiry.get_column("time_to_expiry_years")[0]))
        pricer_variances.append(float(_one_value(nodes, "kernel_input_total_variance")))
        pricer_supports.append(nodes.get_column("normalized_support").cast(pl.Float64).to_numpy())
        pricer_densities.append(nodes.get_column("density_mass").cast(pl.Float64).to_numpy())
    pricer = SanosPricer(
        smoothness=smoothness,
        expiries=np.asarray(pricer_times, dtype=np.float64),
        variances=np.asarray(pricer_variances, dtype=np.float64),
        xstrikes=pricer_supports,
        xdensities=pricer_densities,
    )
    model_business_dtes = np.asarray(geometry.model_business_dtes, dtype=np.float64)
    model_time_y = model_business_dtes / DAYS_PER_YEAR
    model_atm_vol = _interpolate_atm_vols(
        model_time_y,
        np.asarray(pricer.expiries, dtype=np.float64),
        np.asarray(pricer.variances, dtype=np.float64),
    )
    model_atm_total_vol = model_atm_vol * np.sqrt(model_time_y)

    public_book_expiries = set(int(value) for value in book_rows["expiry_index"].to_list())
    retained_expiry_rows = expiry_rows.filter(pl.col("expiry_index").is_in(public_book_expiries))
    public_to_dense = {
        int(value): index
        for index, value in enumerate(retained_expiry_rows["expiry_index"].to_list())
    }
    expiry_time_array = retained_expiry_rows["time_to_expiry_years"].cast(pl.Float64).to_numpy()
    expiry_atm_array = retained_expiry_rows["nms_atm_annual_volatility"].cast(pl.Float64).to_numpy()
    epsilon_atm = float(_one_value(day_rows, "nms_epsilon_atm_annual_volatility"))
    bounds = np.asarray(
        NMS.from_nms(
            np.asarray(geometry.nms_range, dtype=np.float64)[None, :],
            expiry_time_array[:, None],
            volatility=expiry_atm_array[:, None],
            epsilon_volatility=epsilon_atm,
        ),
        dtype=np.float64,
    )
    quote_expiry_array = np.asarray(
        [public_to_dense[int(value)] for value in book_rows["expiry_index"].to_list()],
        dtype=np.int64,
    )
    quote_strike_array = book_rows["normalized_strike"].cast(pl.Float64).to_numpy()
    quote_side_array = book_rows["is_call"].to_numpy().astype(np.bool_)
    quote_half_spread = book_rows["half_spread"].cast(pl.Float64).to_numpy()
    call_fit = np.asarray(
        [
            pricer(float(expiry_time_array[expiry_index]), np.asarray([strike]))[0]
            for expiry_index, strike in zip(
                quote_expiry_array, quote_strike_array, strict=True
            )
        ],
        dtype=np.float64,
    )
    quote_fit_array = call_fit + np.where(
        quote_side_array,
        0.0,
        quote_strike_array - 1.0,
    )
    price_by_key: dict[tuple[int, float], int] = {}
    price_expiry_y: list[float] = []
    price_strike: list[float] = []
    quote_price_index = np.empty(len(quote_strike_array), dtype=np.int64)
    for index, (expiry_index, strike) in enumerate(
        zip(quote_expiry_array, quote_strike_array, strict=True)
    ):
        key = (int(expiry_index), float(strike))
        price_index = price_by_key.get(key)
        if price_index is None:
            price_index = len(price_strike)
            price_by_key[key] = price_index
            price_expiry_y.append(float(expiry_time_array[int(expiry_index)]))
            price_strike.append(float(strike))
        quote_price_index[index] = price_index

    expiry_am_array = retained_expiry_rows["is_am"].to_numpy().astype(np.bool_)
    lower_array = bounds[:, 0]
    upper_array = bounds[:, 1]
    expiry_dates = retained_expiry_rows["expiry_date"].to_list()
    expiry_date_array = _days_since_epoch(expiry_dates)
    price_expiry_array = np.asarray(price_expiry_y, dtype=np.float64)
    price_strike_array = np.asarray(price_strike, dtype=np.float64)
    if not ticker:
        raise DailyDataError("DySANOS 1.1 requires an explicit ticker label")
    quote_ids = tuple(
        f"{trading_date.isoformat()}:{int(expiry_index)}:{strike:.12g}:"
        f"{'C' if is_call else 'P'}"
        for expiry_index, strike, is_call in zip(
            quote_expiry_array,
            quote_strike_array,
            quote_side_array,
            strict=True,
        )
    )
    market_uid = canonical_sha256(
        {
            "source_dataset": source_dataset,
            "public_schema": str(public_polars_schema()),
            "market": market,
            "geometry": geometry,
        }
    )
    fingerprint = _market_fingerprint(
        ticker=ticker,
        trading_date=trading_date,
        market_uid=market_uid,
        source=source_dataset,
        expiry_dates=tuple(value.isoformat() for value in expiry_dates),
        quote_ids=tuple(quote_ids),
        arrays=(
            expiry_time_array,
            expiry_am_array,
            quote_expiry_array,
            quote_strike_array,
            quote_side_array,
            quote_fit_array,
            quote_half_spread,
            expiry_atm_array,
            model_atm_total_vol,
            lower_array,
            upper_array,
        ),
    )
    return DailyFrontDay(
        ticker=ticker,
        trading_date=trading_date,
        is_half_day=is_half_day,
        driftless_spot_return=driftless_return,
        driftless_spot_return_valid=return_valid,
        market_uid=market_uid,
        market_fingerprint=fingerprint,
        expiry_time_y=expiry_time_array,
        expiry_date_days=expiry_date_array,
        expiry_is_am=expiry_am_array,
        expiry_atm_annual_vol=expiry_atm_array,
        expiry_strike_lower=lower_array,
        expiry_strike_upper=upper_array,
        model_atm_total_vol=np.asarray(model_atm_total_vol, dtype=np.float64),
        quote_expiry_index=quote_expiry_array,
        quote_price_index=quote_price_index,
        quote_strike=quote_strike_array,
        quote_is_call=quote_side_array,
        quote_fit=quote_fit_array,
        quote_half_spread=quote_half_spread,
        price_expiry_y=price_expiry_array,
        price_strike=price_strike_array,
        audit=EligibilityAudit(
            active_european_books=len(quote_strike_array),
            horizon_eligible_books=len(quote_strike_array),
            nms_eligible_books=len(quote_strike_array),
            input_expiries=expiry_rows.height,
            eligible_expiries=len(expiry_time_array),
        ),
    )


@version("1.0", dependencies=[business_dates])
def exact_weekly_pairs(dates: tuple[date, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return retained endpoints separated by exactly five NYSE sessions."""

    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("dates must be non-empty, unique, and increasing")
    retained = {value: index for index, value in enumerate(dates)}
    calendar = business_dates(dates[0], dates[-1])
    calendar_index = {value: index for index, value in enumerate(calendar)}
    pairs = [
        (index, retained[calendar[position + 5]])
        for index, value in enumerate(dates)
        if (position := calendar_index[value]) + 5 < len(calendar)
        and calendar[position + 5] in retained
    ]
    return (
        np.asarray([item[0] for item in pairs], dtype=np.int64),
        np.asarray([item[1] for item in pairs], dtype=np.int64),
    )


__all__ = [
    "DailyFrontDay",
    "EligibilityAudit",
    "daily_quote_objective",
    "exact_weekly_pairs",
    "project_daily_front",
    "recompute_market_fingerprint",
]
