"""Exact internal daily-front market projection for the reference Experiment."""

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
from sanos.experiments.config import (
    DailyFrontGeometryConfig,
    DailyFrontMarketConfig,
)
from sanos.widb.calendar import DAYS_PER_YEAR, NMS, business_dates
from sanos.widb.daily import DailyDataError
from sanos.widb.surface import FITTED_SCHEMA, smooth_sanos_pricer


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


@version("1.6", dependencies=[EligibilityAudit, _ordered_fingerprint])
@dataclass(frozen=True, slots=True)
class DailyFrontDay:
    """One immutable logical day before global HDF5 padding."""

    ticker: str
    trading_date: date
    is_half_day: bool
    spot_reference: float
    market_uid: str
    market_fingerprint: str
    expiry_time_y: np.ndarray
    expiry_date_days: np.ndarray
    expiry_id: np.ndarray
    expiry_is_am: np.ndarray
    expiry_atm_annual_vol: np.ndarray
    expiry_strike_lower: np.ndarray
    expiry_strike_upper: np.ndarray
    model_atm_total_vol: np.ndarray
    quote_expiry_index: np.ndarray
    quote_price_index: np.ndarray
    quote_strike: np.ndarray
    quote_is_call: np.ndarray
    quote_bid: np.ndarray
    quote_ask: np.ndarray
    quote_fit: np.ndarray
    quote_log_volume: np.ndarray
    quote_log_open_interest: np.ndarray
    price_expiry_y: np.ndarray
    price_strike: np.ndarray
    audit: EligibilityAudit

    def __post_init__(self) -> None:
        if not self.ticker or not isinstance(self.is_half_day, bool):
            raise ValueError("A daily-front day requires ticker and half-day state")
        if not np.isfinite(self.spot_reference) or self.spot_reference <= 0.0:
            raise ValueError("spot_reference must be finite and positive")
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
            self.expiry_id,
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
            or len(set(int(value) for value in self.expiry_id)) != expiry_count
        ):
            raise ValueError("Expiry geometry is invalid")
        quote_count = len(self.quote_strike)
        quote_arrays = (
            self.quote_expiry_index,
            self.quote_price_index,
            self.quote_is_call,
            self.quote_bid,
            self.quote_ask,
            self.quote_fit,
            self.quote_log_volume,
            self.quote_log_open_interest,
        )
        if quote_count == 0 or any(array.shape != (quote_count,) for array in quote_arrays):
            raise ValueError("Quote arrays must be non-empty and aligned")
        if (
            np.any(self.quote_expiry_index < 0)
            or np.any(self.quote_expiry_index >= expiry_count)
            or np.any(~np.isfinite(self.quote_strike))
            or np.any(self.quote_strike <= 0.0)
            or np.any(~np.isfinite(self.quote_bid))
            or np.any(~np.isfinite(self.quote_ask))
            or np.any(~np.isfinite(self.quote_fit))
            or np.any(self.quote_bid < 0.0)
            or np.any(self.quote_ask <= self.quote_bid)
            or np.any(~np.isfinite(self.quote_log_volume))
            or np.any(~np.isfinite(self.quote_log_open_interest))
            or np.any(self.quote_log_volume < 0.0)
            or np.any(self.quote_log_open_interest < 0.0)
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
        return self.expiry_id

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
        return self.quote_bid

    @property
    def asks(self) -> np.ndarray:
        return self.quote_ask

    @property
    def fit_prices(self) -> np.ndarray:
        return self.quote_fit

    @property
    def log_volumes(self) -> np.ndarray:
        return self.quote_log_volume

    @property
    def log_open_interests(self) -> np.ndarray:
        return self.quote_log_open_interest

    @property
    def quote_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.trading_date.isoformat()}:{self.expiry_id[int(expiry_index)]}:"
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
    def spot_ref(self) -> float:
        return self.spot_reference

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
                self.expiry_ids,
                self.expiry_is_am,
                self.expiry_index,
                self.strikes,
                self.is_call,
                self.bids,
                self.asks,
                self.fit_prices,
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

        return 0.5 * (self.quote_ask - self.quote_bid)

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

        mids = 0.5 * (self.quote_bid + self.quote_ask)
        return mids + np.where(
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


@version("1.1", dependencies=[DailyFrontDay, _market_fingerprint])
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
        f"{day.trading_date.isoformat()}:{day.expiry_id[int(expiry_index)]}:"
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
            day.expiry_id,
            day.expiry_is_am,
            day.quote_expiry_index,
            day.quote_strike,
            day.quote_is_call,
            day.quote_bid,
            day.quote_ask,
            day.quote_fit,
            day.expiry_atm_annual_vol,
            day.model_atm_total_vol,
            day.expiry_strike_lower,
            day.expiry_strike_upper,
        ),
    )


@version(
    "1.2",
    dependencies=[
        DailyFrontDay,
        DailyFrontMarketConfig,
        DailyFrontGeometryConfig,
        _interpolate_atm_vols,
        _market_fingerprint,
        _one_value,
        _days_since_epoch,
        canonical_sha256,
        smooth_sanos_pricer,
        NMS,
    ],
)
def project_daily_front(
    frame: pl.DataFrame,
    *,
    source_dataset: ArtifactPointer,
    market: DailyFrontMarketConfig,
    geometry: DailyFrontGeometryConfig,
) -> DailyFrontDay:
    """Project one fitted internal partition into the exact daily-front universe."""

    if frame.schema != pl.Schema(FITTED_SCHEMA):
        raise DailyDataError("daily-front source does not have the fitted-day schema")
    if frame.is_empty():
        raise DailyDataError("A typed-empty partition has no daily-front market")
    ticker = str(_one_value(frame, "ticker"))
    trading_date = _one_value(frame, "trading_date")
    if not isinstance(trading_date, date):
        raise DailyDataError("trading_date must be a date")
    is_half_day = bool(_one_value(frame, "is_half_day"))
    spot_reference = float(_one_value(frame, "spot_reference"))
    smoothness = float(_one_value(frame, "smoothness"))
    if smoothness != market.smoothness:
        raise DailyDataError(
            f"Daily smoothness {smoothness} differs from Experiment {market.smoothness}"
        )

    eligible = frame.filter(pl.col("options_filter_eligible")).sort(
        "time_to_expiry_years", "expiry_id", "normalized_strike", maintain_order=True
    )
    if eligible.is_empty():
        raise DailyDataError("A populated source day has no options_filter_eligible rows")
    expiry_rows = eligible.unique(
        subset=["time_to_expiry_years", "expiry_id"], maintain_order=True
    ).sort("time_to_expiry_years", "expiry_id", maintain_order=True)
    all_time_y = expiry_rows.get_column("time_to_expiry_years").cast(pl.Float64).to_numpy()
    if len(all_time_y) > 1 and np.any(np.diff(all_time_y) <= 0.0):
        raise DailyDataError("Daily-front input requires one strictly ordered identity per expiry")

    pricer = smooth_sanos_pricer(frame)
    pricer_expiries = np.asarray(pricer.expiries, dtype=np.float64)
    pricer_variances = np.asarray(pricer.variances, dtype=np.float64)
    all_atm_vol = _interpolate_atm_vols(all_time_y, pricer_expiries, pricer_variances)
    model_business_dtes = np.asarray(geometry.model_business_dtes, dtype=np.float64)
    model_time_y = model_business_dtes / DAYS_PER_YEAR
    model_atm_vol = _interpolate_atm_vols(model_time_y, pricer_expiries, pricer_variances)
    model_atm_total_vol = model_atm_vol * np.sqrt(model_time_y)
    bounds = np.asarray(
        NMS.from_nms(
            np.asarray(geometry.nms_range, dtype=np.float64)[None, :],
            all_time_y[:, None],
            volatility=all_atm_vol[:, None],
            epsilon_volatility=float(model_atm_vol[0]),
        ),
        dtype=np.float64,
    )

    expiry_time: list[float] = []
    expiry_dates: list[date] = []
    expiry_ids: list[int] = []
    expiry_is_am: list[bool] = []
    expiry_atm: list[float] = []
    expiry_lower: list[float] = []
    expiry_upper: list[float] = []
    quote_expiry: list[int] = []
    quote_strike: list[float] = []
    quote_side: list[bool] = []
    quote_bid: list[float] = []
    quote_ask: list[float] = []
    quote_fit: list[float] = []
    quote_volume: list[float] = []
    quote_open_interest: list[float] = []
    quote_ids: list[str] = []
    active_european_books = 0
    horizon_eligible_books = 0
    nms_eligible_books = 0
    maximum_time = float(model_time_y[-1])

    for input_expiry_index, expiry_record in enumerate(expiry_rows.iter_rows(named=True)):
        expiry_id = int(expiry_record["expiry_id"])
        time_y = float(expiry_record["time_to_expiry_years"])
        business_dte = time_y * DAYS_PER_YEAR
        group = eligible.filter(pl.col("expiry_id") == expiry_id).sort(
            "normalized_strike", maintain_order=True
        )
        european = ~group.get_column("is_american").to_numpy().astype(bool)
        fit_available = group.get_column("smooth_mid_available").to_numpy().astype(bool)
        strikes = group.get_column("normalized_strike").cast(pl.Float64).to_numpy()
        within_nms = (strikes >= bounds[input_expiry_index, 0]) & (
            strikes <= bounds[input_expiry_index, 1]
        )
        excluded = any(
            abs(business_dte - value) <= 1.0e-4 for value in market.excluded_business_dtes
        )
        within_horizon = not excluded and 0.0 <= time_y <= maximum_time + 1.0e-12
        output_expiry_index = len(expiry_time)
        expiry_quote_start = len(quote_strike)
        side_fields = (
            (
                True,
                "C",
                "call_active",
                "pure_market_call_bid",
                "pure_market_call_ask",
                "smooth_call_mid",
                "market_call_log_volume",
                "market_call_log_open_interest",
            ),
            (
                False,
                "P",
                "put_active",
                "pure_market_put_bid",
                "pure_market_put_ask",
                "smooth_put_mid",
                "market_put_log_volume",
                "market_put_log_open_interest",
            ),
        )
        for (
            is_call,
            side,
            active_column,
            bid_column,
            ask_column,
            fit_column,
            volume_column,
            oi_column,
        ) in side_fields:
            fit = group.get_column(fit_column).cast(pl.Float64).to_numpy()
            active = (
                group.get_column(active_column).to_numpy().astype(bool)
                & european
                & fit_available
                & np.isfinite(fit)
            )
            active_european_books += int(np.sum(active))
            if not within_horizon:
                continue
            horizon_eligible_books += int(np.sum(active))
            selected = active & within_nms
            nms_eligible_books += int(np.sum(selected))
            if not np.any(selected):
                continue
            selected_strikes = strikes[selected]
            count = len(selected_strikes)
            quote_expiry.extend([output_expiry_index] * count)
            quote_strike.extend(selected_strikes.tolist())
            quote_side.extend([is_call] * count)
            quote_bid.extend(
                group.get_column(bid_column).cast(pl.Float64).to_numpy()[selected].tolist()
            )
            quote_ask.extend(
                group.get_column(ask_column).cast(pl.Float64).to_numpy()[selected].tolist()
            )
            quote_fit.extend(fit[selected].tolist())
            quote_volume.extend(
                group.get_column(volume_column).cast(pl.Float64).to_numpy()[selected].tolist()
            )
            quote_open_interest.extend(
                group.get_column(oi_column).cast(pl.Float64).to_numpy()[selected].tolist()
            )
            quote_ids.extend(
                f"{trading_date.isoformat()}:{expiry_id}:{strike:.12g}:{side}"
                for strike in selected_strikes
            )
        if len(quote_strike) == expiry_quote_start:
            continue
        expiry_time.append(time_y)
        expiry_value = expiry_record["expiry_date"]
        if not isinstance(expiry_value, date):
            raise DailyDataError("expiry_date must be a date")
        expiry_dates.append(expiry_value)
        expiry_ids.append(expiry_id)
        expiry_is_am.append(bool(expiry_record["expiry_is_am"]))
        expiry_atm.append(float(all_atm_vol[input_expiry_index]))
        expiry_lower.append(float(bounds[input_expiry_index, 0]))
        expiry_upper.append(float(bounds[input_expiry_index, 1]))

    if not quote_strike:
        raise DailyDataError("A populated source day has no daily-front eligible books")

    quote_expiry_array = np.asarray(quote_expiry, dtype=np.int64)
    quote_strike_array = np.asarray(quote_strike, dtype=np.float64)
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
            price_expiry_y.append(expiry_time[int(expiry_index)])
            price_strike.append(float(strike))
        quote_price_index[index] = price_index

    expiry_time_array = np.asarray(expiry_time, dtype=np.float64)
    expiry_id_array = np.asarray(expiry_ids, dtype=np.int64)
    expiry_am_array = np.asarray(expiry_is_am, dtype=np.bool_)
    quote_side_array = np.asarray(quote_side, dtype=np.bool_)
    quote_bid_array = np.asarray(quote_bid, dtype=np.float64)
    quote_ask_array = np.asarray(quote_ask, dtype=np.float64)
    quote_fit_array = np.asarray(quote_fit, dtype=np.float64)
    quote_volume_array = np.asarray(quote_volume, dtype=np.float64)
    quote_oi_array = np.asarray(quote_open_interest, dtype=np.float64)
    expiry_atm_array = np.asarray(expiry_atm, dtype=np.float64)
    lower_array = np.asarray(expiry_lower, dtype=np.float64)
    upper_array = np.asarray(expiry_upper, dtype=np.float64)
    expiry_date_array = _days_since_epoch(expiry_dates)
    price_expiry_array = np.asarray(price_expiry_y, dtype=np.float64)
    price_strike_array = np.asarray(price_strike, dtype=np.float64)
    market_uid = canonical_sha256(
        {
            "source_dataset": source_dataset,
            "daily_configuration_id": str(_one_value(frame, "configuration_id")),
            "options_filter_configuration_id": str(
                _one_value(frame, "options_filter_configuration_id")
            ),
            "smooth_sanos_configuration_id": str(
                _one_value(frame, "smooth_sanos_configuration_id")
            ),
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
            expiry_id_array,
            expiry_am_array,
            quote_expiry_array,
            quote_strike_array,
            quote_side_array,
            quote_bid_array,
            quote_ask_array,
            quote_fit_array,
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
        spot_reference=spot_reference,
        market_uid=market_uid,
        market_fingerprint=fingerprint,
        expiry_time_y=expiry_time_array,
        expiry_date_days=expiry_date_array,
        expiry_id=expiry_id_array,
        expiry_is_am=expiry_am_array,
        expiry_atm_annual_vol=expiry_atm_array,
        expiry_strike_lower=lower_array,
        expiry_strike_upper=upper_array,
        model_atm_total_vol=np.asarray(model_atm_total_vol, dtype=np.float64),
        quote_expiry_index=quote_expiry_array,
        quote_price_index=quote_price_index,
        quote_strike=quote_strike_array,
        quote_is_call=quote_side_array,
        quote_bid=quote_bid_array,
        quote_ask=quote_ask_array,
        quote_fit=quote_fit_array,
        quote_log_volume=quote_volume_array,
        quote_log_open_interest=quote_oi_array,
        price_expiry_y=price_expiry_array,
        price_strike=price_strike_array,
        audit=EligibilityAudit(
            active_european_books=active_european_books,
            horizon_eligible_books=horizon_eligible_books,
            nms_eligible_books=nms_eligible_books,
            input_expiries=len(all_time_y),
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
