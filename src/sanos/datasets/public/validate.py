"""Self-contained schema, leakage, numerical, and capability validation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import polars as pl
from cdxcore.version import version

from sanos.artifacts import (
    COMPLETION_NAME,
    MANIFEST_NAME,
    ArtifactIntegrityError,
    ArtifactPointer,
    ArtifactStore,
    DatasetDistribution,
    DistributionStatus,
    ValidatedArtifact,
)
from sanos.config import canonical_json, canonical_sha256, canonical_value
from sanos.datasets.public.config import (
    PublicProjectionConfig,
    PublicReleaseConfig,
)
from sanos.datasets.public.documents import candidate_readme, release_readme
from sanos.datasets.public.project import (
    PublicProjectionError,
    _excluded_public_business_dte,
    _inflate_public_day,
    _model_iv_fields,
)
from sanos.datasets.public.schema import (
    PUBLIC_VALIDATION_VERSION,
    public_capabilities,
    public_polars_schema,
    public_schema_document,
)
from sanos.model.tools import SanosPricer
from sanos.widb.calendar import (
    NMS,
    business_clock_policy,
    business_dates,
    count_business_days_between,
    is_half_day,
    normalized_strike_policy,
)


class PublicDatasetValidationError(ArtifactIntegrityError):
    """Raised when an intact public Dataset violates its semantic contract."""


@version("3.0")
@dataclass(frozen=True, slots=True)
class PublicDatasetValidationSummary:
    """Recomputed public-only validation evidence."""

    date_count: int
    expiry_count: int
    book_count: int
    smooth_state_count: int
    dlv_state_count: int
    omitted_date_count: int
    maximum_density_mass_residual: float
    maximum_density_moment_residual: float
    maximum_calendar_call_residual: float
    outcome: str = "passed"

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], canonical_value(self))


@version("1.0")
@dataclass(frozen=True, slots=True)
class PublicDataset:
    """Self-contained reader which never consults the source Dataset or WRDS."""

    artifact: ValidatedArtifact
    config: PublicProjectionConfig
    distribution: DistributionStatus
    release_config: PublicReleaseConfig | None = None

    def scan(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.artifact.path / "data.parquet", hive_partitioning=False)

    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.artifact.path / "data.parquet", hive_partitioning=False)


def _fail(message: str) -> NoReturn:
    raise PublicDatasetValidationError(message)


@version("1.0")
def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"Missing or unsafe {label}: {path}")
    try:
        source = path.read_text(encoding="utf-8")
        value = json.loads(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicDatasetValidationError(f"Cannot decode {label}: {exc}") from exc
    if not isinstance(value, dict) or source != canonical_json(value) + "\n":
        _fail(f"{label} must be a canonical JSON object")
    return cast(dict[str, Any], value)


@version("1.0")
def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label} contains a non-finite float")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")


@version(
    "2.0",
    dependencies=[canonical_sha256, public_polars_schema, public_schema_document],
)
def public_frame_logical_sha256(frame: pl.DataFrame) -> str:
    """Hash the exact typed, ordered public logical content."""

    if frame.schema != public_polars_schema():
        _fail("Cannot hash a frame outside the exact public schema")
    return canonical_sha256(
        {
            "schema": public_schema_document()["physical_schema"],
            "rows": frame.to_dicts(),
        }
    )


@version("2.0", dependencies=[canonical_sha256])
def public_return_values_sha256(frame: pl.DataFrame) -> str:
    """Bind the ordered public return vector without exposing private inputs."""

    return canonical_sha256(
        [
            {
                "trading_date": row["trading_date"],
                "driftless_spot_return": row["driftless_spot_return"],
            }
            for row in frame.filter(pl.col("record_kind") == "day")
            .select("trading_date", "driftless_spot_return")
            .iter_rows(named=True)
        ]
    )


@version("1.0", dependencies=[PublicProjectionConfig, canonical_sha256])
def public_coverage_sha256(
    dates: tuple[date, ...],
    omitted_dates: tuple[Mapping[str, Any], ...],
    config: PublicProjectionConfig,
) -> str:
    """Bind the complete configured NYSE coverage and exact omissions."""

    return canonical_sha256(
        {
            "coverage_start_date": config.coverage_start_date,
            "coverage_end_date": config.coverage_end_date,
            "published_dates": dates,
            "omitted_dates": omitted_dates,
        }
    )


@version("1.0")
def _validate_density(
    support: np.ndarray,
    density: np.ndarray,
    label: str,
) -> tuple[float, float]:
    if (
        support.ndim != 1
        or density.shape != support.shape
        or len(support) < 3
        or np.any(~np.isfinite(support))
        or np.any(~np.isfinite(density))
        or np.any(support <= 0.0)
        or np.any(np.diff(support) <= 0.0)
        or np.any(density < -1.0e-7)
    ):
        _fail(f"{label} has invalid support or density")
    mass = abs(float(np.sum(density)) - 1.0)
    moment = abs(float(support @ density) - 1.0)
    if mass > 1.0e-4 or moment > 1.0e-4:
        _fail(f"{label} density residuals are mass={mass}, moment={moment}")
    return mass, moment


@version("initial-surface-calendar-accepted-maximum-5e-5-v1")
def _public_calendar_call_tolerance() -> float:
    """Return the broadest accepted producer solver/status calendar tolerance."""

    return 5.0e-5


@version("2.0", dependencies=[_public_calendar_call_tolerance])
def _validate_calendar_calls(pricer: SanosPricer, label: str) -> float:
    """Check adjacent support unions at the sanctioned producer maximum."""

    maximum = 0.0
    for index in range(1, len(pricer.expiries)):
        support = np.union1d(
            np.asarray(pricer.xstrikes[index - 1], dtype=np.float64),
            np.asarray(pricer.xstrikes[index], dtype=np.float64),
        )
        previous = np.asarray(pricer(float(pricer.expiries[index - 1]), support))
        current = np.asarray(pricer(float(pricer.expiries[index]), support))
        residual = float(np.max(np.maximum(previous - current, 0.0)))
        if not math.isfinite(residual):
            _fail(f"{label} has a non-finite full-support calendar residual")
        maximum = max(maximum, residual)
    tolerance = _public_calendar_call_tolerance()
    if maximum > tolerance:
        _fail(
            f"{label} normalized calls violate full-support calendar monotonicity: "
            f"{maximum} > {tolerance}"
        )
    return maximum


@version(
    "4.0",
    dependencies=[SanosPricer, _validate_calendar_calls, _validate_density],
)
def _state_pricer(
    state: Mapping[str, Any],
    expiries: list[Mapping[str, Any]],
    *,
    smooth: bool,
) -> tuple[SanosPricer, float, float, float]:
    expected = {"smoothness", "model_expiries"} if smooth else {"model_expiries"}
    if set(state) != expected or not isinstance(state.get("model_expiries"), list):
        _fail("Public model state fields changed")
    items = cast(list[Mapping[str, Any]], state["model_expiries"])
    if not items:
        _fail("Public model state is empty")
    times: list[float] = []
    variances: list[float] = []
    supports: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    maximum_mass = 0.0
    maximum_moment = 0.0
    previous_time = 0.0
    previous_expiry_index = -1
    for index, item in enumerate(items):
        required = {
            "expiry_index",
            "time_increment_years",
            "kernel_input_total_variance" if smooth else "input_total_variance",
            "normalized_support",
            "density_mass",
        }
        if set(item) != required:
            _fail(f"Public model state expiry {index} fields changed")
        expiry_index = int(item["expiry_index"])
        if expiry_index < 0 or expiry_index >= len(expiries):
            _fail("Public model state expiry index is outside union geometry")
        if expiry_index <= previous_expiry_index:
            _fail("Public model state expiry indices must be strictly increasing")
        previous_expiry_index = expiry_index
        increment = float(item["time_increment_years"])
        if not math.isfinite(increment) or increment <= 0.0:
            _fail("Public model time increment must be positive and finite")
        current_time = previous_time + increment
        union_time = float(expiries[expiry_index]["time_to_expiry_years"])
        if not math.isclose(current_time, union_time, rel_tol=0.0, abs_tol=2.0e-8):
            _fail("Public model increments do not reconstruct union expiry time")
        previous_time = current_time
        variance_name = (
            "kernel_input_total_variance" if smooth else "input_total_variance"
        )
        variance = float(item[variance_name])
        if not math.isfinite(variance) or variance <= 0.0:
            _fail("Public model input variance must be positive and finite")
        support = np.asarray(item["normalized_support"], dtype=np.float64)
        density = np.asarray(item["density_mass"], dtype=np.float64)
        mass, moment = _validate_density(support, density, f"model expiry {index}")
        maximum_mass = max(maximum_mass, mass)
        maximum_moment = max(maximum_moment, moment)
        times.append(current_time)
        variances.append(variance)
        supports.append(support)
        densities.append(density)
    smoothness = float(state["smoothness"]) if smooth else None
    if smooth and (not math.isfinite(cast(float, smoothness)) or cast(float, smoothness) <= 0.0):
        _fail("Public SANOS smoothness must be positive and finite")
    pricer = SanosPricer(
        smoothness=smoothness,
        expiries=np.asarray(times, dtype=np.float64),
        variances=np.asarray(variances, dtype=np.float64),
        xstrikes=supports,
        xdensities=densities,
    )
    grid = np.asarray([0.7, 1.0, 1.3], dtype=np.float64)
    for model_time in times:
        calls = np.asarray(pricer(model_time, grid), dtype=np.float64)
        puts = calls - 1.0 + grid
        if np.any(~np.isfinite(calls)) or np.any(~np.isfinite(puts)):
            _fail("Public pricing capability produced non-finite values")
    _, atm_vol = pricer(times[0], np.asarray([1.0]), and_vols=True)
    if np.any(~np.isfinite(atm_vol)) or np.any(atm_vol <= 0.0):
        _fail("Public model ATM implied-volatility capability failed")
    calendar = _validate_calendar_calls(
        pricer, "Public smooth SANOS state" if smooth else "Public DLV state"
    )
    return pricer, maximum_mass, maximum_moment, calendar


@version(
    "3.1",
    dependencies=[
        _excluded_public_business_dte,
        _model_iv_fields,
        normalized_strike_policy,
    ],
)
def _validate_books(
    books: list[Mapping[str, Any]],
    expiries: list[Mapping[str, Any]],
    epsilon_atm_vol: float,
    config: PublicProjectionConfig,
    smooth_pricer: SanosPricer,
    dlv_pricer: SanosPricer,
) -> None:
    if not books:
        _fail("Public row has no books")
    keys: list[tuple[int, int, float]] = []
    for item in books:
        if set(item) != {
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
        }:
            _fail("Public book fields changed")
        expiry_index = int(item["expiry_index"])
        if expiry_index < 0 or expiry_index >= len(expiries):
            _fail("Public book expiry index is outside union geometry")
        if not isinstance(item["is_call"], bool):
            _fail("Public book side is not Boolean")
        strike = float(item["normalized_strike"])
        half_spread = float(item["half_spread"])
        if (
            not math.isfinite(strike)
            or strike <= 0.0
            or not math.isfinite(half_spread)
            or half_spread <= 0.0
        ):
            _fail("Public book strike and half-spread must be positive and finite")
        expiry = expiries[expiry_index]
        business_dte = float(expiry["business_dte"])
        if _excluded_public_business_dte(business_dte, config):
            _fail("Public book refers to an excluded or out-of-horizon expiry")
        bounds = np.asarray(
            NMS.from_nms(
                np.asarray(config.nms_range, dtype=np.float64),
                float(expiry["time_to_expiry_years"]),
                volatility=float(expiry["nms_atm_annual_volatility"]),
                epsilon_volatility=epsilon_atm_vol,
            ),
            dtype=np.float64,
        )
        if strike < bounds[0] or strike > bounds[1]:
            _fail("Public book lies outside the inclusive configured NMS range")
        time_years = float(expiry["time_to_expiry_years"])
        for label, pricer in (("sanos", smooth_pricer), ("dlv", dlv_pricer)):
            call_price = float(pricer(time_years, np.asarray([strike]))[0])
            expected = _model_iv_fields(
                label, strike, call_price, half_spread, time_years
            )
            for field, expected_value in expected.items():
                observed = item[field]
                if expected_value is None:
                    if observed is not None:
                        _fail(f"Public {field} must be null for a non-identifiable bound")
                elif observed is None or not math.isclose(
                    float(observed), expected_value, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    _fail(f"Public {field} is inconsistent with model and half-spread")
        keys.append((expiry_index, 0 if item["is_call"] else 1, strike))
    if keys != sorted(keys):
        _fail("Public books are not ordered by expiry, call-before-put, and strike")
    if len(keys) != len(set(keys)):
        _fail("Public books contain a duplicate expiry/side/strike key")


@version("1.0")
def _validate_market_quality(row: Mapping[str, Any]) -> None:
    for label in ("sanos", "dlv"):
        percent = float(row[f"{label}_market_percent_books_outside"])
        mean = float(row[f"{label}_market_conditional_mean_excess_halfspreads"])
        median = float(row[f"{label}_market_conditional_median_excess_halfspreads"])
        if (
            not all(math.isfinite(value) for value in (percent, mean, median))
            or percent < 0.0
            or percent > 100.0
            or mean < 0.0
            or median < 0.0
            or (percent == 0.0 and (mean != 0.0 or median != 0.0))
            or (percent > 0.0 and (mean <= 0.0 or median <= 0.0))
        ):
            _fail(f"Public {label} market-quality aggregates are invalid")


@version("1.1", dependencies=[business_dates, count_business_days_between])
def _validate_coverage_evidence(
    evidence: Mapping[str, Any],
    dates: tuple[date, ...],
    config: PublicProjectionConfig,
) -> tuple[Mapping[str, Any], ...]:
    omitted_value = evidence["omitted_dates"]
    returns_value = evidence["return_evidence"]
    if not isinstance(omitted_value, list) or not isinstance(returns_value, list):
        _fail("validation.json coverage evidence must contain lists")
    omitted: list[Mapping[str, Any]] = []
    omitted_dates: list[date] = []
    for record in omitted_value:
        if not isinstance(record, Mapping) or set(record) != {"trading_date", "reason"}:
            _fail("validation.json contains malformed omitted-date evidence")
        if not isinstance(record["trading_date"], str) or not isinstance(record["reason"], str):
            _fail("validation.json omitted-date values must be strings")
        try:
            omitted_date = date.fromisoformat(record["trading_date"])
        except ValueError as exc:
            raise PublicDatasetValidationError(
                "validation.json contains an invalid omitted date"
            ) from exc
        if not record["reason"]:
            _fail("validation.json omitted-date reason cannot be empty")
        omitted.append(record)
        omitted_dates.append(omitted_date)
    if omitted_dates != sorted(set(omitted_dates)) or set(omitted_dates) & set(dates):
        _fail("validation.json omitted dates are duplicate, unordered, or published")
    expected_omissions = tuple(
        item.as_dict() for item in config.expected_typed_empty_omissions
    )
    if tuple(omitted) != expected_omissions:
        _fail("validation.json omissions differ from the identity-owned omission policy")
    planned_dates = business_dates(config.coverage_start_date, config.coverage_end_date)
    if tuple(sorted((*dates, *omitted_dates))) != planned_dates:
        _fail("Published and omitted dates do not exactly cover the configured NYSE sessions")
    if len(returns_value) != max(len(dates) - 1, 0):
        _fail("validation.json return evidence does not cover every non-first row")
    for index, record in enumerate(returns_value, start=1):
        if not isinstance(record, Mapping) or set(record) != {
            "trading_date",
            "previous_trading_date",
            "business_sessions",
            "discount_curve_validated",
        }:
            _fail("validation.json contains malformed return evidence")
        if (
            record["trading_date"] != dates[index].isoformat()
            or record["previous_trading_date"] != dates[index - 1].isoformat()
            or record["discount_curve_validated"] is not True
        ):
            _fail("validation.json return evidence date or discount proof changed")
        sessions = record["business_sessions"]
        expected_sessions = int(count_business_days_between(dates[index - 1], dates[index]))
        if (
            isinstance(sessions, bool)
            or not isinstance(sessions, int)
            or sessions != expected_sessions
        ):
            _fail("validation.json return horizon differs from the NYSE session count")
    return tuple(omitted)


@version(
    "3.0",
    dependencies=[
        PublicDatasetValidationSummary,
        PublicProjectionConfig,
        _inflate_public_day,
        _finite_tree,
        _state_pricer,
        _validate_calendar_calls,
        _validate_market_quality,
        _validate_books,
        business_clock_policy,
        is_half_day,
        public_polars_schema,
    ],
)
def validate_public_frame(
    frame: pl.DataFrame,
    config: PublicProjectionConfig,
    *,
    omitted_dates: tuple[Mapping[str, Any], ...] = (),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> PublicDatasetValidationSummary:
    """Execute all advertised capabilities using only the public frame."""

    if frame.schema != public_polars_schema():
        _fail(
            f"Public flat scalar schema changed: expected={public_polars_schema()}, "
            f"observed={frame.schema}"
        )
    if frame.is_empty():
        _fail("Public Dataset frame is empty")
    day_rows = frame.filter(pl.col("record_kind") == "day")
    dates = tuple(cast(date, value) for value in day_rows["trading_date"].to_list())
    if dates != tuple(sorted(set(dates))):
        _fail("Public dates must be unique and strictly increasing")
    try:
        day_frames = tuple(
            frame.filter(pl.col("trading_date") == trading_date)
            for trading_date in dates
        )
        rows = [_inflate_public_day(day_frame) for day_frame in day_frames]
    except PublicProjectionError as exc:
        raise PublicDatasetValidationError(str(exc)) from exc
    if not day_frames or not pl.concat(day_frames, how="vertical").equals(frame):
        _fail("Public dates are interleaved or outside canonical order")
    if rows[0]["driftless_spot_return"] is not None:
        _fail("The first public driftless return must be null")
    if any(row["driftless_spot_return"] is None for row in rows[1:]):
        _fail("Only the first public driftless return may be null")
    if any(float(row["driftless_spot_return"]) <= -1.0 for row in rows[1:]):
        _fail("Public driftless returns must be greater than -1")

    expiry_count = 0
    book_count = 0
    smooth_count = 0
    dlv_count = 0
    maximum_mass = 0.0
    maximum_moment = 0.0
    maximum_calendar = 0.0
    started = time.monotonic()
    for completed, row in enumerate(rows, start=1):
        _finite_tree(row, f"row[{completed - 1}]")
        trading_date = cast(date, row["trading_date"])
        if bool(row["is_half_day"]) != is_half_day(trading_date):
            _fail(f"{trading_date}: trading-session half-day flag changed")
        expiries = cast(list[Mapping[str, Any]], row["expiries"])
        if not expiries:
            _fail(f"{trading_date}: public expiry geometry is empty")
        times: list[float] = []
        for index, expiry in enumerate(expiries):
            if set(expiry) != {
                "expiry_index",
                "expiry_date",
                "business_dte",
                "calendar_dte",
                "is_am",
                "time_to_expiry_years",
                "nms_atm_annual_volatility",
            }:
                _fail("Public expiry fields changed")
            if int(expiry["expiry_index"]) != index:
                _fail("Public expiry indices are not contiguous from zero")
            expiry_date = cast(date, expiry["expiry_date"])
            if int(expiry["calendar_dte"]) != (expiry_date - trading_date).days:
                _fail("Public calendar DTE changed")
            business_dte = float(expiry["business_dte"])
            time_years = float(expiry["time_to_expiry_years"])
            expected_business_dte = float(
                count_business_days_between(trading_date, expiry_date)
            ) - (1.0 - config.overnight_variance_fraction if expiry["is_am"] else 0.0)
            if not math.isclose(
                business_dte,
                expected_business_dte,
                rel_tol=0.0,
                abs_tol=2.0e-5,
            ):
                _fail("Public business DTE differs from the configured AM/PM clock")
            if not math.isclose(
                time_years,
                business_dte / business_clock_policy()[0],
                rel_tol=0.0,
                abs_tol=2.0e-8,
            ):
                _fail("Public model time differs from business_dte/252")
            if (
                not math.isfinite(float(expiry["nms_atm_annual_volatility"]))
                or float(expiry["nms_atm_annual_volatility"]) <= 0.0
            ):
                _fail("Public expiry ATM geometry must be positive and finite")
            times.append(time_years)
        if any(right <= left for left, right in zip(times, times[1:])):
            _fail("Public expiry times are not strictly increasing")
        epsilon_atm = float(row["nms_epsilon_atm_annual_volatility"])
        if not math.isfinite(epsilon_atm) or epsilon_atm <= 0.0:
            _fail("Public NMS epsilon ATM volatility must be positive and finite")
        _validate_market_quality(row)
        smooth_state = cast(Mapping[str, Any], row["smooth_sanos"])
        dlv_state = cast(Mapping[str, Any], row["dlv"])
        smooth_pricer, smooth_mass, smooth_moment, smooth_calendar = _state_pricer(
            smooth_state, expiries, smooth=True
        )
        dlv_pricer, dlv_mass, dlv_moment, dlv_calendar = _state_pricer(
            dlv_state, expiries, smooth=False
        )
        if smooth_pricer.smoothness != config.smoothness:
            _fail("Public SANOS smoothness differs from canonical configuration")
        smooth_indices = {
            int(item["expiry_index"])
            for item in cast(list[Mapping[str, Any]], smooth_state["model_expiries"])
        }
        dlv_indices = {
            int(item["expiry_index"])
            for item in cast(list[Mapping[str, Any]], dlv_state["model_expiries"])
        }
        expected_indices = set(range(len(expiries)))
        if dlv_indices != expected_indices:
            _fail("Public DLV state must cover the complete union expiry geometry")
        if not smooth_indices.issubset(dlv_indices):
            _fail("Public smooth SANOS expiries must be a subset of DLV expiries")
        books = cast(list[Mapping[str, Any]], row["books"])
        try:
            _validate_books(
                books,
                expiries,
                epsilon_atm,
                config,
                smooth_pricer,
                dlv_pricer,
            )
        except PublicProjectionError as exc:
            raise PublicDatasetValidationError(str(exc)) from exc
        smooth_times = np.asarray(smooth_pricer.expiries, dtype=np.float64)
        smooth_variances = np.asarray(smooth_pricer.variances, dtype=np.float64)

        def expected_atm(time_years: float) -> float:
            clipped = float(np.clip(time_years, smooth_times[0], smooth_times[-1]))
            return float(
                np.sqrt(np.interp(clipped, smooth_times, smooth_variances) / clipped)
            )

        expected_epsilon_atm = expected_atm(
            config.nms_epsilon_business_dte / business_clock_policy()[0]
        )
        if not math.isclose(
            epsilon_atm,
            expected_epsilon_atm,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            _fail("Public NMS epsilon ATM volatility is not smooth-SANOS-derived")
        for expiry in expiries:
            if not math.isclose(
                float(expiry["nms_atm_annual_volatility"]),
                expected_atm(float(expiry["time_to_expiry_years"])),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                _fail("Public expiry ATM volatility is not smooth-SANOS-derived")
        # One off-grid comparison proves both reconstructed common-surface
        # pricers are executable without any internal Dataset field.
        query_time = min(float(smooth_pricer.expiries[-1]), float(dlv_pricer.expiries[-1]))
        query = np.asarray([0.83, 1.07, 1.41], dtype=np.float64)
        if np.any(~np.isfinite(smooth_pricer(query_time, query))) or np.any(
            ~np.isfinite(dlv_pricer(query_time, query))
        ):
            _fail("Public off-grid pricing capability failed")
        expiry_count += len(expiries)
        book_count += len(books)
        smooth_count += len(cast(list[Any], smooth_state["model_expiries"]))
        dlv_count += len(cast(list[Any], dlv_state["model_expiries"]))
        maximum_mass = max(maximum_mass, smooth_mass, dlv_mass)
        maximum_moment = max(maximum_moment, smooth_moment, dlv_moment)
        maximum_calendar = max(maximum_calendar, smooth_calendar, dlv_calendar)
        if progress is not None:
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            progress(
                {
                    "event": "public_validation_progress",
                    "elapsed_seconds": elapsed,
                    "completed": completed,
                    "total": len(rows),
                    "failed": 0,
                    "estimated_remaining_seconds": (
                        (len(rows) - completed) / rate if rate > 0.0 else None
                    ),
                }
            )
    return PublicDatasetValidationSummary(
        date_count=len(rows),
        expiry_count=expiry_count,
        book_count=book_count,
        smooth_state_count=smooth_count,
        dlv_state_count=dlv_count,
        omitted_date_count=len(omitted_dates),
        maximum_density_mass_residual=maximum_mass,
        maximum_density_moment_residual=maximum_moment,
        maximum_calendar_call_residual=maximum_calendar,
    )


@version(
    "2.0",
    dependencies=[
        PublicDatasetValidationSummary,
        public_coverage_sha256,
        public_frame_logical_sha256,
        public_return_values_sha256,
    ],
)
def public_validation_document(
    frame: pl.DataFrame,
    summary: PublicDatasetValidationSummary,
    config: PublicProjectionConfig,
    *,
    omitted_dates: tuple[Mapping[str, Any], ...],
    return_evidence: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Create the exact self-contained validation and projection evidence."""

    dates = tuple(
        cast(date, value)
        for value in frame.filter(pl.col("record_kind") == "day")[
            "trading_date"
        ].to_list()
    )
    return {
        "schema": PUBLIC_VALIDATION_VERSION,
        "outcome": "passed",
        "summary": summary.as_dict(),
        "omitted_dates": list(omitted_dates),
        "return_evidence": list(return_evidence),
        "return_values_sha256": public_return_values_sha256(frame),
        "logical_content_sha256": public_frame_logical_sha256(frame),
        "coverage_sha256": public_coverage_sha256(dates, omitted_dates, config),
    }


@version(
    "1.0",
    dependencies=[
        DistributionStatus,
        PublicDatasetValidationSummary,
        PublicProjectionConfig,
        public_capabilities,
    ],
)
def public_manifest_metadata(
    config: PublicProjectionConfig,
    summary: PublicDatasetValidationSummary,
    validation_document: Mapping[str, Any],
    *,
    distribution: DistributionStatus,
    approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the closed recomputable public manifest metadata contract."""

    metadata: dict[str, Any] = {
        "distribution": distribution.value,
        "source_dataset": config.source_dataset.as_dict(),
        "coverage_start_date": config.coverage_start_date.isoformat(),
        "coverage_end_date": config.coverage_end_date.isoformat(),
        "date_count": summary.date_count,
        "expiry_count": summary.expiry_count,
        "book_count": summary.book_count,
        "omitted_date_count": summary.omitted_date_count,
        "omitted_dates": validation_document["omitted_dates"],
        "validation_outcome": summary.outcome,
        "return_values_sha256": validation_document["return_values_sha256"],
        "logical_content_sha256": validation_document["logical_content_sha256"],
        "coverage_sha256": validation_document["coverage_sha256"],
        "capabilities": public_capabilities(),
    }
    if distribution is DistributionStatus.PUBLIC:
        if approval is None:
            _fail("Public release metadata requires approval evidence")
        metadata["approval"] = dict(approval)
    elif approval is not None:
        _fail("Candidate metadata cannot contain release approval")
    return metadata


@version("1.0")
def _payload_names(path: Path) -> set[str]:
    names: set[str] = set()
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            _fail(f"Public payload contains a symlink: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            _fail(f"Public payload contains a special file: {candidate}")
        names.add(candidate.relative_to(path).as_posix())
    return names - {MANIFEST_NAME, COMPLETION_NAME}


@version(
    "1.0",
    dependencies=[
        DistributionStatus,
        PublicDatasetValidationSummary,
        PublicProjectionConfig,
        PublicReleaseConfig,
        _payload_names,
        _read_json,
        _validate_coverage_evidence,
        candidate_readme,
        public_capabilities,
        public_coverage_sha256,
        public_frame_logical_sha256,
        public_manifest_metadata,
        public_return_values_sha256,
        public_schema_document,
        release_readme,
        validate_public_frame,
    ],
)
def validate_public_payload(
    path: Path,
    artifact_id: str,
    config: PublicProjectionConfig,
    *,
    distribution: DistributionStatus,
    release_config: PublicReleaseConfig | None = None,
    semantic: bool = True,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[PublicDatasetValidationSummary | None, dict[str, Any]]:
    """Validate a staged or committed payload before trusting its manifest."""

    candidate_files = {
        "README.md",
        "capabilities.json",
        "config.json",
        "data.parquet",
        "schema.json",
        "validation.json",
    }
    release_files = candidate_files | {"LICENSE.txt", "approval.json"}
    expected_files = (
        release_files if distribution is DistributionStatus.PUBLIC else candidate_files
    )
    actual_files = _payload_names(path)
    if actual_files != expected_files:
        _fail(
            "Public payload files changed: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    readme_path = path / "README.md"
    if readme_path.is_symlink() or not readme_path.is_file():
        _fail("Missing or unsafe public README.md")
    if distribution is DistributionStatus.PUBLIC_CANDIDATE:
        if release_config is not None:
            _fail("Candidate cannot carry a release configuration")
        expected_config = config.as_dict()
        expected_readme = candidate_readme(config, artifact_id)
        approval = None
    elif distribution is DistributionStatus.PUBLIC:
        if release_config is None or release_config.projection != config:
            _fail("Public release has no matching release configuration")
        expected_config = release_config.as_dict()
        expected_readme = release_readme(release_config, artifact_id)
        approval = release_config.approval.as_dict()
        if _read_json(path / "approval.json", "approval.json") != approval:
            _fail("approval.json differs from the immutable release approval")
        license_path = path / "LICENSE.txt"
        if license_path.is_symlink() or not license_path.is_file():
            _fail("Missing or unsafe LICENSE.txt")
        if (
            hashlib.sha256(license_path.read_bytes()).hexdigest()
            != release_config.approval.license_sha256
        ):
            _fail("LICENSE.txt checksum differs from the approved license checksum")
    else:
        _fail("Public payload distribution must be candidate or public")
    if readme_path.read_text(encoding="utf-8") != expected_readme:
        _fail("README.md differs from the deterministic public document")
    if _read_json(path / "config.json", "config.json") != expected_config:
        _fail("config.json differs from the immutable public configuration")
    if _read_json(path / "schema.json", "schema.json") != public_schema_document():
        _fail("schema.json differs from the immutable public schema contract")
    capabilities = _read_json(path / "capabilities.json", "capabilities.json")
    if capabilities != public_capabilities():
        _fail("capabilities.json differs from the closed public capability contract")
    evidence = _read_json(path / "validation.json", "validation.json")
    required_evidence = {
        "schema",
        "outcome",
        "summary",
        "omitted_dates",
        "return_evidence",
        "return_values_sha256",
        "logical_content_sha256",
        "coverage_sha256",
    }
    if set(evidence) != required_evidence:
        _fail("validation.json fields changed")
    if evidence["schema"] != PUBLIC_VALIDATION_VERSION or evidence["outcome"] != "passed":
        _fail("validation.json is not a passed current validation record")
    frame = pl.read_parquet(path / "data.parquet", hive_partitioning=False)
    dates = tuple(
        cast(date, value)
        for value in frame.filter(pl.col("record_kind") == "day")[
            "trading_date"
        ].to_list()
    )
    omitted = _validate_coverage_evidence(evidence, dates, config)
    if evidence["return_values_sha256"] != public_return_values_sha256(frame):
        _fail("Public return vector differs from immutable projection evidence")
    if evidence["logical_content_sha256"] != public_frame_logical_sha256(frame):
        _fail("Public logical content differs from immutable projection evidence")
    if (
        release_config is not None
        and evidence["logical_content_sha256"]
        != release_config.approval.candidate_logical_content_sha256
    ):
        _fail("Public release logical content differs from the approved candidate")
    if evidence["coverage_sha256"] != public_coverage_sha256(dates, omitted, config):
        _fail("Public coverage differs from immutable projection evidence")
    if not semantic:
        return None, public_manifest_metadata(
            config,
            PublicDatasetValidationSummary(**evidence["summary"]),
            evidence,
            distribution=distribution,
            approval=approval,
        )
    summary = validate_public_frame(
        frame,
        config,
        omitted_dates=omitted,
        progress=progress,
    )
    if evidence["summary"] != summary.as_dict():
        _fail("validation.json summary differs from public-only recomputation")
    return summary, public_manifest_metadata(
        config,
        summary,
        evidence,
        distribution=distribution,
        approval=approval,
    )


@version(
    "1.1",
    dependencies=[PublicDataset, PublicProjectionConfig, PublicReleaseConfig, _read_json],
)
def open_public_dataset(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
) -> PublicDataset:
    """Open a candidate/release shallowly without touching its internal source."""

    validated = store.validate(artifact) if isinstance(artifact, ArtifactPointer) else artifact
    distribution_value = validated.manifest.get("distribution")
    if not isinstance(distribution_value, Mapping) or distribution_value.get("status") not in {
        DistributionStatus.PUBLIC_CANDIDATE.value,
        DistributionStatus.PUBLIC.value,
    }:
        _fail("Requested artifact is not a public candidate or public release")
    distribution = DistributionStatus(distribution_value["status"])
    config_value = _read_json(validated.path / "config.json", "config.json")
    if canonical_json(config_value) != canonical_json(validated.manifest["configuration"]):
        _fail("config.json differs from the immutable manifest configuration")
    if distribution is DistributionStatus.PUBLIC_CANDIDATE:
        release_config = None
        config = PublicProjectionConfig.from_document({"public_dataset": config_value})
    else:
        release_config = PublicReleaseConfig.from_document(
            {"public_release": config_value}
        )
        config = release_config.projection
    return PublicDataset(
        artifact=validated,
        config=config,
        distribution=distribution,
        release_config=release_config,
    )


@version(
    "1.1",
    dependencies=[
        DatasetDistribution,
        PublicDatasetValidationSummary,
        open_public_dataset,
        validate_public_payload,
    ],
)
def validate_public_dataset(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
    *,
    semantic: bool = True,
    deep_lineage: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> PublicDatasetValidationSummary | None:
    """Validate the artifact itself by default and optional lineage explicitly."""

    pointer = artifact.pointer if isinstance(artifact, ValidatedArtifact) else artifact
    validated = store.validate(pointer, deep=deep_lineage)
    dataset = open_public_dataset(store, validated)
    if validated.manifest.get("upstream") != [dataset.config.source_dataset.as_dict()]:
        _fail("Public manifest source differs from the projection configuration")
    if dataset.distribution is DistributionStatus.PUBLIC_CANDIDATE:
        expected_distribution = DatasetDistribution(
            DistributionStatus.PUBLIC_CANDIDATE
        )
    else:
        if dataset.release_config is None:
            _fail("Public release is missing its release configuration")
        approval = dataset.release_config.approval
        expected_distribution = DatasetDistribution(
            DistributionStatus.PUBLIC,
            release_license=approval.release_license,
            redistribution_basis=approval.redistribution_basis,
            approver=approval.approver,
            approval_date=approval.approval_date.isoformat(),
            approval_reference=approval.approval_reference,
        )
    if validated.manifest.get("distribution") != expected_distribution.as_dict():
        _fail("Public manifest distribution differs from immutable approval state")
    summary, expected_metadata = validate_public_payload(
        validated.path,
        validated.pointer.artifact_id,
        dataset.config,
        distribution=dataset.distribution,
        release_config=dataset.release_config,
        semantic=semantic,
        progress=progress,
    )
    if validated.manifest.get("metadata") != expected_metadata:
        _fail("Manifest metadata differs from the closed public metadata contract")
    if deep_lineage:
        # Import locally to keep the pure payload reader independent of the
        # publication/orchestration module during normal shallow use.
        from sanos.datasets.public.publish import reproject_public_source

        expected_frame, expected_omissions, _ = reproject_public_source(
            store, dataset.config
        )
        observed_frame = dataset.load()
        if not observed_frame.equals(expected_frame, null_equal=True):
            _fail("Public content differs from an exact source re-projection")
        evidence = _read_json(validated.path / "validation.json", "validation.json")
        if tuple(evidence["omitted_dates"]) != expected_omissions:
            _fail("Public omissions differ from the exact source re-projection")
    return summary


__all__ = [
    "PublicDataset",
    "PublicDatasetValidationError",
    "PublicDatasetValidationSummary",
    "open_public_dataset",
    "public_coverage_sha256",
    "public_frame_logical_sha256",
    "public_manifest_metadata",
    "public_return_values_sha256",
    "public_validation_document",
    "validate_public_payload",
    "validate_public_dataset",
    "validate_public_frame",
]
