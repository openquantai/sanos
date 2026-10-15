"""Closed-schema HDF5 transport for DySANOS model inputs."""

from __future__ import annotations

import hashlib
import importlib
import os
import time
import uuid
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.version import version

from sanos.artifacts import ArtifactIntegrityError, ArtifactPointer, StorageClass
from sanos.config import canonical_json, canonical_sha256
from sanos.models.dysanos.config import ExperimentConfig, Hdf5StorageConfig
from sanos.models.dysanos.data import (
    DailyFrontDay,
    EligibilityAudit,
    exact_weekly_pairs,
    recompute_market_fingerprint,
)
from sanos.widb.calendar import DAYS_PER_YEAR, NMS

EXPERIMENT_HDF5_SCHEMA = "sanos-dysanos-1.1-v1"
EXPERIMENT_HDF5_NAME = "data.h5"
SHARD_SCHEMA = "sanos-daily-front-public-shard-v2"
_EPOCH = np.datetime64("1970-01-01", "D")


class ExperimentValidationError(ArtifactIntegrityError):
    """Raised when an Experiment violates its closed semantic contract."""


def _h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise RuntimeError("Experiment HDF5 support requires 'sanos[experiment]'") from exc


@version("1.0")
def _date_day(value: date) -> int:
    return int((np.datetime64(value, "D") - _EPOCH).astype(np.int64))


@version("1.0")
def _day_date(value: int) -> date:
    return (_EPOCH + np.timedelta64(int(value), "D")).astype(object)


@version("1.0")
def _encode(value: str, width: int) -> np.bytes_:
    encoded = value.encode("ascii")
    if len(encoded) > width:
        raise ValueError(f"ASCII value exceeds fixed width {width}: {value!r}")
    return np.bytes_(encoded)


@version("1.0")
def _decode(value: object) -> str:
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).rstrip(b"\x00").decode("ascii")
    raise ExperimentValidationError(f"Expected fixed ASCII bytes, found {type(value).__name__}")


_SHARD_ARRAYS = (
    "expiry_time_y",
    "expiry_date_days",
    "expiry_is_am",
    "expiry_atm_annual_vol",
    "expiry_strike_lower",
    "expiry_strike_upper",
    "model_atm_total_vol",
    "quote_expiry_index",
    "quote_price_index",
    "quote_strike",
    "quote_is_call",
    "quote_fit",
    "quote_half_spread",
    "price_expiry_y",
    "price_strike",
)


@version("2.0", dependencies=[DailyFrontDay, _date_day, _encode])
def write_day_shard(path: Path, day: DailyFrontDay) -> None:
    """Atomically persist one restartable, pickle-free daily NPZ shard."""

    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"Unsafe shard destination: {path}")
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f"Unsafe shard destination: {destination}")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    audit = np.asarray(
        (
            day.audit.active_european_books,
            day.audit.horizon_eligible_books,
            day.audit.nms_eligible_books,
            day.audit.input_expiries,
            day.audit.eligible_expiries,
        ),
        dtype=np.int64,
    )
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(_encode(SHARD_SCHEMA, 64), dtype="S64"),
                ticker=np.asarray(_encode(day.ticker, 32), dtype="S32"),
                date_day=np.asarray(_date_day(day.trading_date), dtype=np.int32),
                is_half_day=np.asarray(day.is_half_day, dtype=np.bool_),
                driftless_spot_return=np.asarray(day.driftless_spot_return, dtype=np.float64),
                driftless_spot_return_valid=np.asarray(
                    day.driftless_spot_return_valid, dtype=np.bool_
                ),
                market_uid=np.asarray(_encode(day.market_uid, 64), dtype="S64"),
                market_fingerprint=np.asarray(_encode(day.market_fingerprint, 64), dtype="S64"),
                audit=audit,
                **{name: getattr(day, name) for name in _SHARD_ARRAYS},
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


@version("2.0", dependencies=[DailyFrontDay, EligibilityAudit, _day_date, _decode])
def read_day_shard(path: Path) -> DailyFrontDay:
    """Read and validate one restart shard without enabling pickle."""

    if path.is_symlink() or not path.is_file():
        raise ExperimentValidationError(f"Missing or unsafe shard: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected = {
                "schema",
                "ticker",
                "date_day",
                "is_half_day",
                "driftless_spot_return",
                "driftless_spot_return_valid",
                "market_uid",
                "market_fingerprint",
                "audit",
                *_SHARD_ARRAYS,
            }
            if set(payload.files) != expected:
                raise ExperimentValidationError(
                    f"Shard fields changed: missing={sorted(expected - set(payload.files))}, "
                    f"unexpected={sorted(set(payload.files) - expected)}"
                )
            if _decode(payload["schema"].item()) != SHARD_SCHEMA:
                raise ExperimentValidationError("Unsupported daily-front shard schema")
            audit = np.asarray(payload["audit"], dtype=np.int64)
            if audit.shape != (5,) or np.any(audit < 0):
                raise ExperimentValidationError("Shard eligibility audit is malformed")
            arrays = {name: np.asarray(payload[name]).copy() for name in _SHARD_ARRAYS}
            return DailyFrontDay(
                ticker=_decode(payload["ticker"].item()),
                trading_date=_day_date(int(payload["date_day"].item())),
                is_half_day=bool(payload["is_half_day"].item()),
                driftless_spot_return=float(payload["driftless_spot_return"].item()),
                driftless_spot_return_valid=bool(
                    payload["driftless_spot_return_valid"].item()
                ),
                market_uid=_decode(payload["market_uid"].item()),
                market_fingerprint=_decode(payload["market_fingerprint"].item()),
                audit=EligibilityAudit(*(int(value) for value in audit)),
                **arrays,
            )
    except ExperimentValidationError:
        raise
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ExperimentValidationError(f"Cannot read daily-front shard {path}: {exc}") from exc


@version("1.0")
def _hash_item(digest: Any, name: str, value: np.ndarray) -> None:
    encoded = name.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


@version("2.0", dependencies=[_hash_item, _date_day, canonical_json])
def logical_content_digest(
    days: Sequence[DailyFrontDay],
    *,
    config_id: str,
    source: ArtifactPointer,
    model_business_dtes: np.ndarray,
    weekly_origin: np.ndarray,
    weekly_endpoint: np.ndarray,
    progress: Any = None,
    progress_interval_seconds: float = 30.0,
    progress_event: str = "experiment_logical_digest_progress",
) -> str:
    """Hash typed logical content independently of HDF5 container bytes."""

    digest = hashlib.sha256()
    for label in (EXPERIMENT_HDF5_SCHEMA, config_id, canonical_json(source.as_dict())):
        value = label.encode("utf-8")
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    _hash_item(digest, "model/business_dte", model_business_dtes)
    _hash_item(digest, "dynamics/weekly_origin_index", weekly_origin)
    _hash_item(digest, "dynamics/weekly_endpoint_index", weekly_endpoint)
    started = time.monotonic()
    last_progress = started
    for index, day in enumerate(days):
        prefix = f"day/{index}/"
        for name, value in (
            ("date_day", np.asarray(_date_day(day.trading_date), dtype=np.int32)),
            ("is_half_day", np.asarray(day.is_half_day, dtype=np.bool_)),
            (
                "driftless_spot_return",
                np.asarray(day.driftless_spot_return, dtype=np.float64),
            ),
            (
                "driftless_spot_return_valid",
                np.asarray(day.driftless_spot_return_valid, dtype=np.bool_),
            ),
            ("market_uid", np.frombuffer(day.market_uid.encode("ascii"), dtype=np.uint8)),
            (
                "market_fingerprint",
                np.frombuffer(day.market_fingerprint.encode("ascii"), dtype=np.uint8),
            ),
            *((name, getattr(day, name)) for name in _SHARD_ARRAYS),
            (
                "audit",
                np.asarray(
                    (
                        day.audit.active_european_books,
                        day.audit.horizon_eligible_books,
                        day.audit.nms_eligible_books,
                        day.audit.input_expiries,
                        day.audit.eligible_expiries,
                    ),
                    dtype=np.int64,
                ),
            ),
        ):
            _hash_item(digest, prefix + name, np.asarray(value))
        now = time.monotonic()
        if progress is not None and (
            index + 1 == len(days) or now - last_progress >= progress_interval_seconds
        ):
            elapsed = now - started
            rate = (index + 1) / elapsed if elapsed > 0.0 else 0.0
            eta = (len(days) - index - 1) / rate if rate > 0.0 else None
            progress(
                {
                    "event": progress_event,
                    "completed": index + 1,
                    "total": len(days),
                    "failures": 0,
                    "elapsed_seconds": round(elapsed, 6),
                    "eta_seconds": None if eta is None else round(eta, 6),
                }
            )
            last_progress = now
    return digest.hexdigest()


@version("1.0")
def _create_dataset(
    group: Any,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    *,
    fillvalue: object,
    storage: Hdf5StorageConfig,
) -> Any:
    chunks = tuple(1 if index == 0 else min(size, 1024) for index, size in enumerate(shape))
    maxshape = tuple(None if size == 0 else size for size in shape) if 0 in shape else None
    return group.create_dataset(
        name,
        shape=shape,
        dtype=np.dtype(dtype),
        chunks=chunks,
        compression=storage.compression,
        compression_opts=storage.compression_level,
        shuffle=storage.shuffle,
        fletcher32=storage.fletcher32,
        fillvalue=fillvalue,
        maxshape=maxshape,
    )


@version(
    "2.0",
    dependencies=[
        DailyFrontDay,
        ExperimentConfig,
        exact_weekly_pairs,
        logical_content_digest,
        recompute_market_fingerprint,
        _create_dataset,
        _date_day,
        _encode,
    ],
)
def write_experiment_hdf5(
    path: Path,
    days: Sequence[DailyFrontDay],
    *,
    config: ExperimentConfig,
    producer_version: str,
    progress: Any = None,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Write a complete closed-schema Experiment HDF5 file."""

    if not days:
        raise ValueError("Cannot materialize an empty Experiment")
    dates = tuple(day.trading_date for day in days)
    if dates != tuple(sorted(set(dates))):
        raise ValueError("Experiment days must be unique and increasing")
    tickers = {day.ticker for day in days}
    if len(tickers) != 1:
        raise ValueError("The reference Experiment requires one ticker")
    source = config.source_dataset.pointer
    for day in days:
        if day.market_fingerprint != recompute_market_fingerprint(day, source):
            raise ValueError(f"Market fingerprint does not reconstruct for {day.trading_date}")
    model_business_dtes = np.asarray(config.geometry.model_business_dtes, dtype=np.float64)
    model_time_y = model_business_dtes / DAYS_PER_YEAR
    weekly_origin, weekly_endpoint = exact_weekly_pairs(dates)
    config_id = canonical_sha256(config)
    logical_digest = logical_content_digest(
        days,
        config_id=config_id,
        source=source,
        model_business_dtes=model_business_dtes,
        weekly_origin=weekly_origin,
        weekly_endpoint=weekly_endpoint,
        progress=progress,
        progress_interval_seconds=progress_interval_seconds,
        progress_event="experiment_hdf5_digest_progress",
    )
    D = len(days)
    E = max(day.expiry_count for day in days)
    Q = max(day.quote_count for day in days)
    P = max(day.price_count for day in days)
    M = len(model_business_dtes)
    h5py = _h5py()
    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        raise ValueError(f"HDF5 destination must not already exist: {destination}")
    with h5py.File(destination, "w", libver="latest", track_order=True) as handle:
        attributes = {
            "schema": EXPERIMENT_HDF5_SCHEMA,
            "profile": config.profile,
            "consumer_contract": config.consumer_contract,
            "configuration_id": config_id,
            "producer_version": producer_version,
            "source_storage_class": source.storage_class.value,
            "source_namespace": source.namespace,
            "source_artifact_id": source.artifact_id,
            "ticker": next(iter(tickers)),
            "price_units": "discounted_forward",
            "strike_units": "forward_relative_K_over_F",
            "time_units": "NYSE_business_year_252_sessions",
            "logical_content_sha256": logical_digest,
            "D": D,
            "E": E,
            "Q": Q,
            "P": P,
            "M": M,
        }
        for name, value in attributes.items():
            handle.attrs[name] = value
        day_group = handle.create_group("day", track_order=True)
        quote_group = handle.create_group("quote", track_order=True)
        price_group = handle.create_group("price", track_order=True)
        expiry_group = handle.create_group("expiry", track_order=True)
        model_group = handle.create_group("model", track_order=True)
        split_group = handle.create_group("split", track_order=True)
        period_group = handle.create_group("period", track_order=True)
        dynamics_group = handle.create_group("dynamics", track_order=True)
        audit_group = handle.create_group("audit", track_order=True)

        one_dim = lambda group, name, dtype, fill: _create_dataset(  # noqa: E731
            group, name, (D,), dtype, fillvalue=fill, storage=config.storage
        )
        two_dim = lambda group, name, width, dtype, fill: _create_dataset(  # noqa: E731
            group, name, (D, width), dtype, fillvalue=fill, storage=config.storage
        )
        day_date = one_dim(day_group, "date_days", "<i4", 0)
        day_half = one_dim(day_group, "is_half_day", "?", False)
        day_return = one_dim(day_group, "driftless_spot_return", "<f8", 0.0)
        day_return_valid = one_dim(
            day_group, "driftless_spot_return_valid", "?", False
        )
        day_qcount = one_dim(day_group, "quote_count", "<i4", 0)
        day_pcount = one_dim(day_group, "price_count", "<i4", 0)
        day_ecount = one_dim(day_group, "expiry_count", "<i4", 0)
        day_uid = one_dim(day_group, "market_uid", "S64", b"")
        day_fingerprint = one_dim(day_group, "market_fingerprint", "S64", b"")

        quote_mask = two_dim(quote_group, "mask", Q, "?", False)
        quote_expiry = two_dim(quote_group, "expiry_index", Q, "<i8", 0)
        quote_price = two_dim(quote_group, "price_index", Q, "<i8", 0)
        quote_strike = two_dim(quote_group, "strike", Q, "<f8", 1.0)
        quote_side = two_dim(quote_group, "is_call", Q, "?", True)
        quote_fit = two_dim(quote_group, "fit", Q, "<f8", 0.0)
        quote_half_spread = two_dim(quote_group, "half_spread", Q, "<f8", 0.0)

        price_mask = two_dim(price_group, "mask", P, "?", False)
        price_expiry = two_dim(price_group, "expiry_y", P, "<f8", 0.0)
        price_strike = two_dim(price_group, "strike", P, "<f8", 1.0)

        expiry_mask = two_dim(expiry_group, "mask", E, "?", False)
        expiry_time = two_dim(expiry_group, "time_y", E, "<f8", 0.0)
        expiry_date = two_dim(expiry_group, "date_days", E, "<i4", 0)
        expiry_am = two_dim(expiry_group, "is_am", E, "?", False)
        expiry_atm = two_dim(expiry_group, "atm_annual_vol", E, "<f8", 0.0)
        expiry_lower = two_dim(expiry_group, "strike_lower", E, "<f8", 1.0)
        expiry_upper = two_dim(expiry_group, "strike_upper", E, "<f8", 1.0)

        model_business = _create_dataset(
            model_group,
            "business_dte",
            (M,),
            "<f8",
            fillvalue=0.0,
            storage=config.storage,
        )
        model_expiry = _create_dataset(
            model_group,
            "expiry_y",
            (M,),
            "<f8",
            fillvalue=0.0,
            storage=config.storage,
        )
        model_business[...] = model_business_dtes
        model_expiry[...] = model_time_y
        model_atm = two_dim(model_group, "atm_total_vol", M, "<f8", 0.0)
        train_mask = one_dim(split_group, "train_mask", "?", True)
        validation_mask = one_dim(split_group, "validation_mask", "?", False)
        test_mask = one_dim(split_group, "test_mask", "?", False)
        pre_boundary = one_dim(period_group, "pre_2025_mask", "?", False)
        from_boundary = one_dim(period_group, "from_2025_mask", "?", False)
        weekly_origin_dataset = _create_dataset(
            dynamics_group,
            "weekly_origin_index",
            (len(weekly_origin),),
            "<i8",
            fillvalue=-1,
            storage=config.storage,
        )
        weekly_endpoint_dataset = _create_dataset(
            dynamics_group,
            "weekly_endpoint_index",
            (len(weekly_endpoint),),
            "<i8",
            fillvalue=-1,
            storage=config.storage,
        )
        weekly_origin_dataset[...] = weekly_origin
        weekly_endpoint_dataset[...] = weekly_endpoint
        audit_names = (
            "active_european_books",
            "horizon_eligible_books",
            "nms_eligible_books",
            "input_expiries",
            "eligible_expiries",
        )
        audit_data = {name: one_dim(audit_group, name, "<i8", 0) for name in audit_names}
        boundary = date.fromisoformat(config.split.diagnostic_boundary)
        assembly_started = time.monotonic()
        last_assembly_progress = assembly_started
        for row, day in enumerate(days):
            q, p, e = day.quote_count, day.price_count, day.expiry_count
            day_date[row] = _date_day(day.trading_date)
            day_half[row] = day.is_half_day
            day_return[row] = day.driftless_spot_return if day.driftless_spot_return_valid else 0.0
            day_return_valid[row] = day.driftless_spot_return_valid
            day_qcount[row], day_pcount[row], day_ecount[row] = q, p, e
            day_uid[row] = _encode(day.market_uid, 64)
            day_fingerprint[row] = _encode(day.market_fingerprint, 64)
            quote_mask[row, :q] = True
            quote_expiry[row, :q] = day.quote_expiry_index
            quote_price[row, :q] = day.quote_price_index
            quote_strike[row, :q] = day.quote_strike
            quote_side[row, :q] = day.quote_is_call
            quote_fit[row, :q] = day.quote_fit
            quote_half_spread[row, :q] = day.quote_half_spread
            price_mask[row, :p] = True
            price_expiry[row, :p] = day.price_expiry_y
            price_strike[row, :p] = day.price_strike
            expiry_mask[row, :e] = True
            expiry_time[row, :e] = day.expiry_time_y
            expiry_date[row, :e] = day.expiry_date_days
            expiry_am[row, :e] = day.expiry_is_am
            expiry_atm[row, :e] = day.expiry_atm_annual_vol
            expiry_lower[row, :e] = day.expiry_strike_lower
            expiry_upper[row, :e] = day.expiry_strike_upper
            model_atm[row, :] = day.model_atm_total_vol
            train_mask[row] = True
            pre_boundary[row] = day.trading_date < boundary
            from_boundary[row] = day.trading_date >= boundary
            for name in audit_names:
                audit_data[name][row] = getattr(day.audit, name)
            now = time.monotonic()
            if progress is not None and (
                row + 1 == D or now - last_assembly_progress >= progress_interval_seconds
            ):
                elapsed = now - assembly_started
                rate = (row + 1) / elapsed if elapsed > 0.0 else 0.0
                eta = (D - row - 1) / rate if rate > 0.0 else None
                progress(
                    {
                        "event": "experiment_hdf5_assembly_progress",
                        "completed": row + 1,
                        "total": D,
                        "failures": 0,
                        "elapsed_seconds": round(elapsed, 6),
                        "eta_seconds": None if eta is None else round(eta, 6),
                    }
                )
                last_assembly_progress = now
        validation_mask[...] = False
        test_mask[...] = False
        handle.flush()
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "schema": EXPERIMENT_HDF5_SCHEMA,
        "dimensions": {"D": D, "E": E, "Q": Q, "P": P, "M": M},
        "dates": {"first": dates[0], "last": dates[-1], "count": D},
        "books": sum(day.quote_count for day in days),
        "calls": sum(int(np.sum(day.quote_is_call)) for day in days),
        "puts": sum(int(np.sum(~day.quote_is_call)) for day in days),
        "weekly_pairs": len(weekly_origin),
        "logical_content_sha256": logical_digest,
    }


_EXPECTED_DATASETS: Mapping[str, str] = {
    "day/date_days": "<i4",
    "day/is_half_day": "|b1",
    "day/driftless_spot_return": "<f8",
    "day/driftless_spot_return_valid": "|b1",
    "day/quote_count": "<i4",
    "day/price_count": "<i4",
    "day/expiry_count": "<i4",
    "day/market_uid": "|S64",
    "day/market_fingerprint": "|S64",
    "quote/mask": "|b1",
    "quote/expiry_index": "<i8",
    "quote/price_index": "<i8",
    "quote/strike": "<f8",
    "quote/is_call": "|b1",
    "quote/fit": "<f8",
    "quote/half_spread": "<f8",
    "price/mask": "|b1",
    "price/expiry_y": "<f8",
    "price/strike": "<f8",
    "expiry/mask": "|b1",
    "expiry/time_y": "<f8",
    "expiry/date_days": "<i4",
    "expiry/is_am": "|b1",
    "expiry/atm_annual_vol": "<f8",
    "expiry/strike_lower": "<f8",
    "expiry/strike_upper": "<f8",
    "model/business_dte": "<f8",
    "model/expiry_y": "<f8",
    "model/atm_total_vol": "<f8",
    "split/train_mask": "|b1",
    "split/validation_mask": "|b1",
    "split/test_mask": "|b1",
    "period/pre_2025_mask": "|b1",
    "period/from_2025_mask": "|b1",
    "dynamics/weekly_origin_index": "<i8",
    "dynamics/weekly_endpoint_index": "<i8",
    "audit/active_european_books": "<i8",
    "audit/horizon_eligible_books": "<i8",
    "audit/nms_eligible_books": "<i8",
    "audit/input_expiries": "<i8",
    "audit/eligible_expiries": "<i8",
}

_EXPECTED_GROUPS = {
    "audit",
    "day",
    "dynamics",
    "expiry",
    "model",
    "period",
    "price",
    "quote",
    "split",
}


@version("2.0")
def experiment_hdf5_schema_document() -> dict[str, Any]:
    """Return the exact portable HDF5 path, dtype, shape, unit, and pad contract."""

    shapes: dict[str, str] = {}
    for name in _EXPECTED_DATASETS:
        if name in {"model/business_dte", "model/expiry_y"}:
            shapes[name] = "[M]"
        elif name.startswith("dynamics/"):
            shapes[name] = "[W]"
        elif name == "model/atm_total_vol":
            shapes[name] = "[D,M]"
        elif name.startswith("quote/"):
            shapes[name] = "[D,Q]"
        elif name.startswith("price/"):
            shapes[name] = "[D,P]"
        elif name.startswith("expiry/"):
            shapes[name] = "[D,E]"
        else:
            shapes[name] = "[D]"
    padding: dict[str, object | None] = {name: None for name in _EXPECTED_DATASETS}
    padding.update(
        {
            "quote/mask": False,
            "quote/expiry_index": 0,
            "quote/price_index": 0,
            "quote/strike": 1.0,
            "quote/is_call": True,
            "quote/fit": 0.0,
            "quote/half_spread": 0.0,
            "price/mask": False,
            "price/expiry_y": 0.0,
            "price/strike": 1.0,
            "expiry/mask": False,
            "expiry/time_y": 0.0,
            "expiry/date_days": 0,
            "expiry/is_am": False,
            "expiry/atm_annual_vol": 0.0,
            "expiry/strike_lower": 1.0,
            "expiry/strike_upper": 1.0,
        }
    )
    units: dict[str, str] = {name: "not_applicable" for name in _EXPECTED_DATASETS}
    for name in ("quote/strike", "price/strike", "expiry/strike_lower", "expiry/strike_upper"):
        units[name] = "K/F"
    for name in ("quote/fit", "quote/half_spread"):
        units[name] = "option_price/(DF*F)"
    for name in ("expiry/time_y", "price/expiry_y", "model/expiry_y"):
        units[name] = "NYSE_business_years"
    units["model/business_dte"] = "NYSE_business_sessions"
    units["expiry/atm_annual_vol"] = "annualized_volatility"
    units["model/atm_total_vol"] = "volatility*sqrt(years)"
    units["day/date_days"] = "days_since_1970-01-01"
    units["expiry/date_days"] = "days_since_1970-01-01"
    units["day/driftless_spot_return"] = "dimensionless_driftless_return"
    units["day/driftless_spot_return_valid"] = "validity_mask"
    return {
        "schema": EXPERIMENT_HDF5_SCHEMA,
        "dimensions": {
            "D": "populated retained dates",
            "E": "maximum retained expiries per date",
            "Q": "maximum separate-side books per date",
            "P": "maximum unique pricing locations per date",
            "M": "fixed model nodes",
            "W": "exact five-session date pairs",
        },
        "datasets": {
            name: {
                "dtype": dtype,
                "shape": shapes[name],
                "unit": units[name],
                "padding": padding[name],
            }
            for name, dtype in sorted(_EXPECTED_DATASETS.items())
        },
        "storage": {
            "chunking": "day-major",
            "compression": "configured deterministic gzip",
            "validity": "counts and prefix masks are authoritative",
        },
    }


@version("1.1")
def _object_paths(handle: Any) -> tuple[set[str], set[str]]:
    h5py = _h5py()
    datasets: set[str] = set()
    groups: set[str] = set()

    def address(value: Any) -> int:
        return int(h5py.h5o.get_info(value.id).addr)

    expanded_groups = {address(handle)}

    def visit_group(group: Any, prefix: str) -> None:
        # Walk links rather than objects.  h5py.Group.visititems dereferences
        # links and visits a hard-linked object only once, which can hide both
        # dangling links and aliases from a closed-schema validator.
        for child in group.keys():
            name = f"{prefix}/{child}" if prefix else child
            link = group.get(child, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ExperimentValidationError(f"HDF5 links are forbidden: {name}")
            value = group.get(child, getlink=False)
            if value is None:
                raise ExperimentValidationError(f"Unresolvable HDF5 object: {name}")
            if isinstance(value, h5py.Dataset):
                if value.is_virtual or value.external:
                    raise ExperimentValidationError(
                        f"External/virtual HDF5 data is forbidden: {name}"
                    )
                datasets.add(name)
            elif isinstance(value, h5py.Group):
                groups.add(name)
                object_address = address(value)
                if object_address not in expanded_groups:
                    expanded_groups.add(object_address)
                    visit_group(value, name)
            else:
                raise ExperimentValidationError(f"Unsupported HDF5 object: {name}")

    visit_group(handle, "")
    return datasets, groups


@version("1.0")
def _prefix_mask(mask: np.ndarray, count: int) -> bool:
    expected = np.zeros(mask.shape, dtype=np.bool_)
    expected[:count] = True
    return bool(np.array_equal(mask, expected))


@version("1.0")
def _hdf_dataset_digest(dataset: Any) -> str:
    """Hash one typed HDF5 array in bounded row-major blocks."""

    digest = hashlib.sha256()
    digest.update(dataset.name.lstrip("/").encode("utf-8"))
    digest.update(dataset.dtype.str.encode("ascii"))
    digest.update(np.asarray(dataset.shape, dtype=np.int64).tobytes())
    if not dataset.shape or dataset.shape[0] == 0:
        values = np.asarray(dataset[()]) if not dataset.shape else np.asarray([], dataset.dtype)
        digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()
    row_items = int(np.prod(dataset.shape[1:], dtype=np.int64)) if dataset.ndim > 1 else 1
    row_bytes = max(1, row_items * dataset.dtype.itemsize)
    block_rows = max(1, min(dataset.shape[0], (4 * 1024 * 1024) // row_bytes))
    for start in range(0, dataset.shape[0], block_rows):
        values = np.ascontiguousarray(dataset[start : start + block_rows])
        digest.update(values.tobytes())
    return digest.hexdigest()


@version("2.0", dependencies=[DailyFrontDay, EligibilityAudit, _day_date, _decode])
def read_hdf5_day(handle: Any, index: int) -> DailyFrontDay:
    D = int(handle.attrs["D"])
    if not 0 <= index < D:
        raise IndexError(index)
    q = int(handle["day/quote_count"][index])
    p = int(handle["day/price_count"][index])
    e = int(handle["day/expiry_count"][index])
    return DailyFrontDay(
        ticker=str(handle.attrs["ticker"]),
        trading_date=_day_date(int(handle["day/date_days"][index])),
        is_half_day=bool(handle["day/is_half_day"][index]),
        driftless_spot_return=(
            float(handle["day/driftless_spot_return"][index])
            if bool(handle["day/driftless_spot_return_valid"][index])
            else float("nan")
        ),
        driftless_spot_return_valid=bool(
            handle["day/driftless_spot_return_valid"][index]
        ),
        market_uid=_decode(handle["day/market_uid"][index]),
        market_fingerprint=_decode(handle["day/market_fingerprint"][index]),
        expiry_time_y=np.asarray(handle["expiry/time_y"][index, :e], dtype=np.float64),
        expiry_date_days=np.asarray(handle["expiry/date_days"][index, :e], dtype=np.int32),
        expiry_is_am=np.asarray(handle["expiry/is_am"][index, :e], dtype=np.bool_),
        expiry_atm_annual_vol=np.asarray(
            handle["expiry/atm_annual_vol"][index, :e], dtype=np.float64
        ),
        expiry_strike_lower=np.asarray(handle["expiry/strike_lower"][index, :e], dtype=np.float64),
        expiry_strike_upper=np.asarray(handle["expiry/strike_upper"][index, :e], dtype=np.float64),
        model_atm_total_vol=np.asarray(handle["model/atm_total_vol"][index, :], dtype=np.float64),
        quote_expiry_index=np.asarray(handle["quote/expiry_index"][index, :q], dtype=np.int64),
        quote_price_index=np.asarray(handle["quote/price_index"][index, :q], dtype=np.int64),
        quote_strike=np.asarray(handle["quote/strike"][index, :q], dtype=np.float64),
        quote_is_call=np.asarray(handle["quote/is_call"][index, :q], dtype=np.bool_),
        quote_fit=np.asarray(handle["quote/fit"][index, :q], dtype=np.float64),
        quote_half_spread=np.asarray(
            handle["quote/half_spread"][index, :q], dtype=np.float64
        ),
        price_expiry_y=np.asarray(handle["price/expiry_y"][index, :p], dtype=np.float64),
        price_strike=np.asarray(handle["price/strike"][index, :p], dtype=np.float64),
        audit=EligibilityAudit(
            *(
                int(handle[f"audit/{name}"][index])
                for name in (
                    "active_european_books",
                    "horizon_eligible_books",
                    "nms_eligible_books",
                    "input_expiries",
                    "eligible_expiries",
                )
            )
        ),
    )


@version(
    "2.0",
    dependencies=[
        read_hdf5_day,
        logical_content_digest,
        exact_weekly_pairs,
        recompute_market_fingerprint,
        _object_paths,
        _prefix_mask,
        _hdf_dataset_digest,
    ],
)
def validate_experiment_hdf5(
    path: Path,
    *,
    config: ExperimentConfig,
    producer_version: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Validate closed HDF5 structure, padding, invariants, and logical digest."""

    if path.is_symlink() or not path.is_file():
        raise ExperimentValidationError(f"Missing or unsafe Experiment HDF5: {path}")
    h5py = _h5py()
    source = config.source_dataset.pointer
    config_id = canonical_sha256(config)
    with h5py.File(path, "r", swmr=True) as handle:
        dataset_paths, group_paths = _object_paths(handle)
        if dataset_paths != set(_EXPECTED_DATASETS) or group_paths != _EXPECTED_GROUPS:
            missing = sorted(set(_EXPECTED_DATASETS) - dataset_paths)
            unexpected = sorted(dataset_paths - set(_EXPECTED_DATASETS))
            missing_groups = sorted(_EXPECTED_GROUPS - group_paths)
            unexpected_groups = sorted(group_paths - _EXPECTED_GROUPS)
            raise ExperimentValidationError(
                "HDF5 objects changed: "
                f"missing={missing}, unexpected={unexpected}, "
                f"missing_groups={missing_groups}, unexpected_groups={unexpected_groups}"
            )
        for name in dataset_paths | group_paths:
            if set(handle[name].attrs):
                raise ExperimentValidationError(f"Unexpected HDF5 object attributes at {name}")
        for name, dtype in _EXPECTED_DATASETS.items():
            if handle[name].dtype.str != dtype:
                raise ExperimentValidationError(
                    f"HDF5 dtype changed at {name}: {handle[name].dtype.str} != {dtype}"
                )
            dataset = handle[name]
            if (
                dataset.chunks is None
                or dataset.compression != config.storage.compression
                or dataset.compression_opts != config.storage.compression_level
                or bool(dataset.shuffle) != config.storage.shuffle
                or bool(dataset.fletcher32) != config.storage.fletcher32
            ):
                raise ExperimentValidationError(f"HDF5 storage filters changed at {name}")
        required_attrs = {
            "schema",
            "profile",
            "consumer_contract",
            "configuration_id",
            "producer_version",
            "source_storage_class",
            "source_namespace",
            "source_artifact_id",
            "ticker",
            "price_units",
            "strike_units",
            "time_units",
            "logical_content_sha256",
            "D",
            "E",
            "Q",
            "P",
            "M",
        }
        if set(handle.attrs) != required_attrs:
            raise ExperimentValidationError("HDF5 root attributes changed")
        expected_attrs = {
            "schema": EXPERIMENT_HDF5_SCHEMA,
            "profile": config.profile,
            "consumer_contract": config.consumer_contract,
            "configuration_id": config_id,
            "producer_version": producer_version,
            "source_storage_class": StorageClass.DATASET.value,
            "source_namespace": source.namespace,
            "source_artifact_id": source.artifact_id,
            "price_units": "discounted_forward",
            "strike_units": "forward_relative_K_over_F",
            "time_units": "NYSE_business_year_252_sessions",
        }
        for name, expected in expected_attrs.items():
            if str(handle.attrs[name]) != expected:
                raise ExperimentValidationError(f"HDF5 attribute {name} changed")
        D, E, Q, P, M = (int(handle.attrs[name]) for name in ("D", "E", "Q", "P", "M"))
        expected_shapes = {
            name: (
                (M,)
                if name in {"model/business_dte", "model/expiry_y"}
                else (
                    (len(handle[name]),)
                    if name.startswith("dynamics/")
                    else (
                        (D, M)
                        if name == "model/atm_total_vol"
                        else (
                            (D, Q)
                            if name.startswith("quote/")
                            else (
                                (D, P)
                                if name.startswith("price/")
                                else ((D, E) if name.startswith("expiry/") else (D,))
                            )
                        )
                    )
                )
            )
            for name in _EXPECTED_DATASETS
        }
        for name, shape in expected_shapes.items():
            if handle[name].shape != shape:
                raise ExperimentValidationError(
                    f"HDF5 shape changed at {name}: {handle[name].shape} != {shape}"
                )
        model_business_dtes = np.asarray(handle["model/business_dte"][:], dtype=np.float64)
        if not np.array_equal(
            model_business_dtes, np.asarray(config.geometry.model_business_dtes, dtype=np.float64)
        ):
            raise ExperimentValidationError("Model business-DTE geometry changed")
        if not np.array_equal(handle["model/expiry_y"][:], model_business_dtes / DAYS_PER_YEAR):
            raise ExperimentValidationError("Model expiry-year geometry changed")
        dates_raw = np.asarray(handle["day/date_days"][:], dtype=np.int32)
        if len(dates_raw) == 0 or np.any(np.diff(dates_raw.astype(np.int64)) <= 0):
            raise ExperimentValidationError("Experiment dates are not increasing")
        returns = np.asarray(handle["day/driftless_spot_return"][:], dtype=np.float64)
        return_valid = np.asarray(
            handle["day/driftless_spot_return_valid"][:], dtype=np.bool_
        )
        expected_return_valid = np.ones(D, dtype=np.bool_)
        expected_return_valid[0] = False
        if (
            not np.array_equal(return_valid, expected_return_valid)
            or np.any(~np.isfinite(returns))
            or np.any(returns[return_valid] <= -1.0)
            or returns[0] != 0.0
        ):
            raise ExperimentValidationError(
                "Driftless spot returns or their first-value validity mask changed"
            )
        days: list[DailyFrontDay] = []
        total_books = 0
        calls = 0
        training_fingerprints = hashlib.sha256()
        for index in range(D):
            q = int(handle["day/quote_count"][index])
            p = int(handle["day/price_count"][index])
            e = int(handle["day/expiry_count"][index])
            if not (0 < q <= Q and 0 < p <= P and 0 < e <= E):
                raise ExperimentValidationError(f"Invalid day counts at row {index}")
            for group, width, count in (("quote", Q, q), ("price", P, p), ("expiry", E, e)):
                if not _prefix_mask(np.asarray(handle[f"{group}/mask"][index, :]), count):
                    raise ExperimentValidationError(
                        f"{group} mask is not a valid prefix at {index}"
                    )
            padding_checks = (
                ("quote/expiry_index", q, 0),
                ("quote/price_index", q, 0),
                ("quote/strike", q, 1.0),
                ("quote/is_call", q, True),
                ("quote/fit", q, 0.0),
                ("quote/half_spread", q, 0.0),
                ("price/expiry_y", p, 0.0),
                ("price/strike", p, 1.0),
                ("expiry/time_y", e, 0.0),
                ("expiry/date_days", e, 0),
                ("expiry/is_am", e, False),
                ("expiry/atm_annual_vol", e, 0.0),
                ("expiry/strike_lower", e, 1.0),
                ("expiry/strike_upper", e, 1.0),
            )
            for name, count, fill in padding_checks:
                tail = np.asarray(handle[name][index, count:])
                if tail.size and not np.all(tail == fill):
                    raise ExperimentValidationError(f"Noncanonical padding at {name}[{index}]")
            quote_expiry_index = np.asarray(handle["quote/expiry_index"][index, :q], dtype=np.int64)
            if np.any(quote_expiry_index < 0) or np.any(quote_expiry_index >= e):
                raise ExperimentValidationError(
                    f"Quote expiry index is out of range at row {index}"
                )
            quote_strike = np.asarray(handle["quote/strike"][index, :q], dtype=np.float64)
            expiry_lower = np.asarray(handle["expiry/strike_lower"][index, :e], dtype=np.float64)
            expiry_upper = np.asarray(handle["expiry/strike_upper"][index, :e], dtype=np.float64)
            if np.any(quote_strike < expiry_lower[quote_expiry_index]) or np.any(
                quote_strike > expiry_upper[quote_expiry_index]
            ):
                raise ExperimentValidationError(
                    f"Quote strike lies outside stored NMS bounds at row {index}"
                )
            try:
                day = read_hdf5_day(handle, index)
            except (ValueError, IndexError, TypeError) as exc:
                raise ExperimentValidationError(f"Invalid daily-front row {index}: {exc}") from exc
            business_dtes = day.expiry_time_y * DAYS_PER_YEAR
            maximum_time_y = config.geometry.model_business_dtes[-1] / DAYS_PER_YEAR
            if np.any(day.expiry_time_y > maximum_time_y + 1.0e-12):
                raise ExperimentValidationError(
                    f"Expiry horizon exceeds configured maximum at row {index}"
                )
            if any(
                np.any(np.abs(business_dtes - excluded) <= 1.0e-4)
                for excluded in config.market.excluded_business_dtes
            ):
                raise ExperimentValidationError(
                    f"Configured excluded business DTE retained at row {index}"
                )
            model_first_time_y = config.geometry.model_business_dtes[0] / DAYS_PER_YEAR
            model_first_annual_vol = day.model_atm_total_vol[0] / np.sqrt(model_first_time_y)
            expected_bounds = np.asarray(
                NMS.from_nms(
                    np.asarray(config.geometry.nms_range, dtype=np.float64)[None, :],
                    day.expiry_time_y[:, None],
                    volatility=day.expiry_atm_annual_vol[:, None],
                    epsilon_volatility=float(model_first_annual_vol),
                ),
                dtype=np.float64,
            )
            observed_bounds = np.column_stack((day.expiry_strike_lower, day.expiry_strike_upper))
            if not np.allclose(
                observed_bounds,
                expected_bounds,
                rtol=0.0,
                atol=1.0e-14,
            ):
                raise ExperimentValidationError(
                    f"Configured NMS bounds do not reconstruct at row {index}"
                )
            if day.market_fingerprint != recompute_market_fingerprint(day, source):
                raise ExperimentValidationError(
                    f"Market fingerprint does not reconstruct at row {index}"
                )
            training_fingerprint = day.training_fingerprint
            encoded_training_fingerprint = training_fingerprint.encode("ascii")
            training_fingerprints.update(len(encoded_training_fingerprint).to_bytes(8, "little"))
            training_fingerprints.update(encoded_training_fingerprint)
            days.append(day)
            total_books += day.quote_count
            calls += int(np.sum(day.quote_is_call))
            if progress is not None:
                progress(
                    {
                        "event": "experiment_validation_progress",
                        "completed": index + 1,
                        "total": D,
                        "failures": 0,
                    }
                )
        if (
            E != max(day.expiry_count for day in days)
            or Q != max(day.quote_count for day in days)
            or P != max(day.price_count for day in days)
        ):
            raise ExperimentValidationError(
                "HDF5 E/Q/P dimensions are not the exact retained maxima"
            )
        if not np.all(handle["split/train_mask"][:]):
            raise ExperimentValidationError("Reference training mask must include every date")
        if np.any(handle["split/validation_mask"][:]) or np.any(handle["split/test_mask"][:]):
            raise ExperimentValidationError("Reference validation/test masks must be empty")
        boundary = date.fromisoformat(config.split.diagnostic_boundary)
        expected_pre = np.asarray(
            [day.trading_date < boundary for day in days], dtype=np.bool_
        )
        pre_matches = np.array_equal(handle["period/pre_2025_mask"][:], expected_pre)
        post_matches = np.array_equal(handle["period/from_2025_mask"][:], ~expected_pre)
        if not pre_matches or not post_matches:
            raise ExperimentValidationError("Diagnostic period masks changed")
        weekly_origin, weekly_endpoint = exact_weekly_pairs(tuple(day.trading_date for day in days))
        origins_match = np.array_equal(handle["dynamics/weekly_origin_index"][:], weekly_origin)
        endpoints_match = np.array_equal(
            handle["dynamics/weekly_endpoint_index"][:], weekly_endpoint
        )
        if not origins_match or not endpoints_match:
            raise ExperimentValidationError("Exact weekly dynamics pairs changed")
        digest = logical_content_digest(
            days,
            config_id=config_id,
            source=source,
            model_business_dtes=model_business_dtes,
            weekly_origin=weekly_origin,
            weekly_endpoint=weekly_endpoint,
            progress=progress,
            progress_event="experiment_validation_digest_progress",
        )
        if digest != str(handle.attrs["logical_content_sha256"]):
            raise ExperimentValidationError("HDF5 logical content digest changed")
        dataset_digests = {
            name: _hdf_dataset_digest(handle[name]) for name in sorted(_EXPECTED_DATASETS)
        }
        return {
            "schema": EXPERIMENT_HDF5_SCHEMA,
            "status": "passed",
            "dimensions": {"D": D, "E": E, "Q": Q, "P": P, "M": M},
            "dates": {
                "first": days[0].trading_date,
                "last": days[-1].trading_date,
                "count": D,
            },
            "books": total_books,
            "calls": calls,
            "puts": total_books - calls,
            "weekly_pairs": len(weekly_origin),
            "consumer": {
                "contract": config.consumer_contract,
                "adapter_status": "passed",
                "training_fingerprints_sha256": training_fingerprints.hexdigest(),
            },
            "logical_content_sha256": digest,
            "dataset_logical_sha256": dataset_digests,
        }


@dataclass(frozen=True, slots=True)
class DailyFrontExperiment:
    """Self-contained read-only access to one validated Experiment HDF5."""

    path: Path
    config: ExperimentConfig
    producer_version: str

    def __len__(self) -> int:
        h5py = _h5py()
        with h5py.File(self.path, "r", swmr=True) as handle:
            return int(handle.attrs["D"])

    def dates(self) -> tuple[date, ...]:
        h5py = _h5py()
        with h5py.File(self.path, "r", swmr=True) as handle:
            return tuple(_day_date(int(value)) for value in handle["day/date_days"][:])

    def load_day(self, value: int | date) -> DailyFrontDay:
        h5py = _h5py()
        with h5py.File(self.path, "r", swmr=True) as handle:
            if isinstance(value, date):
                values = np.asarray(handle["day/date_days"][:], dtype=np.int32)
                matches = np.flatnonzero(values == _date_day(value))
                if len(matches) != 1:
                    raise KeyError(value.isoformat())
                index = int(matches[0])
            else:
                index = value
            return read_hdf5_day(handle, index)

    def iter_days(self) -> Iterator[DailyFrontDay]:
        for index in range(len(self)):
            yield self.load_day(index)

    def driftless_spot_returns(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored driftless returns and their authoritative validity mask."""

        h5py = _h5py()
        with h5py.File(self.path, "r", swmr=True) as handle:
            returns = np.asarray(handle["day/driftless_spot_return"][:], dtype=np.float64)
            valid = np.asarray(
                handle["day/driftless_spot_return_valid"][:], dtype=np.bool_
            )
        return returns, valid

    def driftless_spot_path(self, *, initial_value: float = 1.0) -> np.ndarray:
        """Reconstruct the spot-one driftless path as cumulative ``prod(1+r)``."""

        if not np.isfinite(initial_value) or initial_value <= 0.0:
            raise ValueError("initial_value must be finite and positive")
        returns, valid = self.driftless_spot_returns()
        expected = np.ones(len(returns), dtype=np.bool_)
        if len(expected):
            expected[0] = False
        if not np.array_equal(valid, expected):
            raise ExperimentValidationError("Driftless return validity mask is noncanonical")
        path = np.empty(len(returns), dtype=np.float64)
        if len(path):
            path[0] = float(initial_value)
            path[1:] = float(initial_value) * np.cumprod(1.0 + returns[1:])
        return path


__all__ = [
    "DailyFrontExperiment",
    "EXPERIMENT_HDF5_NAME",
    "EXPERIMENT_HDF5_SCHEMA",
    "ExperimentValidationError",
    "experiment_hdf5_schema_document",
    "logical_content_digest",
    "read_day_shard",
    "read_hdf5_day",
    "validate_experiment_hdf5",
    "write_day_shard",
    "write_experiment_hdf5",
]
