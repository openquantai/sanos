"""Isolated worker for the pinned DySANOS Experiment migration oracle.

This module is intentionally invoked in a fresh Python process by
``sanos.reference.experiment_compare``.  The process imports the read-only
``quant_data`` snapshot/checkout only after its exact source files have been
hashed, and it never calls cache or WRDS entry points.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
import types
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.version import version

from sanos.config import canonical_value
from sanos.experiments import ExperimentConfig, exact_weekly_pairs
from sanos.experiments.hdf5 import read_hdf5_day
from sanos.reference.legacy_market import read_legacy_pytree_hdf5

PINNED_SOURCE_SHA256 = {
    "quantdata/__init__.py": (
        "ecd4c8611a78dbe928fdb05b5c7b9fdb5d566601b32664885b3606670dbaf060"
    ),
    "quantdata/biz_calendar.py": (
        "3ab556c9a39f7e385ec73b7ff0a953f0765204a73427cb3c0811afc6af8766e3"
    ),
    "quantdata/dysanos/flex/config.py": (
        "d97ba30cedec7dbda3801e3857f178d02ebb7590a9ac98053b974d8a3765baeb"
    ),
    "quantdata/dysanos/__init__.py": (
        "4669c7e60c6bfb2f53bdfea3c4af66ee949e41b03787001e28014d4eef0f3143"
    ),
    "quantdata/dysanos/flex/market.py": (
        "fc1a53efdf041c8326deca84226b47c96f89f7cae632d5934351d70165d6a303"
    ),
    "quantdata/dysanos/flex/metrics.py": (
        "b8b48aa5c37f8ae290471d307325acf17b35f1ca1eb5370fb0f5b5ca0c1ca9e7"
    ),
    "quantdata/dysanos/flex/model.py": (
        "eead865900507f2b423f15e86806edea2dcc5e9f0ee58be3c145a67d39d39609"
    ),
    "quantdata/dysanos/flex/sanos.py": (
        "5b42b761e655f6482a5d9e34e7e3f533c6e4755cab9a4eb02287c42e290b0113"
    ),
    "quantdata/dysanos/flex/training.py": (
        "2b5d19f919932ed64eb6f7304a88551690771f72637517a769de6137f6c5a80a"
    ),
    "quantdata/dysanos/market.py": (
        "31aac8f19d83576d8863d9fa1ab050bca50f609bbc04c58bdf1eaf19cc8ddef6"
    ),
    "quantdata/pool.py": (
        "a2a84aa398180782db475eb483b551556917e1f434e4fb8986ff91ba5a7df032"
    ),
    "quantdata/sanos/__init__.py": (
        "bbeb6fe4060c6d30f2146a5ee60e249a6d65f49ca66b45b64b5ad1c5f526d683"
    ),
    "quantdata/sanos/arbfree.py": (
        "d26dd8236160786dc880f30631217d1a8183503019b748e4a6f9bfa0c02f27ca"
    ),
    "quantdata/sanos/sanos_tools.py": (
        "e250df0db749197f272c8807d2b5f342b07ae303a20096401e377d31da005b43"
    ),
    "quantdata/sanos/tridiag.py": (
        "97333b68e7f726b2e7273b360d1e8b4834f7c11d32212526ef98a75f8f90bf2b"
    ),
    "quantdata/torch/__init__.py": (
        "68197a7ea2977b79f2055efaf478cb56fef1f3809d56ea66cc7434324ae218f5"
    ),
    "quantdata/torch/ffw.py": (
        "65d80326bfdc239bdd1e6f7d442155007dcfc44fd5d8a4800a4a008758165e27"
    ),
    "quantdata/utils/__init__.py": (
        "dfcfff82cbc7a07ccf22766ba5fbfdac9e67ce3b55cda1db1b19f5d1a61cb8e8"
    ),
    "quantdata/utils/backbone.py": (
        "8b61f5b6d3e36ac8b957436fbdf8f0ab71d655ab0212d26475d79836f875a535"
    ),
    "quantdata/utils/cvxpy.py": (
        "681069bf29a56d165c822b203ee603f0741f8d4dbeb90528fab5e4fad335be05"
    ),
    "quantdata/utils/pd.py": (
        "402c4bb1eec258ad3acfa20c518a6d98e4bdd0b8f23448a41e4d29976148d60a"
    ),
    "quantdata/widb/__init__.py": (
        "7b4b353057a614eb233124b291313b063d6a32ca152e4dda16be9b65e5f029f1"
    ),
    "quantdata/widb/arbfree.py": (
        "eb77327259312ea3aecf74165b93c9fea0a44b0237e2d2f0795a362919d2b076"
    ),
    "quantdata/widb/bad.py": (
        "e25f81edea1a9828a2f710e89ff8695853ff87cb6c9c91d6d75fc10e36557286"
    ),
    "quantdata/widb/client.py": (
        "05875dbf217372de92a63e3fc319ad769218db1712bb6293ff192013eb8dfeb7"
    ),
    "quantdata/widb/day.py": (
        "9c15fb926534c730a1ab92626361dfc9da45549e9675a76c8c8a6db020c24e48"
    ),
    "quantdata/widb/options_filter.py": (
        "a2269ba838db0320bf612f566621cec36bbbd90228abdea68a752ed87a79a427"
    ),
}


@version("1.0")
def _worker_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@version("1.0")
def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@version("1.0")
def _strings_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@version("1.0", dependencies=[_array_digest])
def _compare_array(
    name: str,
    fresh: np.ndarray,
    legacy: np.ndarray,
    *,
    float_atol: float,
) -> dict[str, Any]:
    fresh_array = np.asarray(fresh)
    legacy_array = np.asarray(legacy)
    same_shape = fresh_array.shape == legacy_array.shape
    same_dtype = fresh_array.dtype == legacy_array.dtype
    record: dict[str, Any] = {
        "field": name,
        "fresh_dtype": fresh_array.dtype.str,
        "legacy_dtype": legacy_array.dtype.str,
        "fresh_shape": list(fresh_array.shape),
        "legacy_shape": list(legacy_array.shape),
        "fresh_sha256": _array_digest(fresh_array),
        "legacy_sha256": _array_digest(legacy_array),
        "atol": float_atol if fresh_array.dtype.kind == "f" else 0.0,
        "rtol": 0.0,
    }
    if not same_shape:
        record.update(
            {
                "matched": False,
                "mismatch_count": None,
                "maximum_absolute_error": None,
            }
        )
        return record
    if fresh_array.dtype.kind == "f" and legacy_array.dtype.kind == "f":
        fresh_finite = np.isfinite(fresh_array)
        legacy_finite = np.isfinite(legacy_array)
        finite_pattern = fresh_finite == legacy_finite
        both_finite = fresh_finite & legacy_finite
        differences = np.abs(
            fresh_array[both_finite].astype(np.float64)
            - legacy_array[both_finite].astype(np.float64)
        )
        both_nonfinite = ~fresh_finite & ~legacy_finite
        same_nonfinite = np.zeros(fresh_array.shape, dtype=np.bool_)
        same_nonfinite[both_nonfinite] = (
            (np.isnan(fresh_array[both_nonfinite]) & np.isnan(legacy_array[both_nonfinite]))
            | (fresh_array[both_nonfinite] == legacy_array[both_nonfinite])
        )
        nonfinite_mismatches = (~finite_pattern) | (both_nonfinite & ~same_nonfinite)
        mismatch_count = int(np.sum(nonfinite_mismatches)) + int(
            np.sum(differences > float_atol)
        )
        record.update(
            {
                "matched": same_dtype and mismatch_count == 0,
                "mismatch_count": mismatch_count,
                "nonzero_difference_count": int(np.sum(differences != 0.0)),
                "maximum_absolute_error": (
                    float(np.max(differences)) if differences.size else 0.0
                ),
            }
        )
    else:
        unequal = fresh_array != legacy_array
        mismatch_count = int(np.sum(unequal))
        record.update(
            {
                "matched": same_dtype and mismatch_count == 0,
                "mismatch_count": mismatch_count,
                "maximum_absolute_error": None,
            }
        )
    return record


@version(
    "quantdata-de15be96-flex-consumer-closure-v2",
    dependencies=[_worker_sha256_file],
)
def _load_pinned_consumer(code_root: Path) -> tuple[Any, Any, Any, Any, dict[str, str]]:
    observed: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = code_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Pinned quant_data source is missing or unsafe: {path}")
        digest = _worker_sha256_file(path)
        observed[relative] = digest
        if digest != expected:
            raise RuntimeError(
                f"Pinned quant_data source changed: {relative}; "
                f"expected {expected}, observed {digest}"
            )

    sys.path.insert(0, str(code_root))
    import quantdata.dysanos  # noqa: F401, PLC0415

    flex_package = types.ModuleType("quantdata.dysanos.flex")
    flex_package.__path__ = [str(code_root / "quantdata" / "dysanos" / "flex")]
    sys.modules["quantdata.dysanos.flex"] = flex_package
    from quantdata.dysanos.flex.config import (  # noqa: PLC0415
        FlexGeometryConfig,
        FlexMarketConfig,
    )
    from quantdata.dysanos.flex.market import market_day_from_payload  # noqa: PLC0415
    from quantdata.dysanos.flex.training import (  # noqa: PLC0415
        materialize_batch,
        materialize_day,
    )
    from quantdata.torch import BaseEnvironment  # noqa: PLC0415

    code_root_resolved = code_root.resolve(strict=True)
    loaded_sources: set[str] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "quantdata" and not module_name.startswith("quantdata."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            relative = Path(module_file).resolve(strict=True).relative_to(
                code_root_resolved
            )
        except ValueError:
            continue
        loaded_sources.add(relative.as_posix())
    unpinned = loaded_sources - set(PINNED_SOURCE_SHA256)
    if unpinned:
        raise RuntimeError(
            "Pinned quant_data consumer imported unpinned local source: "
            + ", ".join(sorted(unpinned))
        )

    return (
        FlexGeometryConfig,
        FlexMarketConfig,
        market_day_from_payload,
        (materialize_day, materialize_batch, BaseEnvironment),
        observed,
    )


@version("1.0")
def _legacy_price_geometry(day: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    locations: dict[tuple[int, float], int] = {}
    quote_price_index: list[int] = []
    price_expiry_y: list[float] = []
    price_strike: list[float] = []
    for expiry_index, strike in zip(day.expiry_index, day.strikes, strict=True):
        key = (int(expiry_index), float(strike))
        index = locations.get(key)
        if index is None:
            index = len(locations)
            locations[key] = index
            price_expiry_y.append(float(day.expiries_y[key[0]]))
            price_strike.append(key[1])
        quote_price_index.append(index)
    return (
        np.asarray(quote_price_index, dtype=np.int64),
        np.asarray(price_expiry_y, dtype=np.float64),
        np.asarray(price_strike, dtype=np.float64),
    )


@version(
    "1.1",
    dependencies=[_compare_array, _legacy_price_geometry, _strings_digest],
)
def _compare_logical_day(
    fresh: Any,
    legacy: Any,
    *,
    float_atol: float,
) -> dict[str, Any]:
    epoch = np.datetime64("1970-01-01", "D")
    legacy_expiry_days = np.asarray(
        [
            int((np.datetime64(value, "D") - epoch).astype(np.int64))
            for value in legacy.expiry_dates
        ],
        dtype=np.int32,
    )
    legacy_price_index, legacy_price_expiry, legacy_price_strike = (
        _legacy_price_geometry(legacy)
    )
    arrays = (
        ("expiry/time_y", fresh.expiry_time_y, legacy.expiries_y),
        ("expiry/date_days", fresh.expiry_date_days, legacy_expiry_days),
        ("expiry/id", fresh.expiry_id, legacy.expiry_ids),
        ("expiry/is_am", fresh.expiry_is_am, legacy.expiry_is_am),
        (
            "expiry/atm_annual_vol",
            fresh.expiry_atm_annual_vol,
            legacy.atm_annual_vols,
        ),
        ("expiry/strike_lower", fresh.expiry_strike_lower, legacy.strike_lower_bounds),
        ("expiry/strike_upper", fresh.expiry_strike_upper, legacy.strike_upper_bounds),
        ("model/atm_total_vol", fresh.model_atm_total_vol, legacy.model_atm_total_vols),
        ("quote/expiry_index", fresh.quote_expiry_index, legacy.expiry_index),
        ("quote/price_index", fresh.quote_price_index, legacy_price_index),
        ("quote/strike", fresh.quote_strike, legacy.strikes),
        ("quote/is_call", fresh.quote_is_call, legacy.is_call),
        ("quote/bid", fresh.quote_bid, legacy.bids),
        ("quote/ask", fresh.quote_ask, legacy.asks),
        ("quote/fit", fresh.quote_fit, legacy.fit_prices),
        ("quote/log_volume", fresh.quote_log_volume, legacy.log_volumes),
        (
            "quote/log_open_interest",
            fresh.quote_log_open_interest,
            legacy.log_open_interests,
        ),
        ("price/expiry_y", fresh.price_expiry_y, legacy_price_expiry),
        ("price/strike", fresh.price_strike, legacy_price_strike),
        (
            "day/spot_reference",
            np.asarray([fresh.spot_reference], dtype=np.float64),
            np.asarray([legacy.spot_ref], dtype=np.float64),
        ),
        (
            "audit/counts",
            np.asarray(dataclasses.astuple(fresh.audit), dtype=np.int64),
            np.asarray(dataclasses.astuple(legacy.audit), dtype=np.int64),
        ),
    )
    comparisons = [
        _compare_array(name, np.asarray(left), np.asarray(right), float_atol=float_atol)
        for name, left, right in arrays
    ]
    fresh_quote_ids = fresh.quote_ids
    legacy_quote_ids = legacy.quote_ids
    quote_ids_match = fresh_quote_ids == legacy_quote_ids
    strings = {
        "quote_ids": {
            "matched": quote_ids_match,
            "count": len(fresh_quote_ids),
            "fresh_sha256": _strings_digest(fresh_quote_ids),
            "legacy_sha256": _strings_digest(legacy_quote_ids),
        }
    }
    rebased_legacy = dataclasses.replace(legacy, market_uid=fresh.market_uid)
    identity = {
        "ticker_matched": fresh.ticker == legacy.ticker,
        "date_matched": fresh.trading_date == legacy.trading_date,
        "legacy_market_uid": legacy.market_uid,
        "fresh_market_uid": fresh.market_uid,
        "market_uid_expected_to_change": True,
        "fresh_market_fingerprint": fresh.market_fingerprint,
        "legacy_training_fingerprint": legacy.training_fingerprint,
        "rebased_legacy_training_fingerprint": rebased_legacy.training_fingerprint,
        "fresh_training_fingerprint": fresh.training_fingerprint,
        "rebased_training_fingerprint_matched": (
            rebased_legacy.training_fingerprint == fresh.training_fingerprint
        ),
    }
    mismatch_fields = [
        str(record["field"]) for record in comparisons if not record["matched"]
    ]
    if not quote_ids_match:
        mismatch_fields.append("quote_ids")
    for key in ("ticker_matched", "date_matched", "rebased_training_fingerprint_matched"):
        if not identity[key]:
            mismatch_fields.append(key)
    return {
        "matched": not mismatch_fields,
        "mismatch_fields": sorted(mismatch_fields),
        "comparisons": comparisons,
        "strings": strings,
        "identity": identity,
        "population": {
            "quotes": fresh.quote_count,
            "prices": fresh.price_count,
            "expiries": fresh.expiry_count,
            "calls": int(np.sum(fresh.quote_is_call)),
            "puts": int(np.sum(~fresh.quote_is_call)),
        },
    }


@version("1.0", dependencies=[_compare_array])
def _compare_torch_day(
    fresh: Any,
    legacy: Any,
    *,
    materialize_day: Any,
    environment: Any,
    float_atol: float,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    fresh_day = materialize_day(fresh, env=environment)
    legacy_day = materialize_day(legacy, env=environment)
    fields = (
        "expiries_y",
        "strikes",
        "price_expiries_y",
        "price_strikes",
        "quote_price_index",
        "is_call",
        "bids",
        "asks",
        "fit_prices",
    )
    records = [
        _compare_array(
            f"consumer/day/{name}",
            getattr(fresh_day, name).detach().cpu().numpy(),
            getattr(legacy_day, name).detach().cpu().numpy(),
            float_atol=float_atol,
        )
        for name in fields
    ]
    return fresh_day, legacy_day, records


@version("1.0", dependencies=[_compare_array])
def _compare_torch_batch(
    fresh_days: tuple[Any, ...],
    legacy_days: tuple[Any, ...],
    *,
    materialize_batch: Any,
    environment: Any,
    batch_index: int,
    float_atol: float,
) -> dict[str, Any]:
    indices = np.arange(len(fresh_days), dtype=np.int64)
    fresh = materialize_batch(fresh_days, indices, env=environment)
    legacy = materialize_batch(legacy_days, indices, env=environment)
    fields = (
        "day_indices",
        "strikes",
        "price_expiries_y",
        "price_strikes",
        "quote_price_index",
        "is_call",
        "bids",
        "asks",
        "fit_prices",
        "mask",
    )
    comparisons = [
        _compare_array(
            f"consumer/batch/{name}",
            getattr(fresh, name).detach().cpu().numpy(),
            getattr(legacy, name).detach().cpu().numpy(),
            float_atol=float_atol,
        )
        for name in fields
    ]
    return {
        "batch_index": batch_index,
        "days": len(fresh_days),
        "matched": all(item["matched"] for item in comparisons),
        "comparisons": comparisons,
    }


@version(
    "1.0",
    dependencies=[
        ExperimentConfig,
        _compare_logical_day,
        _compare_torch_batch,
        _compare_torch_day,
        _load_pinned_consumer,
        _worker_sha256_file,
        canonical_value,
        exact_weekly_pairs,
        read_hdf5_day,
        read_legacy_pytree_hdf5,
    ],
)
def run(request: Mapping[str, Any]) -> dict[str, Any]:
    experiment_hdf = Path(str(request["experiment_hdf"]))
    legacy_root = Path(str(request["legacy_root"]))
    catalog_path = Path(str(request["catalog_path"]))
    code_root = Path(str(request["legacy_code_root"]))
    float64_atol = float(request["float64_atol"])
    float32_atol = float(request["float32_atol"])
    progress_interval = float(request["progress_interval_seconds"])
    batch_size = int(request["consumer_batch_size"])
    config = ExperimentConfig.from_document(request["experiment_config"])
    if batch_size < 1 or batch_size > 256:
        raise ValueError("consumer_batch_size must lie in [1, 256]")

    (
        Geometry,
        Market,
        market_day_from_payload,
        consumer,
        source_hashes,
    ) = _load_pinned_consumer(code_root)
    materialize_day, materialize_batch, BaseEnvironment = consumer
    geometry = Geometry(
        num_inner_strikes=config.geometry.num_inner_strikes,
        nms_range=config.geometry.nms_range,
        num_model_expiries=len(config.geometry.model_business_dtes),
        explicit_model_business_dtes=config.geometry.model_business_dtes,
    )
    market = Market(
        max_num_options=config.market.max_num_options,
        smoothness=config.market.smoothness,
        excluded_business_dtes=config.market.excluded_business_dtes,
    )
    environment = BaseEnvironment(precision=32, device="cpu", debug=True)

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = [record for record in catalog["files"] if record["populated"]]
    legacy_dates = tuple(date.fromisoformat(record["trading_date"]) for record in records)
    h5py = __import__("h5py")
    started = time.monotonic()
    last_progress = started
    days: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    fresh_batch: list[Any] = []
    legacy_batch: list[Any] = []
    legacy_books = 0
    legacy_calls = 0
    legacy_qmax = 0
    legacy_pmax = 0
    legacy_emax = 0
    with h5py.File(experiment_hdf, "r", swmr=True) as handle:
        fresh_date_values = np.asarray(handle["day/date_days"][:], dtype=np.int32)
        epoch = np.datetime64("1970-01-01", "D")
        fresh_dates = tuple(
            (epoch + np.timedelta64(int(value), "D")).astype(object)
            for value in fresh_date_values
        )
        date_to_index = {value: index for index, value in enumerate(fresh_dates)}
        if len(date_to_index) != len(fresh_dates):
            raise RuntimeError("Fresh Experiment contains duplicate dates")
        missing = [value.isoformat() for value in legacy_dates if value not in date_to_index]
        if missing:
            raise RuntimeError(f"Fresh Experiment omits legacy populated dates: {missing}")
        for completed, (record, trading_date) in enumerate(
            zip(records, legacy_dates, strict=True), start=1
        ):
            legacy_path = legacy_root / record["file"]
            if legacy_path.stat().st_size != record["byte_size"]:
                raise RuntimeError(f"Legacy file size changed: {legacy_path}")
            digest = _worker_sha256_file(legacy_path)
            if digest != record["sha256"]:
                raise RuntimeError(f"Legacy file checksum changed: {legacy_path}")
            payload = read_legacy_pytree_hdf5(legacy_path)
            legacy_day = market_day_from_payload(
                payload,
                ticker=str(request["ticker"]),
                geometry=geometry,
                config=market,
            )
            if legacy_day is None:
                raise RuntimeError(f"Pinned selector returned no day for {trading_date}")
            fresh_day = read_hdf5_day(handle, date_to_index[trading_date])
            logical = _compare_logical_day(
                fresh_day,
                legacy_day,
                float_atol=float64_atol,
            )
            fresh_torch, legacy_torch, tensor_records = _compare_torch_day(
                fresh_day,
                legacy_day,
                materialize_day=materialize_day,
                environment=environment,
                float_atol=float32_atol,
            )
            fresh_batch.append(fresh_torch)
            legacy_batch.append(legacy_torch)
            tensor_matched = all(item["matched"] for item in tensor_records)
            days.append(
                {
                    "date": trading_date.isoformat(),
                    "classification": (
                        "match" if logical["matched"] and tensor_matched else "discrepancy"
                    ),
                    "legacy_file": record["file"],
                    "legacy_file_sha256": digest,
                    "logical": logical,
                    "consumer": {
                        "matched": tensor_matched,
                        "comparisons": tensor_records,
                    },
                }
            )
            population = logical["population"]
            legacy_books += int(population["quotes"])
            legacy_calls += int(population["calls"])
            legacy_qmax = max(legacy_qmax, int(population["quotes"]))
            legacy_pmax = max(legacy_pmax, int(population["prices"]))
            legacy_emax = max(legacy_emax, int(population["expiries"]))
            if len(fresh_batch) == batch_size or completed == len(records):
                batch_records.append(
                    _compare_torch_batch(
                        tuple(fresh_batch),
                        tuple(legacy_batch),
                        materialize_batch=materialize_batch,
                        environment=environment,
                        batch_index=len(batch_records),
                        float_atol=float32_atol,
                    )
                )
                fresh_batch.clear()
                legacy_batch.clear()
            now = time.monotonic()
            if completed == len(records) or now - last_progress >= progress_interval:
                elapsed = now - started
                rate = completed / elapsed if elapsed > 0.0 else 0.0
                _emit(
                    {
                        "kind": "progress",
                        "event": "experiment_reference_comparison_progress",
                        "completed": completed,
                        "total": len(records),
                        "failures": sum(
                            item["classification"] == "discrepancy" for item in days
                        ),
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": (
                            (len(records) - completed) / rate if rate > 0.0 else None
                        ),
                    }
                )
                last_progress = now

    fresh_only = tuple(
        value.isoformat() for value in fresh_dates if value not in set(legacy_dates)
    )
    pre_2025 = sum(value < date(2025, 1, 1) for value in legacy_dates)
    weekly_origin, _ = exact_weekly_pairs(legacy_dates)
    mismatch_dates = tuple(
        item["date"] for item in days if item["classification"] == "discrepancy"
    )
    return canonical_value(
        {
            "source_hashes": source_hashes,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(sys.modules["torch"].__version__),
            },
            "dates": days,
            "consumer_batches": batch_records,
            "summary": {
                "fresh_dates": len(fresh_dates),
                "common_dates": len(legacy_dates),
                "fresh_only_dates": fresh_only,
                "mismatch_dates": mismatch_dates,
                "legacy_books": legacy_books,
                "legacy_calls": legacy_calls,
                "legacy_puts": legacy_books - legacy_calls,
                "legacy_qmax": legacy_qmax,
                "legacy_pmax": legacy_pmax,
                "legacy_emax": legacy_emax,
                "legacy_pre_2025_dates": pre_2025,
                "legacy_from_2025_dates": len(legacy_dates) - pre_2025,
                "legacy_weekly_pairs": len(weekly_origin),
                "consumer_batches": len(batch_records),
                "consumer_batch_mismatches": sum(
                    not item["matched"] for item in batch_records
                ),
            },
        }
    )


@version("1.0", dependencies=[run])
def experiment_oracle_worker_version() -> str:
    """Return the resolved content version of the isolated oracle worker."""

    return str(experiment_oracle_worker_version.version)


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":")), flush=True)


@version("1.0")
def _runtime_preflight() -> dict[str, Any]:
    """Prove the installed worker can import its optional runtime dependencies."""

    h5py = __import__("h5py")
    torch = __import__("torch")
    return {
        "protocol": "sanos-experiment-oracle-worker-preflight-v1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": str(h5py.__version__),
        "torch": str(torch.__version__),
        "pinned_source_files": sorted(PINNED_SOURCE_SHA256),
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, Mapping):
            raise ValueError("Worker request must be a JSON object")
        result = _runtime_preflight() if request.get("operation") == "preflight" else run(request)
        _emit({"kind": "result", "result": result})
        return 0
    except BaseException as exc:  # worker boundary must return structured failure
        _emit({"kind": "error", "error_type": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through parent process
    raise SystemExit(main())
