"""Read the legacy DySANOS HDF5 cache without invoking its cache machinery.

The adapter deliberately has no write method.  It understands only the
``cdxcore.subdir.PYTREE_HDF5`` representation used by the pinned reference
dataset and opens every file in read-only mode.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.version import version


@version("1.0")
@dataclass(frozen=True)
class _LegacyReferenceSchema:
    """Versioned constants that determine accepted and catalogued content."""

    catalog_schema: str = "sanos-legacy-market-catalog-v1"
    payload_schema_version: int = 4
    days_per_year: float = 252.0
    point_two_business_dte: float = 0.2
    business_dte_tolerance: float = 1.0e-4
    required_payload_keys: tuple[str, ...] = (
        "af_pricer",
        "d",
        "fit_result",
        "market",
        "schema_version",
        "spot_ref",
        "spot_ret",
        "ticker",
    )
    required_market_keys: tuple[str, ...] = (
        "call_active",
        "expiries",
        "expiry_dates",
        "expiry_ids",
        "is_am",
        "put_active",
        "strikes",
    )


_REFERENCE_SCHEMA = _LegacyReferenceSchema()


class ReferenceFormatError(ValueError):
    """Raised when a reference artifact does not satisfy the pinned format."""


def _require_h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading the legacy reference requires h5py; install sanos[reference]"
        ) from error


@version("1.0")
def _decode_text(value: object, *, location: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    raise ReferenceFormatError(f"Expected UTF-8 text at {location}, found {type(value)}")


@version("1.0", dependencies=[_decode_text])
def _read_datetime64_dataset(dataset: Any, *, location: str) -> np.ndarray:
    if dataset.attrs.get("__numpy_kind__") != "datetime64":
        raise ReferenceFormatError(
            f"Expected datetime64 metadata at {location}, found {dict(dataset.attrs)}"
        )
    dtype = np.dtype(_decode_text(dataset.attrs["__numpy_dtype__"], location=location))
    return np.asarray(dataset[...], dtype=np.int64).view(dtype)


@version("1.0", dependencies=[_decode_text, _read_datetime64_dataset])
def _read_group(h5py: Any, group: Any, *, location: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for stored_key, node in group.items():
        key = str(stored_key)
        if isinstance(node, h5py.Dataset):
            value = node[()]
            if key.startswith("_1_N"):
                result[key[4:]] = value
            elif key.startswith("_1_S"):
                result[key[4:]] = _decode_text(value, location=f"{location}.{key}")
            elif key.startswith("_1_D"):
                result[key[4:]] = dt.date.fromisoformat(
                    _decode_text(value, location=f"{location}.{key}")
                )
            elif key.startswith("_1_d"):
                result[key[4:]] = dt.datetime.fromisoformat(
                    _decode_text(value, location=f"{location}.{key}")
                )
            elif key.startswith("_1_n"):
                encoded, unit, count = _decode_text(
                    value, location=f"{location}.{key}"
                ).split("@")
                count_value = int(count)
                unit_prefix = "" if count_value == 1 else str(count_value)
                dtype = np.dtype(f"datetime64[{unit_prefix}{unit}]")
                result[key[4:]] = np.asarray(encoded, dtype=dtype)[()]
            elif key.startswith("_1_T"):
                result[key[4:]] = dt.time.fromisoformat(
                    _decode_text(value, location=f"{location}.{key}")
                )
            elif key.startswith("_1_Q"):
                result[key[4:]] = _read_datetime64_dataset(
                    node, location=f"{location}.{key}"
                )
            else:
                result[key] = value
            continue

        if not isinstance(node, h5py.Group):
            raise ReferenceFormatError(
                f"Unsupported HDF5 node at {location}.{key}: {type(node)}"
            )
        if key.startswith("_1_L"):
            logical_key = key[4:]
            members = _read_group(h5py, node, location=f"{location}.{logical_key}")
            expected = {str(index) for index in range(len(members))}
            if set(members) != expected:
                raise ReferenceFormatError(
                    f"List at {location}.{logical_key} has non-contiguous keys: "
                    f"{sorted(members)}"
                )
            result[logical_key] = [members[str(index)] for index in range(len(members))]
        else:
            result[key] = _read_group(h5py, node, location=f"{location}.{key}")
    return result


@version("1.1", dependencies=[_read_group])
def read_legacy_pytree_hdf5(path: str | Path) -> dict[str, object]:
    """Read one legacy pytree HDF5 file, strictly in mode ``"r"``."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Reference artifact does not exist: {artifact}")
    h5py = _require_h5py()
    try:
        with h5py.File(artifact, "r") as handle:
            if set(handle) != {"root"}:
                raise ReferenceFormatError(
                    f"Expected the sole HDF5 root group 'root' in {artifact}, "
                    f"found {sorted(handle)}"
                )
            return _read_group(h5py, handle["root"], location="root")
    except ReferenceFormatError:
        raise
    except OSError as error:
        raise ReferenceFormatError(
            f"Cannot read legacy HDF5 artifact {artifact}: {error}"
        ) from error
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise ReferenceFormatError(
            f"Malformed legacy HDF5 encoding in {artifact}: {error}"
        ) from error


@version("1.0")
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@version("1.0")
def _as_mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceFormatError(f"Expected a mapping at {location}, found {type(value)}")
    return value


@version("1.0")
def _as_list(value: object, *, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ReferenceFormatError(f"Expected a list at {location}, found {type(value)}")
    return value


@version("1.0")
def _as_array(
    value: object,
    *,
    location: str,
    dtype_kinds: frozenset[str],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ReferenceFormatError(
            f"Expected a NumPy array at {location}, found {type(value)}"
        )
    if value.ndim != 1:
        raise ReferenceFormatError(
            f"Expected a one-dimensional array at {location}, found shape {value.shape}"
        )
    if value.dtype.kind not in dtype_kinds:
        raise ReferenceFormatError(
            f"Expected dtype kind {sorted(dtype_kinds)} at {location}, "
            f"found {value.dtype}"
        )
    return value


@version("1.0")
def _finite_real(value: object, *, location: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ReferenceFormatError(
            f"Expected a finite real number at {location}, found {type(value)}"
        )
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ReferenceFormatError(
            f"Expected a finite {qualifier}number at {location}, found {result!r}"
        )
    return result


@version("1.0")
@dataclass(frozen=True)
class _ValidatedLegacyPayload:
    schema_version: int
    ticker: str
    market: Mapping[str, object]
    expiries: np.ndarray
    strikes: tuple[np.ndarray, ...]
    call_active: tuple[np.ndarray, ...]
    put_active: tuple[np.ndarray, ...]


@version(
    "1.0",
    dependencies=[
        _LegacyReferenceSchema,
        _ValidatedLegacyPayload,
        _as_array,
        _as_list,
        _as_mapping,
        _finite_real,
    ],
)
def _validate_payload(
    payload: Mapping[str, object],
    *,
    path: Path,
    expected_date: dt.date,
) -> _ValidatedLegacyPayload:
    missing = set(_REFERENCE_SCHEMA.required_payload_keys) - set(payload)
    if missing:
        raise ReferenceFormatError(f"{path} is missing payload keys {sorted(missing)}")
    if payload["d"] != expected_date:
        raise ReferenceFormatError(
            f"{path} stores date {payload['d']!r}, expected {expected_date} from its name"
        )

    schema_value = payload["schema_version"]
    if isinstance(schema_value, (bool, np.bool_)) or not isinstance(
        schema_value, (int, np.integer)
    ):
        raise ReferenceFormatError(
            f"{path} has non-integer payload schema version {schema_value!r}"
        )
    schema_version = int(schema_value)
    if schema_version != _REFERENCE_SCHEMA.payload_schema_version:
        raise ReferenceFormatError(
            f"{path} has payload schema version {schema_version}, expected "
            f"{_REFERENCE_SCHEMA.payload_schema_version}"
        )

    ticker = payload["ticker"]
    if not isinstance(ticker, str) or not ticker:
        raise ReferenceFormatError(f"{path} has invalid ticker {ticker!r}")
    _finite_real(payload["spot_ref"], location=f"{path}:spot_ref", positive=True)
    _finite_real(payload["spot_ret"], location=f"{path}:spot_ret")
    _as_mapping(payload["af_pricer"], location=f"{path}:af_pricer")
    _as_mapping(payload["fit_result"], location=f"{path}:fit_result")

    market = _as_mapping(payload["market"], location=f"{path}:market")
    missing_market = set(_REFERENCE_SCHEMA.required_market_keys) - set(market)
    if missing_market:
        raise ReferenceFormatError(f"{path} is missing market keys {sorted(missing_market)}")

    expiries = _as_array(
        market["expiries"],
        location=f"{path}:market.expiries",
        dtype_kinds=frozenset({"f"}),
    )
    if (
        len(expiries) == 0
        or not np.all(np.isfinite(expiries))
        or not np.all(expiries > 0.0)
        or (len(expiries) > 1 and not np.all(np.diff(expiries) > 0.0))
    ):
        raise ReferenceFormatError(
            f"{path}:market.expiries must be finite, positive and strictly increasing"
        )

    expiry_dates = _as_list(
        market["expiry_dates"], location=f"{path}:market.expiry_dates"
    )
    for index, value in enumerate(expiry_dates):
        if not isinstance(value, str):
            raise ReferenceFormatError(
                f"Expected an ISO date string at {path}:market.expiry_dates[{index}], "
                f"found {type(value)}"
            )
        try:
            dt.date.fromisoformat(value)
        except ValueError as error:
            raise ReferenceFormatError(
                f"Invalid ISO date at {path}:market.expiry_dates[{index}]: {value!r}"
            ) from error

    expiry_ids = _as_array(
        market["expiry_ids"],
        location=f"{path}:market.expiry_ids",
        dtype_kinds=frozenset({"i", "u"}),
    )
    is_am = _as_array(
        market["is_am"],
        location=f"{path}:market.is_am",
        dtype_kinds=frozenset({"b"}),
    )
    strikes_values = _as_list(market["strikes"], location=f"{path}:market.strikes")
    call_values = _as_list(
        market["call_active"], location=f"{path}:market.call_active"
    )
    put_values = _as_list(
        market["put_active"], location=f"{path}:market.put_active"
    )
    lengths = {
        "expiries": len(expiries),
        "expiry_dates": len(expiry_dates),
        "expiry_ids": len(expiry_ids),
        "is_am": len(is_am),
        "strikes": len(strikes_values),
        "call_active": len(call_values),
        "put_active": len(put_values),
    }
    if len(set(lengths.values())) != 1:
        raise ReferenceFormatError(f"{path} has inconsistent expiry lengths: {lengths}")
    if np.any(expiry_ids <= 0) or (
        len(expiry_ids) > 1 and not np.all(np.diff(expiry_ids) > 0)
    ):
        raise ReferenceFormatError(
            f"{path}:market.expiry_ids must be positive and strictly increasing"
        )

    strikes: list[np.ndarray] = []
    calls: list[np.ndarray] = []
    puts: list[np.ndarray] = []
    for index, (strike_value, call_value, put_value) in enumerate(
        zip(strikes_values, call_values, put_values, strict=True)
    ):
        strike = _as_array(
            strike_value,
            location=f"{path}:market.strikes[{index}]",
            dtype_kinds=frozenset({"f"}),
        )
        call = _as_array(
            call_value,
            location=f"{path}:market.call_active[{index}]",
            dtype_kinds=frozenset({"b"}),
        )
        put = _as_array(
            put_value,
            location=f"{path}:market.put_active[{index}]",
            dtype_kinds=frozenset({"b"}),
        )
        if (
            len(strike) == 0
            or not np.all(np.isfinite(strike))
            or not np.all(strike > 0.0)
            or (len(strike) > 1 and not np.all(np.diff(strike) > 0.0))
        ):
            raise ReferenceFormatError(
                f"{path}:market.strikes[{index}] must be finite, positive and "
                "strictly increasing"
            )
        if call.shape != strike.shape or put.shape != strike.shape:
            raise ReferenceFormatError(
                f"{path} expiry {index} has strike/mask shape mismatch: "
                f"strikes={strike.shape}, calls={call.shape}, puts={put.shape}"
            )
        strikes.append(strike)
        calls.append(call)
        puts.append(put)

    return _ValidatedLegacyPayload(
        schema_version=schema_version,
        ticker=ticker,
        market=market,
        expiries=expiries,
        strikes=tuple(strikes),
        call_active=tuple(calls),
        put_active=tuple(puts),
    )


@version("1.0")
@dataclass(frozen=True)
class LegacyDayRecord:
    """Deterministic content and population summary for one physical file."""

    file: str
    byte_size: int
    sha256: str
    populated: bool
    trading_date: str
    producer_version: str | None
    payload_schema_version: int | None
    ticker: str | None
    num_expiries: int
    num_strike_nodes: int
    active_call_books: int
    active_put_books: int
    active_point_two_day_books: int

    @property
    def active_books(self) -> int:
        return self.active_call_books + self.active_put_books

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["active_books"] = self.active_books
        return result


@version("1.1", dependencies=[_LegacyReferenceSchema, LegacyDayRecord])
@dataclass(frozen=True)
class LegacyMarketCatalog:
    """Content-addressed inventory of one legacy daily-market directory."""

    source_revision: str
    relative_path: str
    collection_sha256: str
    records: tuple[LegacyDayRecord, ...]

    @property
    def populated_records(self) -> tuple[LegacyDayRecord, ...]:
        return tuple(record for record in self.records if record.populated)

    def to_dict(self) -> dict[str, object]:
        populated = self.populated_records
        producer_versions = sorted(
            {record.producer_version for record in self.records if record.producer_version}
        )
        payload_versions = sorted(
            {
                record.payload_schema_version
                for record in populated
                if record.payload_schema_version is not None
            }
        )
        tickers = sorted({record.ticker for record in populated if record.ticker})
        return {
            "schema": _REFERENCE_SCHEMA.catalog_schema,
            "source_revision": self.source_revision,
            "root_locator": "legacy_QUANTDATA_TRAINING_DATA",
            "relative_path": self.relative_path,
            "format": "cdxcore.subdir.PYTREE_HDF5",
            "collection_sha256": self.collection_sha256,
            "physical_file_count": len(self.records),
            "populated_day_count": len(populated),
            "empty_marker_dates": [
                record.trading_date for record in self.records if not record.populated
            ],
            "first_populated_date": populated[0].trading_date if populated else None,
            "last_populated_date": populated[-1].trading_date if populated else None,
            "total_bytes": sum(record.byte_size for record in self.records),
            "producer_versions": producer_versions,
            "payload_schema_versions": payload_versions,
            "tickers": tickers,
            "aggregate": {
                "expiries": sum(record.num_expiries for record in populated),
                "strike_nodes": sum(record.num_strike_nodes for record in populated),
                "active_call_books": sum(record.active_call_books for record in populated),
                "active_put_books": sum(record.active_put_books for record in populated),
                "active_books": sum(record.active_books for record in populated),
                "active_point_two_day_books": sum(
                    record.active_point_two_day_books for record in populated
                ),
            },
            "collection_hash_algorithm": (
                "sha256 over UTF-8 file name, NUL, decimal byte size, NUL, "
                "ASCII file sha256, newline, in lexical file-name order"
            ),
            "files": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


ProgressCallback = Callable[[int, int, Path], None]


@version("1.0", dependencies=[_decode_text])
def _producer_version(path: Path) -> str | None:
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("version")
    return None if value is None else _decode_text(value, location=f"{path}:version")


@version(
    "1.1",
    dependencies=[
        _LegacyReferenceSchema,
        _producer_version,
        _sha256,
        _validate_payload,
        read_legacy_pytree_hdf5,
    ],
)
def _record(path: Path) -> LegacyDayRecord:
    try:
        trading_date = dt.date.fromisoformat(path.stem)
    except ValueError as error:
        raise ReferenceFormatError(
            f"Reference file name must be an ISO trading date: {path.name}"
        ) from error

    payload = read_legacy_pytree_hdf5(path)
    producer_version = _producer_version(path)
    byte_size = path.stat().st_size
    sha256 = _sha256(path)
    if not payload:
        return LegacyDayRecord(
            file=path.name,
            byte_size=byte_size,
            sha256=sha256,
            populated=False,
            trading_date=trading_date.isoformat(),
            producer_version=producer_version,
            payload_schema_version=None,
            ticker=None,
            num_expiries=0,
            num_strike_nodes=0,
            active_call_books=0,
            active_put_books=0,
            active_point_two_day_books=0,
        )

    validated = _validate_payload(payload, path=path, expected_date=trading_date)

    active_call_books = 0
    active_put_books = 0
    point_two_books = 0
    strike_nodes = 0
    for index, expiry in enumerate(validated.expiries):
        node_count = len(validated.strikes[index])
        calls = validated.call_active[index]
        puts = validated.put_active[index]
        call_count = int(calls.sum())
        put_count = int(puts.sum())
        strike_nodes += node_count
        active_call_books += call_count
        active_put_books += put_count
        business_dte = float(expiry) * _REFERENCE_SCHEMA.days_per_year
        if (
            abs(business_dte - _REFERENCE_SCHEMA.point_two_business_dte)
            <= _REFERENCE_SCHEMA.business_dte_tolerance
        ):
            point_two_books += call_count + put_count

    return LegacyDayRecord(
        file=path.name,
        byte_size=byte_size,
        sha256=sha256,
        populated=True,
        trading_date=trading_date.isoformat(),
        producer_version=producer_version,
        payload_schema_version=validated.schema_version,
        ticker=validated.ticker,
        num_expiries=len(validated.expiries),
        num_strike_nodes=strike_nodes,
        active_call_books=active_call_books,
        active_put_books=active_put_books,
        active_point_two_day_books=point_two_books,
    )


@version(
    "1.1",
    dependencies=[
        _LegacyReferenceSchema,
        _record,
        LegacyDayRecord,
        LegacyMarketCatalog,
    ],
)
def catalog_legacy_market(
    root: str | Path,
    *,
    source_revision: str,
    relative_path: str,
    progress: ProgressCallback | None = None,
) -> LegacyMarketCatalog:
    """Hash and inspect every daily HDF5 file beneath ``root``."""

    directory = Path(root)
    if not directory.is_dir():
        raise NotADirectoryError(f"Reference market root does not exist: {directory}")
    paths = tuple(sorted(directory.glob("*.h5"), key=lambda path: path.name))
    if not paths:
        raise FileNotFoundError(f"No .h5 reference files found beneath {directory}")

    records: list[LegacyDayRecord] = []
    total = len(paths)
    for completed, path in enumerate(paths, start=1):
        records.append(_record(path))
        if progress is not None:
            progress(completed, total, path)

    collection = hashlib.sha256()
    for record in records:
        collection.update(record.file.encode("utf-8"))
        collection.update(b"\0")
        collection.update(str(record.byte_size).encode("ascii"))
        collection.update(b"\0")
        collection.update(record.sha256.encode("ascii"))
        collection.update(b"\n")
    return LegacyMarketCatalog(
        source_revision=source_revision,
        relative_path=relative_path,
        collection_sha256=collection.hexdigest(),
        records=tuple(records),
    )


@version("1.1", dependencies=[_validate_payload, read_legacy_pytree_hdf5])
@dataclass(frozen=True)
class LegacyMarketReference:
    """Read-only, date-addressed view of the legacy daily market collection."""

    root: Path

    def __post_init__(self) -> None:
        if not self.root.is_dir():
            raise NotADirectoryError(f"Reference market root does not exist: {self.root}")

    def physical_dates(self) -> tuple[dt.date, ...]:
        dates: list[dt.date] = []
        for path in sorted(self.root.glob("*.h5"), key=lambda item: item.name):
            try:
                dates.append(dt.date.fromisoformat(path.stem))
            except ValueError as error:
                raise ReferenceFormatError(
                    f"Reference file name must be an ISO trading date: {path.name}"
                ) from error
        return tuple(dates)

    def load_day(self, trading_date: dt.date) -> dict[str, object] | None:
        path = self.root / f"{trading_date.isoformat()}.h5"
        payload = read_legacy_pytree_hdf5(path)
        if not payload:
            return None
        _validate_payload(payload, path=path, expected_date=trading_date)
        return payload

    def iter_populated_days(self) -> Iterator[tuple[dt.date, dict[str, object]]]:
        for trading_date in self.physical_dates():
            payload = self.load_day(trading_date)
            if payload is not None:
                yield trading_date, payload
