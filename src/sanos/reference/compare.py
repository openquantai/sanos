"""Read-only parity comparison between fitted SANOS days and the legacy oracle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from cdxcore.version import version

from sanos.artifacts import ArtifactStore, ValidatedArtifact, atomic_write_json
from sanos.config import StorageLocations, canonical_sha256, canonical_value
from sanos.datasets import InternalDatasetConfig
from sanos.datasets.internal import validate_fitted_partition
from sanos.model.arbfree import ArbFreePricer
from sanos.model.tools import SanosPricer
from sanos.reference.legacy_market import read_legacy_pytree_hdf5
from sanos.widb.build import IntermediateBuildRequest, plan_daily_artifacts, read_daily_frame
from sanos.widb.config import DailySurfaceConfig, DailyWidbConfig
from sanos.widb.surface import (
    FITTED_SCHEMA,
    compatibility_dlv_pricer,
    selected_reference_view,
    smooth_sanos_pricer,
)
from sanos.wrds import raw_artifact_validation_record

REFERENCE_REPORT_SCHEMA = "sanos-legacy-daily-comparison-v1"
APPROVAL_SCHEMA = "sanos-legacy-daily-comparison-approvals-v1"
PINNED_CATALOG_SHA256 = "9b68a9638f273271ff4dbde6365c75ac9fd749993f9284b42428682339d516b4"
PINNED_COLLECTION_SHA256 = "f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280"
PHASE10_PILOT_DATES = (
    "2024-12-23",
    "2024-12-24",
    "2024-12-26",
    "2024-12-27",
    "2024-12-30",
    "2024-12-31",
    "2025-01-02",
    "2025-01-03",
    "2025-01-06",
    "2025-01-07",
    "2025-01-08",
    "2025-01-10",
    "2025-01-13",
    "2025-01-14",
    "2025-01-15",
    "2025-01-16",
)
PHASE10_PILOT_ORACLE_GAPS = ("2024-12-24",)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")

DailyClassification = Literal[
    "match",
    "oracle_gap",
    "policy_empty",
    "unexplained_discrepancy",
    "approved_discrepancy",
]


class ReferenceComparisonError(RuntimeError):
    """Raised when comparison evidence is missing, malformed, or unsafe."""


@version("phase10-legacy-daily-comparison-tolerances-v1")
@dataclass(frozen=True, slots=True)
class DailyLegacyComparisonTolerances:
    """Frozen parity and invariant tolerances for the migration oracle."""

    float32_atol: float = 0.0
    float64_atol: float = 1.0e-12
    reconstructed_call_atol: float = 5.0e-8
    density_dlv_inverse_atol: float = 2.0e-14
    density_calendar_validity: float = 1.0e-7
    quote_containment_validity: float = 1.0e-6
    density_mass_moment_validity: float = 1.0e-4

    def __post_init__(self) -> None:
        expected = {
            "float32_atol": 0.0,
            "float64_atol": 1.0e-12,
            "reconstructed_call_atol": 5.0e-8,
            "density_dlv_inverse_atol": 2.0e-14,
            "density_calendar_validity": 1.0e-7,
            "quote_containment_validity": 1.0e-6,
            "density_mass_moment_validity": 1.0e-4,
        }
        for field, required in expected.items():
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
                or float(value) != required
            ):
                raise ValueError(
                    f"{field} is frozen at {required!r} by the Phase 10 v1 contract"
                )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@version("1.0", dependencies=[DailyLegacyComparisonTolerances])
@dataclass(frozen=True, slots=True)
class DailyComparisonRequest:
    """A read-only selection of already completed fitted artifacts."""

    ticker: str
    start_date: date
    end_date: date
    source: str
    source_generation: str
    declared_oracle_gaps: tuple[date, ...] = ()
    enforce_phase10_pilot: bool = False
    tolerances: DailyLegacyComparisonTolerances = DailyLegacyComparisonTolerances()

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("Comparison end_date cannot precede start_date")
        if self.source != "optionmetrics":
            raise ValueError(f"Unsupported comparison source: {self.source!r}")
        if not self.source_generation.strip():
            raise ValueError("source_generation cannot be empty")
        normalized = self.ticker.strip().upper()
        if not normalized:
            raise ValueError("ticker cannot be empty")
        object.__setattr__(self, "ticker", normalized)
        if len(set(self.declared_oracle_gaps)) != len(self.declared_oracle_gaps):
            raise ValueError("declared_oracle_gaps contains duplicate dates")
        if not isinstance(self.enforce_phase10_pilot, bool):
            raise TypeError("enforce_phase10_pilot must be a boolean")


@version("1.0")
def _reference_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@version("1.0")
def repository_source_revision() -> str:
    """Return a clean exact revision, or a validated explicit packaged revision."""

    explicit = os.environ.get("SANOS_SOURCE_REVISION")
    if explicit is not None:
        if _HEX_40.fullmatch(explicit) is None:
            raise ReferenceComparisonError(
                "SANOS_SOURCE_REVISION must be an exact lowercase 40-hex commit"
            )
        return explicit
    root = Path(__file__).resolve().parents[3]
    git_dir: Path | None = None
    for candidate in (root / ".git-real", root / ".git"):
        if candidate.is_dir() and (candidate / "HEAD").is_file():
            git_dir = candidate
            break
        if candidate.is_file():
            try:
                marker = candidate.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if marker.startswith("gitdir: "):
                linked = (candidate.parent / marker.removeprefix("gitdir: ")).resolve()
                if linked.is_dir() and (linked / "HEAD").is_file():
                    git_dir = linked
                    break
    if git_dir is None:
        raise ReferenceComparisonError(
            "Comparison requires SANOS_SOURCE_REVISION or an identifiable Git checkout"
        )
    try:
        status = subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                f"--work-tree={root}",
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = subprocess.run(
            ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReferenceComparisonError(f"Cannot inspect SANOS Git provenance: {exc}") from exc
    if status.stdout.strip():
        raise ReferenceComparisonError(
            "Reference comparison requires a clean Git checkout; commit reviewed code first"
        )
    if _HEX_40.fullmatch(revision) is None:
        raise ReferenceComparisonError(f"Git returned an invalid revision: {revision!r}")
    return revision


@version("1.0", dependencies=[_reference_sha256_file])
def load_reference_catalog(path: Path) -> dict[str, Any]:
    """Load and structurally validate a checked-in legacy catalog."""

    if path.is_symlink() or not path.is_file():
        raise ReferenceComparisonError(f"Legacy catalog is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceComparisonError(f"Cannot decode legacy catalog {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != "sanos-legacy-market-catalog-v1":
        raise ReferenceComparisonError("Legacy catalog has an unsupported schema")
    required = {"collection_sha256", "source_revision", "relative_path", "files"}
    if not required <= set(value) or not isinstance(value["files"], list):
        raise ReferenceComparisonError("Legacy catalog is missing identity or file records")
    collection = value["collection_sha256"]
    if not isinstance(collection, str) or _HEX_64.fullmatch(collection) is None:
        raise ReferenceComparisonError("Legacy catalog collection_sha256 is invalid")
    files = value["files"]
    names: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise ReferenceComparisonError("Legacy catalog contains a malformed file record")
        file_name = record.get("file")
        digest = record.get("sha256")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not file_name.endswith(".h5")
            or file_name in names
            or not isinstance(digest, str)
            or _HEX_64.fullmatch(digest) is None
        ):
            raise ReferenceComparisonError("Legacy catalog contains an invalid file identity")
        names.add(file_name)
    return cast(dict[str, Any], value)


@version("1.0")
def _array_keys(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}[{index}]" for index in range(count))


@version("1.0")
def _ulp_distances(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> np.ndarray:
    if left.dtype != right.dtype or left.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise TypeError("ULP comparison requires equal-width float arrays")
    if left.dtype == np.dtype("float32"):
        left_bits = left.view(np.uint32)
        right_bits = right.view(np.uint32)
        sign = np.uint32(1 << 31)
        left_order = np.where(left_bits & sign, ~left_bits, left_bits | sign).astype(np.int64)
        right_order = np.where(right_bits & sign, ~right_bits, right_bits | sign).astype(
            np.int64
        )
        return np.abs(left_order - right_order)
    left_bits64 = left.view(np.uint64)
    right_bits64 = right.view(np.uint64)
    sign64 = np.uint64(1 << 63)
    left_order64 = np.where(left_bits64 & sign64, ~left_bits64, left_bits64 | sign64)
    right_order64 = np.where(right_bits64 & sign64, ~right_bits64, right_bits64 | sign64)
    return np.fromiter(
        (abs(int(a) - int(b)) for a, b in zip(left_order64, right_order64, strict=True)),
        dtype=np.float64,
        count=left.size,
    )


@version("1.0", dependencies=[_ulp_distances])
def _compare_numeric(
    field: str,
    fresh: np.ndarray[Any, Any],
    legacy: np.ndarray[Any, Any],
    *,
    fresh_dtype: np.dtype[Any],
    legacy_dtype: np.dtype[Any],
    comparison_dtype: np.dtype[Any],
    atol: float,
    keys: Sequence[str],
    family: str,
    mismatches: list[dict[str, Any]],
    numerical: list[dict[str, Any]],
) -> None:
    """Compare one numeric field while retaining nonzero and ULP evidence."""

    structural = False
    if fresh.dtype != fresh_dtype or legacy.dtype != legacy_dtype:
        mismatches.append(
            {
                "field": field,
                "family": family,
                "kind": "dtype",
                "fresh_dtype": str(fresh.dtype),
                "legacy_dtype": str(legacy.dtype),
                "expected_fresh_dtype": str(fresh_dtype),
                "expected_legacy_dtype": str(legacy_dtype),
            }
        )
        structural = True
    if fresh.shape != legacy.shape:
        mismatches.append(
            {
                "field": field,
                "family": family,
                "kind": "shape",
                "fresh_shape": list(fresh.shape),
                "legacy_shape": list(legacy.shape),
            }
        )
        return
    if fresh.ndim != 1:
        mismatches.append(
            {"field": field, "family": family, "kind": "rank", "shape": list(fresh.shape)}
        )
        return
    fresh_null = np.isnan(fresh) if fresh.dtype.kind == "f" else np.zeros(fresh.shape, bool)
    legacy_null = np.isnan(legacy) if legacy.dtype.kind == "f" else np.zeros(legacy.shape, bool)
    fresh_inf = np.isinf(fresh) if fresh.dtype.kind == "f" else np.zeros(fresh.shape, bool)
    legacy_inf = np.isinf(legacy) if legacy.dtype.kind == "f" else np.zeros(legacy.shape, bool)
    if np.any(fresh_inf) or np.any(legacy_inf):
        mismatches.append(
            {"field": field, "family": family, "kind": "infinity", "count": int(
                np.sum(fresh_inf | legacy_inf)
            )}
        )
        structural = True
    null_difference = fresh_null != legacy_null
    if np.any(null_difference):
        first = int(np.flatnonzero(null_difference)[0])
        mismatches.append(
            {
                "field": field,
                "family": family,
                "kind": "null_pattern",
                "count": int(np.sum(null_difference)),
                "first_mismatch_key": keys[first] if first < len(keys) else str(first),
            }
        )
        structural = True
    finite = ~(fresh_null | legacy_null | fresh_inf | legacy_inf)
    fresh_values = fresh[finite].astype(comparison_dtype, copy=False)
    legacy_values = legacy[finite].astype(comparison_dtype, copy=False)
    differences = np.abs(fresh_values - legacy_values)
    bitwise_exact = (
        atol == 0.0
        and fresh.dtype == legacy.dtype == np.dtype("float32")
    )
    if bitwise_exact:
        nonzero = fresh[finite].view(np.uint32) != legacy[finite].view(np.uint32)
        above = nonzero.copy()
    else:
        nonzero = differences != 0.0
        above = differences > atol
    positions = np.flatnonzero(finite)
    first_above = int(np.flatnonzero(above)[0]) if np.any(above) else None
    denominators = np.abs(legacy_values)
    relative = np.divide(
        differences,
        denominators,
        out=np.full_like(differences, np.nan, dtype=np.float64),
        where=denominators != 0.0,
    )
    ulp_max: int | None = None
    if fresh.dtype == legacy.dtype and fresh.dtype.kind == "f" and np.any(finite):
        ulp = _ulp_distances(fresh[finite], legacy[finite])
        ulp_max = int(np.max(ulp))
    record = {
        "field": field,
        "family": family,
        "count": int(fresh.size),
        "finite_comparison_count": int(np.sum(finite)),
        "atol": atol,
        "rtol": 0.0,
        "maximum_absolute_error": float(np.max(differences)) if differences.size else 0.0,
        "maximum_relative_error": (
            float(np.nanmax(relative)) if relative.size and np.any(~np.isnan(relative)) else None
        ),
        "maximum_ulp_distance": ulp_max,
        "nonzero_difference_count": int(np.sum(nonzero)),
        "above_tolerance_count": int(np.sum(above)),
        "first_mismatch_key": (
            keys[int(positions[first_above])]
            if first_above is not None and int(positions[first_above]) < len(keys)
            else None
        ),
    }
    numerical.append(record)
    if np.any(above) or (atol == 0.0 and np.any(nonzero)):
        mismatches.append(
            {
                "field": field,
                "family": family,
                "kind": "numerical",
                "above_tolerance_count": record["above_tolerance_count"],
                "first_mismatch_key": record["first_mismatch_key"],
            }
        )
    elif structural:
        return


@version("1.0")
def _compare_exact(
    field: str,
    fresh: Sequence[Any] | np.ndarray[Any, Any],
    legacy: Sequence[Any] | np.ndarray[Any, Any],
    *,
    keys: Sequence[str],
    family: str,
    mismatches: list[dict[str, Any]],
) -> None:
    left = list(fresh)
    right = list(legacy)
    if len(left) != len(right):
        mismatches.append(
            {
                "field": field,
                "family": family,
                "kind": "shape",
                "fresh_count": len(left),
                "legacy_count": len(right),
            }
        )
        return
    unequal = [
        index
        for index, pair in enumerate(zip(left, right, strict=True))
        if pair[0] != pair[1]
    ]
    if unequal:
        first = unequal[0]
        mismatches.append(
            {
                "field": field,
                "family": family,
                "kind": "exact",
                "count": len(unequal),
                "first_mismatch_key": keys[first] if first < len(keys) else str(first),
            }
        )


@version("1.0")
def _legacy_list(value: object, field: str) -> list[np.ndarray[Any, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, np.ndarray) for item in value):
        raise ReferenceComparisonError(f"Legacy {field} is not a list of NumPy arrays")
    return cast(list[np.ndarray[Any, Any]], value)


@version("1.0")
def _constant_list(group: pl.DataFrame, column: str) -> np.ndarray[Any, Any]:
    values = group.get_column(column).to_list()
    if not values or values[0] is None or any(value != values[0] for value in values[1:]):
        raise ReferenceComparisonError(f"Fresh {column} is missing or inconsistent within expiry")
    return np.asarray(values[0], dtype=np.float64)


@version("1.0", dependencies=[selected_reference_view])
def _fresh_groups(frame: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame], bool, bool]:
    original = frame.filter(pl.col("options_filter_eligible"))
    view = selected_reference_view(frame)
    if original.is_empty():
        return view, [], True, True
    duplicate = original.select("expiry_id", "normalized_strike").is_duplicated().any()
    canonical = original.sort(
        "time_to_expiry_years", "expiry_id", "normalized_strike", maintain_order=True
    )
    already_canonical = original.select(
        "time_to_expiry_years", "expiry_id", "normalized_strike"
    ).equals(canonical.select("time_to_expiry_years", "expiry_id", "normalized_strike"))
    groups = view.partition_by("expiry_id", maintain_order=True)
    return view, groups, already_canonical, not duplicate


@version(
    "1.0",
    dependencies=[
        DailyLegacyComparisonTolerances,
        ArbFreePricer,
        SanosPricer,
        _array_keys,
        _compare_exact,
        _compare_numeric,
        _constant_list,
        _fresh_groups,
        _legacy_list,
        compatibility_dlv_pricer,
        selected_reference_view,
        smooth_sanos_pricer,
    ],
)
def compare_populated_day(
    frame: pl.DataFrame,
    legacy_payload: Mapping[str, object],
    *,
    tolerances: DailyLegacyComparisonTolerances = DailyLegacyComparisonTolerances(),
) -> dict[str, Any]:
    """Compare a validated populated fitted frame with one populated legacy payload."""

    mismatches: list[dict[str, Any]] = []
    numerical: list[dict[str, Any]] = []
    view, groups, fresh_canonical, fresh_unique = _fresh_groups(frame)
    if not fresh_canonical:
        mismatches.append(
            {"field": "fresh.order", "family": "structure", "kind": "noncanonical"}
        )
    if not fresh_unique:
        mismatches.append(
            {"field": "fresh.keys", "family": "structure", "kind": "duplicate"}
        )
    market = legacy_payload.get("market")
    state = legacy_payload.get("af_pricer")
    fit_result = legacy_payload.get("fit_result")
    if not all(isinstance(value, Mapping) for value in (market, state, fit_result)):
        raise ReferenceComparisonError("Legacy populated payload mappings are malformed")
    market = cast(Mapping[str, object], market)
    state = cast(Mapping[str, object], state)
    fit_result = cast(Mapping[str, object], fit_result)
    trading_date = frame.get_column("trading_date").item(0)
    ticker = frame.get_column("ticker").item(0)
    _compare_exact(
        "top_level.date",
        [trading_date],
        [legacy_payload.get("d")],
        keys=[str(trading_date)],
        family="identity",
        mismatches=mismatches,
    )
    _compare_exact(
        "top_level.ticker",
        [ticker],
        [legacy_payload.get("ticker")],
        keys=[str(trading_date)],
        family="identity",
        mismatches=mismatches,
    )
    scalar_keys = [str(trading_date)]
    _compare_numeric(
        "top_level.spot_reference",
        np.asarray([view.get_column("spot_reference").item(0)], dtype=np.float32),
        np.asarray([legacy_payload.get("spot_ref")], dtype=np.float64),
        fresh_dtype=np.dtype("float32"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.float32_atol,
        keys=scalar_keys,
        family="market",
        mismatches=mismatches,
        numerical=numerical,
    )
    _compare_numeric(
        "top_level.spot_return",
        np.asarray([view.get_column("spot_return").item(0)], dtype=np.float32),
        np.asarray([legacy_payload.get("spot_ret")], dtype=np.float64),
        fresh_dtype=np.dtype("float32"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.float32_atol,
        keys=scalar_keys,
        family="market",
        mismatches=mismatches,
        numerical=numerical,
    )

    legacy_expiries = np.asarray(market["expiries"])
    legacy_expiry_ids = np.asarray(market["expiry_ids"])
    legacy_is_am = np.asarray(market["is_am"])
    legacy_expiry_dates = cast(list[str], market["expiry_dates"])
    fresh_expiries = np.asarray(
        [group.get_column("time_to_expiry_years").item(0) for group in groups],
        dtype=np.float32,
    )
    fresh_expiry_ids = np.asarray(
        [group.get_column("expiry_id").item(0) for group in groups], dtype=np.int64
    )
    expiry_keys = tuple(str(value) for value in fresh_expiry_ids)
    for field, values, expected_dtype in (
        ("market.expiry_ids", legacy_expiry_ids, np.dtype("int64")),
        ("market.is_am", legacy_is_am, np.dtype("bool")),
    ):
        if values.dtype != expected_dtype:
            mismatches.append(
                {
                    "field": field,
                    "family": "market_geometry",
                    "kind": "dtype",
                    "legacy_dtype": str(values.dtype),
                    "expected_legacy_dtype": str(expected_dtype),
                }
            )
    if not isinstance(legacy_expiry_dates, list) or not all(
        isinstance(value, str) for value in legacy_expiry_dates
    ):
        mismatches.append(
            {
                "field": "market.expiry_dates",
                "family": "market_geometry",
                "kind": "dtype",
            }
        )
    legacy_strikes_for_keys = _legacy_list(market["strikes"], "market.strikes")
    legacy_pairs = (
        list(zip(legacy_expiries.tolist(), legacy_expiry_ids.tolist(), strict=True))
        if len(legacy_expiries) == len(legacy_expiry_ids)
        else []
    )
    legacy_canonical = bool(
        legacy_pairs
        and legacy_pairs == sorted(legacy_pairs)
        and all(
            values.ndim == 1
            and np.all(np.isfinite(values))
            and (len(values) < 2 or np.all(np.diff(values) > 0.0))
            for values in legacy_strikes_for_keys
        )
    )
    legacy_unique = bool(
        len(set(legacy_expiry_ids.tolist())) == len(legacy_expiry_ids)
        and all(len(np.unique(values)) == len(values) for values in legacy_strikes_for_keys)
    )
    if not legacy_canonical:
        mismatches.append(
            {"field": "legacy.order", "family": "structure", "kind": "noncanonical"}
        )
    if not legacy_unique:
        mismatches.append(
            {"field": "legacy.keys", "family": "structure", "kind": "duplicate"}
        )
    _compare_numeric(
        "market.expiries",
        fresh_expiries,
        legacy_expiries,
        fresh_dtype=np.dtype("float32"),
        legacy_dtype=np.dtype("float32"),
        comparison_dtype=np.dtype("float32"),
        atol=tolerances.float32_atol,
        keys=expiry_keys,
        family="market_geometry",
        mismatches=mismatches,
        numerical=numerical,
    )
    _compare_exact(
        "market.expiry_ids",
        fresh_expiry_ids.tolist(),
        legacy_expiry_ids.tolist(),
        keys=expiry_keys,
        family="market_geometry",
        mismatches=mismatches,
    )
    _compare_exact(
        "market.expiry_dates",
        [group.get_column("expiry_date").item(0).isoformat() for group in groups],
        legacy_expiry_dates,
        keys=expiry_keys,
        family="market_geometry",
        mismatches=mismatches,
    )
    _compare_exact(
        "market.is_am",
        [bool(group.get_column("expiry_is_am").item(0)) for group in groups],
        legacy_is_am.tolist(),
        keys=expiry_keys,
        family="market_geometry",
        mismatches=mismatches,
    )

    field_map = {
        "strikes": "normalized_strike",
        "bids": "pure_bid_price",
        "asks": "pure_ask_price",
        "is_american": "is_american",
        "pre_af_masks": "mask_arb_free",
        "model_masks": "smooth_model_mask",
        "call_bids": "pure_market_call_bid",
        "call_asks": "pure_market_call_ask",
        "put_bids": "pure_market_put_bid",
        "put_asks": "pure_market_put_ask",
        "call_log_volumes": "market_call_log_volume",
        "call_log_open_interests": "market_call_log_open_interest",
        "put_log_volumes": "market_put_log_volume",
        "put_log_open_interests": "market_put_log_open_interest",
        "call_active": "call_active",
        "put_active": "put_active",
        "af_call_mids": "smooth_call_mid",
        "af_put_mids": "smooth_put_mid",
        "af_mid_available": "smooth_mid_available",
    }
    boolean_fields = {
        "is_american",
        "pre_af_masks",
        "model_masks",
        "call_active",
        "put_active",
        "af_mid_available",
    }
    for legacy_name, fresh_name in field_map.items():
        legacy_values = _legacy_list(market[legacy_name], f"market.{legacy_name}")
        if len(legacy_values) != len(groups):
            mismatches.append(
                {
                    "field": f"market.{legacy_name}",
                    "family": "population",
                    "kind": "expiry_count",
                    "fresh_count": len(groups),
                    "legacy_count": len(legacy_values),
                }
            )
            continue
        for index, (group, legacy_array) in enumerate(
            zip(groups, legacy_values, strict=True)
        ):
            prefix = expiry_keys[index] if index < len(expiry_keys) else str(index)
            keys = tuple(
                f"{prefix}/{float(value):.9g}"
                for value in group.get_column("normalized_strike").to_list()
            )
            fresh_array = group.get_column(fresh_name).to_numpy()
            field = f"market.{legacy_name}[{index}]"
            if legacy_name in boolean_fields:
                if fresh_array.dtype != np.dtype("bool") or legacy_array.dtype != np.dtype("bool"):
                    mismatches.append(
                        {
                            "field": field,
                            "family": "mask",
                            "kind": "dtype",
                            "fresh_dtype": str(fresh_array.dtype),
                            "legacy_dtype": str(legacy_array.dtype),
                        }
                    )
                _compare_exact(
                    field,
                    fresh_array.tolist(),
                    legacy_array.tolist(),
                    keys=keys,
                    family="mask",
                    mismatches=mismatches,
                )
            else:
                _compare_numeric(
                    field,
                    fresh_array,
                    legacy_array,
                    fresh_dtype=np.dtype("float32")
                    if fresh_name not in {"smooth_call_mid", "smooth_put_mid"}
                    else np.dtype("float64"),
                    legacy_dtype=np.dtype("float32")
                    if legacy_name
                    not in {"af_call_mids", "af_put_mids"}
                    else np.dtype("float64"),
                    comparison_dtype=np.dtype("float32")
                    if legacy_name not in {"af_call_mids", "af_put_mids"}
                    else np.dtype("float64"),
                    atol=(
                        tolerances.float32_atol
                        if legacy_name not in {"af_call_mids", "af_put_mids"}
                        else tolerances.float64_atol
                    ),
                    keys=keys,
                    family="market",
                    mismatches=mismatches,
                    numerical=numerical,
                )

    serialized_input_index = np.asarray(state.get("input_index"))
    if serialized_input_index.dtype != np.dtype("int32"):
        mismatches.append(
            {
                "field": "af_pricer.input_index",
                "family": "smooth_state",
                "kind": "dtype",
                "legacy_dtype": str(serialized_input_index.dtype),
                "expected_legacy_dtype": "int32",
            }
        )
    serialized_mappings = state.get("xix_to_mix")
    if isinstance(serialized_mappings, list):
        for index, serialized_mapping in enumerate(serialized_mappings):
            if isinstance(serialized_mapping, str):
                continue
            mapping_array = np.asarray(serialized_mapping)
            if mapping_array.dtype != np.dtype("int32"):
                mismatches.append(
                    {
                        "field": f"af_pricer.xix_to_mix[{index}]",
                        "family": "smooth_state",
                        "kind": "dtype",
                        "legacy_dtype": str(mapping_array.dtype),
                        "expected_legacy_dtype": "int32",
                    }
                )

    legacy_pricer = ArbFreePricer.from_dict(dict(state))
    if legacy_pricer.mstrikes is None or legacy_pricer.mdensities is None:
        raise ReferenceComparisonError("Legacy smooth pricer has no surface state")
    legacy_surface = SanosPricer(
        smoothness=float(legacy_pricer.smoothness),
        expiries=np.asarray(legacy_pricer.expiries, dtype=np.float64),
        variances=np.asarray(legacy_pricer.variances, dtype=np.float64),
        xstrikes=[np.asarray(value, dtype=np.float64) for value in legacy_pricer.mstrikes],
        xdensities=[np.asarray(value, dtype=np.float64) for value in legacy_pricer.mdensities],
        min_half_spread=float(legacy_pricer.min_half_spread),
    )
    fresh_pricer = smooth_sanos_pricer(frame)
    retained_groups = [
        group
        for group in groups
        if group.get_column("smooth_input_expiry_index").null_count() == 0
    ]
    fresh_input_index = np.asarray(
        [group.get_column("smooth_input_expiry_index").item(0) for group in retained_groups],
        dtype=np.int32,
    )
    legacy_input_index = np.asarray(legacy_pricer.input_index)
    _compare_exact(
        "af_pricer.input_index",
        fresh_input_index.tolist(),
        legacy_input_index.tolist(),
        keys=_array_keys("retained_expiry", len(fresh_input_index)),
        family="smooth_state",
        mismatches=mismatches,
    )
    state_keys = _array_keys("retained_expiry", len(fresh_pricer.expiries))
    for field, fresh_values, legacy_values in (
        ("expiries", np.asarray(fresh_pricer.expiries), np.asarray(legacy_pricer.expiries)),
        ("variances", np.asarray(fresh_pricer.variances), np.asarray(legacy_pricer.variances)),
    ):
        _compare_numeric(
            f"af_pricer.{field}",
            fresh_values.astype(np.float64, copy=False),
            legacy_values,
            fresh_dtype=np.dtype("float64"),
            legacy_dtype=np.dtype("float64"),
            comparison_dtype=np.dtype("float64"),
            atol=tolerances.float64_atol,
            keys=state_keys,
            family="smooth_state",
            mismatches=mismatches,
            numerical=numerical,
        )
    _compare_numeric(
        "af_pricer.smoothness",
        np.asarray([fresh_pricer.smoothness], dtype=np.float64),
        np.asarray([legacy_pricer.smoothness], dtype=np.float64),
        fresh_dtype=np.dtype("float64"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.float64_atol,
        keys=["smoothness"],
        family="smooth_state",
        mismatches=mismatches,
        numerical=numerical,
    )
    _compare_numeric(
        "af_pricer.min_half_spread",
        np.asarray([fresh_pricer.min_half_spread], dtype=np.float64),
        np.asarray([legacy_pricer.min_half_spread], dtype=np.float64),
        fresh_dtype=np.dtype("float64"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.float64_atol,
        keys=["min_half_spread"],
        family="smooth_state",
        mismatches=mismatches,
        numerical=numerical,
    )
    if not bool(legacy_pricer.expiry_linear):
        mismatches.append(
            {
                "field": "af_pricer.expiry_linear",
                "family": "smooth_state",
                "kind": "exact",
                "count": 1,
            }
        )

    state_fields = (
        ("masks", "smooth_fit_mask", "bool"),
        ("strikes", "normalized_strike", "float32"),
        ("bids", "pure_bid_price", "float32"),
        ("asks", "pure_ask_price", "float32"),
        ("fits1", "smooth_expiry_fit_price", "float64"),
        ("fitsX", "smooth_call_mid", "float64"),
    )
    for legacy_name, fresh_name, dtype_name in state_fields:
        legacy_values = getattr(legacy_pricer, legacy_name)
        if legacy_values is None or len(legacy_values) != len(retained_groups):
            mismatches.append(
                {
                    "field": f"af_pricer.{legacy_name}",
                    "family": "smooth_state",
                    "kind": "expiry_count",
                }
            )
            continue
        for index, (group, legacy_array) in enumerate(
            zip(retained_groups, legacy_values, strict=True)
        ):
            fresh_array = group.get_column(fresh_name).to_numpy()
            keys = tuple(
                f"retained_expiry[{index}]/{float(value):.9g}"
                for value in group.get_column("normalized_strike").to_list()
            )
            if dtype_name == "bool":
                legacy_array = np.asarray(legacy_array)
                if (
                    fresh_array.dtype != np.dtype("bool")
                    or legacy_array.dtype != np.dtype("bool")
                ):
                    mismatches.append(
                        {
                            "field": f"af_pricer.{legacy_name}[{index}]",
                            "family": "smooth_state",
                            "kind": "dtype",
                            "fresh_dtype": str(fresh_array.dtype),
                            "legacy_dtype": str(legacy_array.dtype),
                        }
                    )
                _compare_exact(
                    f"af_pricer.{legacy_name}[{index}]",
                    fresh_array.tolist(),
                    legacy_array.tolist(),
                    keys=keys,
                    family="smooth_state",
                    mismatches=mismatches,
                )
            else:
                dtype = np.dtype(dtype_name)
                _compare_numeric(
                    f"af_pricer.{legacy_name}[{index}]",
                    fresh_array,
                    np.asarray(legacy_array),
                    fresh_dtype=dtype,
                    legacy_dtype=dtype,
                    comparison_dtype=dtype,
                    atol=(
                        tolerances.float32_atol
                        if dtype == np.dtype("float32")
                        else tolerances.float64_atol
                    ),
                    keys=keys,
                    family="smooth_state",
                    mismatches=mismatches,
                    numerical=numerical,
                )

    support_fields = (
        ("xstrikes", "smooth_expiry_model_strikes", "float32", "float64"),
        ("xdensities", "smooth_expiry_model_density", "float64", "float64"),
        ("mstrikes", "smooth_surface_model_strikes", "float32", "float64"),
        ("mdensities", "smooth_surface_model_density", "float64", "float64"),
    )
    for legacy_name, fresh_name, legacy_dtype_name, fresh_dtype_name in support_fields:
        legacy_values = getattr(legacy_pricer, legacy_name)
        if legacy_values is None or len(legacy_values) != len(retained_groups):
            mismatches.append(
                {
                    "field": f"af_pricer.{legacy_name}",
                    "family": "smooth_state",
                    "kind": "expiry_count",
                }
            )
            continue
        for index, (group, legacy_array) in enumerate(
            zip(retained_groups, legacy_values, strict=True)
        ):
            fresh_array = _constant_list(group, fresh_name)
            legacy_array = np.asarray(legacy_array)
            _compare_numeric(
                f"af_pricer.{legacy_name}[{index}]",
                fresh_array,
                legacy_array,
                fresh_dtype=np.dtype(fresh_dtype_name),
                legacy_dtype=np.dtype(legacy_dtype_name),
                comparison_dtype=np.dtype("float64"),
                atol=tolerances.float64_atol,
                keys=_array_keys(f"retained_expiry[{index}].{legacy_name}", len(fresh_array)),
                family="smooth_state",
                mismatches=mismatches,
                numerical=numerical,
            )
    legacy_mappings = legacy_pricer.xix_to_mix
    if legacy_mappings is None or len(legacy_mappings) != len(retained_groups):
        mismatches.append(
            {"field": "af_pricer.xix_to_mix", "family": "smooth_state", "kind": "shape"}
        )
    else:
        for index, (group, legacy_mapping) in enumerate(
            zip(retained_groups, legacy_mappings, strict=True)
        ):
            fresh_values = group.get_column("smooth_x_to_surface_indices").to_list()
            if not fresh_values or any(value != fresh_values[0] for value in fresh_values[1:]):
                raise ReferenceComparisonError("Fresh smooth mapping is inconsistent within expiry")
            fresh_mapping = fresh_values[0]
            _compare_exact(
                f"af_pricer.xix_to_mix[{index}]",
                [] if fresh_mapping is None else fresh_mapping,
                [] if legacy_mapping is None else np.asarray(legacy_mapping).tolist(),
                keys=_array_keys(
                    f"retained_expiry[{index}].mapping",
                    0 if fresh_mapping is None else len(fresh_mapping),
                ),
                family="smooth_state",
                mismatches=mismatches,
            )

    fit_fields = {
        "fits": "smooth_call_mid",
        "fit_errs": "smooth_fit_error_half_spreads",
    }
    for legacy_name, fresh_name in fit_fields.items():
        legacy_values = _legacy_list(fit_result[legacy_name], f"fit_result.{legacy_name}")
        if len(legacy_values) != len(groups):
            mismatches.append(
                {
                    "field": f"fit_result.{legacy_name}",
                    "family": "fit_diagnostics",
                    "kind": "expiry_count",
                }
            )
            continue
        for index, (group, legacy_array) in enumerate(
            zip(groups, legacy_values, strict=True)
        ):
            fresh_array = group.get_column(fresh_name).to_numpy()
            _compare_numeric(
                f"fit_result.{legacy_name}[{index}]",
                fresh_array,
                legacy_array,
                fresh_dtype=np.dtype("float64"),
                legacy_dtype=np.dtype("float64"),
                comparison_dtype=np.dtype("float64"),
                atol=tolerances.float64_atol,
                keys=_array_keys(f"expiry[{index}].{legacy_name}", len(fresh_array)),
                family="fit_diagnostics",
                mismatches=mismatches,
                numerical=numerical,
            )
    summary_fields = {
        "pcnt_fit": "smooth_pcnt_fit",
        "median_err": "smooth_median_error",
        "mean_err": "smooth_mean_error",
        "median_total_err": "smooth_median_total_error",
        "mean_total_err": "smooth_mean_total_error",
    }
    for legacy_name, fresh_name in summary_fields.items():
        _compare_numeric(
            f"fit_result.{legacy_name}",
            np.asarray([view.get_column(fresh_name).item(0)], dtype=np.float64),
            np.asarray([fit_result[legacy_name]], dtype=np.float64),
            fresh_dtype=np.dtype("float64"),
            legacy_dtype=np.dtype("float64"),
            comparison_dtype=np.dtype("float64"),
            atol=tolerances.float64_atol,
            keys=[legacy_name],
            family="fit_diagnostics",
            mismatches=mismatches,
            numerical=numerical,
        )

    legacy_strike_collections = (
        legacy_pricer.strikes,
        legacy_pricer.xstrikes,
        legacy_pricer.mstrikes,
    )
    if any(values is None for values in legacy_strike_collections):
        raise ReferenceComparisonError("Legacy smooth pricer has incomplete strike geometry")
    strike_grid = np.unique(
        np.concatenate(
            [
                np.asarray([0.713, 0.947, 1.0, 1.037, 1.289], dtype=np.float64),
                *(
                    group.get_column("normalized_strike").to_numpy().astype(np.float64)
                    for group in groups
                ),
                *(
                    np.asarray(value, dtype=np.float64)
                    for collection in legacy_strike_collections
                    if collection is not None
                    for value in collection
                ),
                *(
                    _constant_list(group, "smooth_expiry_model_strikes")
                    for group in retained_groups
                ),
                *(np.asarray(value, dtype=np.float64) for value in fresh_pricer.xstrikes),
            ]
        )
    )
    minute = 1.0 / (252.0 * 24.0 * 60.0)
    nodes = np.asarray(legacy_pricer.expiries, dtype=np.float64)
    midpoints = 0.5 * (nodes[:-1] + nodes[1:])
    time_grid = np.unique(
        np.concatenate(
            (
                np.asarray(
                    [0.0, 0.5 * minute, minute, max(0.0, nodes[0] - 2.0 * minute)],
                    dtype=np.float64,
                ),
                nodes,
                midpoints,
                np.asarray([nodes[-1] + 2.0 * minute], dtype=np.float64),
            )
        )
    )
    call_fresh = np.concatenate(
        [np.asarray(fresh_pricer(float(expiry), strike_grid)) for expiry in time_grid]
    )
    call_legacy = np.concatenate(
        [np.asarray(legacy_surface(float(expiry), strike_grid)) for expiry in time_grid]
    )
    call_keys = tuple(
        f"time[{time_index}]/strike[{strike_index}]"
        for time_index in range(len(time_grid))
        for strike_index in range(len(strike_grid))
    )
    _compare_numeric(
        "reconstruction.calls",
        call_fresh.astype(np.float64, copy=False),
        call_legacy.astype(np.float64, copy=False),
        fresh_dtype=np.dtype("float64"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.reconstructed_call_atol,
        keys=call_keys,
        family="reconstruction",
        mismatches=mismatches,
        numerical=numerical,
    )
    put_fresh = call_fresh - 1.0 + np.tile(strike_grid, len(time_grid))
    put_legacy = call_legacy - 1.0 + np.tile(strike_grid, len(time_grid))
    _compare_numeric(
        "reconstruction.puts",
        put_fresh,
        put_legacy,
        fresh_dtype=np.dtype("float64"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.reconstructed_call_atol,
        keys=call_keys,
        family="reconstruction",
        mismatches=mismatches,
        numerical=numerical,
    )
    fresh_atm_vols: list[float] = []
    legacy_atm_vols: list[float] = []
    for expiry in time_grid:
        _, fresh_vol = fresh_pricer(float(expiry), np.asarray([1.0]), and_vols=True)
        _, legacy_vol = legacy_surface(float(expiry), np.asarray([1.0]), and_vols=True)
        fresh_atm_vols.append(float(fresh_vol[0]))
        legacy_atm_vols.append(float(legacy_vol[0]))
    _compare_numeric(
        "reconstruction.atm_implied_volatility",
        np.asarray(fresh_atm_vols, dtype=np.float64),
        np.asarray(legacy_atm_vols, dtype=np.float64),
        fresh_dtype=np.dtype("float64"),
        legacy_dtype=np.dtype("float64"),
        comparison_dtype=np.dtype("float64"),
        atol=tolerances.reconstructed_call_atol,
        keys=_array_keys("time", len(time_grid)),
        family="reconstruction",
        mismatches=mismatches,
        numerical=numerical,
    )
    dlv_pricer = compatibility_dlv_pricer(frame)
    initial_dlvs = dlv_pricer.dlvsqs(on_all_strikes=True)
    initial_state = frame.filter(
        pl.col("initial_no_arb_input_expiry_index").is_not_null()
    ).sort("initial_no_arb_input_expiry_index")
    for index, recomputed in enumerate(initial_dlvs):
        group = initial_state.filter(
            pl.col("initial_no_arb_input_expiry_index") == index
        )
        stored = _constant_list(group, "initial_surface_dlv_squares")
        _compare_numeric(
            f"fresh_only.unsmoothed_dlv_squares[{index}]",
            stored,
            np.asarray(recomputed, dtype=np.float64),
            fresh_dtype=np.dtype("float64"),
            legacy_dtype=np.dtype("float64"),
            comparison_dtype=np.dtype("float64"),
            atol=tolerances.float64_atol,
            keys=_array_keys(f"initial_dlv[{index}]", len(stored)),
            family="fresh_invariant",
            mismatches=mismatches,
            numerical=numerical,
        )
    decoded_initial = dlv_pricer.densities_from_dlvsqs(initial_dlvs)
    initial_inverse_error = 0.0
    for index, (decoded, density) in enumerate(
        zip(decoded_initial, dlv_pricer.xdensities, strict=True)
    ):
        initial_inverse_error = max(
            initial_inverse_error,
            float(np.max(np.abs(np.asarray(decoded) - np.asarray(density)))),
        )
        _compare_numeric(
            f"fresh_only.unsmoothed_density_dlv_inverse[{index}]",
            np.asarray(decoded, dtype=np.float64),
            np.asarray(density, dtype=np.float64),
            fresh_dtype=np.dtype("float64"),
            legacy_dtype=np.dtype("float64"),
            comparison_dtype=np.dtype("float64"),
            atol=tolerances.density_dlv_inverse_atol,
            keys=_array_keys(f"initial_density[{index}]", len(decoded)),
            family="fresh_invariant",
            mismatches=mismatches,
            numerical=numerical,
        )
    if initial_inverse_error > tolerances.density_dlv_inverse_atol:
        mismatches.append(
            {
                "field": "fresh_only.unsmoothed_density_dlv_inverse",
                "family": "fresh_invariant",
                "kind": "numerical",
                "above_tolerance_count": 1,
            }
        )
    item_oracle_gaps = [
        {
            "item": "unsmoothed_no_arbitrage_state",
            "reason": "legacy daily HDF5 does not store complete pre-filter state",
            "fresh_invariants_passed": initial_inverse_error
            <= tolerances.density_dlv_inverse_atol,
            "density_dlv_inverse_maximum_absolute_error": initial_inverse_error,
        }
    ]
    mismatch_fields = sorted({str(item["field"]) for item in mismatches})
    population = {
        "fresh_input_rows": frame.height,
        "fresh_selected_rows": view.height,
        "input_expiries": frame.get_column("expiry_id").n_unique(),
        "selected_expiries": len(groups),
        "initial_retained_expiries": frame.get_column(
            "initial_no_arb_input_expiry_index"
        ).drop_nulls().n_unique(),
        "smooth_retained_expiries": len(retained_groups),
        "active_call_books": int(view.get_column("call_active").sum()),
        "active_put_books": int(view.get_column("put_active").sum()),
        "smooth_masked_strikes": int(
            view.filter(~pl.col("smooth_fit_mask")).height
        ),
        "dropped_expiry_ids": [
            int(group.get_column("expiry_id").item(0))
            for group in groups
            if group.get_column("smooth_input_expiry_index").null_count() == group.height
        ],
    }
    return cast(
        dict[str, Any],
        canonical_value(
            {
                "matched": not mismatches,
                "mismatch_fields": mismatch_fields,
                "mismatches": mismatches,
                "numerical_comparisons": numerical,
                "population": population,
                "source_canonicality": {
                    "fresh_already_canonical": fresh_canonical,
                    "fresh_unique_keys": fresh_unique,
                    "legacy_already_canonical": legacy_canonical,
                    "legacy_unique_keys": legacy_unique,
                },
                "item_oracle_gaps": item_oracle_gaps,
            }
        ),
    )


@version("1.0")
def _manifest_file_digest(artifact: ValidatedArtifact, name: str) -> str:
    files = artifact.manifest.get("files")
    if not isinstance(files, list):
        raise ReferenceComparisonError("Fresh artifact manifest has no file inventory")
    matches = [item for item in files if isinstance(item, Mapping) and item.get("path") == name]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ReferenceComparisonError(f"Fresh artifact manifest has no unique {name} checksum")
    return str(matches[0]["sha256"])


@version("1.0")
def _daily_observations(frame: pl.DataFrame) -> dict[str, Any]:
    """Return non-secret daily facts needed by the guarded pilot contract."""

    if frame.is_empty():
        return {
            "populated": False,
            "is_half_day": None,
            "irregular_expiry_rows_removed": 0,
            "point_two_am_active_side_books": 0,
        }
    diagnostics_values = frame.get_column("normalization_diagnostics_json").unique().to_list()
    if len(diagnostics_values) != 1 or not isinstance(diagnostics_values[0], str):
        raise ReferenceComparisonError("Fresh normalization diagnostics are not constant")
    try:
        diagnostics = json.loads(diagnostics_values[0])
    except json.JSONDecodeError as exc:
        raise ReferenceComparisonError("Fresh normalization diagnostics are invalid JSON") from exc
    if not isinstance(diagnostics, Mapping):
        raise ReferenceComparisonError("Fresh normalization diagnostics are not an object")
    filter_counts = diagnostics.get("filter_counts")
    irregular = (
        filter_counts.get("irregular_expiry")
        if isinstance(filter_counts, Mapping)
        else None
    )
    removed = irregular.get("removed") if isinstance(irregular, Mapping) else 0
    if isinstance(removed, bool) or not isinstance(removed, int) or removed < 0:
        raise ReferenceComparisonError("Fresh irregular-expiry removal count is invalid")
    point_two = (
        frame.get_column("expiry_is_am").to_numpy().astype(bool)
        & np.isclose(
            frame.get_column("business_dte").to_numpy(),
            0.2,
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    active = (
        frame.get_column("call_active").to_numpy().astype(np.int64)
        + frame.get_column("put_active").to_numpy().astype(np.int64)
    )
    half_day = frame.get_column("is_half_day").unique().to_list()
    if len(half_day) != 1 or not isinstance(half_day[0], bool):
        raise ReferenceComparisonError("Fresh half-day indicator is not constant")
    return {
        "populated": True,
        "is_half_day": half_day[0],
        "irregular_expiry_rows_removed": removed,
        "point_two_am_active_side_books": int(np.sum(active[point_two])),
    }


@version(
    "1.0",
    dependencies=[ValidatedArtifact, canonical_value, raw_artifact_validation_record],
)
def _phase10_irregular_raw_evidence(
    artifact: ValidatedArtifact,
    *,
    required_expiry: date = date(2025, 1, 9),
) -> dict[str, Any]:
    """Bind the Raw options artifact and count the fixed irregular-expiry case."""

    provenance = raw_artifact_validation_record(artifact)
    if provenance.get("logical_table") != "options":
        raise ReferenceComparisonError("The irregular-expiry evidence is not Raw options")
    data_path = artifact.path / "data.parquet"
    try:
        expiries = pl.read_parquet(data_path, columns=["exdate"]).get_column("exdate")
        parsed = expiries.cast(pl.String).str.to_date(strict=True)
    except Exception as exc:
        raise ReferenceComparisonError(
            f"Cannot inspect Raw irregular-expiry evidence: {exc}"
        ) from exc
    matching_rows = int((parsed == required_expiry).sum())
    return cast(
        dict[str, Any],
        canonical_value(
            {
                "artifact": provenance["artifact"],
                "path": provenance["path"],
                "source_generation": provenance["source_generation"],
                "physical_table": provenance["physical_table"],
                "observation_date": provenance["observation_date"],
                "query_sha256": provenance["query_sha256"],
                "row_count": provenance["row_count"],
                "empty_vendor_result": provenance["empty_vendor_result"],
                "persisted_schema": provenance["persisted_schema"],
                "data_sha256": provenance["data_sha256"],
                "required_expiry": required_expiry,
                "required_expiry_row_count": matching_rows,
            }
        ),
    )


@version("1.0")
def _phase10_pilot_day_checks(
    day_key: str,
    *,
    observations: Mapping[str, Any],
    typed_empty_reason: str | None,
    record: Mapping[str, Any] | None,
    raw_case_evidence: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the fixed representative cases without changing fresh values."""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "passed": passed,
                "observed": observed,
                "expected": expected,
            }
        )

    if day_key in {"2024-12-23", "2025-01-06"}:
        add(
            "ordinary_day_populated",
            observations["populated"] is True,
            observations["populated"],
            True,
        )
    if day_key == "2024-12-24":
        add(
            "half_day_populated",
            observations["populated"] is True,
            observations["populated"],
            True,
        )
        add(
            "half_day_indicator",
            observations["is_half_day"] is True,
            observations["is_half_day"],
            True,
        )
        add("half_day_oracle_absent", record is None, record is not None, False)
    if day_key == "2025-01-07":
        raw_rows = (
            None
            if raw_case_evidence is None
            else raw_case_evidence.get("required_expiry_row_count")
        )
        add(
            "irregular_expiry_raw_case_present",
            isinstance(raw_rows, int) and raw_rows > 0,
            raw_rows,
            "> 0 rows for expiry 2025-01-09",
        )
        removed = observations["irregular_expiry_rows_removed"]
        add(
            "irregular_expiry_case_exercised",
            isinstance(removed, int) and removed > 0,
            removed,
            "> 0",
        )
    if day_key == "2025-01-08":
        add(
            "policy_empty_frame",
            observations["populated"] is False,
            observations["populated"],
            False,
        )
        add(
            "policy_empty_reason",
            typed_empty_reason == "known_bad_day_all_tickers",
            typed_empty_reason,
            "known_bad_day_all_tickers",
        )
        add(
            "policy_empty_legacy_marker",
            record is not None and record.get("populated") is False,
            None if record is None else record.get("populated"),
            False,
        )
    if day_key == "2025-01-16":
        active = observations["point_two_am_active_side_books"]
        add(
            "point_two_am_node_present",
            isinstance(active, int) and active > 0,
            active,
            "> 0",
        )
        add(
            "legacy_point_two_active_side_books",
            record is not None and record.get("active_point_two_day_books") == 186,
            None if record is None else record.get("active_point_two_day_books"),
            186,
        )
    return checks


@version(
    "1.0",
    dependencies=[
        DailyComparisonRequest,
        DailySurfaceConfig,
        DailyWidbConfig,
        canonical_sha256,
    ],
)
def _validate_phase10_pilot_contract(
    request: DailyComparisonRequest,
    *,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig,
    plans: Sequence[Any],
    catalog_records: Mapping[str, Mapping[str, Any]],
    require_pinned_catalog: bool,
) -> dict[str, Any]:
    """Validate the immutable pilot selection and frozen catalog population."""

    if not require_pinned_catalog:
        raise ReferenceComparisonError("The Phase 10 pilot requires the pinned catalog")
    observed_dates = tuple(item.trading_date.isoformat() for item in plans)
    declared_gaps = tuple(sorted(value.isoformat() for value in request.declared_oracle_gaps))
    failures: list[str] = []
    if request.ticker != "SPX":
        failures.append(f"ticker={request.ticker!r}, expected 'SPX'")
    if request.start_date != date(2024, 12, 23) or request.end_date != date(2025, 1, 16):
        failures.append("date interval must be 2024-12-23 through 2025-01-16")
    if observed_dates != PHASE10_PILOT_DATES:
        failures.append(f"planned sessions={observed_dates!r}, expected={PHASE10_PILOT_DATES!r}")
    if declared_gaps != PHASE10_PILOT_ORACLE_GAPS:
        failures.append(
            f"declared oracle gaps={declared_gaps!r}, expected={PHASE10_PILOT_ORACLE_GAPS!r}"
        )
    if daily_config.clock.overnight_variance_fraction != 0.2:
        failures.append("overnight_variance_fraction must equal 0.2")
    if not daily_config.raw.drop_irregular_expiries:
        failures.append("drop_irregular_expiries must be true")
    if surface_config.options_filter.remove_am:
        failures.append("remove_am must be false")
    expected_daily = DailyWidbConfig()
    expected_surface = DailySurfaceConfig()
    if daily_config != expected_daily:
        failures.append(
            "daily configuration does not equal the frozen reference DailyWidbConfig"
        )
    if surface_config != expected_surface:
        failures.append(
            "surface configuration does not equal the frozen reference DailySurfaceConfig"
        )

    selected = [catalog_records.get(day_key) for day_key in PHASE10_PILOT_DATES]
    populated = [
        record for record in selected if record is not None and record.get("populated") is True
    ]
    empty = [
        record for record in selected if record is not None and record.get("populated") is False
    ]
    if len(populated) != 14 or len(empty) != 1 or selected[1] is not None:
        failures.append(
            "pinned pilot catalog must contain 14 populated files, one empty marker, "
            "and no 2024-12-24 file"
        )
    aggregate_fields = {
        "num_expiries": 448,
        "num_strike_nodes": 40_436,
        "active_call_books": 38_022,
        "active_put_books": 38_022,
        "active_books": 76_044,
        "active_point_two_day_books": 186,
    }
    observed_aggregates: dict[str, int] = {}
    for field, expected in aggregate_fields.items():
        value = sum(
            int(record.get(field, -10**9))
            for record in populated
            if isinstance(record, Mapping)
        )
        observed_aggregates[field] = value
        if value != expected:
            failures.append(f"catalog {field}={value}, expected {expected}")
    if failures:
        raise ReferenceComparisonError(
            "Phase 10 pilot contract validation failed: " + "; ".join(failures)
        )
    return {
        "enabled": True,
        "passed": True,
        "expected_dates": PHASE10_PILOT_DATES,
        "expected_oracle_gaps": PHASE10_PILOT_ORACLE_GAPS,
        "catalog_populated_files": len(populated),
        "catalog_empty_markers": len(empty),
        "catalog_aggregates": observed_aggregates,
        "daily_configuration_id": canonical_sha256(daily_config),
        "surface_configuration_id": canonical_sha256(surface_config),
    }


@version("1.0")
def load_comparison_approvals(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load exact, evidence-bound discrepancy approvals."""

    if path is None:
        return {}
    if path.is_symlink() or not path.is_file():
        raise ReferenceComparisonError(f"Approval file is missing or unsafe: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceComparisonError(f"Cannot decode approval file {path}: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"schema", "approvals"}:
        raise ReferenceComparisonError("Approval document fields are missing or unexpected")
    if document["schema"] != APPROVAL_SCHEMA or not isinstance(document["approvals"], list):
        raise ReferenceComparisonError("Approval document has an unsupported schema")
    required = {
        "date",
        "fields",
        "fresh_artifact_id",
        "fresh_data_sha256",
        "legacy_file_sha256",
        "explanation",
        "approver",
        "approval_date",
        "approval_reference",
    }
    approvals: dict[str, dict[str, Any]] = {}
    for approval in document["approvals"]:
        if not isinstance(approval, dict) or set(approval) != required:
            raise ReferenceComparisonError("Approval fields are missing or unexpected")
        try:
            date.fromisoformat(str(approval["date"]))
            date.fromisoformat(str(approval["approval_date"]))
        except ValueError as exc:
            raise ReferenceComparisonError("Approval dates must be ISO dates") from exc
        fields = approval["fields"]
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(field, str) and field for field in fields)
            or fields != sorted(set(fields))
        ):
            raise ReferenceComparisonError("Approval fields must be sorted unique strings")
        for digest_field in ("fresh_artifact_id", "fresh_data_sha256", "legacy_file_sha256"):
            value = approval[digest_field]
            if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
                raise ReferenceComparisonError(f"Approval {digest_field} must be 64-hex")
        for text_field in ("explanation", "approver", "approval_reference"):
            if not isinstance(approval[text_field], str) or not approval[text_field].strip():
                raise ReferenceComparisonError(f"Approval {text_field} cannot be empty")
        key = str(approval["date"])
        if key in approvals:
            raise ReferenceComparisonError(f"Duplicate comparison approval for {key}")
        approvals[key] = approval
    return approvals


@version("1.0")
def _approval_applies(
    approval: Mapping[str, Any] | None,
    *,
    fields: Sequence[str],
    fresh_artifact_id: str,
    fresh_data_sha256: str,
    legacy_file_sha256: str,
) -> bool:
    return bool(
        approval is not None
        and list(fields) == approval.get("fields")
        and fresh_artifact_id == approval.get("fresh_artifact_id")
        and fresh_data_sha256 == approval.get("fresh_data_sha256")
        and legacy_file_sha256 == approval.get("legacy_file_sha256")
    )


ProgressCallback = Callable[[Mapping[str, Any]], None]


@version(
    "1.0",
    dependencies=[
        DailyComparisonRequest,
        ArtifactStore,
        DailySurfaceConfig,
        DailyWidbConfig,
        IntermediateBuildRequest,
        InternalDatasetConfig,
        _approval_applies,
        _daily_observations,
        _manifest_file_digest,
        _phase10_pilot_day_checks,
        _phase10_irregular_raw_evidence,
        _reference_sha256_file,
        _validate_phase10_pilot_contract,
        canonical_sha256,
        canonical_value,
        compare_populated_day,
        load_comparison_approvals,
        load_reference_catalog,
        plan_daily_artifacts,
        read_daily_frame,
        read_legacy_pytree_hdf5,
        validate_fitted_partition,
    ],
)
def compare_daily_artifacts(
    locations: StorageLocations,
    request: DailyComparisonRequest,
    *,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig,
    legacy_root: Path,
    catalog_path: Path,
    source_revision: str,
    approvals_path: Path | None = None,
    require_pinned_catalog: bool = True,
    progress: ProgressCallback | None = None,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Validate and compare existing fitted artifacts without building or networking."""

    if not 0.0 < progress_interval_seconds <= 300.0:
        raise ValueError("progress_interval_seconds must lie in (0, 300]")
    if _HEX_40.fullmatch(source_revision) is None:
        raise ReferenceComparisonError("source_revision must be an exact lowercase 40-hex commit")
    if not legacy_root.is_dir() or legacy_root.is_symlink():
        raise ReferenceComparisonError(f"Legacy root is missing or unsafe: {legacy_root}")
    catalog = load_reference_catalog(catalog_path)
    catalog_sha256 = _reference_sha256_file(catalog_path)
    if require_pinned_catalog and (
        catalog_sha256 != PINNED_CATALOG_SHA256
        or catalog["collection_sha256"] != PINNED_COLLECTION_SHA256
    ):
        raise ReferenceComparisonError(
            "Legacy catalog does not match the pinned 891-date migration oracle"
        )
    approvals = load_comparison_approvals(approvals_path)
    catalog_records: dict[str, Mapping[str, Any]] = {}
    for record in catalog["files"]:
        if not isinstance(record, Mapping):
            raise ReferenceComparisonError("Legacy catalog contains a malformed record")
        trading_date = record.get("trading_date")
        if not isinstance(trading_date, str):
            raise ReferenceComparisonError("Legacy catalog record has no trading date")
        try:
            date.fromisoformat(trading_date)
        except ValueError as exc:
            raise ReferenceComparisonError(
                f"Legacy catalog has an invalid trading date: {trading_date!r}"
            ) from exc
        if trading_date in catalog_records:
            raise ReferenceComparisonError(
                f"Legacy catalog contains duplicate date {trading_date}"
            )
        catalog_records[trading_date] = record
    plan_request = IntermediateBuildRequest(
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        through="fitted-day",
        include_half_days=True,
        source=request.source,
        source_generation=request.source_generation,
    )
    plans = plan_daily_artifacts(plan_request, daily_config, surface_config)
    planned_dates = {item.trading_date.isoformat() for item in plans}
    declared_gaps = {value.isoformat() for value in request.declared_oracle_gaps}
    gaps_outside_request = sorted(declared_gaps - planned_dates)
    if gaps_outside_request:
        raise ReferenceComparisonError(
            f"Declared oracle gaps fall outside the planned sessions: {gaps_outside_request}"
        )
    pilot_contract = (
        _validate_phase10_pilot_contract(
            request,
            daily_config=daily_config,
            surface_config=surface_config,
            plans=plans,
            catalog_records=catalog_records,
            require_pinned_catalog=require_pinned_catalog,
        )
        if request.enforce_phase10_pilot
        else {"enabled": False, "passed": True}
    )
    store = ArtifactStore(locations)
    validation_config = InternalDatasetConfig(
        name="daily-reference-comparison",
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        include_half_days=True,
        source="optionmetrics",
        source_generation=request.source_generation,
        daily=daily_config,
        surface=surface_config,
    )
    started = time.monotonic()
    last_progress = started
    days: list[dict[str, Any]] = []
    selected_legacy_files: list[dict[str, Any]] = []
    for completed, item in enumerate(plans, start=1):
        artifact = store.validate(item.fitted.pointer, expected=item.fitted)
        frame = read_daily_frame(
            store,
            artifact,
            schema=FITTED_SCHEMA,
            ticker=request.ticker,
            trading_date=item.trading_date,
        )
        metadata = artifact.manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ReferenceComparisonError("Fresh fitted artifact metadata is malformed")
        typed_empty_reason = metadata.get("typed_empty_reason")
        statistics = validate_fitted_partition(
            frame,
            config=validation_config,
            trading_date=item.trading_date,
            typed_empty_reason=(
                str(typed_empty_reason) if typed_empty_reason is not None else None
            ),
            expected_raw_options_artifact_id=item.raw.options.artifact_id,
        )
        data_sha256 = _manifest_file_digest(artifact, "data.parquet")
        day_key = item.trading_date.isoformat()
        record = catalog_records.get(day_key)
        observations = _daily_observations(frame)
        raw_case_evidence: dict[str, Any] | None = None
        if request.enforce_phase10_pilot and day_key == "2025-01-07":
            raw_options = store.validate(item.raw.options)
            raw_case_evidence = _phase10_irregular_raw_evidence(raw_options)
            if (
                raw_case_evidence["source_generation"] != request.source_generation
                or raw_case_evidence["observation_date"] != day_key
            ):
                raise ReferenceComparisonError(
                    "Raw irregular-expiry evidence has the wrong generation or date"
                )
        legacy_digest: str | None = None
        comparison: dict[str, Any] | None = None
        if record is None:
            unpinned_path = legacy_root / f"{day_key}.h5"
            if unpinned_path.exists() or unpinned_path.is_symlink():
                raise ReferenceComparisonError(
                    f"Unpinned legacy file exists for catalog gap {day_key}: {unpinned_path}"
                )
            classification: DailyClassification = (
                "oracle_gap" if day_key in declared_gaps and not frame.is_empty()
                else "unexplained_discrepancy"
            )
            if day_key in approvals:
                raise ReferenceComparisonError(
                    f"Approval for {day_key} cannot apply without a physical legacy file"
                )
        else:
            file_name = record.get("file")
            expected_digest = record.get("sha256")
            expected_size = record.get("byte_size")
            expected_populated = record.get("populated")
            if (
                not isinstance(file_name, str)
                or file_name != f"{day_key}.h5"
                or not isinstance(expected_digest, str)
                or isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
                or not isinstance(expected_populated, bool)
            ):
                raise ReferenceComparisonError(f"Malformed catalog record for {day_key}")
            legacy_path = legacy_root / file_name
            if legacy_path.is_symlink() or not legacy_path.is_file():
                raise ReferenceComparisonError(f"Legacy file is missing or unsafe: {legacy_path}")
            actual_size = legacy_path.stat().st_size
            if actual_size != expected_size:
                raise ReferenceComparisonError(
                    f"Legacy file size changed for {day_key}: "
                    f"expected {expected_size}, observed {actual_size}"
                )
            legacy_digest = _reference_sha256_file(legacy_path)
            if legacy_digest != expected_digest:
                raise ReferenceComparisonError(
                    f"Legacy file checksum changed for {day_key}: "
                    f"expected {expected_digest}, observed {legacy_digest}"
                )
            selected_legacy_files.append(
                {
                    "date": day_key,
                    "file": file_name,
                    "sha256": legacy_digest,
                    "byte_size": actual_size,
                    "populated": expected_populated,
                    "producer_version": record.get("producer_version"),
                }
            )
            payload = read_legacy_pytree_hdf5(legacy_path)
            if bool(payload) is not expected_populated:
                raise ReferenceComparisonError(
                    f"Legacy payload population contradicts its catalog for {day_key}"
                )
            if frame.is_empty() and not payload:
                classification = "policy_empty"
            elif frame.is_empty() or not payload:
                classification = "unexplained_discrepancy"
            else:
                comparison = compare_populated_day(
                    frame,
                    payload,
                    tolerances=request.tolerances,
                )
                classification = (
                    "match" if comparison["matched"] else "unexplained_discrepancy"
                )
                approval = approvals.get(day_key)
                if classification == "unexplained_discrepancy" and _approval_applies(
                    approval,
                    fields=cast(list[str], comparison["mismatch_fields"]),
                    fresh_artifact_id=artifact.pointer.artifact_id,
                    fresh_data_sha256=data_sha256,
                    legacy_file_sha256=legacy_digest,
                ):
                    classification = "approved_discrepancy"
                elif approval is not None and classification != "approved_discrepancy":
                    raise ReferenceComparisonError(
                        f"Approval for {day_key} is stale, broadened, or does not bind "
                        "exact evidence"
                    )
        if day_key in approvals and classification != "approved_discrepancy":
            raise ReferenceComparisonError(
                f"Approval for {day_key} is stale, broadened, or does not bind exact evidence"
            )
        pilot_checks = (
            _phase10_pilot_day_checks(
                day_key,
                observations=observations,
                typed_empty_reason=(
                    str(typed_empty_reason) if typed_empty_reason is not None else None
                ),
                record=record,
                raw_case_evidence=raw_case_evidence,
            )
            if request.enforce_phase10_pilot
            else []
        )
        if any(not check["passed"] for check in pilot_checks):
            classification = "unexplained_discrepancy"
        days.append(
            cast(
                dict[str, Any],
                canonical_value(
                    {
                        "date": day_key,
                        "classification": classification,
                        "fresh": {
                            "artifact": artifact.pointer.as_dict(),
                            "path": str(artifact.path),
                            "data_sha256": data_sha256,
                            "schema_version": artifact.manifest.get("schema_version"),
                            "producer_version": artifact.manifest.get("producer_version"),
                            "configuration_id": artifact.manifest.get("configuration_id"),
                            "source_generation": artifact.manifest.get("source_generation"),
                            "statistics": statistics.as_dict(),
                        },
                        "observations": observations,
                        "raw_case_evidence": raw_case_evidence,
                        "pilot_checks": pilot_checks,
                        "legacy": (
                            None
                            if record is None
                            else {
                                "file": record.get("file"),
                                "sha256": legacy_digest,
                                "producer_version": record.get("producer_version"),
                                "populated": record.get("populated"),
                            }
                        ),
                        "comparison": comparison,
                        "approval": (
                            approvals.get(day_key)
                            if classification == "approved_discrepancy"
                            else None
                        ),
                    }
                ),
            )
        )
        now = time.monotonic()
        if progress is not None and (
            completed == len(plans) or now - last_progress >= progress_interval_seconds
        ):
            elapsed = now - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            progress(
                {
                    "event": "reference_comparison_progress",
                    "elapsed_seconds": elapsed,
                    "completed": completed,
                    "total": len(plans),
                    "failed": sum(
                        day["classification"] == "unexplained_discrepancy" for day in days
                    ),
                    "estimated_remaining_seconds": (
                        (len(plans) - completed) / rate if rate > 0.0 else None
                    ),
                }
            )
            last_progress = now
    unused_approvals = sorted(set(approvals) - {str(day["date"]) for day in days})
    if unused_approvals:
        raise ReferenceComparisonError(
            f"Approval document contains dates outside the request: {unused_approvals}"
        )
    counts = dict(sorted(Counter(str(day["classification"]) for day in days).items()))
    outcome = "blocked" if counts.get("unexplained_discrepancy", 0) else "passed"
    pilot_daily_checks_passed = all(
        check["passed"] for day in days for check in day["pilot_checks"]
    )
    pilot_contract = {
        **pilot_contract,
        "daily_checks_passed": pilot_daily_checks_passed,
        "passed": bool(pilot_contract["passed"] and pilot_daily_checks_passed),
    }
    report_without_id = cast(
        dict[str, Any],
        canonical_value(
            {
                "schema": REFERENCE_REPORT_SCHEMA,
                "outcome": outcome,
                "source_revision": source_revision,
                "request": {
                    "ticker": request.ticker,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "source": request.source,
                    "source_generation": request.source_generation,
                    "declared_oracle_gaps": request.declared_oracle_gaps,
                    "enforce_phase10_pilot": request.enforce_phase10_pilot,
                },
                "resolved_versions": {
                    "comparison": str(compare_daily_artifacts.version),
                    "tolerances": str(DailyLegacyComparisonTolerances.version),
                    "daily_config": str(DailyWidbConfig.version),
                    "surface_config": str(DailySurfaceConfig.version),
                },
                "configuration": {
                    "daily": daily_config,
                    "surface": surface_config,
                },
                "tolerances": request.tolerances.to_dict(),
                "legacy_catalog": {
                    "path": str(catalog_path),
                    "sha256": catalog_sha256,
                    "collection_sha256": catalog["collection_sha256"],
                    "source_revision": catalog["source_revision"],
                    "relative_path": catalog["relative_path"],
                    "selected_files": selected_legacy_files,
                },
                "classification_counts": counts,
                "pilot_contract": pilot_contract,
                "dates": days,
                "network_accessed": False,
                "artifacts_built": False,
            }
        ),
    )
    return {"report_id": canonical_sha256(report_without_id), **report_without_id}


@version("1.0", dependencies=[atomic_write_json])
def _atomic_write_new_json(output: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without ever replacing a winner."""

    temporary = output.parent / f".{output.name}.publish-{uuid.uuid4().hex}"
    try:
        atomic_write_json(temporary, value)
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise ReferenceComparisonError(
                f"Comparison output already exists: {output}"
            ) from exc
        directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@version("1.0", dependencies=[_atomic_write_new_json])
def write_comparison_report(
    output: Path,
    report: Mapping[str, Any],
    *,
    legacy_root: Path,
) -> None:
    """Publish immutable comparison evidence atomically outside the oracle root."""

    destination = output.expanduser().absolute()
    oracle = legacy_root.expanduser().resolve(strict=True)
    repository = Path(__file__).resolve().parents[3]
    resolved_parent = destination.parent.resolve(strict=False)
    if resolved_parent == oracle or oracle in resolved_parent.parents:
        raise ReferenceComparisonError(
            "Comparison output must be outside the read-only legacy root"
        )
    if resolved_parent == repository or repository in resolved_parent.parents:
        raise ReferenceComparisonError("Comparison output must be outside the Git working tree")
    if destination.exists() or destination.is_symlink():
        raise ReferenceComparisonError(f"Comparison output already exists: {destination}")
    _atomic_write_new_json(destination, report)


__all__ = [
    "APPROVAL_SCHEMA",
    "DailyComparisonRequest",
    "DailyLegacyComparisonTolerances",
    "PINNED_CATALOG_SHA256",
    "PINNED_COLLECTION_SHA256",
    "REFERENCE_REPORT_SCHEMA",
    "ReferenceComparisonError",
    "compare_daily_artifacts",
    "compare_populated_day",
    "load_comparison_approvals",
    "load_reference_catalog",
    "repository_source_revision",
    "write_comparison_report",
]
