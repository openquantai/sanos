"""Immutable, self-contained internal WIDB/SANOS Dataset publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import polars as pl
from cdxcore.bs import bs
from cdxcore.version import version

from sanos.artifacts import (
    ArtifactExistsError,
    ArtifactIntegrityError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    DatasetDistribution,
    DistributionStatus,
    ReuseResult,
    StorageClass,
    ValidatedArtifact,
)
from sanos.config import StorageLocations, canonical_json, canonical_sha256, canonical_value
from sanos.datasets.config import DatasetBuildOptions, InternalDatasetConfig
from sanos.model.constants import one_minute_year_fraction
from sanos.model.tools import SanosLPSolver, SanosPricer
from sanos.pipeline import DailyRunSummary, RunJournal
from sanos.resources import packaged_source_revision
from sanos.widb.bad import bad_data_policy, bad_day_reason
from sanos.widb.build import (
    DailyArtifactPlan,
    IntermediateBuildRequest,
    build_intermediates,
    enriched_day_producer_version,
    filtered_day_producer_version,
    fitted_day_producer_version,
    no_arb_day_producer_version,
    paired_day_producer_version,
    plan_daily_artifacts,
    read_daily_frame,
)
from sanos.widb.calendar import DAYS_PER_YEAR, count_business_days_between, half_days, is_half_day
from sanos.widb.daily import (
    _black_implied,
    _black_values,
    _implied_volatility_price_tolerance,
    _numerical_tolerances,
)
from sanos.widb.surface import (
    FILTERED_SCHEMA,
    FITTED_SCHEMA,
    FITTED_SCHEMA_VERSION,
    NO_ARB_SCHEMA,
    SOLVER_TIMEOUT_SECONDS,
    _canonical_discrete_dlv_densities,
    _canonical_discrete_dlv_squares,
    _discrete_dlv_grid,
    _validate_discrete_dlv_projection,
    apply_options_filter,
    discrete_dlv_pricer,
    fit_smooth_sanos,
    initial_no_arb_pricer,
    smooth_sanos_pricer,
)
from sanos.wrds import raw_content_producer_version

RAW_CATALOG_SCHEMA_VERSION = "widb-raw-lineage-catalog-v1"
INTERNAL_DATASET_SCHEMA_VERSION = "sanos-internal-widb-dataset-v4"
PARTITION_CATALOG_SCHEMA = "sanos-internal-partitions-v1"
LINEAGE_SCHEMA = "sanos-internal-lineage-v1"
VALIDATION_SCHEMA = "sanos-internal-validation-v6"
DATASET_SCHEMA_DOCUMENT = "sanos-internal-schema-v4"
LEGACY_COMPARISON_SCHEMA = "sanos-legacy-comparison-v4"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_PARTITION_PATH = re.compile(
    r"^data/ticker=([A-Za-z0-9._-]+)/year=(\d{4})/date=(\d{4}-\d{2}-\d{2})/"
    r"part-00000\.parquet$"
)
_SAFE_PARTITION_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetValidationError(ArtifactIntegrityError):
    """Raised when a Dataset is intact but violates its semantic contract."""


@version("1.0")
@dataclass(frozen=True, slots=True)
class PartitionStatistics:
    """Recomputed populations and residual maxima for one daily partition."""

    trading_date: date
    row_count: int
    input_expiries: int
    initial_retained_expiries: int
    mask_arb_free_rows: int
    eligible_rows: int
    selected_expiries: int
    smooth_retained_expiries: int
    smooth_fit_mask_rows: int
    smooth_model_mask_rows: int
    active_call_rows: int
    active_put_rows: int
    active_point_two_day_side_books: int
    typed_empty: bool
    typed_empty_reason: str | None
    rejection_counts: Mapping[str, int]
    residual_maxima: Mapping[str, float]

    @property
    def active_side_books(self) -> int:
        return self.active_call_rows + self.active_put_rows

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], canonical_value(self))


@version("1.0")
@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    """Self-contained semantic validation result for a published Dataset."""

    requested_dates: int
    populated_dates: int
    typed_empty_dates: int
    scheduled_exclusions: int
    row_count: int
    input_expiries: int
    initial_retained_expiries: int
    mask_arb_free_rows: int
    eligible_rows: int
    selected_expiries: int
    smooth_retained_expiries: int
    smooth_fit_mask_rows: int
    smooth_model_mask_rows: int
    active_call_rows: int
    active_put_rows: int
    active_point_two_day_side_books: int
    rejection_counts: Mapping[str, int]
    residual_maxima: Mapping[str, float]
    exclusions: tuple[Mapping[str, Any], ...]
    outcome: str = "passed"

    @property
    def active_side_books(self) -> int:
        return self.active_call_rows + self.active_put_rows

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], canonical_value(self))


@version("1.0", dependencies=[DailyArtifactPlan, ArtifactSpec])
@dataclass(frozen=True, slots=True)
class InternalDatasetPlan:
    """Pure, location-independent identities for one Dataset publication."""

    configuration: Mapping[str, Any]
    expected_dates: tuple[date, ...]
    daily_artifacts: tuple[DailyArtifactPlan, ...]
    raw_mapping_digest: str
    raw_catalog_spec: ArtifactSpec
    dataset_spec: ArtifactSpec
    plan_id: str


@version(
    "1.0",
    dependencies=[
        ArtifactPointer,
        ReuseResult,
        DailyRunSummary,
        DatasetValidationSummary,
    ],
)
@dataclass(frozen=True, slots=True)
class InternalDatasetBuildResult:
    """Dry-run, reuse, or completed Dataset build outcome."""

    plan: InternalDatasetPlan
    artifact: ArtifactPointer
    path: Path
    dry_run: bool
    built: bool
    reuse: ReuseResult | None
    daily_summary: DailyRunSummary | None
    validation: DatasetValidationSummary | None
    journal_path: Path


@version("1.0")
def _fail(message: str) -> None:
    raise DatasetValidationError(message)


@version("1.0")
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@version("1.0")
def _canonical_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or canonical_json(value) != text:
        _fail(f"{label} must be a canonical JSON object")
    return cast(dict[str, Any], value)


@version("1.0")
def _reason_counts(frame: pl.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for column in (
        "initial_no_arb_rejection_reason",
        "options_filter_rejection_reason",
        "smooth_rejection_reason",
    ):
        for value in frame.get_column(column).drop_nulls().to_list():
            counts[f"{column}:{value}"] += 1
    return dict(sorted(counts.items()))


@version("1.0")
@dataclass(frozen=True, slots=True)
class _SolverCall:
    """One logical solve, including an optional adjacent fallback event."""

    stage: str
    input_expiry_index: int | None
    expiry_attempt: int | None
    terminal_status: str
    used_fallback: bool


@version("1.0", dependencies=[_SolverCall])
@dataclass(frozen=True, slots=True)
class _TerminalSolverEvidence:
    """Validated terminal surface call and any intentionally relaxed support."""

    relaxed_support: frozenset[int]
    terminal_solver_call: _SolverCall


@version("1.0", dependencies=[_SolverCall])
def _solver_calls(events: list[Any], label: str) -> list[_SolverCall]:
    """Parse and validate the exact diagnostic event and fallback grammar."""

    expiry_stages = {"expiry_soft", "expiry_hard"}
    allowed_stages = expiry_stages | {"surface", "discrete_dlv_surface"}
    successful = {"optimal", "optimal_inaccurate"}
    retry_triggers = {"solver_error", "invalid_solution", "user_limit"}
    base_fields = {
        "stage",
        "solver",
        "algorithm",
        "status",
        "is_fallback",
        "for_expiry",
        "objective",
    }

    def validate_event(event: Any) -> tuple[str, int | None, int | None]:
        if not isinstance(event, Mapping):
            _fail(f"{label} diagnostics contains a malformed solver event")
        stage = event.get("stage")
        if not isinstance(stage, str) or stage not in allowed_stages:
            _fail(f"{label} diagnostics has an unknown solver stage {stage!r}")
        is_expiry = stage in expiry_stages
        expected_fields = base_fields | (
            {"input_expiry_index", "expiry_attempt"} if is_expiry else set()
        )
        if set(event) != expected_fields:
            _fail(f"{label} diagnostics {stage} event fields changed")
        if event.get("algorithm") != "simplex":
            _fail(f"{label} diagnostics {stage} solver algorithm changed")
        is_fallback = event.get("is_fallback")
        for_expiry = event.get("for_expiry")
        if not isinstance(is_fallback, bool) or not isinstance(for_expiry, bool):
            _fail(f"{label} diagnostics {stage} has invalid boolean fields")
        if for_expiry is not is_expiry:
            _fail(f"{label} diagnostics {stage} has inconsistent expiry scope")
        expected_solver = "CLARABEL" if is_fallback else "HIGHS"
        if event.get("solver") != expected_solver:
            _fail(f"{label} diagnostics {stage} has inconsistent solver identity")
        status = event.get("status")
        if not isinstance(status, str) or not status:
            _fail(f"{label} diagnostics {stage} has invalid solver status")
        objective = event.get("objective")
        if objective is not None and (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(float(objective))
        ):
            _fail(f"{label} diagnostics {stage} has invalid solver objective")
        if status in successful and objective is None:
            _fail(f"{label} diagnostics {stage} success has no finite objective")
        if status not in successful | {"user_limit"} and objective is not None:
            _fail(f"{label} diagnostics {stage} failure has an objective value")
        if not is_expiry:
            return stage, None, None
        input_index = event.get("input_expiry_index")
        attempt = event.get("expiry_attempt")
        if (
            isinstance(input_index, bool)
            or not isinstance(input_index, int)
            or input_index < 0
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt not in {1, 2}
        ):
            _fail(f"{label} diagnostics expiry event has invalid input index/attempt")
        return stage, input_index, attempt

    calls: list[_SolverCall] = []
    position = 0
    while position < len(events):
        primary = events[position]
        stage, input_index, attempt = validate_event(primary)
        if primary["is_fallback"]:
            _fail(f"{label} diagnostics has an orphan fallback event")
        terminal = primary
        used_fallback = False
        if primary["status"] in retry_triggers:
            if position + 1 >= len(events):
                _fail(f"{label} diagnostics has a dangling fallback trigger")
            fallback = events[position + 1]
            fallback_key = validate_event(fallback)
            if not fallback["is_fallback"] or fallback_key != (
                stage,
                input_index,
                attempt,
            ):
                _fail(f"{label} diagnostics has a mismatched fallback event")
            terminal = fallback
            used_fallback = True
            position += 1
        calls.append(
            _SolverCall(
                stage=stage,
                input_expiry_index=input_index,
                expiry_attempt=attempt,
                terminal_status=cast(str, terminal["status"]),
                used_fallback=used_fallback,
            )
        )
        position += 1
    return calls


@version("5.0", dependencies=[_solver_calls, _TerminalSolverEvidence])
def _terminal_solver_events(
    value: Mapping[str, Any],
    label: str,
    *,
    expected_expiry_indices: tuple[int, ...] | None,
    retained_expiry_indices: tuple[int, ...] | None = None,
) -> _TerminalSolverEvidence:
    """Validate successful solver calls and the intentional hard-fit fallback."""

    if value.get("outcome") != "completed":
        _fail(f"{label} diagnostics does not declare a completed outcome")
    events = value.get("solver_events")
    if not isinstance(events, list) or not events:
        _fail(f"{label} diagnostics has no non-empty solver_events list")
    successful = {"optimal", "optimal_inaccurate"}
    hard_fallback = successful | {"infeasible", "infeasible_inaccurate"}

    if expected_expiry_indices is None:
        calls = _solver_calls(events, label)
        if len(calls) != 1 or calls[0].stage != "discrete_dlv_surface":
            _fail(f"{label} diagnostics does not contain one discrete-DLV solve")
        if calls[0].used_fallback or calls[0].terminal_status not in successful:
            _fail(f"{label} diagnostics discrete-DLV solve did not succeed directly")
        if retained_expiry_indices is not None:
            _fail(f"{label} diagnostics received unexpected retained expiry indices")
        return _TerminalSolverEvidence(
            relaxed_support=frozenset(),
            terminal_solver_call=calls[0],
        )

    if retained_expiry_indices is None:
        _fail(f"{label} diagnostics has no retained-expiry contract")

    predicted_retained: list[int] = []
    relaxed_support: set[int] = set()
    cursor = 0
    for input_index in expected_expiry_indices:
        decision_position = next(
            (
                position
                for position in range(cursor, len(events))
                if isinstance(events[position], Mapping)
                and events[position].get("stage") == "expiry_decision"
            ),
            None,
        )
        if decision_position is None:
            _fail(f"{label} diagnostics expiry {input_index} has no decision event")
        history = _solver_calls(events[cursor:decision_position], label)
        if any(call.input_expiry_index != input_index for call in history):
            _fail(f"{label} diagnostics expiry solve order/count changed")
        decision = events[decision_position]
        assert isinstance(decision, Mapping)
        if set(decision) != {
            "stage",
            "input_expiry_index",
            "retained",
            "support_attempt",
        }:
            _fail(f"{label} diagnostics expiry decision fields changed")
        decision_index = decision.get("input_expiry_index")
        retained = decision.get("retained")
        support_attempt = decision.get("support_attempt")
        if (
            isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or decision_index != input_index
            or not isinstance(retained, bool)
            or (
                support_attempt is not None
                and (
                    isinstance(support_attempt, bool)
                    or not isinstance(support_attempt, int)
                    or support_attempt not in {1, 2}
                )
            )
        ):
            _fail(f"{label} diagnostics expiry decision is malformed")
        cursor = decision_position + 1
        shape = [(call.stage, call.expiry_attempt) for call in history]
        accepted_shapes = {
            (("expiry_soft", 1),),
            (("expiry_soft", 1), ("expiry_hard", 1)),
            (("expiry_soft", 1), ("expiry_hard", 1), ("expiry_soft", 2)),
            (
                ("expiry_soft", 1),
                ("expiry_hard", 1),
                ("expiry_soft", 2),
                ("expiry_hard", 2),
            ),
        }
        if tuple(shape) not in accepted_shapes:
            _fail(f"{label} diagnostics expiry {input_index} has invalid solve ordering")
        for call in history:
            accepted = successful if call.stage == "expiry_soft" else hard_fallback
            if call.terminal_status not in accepted:
                _fail(
                    f"{label} diagnostics expiry "
                    f"{(input_index, call.expiry_attempt)} {call.stage} ends in "
                    f"unexpected state {call.terminal_status!r}"
                )
        if len(history) >= 3 and history[1].terminal_status not in {
            "infeasible",
            "infeasible_inaccurate",
        }:
            _fail(
                f"{label} diagnostics expiry {input_index} retries without "
                "a failed hard fit"
            )
        hard_failed = {"infeasible", "infeasible_inaccurate"}
        if len(history) == 1:
            valid_decision = retained and support_attempt == 1
        elif len(history) == 2 and history[1].terminal_status in successful:
            valid_decision = retained and support_attempt == 1
        elif len(history) == 2:
            valid_decision = (not retained and support_attempt is None) or (
                retained and support_attempt == 1
            )
        elif len(history) == 3:
            valid_decision = retained and support_attempt == 2
        elif history[-1].terminal_status in successful:
            valid_decision = retained and support_attempt == 2
        else:
            valid_decision = retained and support_attempt in {1, 2}
        if not valid_decision:
            _fail(
                f"{label} diagnostics expiry {input_index} decision conflicts "
                "with its solve history"
            )
        if retained:
            predicted_retained.append(input_index)
        if (
            len(history) == 4
            and history[-1].terminal_status in hard_failed
            and support_attempt == 1
        ):
            relaxed_support.add(input_index)
    surface_calls = _solver_calls(events[cursor:], label)
    if (
        len(surface_calls) != 1
        or surface_calls[0].stage != "surface"
        or surface_calls[0].terminal_status not in successful
    ):
        _fail(f"{label} diagnostics does not end in one successful surface solve")
    if tuple(predicted_retained) != retained_expiry_indices:
        _fail(
            f"{label} diagnostics retained-expiry population changed: expected "
            f"{tuple(predicted_retained)}, observed {retained_expiry_indices}"
        )
    return _TerminalSolverEvidence(
        relaxed_support=frozenset(relaxed_support),
        terminal_solver_call=surface_calls[0],
    )


@version("1.0")
def _expected_solver_expiry_indices(
    frame: pl.DataFrame,
    row_mask: np.ndarray[Any, Any],
    minimum_strikes: int,
) -> tuple[int, ...]:
    """Return the input expiry indices for which MkArbFree attempts a solve."""

    expiry_ids = frame.get_column("expiry_id").to_numpy()
    return tuple(
        index
        for index, expiry_id in enumerate(np.unique(expiry_ids[row_mask]))
        if int(np.sum(row_mask & (expiry_ids == expiry_id))) >= minimum_strikes
    )


@version("1.0")
def _max_abs(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


@version("1.0")
def _validate_calendar_residual(
    residual: float,
    *,
    tolerance: float,
    label: str,
) -> None:
    """Apply one inclusive finite calendar-feasibility boundary."""
    if (
        not np.isfinite(residual)
        or residual < 0.0
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        _fail(f"{label} has an invalid calendar residual or tolerance")
    if residual > tolerance:
        _fail(f"{label} normalized calls violate calendar monotonicity")


@version("1.2", dependencies=[_validate_calendar_residual])
def _validate_pricer_state(
    pricer: Any,
    label: str,
    *,
    check_dlvs: bool,
    calendar_tolerance: float = 1.0e-7,
) -> dict[str, float]:
    expiries = np.asarray(pricer.expiries, dtype=np.float64)
    variances = np.asarray(pricer.variances, dtype=np.float64)
    if (
        expiries.ndim != 1
        or variances.shape != expiries.shape
        or len(expiries) == 0
        or np.any(~np.isfinite(expiries))
        or np.any(~np.isfinite(variances))
        or np.any(expiries <= 0.0)
        or np.any(variances <= 0.0)
        or (len(expiries) > 1 and np.any(np.diff(expiries) <= 0.0))
        or (len(variances) > 1 and np.any(np.diff(variances) <= 0.0))
    ):
        _fail(f"{label} has invalid expiry/variance geometry")
    mass_residual = 0.0
    moment_residual = 0.0
    minimum_density = float("inf")
    dlvs = pricer.dlvsqs(on_all_strikes=True) if check_dlvs else [None] * len(expiries)
    for index, (strikes, density, dlv_value) in enumerate(
        zip(pricer.xstrikes, pricer.xdensities, dlvs, strict=True)
    ):
        strikes = np.asarray(strikes, dtype=np.float64)
        density = np.asarray(density, dtype=np.float64)
        dlv = None if dlv_value is None else np.asarray(dlv_value, dtype=np.float64)
        if (
            strikes.shape != density.shape
            or len(strikes) < 3
            or np.any(~np.isfinite(strikes))
            or np.any(~np.isfinite(density))
            or np.any(strikes <= 0.0)
            or np.any(np.diff(strikes) <= 0.0)
        ):
            _fail(f"{label} expiry index {index} has invalid support/state geometry")
        minimum_density = min(minimum_density, float(np.min(density)))
        mass_residual = max(mass_residual, abs(float(np.sum(density)) - 1.0))
        moment_residual = max(moment_residual, abs(float(strikes @ density) - 1.0))
        if dlv is not None and (
            dlv.shape != strikes.shape
            or np.any(~np.isfinite(dlv))
            or np.any(dlv[1:-1] < 0.0)
            or dlv[0] != 0.0
            or dlv[-1] != 0.0
        ):
            _fail(
                f"{label} expiry index {index} has invalid squared-DLV "
                "boundaries/interior"
            )
    if minimum_density < -1.0e-7:
        _fail(f"{label} minimum density is {minimum_density}")
    if mass_residual > 1.0e-4 or moment_residual > 1.0e-4:
        _fail(
            f"{label} density normalization failed: mass={mass_residual}, "
            f"first_moment={moment_residual}"
        )

    grid = np.linspace(0.25, 2.0, 101, dtype=np.float64)
    strike_shape_residual = 0.0
    calendar_residual = 0.0
    bounds_residual = 0.0
    previous_calls: np.ndarray[Any, Any] | None = None
    for expiry in expiries:
        calls = np.asarray(pricer(expiry, grid), dtype=np.float64)
        if calls.shape != grid.shape or np.any(~np.isfinite(calls)):
            _fail(f"{label} reconstructed non-finite normalized calls")
        strike_shape_residual = max(
            strike_shape_residual,
            float(max(np.max(np.diff(calls)), -np.min(np.diff(calls, n=2)), 0.0)),
        )
        lower = np.maximum(1.0 - grid, 0.0)
        bounds_residual = max(
            bounds_residual,
            float(np.max(np.maximum(lower - calls, 0.0))),
            float(np.max(np.maximum(calls - 1.0, 0.0))),
        )
        if previous_calls is not None:
            calendar_residual = max(
                calendar_residual,
                float(np.max(np.maximum(previous_calls - calls, 0.0))),
            )
        previous_calls = calls
    if strike_shape_residual > 1.0e-7:
        _fail(f"{label} normalized calls violate decreasing/convex strike shape")
    if bounds_residual > 1.0e-7:
        _fail(f"{label} normalized calls violate model-free bounds")
    _validate_calendar_residual(
        calendar_residual,
        tolerance=calendar_tolerance,
        label=label,
    )
    return {
        "bounds": bounds_residual,
        "calendar": calendar_residual,
        "density_mass": mass_residual,
        "density_minimum_violation": max(-minimum_density, 0.0),
        "density_moment": moment_residual,
        "strike_shape": strike_shape_residual,
    }


@version("2.0", dependencies=[SanosLPSolver, _SolverCall])
def _surface_calendar_acceptance_tolerance(
    terminal_solver_call: _SolverCall,
    label: str,
) -> float:
    """Return the feasibility tolerance of the accepted surface solve."""

    if terminal_solver_call.stage != "surface" or terminal_solver_call.terminal_status not in {
        "optimal",
        "optimal_inaccurate",
    }:
        _fail(f"{label} diagnostics has no accepted surface solve")
    _, settings = SanosLPSolver()._lp_solver_settings(
        for_expiry=False,
        is_fallback=terminal_solver_call.used_fallback,
    )
    if not terminal_solver_call.used_fallback:
        return float(settings["highs_options"]["kkt_tolerance"])
    key = (
        "tol_feas"
        if terminal_solver_call.terminal_status == "optimal"
        else "reduced_tol_feas"
    )
    return float(settings[key])


@version(
    "producer-float32-count-float64-overnight-final-float32-v1",
    dependencies=[count_business_days_between],
)
def _business_dte_replay(
    *,
    trading_date: date,
    expiry_dates: np.ndarray[Any, Any],
    expiry_is_am: np.ndarray[Any, Any],
    am_business_day_subtraction: float,
) -> np.ndarray[Any, Any]:
    """Replay the daily producer's dtype-sensitive AM/PM business clock."""

    integer_business_dte = np.asarray(
        count_business_days_between(trading_date, expiry_dates),
        dtype=np.float32,
    )
    return np.clip(
        integer_business_dte
        - np.where(expiry_is_am, am_business_day_subtraction, 0.0),
        a_min=0.0,
        a_max=None,
    ).astype(np.float32)


@version("1.0")
def _validate_density_geometry(
    strikes: np.ndarray[Any, Any],
    density: np.ndarray[Any, Any],
    label: str,
) -> tuple[float, float, float]:
    if (
        strikes.ndim != 1
        or density.shape != strikes.shape
        or len(strikes) < 3
        or np.any(~np.isfinite(strikes))
        or np.any(~np.isfinite(density))
        or np.any(strikes <= 0.0)
        or np.any(np.diff(strikes) <= 0.0)
    ):
        _fail(f"{label} has invalid support/density geometry")
    minimum = float(np.min(density))
    mass = abs(float(np.sum(density)) - 1.0)
    moment = abs(float(strikes @ density) - 1.0)
    if minimum < -1.0e-7 or mass > 1.0e-4 or moment > 1.0e-4:
        _fail(
            f"{label} density normalization failed: minimum={minimum}, "
            f"mass={mass}, first_moment={moment}"
        )
    return minimum, mass, moment


@version("1.0", dependencies=[SanosPricer, _validate_density_geometry, _max_abs])
def _expiry_state_pricer(
    frame: pl.DataFrame,
    *,
    index_column: str,
    expiry_strikes_column: str,
    expiry_density_column: str,
    surface_strikes_column: str,
    surface_density_column: str,
    mapping_column: str,
    variance_column: str,
    expected_source_expiry_ids: tuple[int, ...],
    smoothness: float | None,
    label: str,
) -> tuple[SanosPricer, tuple[int, ...], dict[str, float]]:
    state = frame.filter(pl.col(index_column).is_not_null()).sort(index_column)
    if state.is_empty():
        _fail(f"{label} has no retained model state")
    indices = tuple(
        int(value) for value in state.get_column(index_column).unique(maintain_order=True)
    )
    if indices != tuple(sorted(set(indices))) or any(
        value < 0 or value >= len(expected_source_expiry_ids) for value in indices
    ):
        _fail(f"{label} has invalid input-expiry indices: {indices}")

    expiries: list[float] = []
    variances: list[float] = []
    expiry_strikes: list[np.ndarray[Any, Any]] = []
    expiry_densities: list[np.ndarray[Any, Any]] = []
    maxima = {
        "density_mass": 0.0,
        "density_minimum_violation": 0.0,
        "density_moment": 0.0,
        "mapping": 0.0,
        "surface_density_mass": 0.0,
        "surface_density_minimum_violation": 0.0,
        "surface_density_moment": 0.0,
    }
    list_columns = (
        expiry_strikes_column,
        expiry_density_column,
        surface_strikes_column,
        surface_density_column,
        mapping_column,
    )
    for input_index in indices:
        group = state.filter(pl.col(index_column) == input_index)
        expiry_ids = group.get_column("expiry_id").unique().to_list()
        expected_expiry_id = expected_source_expiry_ids[input_index]
        if expiry_ids != [expected_expiry_id]:
            _fail(
                f"{label} input index {input_index} maps to {expiry_ids}, "
                f"expected expiry_id={expected_expiry_id}"
            )
        repeated: dict[str, Any] = {}
        for column in list_columns:
            values = group.get_column(column).to_list()
            if not values or any(value != values[0] for value in values[1:]):
                _fail(f"{label} repeated {column} state is null or inconsistent")
            if column != mapping_column and values[0] is None:
                _fail(f"{label} repeated {column} state is null or inconsistent")
            repeated[column] = values[0]
        expiry_values = group.get_column("time_to_expiry_years").unique().to_list()
        variance_values = group.get_column(variance_column).unique().to_list()
        if len(expiry_values) != 1 or len(variance_values) != 1:
            _fail(f"{label} expiry/variance state is inconsistent")

        xstrikes = np.asarray(repeated[expiry_strikes_column], dtype=np.float64)
        xdensity = np.asarray(repeated[expiry_density_column], dtype=np.float64)
        mstrikes = np.asarray(repeated[surface_strikes_column], dtype=np.float64)
        mdensity = np.asarray(repeated[surface_density_column], dtype=np.float64)
        mapping_value = repeated[mapping_column]
        mapping = (
            np.arange(len(xstrikes), dtype=np.int64)
            if mapping_value is None
            else np.asarray(mapping_value, dtype=np.int64)
        )
        minimum, mass, moment = _validate_density_geometry(
            xstrikes, xdensity, f"{label} expiry index {input_index} expiry fit"
        )
        surface_minimum, surface_mass, surface_moment = _validate_density_geometry(
            mstrikes, mdensity, f"{label} expiry index {input_index} surface fit"
        )
        if (
            mapping.ndim != 1
            or mapping.shape != xstrikes.shape
            or np.any(mapping < 0)
            or np.any(mapping >= len(mstrikes))
            or (len(mapping) > 1 and np.any(np.diff(mapping) <= 0))
        ):
            _fail(f"{label} expiry index {input_index} has an invalid x-to-surface mapping")
        mapping_error = _max_abs(mstrikes[mapping], xstrikes)
        if mapping_error > 1.0e-12:
            _fail(
                f"{label} expiry index {input_index} x-to-surface mapping residual "
                f"is {mapping_error}"
            )
        maxima["density_mass"] = max(maxima["density_mass"], mass)
        maxima["density_minimum_violation"] = max(
            maxima["density_minimum_violation"], max(-minimum, 0.0)
        )
        maxima["density_moment"] = max(maxima["density_moment"], moment)
        maxima["mapping"] = max(maxima["mapping"], mapping_error)
        maxima["surface_density_mass"] = max(maxima["surface_density_mass"], surface_mass)
        maxima["surface_density_minimum_violation"] = max(
            maxima["surface_density_minimum_violation"], max(-surface_minimum, 0.0)
        )
        maxima["surface_density_moment"] = max(
            maxima["surface_density_moment"], surface_moment
        )
        expiries.append(float(expiry_values[0]))
        variances.append(float(variance_values[0]))
        expiry_strikes.append(xstrikes)
        expiry_densities.append(xdensity)
    try:
        pricer = SanosPricer(
            smoothness=smoothness,
            expiries=np.asarray(expiries, dtype=np.float64),
            variances=np.asarray(variances, dtype=np.float64),
            xstrikes=expiry_strikes,
            xdensities=expiry_densities,
            min_half_spread=float(frame.get_column("surface_min_half_spread").item(0)),
        )
    except Exception as exc:
        raise DatasetValidationError(f"{label} expiry state is invalid: {exc}") from exc
    return pricer, indices, maxima


@version("2.0")
def _validate_support_mask_geometry(
    frame: pl.DataFrame,
    *,
    mask_column: str,
    state_index_column: str,
    expiry_strikes_column: str,
    allow_masked_support_superset: frozenset[int],
    label: str,
) -> None:
    """Tie a stored fitter mask to its retained expiry support and solve path."""

    mask = frame.get_column(mask_column).to_numpy().astype(bool)
    state_present = frame.get_column(state_index_column).is_not_null().to_numpy()
    if np.any(mask & ~state_present):
        _fail(f"{label} contains masked rows without retained model state")
    for group in frame.partition_by("expiry_id", maintain_order=True):
        local_present = group.get_column(state_index_column).is_not_null().to_numpy()
        if not np.any(local_present):
            continue
        state_rows = group.filter(pl.Series("__state", local_present))
        supports = state_rows.get_column(expiry_strikes_column).to_list()
        if not supports or supports[0] is None or any(
            value != supports[0] for value in supports[1:]
        ):
            _fail(f"{label} has inconsistent retained expiry support")
        support = np.asarray(supports[0], dtype=np.float64)
        state_indices = state_rows.get_column(state_index_column).unique().to_list()
        if len(state_indices) != 1 or state_indices[0] is None:
            _fail(f"{label} has inconsistent retained input-expiry indices")
        state_index = int(state_indices[0])
        local_mask = state_rows.get_column(mask_column).to_numpy().astype(bool)
        masked_strikes = (
            state_rows.get_column("normalized_strike")
            .filter(pl.Series("__mask", local_mask))
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        if support.ndim != 1 or len(support) < 3:
            _fail(f"{label} has invalid retained expiry support")
        interior = support[1:-1]
        if state_index not in allow_masked_support_superset:
            if not np.array_equal(interior, masked_strikes):
                _fail(f"{label} differs from the exact retained expiry support")
            continue
        positions = np.searchsorted(interior, masked_strikes)
        is_strict_ordered_subset = (
            len(masked_strikes) < len(interior)
            and len(masked_strikes) > 0
            and np.all(positions < len(interior))
            and np.array_equal(interior[positions], masked_strikes)
        )
        if not is_strict_ordered_subset:
            _fail(
                f"{label} must be a strict ordered subset after a failed second "
                "hard fit"
            )


@version("1.0")
@dataclass(frozen=True, slots=True)
class _BlackVectorResiduals:
    """Independent price and Greek residuals for one persisted Black vector."""

    price_repricing: float
    price_rounding_excess: float
    greek_center: float
    greek_rounding_excess: float
    volatility_bound_excess: float

    @property
    def center_residual(self) -> float:
        return max(self.price_repricing, self.greek_center)

    @property
    def maximum_excess(self) -> float:
        return max(
            self.price_rounding_excess,
            self.greek_rounding_excess,
            self.volatility_bound_excess,
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class _BlackSourceVectorResiduals:
    """Residuals from replaying a fitted source through its persisted vector."""

    source_price: float
    source_contract_excess: float
    float32_max_abs: float
    mismatch_count: int
    mismatch_fields: tuple[str, ...]


@version("1.0")
@dataclass(frozen=True, slots=True)
class _SmoothProducerReplay:
    """Exact producer observables recovered from serialized smooth state."""

    retained_input_indices: tuple[int, ...]
    expiry_fits: tuple[np.ndarray[Any, Any], ...]
    surface_fits: tuple[np.ndarray[Any, Any], ...]
    market_fits: tuple[np.ndarray[Any, Any], ...]
    market_fit_errors: tuple[np.ndarray[Any, Any], ...]
    pcnt_fit: float
    median_error: float
    mean_error: float
    median_total_error: float
    mean_total_error: float


@version("1.0")
def _exact_float32_state(values: Any, label: str) -> np.ndarray[Any, Any]:
    """Recover producer Float32 state only when serialization was lossless."""

    values64 = np.asarray(values, dtype=np.float64)
    values32 = values64.astype(np.float32)
    if not np.array_equal(values64, values32.astype(np.float64)):
        _fail(f"{label} is not a lossless serialization of producer Float32 state")
    return values32


@version("cdxcore-bs-price-kernel-2026-10-v1")
def _smooth_black_price(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
    """Version-owned adapter for the copied smooth producer's Black kernel."""

    return np.asarray(bs.price(*args, **kwargs))


@version("sanos-smooth-black-price-epsilon-1e-6-v1")
def _smooth_black_price_epsilon() -> float:
    """Return the copied smooth producer's Black price epsilon."""

    return 1.0e-6


@version(
    "1.0",
    dependencies=[
        _smooth_black_price,
        _smooth_black_price_epsilon,
        one_minute_year_fraction,
    ],
)
def _smooth_call_matrix(
    *,
    expiry: float | np.floating[Any],
    strikes: np.ndarray[Any, Any],
    support: np.ndarray[Any, Any],
    volatility: float | np.floating[Any],
) -> np.ndarray[Any, Any]:
    """Replay the copied fitter's dtype-sensitive smooth Black kernel."""

    if float(expiry) < one_minute_year_fraction() or float(volatility) == 0.0:
        return np.maximum(
            np.float32(0.0),
            support[None, :] - strikes[:, None],
        )
    return (
        _smooth_black_price(
            k=strikes[:, None] / support[None, :],
            vol=volatility,
            sqrtT=math.sqrt(expiry),
            eps=_smooth_black_price_epsilon(),
        )
        * support[None, :]
    )


@version(
    "1.0",
    dependencies=[_smooth_call_matrix, one_minute_year_fraction],
)
def _smooth_market_fit_prices(
    *,
    expiry: float | np.floating[Any],
    strikes32: np.ndarray[Any, Any],
    smoothness: float,
    expiries64: np.ndarray[Any, Any],
    variances64: np.ndarray[Any, Any],
    surface_supports: list[np.ndarray[Any, Any]],
    surface_densities: list[np.ndarray[Any, Any]],
) -> np.ndarray[Any, Any]:
    """Replay copied ``ArbFreePricer.price(..., mode='surface')`` semantics."""

    if float(expiry) < one_minute_year_fraction():
        return np.maximum(np.float32(0.0), np.float32(1.0) - strikes32)

    clipped = np.float64(np.clip(expiry, expiries64[0], expiries64[-1]))
    variance = np.float64(np.interp(clipped, expiries64, variances64))
    market_volatility = np.sqrt(variance / clipped) * math.sqrt(smoothness)

    def at_state(index: int) -> np.ndarray[Any, Any]:
        return (
            _smooth_call_matrix(
                expiry=expiry,
                strikes=strikes32,
                support=surface_supports[index],
                volatility=market_volatility,
            )
            @ surface_densities[index]
        )

    insertion = int(np.searchsorted(expiries64, expiry))
    if insertion == 0:
        return at_state(0)
    if insertion == len(expiries64):
        return at_state(len(expiries64) - 1)
    left_distance = abs(expiry - expiries64[insertion - 1])
    right_distance = abs(expiry - expiries64[insertion])
    if min(left_distance, right_distance) < one_minute_year_fraction():
        return at_state(insertion if right_distance < left_distance else insertion - 1)
    left = at_state(insertion - 1)
    right = at_state(insertion)
    return (
        (expiry - expiries64[insertion - 1]) * right
        + (expiries64[insertion] - expiry) * left
    ) / (expiries64[insertion] - expiries64[insertion - 1])


@version("1.0")
def _smooth_canonical_market_fit_bridge_tolerance() -> float:
    """Bound the copied listed-price path against canonical Float64 SANOS."""

    return 5.0e-8


@version("1.0", dependencies=[_smooth_canonical_market_fit_bridge_tolerance])
def _validate_smooth_canonical_market_fit_bridge(
    residual: float,
    *,
    label: str,
) -> None:
    """Reject a canonical consumer that diverges from the producer market fit."""

    if residual > _smooth_canonical_market_fit_bridge_tolerance():
        _fail(
            f"{label}: smooth canonical market-fit bridge residual is {residual}"
        )


@version("initial-expiry-quote-containment-1e-6-v1")
def _initial_expiry_quote_containment_tolerance() -> float:
    """Return the copied per-expiry fitter's absolute quote-band tolerance."""

    return 1.0e-6


@version("1.0")
def _masked_quote_band_residual(
    *,
    bids: np.ndarray[Any, Any],
    asks: np.ndarray[Any, Any],
    fits: np.ndarray[Any, Any],
    mask: np.ndarray[Any, Any],
) -> float:
    """Return the maximum absolute quote-band violation on retained rows."""

    bid_values = np.asarray(bids)
    ask_values = np.asarray(asks)
    fit_values = np.asarray(fits)
    mask_values = np.asarray(mask, dtype=bool)
    if not (
        bid_values.shape
        == ask_values.shape
        == fit_values.shape
        == mask_values.shape
    ):
        _fail("initial quote-band residual inputs have inconsistent shapes")
    if not np.any(mask_values):
        return 0.0
    below = np.maximum(bid_values - fit_values, 0.0)
    above = np.maximum(fit_values - ask_values, 0.0)
    return float(np.max(np.maximum(below[mask_values], above[mask_values])))


@version("1.1", dependencies=[_initial_expiry_quote_containment_tolerance])
def _validate_initial_fit_quote_contract(
    *,
    expiry_fit_residual: float,
    surface_fit_residual: float,
    label: str,
) -> None:
    """Enforce quote containment only for the copied accepted per-expiry fit.

    ``MkArbFree.fits1`` is accepted only inside the quote band up to its
    numerical tolerance. ``fitsX`` is a globally arbitrage-free soft closest
    fit and is explicitly allowed to leave the quote band; its residual remains
    a validation diagnostic rather than an acceptance criterion.
    """

    for name, residual in (
        ("expiry-fit", expiry_fit_residual),
        ("surface-fit", surface_fit_residual),
    ):
        if not np.isfinite(residual) or residual < 0.0:
            _fail(f"{label}: initial {name} quote-band residual is invalid")
    if expiry_fit_residual > _initial_expiry_quote_containment_tolerance():
        _fail(
            f"{label}: initial expiry-fit quote containment residual is "
            f"{expiry_fit_residual}"
        )


@version(
    "1.0",
    dependencies=[
        _SmoothProducerReplay,
        _exact_float32_state,
        _smooth_call_matrix,
        _smooth_market_fit_prices,
    ],
)
def _smooth_producer_replay(
    frame: pl.DataFrame,
    *,
    selected_groups: list[pl.DataFrame],
    smoothness: float,
    min_half_spread: float,
) -> _SmoothProducerReplay:
    """Replay fits1, fitsX, and market-fit output under their producer contracts.

    The copied fitter deliberately receives Float32 market inputs.  Its expiry
    and common-surface supports remain Float32 in memory but are serialized in
    Float64 list columns without loss.  The solver-time ``fitsX`` kernel and
    the later ``compute_market_fit`` kernel also use algebraically equivalent,
    numerically distinct volatility expressions.  Replaying each observable
    separately prevents the canonical Float64 :class:`SanosPricer` contract
    from being conflated with either producer audit field.
    """

    state = frame.filter(pl.col("smooth_input_expiry_index").is_not_null()).sort(
        "smooth_input_expiry_index"
    )
    retained_indices = tuple(
        int(value)
        for value in state.get_column("smooth_input_expiry_index")
        .unique(maintain_order=True)
        .to_list()
    )
    if not retained_indices:
        _fail("smooth producer replay requires retained model state")

    retained_expiries: list[float] = []
    retained_variances: list[float] = []
    expiry_supports: list[np.ndarray[Any, Any]] = []
    expiry_densities: list[np.ndarray[Any, Any]] = []
    surface_supports: list[np.ndarray[Any, Any]] = []
    surface_densities: list[np.ndarray[Any, Any]] = []
    for input_index in retained_indices:
        group = state.filter(pl.col("smooth_input_expiry_index") == input_index)
        expiry_values = group.get_column("time_to_expiry_years").unique().to_list()
        variance_values = group.get_column("smooth_model_variance").unique().to_list()
        if len(expiry_values) != 1 or len(variance_values) != 1:
            _fail(f"smooth producer state {input_index} has inconsistent time/variance")
        retained_expiries.append(float(np.float32(expiry_values[0])))
        retained_variances.append(float(variance_values[0]))
        for column, target, dtype in (
            ("smooth_expiry_model_strikes", expiry_supports, "support"),
            ("smooth_surface_model_strikes", surface_supports, "support"),
        ):
            values = group.get_column(column).to_list()
            if (
                not values
                or values[0] is None
                or any(value != values[0] for value in values[1:])
            ):
                _fail(f"smooth producer state {input_index} has inconsistent {column}")
            target.append(
                _exact_float32_state(
                    values[0],
                    f"smooth producer state {input_index} {dtype} {column}",
                )
            )
        for column, target in (
            ("smooth_expiry_model_density", expiry_densities),
            ("smooth_surface_model_density", surface_densities),
        ):
            values = group.get_column(column).to_list()
            if (
                not values
                or values[0] is None
                or any(value != values[0] for value in values[1:])
            ):
                _fail(f"smooth producer state {input_index} has inconsistent {column}")
            target.append(np.asarray(values[0], dtype=np.float64))

    expiries64 = np.asarray(retained_expiries, dtype=np.float64)
    variances64 = np.asarray(retained_variances, dtype=np.float64)

    def interpolated_variance(
        expiry: float | np.floating[Any],
    ) -> tuple[np.float64, np.float64]:
        clipped = np.float64(np.clip(expiry, expiries64[0], expiries64[-1]))
        return clipped, np.float64(np.interp(clipped, expiries64, variances64))

    expiry_fits: list[np.ndarray[Any, Any]] = []
    surface_fits: list[np.ndarray[Any, Any]] = []
    for model_index, input_index in enumerate(retained_indices):
        group = selected_groups[input_index]
        strikes32 = group.get_column("normalized_strike").to_numpy().astype(
            np.float32, copy=False
        )
        expiry = expiries64[model_index]
        expiry_fits.append(
            np.maximum(
                np.float32(0.0),
                expiry_supports[model_index][None, :] - strikes32[:, None],
            )
            @ expiry_densities[model_index]
        )
        clipped, variance = interpolated_variance(expiry)
        surface_volatility = math.sqrt(smoothness * variance / clipped)
        surface_fits.append(
            _smooth_call_matrix(
                expiry=expiry,
                strikes=strikes32,
                support=surface_supports[model_index],
                volatility=surface_volatility,
            )
            @ surface_densities[model_index]
        )

    market_fits: list[np.ndarray[Any, Any]] = []
    market_fit_errors: list[np.ndarray[Any, Any]] = []
    for group in selected_groups:
        expiry = np.float32(group.get_column("time_to_expiry_years").item(0))
        strikes32 = group.get_column("normalized_strike").to_numpy().astype(
            np.float32, copy=False
        )
        bids32 = group.get_column("pure_bid_price").to_numpy().astype(
            np.float32, copy=False
        )
        asks32 = group.get_column("pure_ask_price").to_numpy().astype(
            np.float32, copy=False
        )
        fitted = _smooth_market_fit_prices(
            expiry=expiry,
            strikes32=strikes32,
            smoothness=smoothness,
            expiries64=expiries64,
            variances64=variances64,
            surface_supports=surface_supports,
            surface_densities=surface_densities,
        )
        half_spreads = np.maximum(
            (asks32 - bids32) / np.float32(2.0),
            min_half_spread,
        )
        errors = (
            np.maximum(0.0, fitted - asks32)
            + np.maximum(0.0, bids32 - fitted)
        ) / half_spreads
        market_fits.append(fitted)
        market_fit_errors.append(errors)

    all_errors = np.concatenate(market_fit_errors)
    positive = all_errors > 0.0
    return _SmoothProducerReplay(
        retained_input_indices=retained_indices,
        expiry_fits=tuple(expiry_fits),
        surface_fits=tuple(surface_fits),
        market_fits=tuple(market_fits),
        market_fit_errors=tuple(market_fit_errors),
        pcnt_fit=float(np.sum(positive)) / float(len(all_errors)),
        median_error=float(np.median(all_errors[positive])) if np.any(positive) else 0.0,
        mean_error=float(np.mean(all_errors[positive])) if np.any(positive) else 0.0,
        median_total_error=float(np.median(all_errors)),
        mean_total_error=float(np.mean(all_errors)),
    )


@version("1.0")
def _float32_ulp(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a conservative one-ULP radius around finite Float32 values."""

    center = np.asarray(values, dtype=np.float32)
    lower = np.nextafter(center, np.float32(-np.inf), dtype=np.float32)
    upper = np.nextafter(center, np.float32(np.inf), dtype=np.float32)
    center64 = center.astype(np.float64)
    return np.maximum(
        center64 - lower.astype(np.float64),
        upper.astype(np.float64) - center64,
    )


@version(
    "1.0",
    dependencies=[_float32_ulp, _implied_volatility_price_tolerance],
)
def _black_source_price_residual(
    source_prices: np.ndarray[Any, Any],
    persisted_prices: np.ndarray[Any, Any],
) -> tuple[float, float]:
    """Measure source-to-Black-round-trip error against its exact contract.

    The producer first converts each fitted price to implied volatility with
    cdxcore's configured price tolerance, reprices it, and then persists the
    result as Float32.  One full Float32 ULP is a conservative serialization
    allowance around the stored value.
    """

    source = np.asarray(source_prices, dtype=np.float64)
    persisted32 = np.asarray(persisted_prices, dtype=np.float32)
    persisted = persisted32.astype(np.float64)
    if source.shape != persisted.shape:
        _fail(
            "source and persisted Black prices have inconsistent shapes: "
            f"{source.shape} != {persisted.shape}"
        )
    error = np.abs(source - persisted)
    allowance = _implied_volatility_price_tolerance() + _float32_ulp(persisted32)
    return (
        float(np.max(error, initial=0.0)),
        float(np.max(np.maximum(error - allowance, 0.0), initial=0.0)),
    )


@version(
    "1.0",
    dependencies=[
        _BlackSourceVectorResiduals,
        _black_implied,
        _black_source_price_residual,
        _black_values,
        _numerical_tolerances,
    ],
)
def _black_source_vector_residual(
    frame: pl.DataFrame,
    *,
    source_prices: np.ndarray[Any, Any],
    price_column: str,
    volatility_column: str,
    greek_prefix: str,
) -> _BlackSourceVectorResiduals:
    """Replay the deterministic source-price-to-Float32 Black vector contract."""

    source = np.asarray(source_prices, dtype=np.float64)
    if source.shape != (frame.height,):
        _fail(
            "source Black prices have the wrong shape for deterministic replay: "
            f"{source.shape} != {(frame.height,)}"
        )
    times = (
        frame.get_column("time_to_expiry_years")
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    if not len(times) or np.any(times != times[0]) or times[0] <= 0.0:
        _fail("deterministic Black replay requires one positive expiry")
    strikes = (
        frame.get_column("normalized_strike")
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    implied_epsilon, greek_epsilon = _numerical_tolerances()
    volatility = _black_implied(
        k=strikes,
        prices=source,
        sqrtT=math.sqrt(float(times[0])),
        eps=implied_epsilon,
        on_exceed_bounds="quiet",
    )
    values = _black_values(
        k=strikes,
        vol=volatility,
        sqrtT=math.sqrt(float(times[0])),
        eps=greek_epsilon,
        what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
    )
    replayed = {
        volatility_column: volatility,
        price_column: values.price,
        **{
            f"{greek_prefix}_{name}": getattr(values, name)
            for name in ("delta", "gamma", "vega", "theta", "dk")
        },
    }
    mismatch_count = 0
    mismatch_fields: list[str] = []
    maximum = 0.0
    for column, calculated in replayed.items():
        observed_series = frame.get_column(column)
        if observed_series.dtype != pl.Float32 or observed_series.null_count():
            _fail(f"{column} is not a complete Float32 Black replay vector")
        observed = observed_series.to_numpy().astype(np.float32, copy=False)
        expected = np.asarray(calculated, dtype=np.float64).astype(np.float32)
        if np.any(~np.isfinite(expected)):
            _fail(f"deterministic Black replay produced non-finite {column}")
        mismatched = observed.view(np.uint32) != expected.view(np.uint32)
        count = int(np.sum(mismatched))
        if count:
            mismatch_count += count
            mismatch_fields.append(column)
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            observed[mismatched].astype(np.float64)
                            - expected[mismatched].astype(np.float64)
                        )
                    )
                ),
            )
    source_price, source_contract_excess = _black_source_price_residual(
        source,
        frame.get_column(price_column).to_numpy(),
    )
    return _BlackSourceVectorResiduals(
        source_price=source_price,
        source_contract_excess=source_contract_excess,
        float32_max_abs=maximum,
        mismatch_count=mismatch_count,
        mismatch_fields=tuple(mismatch_fields),
    )


@version("1.0", dependencies=[_black_values])
def _black_value_envelope(
    *,
    strikes32: np.ndarray[Any, Any],
    volatilities32: np.ndarray[Any, Any],
    sqrt_time: np.ndarray[Any, Any],
    strike_was_persisted_input: bool,
    volatility_minimum: float,
    volatility_maximum: float,
    epsilon: float,
) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    """Return conservative Black extrema over the persisted input cells.

    The input interval deliberately spans the adjacent Float32 values, which is
    wider than the true round-to-nearest cell. Candidate points include the
    analytic stationary points of d1, d2, gamma, and theta on every boundary;
    they therefore cover the non-monotone Greek extrema, not just corners.
    """

    strike_center = strikes32.astype(np.float64)
    if strike_was_persisted_input:
        strike_lower = strike_center
        strike_upper = strike_center
    else:
        strike_lower = np.nextafter(
            strikes32, np.float32(-np.inf), dtype=np.float32
        ).astype(np.float64)
        strike_upper = np.nextafter(
            strikes32, np.float32(np.inf), dtype=np.float32
        ).astype(np.float64)
    volatility_lower = np.clip(
        np.nextafter(
            volatilities32, np.float32(-np.inf), dtype=np.float32
        ).astype(np.float64),
        volatility_minimum,
        volatility_maximum,
    )
    volatility_upper = np.clip(
        np.nextafter(
            volatilities32, np.float32(np.inf), dtype=np.float32
        ).astype(np.float64),
        volatility_minimum,
        volatility_maximum,
    )
    log_strike_lower = np.log(strike_lower)
    log_strike_upper = np.log(strike_upper)
    total_volatility_lower = volatility_lower * sqrt_time
    total_volatility_upper = volatility_upper * sqrt_time
    names = ("price", "delta", "gamma", "vega", "theta", "dk")
    candidate_values: dict[str, list[np.ndarray[Any, Any]]] = {
        name: [] for name in names
    }

    def append_candidate(
        strike: np.ndarray[Any, Any],
        volatility: np.ndarray[Any, Any],
    ) -> None:
        values = _black_values(
            k=strike,
            vol=volatility,
            sqrtT=sqrt_time,
            eps=epsilon,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        for name in names:
            candidate_values[name].append(
                np.asarray(getattr(values, name), dtype=np.float64)
            )

    for strike in (strike_lower, strike_upper):
        for volatility in (volatility_lower, volatility_upper):
            append_candidate(strike, volatility)

    for strike, log_strike in (
        (strike_lower, log_strike_lower),
        (strike_upper, log_strike_upper),
    ):
        stationary_total_volatilities = (
            np.sqrt(np.maximum(-2.0 * log_strike, 0.0)),
            np.sqrt(np.maximum(2.0 * log_strike, 0.0)),
            np.sqrt(2.0 * (np.sqrt(1.0 + np.square(log_strike)) - 1.0)),
            np.sqrt(2.0 * (1.0 + np.sqrt(1.0 + np.square(log_strike)))),
        )
        for stationary in stationary_total_volatilities:
            clipped = np.clip(
                stationary,
                total_volatility_lower,
                total_volatility_upper,
            )
            volatility = np.where(
                stationary <= total_volatility_lower,
                volatility_lower,
                np.where(
                    stationary >= total_volatility_upper,
                    volatility_upper,
                    clipped / sqrt_time,
                ),
            )
            append_candidate(strike, volatility)

    for total_volatility, volatility in (
        (total_volatility_lower, volatility_lower),
        (total_volatility_upper, volatility_upper),
    ):
        stationary_log_strike = 0.5 * np.square(total_volatility)
        clipped = np.clip(
            stationary_log_strike,
            log_strike_lower,
            log_strike_upper,
        )
        strike = np.where(
            stationary_log_strike <= log_strike_lower,
            strike_lower,
            np.where(
                stationary_log_strike >= log_strike_upper,
                strike_upper,
                np.exp(clipped),
            ),
        )
        append_candidate(strike, volatility)

    return {
        name: (
            np.min(candidate_values[name], axis=0),
            np.max(candidate_values[name], axis=0),
        )
        for name in names
    }


@version("1.0", dependencies=[_BlackVectorResiduals])
def _merge_black_vector_residuals(
    *items: _BlackVectorResiduals,
) -> _BlackVectorResiduals:
    """Take componentwise maxima across independently validated Black vectors."""

    if not items:
        return _BlackVectorResiduals(0.0, 0.0, 0.0, 0.0, 0.0)
    fields = (
        "price_repricing",
        "price_rounding_excess",
        "greek_center",
        "greek_rounding_excess",
        "volatility_bound_excess",
    )
    return _BlackVectorResiduals(
        *(max(getattr(item, field) for item in items) for field in fields)
    )


@version(
    "2.0",
    dependencies=[
        _BlackVectorResiduals,
        _black_value_envelope,
        _black_values,
        _float32_ulp,
        _numerical_tolerances,
    ],
)
def _black_vector_residual(
    frame: pl.DataFrame,
    *,
    price_column: str,
    volatility_column: str,
    greek_prefix: str,
    strike_was_persisted_input: bool,
    volatility_minimum: float,
    volatility_maximum: float,
) -> _BlackVectorResiduals:
    """Validate a persisted Black vector against its Float32 production contract.

    Implied volatility is deliberately not re-inverted. Deep in-the-money prices
    can identify volatility very poorly even when repricing is well inside the
    producer's price tolerance. Greeks are instead checked over every possible
    source input represented by the persisted Float32 rounding cells.
    """

    _, greek_epsilon = _numerical_tolerances()
    strikes32 = (
        frame.get_column("normalized_strike")
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    prices32 = frame.get_column(price_column).to_numpy().astype(np.float32, copy=False)
    volatilities32 = (
        frame.get_column(volatility_column).to_numpy().astype(np.float32, copy=False)
    )
    sqrt_time = np.sqrt(
        frame.get_column("time_to_expiry_years").to_numpy().astype(np.float64)
    )
    strikes = strikes32.astype(np.float64)
    volatilities = volatilities32.astype(np.float64)
    center = _black_values(
        k=strikes,
        vol=volatilities,
        sqrtT=sqrt_time,
        eps=greek_epsilon,
        what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
    )

    names = ("price", "delta", "gamma", "vega", "theta", "dk")
    envelope = _black_value_envelope(
        strikes32=strikes32,
        volatilities32=volatilities32,
        sqrt_time=sqrt_time,
        strike_was_persisted_input=strike_was_persisted_input,
        volatility_minimum=volatility_minimum,
        volatility_maximum=volatility_maximum,
        epsilon=greek_epsilon,
    )
    price_lower, price_upper = envelope["price"]
    observed_price = prices32.astype(np.float64)
    price_repricing = _max_abs(
        observed_price,
        np.asarray(center.price, dtype=np.float64),
    )
    price_allowance = _float32_ulp(prices32)
    price_rounding_excess = float(
        np.max(
            np.maximum.reduce(
                (
                    price_lower - observed_price - price_allowance,
                    observed_price - price_upper - price_allowance,
                    np.zeros_like(observed_price),
                )
            )
        )
    )

    greek_center = 0.0
    greek_rounding_excess = 0.0
    for name in names[1:]:
        observed32 = (
            frame.get_column(f"{greek_prefix}_{name}")
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        observed = observed32.astype(np.float64)
        greek_center = max(
            greek_center,
            _max_abs(observed, np.asarray(getattr(center, name), dtype=np.float64)),
        )
        lower, upper = envelope[name]
        allowance = _float32_ulp(observed32)
        greek_rounding_excess = max(
            greek_rounding_excess,
            float(
                np.max(
                    np.maximum.reduce(
                        (
                            lower - observed - allowance,
                            observed - upper - allowance,
                            np.zeros_like(observed),
                        )
                    )
                )
            ),
        )

    minimum32 = np.float32(volatility_minimum).astype(np.float64)
    maximum32 = np.float32(volatility_maximum).astype(np.float64)
    volatility_bound_excess = float(
        np.max(
            np.maximum.reduce(
                (
                    minimum32 - volatilities,
                    volatilities - maximum32,
                    np.zeros_like(volatilities),
                )
            )
        )
    )
    return _BlackVectorResiduals(
        price_repricing=price_repricing,
        price_rounding_excess=price_rounding_excess,
        greek_center=greek_center,
        greek_rounding_excess=greek_rounding_excess,
        volatility_bound_excess=volatility_bound_excess,
    )


@version("1.0")
def _validate_fitted_numeric_finiteness(frame: pl.DataFrame) -> None:
    """Reject non-finite values anywhere a fitted numeric value is present."""

    scalar_float_types = (pl.Float32, pl.Float64)
    list_float_types = (pl.List(pl.Float32), pl.List(pl.Float64))
    for column, dtype in FITTED_SCHEMA.items():
        values = frame.get_column(column)
        if dtype in scalar_float_types:
            present = values.drop_nulls()
            if present.len() and not bool(present.is_finite().all()):
                _fail(f"{column} contains a non-finite numeric value")
        elif dtype in list_float_types:
            for row_index, items in enumerate(values.to_list()):
                if items is None:
                    continue
                if any(
                    item is not None and not np.isfinite(float(item))
                    for item in items
                ):
                    _fail(
                        f"{column} contains a non-finite numeric value at row "
                        f"{row_index}"
                    )


@version(
    "4.3",
    dependencies=[
        PartitionStatistics,
        _canonical_object,
        _reason_counts,
        _expected_solver_expiry_indices,
        _terminal_solver_events,
        _max_abs,
        _validate_pricer_state,
        _validate_calendar_residual,
        _surface_calendar_acceptance_tolerance,
        _business_dte_replay,
        _expiry_state_pricer,
        _validate_support_mask_geometry,
        _black_vector_residual,
        _black_source_price_residual,
        _black_source_vector_residual,
        _merge_black_vector_residuals,
        _smooth_producer_replay,
        _validate_smooth_canonical_market_fit_bridge,
        _masked_quote_band_residual,
        _validate_initial_fit_quote_contract,
        _validate_fitted_numeric_finiteness,
        _canonical_discrete_dlv_densities,
        _canonical_discrete_dlv_squares,
        _discrete_dlv_grid,
        _validate_discrete_dlv_projection,
        bad_data_policy,
        bad_day_reason,
        count_business_days_between,
        is_half_day,
        apply_options_filter,
        discrete_dlv_pricer,
        fit_smooth_sanos,
        initial_no_arb_pricer,
        smooth_sanos_pricer,
    ],
)
def validate_fitted_partition(
    frame: pl.DataFrame,
    *,
    config: InternalDatasetConfig,
    trading_date: date,
    typed_empty_reason: str | None,
    expected_raw_options_artifact_id: str | None = None,
) -> PartitionStatistics:
    """Recompute population, unit, mask, state, and solver invariants for one day."""

    if frame.schema != pl.Schema(FITTED_SCHEMA):
        _fail(
            f"{trading_date}: fitted schema mismatch: expected={pl.Schema(FITTED_SCHEMA)}, "
            f"observed={frame.schema}"
        )
    _validate_fitted_numeric_finiteness(frame)
    expected_empty_reason = bad_day_reason(config.ticker, trading_date)
    if frame.is_empty():
        if typed_empty_reason != expected_empty_reason or expected_empty_reason is None:
            _fail(
                f"{trading_date}: typed-empty partition is not justified by the "
                "versioned bad-data policy"
            )
        return PartitionStatistics(
            trading_date=trading_date,
            row_count=0,
            input_expiries=0,
            initial_retained_expiries=0,
            mask_arb_free_rows=0,
            eligible_rows=0,
            selected_expiries=0,
            smooth_retained_expiries=0,
            smooth_fit_mask_rows=0,
            smooth_model_mask_rows=0,
            active_call_rows=0,
            active_put_rows=0,
            active_point_two_day_side_books=0,
            typed_empty=True,
            typed_empty_reason=typed_empty_reason,
            rejection_counts={},
            residual_maxima={},
        )
    if expected_empty_reason is not None:
        _fail(f"{trading_date}: versioned bad-data date must be a typed-empty partition")
    if typed_empty_reason is not None:
        _fail(f"{trading_date}: a populated partition cannot declare typed_empty_reason")
    if frame.get_column("ticker").unique().to_list() != [config.ticker]:
        _fail(f"{trading_date}: partition contains the wrong ticker")
    if frame.get_column("trading_date").unique().to_list() != [trading_date]:
        _fail(f"{trading_date}: partition contains the wrong trading date")
    half_day_values = frame.get_column("is_half_day").unique().to_list()
    if half_day_values != [is_half_day(trading_date)]:
        _fail(f"{trading_date}: trading-session half-day flag is inconsistent")

    expiry_ids = frame.get_column("expiry_id").to_numpy()
    normalized_strikes = frame.get_column("normalized_strike").to_numpy()
    for index in range(1, frame.height):
        if expiry_ids[index] < expiry_ids[index - 1] or (
            expiry_ids[index] == expiry_ids[index - 1]
            and normalized_strikes[index] <= normalized_strikes[index - 1]
        ):
            _fail(
                f"{trading_date}: rows are not strictly ordered by unique "
                "expiry/normalized-strike keys"
            )

    daily_constant_columns = (
        "spot",
        "previous_spot",
        "spot_return",
        "source_secid",
        "stage_version",
        "configuration_id",
        "raw_options_artifact_id",
        "normalization_diagnostics_json",
        "enrichment_diagnostics_json",
        "no_arb_diagnostics_json",
        "filter_diagnostics_json",
        "smooth_diagnostics_json",
        "discrete_dlv_diagnostics_json",
    )
    for column in daily_constant_columns:
        if frame.get_column(column).n_unique() != 1:
            _fail(f"{trading_date}: daily field {column} is not constant")
    expiry_constant_columns = (
        "is_american",
        "expiry_is_am",
        "vendor_forward",
        "discount_factor",
        "expiry_date",
        "business_dte",
        "calendar_dte",
        "time_to_expiry_years",
        "implied_forward",
        "initial_no_arb_expiry_retained",
        "initial_no_arb_input_expiry_index",
        "initial_model_variance",
    )
    for expiry_frame in frame.partition_by("expiry_id", maintain_order=True):
        expiry_id = expiry_frame.get_column("expiry_id").item(0)
        for column in expiry_constant_columns:
            if expiry_frame.get_column(column).n_unique() != 1:
                _fail(
                    f"{trading_date}: expiry field {column} is not constant for "
                    f"expiry_id={expiry_id}"
                )

    initial_index_null = frame.get_column(
        "initial_no_arb_input_expiry_index"
    ).is_null().to_numpy()
    for column in (
        "initial_model_variance",
        "fit1_volatility",
        "pure_fit1_price",
        "pure_fit1_delta",
        "pure_fit1_gamma",
        "pure_fit1_vega",
        "pure_fit1_theta",
        "pure_fit1_dk",
        "fitX_volatility",
        "pure_fitX_price",
        "pure_fitX_delta",
        "pure_fitX_gamma",
        "pure_fitX_vega",
        "pure_fitX_theta",
        "pure_fitX_dk",
        "initial_expiry_model_strikes",
        "initial_expiry_model_density",
        "initial_surface_model_strikes",
        "initial_surface_model_density",
    ):
        if not np.array_equal(frame.get_column(column).is_null().to_numpy(), initial_index_null):
            _fail(f"{trading_date}: {column} nullability differs from initial model index")
    if not np.array_equal(
        ~initial_index_null,
        frame.get_column("initial_no_arb_expiry_retained").to_numpy().astype(bool),
    ):
        _fail(f"{trading_date}: initial retained flag differs from model-state presence")
    smooth_index_null = frame.get_column("smooth_input_expiry_index").is_null().to_numpy()
    for column in (
        "smooth_model_variance",
        "smooth_expiry_fit_price",
        "smooth_surface_fit_price",
        "smooth_expiry_model_strikes",
        "smooth_expiry_model_density",
        "smooth_surface_model_strikes",
        "smooth_surface_model_density",
    ):
        if not np.array_equal(frame.get_column(column).is_null().to_numpy(), smooth_index_null):
            _fail(f"{trading_date}: {column} nullability differs from smooth model index")
    dlv_anchor = frame.get_column("discrete_dlv_anchor").to_numpy().astype(bool)
    dlv_index_null = frame.get_column("discrete_dlv_input_expiry_index").is_null().to_numpy()
    if not np.array_equal(dlv_anchor, ~dlv_index_null):
        _fail(f"{trading_date}: DLV anchor flag differs from state-index presence")
    for column in (
        "discrete_dlv_time_increment_years",
        "discrete_dlv_atm_annual_volatility",
        "discrete_dlv_input_total_variance",
        "discrete_dlv_nms",
        "discrete_dlv_model_strikes",
        "discrete_dlv_model_density",
        "discrete_dlv_squares",
        "discrete_dlv_target_calls",
        "discrete_dlv_fitted_calls",
    ):
        if not np.array_equal(frame.get_column(column).is_null().to_numpy(), dlv_index_null):
            _fail(f"{trading_date}: {column} must be present exactly on DLV anchors")
    dlv_indices = frame.get_column("discrete_dlv_input_expiry_index").drop_nulls().to_list()
    if dlv_indices != list(range(len(dlv_indices))):
        _fail(f"{trading_date}: DLV anchor indices must be contiguous from zero")
    for source_index in frame.get_column("smooth_input_expiry_index").drop_nulls().unique(
        maintain_order=True
    ):
        positions = np.flatnonzero(
            frame.get_column("smooth_input_expiry_index").to_numpy() == source_index
        )
        if len(positions) == 0 or not dlv_anchor[int(positions[0])]:
            _fail(f"{trading_date}: DLV anchor is not the first canonical row of its expiry")
        if int(np.sum(dlv_anchor[positions])) != 1:
            _fail(f"{trading_date}: smooth expiry does not have exactly one DLV anchor")

    expected_ids = {
        "configuration_id": canonical_sha256(config.daily),
        "no_arb_configuration_id": canonical_sha256(
            {"no_arbitrage": config.surface.no_arbitrage}
        ),
        "options_filter_configuration_id": canonical_sha256(
            {"options_filter": config.surface.options_filter}
        ),
        "smooth_sanos_configuration_id": canonical_sha256(
            {
                "no_arbitrage": config.surface.no_arbitrage,
                "smooth_sanos": config.surface.smooth_sanos,
            }
        ),
        "discrete_dlv_configuration_id": canonical_sha256(
            {"discrete_dlv": config.surface.discrete_dlv}
        ),
    }
    for column, expected in expected_ids.items():
        if frame.get_column(column).unique().to_list() != [expected]:
            _fail(f"{trading_date}: {column} does not match canonical configuration")
    if frame.get_column("stage_version").unique().to_list() != [str(fit_smooth_sanos.version)]:
        _fail(f"{trading_date}: stage_version is not the terminal fitted-stage version")
    if frame.get_column("raw_options_artifact_id").unique().to_list() != [
        expected_raw_options_artifact_id
    ]:
        _fail(f"{trading_date}: Raw options identity differs from planned lineage")
    spread_values = frame.get_column("surface_min_half_spread").unique().to_list()
    if spread_values != [config.surface.no_arbitrage.min_half_spread]:
        _fail(f"{trading_date}: surface_min_half_spread differs from configuration")
    if frame.get_column("smoothness").unique().to_list() != [
        config.surface.smooth_sanos.smoothness
    ]:
        _fail(f"{trading_date}: smoothness differs from configuration")

    recomputed_filtered = apply_options_filter(
        frame.select(*NO_ARB_SCHEMA),
        config=config.surface,
    )
    filter_output_columns = tuple(
        name for name in FILTERED_SCHEMA if name not in NO_ARB_SCHEMA
    )
    for column in filter_output_columns:
        if not frame.get_column(column).equals(recomputed_filtered.get_column(column)):
            _fail(
                f"{trading_date}: filtered field {column} differs from the canonical "
                "OptionsFilter projection"
            )

    filter_columns = (
        "filter_initial_eligible",
        "filter_horizon_eligible",
        "filter_nms_eligible",
        "filter_volume_eligible",
        "filter_open_interest_eligible",
        "filter_am_eligible",
        "filter_expiry_population_eligible",
        "filter_cap_eligible",
    )
    filter_masks = [frame.get_column(name).to_numpy().astype(bool) for name in filter_columns]
    for before, after, name in zip(
        filter_masks[:-1], filter_masks[1:], filter_columns[1:], strict=True
    ):
        if np.any(after & ~before):
            _fail(f"{trading_date}: cumulative filter mask {name} revives rejected rows")
    selected = frame.get_column("options_filter_eligible").to_numpy().astype(bool)
    if not np.array_equal(selected, filter_masks[-1]):
        _fail(f"{trading_date}: options_filter_eligible differs from filter_cap_eligible")
    filter_reasons = frame.get_column("options_filter_rejection_reason").to_numpy()
    if any(filter_reasons[index] is not None for index in np.flatnonzero(selected)):
        _fail(f"{trading_date}: selected rows carry filter rejection reasons")
    if any(filter_reasons[index] is None for index in np.flatnonzero(~selected)):
        _fail(f"{trading_date}: rejected rows lack a filter rejection reason")

    pre_mask = frame.get_column("mask_arb_free").to_numpy().astype(bool)
    initial_retained = frame.get_column(
        "initial_no_arb_expiry_retained"
    ).to_numpy().astype(bool)
    initial_reasons = frame.get_column("initial_no_arb_rejection_reason").to_numpy()
    expected_initial_reasons = np.full(frame.height, None, dtype=object)
    expected_initial_reasons[~initial_retained] = "initial_no_arb_expiry_dropped"
    expected_initial_reasons[initial_retained & ~pre_mask] = "initial_no_arb_strike_masked"
    if initial_reasons.tolist() != expected_initial_reasons.tolist():
        _fail(f"{trading_date}: initial no-arbitrage rejection reasons are inconsistent")

    no_arb_text = frame.get_column("no_arb_diagnostics_json").item(0)
    filter_text = frame.get_column("filter_diagnostics_json").item(0)
    smooth_text = frame.get_column("smooth_diagnostics_json").item(0)
    dlv_text = frame.get_column("discrete_dlv_diagnostics_json").item(0)
    no_arb = _canonical_object(no_arb_text, "no-arbitrage diagnostics")
    filter_diagnostics = _canonical_object(filter_text, "filter diagnostics")
    smooth = _canonical_object(smooth_text, "smooth diagnostics")
    dlv_diagnostics = _canonical_object(dlv_text, "discrete-DLV diagnostics")
    no_arb_solver_indices = _expected_solver_expiry_indices(
        frame,
        np.ones(frame.height, dtype=bool),
        config.surface.no_arbitrage.min_strikes_per_expiry,
    )
    initial_retained_indices = tuple(
        int(value)
        for value in frame.get_column("initial_no_arb_input_expiry_index")
        .drop_nulls()
        .unique(maintain_order=True)
    )
    initial_solver_evidence = _terminal_solver_events(
        no_arb,
        "no-arbitrage",
        expected_expiry_indices=no_arb_solver_indices,
        retained_expiry_indices=initial_retained_indices,
    )
    initial_relaxed_support = initial_solver_evidence.relaxed_support
    input_expiry_ids = frame.get_column("expiry_id").unique(maintain_order=True).to_list()
    retained_expiry_ids = (
        frame.filter(pl.col("initial_no_arb_expiry_retained"))
        .get_column("expiry_id")
        .unique(maintain_order=True)
        .to_list()
    )
    smooth_relaxed_support: frozenset[int] = frozenset()
    smooth_solver_evidence: _TerminalSolverEvidence | None = None
    if np.any(selected):
        smooth_solver_indices = _expected_solver_expiry_indices(
            frame,
            selected,
            config.surface.no_arbitrage.min_strikes_per_expiry,
        )
        smooth_retained_indices = tuple(
            int(value)
            for value in frame.get_column("smooth_input_expiry_index")
            .drop_nulls()
            .unique(maintain_order=True)
        )
        smooth_solver_evidence = _terminal_solver_events(
            smooth,
            "smooth",
            expected_expiry_indices=smooth_solver_indices,
            retained_expiry_indices=smooth_retained_indices,
        )
        smooth_relaxed_support = smooth_solver_evidence.relaxed_support
        _terminal_solver_events(
            dlv_diagnostics,
            "discrete-DLV",
            expected_expiry_indices=None,
        )
    _validate_support_mask_geometry(
        frame,
        mask_column="mask_arb_free",
        state_index_column="initial_no_arb_input_expiry_index",
        expiry_strikes_column="initial_expiry_model_strikes",
        allow_masked_support_superset=initial_relaxed_support,
        label=f"{trading_date}: initial no-arbitrage mask",
    )
    smooth_fit_mask = frame.get_column("smooth_fit_mask").to_numpy().astype(bool)
    smooth_model_mask = frame.get_column("smooth_model_mask").to_numpy().astype(bool)
    if np.any(smooth_fit_mask & ~selected):
        _fail(f"{trading_date}: smooth_fit_mask contains an unselected row")
    if not np.array_equal(smooth_model_mask, selected & pre_mask & smooth_fit_mask):
        _fail(f"{trading_date}: smooth_model_mask has changed semantics")
    smooth_reasons = frame.get_column("smooth_rejection_reason").to_numpy()
    expected_smooth_reasons = np.full(frame.height, None, dtype=object)
    expected_smooth_reasons[~selected] = "not_options_filter_eligible"
    expected_smooth_reasons[selected & smooth_index_null] = "smooth_expiry_dropped"
    expected_smooth_reasons[selected & ~smooth_index_null & ~smooth_fit_mask] = (
        "smooth_strike_masked"
    )
    if smooth_reasons.tolist() != expected_smooth_reasons.tolist():
        _fail(f"{trading_date}: smooth rejection reasons are inconsistent")
    _validate_support_mask_geometry(
        frame,
        mask_column="smooth_fit_mask",
        state_index_column="smooth_input_expiry_index",
        expiry_strikes_column="smooth_expiry_model_strikes",
        allow_masked_support_superset=smooth_relaxed_support,
        label=f"{trading_date}: smooth fit mask",
    )

    def floats(name: str) -> np.ndarray[Any, Any]:
        return frame.get_column(name).to_numpy().astype(np.float64, copy=False)

    expiry_dates = frame.get_column("expiry_date").to_numpy().astype("datetime64[D]")
    expiry_is_am = frame.get_column("expiry_is_am").to_numpy().astype(bool)
    expected_calendar_dte = (
        expiry_dates - np.datetime64(trading_date, "D")
    ).astype(np.int32)
    observed_calendar_dte = frame.get_column("calendar_dte").to_numpy()
    if not np.array_equal(observed_calendar_dte, expected_calendar_dte):
        _fail(f"{trading_date}: calendar_dte differs from expiry_date minus trading_date")
    expected_business_dte = _business_dte_replay(
        trading_date=trading_date,
        expiry_dates=expiry_dates,
        expiry_is_am=expiry_is_am,
        am_business_day_subtraction=config.daily.clock.am_business_day_subtraction,
    )
    observed_business_dte = frame.get_column("business_dte").to_numpy()
    if not np.array_equal(observed_business_dte, expected_business_dte):
        _fail(
            f"{trading_date}: business_dte differs from the configured AM/PM business clock"
        )
    expected_time = (expected_business_dte / np.float32(DAYS_PER_YEAR)).astype(np.float32)
    if not np.array_equal(frame.get_column("time_to_expiry_years").to_numpy(), expected_time):
        _fail(f"{trading_date}: time_to_expiry_years differs from business_dte/252")
    expiry_objects = frame.get_column("expiry_date").to_list()
    expected_expiry_id = np.asarray(
        [
            value.year * 100000 + value.month * 1000 + value.day * 10 + (0 if am else 1)
            for value, am in zip(expiry_objects, expiry_is_am, strict=True)
        ],
        dtype=np.int32,
    )
    if not np.array_equal(expiry_ids, expected_expiry_id):
        _fail(f"{trading_date}: expiry_id differs from expiry date and AM/PM settlement")
    observed_spot = frame.get_column("spot").to_numpy()
    observed_previous_spot = frame.get_column("previous_spot").to_numpy()
    expected_spot_return = np.float32(
        np.float32(observed_spot[0])
        / np.float32(observed_previous_spot[0])
        - np.float32(1.0)
    )
    if not np.array_equal(
        frame.get_column("spot_return").to_numpy(),
        np.full(frame.height, expected_spot_return, dtype=np.float32),
    ):
        _fail(f"{trading_date}: spot_return differs from the Float32 close return")

    black_vector_residual = _merge_black_vector_residuals(
        _black_vector_residual(
            frame,
            price_column="pure_bid_price",
            volatility_column="bid_implied_volatility",
            greek_prefix="pure_bid",
            strike_was_persisted_input=False,
            volatility_minimum=config.daily.enrichment.min_implied_vol,
            volatility_maximum=config.daily.enrichment.max_implied_vol,
        ),
        _black_vector_residual(
            frame,
            price_column="pure_ask_price",
            volatility_column="ask_implied_volatility",
            greek_prefix="pure_ask",
            strike_was_persisted_input=False,
            volatility_minimum=config.daily.enrichment.min_implied_vol,
            volatility_maximum=config.daily.enrichment.max_implied_vol,
        ),
    )
    if black_vector_residual.maximum_excess > 0.0:
        _fail(
            f"{trading_date}: stored enrichment Black vector exceeds its persisted "
            f"Float32 contract by {black_vector_residual.maximum_excess}"
        )

    call_bid = floats("pure_market_call_bid")
    call_ask = floats("pure_market_call_ask")
    put_bid = floats("pure_market_put_bid")
    put_ask = floats("pure_market_put_ask")
    valid_call = np.isfinite(call_bid) & np.isfinite(call_ask) & (call_bid >= 0.0) & (
        call_ask > call_bid
    )
    valid_put = np.isfinite(put_bid) & np.isfinite(put_ask) & (put_bid >= 0.0) & (
        put_ask > put_bid
    )
    expected_call_active = selected & pre_mask & valid_call
    expected_put_active = selected & pre_mask & valid_put
    call_active = frame.get_column("call_active").to_numpy().astype(bool)
    put_active = frame.get_column("put_active").to_numpy().astype(bool)
    if not np.array_equal(call_active, expected_call_active):
        _fail(f"{trading_date}: call_active has changed semantics")
    if not np.array_equal(put_active, expected_put_active):
        _fail(f"{trading_date}: put_active has changed semantics")

    smooth_call = floats("smooth_call_mid")
    smooth_put = floats("smooth_put_mid")
    available = frame.get_column("smooth_mid_available").to_numpy().astype(bool)
    expected_available = selected & np.isfinite(smooth_call) & np.isfinite(smooth_put)
    if not np.array_equal(available, expected_available):
        _fail(f"{trading_date}: smooth_mid_available has changed semantics")
    put_call_residual = _max_abs(
        smooth_put[available],
        smooth_call[available] - 1.0 + normalized_strikes[available],
    )
    if put_call_residual > 1.0e-12:
        _fail(f"{trading_date}: smooth put/call parity residual is {put_call_residual}")

    spot = floats("spot")
    forward = floats("implied_forward")
    discount = floats("discount_factor")
    strike = floats("strike")
    if (
        np.any(~np.isfinite(spot) | (spot <= 0.0))
        or np.any(~np.isfinite(forward) | (forward <= 0.0))
        or np.any(~np.isfinite(discount) | (discount <= 0.0))
    ):
        _fail(f"{trading_date}: spot, forward and discount bridges must be finite and positive")
    strike_residual = _max_abs(normalized_strikes, strike / forward)
    spot_strike_residual = _max_abs(floats("spot_normalized_strike"), strike / spot)
    spot_reference_residual = _max_abs(floats("spot_reference"), spot)
    previous_spot_residual = _max_abs(
        floats("normalized_previous_spot"), floats("previous_spot") / spot
    )
    forward_residual = _max_abs(
        floats("spot_normalized_implied_forward"), forward / spot
    )
    discounted_forward = discount * floats("spot_normalized_implied_forward")
    direct_call_residual = _max_abs(
        call_bid[selected],
        floats("spot_normalized_market_call_bid")[selected] / discounted_forward[selected],
    )
    direct_call_residual = max(
        direct_call_residual,
        _max_abs(
            call_ask[selected],
            floats("spot_normalized_market_call_ask")[selected]
            / discounted_forward[selected],
        ),
    )
    direct_put_residual = _max_abs(
        put_bid[selected],
        floats("spot_normalized_market_put_bid")[selected] / discounted_forward[selected],
    )
    direct_put_residual = max(
        direct_put_residual,
        _max_abs(
            put_ask[selected],
            floats("spot_normalized_market_put_ask")[selected]
            / discounted_forward[selected],
        ),
    )
    if max(
        strike_residual,
        spot_strike_residual,
        spot_reference_residual,
        previous_spot_residual,
        forward_residual,
        direct_call_residual,
        direct_put_residual,
    ) > 2e-6:
        _fail(f"{trading_date}: normalized-unit bridge residual exceeds 2e-6")

    expected_no_arb = {
        "input_rows": frame.height,
        "input_expiries": len(input_expiry_ids),
        "retained_expiries": len(retained_expiry_ids),
        "dropped_expiry_ids": [
            value for value in input_expiry_ids if value not in set(retained_expiry_ids)
        ],
        "masked_strikes": int(
            np.sum(
                frame.get_column("initial_no_arb_expiry_retained").to_numpy().astype(bool)
                & ~pre_mask
            )
        ),
        "outcome": "completed",
        "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
    }
    if any(no_arb.get(name) != value for name, value in expected_no_arb.items()):
        _fail(f"{trading_date}: no-arbitrage diagnostic counts are inconsistent")
    filter_counts = filter_diagnostics.get("counts")
    expected_filter_counts = {
        "initial": int(np.sum(filter_masks[0])),
        "horizon": int(np.sum(filter_masks[1])),
        "nms": int(np.sum(filter_masks[2])),
        "volume": int(np.sum(filter_masks[3])),
        "open_interest": int(np.sum(filter_masks[4])),
        "am": int(np.sum(filter_masks[5])),
        "expiry_population": int(np.sum(filter_masks[6])),
        "cap": int(np.sum(filter_masks[7])),
    }
    if filter_counts != expected_filter_counts:
        _fail(f"{trading_date}: filter diagnostic counts are inconsistent")
    expected_filter_rejections = {
        str(reason): int(np.sum(filter_reasons == reason))
        for reason in sorted({value for value in filter_reasons if value is not None})
    }
    if filter_diagnostics.get("rejections") != expected_filter_rejections:
        _fail(f"{trading_date}: filter diagnostic rejections are inconsistent")
    if int(smooth.get("selected_rows", -1)) != int(np.sum(selected)):
        _fail(f"{trading_date}: smooth diagnostic selected count is inconsistent")
    if np.any(selected):
        if (
            dlv_diagnostics.get("schema")
            != "sanos-discrete-dlv-projection-diagnostics-v3"
            or dlv_diagnostics.get("outcome") != "completed"
            or dlv_diagnostics.get("source") != "smooth_sanos_surface"
            or dlv_diagnostics.get("inner_nodes") != config.surface.discrete_dlv.inner_nodes
        ):
            _fail(f"{trading_date}: discrete-DLV diagnostics contract changed")
        selected_expiry_ids = (
            frame.filter(pl.Series(selected))
            .get_column("expiry_id")
            .unique(maintain_order=True)
            .to_list()
        )
        smooth_retained_ids = (
            frame.filter(pl.col("smooth_input_expiry_index").is_not_null())
            .get_column("expiry_id")
            .unique(maintain_order=True)
            .to_list()
        )
        expected_smooth = {
            "input_rows": frame.height,
            "selected_rows": int(np.sum(selected)),
            "selected_expiries": len(selected_expiry_ids),
            "retained_expiries": len(smooth_retained_ids),
            "dropped_expiry_ids": [
                value
                for value in selected_expiry_ids
                if value not in set(smooth_retained_ids)
            ],
            "smooth_masked_strikes": int(np.sum(selected & ~smooth_fit_mask)),
            "model_mask_rows": int(np.sum(smooth_model_mask)),
            "call_active_rows": int(np.sum(call_active)),
            "put_active_rows": int(np.sum(put_active)),
            "outcome": "completed",
            "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
        }
        if any(smooth.get(name) != value for name, value in expected_smooth.items()):
            _fail(f"{trading_date}: smooth diagnostic counts are inconsistent")
    elif (
        smooth.get("outcome") != "no_eligible_options"
        or smooth.get("solver_timeout_seconds") != SOLVER_TIMEOUT_SECONDS
    ):
        _fail(f"{trading_date}: zero-selected day lacks explicit smooth outcome")
    elif dlv_diagnostics != {
        "schema": "sanos-discrete-dlv-projection-diagnostics-v3",
        "outcome": "no_smooth_state",
        "source": "smooth_sanos_surface",
    }:
        _fail(f"{trading_date}: empty discrete-DLV diagnostics changed")

    input_expiry_ids_tuple = tuple(int(value) for value in input_expiry_ids)
    initial_expiry_pricer, initial_state_indices, initial_expiry_residuals = (
        _expiry_state_pricer(
            frame,
            index_column="initial_no_arb_input_expiry_index",
            expiry_strikes_column="initial_expiry_model_strikes",
            expiry_density_column="initial_expiry_model_density",
            surface_strikes_column="initial_surface_model_strikes",
            surface_density_column="initial_surface_model_density",
            mapping_column="initial_x_to_surface_indices",
            variance_column="initial_model_variance",
            expected_source_expiry_ids=input_expiry_ids_tuple,
            smoothness=None,
            label=f"{trading_date} initial",
        )
    )
    selected_expiry_ids_tuple = tuple(
        int(value)
        for value in frame.filter(pl.Series(selected))
        .get_column("expiry_id")
        .unique(maintain_order=True)
        .to_list()
    )
    smooth_expiry_pricer: SanosPricer | None = None
    smooth_state_indices: tuple[int, ...] = ()
    smooth_expiry_residuals: dict[str, float] = {}
    if np.any(selected):
        smooth_expiry_pricer, smooth_state_indices, smooth_expiry_residuals = (
            _expiry_state_pricer(
                frame,
                index_column="smooth_input_expiry_index",
                expiry_strikes_column="smooth_expiry_model_strikes",
                expiry_density_column="smooth_expiry_model_density",
                surface_strikes_column="smooth_surface_model_strikes",
                surface_density_column="smooth_surface_model_density",
                mapping_column="smooth_x_to_surface_indices",
                variance_column="smooth_model_variance",
                expected_source_expiry_ids=selected_expiry_ids_tuple,
                # The recovered smooth fitter deliberately performs each expiry fit
                # with a linear payoff before fitting the smoothed surface.
                smoothness=None,
                label=f"{trading_date} smooth",
            )
        )

    initial_surface_pricer = initial_no_arb_pricer(frame)
    smooth_pricer = smooth_sanos_pricer(frame) if np.any(selected) else None
    initial_surface_calendar_tolerance = _surface_calendar_acceptance_tolerance(
        initial_solver_evidence.terminal_solver_call,
        f"{trading_date} initial",
    )
    smooth_calendar_tolerance = (
        0.0
        if smooth_pricer is None
        else _surface_calendar_acceptance_tolerance(
            cast(_TerminalSolverEvidence, smooth_solver_evidence).terminal_solver_call,
            f"{trading_date} smooth",
        )
    )
    initial_surface_state_residuals = _validate_pricer_state(
        initial_surface_pricer,
        "initial SANOS state",
        check_dlvs=False,
        calendar_tolerance=initial_surface_calendar_tolerance,
    )
    dlv_state_residuals: dict[str, float] = {}
    smooth_state_residuals = (
        {}
        if smooth_pricer is None
        else _validate_pricer_state(
            smooth_pricer,
            "smooth SANOS state",
            check_dlvs=False,
            calendar_tolerance=smooth_calendar_tolerance,
        )
    )
    dlv_density_error = 0.0
    dlv_canonical_density_error = 0.0
    dlv_call_error = 0.0
    dlv_projection_call_error = 0.0
    dlv_atm_iv_error = 0.0
    dlv_conditioned_iv_error = 0.0
    dlv_conditioned_iv_p99 = 0.0
    dlv_minimum_bound_shortfall = 0.0
    dlv_maximum_bound_excess = 0.0
    if smooth_pricer is not None:
        dlv_pricer = discrete_dlv_pricer(frame)
        dlv_state_residuals = _validate_pricer_state(
            dlv_pricer,
            "discrete DLV projection",
            check_dlvs=True,
        )
        stored_dlvs: list[np.ndarray[Any, Any]] = []
        dlv_state = frame.filter(pl.col("discrete_dlv_anchor")).sort(
            "discrete_dlv_input_expiry_index"
        )
        if dlv_state.height != len(dlv_pricer.expiries):
            _fail(f"{trading_date}: DLV anchor count differs from model expiries")
        for input_index in range(len(dlv_pricer.expiries)):
            group = dlv_state.filter(
                pl.col("discrete_dlv_input_expiry_index") == input_index
            )
            values = group.get_column("discrete_dlv_squares").to_list()
            if len(values) != 1 or values[0] is None:
                _fail(f"{trading_date}: DLV state must have one anchor per expiry")
            stored_dlvs.append(np.asarray(values[0], dtype=np.float64))
        recomputed_dlvs = _canonical_discrete_dlv_squares(dlv_pricer)
        for stored, recomputed in zip(stored_dlvs, recomputed_dlvs, strict=True):
            error = _max_abs(stored, recomputed)
            if error > config.surface.discrete_dlv.tolerances.dlv_fixed_point_max_abs:
                _fail(f"{trading_date}: stored DLV differs from encoded density by {error}")
        canonical_decoded = _canonical_discrete_dlv_densities(
            dlv_pricer, stored_dlvs
        )
        for observed, expected in zip(
            canonical_decoded,
            dlv_pricer.xdensities,
            strict=True,
        ):
            dlv_canonical_density_error = max(
                dlv_canonical_density_error,
                _max_abs(observed, expected),
            )
        if (
            dlv_canonical_density_error
            > config.surface.discrete_dlv.tolerances.density_inverse_max_abs
        ):
            _fail(
                f"{trading_date}: canonical DLV density round-trip error is "
                f"{dlv_canonical_density_error}"
            )
        decoded = dlv_pricer.densities_from_dlvsqs(stored_dlvs)
        for observed, expected in zip(decoded, dlv_pricer.xdensities, strict=True):
            dlv_density_error = max(dlv_density_error, _max_abs(observed, expected))
        if (
            dlv_density_error
            > config.surface.discrete_dlv.tolerances.density_inverse_max_abs
        ):
            _fail(f"{trading_date}: DLV density round-trip error is {dlv_density_error}")
        decoded_pricer = type(dlv_pricer)(
            smoothness=dlv_pricer.smoothness,
            expiries=dlv_pricer.expiries,
            variances=dlv_pricer.variances,
            xstrikes=dlv_pricer.xstrikes,
            xdensities=decoded,
            min_half_spread=dlv_pricer.min_half_spread,
            min_dlv=dlv_pricer.min_dlv,
            max_dlv=dlv_pricer.max_dlv,
        )
        query = np.asarray([0.713, 0.947, 1.037, 1.289], dtype=np.float64)
        for expiry in dlv_pricer.expiries:
            dlv_call_error = max(
                dlv_call_error,
                _max_abs(decoded_pricer(expiry, query), dlv_pricer(expiry, query)),
            )
        if dlv_call_error > config.surface.discrete_dlv.tolerances.call_inverse_max_abs:
            _fail(f"{trading_date}: DLV off-grid call round-trip error is {dlv_call_error}")

        smooth_expiries = np.asarray(smooth_pricer.expiries, dtype=np.float64)
        smooth_variances = np.asarray(smooth_pricer.variances, dtype=np.float64)
        if not np.array_equal(
            smooth_expiries, np.asarray(dlv_pricer.expiries, dtype=np.float64)
        ):
            _fail(f"{trading_date}: DLV expiries differ from source smooth SANOS")
        if not np.array_equal(
            smooth_variances, np.asarray(dlv_pricer.variances, dtype=np.float64)
        ):
            _fail(f"{trading_date}: DLV total variances differ from source smooth SANOS")
        stored_time_increments = dlv_state.get_column(
            "discrete_dlv_time_increment_years"
        ).to_numpy()
        expected_time_increments = np.diff(smooth_expiries, prepend=0.0)
        if not np.array_equal(stored_time_increments, expected_time_increments):
            _fail(f"{trading_date}: stored DLV time increments changed")
        stored_atm_volatilities = dlv_state.get_column(
            "discrete_dlv_atm_annual_volatility"
        ).to_numpy()
        expected_atm_volatilities = np.sqrt(smooth_variances / smooth_expiries)
        if not np.array_equal(stored_atm_volatilities, expected_atm_volatilities):
            _fail(f"{trading_date}: stored DLV ATM annual volatilities changed")
        expected_nms, expected_inner, expected_supports = _discrete_dlv_grid(
            smooth_expiries,
            expected_atm_volatilities,
            config.surface.discrete_dlv,
        )
        stored_nms = [
            np.asarray(value, dtype=np.float64)
            for value in dlv_state.get_column("discrete_dlv_nms").to_list()
        ]
        stored_targets = np.stack(
            [
                np.asarray(value, dtype=np.float64)
                for value in dlv_state.get_column("discrete_dlv_target_calls").to_list()
            ]
        )
        stored_fitted = np.stack(
            [
                np.asarray(value, dtype=np.float64)
                for value in dlv_state.get_column("discrete_dlv_fitted_calls").to_list()
            ]
        )
        recomputed_targets = np.asarray(
            smooth_pricer(smooth_expiries[:, None], expected_inner), dtype=np.float64
        )
        recomputed_fitted = np.asarray(
            dlv_pricer(smooth_expiries[:, None], expected_inner), dtype=np.float64
        )
        if _max_abs(stored_targets, recomputed_targets) > 1.0e-14:
            _fail(f"{trading_date}: stored DLV target calls changed")
        if _max_abs(stored_fitted, recomputed_fitted) > 1.0e-14:
            _fail(f"{trading_date}: stored DLV fitted calls changed")
        for index, (nms, support) in enumerate(
            zip(stored_nms, dlv_pricer.xstrikes, strict=True)
        ):
            if not np.array_equal(nms, expected_nms):
                _fail(f"{trading_date}: DLV NMS grid changed at expiry {index}")
            if _max_abs(np.asarray(support), expected_supports[index]) > 1.0e-14:
                _fail(f"{trading_date}: DLV support differs from source smooth geometry")
        smooth_source_indices = sorted(
            int(value)
            for value in frame.get_column("smooth_input_expiry_index")
            .drop_nulls()
            .unique()
            .to_list()
        )
        listed_strikes = [
            frame.filter(pl.col("smooth_input_expiry_index") == source_index)
            .get_column("normalized_strike")
            .to_numpy()
            .astype(np.float64, copy=False)
            for source_index in smooth_source_indices
        ]
        target_fit_price_band = cast(
            Mapping[str, Any],
            dlv_diagnostics["target_fit_price_band"],
        )
        numerical_regularization = cast(
            Mapping[str, Any],
            dlv_diagnostics["numerical_regularization"],
        )
        revalidated_dlv_diagnostics = _validate_discrete_dlv_projection(
            smooth=smooth_pricer,
            discrete=dlv_pricer,
            raw_nms=expected_nms,
            inner=expected_inner,
            supports=expected_supports,
            target_calls=stored_targets,
            fitted_calls=stored_fitted,
            listed_strikes=listed_strikes,
            dlv_squares=stored_dlvs,
            solver_events=[dict(value) for value in dlv_diagnostics["solver_events"]],
            solver_warnings=[str(value) for value in dlv_diagnostics["solver_warnings"]],
            stabilization={
                "density_max_abs": float(
                    dlv_diagnostics["stabilization_density_max_abs"]
                ),
                "dlv_max_abs": float(dlv_diagnostics["stabilization_dlv_max_abs"]),
                "pre_target_call_max_abs": float(
                    dlv_diagnostics["pre_stabilization_target_call_max_abs"]
                ),
                "post_target_call_max_abs": float(
                    dlv_diagnostics["post_stabilization_target_call_max_abs"]
                ),
                "pre_target_iv_max_abs": float(
                    target_fit_price_band[
                        "pre_stabilization_maximum_absolute_iv_error"
                    ]
                ),
                "post_target_iv_max_abs": float(
                    target_fit_price_band["maximum_absolute_iv_error"]
                ),
                "pre_target_band_max_violation": float(
                    target_fit_price_band["pre_stabilization_maximum_violation"]
                ),
                "post_target_band_max_violation": float(
                    target_fit_price_band["maximum_violation"]
                ),
            },
            pre_state_metrics=cast(
                Mapping[str, float | int],
                numerical_regularization["pre_stabilization_state"],
            ),
            config=config.surface.discrete_dlv,
        )
        if canonical_json(revalidated_dlv_diagnostics) != dlv_text:
            _fail(f"{trading_date}: stored DLV diagnostics differ from strict recomputation")
        dlv_projection_call_error = float(
            cast(Mapping[str, Any], revalidated_dlv_diagnostics["projection_call"])[
                "max"
            ]
        )
        dlv_atm_iv_error = float(
            cast(Mapping[str, Any], revalidated_dlv_diagnostics["atm_annual_iv"])[
                "max"
            ]
        )
        conditioned_iv = cast(
            Mapping[str, Any], revalidated_dlv_diagnostics["conditioned_annual_iv"]
        )
        dlv_conditioned_iv_error = float(conditioned_iv["max"])
        dlv_conditioned_iv_p99 = float(conditioned_iv["p99"])
        dlv_minimum_bound_shortfall = float(
            revalidated_dlv_diagnostics["minimum_bound_shortfall"]
        )
        dlv_maximum_bound_excess = float(
            revalidated_dlv_diagnostics["maximum_bound_excess"]
        )

    initial_surface_fit_reconstruction = 0.0
    initial_surface_fit_contract_excess = 0.0
    initial_surface_fit_black_roundtrip = 0.0
    expiry_fit_reconstruction = 0.0
    expiry_fit_contract_excess = 0.0
    expiry_fit_black_roundtrip = 0.0
    initial_surface_fit_quote_band_violation = 0.0
    expiry_fit_quote_containment = 0.0
    fitted_black_vector_residual = _merge_black_vector_residuals()
    for model_index, (input_index, expiry) in enumerate(
        zip(initial_state_indices, initial_surface_pricer.expiries, strict=True)
    ):
        rows = frame.filter(pl.col("initial_no_arb_input_expiry_index") == input_index)
        observed = np.asarray(
            initial_surface_pricer(expiry, rows["normalized_strike"].to_numpy())
        )
        surface_replay = _black_source_vector_residual(
            rows,
            source_prices=observed,
            price_column="pure_fitX_price",
            volatility_column="fitX_volatility",
            greek_prefix="pure_fitX",
        )
        if surface_replay.mismatch_count:
            _fail(
                f"{trading_date}: initial surface-fit deterministic Black "
                f"round-trip changed {surface_replay.mismatch_count} values in "
                f"{surface_replay.mismatch_fields}"
            )
        initial_surface_fit_reconstruction = max(
            initial_surface_fit_reconstruction,
            surface_replay.source_price,
        )
        initial_surface_fit_contract_excess = max(
            initial_surface_fit_contract_excess,
            surface_replay.source_contract_excess,
        )
        initial_surface_fit_black_roundtrip = max(
            initial_surface_fit_black_roundtrip,
            surface_replay.float32_max_abs,
        )
        expiry_observed = np.asarray(
            initial_expiry_pricer(
                initial_expiry_pricer.expiries[model_index],
                rows["normalized_strike"].to_numpy(),
            )
        )
        expiry_replay = _black_source_vector_residual(
            rows,
            source_prices=expiry_observed,
            price_column="pure_fit1_price",
            volatility_column="fit1_volatility",
            greek_prefix="pure_fit1",
        )
        if expiry_replay.mismatch_count:
            _fail(
                f"{trading_date}: initial expiry-fit deterministic Black "
                f"round-trip changed {expiry_replay.mismatch_count} values in "
                f"{expiry_replay.mismatch_fields}"
            )
        expiry_fit_reconstruction = max(
            expiry_fit_reconstruction,
            expiry_replay.source_price,
        )
        expiry_fit_contract_excess = max(
            expiry_fit_contract_excess,
            expiry_replay.source_contract_excess,
        )
        expiry_fit_black_roundtrip = max(
            expiry_fit_black_roundtrip,
            expiry_replay.float32_max_abs,
        )
        fitted_black_vector_residual = _merge_black_vector_residuals(
            fitted_black_vector_residual,
            _black_vector_residual(
                rows,
                price_column="pure_fit1_price",
                volatility_column="fit1_volatility",
                greek_prefix="pure_fit1",
                strike_was_persisted_input=True,
                volatility_minimum=0.01,
                volatility_maximum=5.0,
            ),
            _black_vector_residual(
                rows,
                price_column="pure_fitX_price",
                volatility_column="fitX_volatility",
                greek_prefix="pure_fitX",
                strike_was_persisted_input=True,
                volatility_minimum=0.01,
                volatility_maximum=5.0,
            ),
        )
        local_mask = rows["mask_arb_free"].to_numpy().astype(bool)
        if np.any(local_mask):
            initial_surface_fit_quote_band_violation = max(
                initial_surface_fit_quote_band_violation,
                _masked_quote_band_residual(
                    bids=rows["pure_bid_price"].to_numpy(),
                    asks=rows["pure_ask_price"].to_numpy(),
                    fits=observed,
                    mask=local_mask,
                ),
            )
            expiry_fit_quote_containment = max(
                expiry_fit_quote_containment,
                _masked_quote_band_residual(
                    bids=rows["pure_bid_price"].to_numpy(),
                    asks=rows["pure_ask_price"].to_numpy(),
                    fits=expiry_observed,
                    mask=local_mask,
                ),
            )
    if initial_surface_fit_contract_excess > 0.0:
        _fail(
            f"{trading_date}: reconstructed initial surface-fit prices exceed "
            "the implied-volatility/Float32 round-trip contract by "
            f"{initial_surface_fit_contract_excess}"
        )
    if expiry_fit_contract_excess > 0.0:
        _fail(
            f"{trading_date}: reconstructed initial expiry-fit prices exceed "
            "the implied-volatility/Float32 round-trip contract by "
            f"{expiry_fit_contract_excess}"
        )
    if fitted_black_vector_residual.maximum_excess > 0.0:
        _fail(
            f"{trading_date}: stored fitted Black vector exceeds its persisted "
            f"Float32 contract by {fitted_black_vector_residual.maximum_excess}"
        )
    _validate_initial_fit_quote_contract(
        expiry_fit_residual=expiry_fit_quote_containment,
        surface_fit_residual=initial_surface_fit_quote_band_violation,
        label=str(trading_date),
    )

    smooth_reconstruction = 0.0
    smooth_expiry_reconstruction = 0.0
    smooth_surface_fit_reconstruction = 0.0
    smooth_fit_error_reconstruction = 0.0
    smooth_canonical_market_fit_bridge = 0.0
    smooth_canonical_surface_fit_bridge = 0.0
    calendar_residual = 0.0
    if smooth_pricer is not None:
        assert smooth_expiry_pricer is not None
        selected_groups = [
            frame.filter(pl.Series(selected & (expiry_ids == expiry_id)))
            for expiry_id in selected_expiry_ids_tuple
        ]
        producer_replay = _smooth_producer_replay(
            frame,
            selected_groups=selected_groups,
            smoothness=config.surface.smooth_sanos.smoothness,
            min_half_spread=config.surface.no_arbitrage.min_half_spread,
        )
        if producer_replay.retained_input_indices != smooth_state_indices:
            _fail(f"{trading_date}: smooth producer replay indices changed")
        for model_index, input_index in enumerate(smooth_state_indices):
            rows = selected_groups[input_index]
            expiry_observed = producer_replay.expiry_fits[model_index]
            surface_observed = producer_replay.surface_fits[model_index]
            smooth_expiry_reconstruction = max(
                smooth_expiry_reconstruction,
                _max_abs(
                    expiry_observed,
                    rows["smooth_expiry_fit_price"].to_numpy(),
                ),
            )
            smooth_surface_fit_reconstruction = max(
                smooth_surface_fit_reconstruction,
                _max_abs(
                    surface_observed,
                    rows["smooth_surface_fit_price"].to_numpy(),
                ),
            )
            canonical = np.asarray(
                smooth_pricer(
                    smooth_pricer.expiries[model_index],
                    rows["normalized_strike"].to_numpy(),
                )
            )
            smooth_canonical_surface_fit_bridge = max(
                smooth_canonical_surface_fit_bridge,
                _max_abs(canonical, surface_observed),
            )
        if smooth_expiry_reconstruction > 1.0e-12:
            _fail(
                f"{trading_date}: smooth expiry-fit producer replay changed by "
                f"{smooth_expiry_reconstruction}"
            )
        if smooth_surface_fit_reconstruction > 1.0e-12:
            _fail(
                f"{trading_date}: smooth surface-fit producer replay changed by "
                f"{smooth_surface_fit_reconstruction}"
            )

        for group, fitted_values, fit_errors in zip(
            selected_groups,
            producer_replay.market_fits,
            producer_replay.market_fit_errors,
            strict=True,
        ):
            smooth_reconstruction = max(
                smooth_reconstruction,
                _max_abs(fitted_values, group["smooth_call_mid"].to_numpy()),
            )
            smooth_fit_error_reconstruction = max(
                smooth_fit_error_reconstruction,
                _max_abs(
                    fit_errors,
                    group["smooth_fit_error_half_spreads"].to_numpy(),
                ),
            )
            expiry = np.float32(group.get_column("time_to_expiry_years").item(0))
            canonical = np.asarray(
                smooth_pricer(expiry, group["normalized_strike"].to_numpy())
            )
            smooth_canonical_market_fit_bridge = max(
                smooth_canonical_market_fit_bridge,
                _max_abs(canonical, fitted_values),
            )
        _validate_smooth_canonical_market_fit_bridge(
            smooth_canonical_market_fit_bridge,
            label=str(trading_date),
        )
        if smooth_reconstruction > 1.0e-12 or smooth_fit_error_reconstruction > 1.0e-12:
            _fail(f"{trading_date}: smooth market-fit values or errors changed")
        summary_fields = {
            "smooth_pcnt_fit": ("pcnt_fit", producer_replay.pcnt_fit),
            "smooth_median_error": ("median_error", producer_replay.median_error),
            "smooth_mean_error": ("mean_error", producer_replay.mean_error),
            "smooth_median_total_error": (
                "median_total_error",
                producer_replay.median_total_error,
            ),
            "smooth_mean_total_error": (
                "mean_total_error",
                producer_replay.mean_total_error,
            ),
        }
        for column, (diagnostic_name, result_value) in summary_fields.items():
            values = frame.get_column(column).unique().to_list()
            if (
                len(values) != 1
                or values[0] is None
                or abs(float(values[0]) - result_value) > 1e-12
            ):
                _fail(f"{trading_date}: {column} differs from recomputed smooth fit")
            if smooth.get(diagnostic_name) != result_value:
                _fail(f"{trading_date}: {diagnostic_name} diagnostics differ from smooth fit")
        grid = np.linspace(0.25, 2.0, 101, dtype=np.float64)
        previous_calls: np.ndarray[Any, Any] | None = None
        for expiry in smooth_pricer.expiries:
            calls = np.asarray(smooth_pricer(expiry, grid), dtype=np.float64)
            if np.max(np.diff(calls)) > 1.0e-7 or np.min(np.diff(calls, n=2)) < -1.0e-7:
                _fail(f"{trading_date}: reconstructed smooth calls violate strike shape")
            if previous_calls is not None:
                calendar_residual = max(
                    calendar_residual,
                    float(np.max(np.maximum(previous_calls - calls, 0.0))),
                )
            previous_calls = calls
        _validate_calendar_residual(
            calendar_residual,
            tolerance=smooth_calendar_tolerance,
            label=f"{trading_date}: smooth state",
        )
    else:
        for column in (
            "smooth_call_mid",
            "smooth_put_mid",
            "smooth_fit_error_half_spreads",
            "smooth_pcnt_fit",
            "smooth_median_error",
            "smooth_mean_error",
            "smooth_median_total_error",
            "smooth_mean_total_error",
        ):
            if frame.get_column(column).null_count() != frame.height:
                _fail(f"{trading_date}: zero-selected day has non-null {column}")

    for column in ("smooth_call_mid", "smooth_put_mid", "smooth_fit_error_half_spreads"):
        nulls = frame.get_column(column).is_null().to_numpy()
        if not np.array_equal(nulls, ~selected):
            _fail(f"{trading_date}: {column} nullability differs from selected population")

    point_two = np.isclose(floats("business_dte"), 0.2, rtol=0.0, atol=1.0e-6) & frame[
        "expiry_is_am"
    ].to_numpy().astype(bool)
    initial_retained = frame["initial_no_arb_expiry_retained"].to_numpy().astype(bool)
    smooth_indices = frame["smooth_input_expiry_index"].drop_nulls().unique()
    residuals = {
        "black_enrichment_vectors": black_vector_residual.center_residual,
        "black_enrichment_price_repricing": black_vector_residual.price_repricing,
        "black_enrichment_price_rounding_excess": (
            black_vector_residual.price_rounding_excess
        ),
        "black_enrichment_greek_center": black_vector_residual.greek_center,
        "black_enrichment_greek_rounding_excess": (
            black_vector_residual.greek_rounding_excess
        ),
        "black_enrichment_volatility_bound_excess": (
            black_vector_residual.volatility_bound_excess
        ),
        "black_fitted_vectors": fitted_black_vector_residual.center_residual,
        "black_fitted_price_repricing": fitted_black_vector_residual.price_repricing,
        "black_fitted_price_rounding_excess": (
            fitted_black_vector_residual.price_rounding_excess
        ),
        "black_fitted_greek_center": fitted_black_vector_residual.greek_center,
        "black_fitted_greek_rounding_excess": (
            fitted_black_vector_residual.greek_rounding_excess
        ),
        "black_fitted_volatility_bound_excess": (
            fitted_black_vector_residual.volatility_bound_excess
        ),
        "calendar": calendar_residual,
        "direct_call_unit": direct_call_residual,
        "direct_put_unit": direct_put_residual,
        "dlv_call_roundtrip": dlv_call_error,
        "dlv_canonical_density_roundtrip": dlv_canonical_density_error,
        "dlv_projection_call": dlv_projection_call_error,
        "dlv_atm_annual_iv": dlv_atm_iv_error,
        "dlv_conditioned_annual_iv": dlv_conditioned_iv_error,
        "dlv_conditioned_annual_iv_p99": dlv_conditioned_iv_p99,
        "dlv_minimum_bound_shortfall": dlv_minimum_bound_shortfall,
        "dlv_maximum_bound_excess": dlv_maximum_bound_excess,
        "dlv_density_roundtrip": dlv_density_error,
        "expiry_fit_quote_containment": expiry_fit_quote_containment,
        "expiry_fit_reconstruction": expiry_fit_reconstruction,
        "expiry_fit_contract_excess": expiry_fit_contract_excess,
        "expiry_fit_black_roundtrip": expiry_fit_black_roundtrip,
        "forward_over_spot": forward_residual,
        "normalized_strike": strike_residual,
        "previous_spot_over_spot": previous_spot_residual,
        "put_call_parity": put_call_residual,
        "initial_surface_fit_quote_band_violation": (
            initial_surface_fit_quote_band_violation
        ),
        "initial_surface_fit_reconstruction": initial_surface_fit_reconstruction,
        "initial_surface_fit_contract_excess": (
            initial_surface_fit_contract_excess
        ),
        "initial_surface_fit_black_roundtrip": (
            initial_surface_fit_black_roundtrip
        ),
        "initial_surface_calendar_acceptance_tolerance": (
            initial_surface_calendar_tolerance
        ),
        "smooth_expiry_reconstruction": smooth_expiry_reconstruction,
        "smooth_surface_fit_reconstruction": smooth_surface_fit_reconstruction,
        "smooth_fit_error_reconstruction": smooth_fit_error_reconstruction,
        "smooth_reconstruction": smooth_reconstruction,
        "smooth_canonical_market_fit_bridge": (
            smooth_canonical_market_fit_bridge
        ),
        "smooth_canonical_surface_fit_bridge": (
            smooth_canonical_surface_fit_bridge
        ),
        "smooth_calendar_acceptance_tolerance": smooth_calendar_tolerance,
        "spot_reference": spot_reference_residual,
        "spot_normalized_strike": spot_strike_residual,
        **{f"initial_expiry_{name}": value for name, value in initial_expiry_residuals.items()},
        **{
            f"initial_surface_{name}": value
            for name, value in initial_surface_state_residuals.items()
        },
        **{f"smooth_expiry_{name}": value for name, value in smooth_expiry_residuals.items()},
        **{f"dlv_{name}": value for name, value in dlv_state_residuals.items()},
        **{f"smooth_{name}": value for name, value in smooth_state_residuals.items()},
    }
    return PartitionStatistics(
        trading_date=trading_date,
        row_count=frame.height,
        input_expiries=frame["expiry_id"].n_unique(),
        initial_retained_expiries=frame.filter(pl.Series(initial_retained))[
            "expiry_id"
        ].n_unique(),
        mask_arb_free_rows=int(np.sum(pre_mask)),
        eligible_rows=int(np.sum(selected)),
        selected_expiries=frame.filter(pl.Series(selected))["expiry_id"].n_unique(),
        smooth_retained_expiries=len(smooth_indices),
        smooth_fit_mask_rows=int(np.sum(smooth_fit_mask)),
        smooth_model_mask_rows=int(np.sum(smooth_model_mask)),
        active_call_rows=int(np.sum(call_active)),
        active_put_rows=int(np.sum(put_active)),
        active_point_two_day_side_books=int(np.sum(call_active & point_two))
        + int(np.sum(put_active & point_two)),
        typed_empty=False,
        typed_empty_reason=None,
        rejection_counts=_reason_counts(frame),
        residual_maxima=residuals,
    )


@version("1.0", dependencies=[DatasetValidationSummary, PartitionStatistics])
def aggregate_statistics(
    partitions: tuple[PartitionStatistics, ...],
    *,
    requested_dates: int,
    exclusions: tuple[Mapping[str, Any], ...],
) -> DatasetValidationSummary:
    """Aggregate recomputed daily populations without conflating their meanings."""

    rejection_counts: Counter[str] = Counter()
    residuals: dict[str, float] = {}
    for partition in partitions:
        rejection_counts.update(partition.rejection_counts)
        for name, value in partition.residual_maxima.items():
            residuals[name] = max(residuals.get(name, 0.0), float(value))
    return DatasetValidationSummary(
        requested_dates=requested_dates,
        populated_dates=sum(not item.typed_empty for item in partitions),
        typed_empty_dates=sum(item.typed_empty for item in partitions),
        scheduled_exclusions=len(exclusions),
        row_count=sum(item.row_count for item in partitions),
        input_expiries=sum(item.input_expiries for item in partitions),
        initial_retained_expiries=sum(item.initial_retained_expiries for item in partitions),
        mask_arb_free_rows=sum(item.mask_arb_free_rows for item in partitions),
        eligible_rows=sum(item.eligible_rows for item in partitions),
        selected_expiries=sum(item.selected_expiries for item in partitions),
        smooth_retained_expiries=sum(item.smooth_retained_expiries for item in partitions),
        smooth_fit_mask_rows=sum(item.smooth_fit_mask_rows for item in partitions),
        smooth_model_mask_rows=sum(item.smooth_model_mask_rows for item in partitions),
        active_call_rows=sum(item.active_call_rows for item in partitions),
        active_put_rows=sum(item.active_put_rows for item in partitions),
        active_point_two_day_side_books=sum(
            item.active_point_two_day_side_books for item in partitions
        ),
        rejection_counts=dict(sorted(rejection_counts.items())),
        residual_maxima=dict(sorted(residuals.items())),
        exclusions=exclusions,
    )


@version("1.0")
def _safe_dataset_label(value: str) -> str:
    """Return the already validated portable Dataset namespace label."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError(f"Unsafe Dataset label: {value!r}")
    return value


@version("1.0")
def _raw_role_mapping(
    daily_artifacts: tuple[DailyArtifactPlan, ...],
) -> tuple[dict[str, Any], ...]:
    """Retain date/role multiplicity that a deduplicated upstream tuple cannot express."""

    records: list[dict[str, Any]] = []
    for item in daily_artifacts:
        records.append(
            {
                "trading_date": item.trading_date,
                "roles": {
                    "security": item.raw.security.as_dict(),
                    "options": item.raw.options.as_dict(),
                    "spot": item.raw.spot.as_dict(),
                    "previous_spot": item.raw.previous_spot.as_dict(),
                    "forwards": item.raw.forwards.as_dict(),
                    "zero_curve": item.raw.zero_curve.as_dict(),
                },
            }
        )
    return tuple(records)


@version("1.0")
def _dataset_configuration(
    config: InternalDatasetConfig,
    expected_dates: tuple[date, ...],
    raw_mapping_digest: str,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        canonical_value(
            {
                "dataset": {
                    "name": config.name,
                    "ticker": config.ticker,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "include_half_days": config.include_half_days,
                    "source": config.source,
                    "source_generation": config.source_generation,
                },
                "daily": config.daily,
                "surface": config.surface,
                "planned_dates": expected_dates,
                "raw_mapping_digest": raw_mapping_digest,
            }
        ),
    )


@version(
    "1.0",
    dependencies=[_raw_role_mapping, "_raw_artifact_records", "publish_raw_lineage_catalog"],
)
def raw_catalog_producer_version() -> str:
    """Return the resolved producer closure for Raw lineage catalogs."""

    return str(raw_catalog_producer_version.version)


@version(
    "1.0",
    dependencies=[
        validate_fitted_partition,
        aggregate_statistics,
        paired_day_producer_version,
        enriched_day_producer_version,
        no_arb_day_producer_version,
        filtered_day_producer_version,
        fitted_day_producer_version,
        "_publish_internal_dataset",
        "validate_internal_dataset",
    ],
)
def internal_dataset_producer_version() -> str:
    """Return the complete daily-to-Dataset content version closure."""

    return str(internal_dataset_producer_version.version)


@version(
    "1.0",
    dependencies=[
        InternalDatasetPlan,
        IntermediateBuildRequest,
        plan_daily_artifacts,
        _safe_dataset_label,
        _raw_role_mapping,
        _dataset_configuration,
        raw_catalog_producer_version,
        internal_dataset_producer_version,
        raw_content_producer_version,
        ArtifactSpec,
        DatasetDistribution,
    ],
)
def plan_internal_dataset(config: InternalDatasetConfig) -> InternalDatasetPlan:
    """Plan exact daily, Raw-catalog, and Dataset identities without touching storage."""

    request = IntermediateBuildRequest(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
        through="fitted-day",
        include_half_days=config.include_half_days,
        source=config.source,
        source_generation=config.source_generation,
    )
    daily_artifacts = plan_daily_artifacts(request, config.daily, config.surface)
    expected_dates = tuple(item.trading_date for item in daily_artifacts)
    if expected_dates != tuple(sorted(set(expected_dates))):
        raise ValueError("Dataset planning produced duplicate or unordered sessions")
    mapping = _raw_role_mapping(daily_artifacts)
    mapping_digest = canonical_sha256(mapping)
    raw_pointers = tuple(
        sorted({pointer for item in daily_artifacts for pointer in item.raw.all()})
    )
    label = _safe_dataset_label(config.name)
    catalog_configuration = {
        "name": config.name,
        "ticker": config.ticker,
        "source": config.source,
        "source_generation": config.source_generation,
        "planned_dates": expected_dates,
        "raw_mapping_digest": mapping_digest,
    }
    raw_catalog_spec = ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=f"widb/raw-lineage-catalog/name-{label}",
        configuration=catalog_configuration,
        schema_version=RAW_CATALOG_SCHEMA_VERSION,
        producer_version=raw_catalog_producer_version(),
        upstream=raw_pointers,
    )
    configuration = _dataset_configuration(config, expected_dates, mapping_digest)
    dataset_spec = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace=f"widb/internal/name-{label}",
        configuration=configuration,
        schema_version="+".join(
            (
                INTERNAL_DATASET_SCHEMA_VERSION,
                FITTED_SCHEMA_VERSION,
                PARTITION_CATALOG_SCHEMA,
                LINEAGE_SCHEMA,
                VALIDATION_SCHEMA,
                DATASET_SCHEMA_DOCUMENT,
                LEGACY_COMPARISON_SCHEMA,
            )
        ),
        producer_version=internal_dataset_producer_version(),
        upstream=(raw_catalog_spec.pointer,),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )
    plan_id = canonical_sha256(
        {
            "dataset": dataset_spec.pointer,
            "raw_catalog": raw_catalog_spec.pointer,
            "fitted": tuple(item.fitted.pointer for item in daily_artifacts),
        }
    )
    return InternalDatasetPlan(
        configuration=configuration,
        expected_dates=expected_dates,
        daily_artifacts=daily_artifacts,
        raw_mapping_digest=mapping_digest,
        raw_catalog_spec=raw_catalog_spec,
        dataset_spec=dataset_spec,
        plan_id=plan_id,
    )


@version("1.0", dependencies=[ArtifactPointer])
def _pointer_from_value(value: Any, label: str) -> ArtifactPointer:
    if not isinstance(value, Mapping):
        _fail(f"{label} is not an artifact pointer object")
    if set(value) != {"storage_class", "namespace", "artifact_id"}:
        _fail(f"{label} has missing or unexpected pointer fields")
    try:
        return ArtifactPointer(
            StorageClass(str(value["storage_class"])),
            str(value["namespace"]),
            str(value["artifact_id"]),
        )
    except ValueError as exc:
        raise DatasetValidationError(f"Invalid {label}: {exc}") from exc


@version("1.0")
def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"Dataset has no safe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return cast(dict[str, Any], value)


@version("1.0", dependencies=[InternalDatasetConfig])
def _config_from_published(value: Mapping[str, Any]) -> InternalDatasetConfig:
    required = {"dataset", "daily", "surface", "planned_dates", "raw_mapping_digest"}
    if set(value) != required:
        _fail("Published config.json has missing or unexpected sections")
    try:
        return InternalDatasetConfig.from_document(
            {name: value[name] for name in ("dataset", "daily", "surface")}
        )
    except ValueError as exc:
        raise DatasetValidationError(f"Published Dataset configuration is invalid: {exc}") from exc


@version("1.0")
def _safe_partition_ticker_label(ticker: str) -> str:
    """Return the same collision-resistant label used by daily artifacts."""

    label = _SAFE_PARTITION_LABEL.sub("-", ticker).strip("-._") or "ticker"
    if label != ticker:
        suffix = hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:8]
        label = f"{label}-{suffix}"
    return label


@version("1.0", dependencies=[_safe_partition_ticker_label])
def _partition_relative_path(ticker: str, trading_date: date) -> str:
    ticker_label = _safe_partition_ticker_label(ticker)
    return (
        f"data/ticker={ticker_label}/year={trading_date.year}/"
        f"date={trading_date.isoformat()}/part-00000.parquet"
    )


_DAY_FIELDS = frozenset(
    {
        "ticker",
        "trading_date",
        "spot",
        "previous_spot",
        "spot_return",
        "is_half_day",
        "source_secid",
        "stage_version",
        "configuration_id",
        "raw_options_artifact_id",
        "normalization_diagnostics_json",
        "enrichment_diagnostics_json",
        "no_arb_configuration_id",
        "surface_min_half_spread",
        "no_arb_diagnostics_json",
        "filter_diagnostics_json",
        "options_filter_configuration_id",
        "spot_reference",
        "normalized_previous_spot",
        "smoothness",
        "smooth_pcnt_fit",
        "smooth_median_error",
        "smooth_mean_error",
        "smooth_median_total_error",
        "smooth_mean_total_error",
        "smooth_diagnostics_json",
        "smooth_sanos_configuration_id",
        "discrete_dlv_diagnostics_json",
        "discrete_dlv_configuration_id",
    }
)
_EXPIRY_FIELDS = frozenset(
    {
        "expiry_id",
        "is_american",
        "expiry_is_am",
        "vendor_forward",
        "discount_factor",
        "expiry_date",
        "business_dte",
        "calendar_dte",
        "time_to_expiry_years",
        "implied_forward",
        "initial_no_arb_expiry_retained",
        "initial_no_arb_input_expiry_index",
        "initial_model_variance",
        "initial_expiry_model_strikes",
        "initial_expiry_model_density",
        "initial_surface_model_strikes",
        "initial_surface_model_density",
        "initial_x_to_surface_indices",
        "smooth_input_expiry_index",
        "smooth_model_variance",
        "smooth_expiry_model_strikes",
        "smooth_expiry_model_density",
        "smooth_surface_model_strikes",
        "smooth_surface_model_density",
        "smooth_x_to_surface_indices",
    }
)
_INITIAL_STATE_FIELDS = frozenset(
    name
    for name in FITTED_SCHEMA
    if name.startswith("fit1_")
    or name.startswith("fitX_")
    or name.startswith("pure_fit1_")
    or name.startswith("pure_fitX_")
    or name.startswith("initial_model_")
    or name.startswith("initial_expiry_model_")
    or name.startswith("initial_surface_model_")
    or name == "initial_no_arb_input_expiry_index"
)
_SMOOTH_STATE_FIELDS = frozenset(
    name
    for name in FITTED_SCHEMA
    if name.startswith("smooth_expiry_")
    or name.startswith("smooth_surface_")
    or name in {"smooth_input_expiry_index", "smooth_model_variance"}
)
_DLV_STATE_FIELDS = frozenset(
    name
    for name in FITTED_SCHEMA
    if name.startswith("discrete_dlv_")
    and name
    not in {
        "discrete_dlv_anchor",
        "discrete_dlv_diagnostics_json",
        "discrete_dlv_configuration_id",
    }
)
_SELECTED_ONLY_FIELDS = frozenset(
    {
        "pure_market_call_bid",
        "pure_market_call_ask",
        "pure_market_put_bid",
        "pure_market_put_ask",
        "smooth_call_mid",
        "smooth_put_mid",
        "smooth_fit_error_half_spreads",
    }
)
_SMOOTH_SUMMARY_FIELDS = frozenset(
    {
        "smooth_pcnt_fit",
        "smooth_median_error",
        "smooth_mean_error",
        "smooth_median_total_error",
        "smooth_mean_total_error",
    }
)


@version("1.1")
def _field_units(name: str) -> tuple[str, str]:
    if name in {"trading_date", "expiry_date"}:
        return "calendar date", "none"
    if name == "business_dte":
        return "NYSE business days", "AM expiries retain overnight_variance_fraction"
    if name == "calendar_dte":
        return "calendar days", "expiry_date - trading_date"
    if name == "time_to_expiry_years":
        return "business years", "business_dte / 252"
    if name == "spot_return":
        return "dimensionless simple return", "Float32 spot / previous_spot - 1"
    if name in {
        "spot",
        "previous_spot",
        "strike",
        "vendor_forward",
        "implied_forward",
        "spot_reference",
    }:
        return "absolute underlying/price units", "none"
    if name.startswith("market_") and not (
        name.endswith("log_volume") or name.endswith("log_open_interest")
    ):
        return "absolute option-price units", "none"
    if name in {"synthetic_call_ask", "synthetic_call_bid"}:
        return "absolute option-price units", "put/call-parity synthesized call"
    if name.endswith("log_volume") or name.endswith("log_open_interest") or name.startswith(
        "combined_log_"
    ):
        return "dimensionless", "log10(1 + vendor count)"
    if name in {"normalized_moneyness", "atm_normalized_moneyness"}:
        return "dimensionless NMS", "versioned log(K/F) time/volatility scaling"
    if name == "normalized_strike":
        return "dimensionless", "strike / implied_forward (K/F)"
    if name.startswith("spot_normalized_"):
        return "dimensionless", "named absolute field / spot"
    if name == "normalized_previous_spot":
        return "dimensionless", "previous_spot / spot"
    if name == "discount_factor":
        return "dimensionless", "ACT/365 continuously compounded zero curve"
    if name.endswith("implied_volatility") or name in {"fit1_volatility", "fitX_volatility"}:
        return "annualized volatility", "Black implied volatility"
    if (
        name in {"pure_ask_price", "pure_bid_price", "pure_fit1_price", "pure_fitX_price"}
        or name.startswith("pure_market_")
        or name
        in {
            "smooth_expiry_fit_price",
            "smooth_surface_fit_price",
            "smooth_call_mid",
            "smooth_put_mid",
        }
    ):
        return "discounted-forward-normalized price", "option price / (DF * F)"
    if name.endswith("_delta") or name.endswith("_dk"):
        return "dimensionless normalized-price derivative", "Black derivative in K/F units"
    if name.endswith("_gamma"):
        return "inverse normalized-underlying units", "Black gamma in K/F geometry"
    if name.endswith("_vega"):
        return "normalized price per unit volatility", "Black vega in K/F geometry"
    if name.endswith("_theta"):
        return "normalized price per business year", "Black theta in K/F geometry"
    if name in {
        "initial_model_variance",
        "smooth_model_variance",
        "discrete_dlv_input_total_variance",
    }:
        return "dimensionless total variance", "ATM variance coordinate used by SANOS"
    if name == "discrete_dlv_time_increment_years":
        return "business years", "increment from the preceding retained model expiry"
    if name == "discrete_dlv_atm_annual_volatility":
        return "annualized volatility", "source smooth-SANOS ATM volatility"
    if name == "discrete_dlv_nms":
        return "dimensionless NMS", "canonical source-smooth ATM/NMS target geometry"
    if name.endswith("_model_strikes"):
        return "dimensionless support", "K/F support nodes"
    if name.endswith("_model_density"):
        return "probability mass", "discrete martingale mass on corresponding support"
    if name == "discrete_dlv_squares":
        return "annualized variance rate (1/year)", "squared DLV with zero boundaries"
    if name in {"discrete_dlv_target_calls", "discrete_dlv_fitted_calls"}:
        return "discounted-forward-normalized price", "call prices on discrete_dlv_nms"
    if name.endswith("_indices") or name.endswith("_index"):
        return "integer index", "zero-based interpreting geometry"
    if name == "surface_min_half_spread":
        return "discounted-forward-normalized price", "minimum fit-error half-spread"
    if name == "smoothness":
        return "dimensionless variance fraction", "SANOS smoothing fraction"
    if name == "smooth_fit_error_half_spreads":
        return "half-spread units", "distance outside quote divided by effective half-spread"
    if name in _SMOOTH_SUMMARY_FIELDS:
        return "dimensionless fit statistic", "summary of smooth_fit_error_half_spreads"
    if (
        name.endswith("_json")
        or name.endswith("_id")
        or name.endswith("_eligible")
        or name.endswith("_active")
        or name.endswith("_available")
        or name.endswith("_retained")
        or name.endswith("_mask")
        or name.endswith("_reason")
        or name
        in {
            "ticker",
            "is_half_day",
            "is_american",
            "expiry_is_am",
            "is_call",
            "mask_arb_free",
            "source_secid",
            "stage_version",
            "configuration_id",
            "raw_options_artifact_id",
            "call_symbol",
            "put_symbol",
            "call_option_id",
            "put_option_id",
            "discrete_dlv_anchor",
        }
    ):
        return "not applicable", "none"
    raise AssertionError(f"No immutable unit contract for Dataset field {name!r}")


@version("1.1")
def _field_presence(name: str) -> tuple[bool, str]:
    if name in _INITIAL_STATE_FIELDS:
        return True, "present exactly when initial_no_arb_input_expiry_index is present"
    if name in _SMOOTH_STATE_FIELDS:
        return True, "present exactly when smooth_input_expiry_index is present"
    if name in _DLV_STATE_FIELDS:
        return True, "present exactly on the one deterministic DLV anchor row per expiry"
    if name in _SELECTED_ONLY_FIELDS:
        return True, "present exactly when options_filter_eligible is true"
    if name in _SMOOTH_SUMMARY_FIELDS:
        return True, "one daily value when any row is selected; otherwise null"
    if name in {
        "initial_no_arb_rejection_reason",
        "options_filter_rejection_reason",
        "smooth_rejection_reason",
    }:
        return True, "null exactly for rows accepted by the corresponding stage"
    if name == "initial_x_to_surface_indices" or name == "smooth_x_to_surface_indices":
        return True, "null denotes the identity support mapping; otherwise an explicit list"
    if name == "vendor_forward":
        return True, "nullable when no matching vendor forward row exists"
    return False, "always present on populated partitions"


@version("3.2", dependencies=[_field_units, _field_presence])
def _schema_document() -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for name, dtype in FITTED_SCHEMA.items():
        units, normalization = _field_units(name)
        nullable, presence = _field_presence(name)
        cardinality = (
            "day value repeated on every row"
            if name in _DAY_FIELDS
            else (
                "expiry value/state repeated on its applicable expiry rows"
                if name in _EXPIRY_FIELDS
                else (
                    "one deterministic anchor row per retained DLV expiry"
                    if name in _DLV_STATE_FIELDS or name == "discrete_dlv_anchor"
                    else "paired call/put strike-book row"
                )
            )
        )
        fields.append(
            {
                "name": name,
                "dtype": str(dtype),
                "cardinality": cardinality,
                "nullable": nullable,
                "presence": presence,
                "units": units,
                "normalization": normalization,
            }
        )
    return {
        "schema": DATASET_SCHEMA_DOCUMENT,
        "partition_schema_version": FITTED_SCHEMA_VERSION,
        "row_contract": "one paired call/put listed-strike audit row",
        "session_half_day_field": "is_half_day",
        "expiry_settlement_field": "expiry_is_am",
        "list_contract": (
            "Initial and smooth list fields repeat complete per-expiry model state on "
            "applicable rows. Discrete-DLV list fields occur only on the first canonical "
            "row of each retained expiry. Density values are probability masses, not "
            "continuous density ordinates."
        ),
        "dlv_contract": {
            "source": "completed smooth SANOS surface",
            "direct_vendor_book_fit": False,
            "initial_no_arb_audit_state": False,
            "retained_expiries": "identical to smooth SANOS",
            "configuration_path": "config.json:surface.discrete_dlv",
            "projection_domain_nms": "configured by discrete_dlv.nms_range",
            "inner_nodes": "configured by discrete_dlv.inner_nodes",
            "support_boundaries": (
                "configured by discrete_dlv.lower_support_multiplier and "
                "discrete_dlv.upper_support_multiplier"
            ),
            "acceptance_tolerances": "configured by discrete_dlv.tolerances",
            "equivalence_scope": "retained expiries within inclusive NMS domain",
            "arbitrary_time_equivalence": False,
        },
        "fields": fields,
    }


@version("1.1")
def _readme(config: InternalDatasetConfig, plan: InternalDatasetPlan) -> str:
    return f"""# {config.name}

Immutable internal WIDB-to-SANOS Dataset candidate for `{config.ticker}` from
{config.start_date.isoformat()} through {config.end_date.isoformat()}. It contains one
losslessly copied fitted-day Polars/Parquet frame per planned NYSE session, including
all audit rows, no-arbitrage/filter masks, smooth SANOS state, its canonical
daily-front discrete-DLV projection, fitted values, and diagnostics. Half days are
{'included' if config.include_half_days else 'excluded'}.

The source stream is WRDS OptionMetrics/IvyDB security identity, option quotes,
underlying closes, forward prices, and zero curves. Prices named `pure_*` and smooth
prices are in discounted-forward-normalized units; `normalized_strike` is K/F.
Absolute spot and source quote fields make this an internal Dataset, not a public
redistribution artifact.

The legacy 891-date quant_data artifact is a read-only migration/parity oracle only;
it is neither a build input nor a runtime dependency. This publication becomes the
canonical SANOS reference Dataset only after the fresh-WRDS numerical comparison and
dual artifact review are complete and its immutable promotion evidence records
`canonical_promotion_ready=true`. Unexplained differences remain blocking review
evidence rather than hidden exclusions.

Build and validate with:

```console
sanos dataset build --config configs/widb-spx-reference.json
sanos dataset validate --config configs/widb-spx-reference.json
sanos dataset validate --config configs/widb-spx-reference.json --deep-lineage
```

Default loading and validation are self-contained and do not require Raw or
Intermediate artifacts. Deep lineage validation deliberately does. Dataset identity:
`{plan.dataset_spec.artifact_id}`.
"""


@version("1.2")
def _legacy_comparison_document(config: InternalDatasetConfig) -> dict[str, Any]:
    return {
        "schema": LEGACY_COMPARISON_SCHEMA,
        "role": "migration_parity_oracle_only",
        "legacy_reference": "quant_data 891-populated-date fitted collection",
        "legacy_is_build_input": False,
        "legacy_is_runtime_dependency": False,
        "sanos_dataset_is_reference_after_promotion": True,
        "status": "structural_only_pending_fresh_wrds_build_and_guarded_comparison",
        "canonical_promotion_ready": False,
        "configured_start_date": config.start_date,
        "configured_end_date": config.end_date,
        "oracle_inventory": {
            "path": "reviews/phase-1/reference-artifacts.json",
            "sha256": "59ebb8e75143e5d9fc79dbc52f674177c9bae24f3ec7076a9ada2554d15876c6",
            "quant_data_revision": "de15be96d2f283b6c31e60d3ea47325a0c093ff0",
        },
        "smooth_daily_cache_oracle": {
            "catalog_path": "reviews/phase-1/reference-market-catalog.json",
            "catalog_sha256": "9b68a9638f273271ff4dbde6365c75ac9fd749993f9284b42428682339d516b4",
            "collection_sha256": "f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280",
            "physical_files": 892,
            "populated_dates": 891,
            "expiries": 26732,
            "strike_nodes": 2414341,
            "active_side_books": 4651416,
            "active_point_two_day_side_books": 8090,
        },
        "dysanos_daily_front_experiment_oracle": {
            "completion_sha256": "bffe601161d162e316910a638fb4bc48ef84a73011cddd10538fc5270a8613c7",
            "dates": 891,
            "option_books": 3566258,
            "role": "downstream_experiment_oracle_not_internal_dataset_population",
        },
        "daily25_sdlv_experiment_oracle": {
            "sha256": "79686e911ac256e742ef67ecdf6a046df6b5ad5782e1d13939330894f916c2a6",
            "dates": 891,
            "option_books": 2192458,
            "role": "downstream_experiment_oracle_not_internal_dataset_population",
        },
        "notes": (
            "Full per-date numerical comparison is a guarded post-build acceptance gate; "
            "legacy gaps are not copied into the SANOS schedule."
        ),
    }


@version("1.0")
def _source_revision() -> str:
    """Return exact build provenance without accepting an uncommitted checkout."""

    explicit = os.environ.get("SANOS_SOURCE_REVISION")
    if explicit is not None:
        if _HEX_40.fullmatch(explicit) is None:
            raise RuntimeError("SANOS_SOURCE_REVISION must be an exact lowercase 40-hex commit")
        return explicit
    root = Path(__file__).resolve().parents[3]
    git_dir = root / ".git"
    if not git_dir.exists():
        git_dir = root / ".git-real"
    if not git_dir.exists():
        packaged = packaged_source_revision()
        if packaged is not None:
            return packaged
        raise RuntimeError(
            "Dataset publication requires packaged provenance, SANOS_SOURCE_REVISION, "
            "or an identifiable Git checkout"
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot inspect SANOS Git provenance: {exc}") from exc
    if status.stdout.strip():
        raise RuntimeError(
            "Dataset publication requires a clean Git checkout; commit reviewed code first"
        )
    try:
        revision = subprocess.run(
            ["git", f"--git-dir={git_dir}", f"--work-tree={root}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot resolve SANOS Git revision: {exc}") from exc
    if _HEX_40.fullmatch(revision) is None:
        raise RuntimeError("Git did not return an exact lowercase 40-hex source revision")
    return revision


@version("1.0", dependencies=[_sha256_file])
def _raw_artifact_records(
    store: ArtifactStore,
    plan: InternalDatasetPlan,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for pointer in plan.raw_catalog_spec.upstream:
        artifact = store.validate(pointer)
        configuration = artifact.manifest["configuration"]
        if not isinstance(configuration, Mapping):
            _fail(f"Raw artifact has malformed configuration: {pointer.namespace}")
        records.append(
            {
                "pointer": pointer.as_dict(),
                "source": configuration.get("source"),
                "table": configuration.get("table"),
                "ticker": configuration.get("ticker"),
                "observation_date": configuration.get("observation_date"),
                "source_generation": artifact.manifest["source_generation"],
                "manifest_sha256": _sha256_file(artifact.path / "manifest.json"),
                "payloads": artifact.manifest["files"],
            }
        )
    return tuple(records)


@version("1.0", dependencies=[_raw_artifact_records, _raw_role_mapping])
def publish_raw_lineage_catalog(
    store: ArtifactStore,
    plan: InternalDatasetPlan,
) -> ValidatedArtifact:
    """Publish or validate the exact Raw closure used by a Dataset assembly."""

    reuse = store.resolve(plan.raw_catalog_spec)
    if reuse is not None:
        return store.validate(reuse.artifact.pointer, expected=plan.raw_catalog_spec, deep=True)
    records = _raw_artifact_records(store, plan)
    mapping = _raw_role_mapping(plan.daily_artifacts)
    if canonical_sha256(mapping) != plan.raw_mapping_digest:
        _fail("Planned Raw role mapping digest changed during publication")
    document = {
        "schema": RAW_CATALOG_SCHEMA_VERSION,
        "raw_mapping_digest": plan.raw_mapping_digest,
        "date_roles": mapping,
        "artifacts": records,
    }
    builder = store.writer(
        plan.raw_catalog_spec,
        metadata={
            "planned_date_count": len(plan.expected_dates),
            "raw_artifact_count": len(records),
            "raw_mapping_digest": plan.raw_mapping_digest,
        },
    )
    try:
        with builder as staging:
            staging.write_json("raw-catalog.json", document)
            staging.write_text(
                "README.md",
                "# WIDB Raw lineage catalog\n\n"
                "Rebuildable Intermediate catalog committing the exact date/role Raw "
                "mapping and complete deduplicated Raw artifact closure for one immutable "
                "SANOS Dataset. It may be removed after publication; shallow Dataset use "
                "does not depend on it.\n",
            )
    except ArtifactExistsError:
        return store.validate(
            plan.raw_catalog_spec.pointer,
            expected=plan.raw_catalog_spec,
            deep=True,
        )
    return builder.artifact


@version("1.0")
@dataclass(frozen=True, slots=True)
class InternalDataset:
    """Self-contained reader over an exact immutable partition catalog."""

    artifact: ValidatedArtifact
    config: InternalDatasetConfig
    partitions: tuple[Mapping[str, Any], ...]

    def scan(self) -> pl.LazyFrame:
        """Scan partitions in catalog order without Hive path inference."""

        scans = [
            pl.scan_parquet(
                self.artifact.path / str(record["path"]),
                hive_partitioning=False,
            )
            for record in self.partitions
        ]
        if not scans:
            raise DatasetValidationError("Dataset partition catalog is empty")
        return pl.concat(scans, how="vertical")

    def load_day(self, trading_date: date) -> pl.DataFrame:
        """Load one exact daily partition without consulting upstream artifacts."""

        matches = [
            record
            for record in self.partitions
            if record.get("trading_date") == trading_date.isoformat()
        ]
        if len(matches) != 1:
            raise KeyError(f"Dataset has no unique partition for {trading_date.isoformat()}")
        return pl.read_parquet(
            self.artifact.path / str(matches[0]["path"]),
            hive_partitioning=False,
        )


@version(
    "1.0",
    dependencies=[_read_json_file, _config_from_published, InternalDataset],
)
def _open_validated_internal_dataset(artifact: ValidatedArtifact) -> InternalDataset:
    distribution = artifact.manifest.get("distribution")
    if not isinstance(distribution, Mapping) or distribution.get("status") != "internal":
        _fail("Requested artifact is not an internal Dataset")
    config_value = _read_json_file(artifact.path / "config.json", "config.json")
    if canonical_json(config_value) != canonical_json(artifact.manifest["configuration"]):
        _fail("config.json differs from the immutable manifest configuration")
    config = _config_from_published(config_value)
    catalog = _read_json_file(artifact.path / "partitions.json", "partitions.json")
    if catalog.get("schema") != PARTITION_CATALOG_SCHEMA:
        _fail("Unsupported Dataset partition catalog schema")
    values = catalog.get("partitions")
    if not isinstance(values, list):
        _fail("Dataset partition catalog has no partitions list")
    partitions: list[Mapping[str, Any]] = []
    for index, record in enumerate(values):
        if not isinstance(record, Mapping):
            _fail(f"Partition catalog record {index} is not an object")
        relative = record.get("path")
        if not isinstance(relative, str) or _PARTITION_PATH.fullmatch(relative) is None:
            _fail(f"Partition catalog record {index} has an unsafe path")
        path = artifact.path / PurePosixPath(relative)
        if path.is_symlink() or not path.is_file():
            _fail(f"Partition catalog payload is missing or unsafe: {relative}")
        partitions.append(record)
    return InternalDataset(artifact=artifact, config=config, partitions=tuple(partitions))


@version("1.0", dependencies=[InternalDataset, _open_validated_internal_dataset])
def open_internal_dataset(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
) -> InternalDataset:
    """Open a Dataset shallowly; Raw and Intermediate storage is never consulted."""

    validated = store.validate(artifact) if isinstance(artifact, ArtifactPointer) else artifact
    return _open_validated_internal_dataset(validated)


@version("1.0")
def _partition_record_statistics(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("statistics")
    if not isinstance(value, Mapping):
        _fail("Partition record has no statistics object")
    return cast(dict[str, Any], canonical_value(value))


@version("1.0", dependencies=[half_days])
def _scheduled_exclusions(config: InternalDatasetConfig) -> tuple[dict[str, Any], ...]:
    if config.include_half_days:
        return ()
    return tuple(
        {"trading_date": item, "reason": "configured_half_day_exclusion"}
        for item in half_days(config.start_date, config.end_date)
    )


@version(
    "3.0",
    dependencies=[
        _fail,
        _legacy_comparison_document,
        _open_validated_internal_dataset,
        _partition_record_statistics,
        _pointer_from_value,
        _read_json_file,
        validate_fitted_partition,
        aggregate_statistics,
        _scheduled_exclusions,
        _safe_partition_ticker_label,
        _schema_document,
        _sha256_file,
        bad_data_policy,
        canonical_sha256,
        canonical_value,
        is_half_day,
    ],
)
def validate_internal_dataset(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
    *,
    semantic: bool = True,
    deep_lineage: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    progress_interval_seconds: float = 30.0,
) -> DatasetValidationSummary | None:
    """Validate an immutable Dataset shallowly by default and optionally its lineage."""

    if not 0.0 < progress_interval_seconds <= 300.0:
        raise ValueError("progress_interval_seconds must lie in (0, 300]")
    validation_started = time.monotonic()
    _emit_progress(
        progress,
        "dataset_validation_started",
        semantic=semantic,
        deep_lineage=deep_lineage,
    )

    pointer = artifact.pointer if isinstance(artifact, ValidatedArtifact) else artifact
    validated = store.validate(pointer)
    dataset = _open_validated_internal_dataset(validated)
    configuration = _read_json_file(validated.path / "config.json", "config.json")
    planned_values = configuration.get("planned_dates")
    if not isinstance(planned_values, list) or any(
        not isinstance(value, str) for value in planned_values
    ):
        _fail("Published configuration has an invalid planned date sequence")
    expected_dates = tuple(date.fromisoformat(value) for value in planned_values)
    observed_dates = tuple(
        date.fromisoformat(str(record.get("trading_date"))) for record in dataset.partitions
    )
    if observed_dates != expected_dates or observed_dates != tuple(sorted(set(observed_dates))):
        _fail("Partition dates do not exactly match the ordered planned date sequence")
    schema = _read_json_file(validated.path / "schema.json", "schema.json")
    if schema != _schema_document():
        _fail("schema.json differs from the current immutable Dataset schema contract")
    lineage = _read_json_file(validated.path / "lineage.json", "lineage.json")
    if lineage.get("schema") != LINEAGE_SCHEMA:
        _fail("Unsupported Dataset lineage schema")
    if lineage.get("raw_mapping_digest") != configuration.get("raw_mapping_digest"):
        _fail("Dataset lineage Raw mapping digest differs from configuration")
    raw_catalog = _pointer_from_value(lineage.get("raw_catalog"), "lineage Raw catalog")
    manifest_upstream = validated.manifest.get("upstream")
    if not isinstance(manifest_upstream, list) or len(manifest_upstream) != 1:
        _fail("Internal Dataset must record exactly one Raw catalog upstream")
    if raw_catalog != _pointer_from_value(manifest_upstream[0], "manifest Raw catalog"):
        _fail("Lineage Raw catalog differs from the manifest upstream")
    fitted_values = lineage.get("daily_chains")
    if not isinstance(fitted_values, list) or len(fitted_values) != len(expected_dates):
        _fail("Dataset lineage has an incomplete daily chain catalog")
    role_mapping: list[dict[str, Any]] = []
    chain_raw_roles: list[dict[str, ArtifactPointer]] = []
    chain_fitted: list[ArtifactPointer] = []
    chain_stages: list[dict[str, ArtifactPointer]] = []
    expected_chain_fields = {
        "trading_date",
        "raw_roles",
        "paired",
        "enriched",
        "no_arb",
        "filtered",
        "fitted",
    }
    expected_roles = {
        "security",
        "options",
        "spot",
        "previous_spot",
        "forwards",
        "zero_curve",
    }
    for index, (expected_date, chain) in enumerate(
        zip(expected_dates, fitted_values, strict=True)
    ):
        if not isinstance(chain, Mapping) or set(chain) != expected_chain_fields:
            _fail(f"Dataset lineage chain {index} has missing or unexpected fields")
        if chain.get("trading_date") != expected_date.isoformat():
            _fail(f"Dataset lineage chain {index} has the wrong date")
        raw_roles = chain.get("raw_roles")
        if not isinstance(raw_roles, Mapping) or set(raw_roles) != expected_roles:
            _fail(f"Dataset lineage chain {index} has invalid Raw roles")
        parsed_roles = {
            role: _pointer_from_value(raw_roles[role], f"daily chain {index} {role}")
            for role in sorted(expected_roles)
        }
        if any(pointer.storage_class is not StorageClass.RAW for pointer in parsed_roles.values()):
            _fail(f"Dataset lineage chain {index} contains a non-Raw role pointer")
        normalized_roles = {
            role: pointer.as_dict() for role, pointer in parsed_roles.items()
        }
        chain_raw_roles.append(parsed_roles)
        role_mapping.append(
            {"trading_date": expected_date, "roles": normalized_roles}
        )
        parsed_stages: dict[str, ArtifactPointer] = {}
        for stage in ("paired", "enriched", "no_arb", "filtered", "fitted"):
            stage_pointer = _pointer_from_value(
                chain.get(stage), f"daily chain {index} {stage}"
            )
            if stage_pointer.storage_class is not StorageClass.INTERMEDIATE:
                _fail(f"Dataset lineage chain {index} {stage} is not Intermediate")
            parsed_stages[stage] = stage_pointer
        chain_stages.append(parsed_stages)
        chain_fitted.append(parsed_stages["fitted"])
    if canonical_sha256(role_mapping) != configuration.get("raw_mapping_digest"):
        _fail("Dataset lineage date/role Raw mapping does not match its digest")
    expected_raw_pointers = tuple(
        sorted({pointer for roles in chain_raw_roles for pointer in roles.values()})
    )
    metadata = validated.manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("Dataset manifest metadata is malformed")
    expected_metadata_fields = {
        "source_revision",
        "dataset_schema_version",
        "partition_schema_version",
        "producer_version",
        "configuration_id",
        "raw_catalog",
        "raw_mapping_digest",
        "raw_artifacts",
        "fitted_artifacts",
        "planned_date_count",
        "partition_count",
        "populated_date_count",
        "typed_empty_date_count",
        "scheduled_exclusion_count",
        "scheduled_exclusions",
        "typed_empty_partitions",
        "bad_data_policy_version",
        "row_count",
        "eligible_row_count",
        "active_side_book_count",
        "validation_outcome",
        "canonical_promotion_ready",
        "evidence_checksums",
    }
    if set(metadata) != expected_metadata_fields:
        _fail("Dataset manifest metadata has missing or unexpected fields")
    if not isinstance(metadata.get("source_revision"), str) or _HEX_40.fullmatch(
        str(metadata["source_revision"])
    ) is None:
        _fail("Dataset manifest has no exact source revision")
    if metadata.get("dataset_schema_version") != INTERNAL_DATASET_SCHEMA_VERSION:
        _fail("Dataset manifest schema version differs from the payload contract")
    if metadata.get("partition_schema_version") != FITTED_SCHEMA_VERSION:
        _fail("Dataset manifest partition schema version differs")
    if metadata.get("producer_version") != validated.manifest.get("producer_version"):
        _fail("Dataset manifest metadata producer version differs")
    if metadata.get("configuration_id") != validated.manifest.get("configuration_id"):
        _fail("Dataset manifest metadata configuration identity differs")
    if _pointer_from_value(metadata.get("raw_catalog"), "metadata Raw catalog") != raw_catalog:
        _fail("Dataset manifest Raw catalog differs from lineage")
    if metadata.get("raw_mapping_digest") != configuration.get("raw_mapping_digest"):
        _fail("Dataset manifest Raw mapping digest differs")
    raw_metadata = metadata.get("raw_artifacts")
    fitted_metadata = metadata.get("fitted_artifacts")
    if not isinstance(raw_metadata, list) or tuple(
        _pointer_from_value(value, "metadata Raw artifact") for value in raw_metadata
    ) != expected_raw_pointers:
        _fail("Dataset manifest flattened Raw identities differ from daily lineage")
    if not isinstance(fitted_metadata, list) or tuple(
        _pointer_from_value(value, "metadata fitted artifact") for value in fitted_metadata
    ) != tuple(chain_fitted):
        _fail("Dataset manifest fitted identities differ from daily lineage")
    expected_exclusions = canonical_value(_scheduled_exclusions(dataset.config))
    if metadata.get("scheduled_exclusions") != expected_exclusions:
        _fail("Dataset manifest scheduled exclusions differ from configuration")
    if metadata.get("bad_data_policy_version") != str(bad_data_policy.version):
        _fail("Dataset manifest bad-data policy version differs")
    if metadata.get("canonical_promotion_ready") is not False:
        _fail("Phase 7 Dataset must remain an unpromoted reference candidate")
    if metadata.get("planned_date_count") != len(expected_dates) or metadata.get(
        "partition_count"
    ) != len(dataset.partitions):
        _fail("Dataset manifest planned/partition counts differ")

    if deep_lineage:
        deep_started = time.monotonic()
        deep_last_progress = deep_started
        deep_total = len(expected_raw_pointers) + len(chain_stages)
        deep_completed = 0
        catalog_artifact = store.validate(raw_catalog)
        catalog_upstream_values = catalog_artifact.manifest.get("upstream")
        if not isinstance(catalog_upstream_values, list):
            _fail("Raw catalog manifest has no upstream list")
        catalog_upstream = tuple(
            _pointer_from_value(value, "Raw catalog manifest upstream")
            for value in catalog_upstream_values
        )
        if catalog_upstream != expected_raw_pointers:
            _fail("Raw catalog upstream does not equal flattened daily Raw lineage")
        catalog_document = _read_json_file(
            catalog_artifact.path / "raw-catalog.json", "raw-catalog.json"
        )
        if set(catalog_document) != {
            "schema",
            "raw_mapping_digest",
            "date_roles",
            "artifacts",
        } or catalog_document.get("schema") != RAW_CATALOG_SCHEMA_VERSION:
            _fail("Raw catalog payload has an invalid schema")
        if catalog_document.get("raw_mapping_digest") != configuration.get(
            "raw_mapping_digest"
        ) or catalog_document.get("date_roles") != canonical_value(role_mapping):
            _fail("Raw catalog date/role mapping differs from Dataset lineage")
        raw_records = catalog_document.get("artifacts")
        if not isinstance(raw_records, list) or len(raw_records) != len(expected_raw_pointers):
            _fail("Raw catalog artifact records are incomplete")
        observed_record_pointers: list[ArtifactPointer] = []
        for record in raw_records:
            if not isinstance(record, Mapping) or set(record) != {
                "pointer",
                "source",
                "table",
                "ticker",
                "observation_date",
                "source_generation",
                "manifest_sha256",
                "payloads",
            }:
                _fail("Raw catalog contains a malformed artifact record")
            raw_pointer = _pointer_from_value(record.get("pointer"), "Raw catalog record")
            if raw_pointer.storage_class is not StorageClass.RAW:
                _fail("Raw catalog contains a non-Raw artifact record")
            raw_artifact = store.validate(raw_pointer)
            if record.get("manifest_sha256") != _sha256_file(
                raw_artifact.path / "manifest.json"
            ) or record.get("payloads") != raw_artifact.manifest.get("files"):
                _fail("Raw catalog record checksum evidence differs from retained Raw")
            raw_configuration = raw_artifact.manifest.get("configuration")
            if not isinstance(raw_configuration, Mapping) or any(
                record.get(name) != raw_configuration.get(name)
                for name in ("source", "table", "ticker", "observation_date")
            ) or record.get("source_generation") != raw_artifact.manifest.get(
                "source_generation"
            ):
                _fail("Raw catalog record semantics differ from retained Raw manifest")
            observed_record_pointers.append(raw_pointer)
            deep_completed += 1
            now = time.monotonic()
            if (
                deep_completed == deep_total
                or now - deep_last_progress >= progress_interval_seconds
            ):
                elapsed = now - deep_started
                rate = deep_completed / elapsed if elapsed > 0.0 else 0.0
                _emit_progress(
                    progress,
                    "dataset_lineage_progress",
                    elapsed_seconds=elapsed,
                    completed=deep_completed,
                    total=deep_total,
                    failed=0,
                    estimated_remaining_seconds=(
                        (deep_total - deep_completed) / rate if rate > 0.0 else None
                    ),
                )
                deep_last_progress = now
        if tuple(observed_record_pointers) != expected_raw_pointers:
            _fail("Raw catalog records differ from its manifest upstream")

        for index, (roles, stages) in enumerate(
            zip(chain_raw_roles, chain_stages, strict=True)
        ):
            validated_stages = {
                name: store.validate(stage_pointer)
                for name, stage_pointer in stages.items()
            }
            expected_upstreams = {
                "paired": tuple(sorted(set(roles.values()))),
                "enriched": (stages["paired"],),
                "no_arb": (stages["enriched"],),
                "filtered": (stages["no_arb"],),
                "fitted": (stages["filtered"],),
            }
            for stage, stage_artifact in validated_stages.items():
                upstream_values = stage_artifact.manifest.get("upstream")
                if not isinstance(upstream_values, list):
                    _fail(f"Dataset daily chain {index} {stage} has no upstream list")
                observed_upstream = tuple(
                    _pointer_from_value(value, f"daily chain {index} {stage} upstream")
                    for value in upstream_values
                )
                if observed_upstream != expected_upstreams[stage]:
                    _fail(f"Dataset daily chain {index} {stage} ancestry differs")
            deep_completed += 1
            now = time.monotonic()
            if (
                deep_completed == deep_total
                or now - deep_last_progress >= progress_interval_seconds
            ):
                elapsed = now - deep_started
                rate = deep_completed / elapsed if elapsed > 0.0 else 0.0
                _emit_progress(
                    progress,
                    "dataset_lineage_progress",
                    elapsed_seconds=elapsed,
                    completed=deep_completed,
                    total=deep_total,
                    failed=0,
                    estimated_remaining_seconds=(
                        (deep_total - deep_completed) / rate if rate > 0.0 else None
                    ),
                )
                deep_last_progress = now
    if not semantic:
        _emit_progress(
            progress,
            "dataset_validation_completed",
            elapsed_seconds=time.monotonic() - validation_started,
            completed=0,
            total=0,
            failed=0,
        )
        return None

    partition_statistics: list[PartitionStatistics] = []
    semantic_started = time.monotonic()
    semantic_last_progress = semantic_started
    expected_partition_fields = {
        "trading_date",
        "is_half_day",
        "path",
        "sha256",
        "byte_size",
        "fitted_artifact",
        "fitted_manifest_sha256",
        "fitted_source_sha256",
        "typed_empty",
        "typed_empty_reason",
        "statistics",
    }
    for index, (expected_date, record) in enumerate(
        zip(expected_dates, dataset.partitions, strict=True)
    ):
        if set(record) != expected_partition_fields:
            _fail(f"{expected_date}: partition record has missing or unexpected fields")
        relative = str(record["path"])
        match = _PARTITION_PATH.fullmatch(relative)
        if match is None or match.group(1) != _safe_partition_ticker_label(
            dataset.config.ticker
        ):
            _fail(f"{expected_date}: partition path ticker is inconsistent")
        if match.group(2) != str(expected_date.year) or match.group(3) != expected_date.isoformat():
            _fail(f"{expected_date}: partition path date is inconsistent")
        path = validated.path / PurePosixPath(relative)
        if record.get("sha256") != _sha256_file(path):
            _fail(f"{expected_date}: partition catalog checksum is inconsistent")
        if record.get("fitted_source_sha256") != record.get("sha256"):
            _fail(f"{expected_date}: source/copy checksum evidence differs")
        if record.get("byte_size") != path.stat().st_size:
            _fail(f"{expected_date}: partition byte-size evidence differs")
        manifest_digest = record.get("fitted_manifest_sha256")
        if not isinstance(manifest_digest, str) or _HEX_64.fullmatch(manifest_digest) is None:
            _fail(f"{expected_date}: fitted manifest checksum evidence is invalid")
        fitted_pointer = _pointer_from_value(
            record.get("fitted_artifact"), f"{expected_date} fitted artifact"
        )
        if fitted_pointer != chain_fitted[index]:
            _fail(f"{expected_date}: partition and lineage fitted pointers differ")
        if deep_lineage:
            source_artifact = store.validate(fitted_pointer)
            if record.get("fitted_manifest_sha256") != _sha256_file(
                source_artifact.path / "manifest.json"
            ):
                _fail(f"{expected_date}: fitted manifest checksum evidence differs")
            if record.get("fitted_source_sha256") != _sha256_file(
                source_artifact.path / "data.parquet"
            ):
                _fail(f"{expected_date}: fitted source checksum evidence differs")
        frame = pl.read_parquet(path, hive_partitioning=False)
        typed_reason = record.get("typed_empty_reason")
        if typed_reason is not None and not isinstance(typed_reason, str):
            _fail(f"{expected_date}: typed-empty reason must be a string or null")
        statistics = validate_fitted_partition(
            frame,
            config=dataset.config,
            trading_date=expected_date,
            typed_empty_reason=typed_reason,
            expected_raw_options_artifact_id=chain_raw_roles[index]["options"].artifact_id,
        )
        if record.get("typed_empty") is not frame.is_empty():
            _fail(f"{expected_date}: typed-empty flag differs from partition contents")
        if record.get("is_half_day") is not is_half_day(expected_date):
            _fail(f"{expected_date}: partition half-day evidence differs")
        if statistics.as_dict() != _partition_record_statistics(record):
            _fail(f"{expected_date}: recorded partition statistics differ from recomputation")
        partition_statistics.append(statistics)
        completed = index + 1
        now = time.monotonic()
        if (
            completed == len(expected_dates)
            or now - semantic_last_progress >= progress_interval_seconds
        ):
            elapsed = now - semantic_started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            _emit_progress(
                progress,
                "dataset_semantic_validation_progress",
                elapsed_seconds=elapsed,
                completed=completed,
                total=len(expected_dates),
                failed=0,
                estimated_remaining_seconds=(
                    (len(expected_dates) - completed) / rate if rate > 0.0 else None
                ),
            )
            semantic_last_progress = now
    summary = aggregate_statistics(
        tuple(partition_statistics),
        requested_dates=len(expected_dates),
        exclusions=_scheduled_exclusions(dataset.config),
    )
    expected_typed_empty = canonical_value(
        tuple(
            {"trading_date": item.trading_date, "reason": item.typed_empty_reason}
            for item in partition_statistics
            if item.typed_empty
        )
    )
    aggregate_metadata = {
        "populated_date_count": summary.populated_dates,
        "typed_empty_date_count": summary.typed_empty_dates,
        "scheduled_exclusion_count": summary.scheduled_exclusions,
        "typed_empty_partitions": expected_typed_empty,
        "row_count": summary.row_count,
        "eligible_row_count": summary.eligible_rows,
        "active_side_book_count": summary.active_side_books,
        "validation_outcome": summary.outcome,
    }
    if any(metadata.get(name) != value for name, value in aggregate_metadata.items()):
        _fail("Dataset manifest aggregate validation evidence differs")
    evidence = metadata.get("evidence_checksums")
    evidence_names = (
        "schema.json",
        "partitions.json",
        "lineage.json",
        "validation.json",
        "legacy-comparison.json",
    )
    expected_evidence = {
        name: _sha256_file(validated.path / name) for name in evidence_names
    }
    if evidence != expected_evidence:
        _fail("Dataset manifest evidence checksums differ from immutable payloads")
    report = _read_json_file(validated.path / "validation.json", "validation.json")
    if set(report) != {
        "schema",
        "validator_version",
        "outcome",
        "failure_count",
        "summary",
    } or not isinstance(report.get("validator_version"), str) or not report[
        "validator_version"
    ].strip():
        _fail("validation.json has an invalid immutable report envelope")
    if (
        report.get("schema") != VALIDATION_SCHEMA
        or report.get("outcome") != "passed"
        or report.get("failure_count") != 0
        or report.get("summary") != summary.as_dict()
    ):
        _fail("validation.json differs from recomputed semantic validation")
    comparison = _read_json_file(
        validated.path / "legacy-comparison.json", "legacy-comparison.json"
    )
    expected_comparison = canonical_value(_legacy_comparison_document(dataset.config))
    if set(comparison) != set(expected_comparison) or any(
        comparison.get(name) != value
        for name, value in expected_comparison.items()
        if name != "notes"
    ) or not isinstance(comparison.get("notes"), str) or not comparison["notes"].strip():
        _fail("legacy-comparison.json differs from the migration-oracle contract")
    _emit_progress(
        progress,
        "dataset_validation_completed",
        elapsed_seconds=time.monotonic() - validation_started,
        completed=len(expected_dates),
        total=len(expected_dates),
        failed=0,
    )
    return summary


@version("1.0")
def _daily_chain_record(item: DailyArtifactPlan) -> dict[str, Any]:
    return {
        "trading_date": item.trading_date,
        "raw_roles": {
            "security": item.raw.security,
            "options": item.raw.options,
            "spot": item.raw.spot,
            "previous_spot": item.raw.previous_spot,
            "forwards": item.raw.forwards,
            "zero_curve": item.raw.zero_curve,
        },
        "paired": item.paired.pointer,
        "enriched": item.enriched.pointer,
        "no_arb": item.no_arb.pointer,
        "filtered": item.filtered.pointer,
        "fitted": item.fitted.pointer,
    }


@version("1.0", dependencies=[canonical_json])
def _json_payload(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


@version("1.0")
def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@version("1.0")
def _emit_progress(
    progress: Callable[[Mapping[str, Any]], None] | None,
    event: str,
    **fields: Any,
) -> None:
    if progress is not None:
        progress(cast(dict[str, Any], canonical_value({"event": event, **fields})))


@version(
    "1.0",
    dependencies=[read_daily_frame, validate_fitted_partition, _emit_progress],
)
def _validate_fitted_sources(
    store: ArtifactStore,
    config: InternalDatasetConfig,
    plan: InternalDatasetPlan,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None,
    progress_interval_seconds: float,
) -> tuple[
    tuple[ValidatedArtifact, ...],
    tuple[PartitionStatistics, ...],
]:
    artifacts: list[ValidatedArtifact] = []
    statistics: list[PartitionStatistics] = []
    started = time.monotonic()
    last_progress = started
    total = len(plan.daily_artifacts)
    for completed, item in enumerate(plan.daily_artifacts, start=1):
        artifact = store.validate(item.fitted.pointer, expected=item.fitted, deep=True)
        metadata = artifact.manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            _fail(f"{item.trading_date}: fitted artifact metadata is malformed")
        frame = read_daily_frame(
            store,
            artifact,
            schema=FITTED_SCHEMA,
            ticker=config.ticker,
            trading_date=item.trading_date,
        )
        typed_reason = metadata.get("typed_empty_reason")
        if typed_reason is not None and not isinstance(typed_reason, str):
            _fail(f"{item.trading_date}: invalid fitted typed-empty reason")
        item_statistics = validate_fitted_partition(
            frame,
            config=config,
            trading_date=item.trading_date,
            typed_empty_reason=typed_reason,
            expected_raw_options_artifact_id=item.raw.options.artifact_id,
        )
        artifacts.append(artifact)
        statistics.append(item_statistics)
        now = time.monotonic()
        if completed == total or now - last_progress >= progress_interval_seconds:
            elapsed = now - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = (total - completed) / rate if rate > 0.0 else None
            _emit_progress(
                progress,
                "dataset_validation_progress",
                elapsed_seconds=elapsed,
                completed=completed,
                total=total,
                failed=0,
                estimated_remaining_seconds=remaining,
            )
            last_progress = now
    planned_raw = {
        pointer for item in plan.daily_artifacts for pointer in item.raw.all()
    }
    catalog_raw = set(plan.raw_catalog_spec.upstream)
    if planned_raw != catalog_raw:
        _fail("Validated fitted chains do not close to exactly the planned Raw catalog")
    return tuple(artifacts), tuple(statistics)


@version(
    "1.1",
    dependencies=[
        _scheduled_exclusions,
        aggregate_statistics,
        _partition_relative_path,
        _daily_chain_record,
        _readme,
        _json_payload,
        _schema_document,
        _legacy_comparison_document,
        _sha256_bytes,
        _sha256_file,
        validate_fitted_partition,
    ],
)
def _publish_internal_dataset(
    store: ArtifactStore,
    config: InternalDatasetConfig,
    plan: InternalDatasetPlan,
    fitted_artifacts: tuple[ValidatedArtifact, ...],
    statistics: tuple[PartitionStatistics, ...],
    *,
    source_revision: str,
    progress: Callable[[Mapping[str, Any]], None] | None,
    progress_interval_seconds: float,
) -> ValidatedArtifact:
    if not len(fitted_artifacts) == len(statistics) == len(plan.daily_artifacts):
        raise ValueError("Dataset assembly evidence lengths differ")
    exclusions = _scheduled_exclusions(config)
    summary = aggregate_statistics(
        statistics,
        requested_dates=len(plan.expected_dates),
        exclusions=exclusions,
    )
    partitions: list[dict[str, Any]] = []
    for item, artifact, item_statistics in zip(
        plan.daily_artifacts,
        fitted_artifacts,
        statistics,
        strict=True,
    ):
        source_path = artifact.path / "data.parquet"
        relative = _partition_relative_path(config.ticker, item.trading_date)
        source_checksum = _sha256_file(source_path)
        partitions.append(
            {
                "trading_date": item.trading_date,
                "is_half_day": is_half_day(item.trading_date),
                "path": relative,
                "sha256": source_checksum,
                "byte_size": source_path.stat().st_size,
                "fitted_artifact": artifact.pointer,
                "fitted_manifest_sha256": _sha256_file(artifact.path / "manifest.json"),
                "fitted_source_sha256": source_checksum,
                "typed_empty": item_statistics.typed_empty,
                "typed_empty_reason": item_statistics.typed_empty_reason,
                "statistics": item_statistics,
            }
        )
    partition_document = {
        "schema": PARTITION_CATALOG_SCHEMA,
        "ticker": config.ticker,
        "partitions": partitions,
    }
    lineage_document = {
        "schema": LINEAGE_SCHEMA,
        "raw_catalog": plan.raw_catalog_spec.pointer,
        "raw_mapping_digest": plan.raw_mapping_digest,
        "daily_chains": tuple(_daily_chain_record(item) for item in plan.daily_artifacts),
    }
    validation_document = {
        "schema": VALIDATION_SCHEMA,
        "validator_version": str(validate_internal_dataset.version),
        "outcome": "passed",
        "failure_count": 0,
        "summary": summary,
    }
    documents: dict[str, bytes] = {
        "README.md": _readme(config, plan).encode("utf-8"),
        "config.json": _json_payload(plan.configuration),
        "schema.json": _json_payload(_schema_document()),
        "partitions.json": _json_payload(partition_document),
        "lineage.json": _json_payload(lineage_document),
        "validation.json": _json_payload(validation_document),
        "legacy-comparison.json": _json_payload(_legacy_comparison_document(config)),
    }
    evidence_checksums = {
        name: _sha256_bytes(payload)
        for name, payload in documents.items()
        if name.endswith(".json") and name != "config.json"
    }
    metadata = {
        "source_revision": source_revision,
        "dataset_schema_version": INTERNAL_DATASET_SCHEMA_VERSION,
        "partition_schema_version": FITTED_SCHEMA_VERSION,
        "producer_version": plan.dataset_spec.producer_version,
        "configuration_id": plan.dataset_spec.configuration_id,
        "raw_catalog": plan.raw_catalog_spec.pointer,
        "raw_mapping_digest": plan.raw_mapping_digest,
        "raw_artifacts": plan.raw_catalog_spec.upstream,
        "fitted_artifacts": tuple(artifact.pointer for artifact in fitted_artifacts),
        "planned_date_count": len(plan.expected_dates),
        "partition_count": len(partitions),
        "populated_date_count": summary.populated_dates,
        "typed_empty_date_count": summary.typed_empty_dates,
        "scheduled_exclusion_count": summary.scheduled_exclusions,
        "scheduled_exclusions": exclusions,
        "typed_empty_partitions": tuple(
            {
                "trading_date": item.trading_date,
                "reason": item.typed_empty_reason,
            }
            for item in statistics
            if item.typed_empty
        ),
        "bad_data_policy_version": str(bad_data_policy.version),
        "row_count": summary.row_count,
        "eligible_row_count": summary.eligible_rows,
        "active_side_book_count": summary.active_side_books,
        "validation_outcome": summary.outcome,
        "canonical_promotion_ready": False,
        "evidence_checksums": evidence_checksums,
    }
    builder = store.writer(plan.dataset_spec, metadata=metadata)
    try:
        with builder as staging:
            for relative, payload in documents.items():
                staging.write_bytes(relative, payload)
            started = time.monotonic()
            last_progress = started
            total = len(partitions)
            for completed, (item, artifact, record) in enumerate(
                zip(plan.daily_artifacts, fitted_artifacts, partitions, strict=True),
                start=1,
            ):
                copied = staging.copy_file(
                    str(record["path"]), artifact.path / "data.parquet"
                )
                if _sha256_file(copied) != record["sha256"]:
                    _fail(f"{item.trading_date}: copied partition checksum changed")
                copied_frame = pl.read_parquet(copied, hive_partitioning=False)
                validate_fitted_partition(
                    copied_frame,
                    config=config,
                    trading_date=item.trading_date,
                    typed_empty_reason=cast(str | None, record["typed_empty_reason"]),
                    expected_raw_options_artifact_id=item.raw.options.artifact_id,
                )
                now = time.monotonic()
                if completed == total or now - last_progress >= progress_interval_seconds:
                    elapsed = now - started
                    rate = completed / elapsed if elapsed > 0.0 else 0.0
                    remaining = (total - completed) / rate if rate > 0.0 else None
                    _emit_progress(
                        progress,
                        "dataset_copy_progress",
                        elapsed_seconds=elapsed,
                        completed=completed,
                        total=total,
                        failed=0,
                        estimated_remaining_seconds=remaining,
                    )
                    last_progress = now
    except ArtifactExistsError:
        return store.validate(plan.dataset_spec.pointer, expected=plan.dataset_spec)
    return builder.artifact


@version(
    "1.0",
    dependencies=[
        InternalDatasetBuildResult,
        plan_internal_dataset,
        build_intermediates,
        publish_raw_lineage_catalog,
        _validate_fitted_sources,
        _publish_internal_dataset,
        validate_internal_dataset,
        _source_revision,
        RunJournal,
    ],
)
def build_internal_dataset(
    locations: StorageLocations,
    config: InternalDatasetConfig,
    options: DatasetBuildOptions = DatasetBuildOptions(),
    *,
    dry_run: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> InternalDatasetBuildResult:
    """Resolve output first, then build, validate, and atomically publish if absent."""

    plan = plan_internal_dataset(config)
    store = ArtifactStore(locations)
    journal_path = locations.datasets / ".runs" / "widb" / f"{plan.plan_id}.jsonl"
    reuse = store.resolve(plan.dataset_spec, version=options.version_override)
    if reuse is not None:
        validation = validate_internal_dataset(
            store,
            reuse.artifact,
            progress=progress,
            progress_interval_seconds=options.progress_interval_seconds,
        )
        return InternalDatasetBuildResult(
            plan=plan,
            artifact=reuse.artifact.pointer,
            path=reuse.artifact.path,
            dry_run=dry_run,
            built=False,
            reuse=reuse,
            daily_summary=None,
            validation=validation,
            journal_path=journal_path,
        )
    if dry_run:
        return InternalDatasetBuildResult(
            plan=plan,
            artifact=plan.dataset_spec.pointer,
            path=store.path_for(plan.dataset_spec.pointer),
            dry_run=True,
            built=False,
            reuse=None,
            daily_summary=None,
            validation=None,
            journal_path=journal_path,
        )

    journal = RunJournal(journal_path)
    journal.require_plan(plan.plan_id)
    journal.emit(
        "run_started",
        plan_id=plan.plan_id,
        dataset=plan.dataset_spec.pointer,
        total=len(plan.expected_dates),
    )
    _emit_progress(
        progress,
        "dataset_build_started",
        plan_id=plan.plan_id,
        total=len(plan.expected_dates),
    )
    daily_summary: DailyRunSummary | None = None
    try:
        if options.require_existing_intermediates:
            for item in plan.daily_artifacts:
                store.validate(item.fitted.pointer, expected=item.fitted)
        else:
            daily_result = build_intermediates(
                locations,
                IntermediateBuildRequest(
                    ticker=config.ticker,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    through="fitted-day",
                    include_half_days=config.include_half_days,
                    source=config.source,
                    source_generation=config.source_generation,
                    max_workers=options.max_workers,
                    progress_interval_seconds=options.progress_interval_seconds,
                ),
                config.daily,
                config.surface,
                progress=progress,
            )
            daily_summary = daily_result.summary
        fitted_artifacts, statistics = _validate_fitted_sources(
            store,
            config,
            plan,
            progress=progress,
            progress_interval_seconds=options.progress_interval_seconds,
        )
        catalog = publish_raw_lineage_catalog(store, plan)
        if catalog.pointer != plan.raw_catalog_spec.pointer:
            _fail("Published Raw catalog identity changed")
        source_revision = _source_revision()
        artifact = _publish_internal_dataset(
            store,
            config,
            plan,
            fitted_artifacts,
            statistics,
            source_revision=source_revision,
            progress=progress,
            progress_interval_seconds=options.progress_interval_seconds,
        )
        validation = validate_internal_dataset(
            store,
            artifact,
            progress=progress,
            progress_interval_seconds=options.progress_interval_seconds,
        )
    except BaseException as exc:
        try:
            journal.emit(
                "run_interrupted",
                plan_id=plan.plan_id,
                error_type=type(exc).__name__,
                message=str(exc),
                completed=0 if daily_summary is None else daily_summary.completed,
                total=len(plan.expected_dates),
            )
        except BaseException as journal_error:
            exc.add_note(f"Could not journal Dataset interruption: {journal_error}")
        raise
    journal.emit(
        "run_completed",
        plan_id=plan.plan_id,
        status="ok",
        dataset=artifact.pointer,
        completed=len(plan.expected_dates),
        total=len(plan.expected_dates),
        failed=0,
    )
    _emit_progress(
        progress,
        "dataset_build_completed",
        artifact=artifact.pointer,
        completed=len(plan.expected_dates),
        total=len(plan.expected_dates),
        failed=0,
    )
    return InternalDatasetBuildResult(
        plan=plan,
        artifact=artifact.pointer,
        path=artifact.path,
        dry_run=False,
        built=True,
        reuse=None,
        daily_summary=daily_summary,
        validation=validation,
        journal_path=journal_path,
    )


__all__ = [
    "DatasetValidationError",
    "DatasetValidationSummary",
    "InternalDataset",
    "InternalDatasetBuildResult",
    "InternalDatasetPlan",
    "PartitionStatistics",
    "aggregate_statistics",
    "build_internal_dataset",
    "internal_dataset_producer_version",
    "open_internal_dataset",
    "plan_internal_dataset",
    "publish_raw_lineage_catalog",
    "raw_catalog_producer_version",
    "validate_fitted_partition",
    "validate_internal_dataset",
]
