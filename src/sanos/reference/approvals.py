"""Owner-authorized, evidence-bound daily migration discrepancy approvals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, cast

from cdxcore.version import version

from sanos.artifacts import ArtifactPointer, ArtifactStore, StorageClass, atomic_create_json
from sanos.config import (
    StorageLocations,
    canonical_json,
    canonical_sha256,
    canonical_value,
)
from sanos.datasets import InternalDatasetConfig
from sanos.datasets.internal import validate_fitted_partition
from sanos.reference.compare import (
    APPROVAL_SCHEMA,
    REFERENCE_REPORT_SCHEMA,
    DailyLegacyComparisonTolerances,
    ReferenceComparisonError,
    compare_populated_day,
)
from sanos.reference.legacy_market import read_legacy_pytree_hdf5
from sanos.widb.build import IntermediateBuildRequest, plan_daily_artifacts, read_daily_frame
from sanos.widb.config import DailySurfaceConfig, DailyWidbConfig
from sanos.widb.surface import FITTED_SCHEMA

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_FIELDS = {
    "approval_document",
    "artifacts_built",
    "classification_counts",
    "configuration",
    "dates",
    "legacy_catalog",
    "network_accessed",
    "outcome",
    "pilot_contract",
    "production_contract",
    "report_id",
    "request",
    "resolved_versions",
    "schema",
    "source_revision",
    "tolerances",
}
_DAY_FIELDS = {
    "approval",
    "classification",
    "comparison",
    "date",
    "fresh",
    "legacy",
    "observations",
    "pilot_checks",
    "production_checks",
    "raw_case_evidence",
}
_CLASSIFICATIONS = {
    "approved_discrepancy",
    "match",
    "oracle_gap",
    "policy_empty",
    "unexplained_discrepancy",
}


@version("1.0")
def _assert_approval_path_has_no_symlinks(path: Path) -> Path:
    """Return a lexical absolute path after rejecting every symlink component."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ReferenceComparisonError("Approval evidence path contains a symlink")
    return absolute


@version("1.0")
def _approval_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@version("1.0", dependencies=[_assert_approval_path_has_no_symlinks, canonical_json])
def _read_canonical_json_object(path: Path, label: str) -> dict[str, Any]:
    safe_path = _assert_approval_path_has_no_symlinks(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(safe_path, flags)
    except OSError as exc:
        raise ReferenceComparisonError(f"{label} is missing or unsafe") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 256 * 1024 * 1024:
            raise ReferenceComparisonError(f"{label} is not a safe regular JSON file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            source = stream.read()
    except (OSError, UnicodeError) as exc:
        raise ReferenceComparisonError(f"Cannot read {label}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        value = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ReferenceComparisonError(f"Cannot decode {label}") from exc
    if not isinstance(value, dict):
        raise ReferenceComparisonError(f"{label} must be a JSON object")
    try:
        expected = canonical_json(value) + "\n"
    except (TypeError, ValueError) as exc:
        raise ReferenceComparisonError(f"{label} is not canonical JSON") from exc
    if source != expected:
        raise ReferenceComparisonError(f"{label} is not canonically encoded")
    return value


@version("1.0")
def _iso_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise ReferenceComparisonError(f"{label} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ReferenceComparisonError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ReferenceComparisonError(f"{label} must be a canonical ISO date")
    return parsed


@version("1.0")
def _passing_checks(value: object, label: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) or item.get("passed") is not True for item in value
    ):
        raise ReferenceComparisonError(f"{label} must contain only passing checks")


@version(
    "1.0",
    dependencies=[_iso_date, _passing_checks, canonical_sha256, canonical_value],
)
def validate_comparison_report_for_approval(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the closed logical contract of a blocked comparison report."""

    document = cast(dict[str, Any], canonical_value(dict(value)))
    if set(document) != _REPORT_FIELDS:
        raise ReferenceComparisonError("Comparison report fields are missing or unexpected")
    if document.get("schema") != REFERENCE_REPORT_SCHEMA:
        raise ReferenceComparisonError("Comparison report has an unsupported schema")
    report_id = document.get("report_id")
    if not isinstance(report_id, str) or _HEX_64.fullmatch(report_id) is None:
        raise ReferenceComparisonError("Comparison report ID is invalid")
    body = {key: item for key, item in document.items() if key != "report_id"}
    if canonical_sha256(body) != report_id:
        raise ReferenceComparisonError("Comparison report ID does not bind its content")
    if (
        document.get("outcome") != "blocked"
        or document.get("network_accessed") is not False
        or document.get("artifacts_built") is not False
        or document.get("approval_document") is not None
    ):
        raise ReferenceComparisonError(
            "Approval requires an offline, no-build, blocked report without an approval overlay"
        )
    source_revision = document.get("source_revision")
    if not isinstance(source_revision, str) or _HEX_40.fullmatch(source_revision) is None:
        raise ReferenceComparisonError("Comparison report source revision is invalid")

    request = document.get("request")
    request_fields = {
        "declared_oracle_gaps",
        "end_date",
        "enforce_phase10_pilot",
        "enforce_phase11_production",
        "source",
        "source_generation",
        "start_date",
        "ticker",
    }
    if not isinstance(request, Mapping) or set(request) != request_fields:
        raise ReferenceComparisonError("Comparison report request is malformed")
    start = _iso_date(request.get("start_date"), "request.start_date")
    end = _iso_date(request.get("end_date"), "request.end_date")
    if end < start:
        raise ReferenceComparisonError("Comparison report request dates are reversed")
    for name in ("enforce_phase10_pilot", "enforce_phase11_production"):
        if not isinstance(request.get(name), bool):
            raise ReferenceComparisonError(f"request.{name} must be a boolean")
    if request["enforce_phase10_pilot"] and request["enforce_phase11_production"]:
        raise ReferenceComparisonError("Comparison report enables contradictory contracts")
    if (
        request.get("source") != "optionmetrics"
        or not isinstance(request.get("source_generation"), str)
        or not request["source_generation"]
        or not isinstance(request.get("ticker"), str)
        or not request["ticker"]
    ):
        raise ReferenceComparisonError("Comparison report source identity is malformed")

    for name in ("pilot_contract", "production_contract"):
        contract = document.get(name)
        if not isinstance(contract, Mapping) or contract.get("passed") is not True:
            raise ReferenceComparisonError(f"Comparison report {name} did not pass")
        if "daily_checks_passed" in contract and contract["daily_checks_passed"] is not True:
            raise ReferenceComparisonError(f"Comparison report {name} daily checks did not pass")

    days = document.get("dates")
    if not isinstance(days, list) or not days:
        raise ReferenceComparisonError("Comparison report has no daily evidence")
    observed_dates: list[str] = []
    classifications: Counter[str] = Counter()
    for index, day_record in enumerate(days):
        if not isinstance(day_record, Mapping) or set(day_record) != _DAY_FIELDS:
            raise ReferenceComparisonError("Comparison daily fields are missing or unexpected")
        day_key = day_record.get("date")
        parsed = _iso_date(day_key, f"dates[{index}].date")
        if parsed < start or parsed > end:
            raise ReferenceComparisonError("Comparison daily evidence falls outside the request")
        observed_dates.append(cast(str, day_key))
        classification = day_record.get("classification")
        if classification not in _CLASSIFICATIONS:
            raise ReferenceComparisonError("Comparison daily classification is invalid")
        classifications[cast(str, classification)] += 1
        if day_record.get("approval") is not None:
            raise ReferenceComparisonError("Comparison report already contains a daily approval")
        _passing_checks(day_record.get("pilot_checks"), "pilot_checks")
        _passing_checks(day_record.get("production_checks"), "production_checks")
    if observed_dates != sorted(set(observed_dates)):
        raise ReferenceComparisonError("Comparison report dates must be sorted and unique")
    if "approved_discrepancy" in classifications:
        raise ReferenceComparisonError("Comparison report already contains approved discrepancies")
    expected_counts = {key: classifications[key] for key in sorted(classifications)}
    if document.get("classification_counts") != expected_counts:
        raise ReferenceComparisonError("Comparison classification counts are inconsistent")
    if classifications.get("unexplained_discrepancy", 0) < 1:
        raise ReferenceComparisonError("Comparison report has no discrepancy to approve")
    return document


@version(
    "1.0",
    dependencies=[_read_canonical_json_object, validate_comparison_report_for_approval],
)
def load_comparison_report_for_approval(path: Path) -> dict[str, Any]:
    """Load a canonical blocked report without following symlink components."""

    return validate_comparison_report_for_approval(
        _read_canonical_json_object(path, "Comparison report")
    )


@version("1.0")
def _authority_text(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReferenceComparisonError(f"{label} must be nonempty and trimmed")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise ReferenceComparisonError(f"{label} contains unsafe text")
    return value


@version("1.0", dependencies=[_authority_text, _iso_date, canonical_value])
def validate_comparison_approval_document(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the closed approval-document schema without opening evidence."""

    document = cast(dict[str, Any], canonical_value(dict(value)))
    if (
        set(document) != {"schema", "approvals"}
        or document.get("schema") != APPROVAL_SCHEMA
        or not isinstance(document.get("approvals"), list)
        or not document["approvals"]
    ):
        raise ReferenceComparisonError("Approval document is malformed")
    required = {
        "approval_date",
        "approval_reference",
        "approver",
        "date",
        "explanation",
        "fields",
        "fresh_artifact_id",
        "fresh_data_sha256",
        "legacy_file_sha256",
    }
    observed_dates: list[str] = []
    for record in document["approvals"]:
        if not isinstance(record, Mapping) or set(record) != required:
            raise ReferenceComparisonError("Approval record fields are malformed")
        approval_day = _iso_date(record.get("date"), "approval date scope")
        _iso_date(record.get("approval_date"), "owner approval date")
        observed_dates.append(approval_day.isoformat())
        fields = record.get("fields")
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(item, str) and item for item in fields)
            or fields != sorted(set(fields))
        ):
            raise ReferenceComparisonError("Approval fields must be sorted and unique")
        for name in (
            "fresh_artifact_id",
            "fresh_data_sha256",
            "legacy_file_sha256",
        ):
            digest = record.get(name)
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                raise ReferenceComparisonError(f"Approval {name} is invalid")
        _authority_text(record.get("explanation"), "explanation", maximum=4096)
        _authority_text(record.get("approver"), "approver", maximum=256)
        _authority_text(
            record.get("approval_reference"), "approval_reference", maximum=1024
        )
    if observed_dates != sorted(set(observed_dates)):
        raise ReferenceComparisonError("Approval dates must be sorted and unique")
    return document


@version("1.0")
def _manifest_digest(artifact: Any, name: str) -> str:
    files = artifact.manifest.get("files")
    if not isinstance(files, list):
        raise ReferenceComparisonError("Fresh artifact manifest has no file inventory")
    records = [
        item for item in files if isinstance(item, Mapping) and item.get("path") == name
    ]
    if len(records) != 1 or not isinstance(records[0].get("sha256"), str):
        raise ReferenceComparisonError(f"Fresh artifact has no unique {name} digest")
    return cast(str, records[0]["sha256"])


@version("1.0", dependencies=[ArtifactPointer, StorageClass])
def _exact_pointer(value: object, label: str) -> ArtifactPointer:
    if not isinstance(value, Mapping) or set(value) != {
        "artifact_id",
        "namespace",
        "storage_class",
    }:
        raise ReferenceComparisonError(f"{label} is malformed")
    try:
        return ArtifactPointer(
            StorageClass(value["storage_class"]),
            cast(str, value["namespace"]),
            cast(str, value["artifact_id"]),
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceComparisonError(f"{label} is malformed") from exc


@version(
    "1.0",
    dependencies=[
        ArtifactPointer,
        ArtifactStore,
        DailyLegacyComparisonTolerances,
        DailySurfaceConfig,
        DailyWidbConfig,
        IntermediateBuildRequest,
        InternalDatasetConfig,
        _approval_sha256_file,
        _assert_approval_path_has_no_symlinks,
        _authority_text,
        _exact_pointer,
        _iso_date,
        _manifest_digest,
        _passing_checks,
        canonical_value,
        compare_populated_day,
        load_comparison_report_for_approval,
        plan_daily_artifacts,
        read_daily_frame,
        read_legacy_pytree_hdf5,
        validate_comparison_approval_document,
        validate_fitted_partition,
    ],
)
def build_comparison_approval_document(
    locations: StorageLocations,
    *,
    comparison_report: Path,
    legacy_root: Path,
    approval_dates: Sequence[date],
    explanation: str,
    approver: str,
    approval_date: date,
    approval_reference: str,
) -> dict[str, Any]:
    """Reopen exact evidence and materialize an owner-authorized approval document."""

    if not approval_dates or len(set(approval_dates)) != len(approval_dates):
        raise ReferenceComparisonError("Approval dates must be nonempty and unique")
    requested_dates = tuple(sorted(approval_dates))
    explanation = _authority_text(explanation, "explanation", maximum=4096)
    approver = _authority_text(approver, "approver", maximum=256)
    approval_reference = _authority_text(
        approval_reference, "approval_reference", maximum=1024
    )
    report = load_comparison_report_for_approval(comparison_report)

    root = _assert_approval_path_has_no_symlinks(legacy_root)
    if not root.is_dir():
        raise ReferenceComparisonError("Legacy root is missing or unsafe")
    request_value = cast(Mapping[str, Any], report["request"])
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping) or set(configuration) != {"daily", "surface"}:
        raise ReferenceComparisonError("Comparison configuration is malformed")
    try:
        daily = DailyWidbConfig.from_mapping(cast(Mapping[str, Any], configuration["daily"]))
        surface = DailySurfaceConfig.from_mapping(
            cast(Mapping[str, Any], configuration["surface"])
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceComparisonError("Comparison configuration is malformed") from exc
    if canonical_value(asdict(daily)) != configuration["daily"] or canonical_value(
        asdict(surface)
    ) != configuration["surface"]:
        raise ReferenceComparisonError("Comparison configuration is not canonical")
    tolerances_value = report.get("tolerances")
    if not isinstance(tolerances_value, Mapping):
        raise ReferenceComparisonError("Comparison tolerances are malformed")
    try:
        tolerances = DailyLegacyComparisonTolerances(**dict(tolerances_value))
    except (TypeError, ValueError) as exc:
        raise ReferenceComparisonError("Comparison tolerances are malformed") from exc
    if tolerances.to_dict() != dict(tolerances_value):
        raise ReferenceComparisonError("Comparison tolerances are not canonical")

    request = IntermediateBuildRequest(
        ticker=cast(str, request_value["ticker"]),
        start_date=_iso_date(request_value["start_date"], "request.start_date"),
        end_date=_iso_date(request_value["end_date"], "request.end_date"),
        through="fitted-day",
        include_half_days=True,
        source=cast(str, request_value["source"]),
        source_generation=cast(str, request_value["source_generation"]),
    )
    plans = plan_daily_artifacts(request, daily, surface)
    plans_by_date = {item.trading_date: item for item in plans}
    report_days = cast(list[Mapping[str, Any]], report["dates"])
    if [item["date"] for item in report_days] != [
        item.trading_date.isoformat() for item in plans
    ]:
        raise ReferenceComparisonError("Comparison daily coverage contradicts its request")
    days_by_date = {
        _iso_date(item["date"], "daily date"): item for item in report_days
    }
    missing = [item.isoformat() for item in requested_dates if item not in days_by_date]
    if missing:
        raise ReferenceComparisonError(f"Approval dates are absent from the report: {missing}")

    validation_config = InternalDatasetConfig(
        name="daily-reference-approval-validation",
        ticker=request.ticker,
        start_date=request.start_date,
        end_date=request.end_date,
        include_half_days=True,
        source=request.source,
        source_generation=request.source_generation,
        daily=daily,
        surface=surface,
    )
    store = ArtifactStore(locations)
    catalog = report.get("legacy_catalog")
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("selected_files"), list):
        raise ReferenceComparisonError("Comparison legacy catalog evidence is malformed")
    selected_files = cast(list[Mapping[str, Any]], catalog["selected_files"])
    approvals: list[dict[str, Any]] = []
    for trading_date in requested_dates:
        day_record = days_by_date[trading_date]
        day_key = trading_date.isoformat()
        if day_record.get("classification") != "unexplained_discrepancy":
            raise ReferenceComparisonError(
                f"{day_key} is not an unexplained populated discrepancy"
            )
        _passing_checks(day_record.get("pilot_checks"), f"{day_key} pilot_checks")
        _passing_checks(
            day_record.get("production_checks"), f"{day_key} production_checks"
        )
        comparison = day_record.get("comparison")
        if not isinstance(comparison, Mapping) or comparison.get("matched") is not False:
            raise ReferenceComparisonError(
                f"{day_key} is not a populated numerical discrepancy"
            )
        mismatch_fields = comparison.get("mismatch_fields")
        if (
            not isinstance(mismatch_fields, list)
            or not mismatch_fields
            or not all(isinstance(item, str) and item for item in mismatch_fields)
            or mismatch_fields != sorted(set(mismatch_fields))
        ):
            raise ReferenceComparisonError(f"{day_key} mismatch fields are not canonical")

        fresh = day_record.get("fresh")
        if not isinstance(fresh, Mapping):
            raise ReferenceComparisonError(f"{day_key} fresh evidence is malformed")
        pointer = _exact_pointer(fresh.get("artifact"), f"{day_key} fresh artifact")
        plan = plans_by_date[trading_date]
        if pointer != plan.fitted.pointer:
            raise ReferenceComparisonError(f"{day_key} fresh artifact is not the planned output")
        artifact = store.validate(pointer, expected=plan.fitted)
        frame = read_daily_frame(
            store,
            artifact,
            schema=FITTED_SCHEMA,
            ticker=request.ticker,
            trading_date=trading_date,
        )
        if frame.is_empty():
            raise ReferenceComparisonError(f"{day_key} fresh artifact is not populated")
        metadata = artifact.manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ReferenceComparisonError(f"{day_key} fresh metadata is malformed")
        typed_empty_reason = metadata.get("typed_empty_reason")
        statistics = validate_fitted_partition(
            frame,
            config=validation_config,
            trading_date=trading_date,
            typed_empty_reason=(
                str(typed_empty_reason) if typed_empty_reason is not None else None
            ),
            expected_raw_options_artifact_id=plan.raw.options.artifact_id,
        )
        if canonical_value(statistics.as_dict()) != fresh.get("statistics"):
            raise ReferenceComparisonError(f"{day_key} fresh statistics are stale")
        fresh_digest = _manifest_digest(artifact, "data.parquet")
        if fresh_digest != fresh.get("data_sha256"):
            raise ReferenceComparisonError(f"{day_key} fresh data digest is stale")

        legacy = day_record.get("legacy")
        if (
            not isinstance(legacy, Mapping)
            or legacy.get("file") != f"{day_key}.h5"
            or legacy.get("populated") is not True
            or not isinstance(legacy.get("sha256"), str)
            or _HEX_64.fullmatch(cast(str, legacy["sha256"])) is None
        ):
            raise ReferenceComparisonError(f"{day_key} legacy evidence is malformed")
        catalog_records = [item for item in selected_files if item.get("date") == day_key]
        if len(catalog_records) != 1:
            raise ReferenceComparisonError(f"{day_key} catalog evidence is not unique")
        catalog_record = catalog_records[0]
        if (
            catalog_record.get("file") != legacy["file"]
            or catalog_record.get("sha256") != legacy["sha256"]
            or catalog_record.get("populated") is not True
        ):
            raise ReferenceComparisonError(f"{day_key} catalog evidence is inconsistent")
        legacy_path = _assert_approval_path_has_no_symlinks(root / cast(str, legacy["file"]))
        if not legacy_path.is_file():
            raise ReferenceComparisonError(f"{day_key} legacy file is missing or unsafe")
        expected_size = catalog_record.get("byte_size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or legacy_path.stat().st_size != expected_size
        ):
            raise ReferenceComparisonError(f"{day_key} legacy file size is stale")
        legacy_digest = _approval_sha256_file(legacy_path)
        if legacy_digest != legacy["sha256"]:
            raise ReferenceComparisonError(f"{day_key} legacy file digest is stale")
        payload = read_legacy_pytree_hdf5(legacy_path)
        if not payload:
            raise ReferenceComparisonError(f"{day_key} legacy file is not populated")

        recomputed = compare_populated_day(frame, payload, tolerances=tolerances)
        if canonical_value(recomputed) != canonical_value(dict(comparison)):
            raise ReferenceComparisonError(
                f"{day_key} comparison evidence does not reproduce exactly"
            )
        approvals.append(
            {
                "date": day_key,
                "fields": list(mismatch_fields),
                "fresh_artifact_id": pointer.artifact_id,
                "fresh_data_sha256": fresh_digest,
                "legacy_file_sha256": legacy_digest,
                "explanation": explanation,
                "approver": approver,
                "approval_date": approval_date.isoformat(),
                "approval_reference": approval_reference,
            }
        )
    return validate_comparison_approval_document(
        {"schema": APPROVAL_SCHEMA, "approvals": approvals}
    )


@version("1.0", dependencies=[_assert_approval_path_has_no_symlinks])
def preflight_comparison_approval_output(output: Path, *, legacy_root: Path) -> Path:
    """Validate a no-replace destination without mutating the filesystem."""

    destination = _assert_approval_path_has_no_symlinks(output)
    parent = _assert_approval_path_has_no_symlinks(destination.parent)
    if not parent.is_dir():
        raise ReferenceComparisonError("Approval output parent must already exist")
    oracle = _assert_approval_path_has_no_symlinks(legacy_root)
    if not oracle.is_dir():
        raise ReferenceComparisonError("Legacy root is missing or unsafe")
    repository = Path(__file__).resolve().parents[3]
    if parent == oracle or oracle in parent.parents:
        raise ReferenceComparisonError("Approval output must be outside the legacy root")
    if parent == repository or repository in parent.parents:
        raise ReferenceComparisonError("Approval output must be outside the Git working tree")
    if destination.exists() or destination.is_symlink():
        raise ReferenceComparisonError("Approval output already exists")
    return destination


@version(
    "1.0",
    dependencies=[
        atomic_create_json,
        preflight_comparison_approval_output,
        validate_comparison_approval_document,
    ],
)
def write_comparison_approval_document(
    output: Path,
    document: Mapping[str, Any],
    *,
    legacy_root: Path,
) -> None:
    """Atomically publish a new exact approval document without replacement."""

    value = validate_comparison_approval_document(document)
    destination = preflight_comparison_approval_output(output, legacy_root=legacy_root)
    try:
        atomic_create_json(destination, value)
    except FileExistsError as exc:
        raise ReferenceComparisonError("Approval output already exists") from exc


__all__ = [
    "build_comparison_approval_document",
    "load_comparison_report_for_approval",
    "preflight_comparison_approval_output",
    "validate_comparison_approval_document",
    "validate_comparison_report_for_approval",
    "write_comparison_approval_document",
]
