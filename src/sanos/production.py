"""Frozen Phase 11 configuration and evidence contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

from cdxcore.version import version
from filelock import FileLock

from sanos.artifacts import ArtifactPointer, StorageClass, atomic_write_json
from sanos.config import canonical_json, canonical_sha256, canonical_value
from sanos.datasets import PublicProjectionConfig, plan_public_candidate
from sanos.models.dysanos import ExperimentConfig, plan_experiment
from sanos.reference.compare import (
    PHASE11_PRODUCTION_ORACLE_GAPS,
    REFERENCE_REPORT_SCHEMA,
)
from sanos.reference.experiment_compare import EXPERIMENT_REFERENCE_REPORT_SCHEMA
from sanos.runtime import (
    RuntimeFingerprintError,
    load_runtime_fingerprint,
    runtime_fingerprint,
)
from sanos.widb.calendar import business_dates

PHASE11_CONFIG_BUNDLE_SCHEMA = "sanos-phase11-downstream-config-bundle-v1"
PHASE11_EVIDENCE_INDEX_SCHEMA = "sanos-phase11-evidence-index-v3"
PHASE11_DATASET_NAMESPACE = "widb/internal/name-widb-spx-2022-2025-reference"
PHASE11_START_DATE = "2022-01-03"
PHASE11_END_DATE = "2025-08-29"
PHASE11_BOUNDARY_DATE = "2021-12-31"
PHASE11_SESSIONS = 918
PHASE11_POPULATED_DATES = 917
PHASE11_RAW_ARTIFACTS = 3676
PHASE11_SOURCE_GENERATION = "default-v1"
PHASE11_PUBLIC_TEMPLATE_POLICY_ID = (
    "944dbd0eca379b267a4b5a6ef88764ca15980ddb9534eab65f0ceffd6be0ec47"
)
PHASE11_EXPERIMENT_TEMPLATE_POLICY_ID = (
    "701edff74afc46b91df72de20b462f2f6d775a042a6f950231f8313c443be5bd"
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PUBLIC_PLAN_VERSION = str(plan_public_candidate.version)
_EXPERIMENT_PLAN_VERSION = str(plan_experiment.version)
_DOWNSTREAM_PLAN_VERSION_SHA256 = hashlib.sha256(
    (_PUBLIC_PLAN_VERSION + "\n" + _EXPERIMENT_PLAN_VERSION).encode("utf-8")
).hexdigest()


class Phase11EvidenceError(RuntimeError):
    """Raised when production evidence is incomplete or internally inconsistent."""


@version("1.0")
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@version("1.0")
def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase11EvidenceError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase11EvidenceError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase11EvidenceError(f"{label} must be a JSON object")
    return value


@version("1.0")
def _read_jsonl(path: Path, label: str) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file():
        raise Phase11EvidenceError(f"{label} is missing or unsafe: {path}")
    events: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase11EvidenceError(
                    f"{label} line {line_number} must be a JSON object"
                )
            events.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase11EvidenceError(f"Cannot read {label}: {exc}") from exc
    if not events:
        raise Phase11EvidenceError(f"{label} contains no structured events")
    return tuple(events)


@version("1.0")
def _one_event(
    events: Sequence[Mapping[str, Any]], event: str, label: str
) -> Mapping[str, Any]:
    selected = [value for value in events if value.get("event") == event]
    if len(selected) != 1:
        raise Phase11EvidenceError(
            f"{label} must contain exactly one {event!r} event; found {len(selected)}"
        )
    return selected[0]


@version("1.0")
def _require_fields(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    failures = {
        key: {"expected": required, "observed": value.get(key)}
        for key, required in expected.items()
        if value.get(key) != required
    }
    if failures:
        raise Phase11EvidenceError(f"{label} violates Phase 11: {failures}")


@version("1.0", dependencies=[ArtifactPointer])
def _production_pointer(value: object, label: str) -> ArtifactPointer:
    if not isinstance(value, Mapping):
        raise Phase11EvidenceError(f"{label} must be an artifact pointer")
    try:
        return ArtifactPointer(
            StorageClass(str(value["storage_class"])),
            str(value["namespace"]),
            str(value["artifact_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase11EvidenceError(f"{label} is invalid: {exc}") from exc


@version("1.1", dependencies=[_sha256_file])
def _file_evidence(path: Path, *, identity: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if identity is not None:
        result["identity"] = identity
    return result


@version("1.0", dependencies=[canonical_sha256])
def _template_policy_id(document: Mapping[str, Any], section: str) -> str:
    value = cast(dict[str, Any], canonical_value(document))
    section_value = value.get(section)
    if not isinstance(section_value, dict) or "source_dataset" not in section_value:
        raise ValueError(f"{section} template has no source_dataset")
    section_value.pop("source_dataset")
    return canonical_sha256(value)


@version(
    "1.2",
    dependencies=[
        PublicProjectionConfig,
        ExperimentConfig,
        _template_policy_id,
        canonical_sha256,
        plan_public_candidate,
    ],
)
def finalize_phase11_downstream_configs(
    dataset: ArtifactPointer,
    *,
    public_template: Mapping[str, Any],
    experiment_template: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind public projection to internal data and Experiment to that projection."""

    if (
        dataset.storage_class is not StorageClass.DATASET
        or dataset.namespace != PHASE11_DATASET_NAMESPACE
        or _HEX_64.fullmatch(dataset.artifact_id) is None
    ):
        raise ValueError("Phase 11 requires the exact reference internal Dataset namespace")
    public_document = cast(dict[str, Any], canonical_value(public_template))
    experiment_document = cast(dict[str, Any], canonical_value(experiment_template))
    if (
        _template_policy_id(public_document, "public_dataset")
        != PHASE11_PUBLIC_TEMPLATE_POLICY_ID
        or _template_policy_id(experiment_document, "experiment")
        != PHASE11_EXPERIMENT_TEMPLATE_POLICY_ID
    ):
        raise ValueError("Downstream template policy differs from the frozen Phase 11 profile")
    if set(public_document) != {"public_dataset"} or not isinstance(
        public_document["public_dataset"], dict
    ):
        raise ValueError("Public template must contain exactly public_dataset")
    if not isinstance(experiment_document.get("experiment"), dict):
        raise ValueError("Experiment template has no experiment section")
    public_document["public_dataset"]["source_dataset"] = dataset.as_dict()
    public_config = PublicProjectionConfig.from_document(public_document)
    public_pointer = plan_public_candidate(public_config).dataset_spec.pointer
    experiment_document["experiment"]["source_dataset"] = public_pointer.as_dict()
    experiment_config = ExperimentConfig.from_document(experiment_document)
    public_final = public_config.to_document()
    experiment_final = experiment_config.to_document()
    body = cast(
        dict[str, Any],
        canonical_value(
            {
                "schema": PHASE11_CONFIG_BUNDLE_SCHEMA,
                "source_dataset": dataset,
                "public_candidate": {
                    "file": "public-candidate.json",
                    "configuration_id": canonical_sha256(public_final),
                    "configuration": public_final,
                },
                "experiment": {
                    "file": "experiment.json",
                    "configuration_id": canonical_sha256(experiment_final),
                    "configuration": experiment_final,
                },
            }
        ),
    )
    return {"bundle_id": canonical_sha256(body), **body}


@version("1.1", dependencies=[atomic_write_json, finalize_phase11_downstream_configs])
def write_phase11_downstream_configs(output_directory: Path, bundle: Mapping[str, Any]) -> None:
    """Write final configs and their binding manifest without replacing evidence."""

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_directory.parent / f".{output_directory.name}.lock"))
    with lock:
        if output_directory.exists() or output_directory.is_symlink():
            raise Phase11EvidenceError(
                f"Configuration bundle output already exists: {output_directory}"
            )
        staging = output_directory.parent / (
            f".{output_directory.name}.publish-{uuid.uuid4().hex}"
        )
        staging.mkdir()
        try:
            public = cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], bundle["public_candidate"])["configuration"],
            )
            experiment = cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], bundle["experiment"])["configuration"],
            )
            atomic_write_json(staging / "public-candidate.json", public)
            atomic_write_json(staging / "experiment.json", experiment)
            atomic_write_json(staging / "bundle.json", bundle)
            staging.rename(output_directory)
            descriptor = os.open(
                output_directory.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise


@version(
    f"1.3-plans-{_DOWNSTREAM_PLAN_VERSION_SHA256}",
    dependencies=[
        _read_json,
        _read_jsonl,
        _sha256_file,
        _file_evidence,
        _one_event,
        _production_pointer,
        _require_fields,
        canonical_json,
        canonical_sha256,
        business_dates,
        finalize_phase11_downstream_configs,
        load_runtime_fingerprint,
        runtime_fingerprint,
    ],
)
def build_phase11_evidence_index(
    *,
    run_id: str,
    source_revision: str,
    runtime_fingerprint_path: Path,
    raw_preflight_path: Path,
    raw_validation_path: Path,
    dataset_validation_path: Path,
    daily_report_path: Path,
    downstream_bundle_path: Path,
    public_validation_path: Path,
    experiment_validation_path: Path,
    experiment_report_path: Path,
) -> dict[str, Any]:
    """Validate and bind the complete pre-review Phase 11 evidence set."""

    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise Phase11EvidenceError("run_id must be a portable 1-128 character label")
    if _HEX_40.fullmatch(source_revision) is None:
        raise Phase11EvidenceError("source_revision must be exact lowercase 40-hex")

    try:
        runtime = load_runtime_fingerprint(runtime_fingerprint_path)
        current_runtime = runtime_fingerprint()
    except RuntimeFingerprintError as exc:
        raise Phase11EvidenceError(f"Runtime fingerprint is invalid: {exc}") from exc
    if canonical_json(runtime) != canonical_json(current_runtime):
        raise Phase11EvidenceError(
            "Runtime fingerprint differs from the active evidence-index environment"
        )

    preflight_events = _read_jsonl(raw_preflight_path, "Raw preflight")
    preflight_plan = _one_event(preflight_events, "raw_plan", "Raw preflight")
    preflight_result = _one_event(preflight_events, "raw_result", "Raw preflight")
    common_raw = {
        "source": "optionmetrics",
        "source_generation": PHASE11_SOURCE_GENERATION,
        "ticker": "SPX",
        "start_date": PHASE11_START_DATE,
        "end_date": PHASE11_END_DATE,
        "planned": PHASE11_RAW_ARTIFACTS,
        "network_accessed": False,
    }
    _require_fields(
        preflight_plan,
        {
            **common_raw,
            "reused": 0,
            "missing": PHASE11_RAW_ARTIFACTS,
            "boundary_partitions": 3,
            "dry_run": True,
        },
        "Raw preflight plan",
    )
    _require_fields(
        preflight_result,
        {
            **common_raw,
            "reused": 0,
            "downloaded": 0,
            "missing": PHASE11_RAW_ARTIFACTS,
            "dry_run": True,
        },
        "Raw preflight result",
    )
    expected_boundary = [
        {
            "table": table,
            "ticker": "SPX",
            "observation_date": PHASE11_BOUNDARY_DATE,
            "roles": ["boundary"],
        }
        for table in ("forwards", "spot", "zero_curve")
    ]
    if preflight_plan.get("boundary_dependencies") != expected_boundary:
        raise Phase11EvidenceError("Raw preflight boundary dependencies changed")

    raw_events = _read_jsonl(raw_validation_path, "Raw validation")
    raw_plan = _one_event(raw_events, "raw_plan", "Raw validation")
    raw_result = _one_event(raw_events, "raw_result", "Raw validation")
    _require_fields(
        raw_plan,
        {
            **common_raw,
            "reused": PHASE11_RAW_ARTIFACTS,
            "missing": 0,
            "boundary_partitions": 3,
            "dry_run": False,
        },
        "Raw validation plan",
    )
    _require_fields(
        raw_result,
        {
            **common_raw,
            "reused": PHASE11_RAW_ARTIFACTS,
            "downloaded": 0,
            "missing": 0,
            "dry_run": False,
        },
        "Raw validation result",
    )
    raw_artifacts = [
        value for value in raw_events if value.get("event") == "raw_artifact_validated"
    ]
    try:
        raw_pointers = tuple(
            _production_pointer(value.get("artifact"), "Raw validation artifact")
            for value in raw_artifacts
        )
    except Phase11EvidenceError:
        raise
    raw_ids = set(raw_pointers)
    raw_table_counts = Counter(value.get("logical_table") for value in raw_artifacts)
    boundary_records = {
        value.get("logical_table")
        for value in raw_artifacts
        if value.get("observation_date") == PHASE11_BOUNDARY_DATE
    }
    sessions = tuple(
        value.isoformat()
        for value in business_dates(
            date.fromisoformat(PHASE11_START_DATE),
            date.fromisoformat(PHASE11_END_DATE),
        )
    )
    if len(sessions) != PHASE11_SESSIONS:
        raise Phase11EvidenceError("Phase 11 software calendar no longer has 918 sessions")
    observations_by_table = {
        table: tuple(
            sorted(
                (
                    cast(str | None, value.get("observation_date"))
                    for value in raw_artifacts
                    if value.get("logical_table") == table
                ),
                key=lambda item: (item is not None, item or ""),
            )
        )
        for table in ("security", "options", "spot", "forwards", "zero_curve")
    }
    expected_observations = {
        "security": (None,),
        "options": sessions,
        "spot": tuple(sorted((PHASE11_BOUNDARY_DATE, *sessions))),
        "forwards": tuple(sorted((PHASE11_BOUNDARY_DATE, *sessions))),
        "zero_curve": tuple(sorted((PHASE11_BOUNDARY_DATE, *sessions))),
    }
    if (
        len(raw_artifacts) != PHASE11_RAW_ARTIFACTS
        or len(raw_ids) != PHASE11_RAW_ARTIFACTS
        or any(pointer.storage_class is not StorageClass.RAW for pointer in raw_pointers)
        or raw_table_counts
        != Counter(
            {
                "security": 1,
                "options": 918,
                "spot": 919,
                "forwards": 919,
                "zero_curve": 919,
            }
        )
        or boundary_records != {"spot", "forwards", "zero_curve"}
        or observations_by_table != expected_observations
        or any(
            value.get("ticker") != "SPX"
            or value.get("source_generation") != PHASE11_SOURCE_GENERATION
            for value in raw_artifacts
        )
    ):
        raise Phase11EvidenceError(
            "Raw validation does not contain the exact 3,676-artifact SPX closure"
        )

    dataset_events = _read_jsonl(dataset_validation_path, "Dataset validation")
    dataset_event = _one_event(
        dataset_events, "dataset_validation_result", "Dataset validation"
    )
    _require_fields(
        dataset_event,
        {
            "status": "passed",
            "source_revision": source_revision,
            "semantic": True,
            "deep_lineage": True,
            "network_accessed": False,
            "filesystem_modified": False,
        },
        "Dataset validation",
    )
    dataset_pointer = _production_pointer(
        dataset_event.get("artifact"), "Dataset validation artifact"
    )
    if dataset_pointer.namespace != PHASE11_DATASET_NAMESPACE:
        raise Phase11EvidenceError("Dataset validation names the wrong Phase 11 namespace")
    for field in ("manifest_sha256", "configuration_id"):
        if not isinstance(dataset_event.get(field), str) or _HEX_64.fullmatch(
            str(dataset_event[field])
        ) is None:
            raise Phase11EvidenceError(f"Dataset validation has no exact {field}")
    dataset_summary = dataset_event.get("summary")
    if not isinstance(dataset_summary, Mapping):
        raise Phase11EvidenceError("Dataset validation has no semantic summary")
    _require_fields(
        dataset_summary,
        {"requested_dates": 918, "populated_dates": 917, "typed_empty_dates": 1},
        "Dataset validation summary",
    )

    daily = _read_json(daily_report_path, "daily comparison report")
    daily_body = {key: value for key, value in daily.items() if key != "report_id"}
    _require_fields(
        daily,
        {
            "schema": REFERENCE_REPORT_SCHEMA,
            "outcome": "passed",
            "source_revision": source_revision,
            "network_accessed": False,
            "artifacts_built": False,
        },
        "Daily comparison report",
    )
    if daily.get("report_id") != canonical_sha256(daily_body):
        raise Phase11EvidenceError("Daily comparison report identity is invalid")
    request = daily.get("request")
    production_contract = daily.get("production_contract")
    if not isinstance(request, Mapping) or not isinstance(production_contract, Mapping):
        raise Phase11EvidenceError("Daily comparison lacks its Phase 11 contract")
    _require_fields(
        request,
        {
            "ticker": "SPX",
            "start_date": PHASE11_START_DATE,
            "end_date": PHASE11_END_DATE,
            "source": "optionmetrics",
            "source_generation": PHASE11_SOURCE_GENERATION,
            "declared_oracle_gaps": list(PHASE11_PRODUCTION_ORACLE_GAPS),
            "enforce_phase11_production": True,
        },
        "Daily comparison request",
    )
    if production_contract.get("passed") is not True:
        raise Phase11EvidenceError("Daily Phase 11 production contract did not pass")
    counts = daily.get("classification_counts")
    if not isinstance(counts, Mapping):
        raise Phase11EvidenceError("Daily comparison has no classification counts")
    allowed_count_keys = {
        "match",
        "approved_discrepancy",
        "oracle_gap",
        "policy_empty",
        "unexplained_discrepancy",
    }
    if (
        not set(counts).issubset(allowed_count_keys)
        or counts.get("oracle_gap") != 26
        or counts.get("policy_empty") != 1
        or counts.get("unexplained_discrepancy", 0) != 0
        or counts.get("match", 0) + counts.get("approved_discrepancy", 0) != 891
        or (
            counts.get("approved_discrepancy", 0) != 0
            and not isinstance(daily.get("approval_document"), Mapping)
        )
    ):
        raise Phase11EvidenceError(
            f"Daily classification counts violate Phase 11: {dict(counts)}"
        )

    bundle = _read_json(downstream_bundle_path, "downstream configuration bundle")
    bundle_body = {key: value for key, value in bundle.items() if key != "bundle_id"}
    if (
        bundle.get("schema") != PHASE11_CONFIG_BUNDLE_SCHEMA
        or bundle.get("bundle_id") != canonical_sha256(bundle_body)
        or _production_pointer(bundle.get("source_dataset"), "bundle source")
        != dataset_pointer
    ):
        raise Phase11EvidenceError("Downstream configuration bundle identity is invalid")
    public_section = bundle.get("public_candidate")
    experiment_section = bundle.get("experiment")
    if not isinstance(public_section, Mapping) or not isinstance(experiment_section, Mapping):
        raise Phase11EvidenceError("Downstream bundle sections are missing")
    public_config = PublicProjectionConfig.from_document(
        cast(Mapping[str, Any], public_section.get("configuration"))
    )
    experiment_config = ExperimentConfig.from_document(
        cast(Mapping[str, Any], experiment_section.get("configuration"))
    )
    if (
        public_config.source_dataset != dataset_pointer
        or experiment_config.source_dataset.pointer
        != plan_public_candidate(public_config).dataset_spec.pointer
        or public_section.get("configuration_id")
        != canonical_sha256(public_config.to_document())
        or experiment_section.get("configuration_id")
        != canonical_sha256(experiment_config.to_document())
    ):
        raise Phase11EvidenceError("Downstream configurations are not bound to the Dataset")

    public_events = _read_jsonl(public_validation_path, "public-candidate validation")
    public_event = _one_event(
        public_events, "public_dataset_validation_result", "public-candidate validation"
    )
    _require_fields(
        public_event,
        {
            "status": "passed",
            "distribution": "public_candidate",
            "semantic": True,
            "deep_lineage": True,
            "network_accessed": False,
            "filesystem_modified": False,
        },
        "Public-candidate validation",
    )
    public_pointer = _production_pointer(
        public_event.get("artifact"), "public-candidate artifact"
    )
    public_event_config = public_event.get("config")
    if canonical_json(public_event_config) != canonical_json(public_config):
        raise Phase11EvidenceError("Public validation used a different final configuration")
    if public_pointer != plan_public_candidate(public_config).dataset_spec.pointer:
        raise Phase11EvidenceError("Public validation artifact identity is not canonical")
    public_manifest_sha256 = public_event.get("manifest_sha256")
    if (
        not isinstance(public_manifest_sha256, str)
        or _HEX_64.fullmatch(public_manifest_sha256) is None
    ):
        raise Phase11EvidenceError(
            "Public validation has no exact public-candidate manifest_sha256"
        )

    experiment_events = _read_jsonl(experiment_validation_path, "Experiment validation")
    experiment_event = _one_event(
        experiment_events, "experiment_validation_result", "Experiment validation"
    )
    _require_fields(
        experiment_event,
        {
            "status": "passed",
            "semantic": True,
            "deep_lineage": True,
            "network_accessed": False,
            "filesystem_modified": False,
        },
        "Experiment validation",
    )
    experiment_pointer = _production_pointer(
        experiment_event.get("artifact"), "Experiment artifact"
    )
    if canonical_json(experiment_event.get("config")) != canonical_json(
        experiment_config.to_document()
    ):
        raise Phase11EvidenceError("Experiment validation used a different final configuration")
    if experiment_pointer != plan_experiment(experiment_config).spec.pointer:
        raise Phase11EvidenceError("Experiment validation artifact identity is not canonical")

    experiment_report = _read_json(experiment_report_path, "Experiment comparison report")
    experiment_body = {
        key: value for key, value in experiment_report.items() if key != "report_id"
    }
    _require_fields(
        experiment_report,
        {
            "schema": EXPERIMENT_REFERENCE_REPORT_SCHEMA,
            "outcome": "passed",
            "source_revision": source_revision,
            "network_accessed": False,
            "artifacts_built": False,
        },
        "Experiment comparison report",
    )
    if experiment_report.get("report_id") != canonical_sha256(experiment_body):
        raise Phase11EvidenceError("Experiment comparison report identity is invalid")
    if cast(Mapping[str, Any], experiment_report.get("phase11_contract", {})).get(
        "passed"
    ) is not True:
        raise Phase11EvidenceError("Experiment Phase 11 production contract did not pass")
    experiment_record = experiment_report.get("experiment")
    if (
        not isinstance(experiment_record, Mapping)
        or not isinstance(experiment_record.get("manifest_sha256"), str)
        or _HEX_64.fullmatch(str(experiment_record["manifest_sha256"])) is None
        or _production_pointer(
            experiment_record.get("artifact"), "Experiment report artifact"
        )
        != experiment_pointer
        or _production_pointer(
            experiment_record.get("source_dataset"), "Experiment report source"
        )
        != public_pointer
    ):
        raise Phase11EvidenceError("Experiment report is not bound to final artifacts")

    body = cast(
        dict[str, Any],
        canonical_value(
            {
                "schema": PHASE11_EVIDENCE_INDEX_SCHEMA,
                "run_id": run_id,
                "source_revision": source_revision,
                "network_accessed_during_validation": False,
                "source_generation": PHASE11_SOURCE_GENERATION,
                "resolved_versions": {
                    "public_plan": _PUBLIC_PLAN_VERSION,
                    "experiment_plan": _EXPERIMENT_PLAN_VERSION,
                    "runtime_fingerprint": runtime["fingerprint_id"],
                },
                "coverage": {
                    "ticker": "SPX",
                    "start_date": PHASE11_START_DATE,
                    "end_date": PHASE11_END_DATE,
                    "sessions": PHASE11_SESSIONS,
                    "populated_dates": PHASE11_POPULATED_DATES,
                    "typed_empty_dates": ["2025-01-08"],
                    "oracle_gaps": PHASE11_PRODUCTION_ORACLE_GAPS,
                },
                "artifacts": {
                    "internal_dataset": {
                        **dataset_pointer.as_dict(),
                        "manifest_sha256": dataset_event["manifest_sha256"],
                        "configuration_id": dataset_event["configuration_id"],
                        "source_revision": dataset_event["source_revision"],
                    },
                    "public_candidate": {
                        **public_pointer.as_dict(),
                        "manifest_sha256": public_manifest_sha256,
                    },
                    "experiment": {
                        **experiment_pointer.as_dict(),
                        "manifest_sha256": experiment_record["manifest_sha256"],
                    },
                },
                "artifact_binding": None,
                "evidence": {
                    "runtime_fingerprint": _file_evidence(
                        runtime_fingerprint_path,
                        identity=str(runtime["fingerprint_id"]),
                    ),
                    "raw_preflight": _file_evidence(raw_preflight_path),
                    "raw_validation": _file_evidence(raw_validation_path),
                    "dataset_validation": _file_evidence(
                        dataset_validation_path, identity=dataset_pointer.artifact_id
                    ),
                    "daily_comparison": _file_evidence(
                        daily_report_path, identity=str(daily["report_id"])
                    ),
                    "downstream_configs": _file_evidence(
                        downstream_bundle_path, identity=str(bundle["bundle_id"])
                    ),
                    "public_candidate_validation": _file_evidence(
                        public_validation_path, identity=public_pointer.artifact_id
                    ),
                    "experiment_validation": _file_evidence(
                        experiment_validation_path, identity=experiment_pointer.artifact_id
                    ),
                    "experiment_comparison": _file_evidence(
                        experiment_report_path,
                        identity=str(experiment_report["report_id"]),
                    ),
                },
            }
        ),
    )
    return {"evidence_id": canonical_sha256(body), **body}


@version("1.1", dependencies=[_production_pointer])
def validate_phase11_public_candidate_binding(
    index: Mapping[str, Any],
    *,
    public_candidate: ArtifactPointer,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Bind indexed public-candidate evidence to the manifest bytes on disk."""

    if index.get("schema") != PHASE11_EVIDENCE_INDEX_SCHEMA:
        raise Phase11EvidenceError("Cannot bind an unsupported evidence index")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Phase11EvidenceError("Evidence index has no artifact identities")
    indexed = artifacts.get("public_candidate")
    if not isinstance(indexed, Mapping):
        raise Phase11EvidenceError("Evidence index has no public-candidate identity")
    indexed_pointer = _production_pointer(indexed, "indexed public candidate")
    indexed_manifest = indexed.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or _HEX_64.fullmatch(manifest_sha256) is None
        or not isinstance(indexed_manifest, str)
        or _HEX_64.fullmatch(indexed_manifest) is None
        or indexed_pointer != public_candidate
        or indexed_manifest != manifest_sha256
    ):
        raise Phase11EvidenceError(
            "Public-candidate artifact or manifest differs from the indexed validation"
        )
    return {
        "status": "passed",
        "public_candidate": public_candidate.as_dict(),
        "public_candidate_manifest_sha256": manifest_sha256,
    }


@version("1.2", dependencies=[canonical_sha256])
def attach_phase11_artifact_binding(
    index: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach byte-level Dataset/report/Experiment binding to a checked index."""

    if index.get("schema") != PHASE11_EVIDENCE_INDEX_SCHEMA:
        raise Phase11EvidenceError("Cannot bind an unsupported evidence index")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Phase11EvidenceError("Evidence index has no artifact identities")
    dataset = artifacts.get("internal_dataset")
    public_candidate = artifacts.get("public_candidate")
    experiment = artifacts.get("experiment")
    partition = binding.get("partition_binding")
    public_binding = binding.get("public_candidate_binding")
    if (
        binding.get("status") != "passed"
        or not isinstance(dataset, Mapping)
        or not isinstance(public_candidate, Mapping)
        or not isinstance(experiment, Mapping)
        or not isinstance(partition, Mapping)
        or not isinstance(public_binding, Mapping)
        or partition.get("status") != "passed"
        or partition.get("partition_count") != PHASE11_SESSIONS
        or binding.get("dataset")
        != {key: dataset.get(key) for key in ("storage_class", "namespace", "artifact_id")}
        or binding.get("dataset_manifest_sha256") != dataset.get("manifest_sha256")
        or binding.get("dataset_configuration_id") != dataset.get("configuration_id")
        or binding.get("dataset_source_revision") != dataset.get("source_revision")
        or binding.get("experiment")
        != {
            key: experiment.get(key)
            for key in ("storage_class", "namespace", "artifact_id")
        }
        or binding.get("experiment_manifest_sha256")
        != experiment.get("manifest_sha256")
        or public_binding.get("status") != "passed"
        or public_binding.get("public_candidate")
        != {
            key: public_candidate.get(key)
            for key in ("storage_class", "namespace", "artifact_id")
        }
        or public_binding.get("public_candidate_manifest_sha256")
        != public_candidate.get("manifest_sha256")
        or binding.get("daily_report_id")
        != cast(Mapping[str, Any], index["evidence"])["daily_comparison"]["identity"]
        or binding.get("daily_report_sha256")
        != cast(Mapping[str, Any], index["evidence"])["daily_comparison"]["sha256"]
        or binding.get("experiment_report_id")
        != cast(Mapping[str, Any], index["evidence"])["experiment_comparison"][
            "identity"
        ]
        or binding.get("experiment_report_sha256")
        != cast(Mapping[str, Any], index["evidence"])["experiment_comparison"][
            "sha256"
        ]
    ):
        raise Phase11EvidenceError("Artifact binding differs from the indexed evidence")
    body = cast(
        dict[str, Any],
        canonical_value(
            {
                **{key: value for key, value in index.items() if key != "evidence_id"},
                "artifact_binding": binding,
            }
        ),
    )
    return {"evidence_id": canonical_sha256(body), **body}


@version("1.2", dependencies=[atomic_write_json])
def write_phase11_evidence_index(output: Path, index: Mapping[str, Any]) -> bool:
    """Write once, or shallowly reuse an identical complete evidence index."""

    binding = index.get("artifact_binding")
    if not isinstance(binding, Mapping) or binding.get("status") != "passed":
        raise Phase11EvidenceError("Evidence index lacks byte-level artifact binding")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output.parent / f".{output.name}.lock"))
    with lock:
        if output.exists() or output.is_symlink():
            existing = _read_json(output, "existing Phase 11 evidence index")
            if canonical_json(existing) != canonical_json(index):
                raise Phase11EvidenceError(
                    f"Evidence index already exists with other content: {output}"
                )
            return False
        atomic_write_json(output, index)
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True


__all__ = [
    "PHASE11_CONFIG_BUNDLE_SCHEMA",
    "PHASE11_EVIDENCE_INDEX_SCHEMA",
    "Phase11EvidenceError",
    "attach_phase11_artifact_binding",
    "build_phase11_evidence_index",
    "finalize_phase11_downstream_configs",
    "validate_phase11_public_candidate_binding",
    "write_phase11_downstream_configs",
    "write_phase11_evidence_index",
]
