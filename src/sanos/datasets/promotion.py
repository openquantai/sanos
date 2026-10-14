"""Immutable designation of a validated Dataset as the canonical SANOS reference."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

from cdxcore.version import version

from sanos.artifacts import (
    ArtifactExistsError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    DatasetDistribution,
    DistributionStatus,
    ReuseResult,
    StorageClass,
    ValidatedArtifact,
)
from sanos.config import StorageLocations, canonical_sha256, canonical_value
from sanos.datasets.internal import validate_internal_dataset
from sanos.experiments import validate_experiment
from sanos.reference import (
    EXPERIMENT_REFERENCE_REPORT_SCHEMA,
    REFERENCE_REPORT_SCHEMA,
)
from sanos.runtime import RuntimeFingerprintError, load_runtime_fingerprint

PROMOTION_CONFIG_SCHEMA = "sanos-canonical-reference-promotion-config-v2"
PROMOTION_SCHEMA = "sanos-canonical-reference-promotion-v3"
REVIEW_ATTESTATION_SCHEMA = "sanos-canonical-reference-review-v2"
_PHASE11_EVIDENCE_INDEX_SCHEMA = "sanos-phase11-evidence-index-v3"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ReferencePromotionError(RuntimeError):
    """Raised when reference-promotion evidence is missing, stale, or malformed."""


@version("1.0")
def _promotion_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@version("1.0")
def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReferencePromotionError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferencePromotionError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferencePromotionError(f"{label} must be a JSON object")
    return value


@version("1.0", dependencies=[ArtifactPointer])
def _pointer(value: object, label: str) -> ArtifactPointer:
    if not isinstance(value, Mapping) or set(value) != {
        "storage_class",
        "namespace",
        "artifact_id",
    }:
        raise ReferencePromotionError(f"{label} is not an exact artifact pointer")
    try:
        return ArtifactPointer(
            storage_class=StorageClass(value["storage_class"]),
            namespace=str(value["namespace"]),
            artifact_id=str(value["artifact_id"]),
        )
    except (TypeError, ValueError) as exc:
        raise ReferencePromotionError(f"{label} is invalid: {exc}") from exc


@version("1.0")
def _digest(value: object, label: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ReferencePromotionError(f"{label} must be a lowercase SHA-256 digest")
    return value


@version(
    "1.0",
    dependencies=[_digest, _load_json, _promotion_sha256_file, canonical_sha256],
)
def _verified_report(path: Path, *, schema: str, label: str) -> tuple[dict[str, Any], str]:
    report = _load_json(path, label)
    if report.get("schema") != schema or report.get("outcome") != "passed":
        raise ReferencePromotionError(f"{label} is not a passed {schema} report")
    report_id = _digest(report.get("report_id"), f"{label} report_id")
    without_id = {key: value for key, value in report.items() if key != "report_id"}
    if canonical_sha256(without_id) != report_id:
        raise ReferencePromotionError(f"{label} report_id does not bind its content")
    return report, _promotion_sha256_file(path)


@version(
    "1.2",
    dependencies=[
        _digest,
        _load_json,
        _pointer,
        _promotion_sha256_file,
        canonical_sha256,
        canonical_value,
    ],
)
def _verified_evidence_index(
    path: Path,
    *,
    expected_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Verify the immutable Phase 11 index and its core artifact binding."""

    index = _load_json(path, "Phase 11 evidence index")
    required_fields = {
        "evidence_id",
        "schema",
        "run_id",
        "source_revision",
        "network_accessed_during_validation",
        "source_generation",
        "resolved_versions",
        "coverage",
        "artifacts",
        "artifact_binding",
        "evidence",
    }
    if set(index) != required_fields or index.get("schema") != (
        _PHASE11_EVIDENCE_INDEX_SCHEMA
    ):
        raise ReferencePromotionError("Phase 11 evidence index fields or schema changed")
    if (
        not isinstance(index.get("run_id"), str)
        or not str(index["run_id"]).strip()
        or _HEX_40.fullmatch(str(index.get("source_revision"))) is None
        or index.get("network_accessed_during_validation") is not False
        or not isinstance(index.get("source_generation"), str)
        or not isinstance(index.get("resolved_versions"), Mapping)
        or not isinstance(index.get("coverage"), Mapping)
        or not isinstance(index.get("artifacts"), Mapping)
        or not isinstance(index.get("evidence"), Mapping)
    ):
        raise ReferencePromotionError("Phase 11 evidence index metadata is malformed")
    if index.get("source_revision") != expected_binding.get("dataset_source_revision"):
        raise ReferencePromotionError("Phase 11 evidence index source revision differs")
    expected_evidence = {
        "runtime_fingerprint",
        "raw_preflight",
        "raw_validation",
        "dataset_validation",
        "daily_comparison",
        "downstream_configs",
        "public_candidate_validation",
        "experiment_validation",
        "experiment_comparison",
    }
    evidence = cast(Mapping[str, Any], index["evidence"])
    if set(evidence) != expected_evidence or any(
        not isinstance(value, Mapping)
        or not isinstance(value.get("file"), str)
        or _HEX_64.fullmatch(str(value.get("sha256"))) is None
        or not isinstance(value.get("bytes"), int)
        or value.get("bytes", -1) < 0
        for value in evidence.values()
    ):
        raise ReferencePromotionError("Phase 11 evidence file inventory is malformed")
    runtime_evidence = evidence["runtime_fingerprint"]
    if (
        set(runtime_evidence) != {"file", "sha256", "bytes", "identity"}
        or runtime_evidence.get("file") != "runtime-fingerprint.json"
        or _HEX_64.fullmatch(str(runtime_evidence.get("identity"))) is None
    ):
        raise ReferencePromotionError("Phase 11 runtime evidence is malformed")
    resolved_versions = cast(Mapping[str, Any], index["resolved_versions"])
    if (
        set(resolved_versions)
        != {"public_plan", "experiment_plan", "runtime_fingerprint"}
        or any(
            not isinstance(item, str) or not item
            for item in resolved_versions.values()
        )
        or resolved_versions.get("runtime_fingerprint")
        != runtime_evidence.get("identity")
    ):
        raise ReferencePromotionError(
            "Phase 11 resolved runtime identity differs from runtime evidence"
        )
    evidence_id = _digest(index.get("evidence_id"), "Phase 11 evidence_id")
    body = {key: value for key, value in index.items() if key != "evidence_id"}
    if canonical_sha256(body) != evidence_id:
        raise ReferencePromotionError("Phase 11 evidence_id does not bind its content")
    binding = index.get("artifact_binding")
    binding_fields = {
        "status",
        "dataset",
        "dataset_manifest_sha256",
        "dataset_configuration_id",
        "dataset_source_revision",
        "daily_report_id",
        "daily_report_sha256",
        "daily_approvals_sha256",
        "partition_binding",
        "public_candidate_binding",
        "experiment",
        "experiment_manifest_sha256",
        "experiment_report_id",
        "experiment_report_sha256",
    }
    if (
        not isinstance(binding, Mapping)
        or set(binding) != binding_fields
        or binding.get("status") != "passed"
    ):
        raise ReferencePromotionError("Phase 11 evidence index has no passed artifact binding")
    public_binding = binding.get("public_candidate_binding")
    if (
        not isinstance(public_binding, Mapping)
        or set(public_binding)
        != {
            "status",
            "public_candidate",
            "public_candidate_manifest_sha256",
        }
        or public_binding.get("status") != "passed"
        or _HEX_64.fullmatch(
            str(public_binding.get("public_candidate_manifest_sha256"))
        )
        is None
    ):
        raise ReferencePromotionError(
            "Phase 11 evidence index has no public-candidate manifest binding"
        )
    _pointer(
        public_binding.get("public_candidate"),
        "Phase 11 public-candidate binding",
    )
    artifacts = cast(Mapping[str, Any], index["artifacts"])
    public_artifact = artifacts.get("public_candidate")
    if (
        not isinstance(public_artifact, Mapping)
        or _pointer(
            {
                key: public_artifact.get(key)
                for key in ("storage_class", "namespace", "artifact_id")
            },
            "Phase 11 indexed public candidate",
        )
        != _pointer(
            public_binding["public_candidate"],
            "Phase 11 public-candidate binding",
        )
        or public_artifact.get("manifest_sha256")
        != public_binding.get("public_candidate_manifest_sha256")
    ):
        raise ReferencePromotionError(
            "Phase 11 public-candidate binding differs from indexed artifacts"
        )
    for name, expected in expected_binding.items():
        if canonical_value(binding.get(name)) != canonical_value(expected):
            raise ReferencePromotionError(
                f"Phase 11 evidence index does not bind exact {name}"
            )
    return index, _promotion_sha256_file(path)


@version(
    "1.0",
    dependencies=[_promotion_sha256_file, load_runtime_fingerprint],
)
def _verified_runtime_evidence(
    index: Mapping[str, Any], evidence_index_path: Path
) -> tuple[dict[str, Any], Path, str]:
    """Verify the runtime file named and hashed by a checked evidence index."""

    evidence = index.get("evidence")
    runtime_record = (
        evidence.get("runtime_fingerprint") if isinstance(evidence, Mapping) else None
    )
    if not isinstance(runtime_record, Mapping):
        raise ReferencePromotionError("Phase 11 index has no runtime fingerprint")
    file_name = runtime_record.get("file")
    if file_name != "runtime-fingerprint.json":
        raise ReferencePromotionError("Phase 11 runtime filename changed")
    path = evidence_index_path.parent / file_name
    try:
        fingerprint = load_runtime_fingerprint(path)
    except RuntimeFingerprintError as exc:
        raise ReferencePromotionError(f"Runtime fingerprint is invalid: {exc}") from exc
    sha256 = _promotion_sha256_file(path)
    if (
        runtime_record.get("identity") != fingerprint.get("fingerprint_id")
        or runtime_record.get("sha256") != sha256
        or runtime_record.get("bytes") != path.stat().st_size
    ):
        raise ReferencePromotionError("Runtime fingerprint differs from evidence index")
    return fingerprint, path, sha256


@version("1.1")
@dataclass(frozen=True, slots=True)
class CanonicalReferencePromotionConfig:
    """Portable identities required to create or output-first reuse a designation."""

    name: str
    dataset: ArtifactPointer
    dataset_manifest_sha256: str
    dataset_configuration_id: str
    dataset_source_revision: str
    daily_report_id: str
    daily_report_sha256: str
    daily_approvals_sha256: str | None
    experiment: ArtifactPointer
    experiment_manifest_sha256: str
    experiment_report_id: str
    experiment_report_sha256: str
    evidence_index_id: str
    evidence_index_sha256: str
    code_review_sha256: str
    quantitative_review_sha256: str

    def __post_init__(self) -> None:
        if _SAFE_NAME.fullmatch(self.name) is None:
            raise ValueError("Promotion name must be a portable 1-128 character label")
        if self.dataset.storage_class is not StorageClass.DATASET:
            raise ValueError("Promotion source must be a Dataset")
        if self.experiment.storage_class is not StorageClass.EXPERIMENT:
            raise ValueError("Promotion Experiment evidence must identify an Experiment")
        for name in (
            "dataset_manifest_sha256",
            "dataset_configuration_id",
            "daily_report_id",
            "daily_report_sha256",
            "experiment_manifest_sha256",
            "experiment_report_id",
            "experiment_report_sha256",
            "evidence_index_id",
            "evidence_index_sha256",
            "code_review_sha256",
            "quantitative_review_sha256",
        ):
            if _HEX_64.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.daily_approvals_sha256 is not None and _HEX_64.fullmatch(
            self.daily_approvals_sha256
        ) is None:
            raise ValueError("daily_approvals_sha256 must be null or lowercase SHA-256")
        if _HEX_40.fullmatch(self.dataset_source_revision) is None:
            raise ValueError("dataset_source_revision must be exact lowercase 40-hex")

    def to_document(self) -> dict[str, Any]:
        return cast(dict[str, Any], canonical_value(self))

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> CanonicalReferencePromotionConfig:
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("Promotion configuration fields are missing or unexpected")
        return cls(
            name=str(value["name"]),
            dataset=_pointer(value["dataset"], "promotion Dataset"),
            dataset_manifest_sha256=str(value["dataset_manifest_sha256"]),
            dataset_configuration_id=str(value["dataset_configuration_id"]),
            dataset_source_revision=str(value["dataset_source_revision"]),
            daily_report_id=str(value["daily_report_id"]),
            daily_report_sha256=str(value["daily_report_sha256"]),
            daily_approvals_sha256=cast(str | None, value["daily_approvals_sha256"]),
            experiment=_pointer(value["experiment"], "promotion Experiment"),
            experiment_manifest_sha256=str(value["experiment_manifest_sha256"]),
            experiment_report_id=str(value["experiment_report_id"]),
            experiment_report_sha256=str(value["experiment_report_sha256"]),
            evidence_index_id=str(value["evidence_index_id"]),
            evidence_index_sha256=str(value["evidence_index_sha256"]),
            code_review_sha256=str(value["code_review_sha256"]),
            quantitative_review_sha256=str(value["quantitative_review_sha256"]),
        )


@version("1.0")
@dataclass(frozen=True, slots=True)
class CanonicalReferencePromotionPlan:
    config: CanonicalReferencePromotionConfig
    spec: ArtifactSpec


@version("1.0", dependencies=[CanonicalReferencePromotionPlan, ReuseResult])
@dataclass(frozen=True, slots=True)
class CanonicalReferencePromotionResult:
    plan: CanonicalReferencePromotionPlan
    artifact: ArtifactPointer
    path: Path
    dry_run: bool
    built: bool
    reuse: ReuseResult | None
    validation: Mapping[str, Any] | None


@version("1.1")
def _review_identity(
    *,
    dataset: ArtifactPointer,
    dataset_manifest_sha256: str,
    daily_report_id: str,
    daily_report_sha256: str,
    experiment: ArtifactPointer,
    experiment_report_id: str,
    experiment_report_sha256: str,
    evidence_index_id: str,
    evidence_index_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "daily_report_id": daily_report_id,
        "daily_report_sha256": daily_report_sha256,
        "experiment": experiment,
        "experiment_report_id": experiment_report_id,
        "experiment_report_sha256": experiment_report_sha256,
        "evidence_index_id": evidence_index_id,
        "evidence_index_sha256": evidence_index_sha256,
    }


@version("1.1", dependencies=[_review_identity])
def make_review_attestation(
    *,
    role: Literal["code", "quantitative"],
    reviewer: str,
    review_date: date,
    review_reference: str,
    dataset: ArtifactPointer,
    dataset_manifest_sha256: str,
    daily_report_id: str,
    daily_report_sha256: str,
    experiment: ArtifactPointer,
    experiment_report_id: str,
    experiment_report_sha256: str,
    evidence_index_id: str,
    evidence_index_sha256: str,
) -> dict[str, Any]:
    """Create one exact approval attestation after an independent review."""

    if role not in {"code", "quantitative"}:
        raise ValueError("Review role must be 'code' or 'quantitative'")
    if not reviewer.strip() or not review_reference.strip():
        raise ValueError("Reviewer and durable review reference cannot be empty")
    for name, value in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("daily_report_id", daily_report_id),
        ("daily_report_sha256", daily_report_sha256),
        ("experiment_report_id", experiment_report_id),
        ("experiment_report_sha256", experiment_report_sha256),
        ("evidence_index_id", evidence_index_id),
        ("evidence_index_sha256", evidence_index_sha256),
    ):
        if _HEX_64.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    identity = _review_identity(
        dataset=dataset,
        dataset_manifest_sha256=dataset_manifest_sha256,
        daily_report_id=daily_report_id,
        daily_report_sha256=daily_report_sha256,
        experiment=experiment,
        experiment_report_id=experiment_report_id,
        experiment_report_sha256=experiment_report_sha256,
        evidence_index_id=evidence_index_id,
        evidence_index_sha256=evidence_index_sha256,
    )
    return cast(
        dict[str, Any],
        canonical_value(
            {
                "schema": REVIEW_ATTESTATION_SCHEMA,
                "role": role,
                "outcome": "approved",
                "reviewer": reviewer,
                "review_date": review_date,
                "review_reference": review_reference,
                **identity,
            }
        ),
    )


@version("1.1", dependencies=[_pointer])
def _validate_review(
    value: Mapping[str, Any],
    *,
    role: Literal["code", "quantitative"],
    identity: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema",
        "role",
        "outcome",
        "reviewer",
        "review_date",
        "review_reference",
        *identity,
    }
    if set(value) != expected_fields:
        raise ReferencePromotionError(f"{role} review fields are missing or unexpected")
    if (
        value.get("schema") != REVIEW_ATTESTATION_SCHEMA
        or value.get("role") != role
        or value.get("outcome") != "approved"
    ):
        raise ReferencePromotionError(f"{role} review is not an approval attestation")
    review_date = value.get("review_date")
    try:
        if not isinstance(review_date, str):
            raise ValueError("not a string")
        date.fromisoformat(review_date)
    except ValueError as exc:
        raise ReferencePromotionError(f"{role} review_date must be ISO format") from exc
    for name in ("reviewer", "review_reference"):
        if not isinstance(value.get(name), str) or not str(value[name]).strip():
            raise ReferencePromotionError(f"{role} {name} cannot be empty")
    for name, expected in identity.items():
        observed = value.get(name)
        if name in {"dataset", "experiment"}:
            observed = _pointer(observed, f"{role} review {name}").as_dict()
            expected = cast(ArtifactPointer, expected).as_dict()
        if observed != expected:
            raise ReferencePromotionError(f"{role} review does not bind exact {name}")


@version("1.0", dependencies=[_verified_report])
def _report_evidence(
    daily_path: Path,
    experiment_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    daily, daily_sha = _verified_report(
        daily_path,
        schema=REFERENCE_REPORT_SCHEMA,
        label="daily comparison report",
    )
    experiment, experiment_sha = _verified_report(
        experiment_path,
        schema=EXPERIMENT_REFERENCE_REPORT_SCHEMA,
        label="Experiment comparison report",
    )
    production_contract = daily.get("production_contract")
    experiment_contract = experiment.get("phase11_contract")
    if (
        not isinstance(production_contract, Mapping)
        or production_contract.get("passed") is not True
    ):
        raise ReferencePromotionError("Daily report did not pass its Phase 11 contract")
    if (
        not isinstance(experiment_contract, Mapping)
        or experiment_contract.get("passed") is not True
    ):
        raise ReferencePromotionError("Experiment report did not pass its Phase 11 contract")
    return daily, daily_sha, experiment, experiment_sha


@version("1.0", dependencies=[ValidatedArtifact, _promotion_sha256_file])
def _manifest_identity(artifact: ValidatedArtifact, label: str) -> tuple[str, str, str]:
    manifest_sha256 = _promotion_sha256_file(artifact.path / "manifest.json")
    configuration_id = artifact.manifest.get("configuration_id")
    metadata = artifact.manifest.get("metadata")
    source_revision = metadata.get("source_revision") if isinstance(metadata, Mapping) else None
    if not isinstance(configuration_id, str) or _HEX_64.fullmatch(configuration_id) is None:
        raise ReferencePromotionError(f"{label} configuration identity is invalid")
    if not isinstance(source_revision, str) or _HEX_40.fullmatch(source_revision) is None:
        raise ReferencePromotionError(f"{label} source revision is invalid")
    return manifest_sha256, configuration_id, source_revision


@version("1.0", dependencies=[_digest, _pointer])
def _experiment_report_pointer(report: Mapping[str, Any]) -> tuple[ArtifactPointer, str]:
    evidence = report.get("experiment")
    if not isinstance(evidence, Mapping):
        raise ReferencePromotionError("Experiment report has no artifact evidence")
    pointer = _pointer(evidence.get("artifact"), "Experiment report artifact")
    manifest_sha = _digest(
        evidence.get("manifest_sha256"), "Experiment report manifest_sha256"
    )
    return pointer, cast(str, manifest_sha)


@version("1.0", dependencies=[_pointer])
def _experiment_report_source(report: Mapping[str, Any]) -> ArtifactPointer:
    evidence = report.get("experiment")
    if not isinstance(evidence, Mapping):
        raise ReferencePromotionError("Experiment report has no artifact evidence")
    return _pointer(evidence.get("source_dataset"), "Experiment report source Dataset")


@version("1.0", dependencies=[_digest])
def _daily_approvals_sha256(report: Mapping[str, Any]) -> str | None:
    counts = report.get("classification_counts")
    if not isinstance(counts, Mapping):
        raise ReferencePromotionError("Daily report classification counts are malformed")
    approved = counts.get("approved_discrepancy", 0)
    document = report.get("approval_document")
    if approved:
        if not isinstance(document, Mapping):
            raise ReferencePromotionError(
                "Approved daily discrepancies require complete approval-document evidence"
            )
        return cast(
            str,
            _digest(document.get("sha256"), "daily approval document sha256"),
        )
    if document is not None:
        if not isinstance(document, Mapping):
            raise ReferencePromotionError("Daily approval-document evidence is malformed")
        return cast(
            str,
            _digest(document.get("sha256"), "daily approval document sha256"),
        )
    return None


@version(
    "1.2",
    dependencies=[
        ArtifactStore,
        CanonicalReferencePromotionConfig,
        _daily_approvals_sha256,
        _experiment_report_pointer,
        _experiment_report_source,
        _manifest_identity,
        _report_evidence,
        _promotion_sha256_file,
        _validate_review,
        _verified_evidence_index,
        _verified_runtime_evidence,
    ],
)
def create_promotion_config(
    store: ArtifactStore,
    *,
    name: str,
    dataset: ArtifactPointer,
    daily_report_path: Path,
    experiment_report_path: Path,
    evidence_index_path: Path,
    code_review_path: Path,
    quantitative_review_path: Path,
) -> CanonicalReferencePromotionConfig:
    """Validate evidence identities and create a portable promotion request."""

    dataset_artifact = store.validate(dataset)
    dataset_manifest, dataset_config, dataset_revision = _manifest_identity(
        dataset_artifact, "Dataset"
    )
    daily, daily_sha, experiment_report, experiment_sha = _report_evidence(
        daily_report_path, experiment_report_path
    )
    experiment_pointer, experiment_manifest = _experiment_report_pointer(
        experiment_report
    )
    if _experiment_report_source(experiment_report) != dataset:
        raise ReferencePromotionError(
            "Experiment comparison report does not derive from the designated Dataset"
        )
    experiment_artifact = store.validate(experiment_pointer)
    if _promotion_sha256_file(experiment_artifact.path / "manifest.json") != (
        experiment_manifest
    ):
        raise ReferencePromotionError("Experiment report manifest identity is stale")
    daily_id = cast(str, daily["report_id"])
    experiment_id = cast(str, experiment_report["report_id"])
    approvals_sha = _daily_approvals_sha256(daily)
    evidence_index, evidence_index_sha = _verified_evidence_index(
        evidence_index_path,
        expected_binding={
            "dataset": dataset,
            "dataset_manifest_sha256": dataset_manifest,
            "dataset_configuration_id": dataset_config,
            "dataset_source_revision": dataset_revision,
            "daily_report_id": daily_id,
            "daily_report_sha256": daily_sha,
            "daily_approvals_sha256": approvals_sha,
            "experiment": experiment_pointer,
            "experiment_manifest_sha256": experiment_manifest,
            "experiment_report_id": experiment_id,
            "experiment_report_sha256": experiment_sha,
        },
    )
    _verified_runtime_evidence(evidence_index, evidence_index_path)
    evidence_index_id = cast(str, evidence_index["evidence_id"])
    identity = _review_identity(
        dataset=dataset,
        dataset_manifest_sha256=dataset_manifest,
        daily_report_id=daily_id,
        daily_report_sha256=daily_sha,
        experiment=experiment_pointer,
        experiment_report_id=experiment_id,
        experiment_report_sha256=experiment_sha,
        evidence_index_id=evidence_index_id,
        evidence_index_sha256=evidence_index_sha,
    )
    code_review = _load_json(code_review_path, "code review")
    quantitative_review = _load_json(quantitative_review_path, "quantitative review")
    _validate_review(code_review, role="code", identity=identity)
    _validate_review(quantitative_review, role="quantitative", identity=identity)
    return CanonicalReferencePromotionConfig(
        name=name,
        dataset=dataset,
        dataset_manifest_sha256=dataset_manifest,
        dataset_configuration_id=dataset_config,
        dataset_source_revision=dataset_revision,
        daily_report_id=daily_id,
        daily_report_sha256=daily_sha,
        daily_approvals_sha256=approvals_sha,
        experiment=experiment_pointer,
        experiment_manifest_sha256=experiment_manifest,
        experiment_report_id=experiment_id,
        experiment_report_sha256=experiment_sha,
        evidence_index_id=evidence_index_id,
        evidence_index_sha256=evidence_index_sha,
        code_review_sha256=_promotion_sha256_file(code_review_path),
        quantitative_review_sha256=_promotion_sha256_file(quantitative_review_path),
    )


@version("1.0", dependencies=[_load_json, _pointer, canonical_sha256])
def _partition_binding(
    dataset_artifact: ValidatedArtifact,
    daily_report: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = _load_json(dataset_artifact.path / "partitions.json", "Dataset partitions")
    values = catalog.get("partitions")
    report_days = daily_report.get("dates")
    if not isinstance(values, list) or not isinstance(report_days, list):
        raise ReferencePromotionError("Dataset/report partition evidence is malformed")
    if len(values) != 918 or len(report_days) != 918:
        raise ReferencePromotionError("Promotion requires exactly 918 Dataset/report dates")
    for index, (partition, report_day) in enumerate(zip(values, report_days, strict=True)):
        if not isinstance(partition, Mapping) or not isinstance(report_day, Mapping):
            raise ReferencePromotionError(f"Partition binding record {index} is malformed")
        fresh = report_day.get("fresh")
        if not isinstance(fresh, Mapping):
            raise ReferencePromotionError(f"Daily report record {index} has no fresh evidence")
        date_value = partition.get("trading_date")
        if date_value != report_day.get("date"):
            raise ReferencePromotionError(f"Partition/report date differs at index {index}")
        fitted = _pointer(partition.get("fitted_artifact"), f"partition {index} fitted")
        report_fitted = _pointer(fresh.get("artifact"), f"report {index} fitted")
        if fitted != report_fitted:
            raise ReferencePromotionError(
                f"Partition/report fitted identity differs on {date_value}"
            )
        if partition.get("fitted_source_sha256") != fresh.get("data_sha256"):
            raise ReferencePromotionError(f"Partition/report bytes differ on {date_value}")
    return {
        "status": "passed",
        "partition_count": len(values),
        "bound_fitted_artifacts": len(values),
        "binding_sha256": canonical_sha256(
            [
                {
                    "date": item["trading_date"],
                    "fitted_artifact": item["fitted_artifact"],
                    "data_sha256": item["fitted_source_sha256"],
                }
                for item in values
            ]
        ),
    }


@version(
    "1.2",
    dependencies=[
        ArtifactStore,
        _daily_approvals_sha256,
        _experiment_report_pointer,
        _experiment_report_source,
        _manifest_identity,
        _partition_binding,
        _promotion_sha256_file,
        _report_evidence,
    ],
)
def validate_reference_evidence_binding(
    store: ArtifactStore,
    *,
    dataset: ArtifactPointer,
    daily_report_path: Path,
    experiment_report_path: Path,
) -> dict[str, Any]:
    """Bind pre-review comparison reports to exact published artifact bytes."""

    dataset_artifact = store.validate(dataset)
    dataset_manifest, dataset_config, dataset_revision = _manifest_identity(
        dataset_artifact, "Dataset"
    )
    daily, daily_sha, experiment_report, experiment_sha = _report_evidence(
        daily_report_path, experiment_report_path
    )
    partition_binding = _partition_binding(dataset_artifact, daily)
    experiment, experiment_manifest = _experiment_report_pointer(experiment_report)
    if _experiment_report_source(experiment_report) != dataset:
        raise ReferencePromotionError(
            "Experiment comparison report does not derive from the Dataset"
        )
    experiment_artifact = store.validate(experiment)
    observed_experiment_manifest = _promotion_sha256_file(
        experiment_artifact.path / "manifest.json"
    )
    if observed_experiment_manifest != experiment_manifest:
        raise ReferencePromotionError("Experiment report manifest identity is stale")
    return cast(
        dict[str, Any],
        canonical_value(
            {
                "status": "passed",
                "dataset": dataset,
                "dataset_manifest_sha256": dataset_manifest,
                "dataset_configuration_id": dataset_config,
                "dataset_source_revision": dataset_revision,
                "daily_report_id": daily["report_id"],
                "daily_report_sha256": daily_sha,
                "daily_approvals_sha256": _daily_approvals_sha256(daily),
                "partition_binding": partition_binding,
                "experiment": experiment,
                "experiment_manifest_sha256": experiment_manifest,
                "experiment_report_id": experiment_report["report_id"],
                "experiment_report_sha256": experiment_sha,
            }
        ),
    )


@version(
    "1.2",
    dependencies=[
        ArtifactStore,
        _experiment_report_pointer,
        _experiment_report_source,
        _manifest_identity,
        _partition_binding,
        _promotion_sha256_file,
        _report_evidence,
        make_review_attestation,
        _verified_evidence_index,
        _verified_runtime_evidence,
    ],
)
def create_review_attestation(
    store: ArtifactStore,
    *,
    role: Literal["code", "quantitative"],
    reviewer: str,
    review_date: date,
    review_reference: str,
    dataset: ArtifactPointer,
    daily_report_path: Path,
    experiment_report_path: Path,
    evidence_index_path: Path,
) -> dict[str, Any]:
    """Bind a completed independent review to exact validated evidence."""

    dataset_artifact = store.validate(dataset)
    dataset_manifest, _configuration_id, _source_revision = _manifest_identity(
        dataset_artifact, "Dataset"
    )
    daily, daily_sha, experiment_report, experiment_sha = _report_evidence(
        daily_report_path, experiment_report_path
    )
    partition_binding = _partition_binding(dataset_artifact, daily)
    experiment_pointer, experiment_manifest = _experiment_report_pointer(
        experiment_report
    )
    if _experiment_report_source(experiment_report) != dataset:
        raise ReferencePromotionError(
            "Experiment comparison report does not derive from the reviewed Dataset"
        )
    experiment_artifact = store.validate(experiment_pointer)
    if _promotion_sha256_file(experiment_artifact.path / "manifest.json") != (
        experiment_manifest
    ):
        raise ReferencePromotionError("Experiment report manifest identity is stale")
    approvals_sha = _daily_approvals_sha256(daily)
    evidence_index, evidence_index_sha = _verified_evidence_index(
        evidence_index_path,
        expected_binding={
            "dataset": dataset,
            "dataset_manifest_sha256": dataset_manifest,
            "dataset_configuration_id": _configuration_id,
            "dataset_source_revision": _source_revision,
            "daily_report_id": daily["report_id"],
            "daily_report_sha256": daily_sha,
            "daily_approvals_sha256": approvals_sha,
            "partition_binding": partition_binding,
            "experiment": experiment_pointer,
            "experiment_manifest_sha256": experiment_manifest,
            "experiment_report_id": experiment_report["report_id"],
            "experiment_report_sha256": experiment_sha,
        },
    )
    _verified_runtime_evidence(evidence_index, evidence_index_path)
    return make_review_attestation(
        role=role,
        reviewer=reviewer,
        review_date=review_date,
        review_reference=review_reference,
        dataset=dataset,
        dataset_manifest_sha256=dataset_manifest,
        daily_report_id=cast(str, daily["report_id"]),
        daily_report_sha256=daily_sha,
        experiment=experiment_pointer,
        experiment_report_id=cast(str, experiment_report["report_id"]),
        experiment_report_sha256=experiment_sha,
        evidence_index_id=cast(str, evidence_index["evidence_id"]),
        evidence_index_sha256=evidence_index_sha,
    )


@version(
    "1.1",
    dependencies=[
        CanonicalReferencePromotionConfig,
        _daily_approvals_sha256,
        _experiment_report_pointer,
        _experiment_report_source,
        _load_json,
        _manifest_identity,
        _partition_binding,
        _report_evidence,
        _review_identity,
        _promotion_sha256_file,
        _validate_review,
        _verified_evidence_index,
        _verified_runtime_evidence,
    ],
)
def _config_matches_evidence(
    config: CanonicalReferencePromotionConfig,
    *,
    dataset_artifact: ValidatedArtifact,
    daily_path: Path,
    experiment_path: Path,
    evidence_index_path: Path,
    code_path: Path,
    quantitative_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    manifest_sha, configuration_id, source_revision = _manifest_identity(
        dataset_artifact, "Dataset"
    )
    if (
        manifest_sha != config.dataset_manifest_sha256
        or configuration_id != config.dataset_configuration_id
        or source_revision != config.dataset_source_revision
    ):
        raise ReferencePromotionError("Promotion configuration identifies stale Dataset evidence")
    daily, daily_sha, experiment, experiment_sha = _report_evidence(
        daily_path, experiment_path
    )
    if daily_sha != config.daily_report_sha256 or daily.get("report_id") != config.daily_report_id:
        raise ReferencePromotionError("Daily comparison report differs from promotion config")
    if (
        experiment_sha != config.experiment_report_sha256
        or experiment.get("report_id") != config.experiment_report_id
    ):
        raise ReferencePromotionError("Experiment comparison report differs from promotion config")
    if _daily_approvals_sha256(daily) != config.daily_approvals_sha256:
        raise ReferencePromotionError("Daily approval document differs from promotion config")
    pointer, experiment_manifest = _experiment_report_pointer(experiment)
    if pointer != config.experiment or experiment_manifest != config.experiment_manifest_sha256:
        raise ReferencePromotionError("Experiment evidence differs from promotion config")
    if _experiment_report_source(experiment) != config.dataset:
        raise ReferencePromotionError(
            "Experiment report source differs from the designated Dataset"
        )
    partition_binding = _partition_binding(dataset_artifact, daily)
    evidence_index, evidence_index_sha = _verified_evidence_index(
        evidence_index_path,
        expected_binding={
            "dataset": config.dataset,
            "dataset_manifest_sha256": config.dataset_manifest_sha256,
            "dataset_configuration_id": config.dataset_configuration_id,
            "dataset_source_revision": config.dataset_source_revision,
            "daily_report_id": config.daily_report_id,
            "daily_report_sha256": config.daily_report_sha256,
            "daily_approvals_sha256": config.daily_approvals_sha256,
            "partition_binding": partition_binding,
            "experiment": config.experiment,
            "experiment_manifest_sha256": config.experiment_manifest_sha256,
            "experiment_report_id": config.experiment_report_id,
            "experiment_report_sha256": config.experiment_report_sha256,
        },
    )
    if (
        evidence_index.get("evidence_id") != config.evidence_index_id
        or evidence_index_sha != config.evidence_index_sha256
    ):
        raise ReferencePromotionError("Phase 11 evidence index differs from promotion config")
    _verified_runtime_evidence(evidence_index, evidence_index_path)
    identity = _review_identity(
        dataset=config.dataset,
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        daily_report_id=config.daily_report_id,
        daily_report_sha256=config.daily_report_sha256,
        experiment=config.experiment,
        experiment_report_id=config.experiment_report_id,
        experiment_report_sha256=config.experiment_report_sha256,
        evidence_index_id=config.evidence_index_id,
        evidence_index_sha256=config.evidence_index_sha256,
    )
    code = _load_json(code_path, "code review")
    quantitative = _load_json(quantitative_path, "quantitative review")
    _validate_review(code, role="code", identity=identity)
    _validate_review(quantitative, role="quantitative", identity=identity)
    if _promotion_sha256_file(code_path) != config.code_review_sha256:
        raise ReferencePromotionError("Code review bytes differ from promotion config")
    if _promotion_sha256_file(quantitative_path) != config.quantitative_review_sha256:
        raise ReferencePromotionError("Quantitative review bytes differ from promotion config")
    return daily, experiment, code, quantitative, evidence_index


@version("1.2", dependencies=[CanonicalReferencePromotionConfig])
def _promotion_readme(config: CanonicalReferencePromotionConfig) -> str:
    return f"""# Canonical SANOS reference designation: {config.name}

This immutable evidence artifact designates Dataset `{config.dataset.artifact_id}` as the
canonical SANOS research reference after full fresh-WRDS validation, 891-date migration
comparison, Experiment tensor comparison, and independent code and quantitative reviews.
The exact Phase 11 evidence index reviewed by both reviewers is included and bound by
identity and checksum. The exact solver and numerical runtime fingerprint is retained
beside it and independently revalidated.
It does not modify or replace the Dataset and it grants no redistribution rights.

Validate it without upstream artifacts using `sanos dataset reference validate`; add
`--deep-lineage` to re-open the designated Dataset and Experiment.
"""


@version(
    "1.2",
    dependencies=[
        ArtifactSpec,
        CanonicalReferencePromotionConfig,
        _config_matches_evidence,
        _partition_binding,
        _promotion_readme,
        _report_evidence,
        _validate_review,
        validate_experiment,
        validate_internal_dataset,
    ],
)
def reference_promotion_producer_version() -> str:
    return str(reference_promotion_producer_version.version)


@version(
    "1.2",
    dependencies=[
        ArtifactSpec,
        CanonicalReferencePromotionConfig,
        CanonicalReferencePromotionPlan,
        reference_promotion_producer_version,
    ],
)
def plan_reference_promotion(
    config: CanonicalReferencePromotionConfig,
) -> CanonicalReferencePromotionPlan:
    spec = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace=f"sanos/canonical-reference/name-{config.name}",
        configuration={
            "schema": PROMOTION_CONFIG_SCHEMA,
            "promotion": config.to_document(),
        },
        schema_version=PROMOTION_SCHEMA,
        producer_version=reference_promotion_producer_version(),
        upstream=(config.dataset,),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )
    return CanonicalReferencePromotionPlan(config=config, spec=spec)


@version(
    "1.2",
    dependencies=[
        ArtifactStore,
        CanonicalReferencePromotionConfig,
        CanonicalReferencePromotionResult,
        _config_matches_evidence,
        _partition_binding,
        _promotion_readme,
        _verified_runtime_evidence,
        plan_reference_promotion,
        validate_experiment,
        validate_internal_dataset,
    ],
)
def promote_reference_dataset(
    locations: StorageLocations,
    config: CanonicalReferencePromotionConfig,
    *,
    daily_report_path: Path | None = None,
    experiment_report_path: Path | None = None,
    evidence_index_path: Path | None = None,
    code_review_path: Path | None = None,
    quantitative_review_path: Path | None = None,
    version_override: Literal["*"] | None = None,
    dry_run: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> CanonicalReferencePromotionResult:
    """Output-first reuse or atomically publish one canonical designation."""

    plan = plan_reference_promotion(config)
    store = ArtifactStore(locations)
    reuse = store.resolve(plan.spec, version=version_override)
    if reuse is not None:
        validation = validate_reference_promotion(store, reuse.artifact)
        return CanonicalReferencePromotionResult(
            plan=plan,
            artifact=reuse.artifact.pointer,
            path=reuse.artifact.path,
            dry_run=dry_run,
            built=False,
            reuse=reuse,
            validation=validation,
        )
    if dry_run:
        return CanonicalReferencePromotionResult(
            plan=plan,
            artifact=plan.spec.pointer,
            path=store.path_for(plan.spec.pointer),
            dry_run=True,
            built=False,
            reuse=None,
            validation=None,
        )
    evidence_paths = {
        "daily comparison report": daily_report_path,
        "Experiment comparison report": experiment_report_path,
        "Phase 11 evidence index": evidence_index_path,
        "code review": code_review_path,
        "quantitative review": quantitative_review_path,
    }
    missing = [name for name, path in evidence_paths.items() if path is None]
    if missing:
        raise ReferencePromotionError(
            "Promotion miss requires evidence paths: " + ", ".join(missing)
        )
    daily_path = cast(Path, daily_report_path)
    experiment_path = cast(Path, experiment_report_path)
    evidence_path = cast(Path, evidence_index_path)
    code_path = cast(Path, code_review_path)
    quantitative_path = cast(Path, quantitative_review_path)
    dataset_artifact = store.validate(config.dataset)
    daily, _experiment, _code, _quantitative, evidence_index = _config_matches_evidence(
        config,
        dataset_artifact=dataset_artifact,
        daily_path=daily_path,
        experiment_path=experiment_path,
        evidence_index_path=evidence_path,
        code_path=code_path,
        quantitative_path=quantitative_path,
    )
    runtime, runtime_path, runtime_sha256 = _verified_runtime_evidence(
        evidence_index, evidence_path
    )
    dataset_summary = validate_internal_dataset(
        store,
        dataset_artifact,
        semantic=True,
        deep_lineage=True,
        progress=progress,
    )
    if dataset_summary is None:
        raise ReferencePromotionError("Deep Dataset validation produced no summary")
    experiment_artifact = store.validate(config.experiment)
    if _promotion_sha256_file(experiment_artifact.path / "manifest.json") != (
        config.experiment_manifest_sha256
    ):
        raise ReferencePromotionError("Experiment manifest differs from promotion config")
    experiment_summary = validate_experiment(
        store,
        experiment_artifact,
        semantic=True,
        deep_lineage=False,
        progress=progress,
    )
    if experiment_summary is None:
        raise ReferencePromotionError("Experiment validation produced no summary")
    binding = _partition_binding(dataset_artifact, daily)
    promotion_document = {
        "schema": PROMOTION_SCHEMA,
        "canonical_promotion_ready": True,
        "dataset": config.dataset,
        "dataset_manifest_sha256": config.dataset_manifest_sha256,
        "dataset_configuration_id": config.dataset_configuration_id,
        "dataset_source_revision": config.dataset_source_revision,
        "daily_report_id": config.daily_report_id,
        "daily_report_sha256": config.daily_report_sha256,
        "daily_approvals_sha256": config.daily_approvals_sha256,
        "experiment": config.experiment,
        "experiment_manifest_sha256": config.experiment_manifest_sha256,
        "experiment_report_id": config.experiment_report_id,
        "experiment_report_sha256": config.experiment_report_sha256,
        "evidence_index_id": config.evidence_index_id,
        "evidence_index_sha256": config.evidence_index_sha256,
        "runtime_fingerprint_id": runtime["fingerprint_id"],
        "runtime_fingerprint_sha256": runtime_sha256,
        "code_review_sha256": config.code_review_sha256,
        "quantitative_review_sha256": config.quantitative_review_sha256,
        "validation": {
            "dataset_shallow_semantic": "passed",
            "dataset_deep_lineage": "passed",
            "dataset_summary": dataset_summary.as_dict(),
            "experiment_semantic": "passed",
            "experiment_summary": experiment_summary.as_dict(),
            "partition_binding": binding,
        },
        "distribution_effect": "none; designation does not grant redistribution rights",
    }
    metadata = {
        "canonical_promotion_ready": True,
        "designated_dataset": config.dataset,
        "daily_report_id": config.daily_report_id,
        "experiment_report_id": config.experiment_report_id,
        "evidence_index_id": config.evidence_index_id,
        "runtime_fingerprint_id": runtime["fingerprint_id"],
        "partition_count": binding["partition_count"],
        "validation_outcome": "passed",
    }
    builder = store.writer(plan.spec, metadata=metadata)
    built = True
    try:
        with builder as staging:
            staging.write_text("README.md", _promotion_readme(config))
            staging.write_json("promotion.json", promotion_document)
            staging.copy_file("daily-comparison.json", daily_path)
            staging.copy_file("experiment-comparison.json", experiment_path)
            staging.copy_file("evidence-index.json", evidence_path)
            staging.copy_file("runtime-fingerprint.json", runtime_path)
            staging.copy_file("code-review.json", code_path)
            staging.copy_file("quantitative-review.json", quantitative_path)
    except ArtifactExistsError:
        artifact = store.validate(plan.spec.pointer, expected=plan.spec)
        built = False
    else:
        artifact = builder.artifact
    validation = validate_reference_promotion(store, artifact)
    return CanonicalReferencePromotionResult(
        plan=plan,
        artifact=artifact.pointer,
        path=artifact.path,
        dry_run=False,
        built=built,
        reuse=None,
        validation=validation,
    )


@version("1.1", dependencies=[CanonicalReferencePromotionConfig])
def _parse_promotion_config(value: object) -> CanonicalReferencePromotionConfig:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "promotion"}
        or value.get("schema") != PROMOTION_CONFIG_SCHEMA
        or not isinstance(value.get("promotion"), Mapping)
    ):
        raise ReferencePromotionError("Promotion artifact configuration is malformed")
    try:
        return CanonicalReferencePromotionConfig.from_document(
            cast(Mapping[str, Any], value["promotion"])
        )
    except (TypeError, ValueError) as exc:
        raise ReferencePromotionError(f"Promotion configuration is invalid: {exc}") from exc


@version(
    "1.2",
    dependencies=[
        _parse_promotion_config,
        _report_evidence,
        _validate_review,
        _verified_evidence_index,
        _verified_runtime_evidence,
        canonical_value,
        validate_experiment,
        validate_internal_dataset,
    ],
)
def validate_reference_promotion(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
    *,
    deep_lineage: bool = False,
) -> dict[str, Any]:
    """Validate a self-contained designation; upstream access is opt-in."""

    pointer = artifact.pointer if isinstance(artifact, ValidatedArtifact) else artifact
    validated = store.validate(pointer)
    if validated.pointer.storage_class is not StorageClass.DATASET:
        raise ReferencePromotionError("Canonical designation is not a Dataset-class artifact")
    if validated.manifest.get("schema_version") != PROMOTION_SCHEMA:
        raise ReferencePromotionError("Canonical designation schema changed")
    config = _parse_promotion_config(validated.manifest.get("configuration"))
    upstream = validated.manifest.get("upstream")
    if upstream != [config.dataset.as_dict()]:
        raise ReferencePromotionError("Canonical designation upstream differs")
    distribution = validated.manifest.get("distribution")
    if not isinstance(distribution, Mapping) or distribution.get("status") != "internal":
        raise ReferencePromotionError("Canonical designation must remain internal")
    promotion = _load_json(validated.path / "promotion.json", "promotion evidence")
    required = {
        "schema",
        "canonical_promotion_ready",
        "dataset",
        "dataset_manifest_sha256",
        "dataset_configuration_id",
        "dataset_source_revision",
        "daily_report_id",
        "daily_report_sha256",
        "daily_approvals_sha256",
        "experiment",
        "experiment_manifest_sha256",
        "experiment_report_id",
        "experiment_report_sha256",
        "evidence_index_id",
        "evidence_index_sha256",
        "runtime_fingerprint_id",
        "runtime_fingerprint_sha256",
        "code_review_sha256",
        "quantitative_review_sha256",
        "validation",
        "distribution_effect",
    }
    if set(promotion) != required or promotion.get("schema") != PROMOTION_SCHEMA:
        raise ReferencePromotionError("Promotion evidence fields are missing or unexpected")
    if promotion.get("canonical_promotion_ready") is not True:
        raise ReferencePromotionError("Canonical promotion outcome is not true")
    for name, expected in config.to_document().items():
        if name == "name":
            continue
        observed = promotion.get(name)
        if name in {"dataset", "experiment"}:
            observed = _pointer(observed, f"promotion {name}").as_dict()
            expected = cast(Mapping[str, Any], expected)
        if observed != expected:
            raise ReferencePromotionError(f"Promotion evidence differs from config: {name}")
    daily_path = validated.path / "daily-comparison.json"
    experiment_path = validated.path / "experiment-comparison.json"
    daily, daily_sha, experiment, experiment_sha = _report_evidence(
        daily_path, experiment_path
    )
    if daily_sha != config.daily_report_sha256 or daily.get("report_id") != config.daily_report_id:
        raise ReferencePromotionError("Copied daily report identity differs")
    if (
        experiment_sha != config.experiment_report_sha256
        or experiment.get("report_id") != config.experiment_report_id
    ):
        raise ReferencePromotionError("Copied Experiment report identity differs")
    if _daily_approvals_sha256(daily) != config.daily_approvals_sha256:
        raise ReferencePromotionError("Copied daily approval evidence differs")
    experiment_pointer, experiment_manifest = _experiment_report_pointer(experiment)
    if (
        experiment_pointer != config.experiment
        or experiment_manifest != config.experiment_manifest_sha256
        or _experiment_report_source(experiment) != config.dataset
    ):
        raise ReferencePromotionError("Copied Experiment evidence differs from config")
    evidence_index_path = validated.path / "evidence-index.json"
    evidence_index, evidence_index_sha = _verified_evidence_index(
        evidence_index_path,
        expected_binding={
            "dataset": config.dataset,
            "dataset_manifest_sha256": config.dataset_manifest_sha256,
            "dataset_configuration_id": config.dataset_configuration_id,
            "dataset_source_revision": config.dataset_source_revision,
            "daily_report_id": config.daily_report_id,
            "daily_report_sha256": config.daily_report_sha256,
            "daily_approvals_sha256": config.daily_approvals_sha256,
            "experiment": config.experiment,
            "experiment_manifest_sha256": config.experiment_manifest_sha256,
            "experiment_report_id": config.experiment_report_id,
            "experiment_report_sha256": config.experiment_report_sha256,
        },
    )
    if (
        evidence_index.get("evidence_id") != config.evidence_index_id
        or evidence_index_sha != config.evidence_index_sha256
    ):
        raise ReferencePromotionError("Copied Phase 11 evidence index differs")
    runtime, _runtime_path, runtime_sha256 = _verified_runtime_evidence(
        evidence_index, evidence_index_path
    )
    if (
        promotion.get("runtime_fingerprint_id") != runtime.get("fingerprint_id")
        or promotion.get("runtime_fingerprint_sha256") != runtime_sha256
    ):
        raise ReferencePromotionError("Copied runtime fingerprint evidence differs")
    identity = _review_identity(
        dataset=config.dataset,
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        daily_report_id=config.daily_report_id,
        daily_report_sha256=config.daily_report_sha256,
        experiment=config.experiment,
        experiment_report_id=config.experiment_report_id,
        experiment_report_sha256=config.experiment_report_sha256,
        evidence_index_id=config.evidence_index_id,
        evidence_index_sha256=config.evidence_index_sha256,
    )
    code_path = validated.path / "code-review.json"
    quantitative_path = validated.path / "quantitative-review.json"
    code = _load_json(code_path, "copied code review")
    quantitative = _load_json(quantitative_path, "copied quantitative review")
    _validate_review(code, role="code", identity=identity)
    _validate_review(quantitative, role="quantitative", identity=identity)
    if _promotion_sha256_file(code_path) != config.code_review_sha256:
        raise ReferencePromotionError("Copied code review digest differs")
    if _promotion_sha256_file(quantitative_path) != config.quantitative_review_sha256:
        raise ReferencePromotionError("Copied quantitative review digest differs")
    validation = promotion.get("validation")
    if not isinstance(validation, Mapping) or any(
        validation.get(name) != "passed"
        for name in (
            "dataset_shallow_semantic",
            "dataset_deep_lineage",
            "experiment_semantic",
        )
    ):
        raise ReferencePromotionError("Promotion validation evidence is incomplete")
    partition_binding = validation.get("partition_binding")
    if (
        not isinstance(partition_binding, Mapping)
        or partition_binding.get("status") != "passed"
        or partition_binding.get("partition_count") != 918
        or partition_binding.get("bound_fitted_artifacts") != 918
        or _HEX_64.fullmatch(str(partition_binding.get("binding_sha256"))) is None
    ):
        raise ReferencePromotionError("Promotion partition binding is incomplete")
    indexed_binding = evidence_index.get("artifact_binding")
    if (
        not isinstance(indexed_binding, Mapping)
        or canonical_value(indexed_binding.get("partition_binding"))
        != canonical_value(partition_binding)
    ):
        raise ReferencePromotionError("Promotion partition binding differs from evidence index")
    if promotion.get("distribution_effect") != (
        "none; designation does not grant redistribution rights"
    ):
        raise ReferencePromotionError("Promotion distribution effect changed")
    if deep_lineage:
        dataset = store.validate(config.dataset)
        if _promotion_sha256_file(dataset.path / "manifest.json") != (
            config.dataset_manifest_sha256
        ):
            raise ReferencePromotionError("Designated Dataset manifest changed")
        if validate_internal_dataset(store, dataset, semantic=True, deep_lineage=True) is None:
            raise ReferencePromotionError("Deep designated Dataset validation returned no result")
        experiment_artifact = store.validate(config.experiment)
        if _promotion_sha256_file(experiment_artifact.path / "manifest.json") != (
            config.experiment_manifest_sha256
        ):
            raise ReferencePromotionError("Bound Experiment manifest changed")
        if validate_experiment(store, experiment_artifact, semantic=True) is None:
            raise ReferencePromotionError("Bound Experiment validation returned no result")
    return cast(
        dict[str, Any],
        canonical_value(
            {
                "status": "passed",
                "canonical_promotion_ready": True,
                "promotion": validated.pointer,
                "dataset": config.dataset,
                "experiment": config.experiment,
                "daily_report_id": config.daily_report_id,
                "experiment_report_id": config.experiment_report_id,
                "deep_lineage": deep_lineage,
            }
        ),
    )


__all__ = [
    "PROMOTION_CONFIG_SCHEMA",
    "PROMOTION_SCHEMA",
    "REVIEW_ATTESTATION_SCHEMA",
    "CanonicalReferencePromotionConfig",
    "CanonicalReferencePromotionPlan",
    "CanonicalReferencePromotionResult",
    "ReferencePromotionError",
    "create_promotion_config",
    "create_review_attestation",
    "make_review_attestation",
    "plan_reference_promotion",
    "promote_reference_dataset",
    "reference_promotion_producer_version",
    "validate_reference_promotion",
    "validate_reference_evidence_binding",
]
