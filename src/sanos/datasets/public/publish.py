"""Output-first atomic publication of public-candidate SANOS datasets."""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from cdxcore.version import version

from sanos.artifacts import (
    MANIFEST_NAME,
    ArtifactExistsError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    DatasetDistribution,
    DistributionStatus,
    ReuseResult,
    StorageClass,
)
from sanos.config import StorageLocations, canonical_sha256
from sanos.datasets.internal import (
    InternalDataset,
    open_internal_dataset,
    validate_internal_dataset,
)
from sanos.datasets.public.config import (
    PublicBuildOptions,
    PublicProjectionConfig,
    PublicReleaseConfig,
)
from sanos.datasets.public.documents import candidate_readme, release_readme
from sanos.datasets.public.project import (
    ProjectedPublicDay,
    PublicProjectionError,
    assemble_public_frame,
    project_public_day,
)
from sanos.datasets.public.schema import (
    PUBLIC_DATASET_SCHEMA_VERSION,
    PUBLIC_SCHEMA_DOCUMENT_VERSION,
    PUBLIC_VALIDATION_VERSION,
    public_capabilities,
    public_schema_document,
)
from sanos.datasets.public.validate import (
    PublicDatasetValidationSummary,
    open_public_dataset,
    public_frame_logical_sha256,
    public_manifest_metadata,
    public_validation_document,
    validate_public_dataset,
    validate_public_frame,
    validate_public_payload,
)
from sanos.widb.calendar import business_dates
from sanos.widb.surface import FITTED_SCHEMA_VERSION

_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


@version("1.0", dependencies=[ArtifactSpec])
@dataclass(frozen=True, slots=True)
class PublicDatasetPlan:
    """Pure location-independent public-candidate identity."""

    configuration: Mapping[str, Any]
    dataset_spec: ArtifactSpec
    plan_id: str


@version(
    "1.0",
    dependencies=[
        ArtifactPointer,
        ReuseResult,
        PublicDatasetPlan,
        PublicDatasetValidationSummary,
    ],
)
@dataclass(frozen=True, slots=True)
class PublicDatasetBuildResult:
    """Dry-run, reuse, or completed public-candidate build result."""

    plan: PublicDatasetPlan
    artifact: ArtifactPointer
    path: Path
    dry_run: bool
    built: bool
    reuse: ReuseResult | None
    validation: PublicDatasetValidationSummary | None
    source_accessed: bool


@version("1.0")
def public_candidate_label(name: str) -> str:
    label = _SAFE_LABEL.sub("-", name).strip("-").lower()
    if not label:
        raise ValueError("Public Dataset name has no safe namespace representation")
    return label


@version("parquet-zstd-level3-statistics-v1")
def public_parquet_contract() -> tuple[str, int, bool]:
    """Own the deterministic public Parquet writer settings."""

    return "zstd", 3, True


def _emit(
    progress: Callable[[Mapping[str, Any]], None] | None,
    event: str,
    **values: Any,
) -> None:
    if progress is not None:
        progress({"event": event, **values})


@version("1.0", dependencies=[public_parquet_contract])
def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    compression, compression_level, statistics = public_parquet_contract()
    frame.write_parquet(
        buffer,
        compression=compression,
        compression_level=compression_level,
        statistics=statistics,
    )
    return buffer.getvalue()


@version(
    "1.0",
    dependencies=[
        InternalDataset,
        PublicProjectionConfig,
        assemble_public_frame,
        business_dates,
        project_public_day,
        validate_public_frame,
    ],
)
def _project_source_dataset(
    source: InternalDataset,
    config: PublicProjectionConfig,
    *,
    progress_interval_seconds: float,
    progress_event: str,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[
    pl.DataFrame,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    PublicDatasetValidationSummary,
]:
    """Reproject one validated internal Dataset under the exact public contract."""

    if (
        source.config.start_date != config.coverage_start_date
        or source.config.end_date != config.coverage_end_date
    ):
        raise PublicProjectionError(
            "Public coverage dates differ from the internal source configuration"
        )
    expected_dates = business_dates(config.coverage_start_date, config.coverage_end_date)
    partition_dates = tuple(
        date.fromisoformat(str(record["trading_date"])) for record in source.partitions
    )
    if partition_dates != expected_dates:
        raise PublicProjectionError(
            "Internal partitions do not exactly cover the configured NYSE sessions"
        )
    source_overnight_fraction = source.config.daily.clock.overnight_variance_fraction
    if source_overnight_fraction != config.overnight_variance_fraction:
        raise PublicProjectionError(
            "Public overnight_variance_fraction differs from the internal source "
            f"({config.overnight_variance_fraction} != {source_overnight_fraction})"
        )
    projected: list[ProjectedPublicDay] = []
    omissions: list[dict[str, Any]] = []
    started = time.monotonic()
    last_progress = started
    total = len(source.partitions)
    for completed, record in enumerate(source.partitions, start=1):
        trading_date_text = record.get("trading_date")
        if not isinstance(trading_date_text, str):
            raise PublicProjectionError("Internal partition has no trading_date")
        typed_empty = record.get("typed_empty")
        reason = record.get("typed_empty_reason")
        if not isinstance(typed_empty, bool):
            raise PublicProjectionError("Internal partition has invalid typed_empty evidence")
        frame = source.load_day(date.fromisoformat(trading_date_text))
        if typed_empty:
            if not frame.is_empty() or not isinstance(reason, str) or not reason:
                raise PublicProjectionError("Typed-empty partition evidence is inconsistent")
            omissions.append({"trading_date": trading_date_text, "reason": reason})
        else:
            if frame.is_empty():
                raise PublicProjectionError("Unexpected empty internal partition")
            projected.append(project_public_day(frame, config))
        now = time.monotonic()
        if completed == total or now - last_progress >= progress_interval_seconds:
            elapsed = now - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = (total - completed) / rate if rate > 0.0 else None
            _emit(
                progress,
                progress_event,
                elapsed_seconds=elapsed,
                completed=completed,
                total=total,
                failed=0,
                estimated_remaining_seconds=remaining,
            )
            last_progress = now
    expected_omissions = tuple(
        item.as_dict() for item in config.expected_typed_empty_omissions
    )
    if tuple(omissions) != expected_omissions:
        raise PublicProjectionError(
            "Internal typed-empty omissions differ from the public identity policy"
        )
    frame, private_return_evidence = assemble_public_frame(tuple(projected))
    public_return_evidence = tuple(
        {
            "trading_date": item["trading_date"],
            "previous_trading_date": item["previous_trading_date"],
            "business_sessions": item["business_sessions"],
            "discount_curve_validated": True,
        }
        for item in private_return_evidence
    )
    summary = validate_public_frame(
        frame,
        config,
        omitted_dates=tuple(omissions),
        progress=progress,
    )
    return frame, tuple(omissions), public_return_evidence, summary


@version(
    "1.0",
    dependencies=[
        _project_source_dataset,
        open_internal_dataset,
        validate_internal_dataset,
    ],
)
def reproject_public_source(
    store: ArtifactStore,
    config: PublicProjectionConfig,
) -> tuple[pl.DataFrame, tuple[dict[str, Any], ...], PublicDatasetValidationSummary]:
    """Reproject the exact internal source for explicit source-fidelity validation."""

    validate_internal_dataset(store, config.source_dataset, semantic=True, deep_lineage=False)
    source = open_internal_dataset(store, config.source_dataset)
    frame, omissions, _, summary = _project_source_dataset(
        source,
        config,
        progress_interval_seconds=300.0,
        progress_event="public_source_reprojection_progress",
    )
    return frame, omissions, summary


@version(
    "1.1",
    dependencies=[
        PublicProjectionConfig,
        _parquet_bytes,
        _project_source_dataset,
        candidate_readme,
        public_capabilities,
        public_manifest_metadata,
        public_schema_document,
        public_validation_document,
        validate_public_payload,
    ],
)
def public_dataset_producer_version() -> str:
    """Return the complete public-candidate producer closure."""

    return str(public_dataset_producer_version.version)


@version(
    "1.1",
    dependencies=[
        ArtifactSpec,
        DatasetDistribution,
        PublicDatasetPlan,
        PublicProjectionConfig,
        public_candidate_label,
        public_dataset_producer_version,
    ],
)
def plan_public_candidate(config: PublicProjectionConfig) -> PublicDatasetPlan:
    """Plan a candidate identity without reading any configured storage root."""

    configuration = config.as_dict()
    spec = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace=f"sanos/public-candidate/name-{public_candidate_label(config.name)}",
        configuration=configuration,
        schema_version="+".join(
            (
                PUBLIC_DATASET_SCHEMA_VERSION,
                PUBLIC_SCHEMA_DOCUMENT_VERSION,
                PUBLIC_VALIDATION_VERSION,
                FITTED_SCHEMA_VERSION,
            )
        ),
        producer_version=public_dataset_producer_version(),
        upstream=(config.source_dataset,),
        distribution=DatasetDistribution(DistributionStatus.PUBLIC_CANDIDATE),
    )
    return PublicDatasetPlan(
        configuration=configuration,
        dataset_spec=spec,
        plan_id=canonical_sha256(
            {"candidate": spec.pointer.as_dict(), "source": config.source_dataset.as_dict()}
        ),
    )


@version(
    "1.0",
    dependencies=[
        PublicDatasetBuildResult,
        PublicBuildOptions,
        _project_source_dataset,
        candidate_readme,
        plan_public_candidate,
        open_internal_dataset,
        public_manifest_metadata,
        public_validation_document,
        validate_public_dataset,
        validate_public_payload,
        validate_internal_dataset,
        public_capabilities,
        public_schema_document,
        _parquet_bytes,
    ],
)
def build_public_candidate(
    locations: StorageLocations,
    config: PublicProjectionConfig,
    options: PublicBuildOptions = PublicBuildOptions(),
    *,
    dry_run: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> PublicDatasetBuildResult:
    """Resolve output first, otherwise project and publish one immutable candidate."""

    plan = plan_public_candidate(config)
    store = ArtifactStore(locations)
    reuse = store.resolve(plan.dataset_spec, version=options.version_override)
    if reuse is not None:
        summary = validate_public_dataset(store, reuse.artifact, progress=progress)
        return PublicDatasetBuildResult(
            plan=plan,
            artifact=reuse.artifact.pointer,
            path=reuse.artifact.path,
            dry_run=dry_run,
            built=False,
            reuse=reuse,
            validation=summary,
            source_accessed=False,
        )
    if dry_run:
        return PublicDatasetBuildResult(
            plan=plan,
            artifact=plan.dataset_spec.pointer,
            path=store.path_for(plan.dataset_spec.pointer),
            dry_run=True,
            built=False,
            reuse=None,
            validation=None,
            source_accessed=False,
        )

    _emit(
        progress,
        "public_candidate_build_started",
        plan_id=plan.plan_id,
        source=config.source_dataset.as_dict(),
    )
    validate_internal_dataset(store, config.source_dataset, semantic=True, deep_lineage=False)
    source = open_internal_dataset(store, config.source_dataset)
    started = time.monotonic()
    frame, omissions, public_return_evidence, summary = _project_source_dataset(
        source,
        config,
        progress_interval_seconds=options.progress_interval_seconds,
        progress_event="public_candidate_projection_progress",
        progress=progress,
    )
    validation_document = public_validation_document(
        frame,
        summary,
        config,
        omitted_dates=omissions,
        return_evidence=public_return_evidence,
    )
    capabilities = public_capabilities()
    metadata = public_manifest_metadata(
        config,
        summary,
        validation_document,
        distribution=DistributionStatus.PUBLIC_CANDIDATE,
    )
    builder = store.writer(plan.dataset_spec, metadata=metadata)
    try:
        with builder as staging:
            staging.write_bytes("data.parquet", _parquet_bytes(frame))
            staging.write_text(
                "README.md", candidate_readme(config, plan.dataset_spec.artifact_id)
            )
            staging.write_json("config.json", plan.configuration)
            staging.write_json("schema.json", public_schema_document())
            staging.write_json("capabilities.json", capabilities)
            staging.write_json("validation.json", validation_document)
            staged_summary, staged_metadata = validate_public_payload(
                builder.staging_path,
                plan.dataset_spec.artifact_id,
                config,
                distribution=DistributionStatus.PUBLIC_CANDIDATE,
            )
            if staged_summary != summary or staged_metadata != metadata:
                raise PublicProjectionError("Staged public validation evidence changed")
    except ArtifactExistsError:
        winner = store.validate(plan.dataset_spec.pointer, expected=plan.dataset_spec)
        winner_summary = validate_public_dataset(store, winner, progress=progress)
        return PublicDatasetBuildResult(
            plan=plan,
            artifact=winner.pointer,
            path=winner.path,
            dry_run=False,
            built=False,
            reuse=None,
            validation=winner_summary,
            source_accessed=True,
        )
    artifact = builder.artifact
    published_summary = validate_public_dataset(store, artifact, progress=progress)
    _emit(
        progress,
        "public_candidate_build_completed",
        elapsed_seconds=time.monotonic() - started,
        completed=summary.date_count,
        total=summary.date_count,
        failed=0,
        artifact=artifact.pointer.as_dict(),
    )
    return PublicDatasetBuildResult(
        plan=plan,
        artifact=artifact.pointer,
        path=artifact.path,
        dry_run=False,
        built=True,
        reuse=None,
        validation=published_summary,
        source_accessed=True,
    )


@version(
    "1.0",
    dependencies=[
        PublicReleaseConfig,
        _parquet_bytes,
        _project_source_dataset,
        public_dataset_producer_version,
        public_manifest_metadata,
        public_validation_document,
        release_readme,
        validate_public_dataset,
        validate_public_payload,
    ],
)
def public_release_producer_version() -> str:
    """Return the complete approved-release producer closure."""

    return str(public_release_producer_version.version)


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        DatasetDistribution,
        PublicDatasetPlan,
        PublicReleaseConfig,
        public_candidate_label,
        public_release_producer_version,
    ],
)
def plan_public_release(config: PublicReleaseConfig) -> PublicDatasetPlan:
    """Plan a distinct public release without opening candidate or source data."""

    projection = config.projection
    approval = config.approval
    distribution = DatasetDistribution(
        DistributionStatus.PUBLIC,
        release_license=approval.release_license,
        redistribution_basis=approval.redistribution_basis,
        approver=approval.approver,
        approval_date=approval.approval_date.isoformat(),
        approval_reference=approval.approval_reference,
    )
    specification = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace=f"sanos/public/name-{public_candidate_label(projection.name)}",
        configuration=config.as_dict(),
        schema_version="+".join(
            (
                PUBLIC_DATASET_SCHEMA_VERSION,
                PUBLIC_SCHEMA_DOCUMENT_VERSION,
                PUBLIC_VALIDATION_VERSION,
                FITTED_SCHEMA_VERSION,
                "sanos-public-release-evidence-v1",
            )
        ),
        producer_version=public_release_producer_version(),
        upstream=(projection.source_dataset,),
        distribution=distribution,
    )
    return PublicDatasetPlan(
        configuration=config.as_dict(),
        dataset_spec=specification,
        plan_id=canonical_sha256(
            {
                "release": specification.pointer.as_dict(),
                "candidate": approval.candidate_dataset.as_dict(),
                "source": projection.source_dataset.as_dict(),
            }
        ),
    )


@version(
    "1.0",
    dependencies=[
        PublicBuildOptions,
        PublicDatasetBuildResult,
        PublicReleaseConfig,
        _parquet_bytes,
        _project_source_dataset,
        open_internal_dataset,
        open_public_dataset,
        plan_public_release,
        public_frame_logical_sha256,
        public_manifest_metadata,
        public_validation_document,
        release_readme,
        validate_internal_dataset,
        validate_public_dataset,
        validate_public_payload,
    ],
)
def build_public_release(
    locations: StorageLocations,
    config: PublicReleaseConfig,
    license_bytes: bytes | None,
    options: PublicBuildOptions = PublicBuildOptions(),
    *,
    dry_run: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> PublicDatasetBuildResult:
    """Reproject and publish an approved immutable public Dataset."""

    plan = plan_public_release(config)
    store = ArtifactStore(locations)
    reuse = store.resolve(plan.dataset_spec, version=options.version_override)
    if reuse is not None:
        summary = validate_public_dataset(store, reuse.artifact, progress=progress)
        return PublicDatasetBuildResult(
            plan=plan,
            artifact=reuse.artifact.pointer,
            path=reuse.artifact.path,
            dry_run=dry_run,
            built=False,
            reuse=reuse,
            validation=summary,
            source_accessed=False,
        )
    if dry_run:
        return PublicDatasetBuildResult(
            plan=plan,
            artifact=plan.dataset_spec.pointer,
            path=store.path_for(plan.dataset_spec.pointer),
            dry_run=True,
            built=False,
            reuse=None,
            validation=None,
            source_accessed=False,
        )
    if license_bytes is None:
        raise ValueError("A concrete public release build requires exact license bytes")
    if hashlib.sha256(license_bytes).hexdigest() != config.approval.license_sha256:
        raise ValueError("Supplied license bytes do not match approval license_sha256")

    started = time.monotonic()
    _emit(
        progress,
        "public_release_build_started",
        plan_id=plan.plan_id,
        candidate=config.approval.candidate_dataset.as_dict(),
        source=config.projection.source_dataset.as_dict(),
    )
    candidate_validated = store.validate(config.approval.candidate_dataset)
    validate_public_dataset(store, candidate_validated, progress=progress)
    candidate = open_public_dataset(store, candidate_validated)
    if candidate.distribution is not DistributionStatus.PUBLIC_CANDIDATE:
        raise PublicProjectionError("Approval evidence does not identify a public candidate")
    if candidate.config != config.projection:
        raise PublicProjectionError("Approved candidate projection configuration changed")
    manifest_digest = hashlib.sha256(
        candidate_validated.path.joinpath(MANIFEST_NAME).read_bytes()
    ).hexdigest()
    if manifest_digest != config.approval.candidate_manifest_sha256:
        raise PublicProjectionError("Approved candidate manifest checksum changed")
    candidate_frame = candidate.load()
    candidate_logical_digest = public_frame_logical_sha256(candidate_frame)
    if candidate_logical_digest != config.approval.candidate_logical_content_sha256:
        raise PublicProjectionError("Approved candidate logical-content checksum changed")

    validate_internal_dataset(
        store,
        config.projection.source_dataset,
        semantic=True,
        deep_lineage=False,
    )
    source = open_internal_dataset(store, config.projection.source_dataset)
    frame, omissions, return_evidence, summary = _project_source_dataset(
        source,
        config.projection,
        progress_interval_seconds=options.progress_interval_seconds,
        progress_event="public_release_projection_progress",
        progress=progress,
    )
    if not frame.equals(candidate_frame, null_equal=True):
        raise PublicProjectionError(
            "Fresh release projection is not exactly equal to the approved candidate"
        )
    parquet_payload = _parquet_bytes(frame)
    if parquet_payload != candidate_validated.path.joinpath("data.parquet").read_bytes():
        raise PublicProjectionError(
            "Fresh release Parquet bytes differ from the approved candidate"
        )
    validation_document = public_validation_document(
        frame,
        summary,
        config.projection,
        omitted_dates=omissions,
        return_evidence=return_evidence,
    )
    approval_document = config.approval.as_dict()
    metadata = public_manifest_metadata(
        config.projection,
        summary,
        validation_document,
        distribution=DistributionStatus.PUBLIC,
        approval=approval_document,
    )
    builder = store.writer(plan.dataset_spec, metadata=metadata)
    try:
        with builder as staging:
            staging.write_bytes("data.parquet", parquet_payload)
            staging.write_text(
                "README.md", release_readme(config, plan.dataset_spec.artifact_id)
            )
            staging.write_json("config.json", plan.configuration)
            staging.write_json("schema.json", public_schema_document())
            staging.write_json("capabilities.json", public_capabilities())
            staging.write_json("validation.json", validation_document)
            staging.write_json("approval.json", approval_document)
            staging.write_bytes("LICENSE.txt", license_bytes)
            staged_summary, staged_metadata = validate_public_payload(
                builder.staging_path,
                plan.dataset_spec.artifact_id,
                config.projection,
                distribution=DistributionStatus.PUBLIC,
                release_config=config,
            )
            if staged_summary != summary or staged_metadata != metadata:
                raise PublicProjectionError("Staged public release evidence changed")
    except ArtifactExistsError:
        winner = store.validate(plan.dataset_spec.pointer, expected=plan.dataset_spec)
        winner_summary = validate_public_dataset(store, winner, progress=progress)
        return PublicDatasetBuildResult(
            plan=plan,
            artifact=winner.pointer,
            path=winner.path,
            dry_run=False,
            built=False,
            reuse=None,
            validation=winner_summary,
            source_accessed=True,
        )
    artifact = builder.artifact
    published_summary = validate_public_dataset(store, artifact, progress=progress)
    _emit(
        progress,
        "public_release_build_completed",
        elapsed_seconds=time.monotonic() - started,
        completed=summary.date_count,
        total=summary.date_count,
        failed=0,
        artifact=artifact.pointer.as_dict(),
    )
    return PublicDatasetBuildResult(
        plan=plan,
        artifact=artifact.pointer,
        path=artifact.path,
        dry_run=False,
        built=True,
        reuse=None,
        validation=published_summary,
        source_accessed=True,
    )


__all__ = [
    "PublicDatasetBuildResult",
    "PublicDatasetPlan",
    "build_public_candidate",
    "build_public_release",
    "plan_public_candidate",
    "plan_public_release",
    "public_candidate_label",
    "public_dataset_producer_version",
    "public_release_producer_version",
    "reproject_public_source",
]
