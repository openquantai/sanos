"""Deterministic, atomic storage for SANOS artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import warnings
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal

from cdxcore.version import version
from filelock import FileLock

from sanos.config import (
    StorageLocations,
    canonical_json,
    canonical_sha256,
    canonical_value,
)

MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "COMPLETE.json"
_MANIFEST_SCHEMA = "sanos-artifact-manifest-v1"
_COMPLETION_SCHEMA = "sanos-artifact-completion-v1"
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactError(RuntimeError):
    """Base class for artifact lifecycle failures."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an explicitly requested artifact does not exist."""


class ArtifactExistsError(ArtifactError):
    """Raised when publication would replace an existing artifact."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when an artifact is incomplete, malformed, or corrupted."""


class WildcardReuseMissError(ArtifactError):
    """Raised when ``version='*'`` cannot reuse a complete artifact."""


class AmbiguousArtifactError(ArtifactError):
    """Raised when wildcard reuse has multiple non-current candidates."""


@version("1.0")
class StorageClass(StrEnum):
    """The four and only four SANOS artifact storage classes."""

    RAW = "raw"
    INTERMEDIATE = "intermediate"
    DATASET = "dataset"
    EXPERIMENT = "experiment"

    @property
    def location_attribute(self) -> str:
        """Map the singular artifact class to its configured root attribute."""

        return {
            StorageClass.RAW: "raw",
            StorageClass.INTERMEDIATE: "intermediate",
            StorageClass.DATASET: "datasets",
            StorageClass.EXPERIMENT: "experiments",
        }[self]


@version("1.0")
class DistributionStatus(StrEnum):
    """Immutable redistribution state of a published Dataset."""

    INTERNAL = "internal"
    PUBLIC_CANDIDATE = "public_candidate"
    PUBLIC = "public"


@version("1.0", dependencies=[DistributionStatus])
@dataclass(frozen=True, slots=True)
class DatasetDistribution:
    """Dataset distribution identity and conditional public approval record."""

    status: DistributionStatus
    release_license: str | None = None
    redistribution_basis: str | None = None
    approver: str | None = None
    approval_date: str | None = None
    approval_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", DistributionStatus(self.status))
        approval = {
            "release_license": self.release_license,
            "redistribution_basis": self.redistribution_basis,
            "approver": self.approver,
            "approval_date": self.approval_date,
            "approval_reference": self.approval_reference,
        }
        if self.status is DistributionStatus.PUBLIC:
            missing = [name for name, value in approval.items() if not value or not value.strip()]
            if missing:
                raise ValueError(
                    f"A public Dataset requires non-empty approval fields: {', '.join(missing)}"
                )
            try:
                date.fromisoformat(self.approval_date or "")
            except ValueError as exc:
                raise ValueError("Dataset approval_date must be an ISO date") from exc
        elif any(value is not None for value in approval.values()):
            raise ValueError(
                "Release approval fields are allowed only when distribution status is public"
            )

    def as_dict(self) -> dict[str, str | None]:
        """Return the exact manifest and identity representation."""

        return {
            "status": self.status.value,
            "release_license": self.release_license,
            "redistribution_basis": self.redistribution_basis,
            "approver": self.approver,
            "approval_date": self.approval_date,
            "approval_reference": self.approval_reference,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetDistribution:
        """Parse an exact distribution object from an artifact manifest."""

        required = {
            "status",
            "release_license",
            "redistribution_basis",
            "approver",
            "approval_date",
            "approval_reference",
        }
        if set(value) != required:
            raise ValueError("Dataset distribution fields are missing or unexpected")
        optional = (
            "release_license",
            "redistribution_basis",
            "approver",
            "approval_date",
            "approval_reference",
        )
        if any(value[name] is not None and not isinstance(value[name], str) for name in optional):
            raise ValueError("Dataset distribution approval fields must be strings or null")
        return cls(
            status=DistributionStatus(value["status"]),
            release_license=value["release_license"],
            redistribution_basis=value["redistribution_basis"],
            approver=value["approver"],
            approval_date=value["approval_date"],
            approval_reference=value["approval_reference"],
        )


@version("1.0", dependencies=[StorageClass])
def _validate_lineage(
    storage_class: StorageClass,
    upstream: tuple[ArtifactPointer, ...],
) -> None:
    allowed = {
        StorageClass.RAW: frozenset({StorageClass.RAW}),
        StorageClass.INTERMEDIATE: frozenset({StorageClass.RAW, StorageClass.INTERMEDIATE}),
        StorageClass.DATASET: frozenset(
            {StorageClass.RAW, StorageClass.INTERMEDIATE, StorageClass.DATASET}
        ),
        StorageClass.EXPERIMENT: frozenset({StorageClass.DATASET}),
    }[storage_class]
    invalid = sorted(
        {
            pointer.storage_class.value
            for pointer in upstream
            if pointer.storage_class not in allowed
        }
    )
    if invalid:
        raise ValueError(
            f"{storage_class.value} cannot depend on storage classes: {', '.join(invalid)}"
        )
    if storage_class in {StorageClass.INTERMEDIATE, StorageClass.DATASET, StorageClass.EXPERIMENT}:
        if not upstream:
            raise ValueError(f"{storage_class.value} requires at least one upstream artifact")
    if storage_class is StorageClass.EXPERIMENT and not any(
        pointer.storage_class is StorageClass.DATASET for pointer in upstream
    ):
        raise ValueError("experiment requires a Dataset upstream artifact")


@version("1.0")
def _validate_namespace(namespace: str) -> str:
    path = PurePosixPath(namespace)
    if (
        len(namespace) > 1024
        or path.is_absolute()
        or not path.parts
        or any(
            len(part) > 128 or part in {"", ".", ".."} or _NAMESPACE_SEGMENT.fullmatch(part) is None
            for part in path.parts
        )
    ):
        raise ValueError(f"Invalid artifact namespace: {namespace!r}")
    return path.as_posix()


@version("1.0")
def _validate_artifact_id(artifact_id: str) -> str:
    if _HEX_256.fullmatch(artifact_id) is None:
        raise ValueError(f"Artifact ID must be a lowercase SHA-256 digest: {artifact_id!r}")
    return artifact_id


@version("1.0")
def _safe_relative_path(relative_path: str) -> PurePosixPath:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Artifact file path must be safe and relative: {relative_path!r}")
    return path


@version(
    "1.0",
    dependencies=[StorageClass, _validate_namespace, _validate_artifact_id],
)
@dataclass(frozen=True, slots=True, order=True)
class ArtifactPointer:
    """Stable location-independent reference to one immutable artifact."""

    storage_class: StorageClass
    namespace: str
    artifact_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_class", StorageClass(self.storage_class))
        object.__setattr__(self, "namespace", _validate_namespace(self.namespace))
        object.__setattr__(self, "artifact_id", _validate_artifact_id(self.artifact_id))

    def as_dict(self) -> dict[str, str]:
        """Return the canonical pointer payload stored in manifests."""

        return {
            "storage_class": self.storage_class.value,
            "namespace": self.namespace,
            "artifact_id": self.artifact_id,
        }


@version(
    "1.0",
    dependencies=[
        ArtifactPointer,
        DatasetDistribution,
        _validate_lineage,
        canonical_json,
        canonical_sha256,
    ],
)
@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Content identity inputs for one immutable artifact."""

    storage_class: StorageClass
    namespace: str
    configuration_json: str
    schema_version: str
    producer_version: str
    upstream: tuple[ArtifactPointer, ...] = ()
    distribution: DatasetDistribution | None = None
    source_generation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_class", StorageClass(self.storage_class))
        object.__setattr__(self, "namespace", _validate_namespace(self.namespace))
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("Artifact schema_version cannot be empty")
        if (
            not isinstance(self.producer_version, str)
            or not self.producer_version.strip()
            or self.producer_version == "*"
        ):
            raise ValueError("Artifact producer_version must be concrete and non-empty")
        try:
            configuration = json.loads(self.configuration_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Artifact configuration_json is not valid JSON") from exc
        normalized = canonical_json(configuration)
        if normalized != self.configuration_json:
            raise ValueError("Artifact configuration_json must already be canonical")
        ordered = tuple(sorted(set(self.upstream)))
        object.__setattr__(self, "upstream", ordered)
        _validate_lineage(self.storage_class, ordered)
        if self.storage_class is StorageClass.DATASET:
            if not isinstance(self.distribution, DatasetDistribution):
                raise ValueError("A Dataset artifact requires an immutable distribution contract")
            if self.distribution.status is not DistributionStatus.INTERNAL:
                if (
                    len(ordered) != 1
                    or ordered[0].storage_class is not StorageClass.DATASET
                ):
                    raise ValueError(
                        "A public_candidate or public Dataset requires its sole upstream "
                        "to be one Dataset projection source"
                    )
        elif self.distribution is not None:
            raise ValueError("Only Dataset artifacts may declare distribution state")
        if self.storage_class is StorageClass.RAW:
            if not isinstance(self.source_generation, str) or not self.source_generation.strip():
                raise ValueError("A Raw artifact requires a non-empty source_generation")
        elif self.source_generation is not None:
            raise ValueError("Only Raw artifacts may declare source_generation")

    @classmethod
    def create(
        cls,
        *,
        storage_class: StorageClass | str,
        namespace: str,
        configuration: Any,
        schema_version: str,
        producer_version: str,
        upstream: tuple[ArtifactPointer, ...] = (),
        distribution: DatasetDistribution | None = None,
        source_generation: str | None = None,
    ) -> ArtifactSpec:
        """Construct a spec while canonicalizing its configuration."""

        return cls(
            storage_class=StorageClass(storage_class),
            namespace=namespace,
            configuration_json=canonical_json(configuration),
            schema_version=schema_version,
            producer_version=producer_version,
            upstream=upstream,
            distribution=distribution,
            source_generation=source_generation,
        )

    @property
    def configuration(self) -> Any:
        """Return a fresh mutable decoding of the canonical configuration."""

        return json.loads(self.configuration_json)

    @property
    def configuration_id(self) -> str:
        """Return the SHA-256 identity of canonical configuration JSON."""

        return hashlib.sha256(self.configuration_json.encode("utf-8")).hexdigest()

    def identity_payload(self, *, include_producer: bool) -> dict[str, Any]:
        """Return identity inputs with or without producer-code version."""

        payload: dict[str, Any] = {
            "storage_class": self.storage_class.value,
            "namespace": self.namespace,
            "configuration_id": self.configuration_id,
            "schema_version": self.schema_version,
            "upstream": [pointer.as_dict() for pointer in self.upstream],
            "distribution": (None if self.distribution is None else self.distribution.as_dict()),
            "source_generation": self.source_generation,
        }
        if include_producer:
            payload["producer_version"] = self.producer_version
        return payload

    @property
    def artifact_id(self) -> str:
        """Return full identity, including producer-code version."""

        return canonical_sha256(self.identity_payload(include_producer=True))

    @property
    def compatibility_id(self) -> str:
        """Return wildcard-reuse identity, excluding only producer version."""

        return canonical_sha256(self.identity_payload(include_producer=False))

    @property
    def pointer(self) -> ArtifactPointer:
        """Return this spec's immutable artifact pointer."""

        return ArtifactPointer(self.storage_class, self.namespace, self.artifact_id)

    def with_producer_version(self, producer_version: str) -> ArtifactSpec:
        """Return the same content request for another concrete producer version."""

        return replace(self, producer_version=producer_version)


@version("1.0", dependencies=[ArtifactPointer])
@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    """A fully validated immutable artifact."""

    pointer: ArtifactPointer
    path: Path
    manifest: Mapping[str, Any]


@version("1.0", dependencies=[ValidatedArtifact])
@dataclass(frozen=True, slots=True)
class ReuseResult:
    """Output-first lookup result, including any explicit wildcard override."""

    artifact: ValidatedArtifact
    requested_version: str
    reused_producer_version: str
    warning: str | None

    @property
    def used_wildcard(self) -> bool:
        """Whether producer-version comparison was explicitly skipped."""

        return self.requested_version == "*"

    def as_log_record(self) -> dict[str, Any]:
        """Return the override record for CLI output or a run journal."""

        return {
            "artifact": self.artifact.pointer.as_dict(),
            "requested_version": self.requested_version,
            "reused_producer_version": self.reused_producer_version,
            "version_override": self.used_wildcard,
            "warning": self.warning,
        }


@version("1.0")
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@version("1.0")
def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@version("1.0")
def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


@version("1.0", dependencies=[_lexical_absolute])
def _assert_no_symlink_components(path: Path, *, include_leaf: bool = True) -> Path:
    absolute = _lexical_absolute(path)
    parts = absolute.parts
    current = Path(parts[0])
    limit = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:limit]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ArtifactIntegrityError(f"Storage path contains a symlink: {current}")
    return absolute


@version("1.0", dependencies=[_assert_no_symlink_components, _fsync_directory])
def _ensure_directory_no_symlinks(path: Path) -> Path:
    absolute = _assert_no_symlink_components(path)
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        parent = current
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                # A concurrent publisher may have created the same namespace
                # parent after lstat(). Re-check its type below before use.
                pass
            else:
                _fsync_directory(parent)
            mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ArtifactIntegrityError(f"Storage directory is unsafe: {current}")
    return absolute


@version("1.0", dependencies=[_assert_no_symlink_components])
def _safe_descendant(root: Path, relative: PurePosixPath) -> Path:
    base = _assert_no_symlink_components(root)
    destination = _assert_no_symlink_components(base.joinpath(*relative.parts))
    try:
        destination.relative_to(base)
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"Storage path escapes its configured root: {destination}"
        ) from exc
    return destination


@version("1.0", dependencies=[_fsync_directory])
def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if directory.is_symlink():
            raise ArtifactIntegrityError(f"Artifact directory is a symlink: {directory}")
        _fsync_directory(directory)
    _fsync_directory(root)


@version(
    "1.0",
    dependencies=[
        _lexical_absolute,
        _assert_no_symlink_components,
        _ensure_directory_no_symlinks,
        _fsync_directory,
    ],
)
def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write one file durably and atomically within its destination directory."""

    destination = _lexical_absolute(path)
    _ensure_directory_no_symlinks(destination.parent)
    _assert_no_symlink_components(destination)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _assert_no_symlink_components(destination)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@version("1.0", dependencies=[atomic_write_bytes, canonical_json])
def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write canonical JSON followed by one newline."""

    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


@version("1.0")
def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        destination.write(chunk)


@version("1.0", dependencies=[ArtifactSpec, ValidatedArtifact, DatasetDistribution])
def _validate_publication_upstreams(
    store: ArtifactStore,
    spec: ArtifactSpec,
) -> tuple[ValidatedArtifact, ...]:
    """Validate declared lineage immediately before publishing a new artifact."""

    validated = tuple(store.validate(pointer) for pointer in spec.upstream)
    distribution = spec.distribution
    if (
        spec.storage_class is StorageClass.DATASET
        and distribution is not None
        and distribution.status is not DistributionStatus.INTERNAL
    ):
        if (
            len(validated) != 1
            or validated[0].pointer.storage_class is not StorageClass.DATASET
        ):
            raise ArtifactIntegrityError(
                "A public projection requires one validated internal Dataset as its sole source"
            )
        source_distribution = validated[0].manifest.get("distribution")
        if not isinstance(source_distribution, Mapping):
            raise ArtifactIntegrityError(
                "A public projection source has no valid Dataset distribution contract"
            )
        try:
            parsed_source = DatasetDistribution.from_dict(source_distribution)
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "A public projection source has an invalid Dataset distribution contract"
            ) from exc
        if parsed_source.status is not DistributionStatus.INTERNAL:
            raise ArtifactIntegrityError(
                "A public projection source must be a completed internal Dataset"
            )
    return validated


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        ValidatedArtifact,
        atomic_write_bytes,
        atomic_write_json,
        _safe_relative_path,
        _safe_descendant,
        _assert_no_symlink_components,
        _ensure_directory_no_symlinks,
        _sha256_file,
        _fsync_directory,
        _fsync_tree_directories,
        _copy_stream,
        _validate_publication_upstreams,
    ],
)
class ArtifactBuilder(AbstractContextManager["ArtifactBuilder"]):
    """Locked staging directory published atomically on successful exit."""

    def __init__(
        self,
        store: ArtifactStore,
        spec: ArtifactSpec,
        *,
        metadata: Mapping[str, Any] | None,
        lock_timeout_seconds: float,
    ) -> None:
        self._store = store
        self.spec = spec
        self._metadata = canonical_value({} if metadata is None else metadata)
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock: FileLock | None = None
        self._staging: Path | None = None
        self._committed: ValidatedArtifact | None = None

    @property
    def staging_path(self) -> Path:
        """Return the private staging path after entering the context."""

        if self._staging is None:
            raise ArtifactError("ArtifactBuilder has not been entered")
        return self._staging

    @property
    def artifact(self) -> ValidatedArtifact:
        """Return the committed artifact after a successful context exit."""

        if self._committed is None:
            raise ArtifactError("ArtifactBuilder has not committed an artifact")
        return self._committed

    def __enter__(self) -> ArtifactBuilder:
        parent = self._store.namespace_path(self.spec.storage_class, self.spec.namespace)
        _ensure_directory_no_symlinks(parent)
        lock_directory = parent / ".locks"
        _ensure_directory_no_symlinks(lock_directory)
        lock_path = lock_directory / f"{self.spec.artifact_id}.lock"
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise ArtifactIntegrityError(f"Artifact lock path is unsafe: {lock_path}")
        self._lock = FileLock(
            str(lock_path),
            timeout=self._lock_timeout_seconds,
        )
        self._lock.acquire()
        try:
            final = self._store.path_for(self.spec.pointer)
            if final.exists() or final.is_symlink():
                raise ArtifactExistsError(
                    f"Artifact already exists and will not be replaced: {final}"
                )
            prefix = f".staging-{self.spec.artifact_id}-"
            for stale in parent.iterdir():
                if not stale.name.startswith(prefix):
                    continue
                if stale.is_symlink() or stale.is_file():
                    stale.unlink()
                elif stale.is_dir():
                    shutil.rmtree(stale)
                else:
                    raise ArtifactIntegrityError(f"Unsafe stale staging path: {stale}")
            self._staging = parent / f"{prefix}{uuid.uuid4().hex}"
            self._staging.mkdir()
            _fsync_directory(parent)
            return self
        except BaseException:
            self._release_lock()
            raise

    def _payload_destination(self, relative: PurePosixPath) -> Path:
        staging = self.staging_path
        _assert_no_symlink_components(staging)
        destination = _safe_descendant(staging, relative)
        _ensure_directory_no_symlinks(destination.parent)
        _assert_no_symlink_components(destination)
        return destination

    def write_bytes(self, relative_path: str, data: bytes) -> Path:
        """Write one payload file below the invisible staging directory."""

        relative = _safe_relative_path(relative_path)
        if relative.as_posix() in {MANIFEST_NAME, COMPLETION_NAME}:
            raise ValueError(f"{relative} is reserved by the artifact store")
        destination = self._payload_destination(relative)
        atomic_write_bytes(destination, data)
        return destination

    def write_text(self, relative_path: str, text: str) -> Path:
        """Write one UTF-8 payload file."""

        return self.write_bytes(relative_path, text.encode("utf-8"))

    def write_json(self, relative_path: str, value: Any) -> Path:
        """Write one canonical JSON payload file."""

        return self.write_text(relative_path, canonical_json(value) + "\n")

    def copy_file(self, relative_path: str, source: Path) -> Path:
        """Copy a regular file into staging without following a source symlink."""

        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Artifact copy source must be a regular file: {source}")
        relative = _safe_relative_path(relative_path)
        if relative.as_posix() in {MANIFEST_NAME, COMPLETION_NAME}:
            raise ValueError(f"{relative} is reserved by the artifact store")
        destination = self._payload_destination(relative)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                0o644,
            )
            with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
                _copy_stream(input_stream, output_stream)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            _assert_no_symlink_components(destination)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def commit(self) -> ValidatedArtifact:
        """Write manifest/completion last and atomically expose the directory."""

        staging = self.staging_path
        if self._committed is not None:
            raise ArtifactError("ArtifactBuilder.commit() was called twice")
        payload_paths: list[Path] = []
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise ArtifactIntegrityError(f"Artifact payload contains a symlink: {path}")
            if path.is_file():
                payload_paths.append(path)
        if not payload_paths:
            raise ArtifactIntegrityError("An artifact must contain at least one payload file")
        file_records = []
        for path in sorted(payload_paths):
            relative = path.relative_to(staging).as_posix()
            file_records.append(
                {
                    "path": relative,
                    "byte_size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        _validate_publication_upstreams(self._store, self.spec)
        manifest = {
            "schema": _MANIFEST_SCHEMA,
            "status": "complete",
            "artifact_id": self.spec.artifact_id,
            "compatibility_id": self.spec.compatibility_id,
            "storage_class": self.spec.storage_class.value,
            "namespace": self.spec.namespace,
            "schema_version": self.spec.schema_version,
            "producer_version": self.spec.producer_version,
            "configuration": self.spec.configuration,
            "configuration_id": self.spec.configuration_id,
            "upstream": [pointer.as_dict() for pointer in self.spec.upstream],
            "distribution": (
                None if self.spec.distribution is None else self.spec.distribution.as_dict()
            ),
            "source_generation": self.spec.source_generation,
            "created_at": datetime.now(UTC).isoformat(),
            "files": file_records,
            "metadata": self._metadata,
        }
        atomic_write_json(staging / MANIFEST_NAME, manifest)
        manifest_sha256 = _sha256_file(staging / MANIFEST_NAME)
        completion = {
            "schema": _COMPLETION_SCHEMA,
            "status": "complete",
            "artifact_id": self.spec.artifact_id,
            "manifest_sha256": manifest_sha256,
        }
        atomic_write_json(staging / COMPLETION_NAME, completion)
        _fsync_tree_directories(staging)
        final = self._store.path_for(self.spec.pointer)
        if final.exists() or final.is_symlink():
            raise ArtifactExistsError(f"Artifact already exists and will not be replaced: {final}")
        _assert_no_symlink_components(final)
        os.replace(staging, final)
        _fsync_directory(final.parent)
        self._staging = None
        self._committed = self._store.validate(self.spec.pointer, expected=self.spec)
        return self._committed

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        pending: BaseException | None = None
        try:
            if exc_type is None and self._committed is None:
                self.commit()
        except BaseException as error:
            pending = error
        finally:
            try:
                if self._staging is not None and (
                    self._staging.exists() or self._staging.is_symlink()
                ):
                    if self._staging.is_symlink():
                        raise ArtifactIntegrityError(
                            f"Staging path became a symlink: {self._staging}"
                        )
                    shutil.rmtree(self._staging)
                self._staging = None
            except BaseException as cleanup_error:
                primary = pending if pending is not None else exc
                if isinstance(primary, BaseException):
                    primary.add_note(f"Artifact staging cleanup also failed: {cleanup_error}")
                else:
                    pending = cleanup_error
            finally:
                self._release_lock()
        if pending is not None:
            raise pending
        return False

    def _release_lock(self) -> None:
        if self._lock is not None and self._lock.is_locked:
            self._lock.release()
        self._lock = None


@version("1.0", dependencies=[canonical_json])
def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(f"Missing or unsafe {label}: {path}")
    try:
        source = path.read_text(encoding="utf-8")
        value = json.loads(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"{label.capitalize()} is not a JSON object: {path}")
    try:
        expected_source = canonical_json(value) + "\n"
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"{label.capitalize()} is not canonical JSON: {path}") from exc
    if source != expected_source:
        raise ArtifactIntegrityError(f"{label.capitalize()} is not canonically encoded: {path}")
    return value


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        ArtifactPointer,
        ValidatedArtifact,
        ReuseResult,
        ArtifactBuilder,
        atomic_write_json,
        _safe_descendant,
        _safe_relative_path,
        _sha256_file,
        _read_json_object,
    ],
)
class ArtifactStore:
    """Explicit artifact store with shallow output-first validation."""

    def __init__(self, locations: StorageLocations) -> None:
        self.locations = locations

    def root_for(self, storage_class: StorageClass | str) -> Path:
        """Return the configured root for one canonical storage class."""

        value = StorageClass(storage_class)
        configured = getattr(self.locations, value.location_attribute)
        root = configured.expanduser().resolve(strict=False)
        if root.exists() and not root.is_dir():
            raise ArtifactIntegrityError(f"Configured storage root is not a directory: {root}")
        return root

    def namespace_path(self, storage_class: StorageClass | str, namespace: str) -> Path:
        """Return a safe namespace path without creating it."""

        normalized = PurePosixPath(_validate_namespace(namespace))
        return _safe_descendant(self.root_for(storage_class), normalized)

    def path_for(self, pointer: ArtifactPointer) -> Path:
        """Return an artifact's directory without requiring it to exist."""

        return self.namespace_path(pointer.storage_class, pointer.namespace) / pointer.artifact_id

    def writer(
        self,
        spec: ArtifactSpec,
        *,
        metadata: Mapping[str, Any] | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> ArtifactBuilder:
        """Create a locked atomic builder; existing artifacts are never replaced."""

        if lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds must be non-negative")
        return ArtifactBuilder(
            self,
            spec,
            metadata=metadata,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    def validate(
        self,
        pointer: ArtifactPointer,
        *,
        expected: ArtifactSpec | None = None,
        ignore_producer_version: bool = False,
        deep: bool = False,
        _visited: set[ArtifactPointer] | None = None,
    ) -> ValidatedArtifact:
        """Validate one artifact; upstream lineage is optional and off by default."""

        path = self.path_for(pointer)
        if path.is_symlink():
            raise ArtifactIntegrityError(f"Artifact path is a symlink: {path}")
        if not path.exists():
            raise ArtifactNotFoundError(f"Artifact does not exist: {path}")
        if not path.is_dir():
            raise ArtifactIntegrityError(f"Artifact path is not a regular directory: {path}")
        manifest_path = path / MANIFEST_NAME
        completion_path = path / COMPLETION_NAME
        manifest = _read_json_object(manifest_path, "manifest")
        completion = _read_json_object(completion_path, "completion marker")
        required_manifest = {
            "schema",
            "status",
            "artifact_id",
            "compatibility_id",
            "storage_class",
            "namespace",
            "schema_version",
            "producer_version",
            "configuration",
            "configuration_id",
            "upstream",
            "distribution",
            "source_generation",
            "created_at",
            "files",
            "metadata",
        }
        if set(manifest) != required_manifest:
            missing = sorted(required_manifest - set(manifest))
            unexpected = sorted(set(manifest) - required_manifest)
            raise ArtifactIntegrityError(
                f"Manifest fields changed for {path}: missing={missing}, unexpected={unexpected}"
            )
        if manifest["schema"] != _MANIFEST_SCHEMA or manifest["status"] != "complete":
            raise ArtifactIntegrityError(f"Unsupported or incomplete manifest: {manifest_path}")
        created_at = manifest["created_at"]
        if not isinstance(created_at, str):
            raise ArtifactIntegrityError(f"Manifest created_at is not a string: {manifest_path}")
        try:
            parsed_created_at = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"Manifest created_at is not ISO-8601: {manifest_path}"
            ) from exc
        if parsed_created_at.tzinfo is None:
            raise ArtifactIntegrityError(f"Manifest created_at has no timezone: {manifest_path}")
        if not isinstance(manifest["metadata"], dict):
            raise ArtifactIntegrityError(f"Manifest metadata is not an object: {manifest_path}")
        try:
            canonical_json(manifest["metadata"])
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"Manifest metadata is not canonicalizable: {manifest_path}"
            ) from exc
        if not isinstance(manifest["upstream"], list):
            raise ArtifactIntegrityError(f"Manifest upstream is not a list: {manifest_path}")
        try:
            manifest_pointer = ArtifactPointer(
                StorageClass(str(manifest["storage_class"])),
                str(manifest["namespace"]),
                str(manifest["artifact_id"]),
            )
            upstream = tuple(
                ArtifactPointer(
                    StorageClass(str(item["storage_class"])),
                    str(item["namespace"]),
                    str(item["artifact_id"]),
                )
                for item in manifest["upstream"]
            )
            distribution_value = manifest["distribution"]
            if distribution_value is not None and not isinstance(distribution_value, dict):
                raise ValueError("Dataset distribution must be an object or null")
            distribution = (
                None
                if distribution_value is None
                else DatasetDistribution.from_dict(distribution_value)
            )
            manifest_spec = ArtifactSpec.create(
                storage_class=manifest_pointer.storage_class,
                namespace=manifest_pointer.namespace,
                configuration=manifest["configuration"],
                schema_version=str(manifest["schema_version"]),
                producer_version=str(manifest["producer_version"]),
                upstream=upstream,
                distribution=distribution,
                source_generation=manifest["source_generation"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"Malformed artifact identity in {manifest_path}: {exc}"
            ) from exc
        if manifest_pointer != pointer:
            raise ArtifactIntegrityError(
                f"Manifest pointer {manifest_pointer} does not match requested {pointer}"
            )
        if manifest_spec.artifact_id != pointer.artifact_id:
            raise ArtifactIntegrityError(f"Artifact identity does not recompute for {path}")
        if manifest_spec.compatibility_id != manifest["compatibility_id"]:
            raise ArtifactIntegrityError(f"Compatibility identity does not recompute for {path}")
        if manifest_spec.configuration_id != manifest["configuration_id"]:
            raise ArtifactIntegrityError(f"Configuration identity does not recompute for {path}")
        if expected is not None:
            if ignore_producer_version:
                if manifest_spec.compatibility_id != expected.compatibility_id:
                    raise ArtifactIntegrityError(
                        "Artifact does not match requested "
                        f"schema/configuration/upstream identity: {path}"
                    )
            elif manifest_spec.artifact_id != expected.artifact_id:
                raise ArtifactIntegrityError(
                    f"Artifact does not match the requested identity: {path}"
                )

        required_completion = {"schema", "status", "artifact_id", "manifest_sha256"}
        if set(completion) != required_completion:
            raise ArtifactIntegrityError(f"Completion marker fields changed: {completion_path}")
        if not isinstance(completion["manifest_sha256"], str):
            raise ArtifactIntegrityError(f"Completion checksum is not a string: {completion_path}")
        if (
            completion["schema"] != _COMPLETION_SCHEMA
            or completion["status"] != "complete"
            or completion["artifact_id"] != pointer.artifact_id
            or completion["manifest_sha256"] != _sha256_file(manifest_path)
        ):
            raise ArtifactIntegrityError(f"Invalid completion marker: {completion_path}")

        records = manifest["files"]
        if not isinstance(records, list) or not records:
            raise ArtifactIntegrityError(f"Manifest has no payload file records: {manifest_path}")
        recorded_paths: list[str] = []
        for record in records:
            if not isinstance(record, dict) or set(record) != {"path", "byte_size", "sha256"}:
                raise ArtifactIntegrityError(f"Malformed file record in {manifest_path}")
            if not isinstance(record["path"], str):
                raise ArtifactIntegrityError(f"Payload path is not a string in {manifest_path}")
            byte_size = record["byte_size"]
            if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
                raise ArtifactIntegrityError(f"Invalid payload byte size in {manifest_path}")
            checksum = record["sha256"]
            if not isinstance(checksum, str) or _HEX_256.fullmatch(checksum) is None:
                raise ArtifactIntegrityError(f"Invalid payload checksum in {manifest_path}")
            try:
                relative = _safe_relative_path(record["path"])
            except ValueError as exc:
                raise ArtifactIntegrityError(str(exc)) from exc
            relative_text = relative.as_posix()
            if relative_text in {MANIFEST_NAME, COMPLETION_NAME}:
                raise ArtifactIntegrityError(f"Reserved file appears as payload: {relative_text}")
            payload = path.joinpath(*relative.parts)
            if payload.is_symlink() or not payload.is_file():
                raise ArtifactIntegrityError(f"Missing or non-regular payload file: {payload}")
            if payload.stat().st_size != byte_size:
                raise ArtifactIntegrityError(f"Payload byte size changed: {payload}")
            if _sha256_file(payload) != checksum:
                raise ArtifactIntegrityError(f"Payload checksum changed: {payload}")
            recorded_paths.append(relative_text)
        if recorded_paths != sorted(recorded_paths) or len(recorded_paths) != len(
            set(recorded_paths)
        ):
            raise ArtifactIntegrityError(
                f"Manifest payload paths are unsorted or duplicated: {path}"
            )
        actual_paths: list[str] = []
        actual_directories: set[str] = set()
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                raise ArtifactIntegrityError(f"Artifact contains a symlink: {candidate}")
            if candidate.is_file():
                relative = candidate.relative_to(path).as_posix()
                if relative not in {MANIFEST_NAME, COMPLETION_NAME}:
                    actual_paths.append(relative)
            elif candidate.is_dir():
                actual_directories.add(candidate.relative_to(path).as_posix())
            else:
                raise ArtifactIntegrityError(f"Artifact contains a special file: {candidate}")
        if sorted(actual_paths) != recorded_paths:
            raise ArtifactIntegrityError(
                f"Artifact payload set differs from manifest: expected={recorded_paths}, "
                f"actual={sorted(actual_paths)}"
            )
        expected_directories: set[str] = set()
        for relative_text in recorded_paths:
            relative = PurePosixPath(relative_text)
            expected_directories.update(
                parent.as_posix() for parent in relative.parents if parent.as_posix() != "."
            )
        if actual_directories != expected_directories:
            raise ArtifactIntegrityError(
                f"Artifact directory set differs from payload paths: expected="
                f"{sorted(expected_directories)}, actual={sorted(actual_directories)}"
            )

        if deep:
            visited = set() if _visited is None else _visited
            if pointer in visited:
                raise ArtifactIntegrityError(f"Artifact lineage contains a cycle at {pointer}")
            visited.add(pointer)
            try:
                validated_upstream = []
                for upstream_pointer in manifest_spec.upstream:
                    validated_upstream.append(
                        self.validate(upstream_pointer, deep=True, _visited=visited)
                    )
                distribution = manifest_spec.distribution
                if (
                    manifest_spec.storage_class is StorageClass.DATASET
                    and distribution is not None
                    and distribution.status is not DistributionStatus.INTERNAL
                ):
                    if (
                        len(validated_upstream) != 1
                        or validated_upstream[0].pointer.storage_class
                        is not StorageClass.DATASET
                    ):
                        raise ArtifactIntegrityError(
                            "A public projection requires one validated internal Dataset "
                            "as its sole source"
                        )
                    source_distribution = validated_upstream[0].manifest.get("distribution")
                    if (
                        not isinstance(source_distribution, Mapping)
                        or source_distribution.get("status")
                        != DistributionStatus.INTERNAL.value
                    ):
                        raise ArtifactIntegrityError(
                            "A public projection source must be a completed internal Dataset"
                        )
            finally:
                visited.remove(pointer)
        return ValidatedArtifact(pointer=pointer, path=path, manifest=manifest)

    def resolve(
        self,
        spec: ArtifactSpec,
        *,
        version: str | None = None,
        deep: bool = False,
    ) -> ReuseResult | None:
        """Resolve output first; wildcard reuse can never authorize a build."""

        requested_version = spec.producer_version if version is None else version
        if requested_version != "*":
            if not requested_version.strip():
                raise ValueError("Requested producer version cannot be empty")
            requested_spec = spec.with_producer_version(requested_version)
            path = self.path_for(requested_spec.pointer)
            if not path.exists():
                return None
            artifact = self.validate(requested_spec.pointer, expected=requested_spec, deep=deep)
            return ReuseResult(
                artifact=artifact,
                requested_version=requested_version,
                reused_producer_version=requested_version,
                warning=None,
            )

        candidates = tuple(self._wildcard_candidates(spec, deep=deep))
        if not candidates:
            raise WildcardReuseMissError(
                "version='*' may reuse only an existing complete artifact; "
                "no matching schema/configuration/upstream identity exists"
            )
        current = [
            candidate
            for candidate in candidates
            if candidate.pointer.artifact_id == spec.artifact_id
        ]
        if len(current) == 1:
            selected = current[0]
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            versions = sorted(
                str(candidate.manifest["producer_version"]) for candidate in candidates
            )
            raise AmbiguousArtifactError(
                "version='*' matched multiple complete producer versions; request one concrete "
                f"version instead: {versions}"
            )
        reused_version = str(selected.manifest["producer_version"])
        warning = (
            "Reusing a complete artifact with version='*'; producer-code-version "
            f"comparison was skipped (stored={reused_version!r}, current={spec.producer_version!r})"
        )
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
        return ReuseResult(
            artifact=selected,
            requested_version="*",
            reused_producer_version=reused_version,
            warning=warning,
        )

    def _wildcard_candidates(
        self, spec: ArtifactSpec, *, deep: bool
    ) -> Iterator[ValidatedArtifact]:
        namespace = self.namespace_path(spec.storage_class, spec.namespace)
        if not namespace.is_dir():
            return
        for path in sorted(namespace.iterdir()):
            if not path.is_dir() or _HEX_256.fullmatch(path.name) is None:
                continue
            manifest_path = path / MANIFEST_NAME
            try:
                manifest = _read_json_object(manifest_path, "manifest")
            except (ArtifactIntegrityError, ArtifactNotFoundError):
                continue
            if manifest.get("compatibility_id") != spec.compatibility_id:
                continue
            pointer = ArtifactPointer(spec.storage_class, spec.namespace, path.name)
            yield self.validate(
                pointer,
                expected=spec,
                ignore_producer_version=True,
                deep=deep,
            )
