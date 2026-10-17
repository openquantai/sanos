from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sanos.artifacts import (
    COMPLETION_NAME,
    MANIFEST_NAME,
    AmbiguousArtifactError,
    ArtifactExistsError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    DatasetDistribution,
    DistributionStatus,
    StorageClass,
    WildcardReuseMissError,
    atomic_create_json,
    atomic_write_bytes,
)
from sanos.config import StorageLocations


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        StorageLocations(
            raw=tmp_path / "raw",
            intermediate=tmp_path / "intermediate",
            datasets=tmp_path / "datasets",
            experiments=tmp_path / "experiments",
        )
    )


def _spec(
    *,
    producer: str = "producer-1",
    storage_class: StorageClass = StorageClass.DATASET,
    namespace: str = "widb/spx",
    upstream: tuple = (),
    distribution: DatasetDistribution | None = None,
    source_generation: str | None = None,
) -> ArtifactSpec:
    if not upstream and storage_class is StorageClass.DATASET:
        upstream = (
            _spec(storage_class=StorageClass.RAW, namespace="wrds/source").pointer,
        )
    if distribution is None and storage_class is StorageClass.DATASET:
        distribution = DatasetDistribution(DistributionStatus.INTERNAL)
    if source_generation is None and storage_class is StorageClass.RAW:
        source_generation = "source-generation-1"
    return ArtifactSpec.create(
        storage_class=storage_class,
        namespace=namespace,
        configuration={"ticker": "SPX", "years": [2022, 2023]},
        schema_version="schema-1",
        producer_version=producer,
        upstream=upstream,
        distribution=distribution,
        source_generation=source_generation,
    )


def _publish(store: ArtifactStore, spec: ArtifactSpec, text: str = "payload") -> Path:
    default_raw = _spec(storage_class=StorageClass.RAW, namespace="wrds/source")
    if (
        default_raw.pointer in spec.upstream
        and not store.path_for(default_raw.pointer).exists()
    ):
        _publish(store, default_raw)
    builder = store.writer(spec, metadata={"rows": 1})
    with builder as staging:
        staging.write_text("data/part.parquet", text)
        staging.write_json("config.json", spec.configuration)
    assert builder.artifact.pointer == spec.pointer
    return builder.artifact.path


def test_identity_is_canonical_and_upstream_order_independent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _spec(storage_class=StorageClass.RAW, namespace="wrds/first")
    second = _spec(storage_class=StorageClass.RAW, namespace="wrds/second")
    left = ArtifactSpec.create(
        storage_class="dataset",
        namespace="widb/spx",
        configuration={"b": 2, "a": 1},
        schema_version="one",
        producer_version="one",
        upstream=(second.pointer, first.pointer),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )
    right = ArtifactSpec.create(
        storage_class="dataset",
        namespace="widb/spx",
        configuration={"a": 1, "b": 2},
        schema_version="one",
        producer_version="one",
        upstream=(first.pointer, second.pointer),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )

    assert left.artifact_id == right.artifact_id
    assert store.path_for(left.pointer).name == left.artifact_id


def test_atomic_publication_and_shallow_validation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec()
    path = _publish(store, spec)

    validated = store.validate(spec.pointer, expected=spec)
    assert validated.path == path
    assert (path / MANIFEST_NAME).is_file()
    assert (path / COMPLETION_NAME).is_file()
    assert validated.manifest["status"] == "complete"
    assert validated.manifest["metadata"] == {"rows": 1}
    assert not list(path.parent.glob(f".staging-{spec.artifact_id}-*"))

    with pytest.raises(ArtifactExistsError):
        with store.writer(spec):
            pass


def test_failed_build_is_invisible_and_stale_staging_is_cleaned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec()
    with pytest.raises(RuntimeError, match="interrupt"):
        with store.writer(spec) as staging:
            staging.write_text("data.parquet", "partial")
            raise RuntimeError("interrupt")
    assert not store.path_for(spec.pointer).exists()

    parent = store.namespace_path(spec.storage_class, spec.namespace)
    stale = parent / f".staging-{spec.artifact_id}-stale"
    stale.mkdir()
    (stale / "partial").write_text("bad", encoding="utf-8")
    _publish(store, spec)
    assert not stale.exists()


def test_shallow_validation_survives_removed_upstream_but_deep_does_not(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    upstream = _spec(storage_class=StorageClass.RAW, namespace="wrds/options")
    upstream_path = _publish(store, upstream)
    downstream = _spec(upstream=(upstream.pointer,))
    _publish(store, downstream)
    shutil.rmtree(upstream_path)

    assert store.validate(downstream.pointer).pointer == downstream.pointer
    with pytest.raises(ArtifactNotFoundError):
        store.validate(downstream.pointer, deep=True)


def test_exact_and_wildcard_output_first_reuse(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _spec(producer="old-code")
    path = _publish(store, old)
    manifest_mtime = (path / MANIFEST_NAME).stat().st_mtime_ns
    current = _spec(producer="new-code")

    assert store.resolve(current) is None
    with pytest.warns(RuntimeWarning, match="version='\\*'"):
        reused = store.resolve(current, version="*")
    assert reused is not None
    assert reused.used_wildcard
    assert reused.reused_producer_version == "old-code"
    assert reused.as_log_record()["version_override"] is True
    assert (path / MANIFEST_NAME).stat().st_mtime_ns == manifest_mtime


def test_wildcard_never_builds_and_ambiguity_requires_concrete_version(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    current = _spec(producer="current")
    with pytest.raises(WildcardReuseMissError):
        store.resolve(current, version="*")
    assert not store.path_for(current.pointer).exists()

    one = _spec(producer="one")
    two = _spec(producer="two")
    _publish(store, one, "one")
    _publish(store, two, "two")
    with pytest.raises(AmbiguousArtifactError):
        store.resolve(current, version="*")

    _publish(store, current, "current")
    with pytest.warns(RuntimeWarning):
        selected = store.resolve(current, version="*")
    assert selected is not None
    assert selected.artifact.pointer == current.pointer


def test_integrity_checks_detect_corruption_unexpected_files_and_symlinks(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    corrupted = _spec(namespace="widb/corrupted")
    corrupted_path = _publish(store, corrupted)
    (corrupted_path / "data" / "part.parquet").write_text("changed", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="changed"):
        store.validate(corrupted.pointer)

    unexpected = _spec(namespace="widb/unexpected")
    unexpected_path = _publish(store, unexpected)
    (unexpected_path / "extra").write_text("extra", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="payload set"):
        store.validate(unexpected.pointer)

    linked = _spec(namespace="widb/linked")
    linked_path = _publish(store, linked)
    (linked_path / "link").symlink_to(linked_path / "config.json")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        store.validate(linked.pointer)


def test_manifest_schema_is_strict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec()
    path = _publish(store, spec)
    manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["unknown"] = True
    (path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Manifest"):
        store.validate(spec.pointer)


def test_atomic_file_write_and_reserved_builder_names(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "value.bin"
    atomic_write_bytes(destination, b"complete")
    assert destination.read_bytes() == b"complete"
    assert not list(destination.parent.glob(".value.bin.tmp-*"))

    store = _store(tmp_path / "store")
    spec = _spec()
    source = tmp_path / "source"
    source.write_text("source", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved"):
        with store.writer(spec) as builder:
            builder.copy_file(MANIFEST_NAME, source)


def test_artifact_foundation_is_versioned() -> None:
    from sanos.artifacts.store import ArtifactBuilder

    assert str(getattr(ArtifactSpec, "version")).startswith("1.0")
    builder_version = str(getattr(ArtifactBuilder, "version"))
    store_version = str(getattr(ArtifactStore, "version"))
    assert "_copy_stream" in builder_version
    assert "_fsync_tree_directories" in builder_version
    assert "_read_json_object" in store_version
    assert "_safe_relative_path" in store_version
    assert "_sha256_file" in store_version


def test_concurrent_publishers_can_create_shared_namespace_parents(tmp_path: Path) -> None:
    store = _store(tmp_path)
    specs = tuple(
        _spec(
            storage_class=StorageClass.RAW,
            namespace=f"wrds/options/ticker-SPX/year-2024/date-2024-01-{day:02d}",
        )
        for day in range(1, 17)
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = tuple(pool.map(lambda spec: _publish(store, spec), specs))

    assert all(path.is_dir() for path in paths)
    assert len(set(paths)) == len(specs)


def test_atomic_create_json_is_no_replace_under_concurrency(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "immutable.json"

    def create(value: int) -> tuple[int, bool]:
        try:
            atomic_create_json(destination, {"value": value})
        except FileExistsError:
            return value, False
        return value, True

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = tuple(pool.map(create, range(8)))

    winners = [value for value, created in outcomes if created]
    assert len(winners) == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "value": winners[0]
    }
    assert not list(destination.parent.glob(".immutable.json.tmp-*"))


def test_atomic_create_json_rejects_symlinked_leaf_and_parent(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("do not replace", encoding="utf-8")
    leaf = tmp_path / "leaf.json"
    leaf.symlink_to(target)
    with pytest.raises((ArtifactIntegrityError, FileExistsError)):
        atomic_create_json(leaf, {"unsafe": True})
    assert target.read_text(encoding="utf-8") == "do not replace"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        atomic_create_json(linked_parent / "value.json", {"unsafe": True})
    assert not (real_parent / "value.json").exists()


def test_atomic_create_json_cleans_up_setup_and_commit_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="unsafe"):
        atomic_create_json(blocked_parent / "value.json", {"value": 1})

    import sanos.artifacts.store as store_module

    destination = tmp_path / "commit" / "value.json"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected atomic-link failure")

    monkeypatch.setattr(store_module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected atomic-link failure"):
        atomic_create_json(destination, {"value": 2})
    assert not destination.exists()
    assert not list(destination.parent.glob(".value.json.tmp-*"))


def test_dataset_distribution_is_mandatory_immutable_and_public_is_approved(
    tmp_path: Path,
) -> None:
    raw = ArtifactPointer(StorageClass.RAW, "wrds/source", "0" * 64)
    with pytest.raises(ValueError, match="distribution"):
        ArtifactSpec.create(
            storage_class="dataset",
            namespace="widb/spx",
            configuration={},
            schema_version="one",
            producer_version="one",
            upstream=(raw,),
        )

    internal = _spec(distribution=DatasetDistribution(DistributionStatus.INTERNAL))
    candidate = _spec(
        distribution=DatasetDistribution(DistributionStatus.PUBLIC_CANDIDATE),
        upstream=(internal.pointer,),
    )
    assert internal.artifact_id != candidate.artifact_id
    assert candidate.distribution is not None
    assert candidate.distribution.release_license is None
    with pytest.raises(ValueError, match="approval fields"):
        DatasetDistribution(DistributionStatus.PUBLIC)

    public = _spec(
        upstream=(internal.pointer,),
        distribution=DatasetDistribution(
            DistributionStatus.PUBLIC,
            release_license="CC-BY-4.0",
            redistribution_basis="Approved derived-data release",
            approver="Dataset owner",
            approval_date="2026-10-12",
            approval_reference="approval:SANOS-1",
        ),
    )
    assert public.artifact_id not in {internal.artifact_id, candidate.artifact_id}
    store = _store(tmp_path)
    _publish(store, internal)
    _publish(store, public)
    manifest = store.validate(public.pointer).manifest
    assert manifest["distribution"]["status"] == "public"
    assert manifest["distribution"]["approval_reference"] == "approval:SANOS-1"

    with pytest.raises(ValueError, match="Dataset projection source"):
        _spec(
            distribution=DatasetDistribution(DistributionStatus.PUBLIC_CANDIDATE),
            upstream=(raw,),
        )


def test_publication_requires_real_upstreams_and_an_internal_projection_source(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    fictitious = ArtifactPointer(StorageClass.RAW, "wrds/missing", "0" * 64)
    missing = _spec(namespace="widb/missing", upstream=(fictitious,))
    with pytest.raises(ArtifactNotFoundError, match="does not exist"):
        _publish(store, missing)
    assert not store.path_for(missing.pointer).exists()

    internal = _spec(namespace="widb/internal")
    _publish(store, internal)
    approved = DatasetDistribution(
        DistributionStatus.PUBLIC,
        release_license="CC-BY-4.0",
        redistribution_basis="Approved derived-data release",
        approver="Dataset owner",
        approval_date="2026-10-12",
        approval_reference="approval:SANOS-1",
    )
    public = _spec(
        namespace="widb/public-source",
        upstream=(internal.pointer,),
        distribution=approved,
    )
    _publish(store, public)
    assert store.validate(public.pointer, deep=True).pointer == public.pointer

    candidate_on_public = _spec(
        namespace="widb/candidate-on-public",
        upstream=(public.pointer,),
        distribution=DatasetDistribution(DistributionStatus.PUBLIC_CANDIDATE),
    )
    with pytest.raises(ArtifactIntegrityError, match="must be a completed internal"):
        _publish(store, candidate_on_public)

    public_on_public = _spec(
        namespace="widb/public-on-public",
        upstream=(public.pointer,),
        distribution=approved,
    )
    with pytest.raises(ArtifactIntegrityError, match="must be a completed internal"):
        _publish(store, public_on_public)

    raw = _spec(storage_class=StorageClass.RAW, namespace="wrds/extra")
    intermediate = ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace="widb/extra",
        configuration={},
        schema_version="one",
        producer_version="one",
        upstream=(raw.pointer,),
    )
    for extra in (raw.pointer, intermediate.pointer):
        with pytest.raises(ValueError, match="sole upstream"):
            _spec(
                namespace="widb/candidate-mixed-source",
                upstream=(internal.pointer, extra),
                distribution=DatasetDistribution(DistributionStatus.PUBLIC_CANDIDATE),
            )
        with pytest.raises(ValueError, match="sole upstream"):
            _spec(
                namespace="widb/public-mixed-source",
                upstream=(internal.pointer, extra),
                distribution=approved,
            )


def test_lifecycle_lineage_policy_accepts_forward_edges_and_rejects_backward_edges() -> None:
    raw = _spec(storage_class=StorageClass.RAW, namespace="wrds/raw")
    intermediate = ArtifactSpec.create(
        storage_class="intermediate",
        namespace="widb/daily",
        configuration={},
        schema_version="one",
        producer_version="one",
        upstream=(raw.pointer,),
    )
    internal = ArtifactSpec.create(
        storage_class="dataset",
        namespace="widb/internal",
        configuration={},
        schema_version="one",
        producer_version="one",
        upstream=(intermediate.pointer,),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )
    public = ArtifactSpec.create(
        storage_class="dataset",
        namespace="widb/public",
        configuration={},
        schema_version="one",
        producer_version="one",
        upstream=(internal.pointer,),
        distribution=DatasetDistribution(DistributionStatus.PUBLIC_CANDIDATE),
    )
    experiment = ArtifactSpec.create(
        storage_class="experiment",
        namespace="dysanos/train",
        configuration={},
        schema_version="one",
        producer_version="one",
        upstream=(internal.pointer,),
    )
    trained_experiment = ArtifactSpec.create(
        storage_class="experiment",
        namespace="dysanos/direct200",
        configuration={},
        schema_version="one",
        producer_version="one",
        upstream=(experiment.pointer,),
    )
    assert public.upstream == (internal.pointer,)
    assert experiment.upstream == (internal.pointer,)
    assert trained_experiment.upstream == (experiment.pointer,)

    with pytest.raises(ValueError, match="raw cannot depend"):
        ArtifactSpec.create(
            storage_class="raw",
            namespace="bad/raw",
            configuration={},
            schema_version="one",
            producer_version="one",
            upstream=(experiment.pointer,),
            source_generation="generation",
        )
    with pytest.raises(ValueError, match="intermediate cannot depend"):
        ArtifactSpec.create(
            storage_class="intermediate",
            namespace="bad/intermediate",
            configuration={},
            schema_version="one",
            producer_version="one",
            upstream=(internal.pointer,),
        )
    with pytest.raises(ValueError, match="requires at least"):
        ArtifactSpec.create(
            storage_class="experiment",
            namespace="bad/experiment",
            configuration={},
            schema_version="one",
            producer_version="one",
        )
    with pytest.raises(ValueError, match="exactly one"):
        ArtifactSpec.create(
            storage_class="experiment",
            namespace="bad/multiple-experiment-inputs",
            configuration={},
            schema_version="one",
            producer_version="one",
            upstream=(internal.pointer, public.pointer),
        )
    with pytest.raises(ValueError, match="experiment cannot depend"):
        ArtifactSpec.create(
            storage_class="experiment",
            namespace="bad/intermediate-experiment-input",
            configuration={},
            schema_version="one",
            producer_version="one",
            upstream=(intermediate.pointer,),
        )


def test_raw_source_generation_distinguishes_refresh_and_exact_retry() -> None:
    first = _spec(storage_class=StorageClass.RAW, source_generation="wrds-run-1")
    retry = _spec(storage_class=StorageClass.RAW, source_generation="wrds-run-1")
    refreshed = _spec(storage_class=StorageClass.RAW, source_generation="wrds-run-2")

    assert first.artifact_id == retry.artifact_id
    assert first.artifact_id != refreshed.artifact_id
    with pytest.raises(ValueError, match="source_generation"):
        ArtifactSpec.create(
            storage_class="raw",
            namespace="wrds/raw",
            configuration={},
            schema_version="one",
            producer_version="one",
        )


def test_atomic_and_artifact_writes_reject_symlink_escapes(tmp_path: Path) -> None:
    target = tmp_path / "outside-target"
    target.write_text("unchanged", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.symlink_to(target)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        atomic_write_bytes(destination, b"changed")
    assert target.read_text(encoding="utf-8") == "unchanged"

    external_parent = tmp_path / "external-parent"
    external_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(external_parent, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        atomic_write_bytes(linked_parent / "escaped", b"changed")
    assert not (external_parent / "escaped").exists()

    store = _store(tmp_path / "store")
    external = tmp_path / "external"
    external.mkdir()
    store.locations.datasets.mkdir(parents=True)
    (store.locations.datasets / "widb").symlink_to(external, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        with store.writer(_spec(namespace="widb/escape")):
            pass

    safe = _spec(namespace="safe/nested")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        with store.writer(safe) as builder:
            (builder.staging_path / "data").symlink_to(external, target_is_directory=True)
            builder.write_text("data/escaped", "bad")
    assert not (external / "escaped").exists()


def test_dangling_final_and_lock_symlinks_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    spec = _spec()
    parent = store.namespace_path(spec.storage_class, spec.namespace)
    parent.mkdir(parents=True)
    final = store.path_for(spec.pointer)
    final.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ArtifactExistsError):
        with store.writer(spec):
            pass
    final.unlink()

    lock_directory = parent / ".locks"
    lock_directory.mkdir(exist_ok=True)
    lock_path = lock_directory / f"{spec.artifact_id}.lock"
    lock_path.unlink()
    lock_path.symlink_to(tmp_path / "outside-lock")
    with pytest.raises(ArtifactIntegrityError, match="lock"):
        with store.writer(spec):
            pass


def test_lock_is_released_when_stale_or_exit_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sanos.artifacts.store as store_module

    store = _store(tmp_path)
    spec = _spec()
    parent = store.namespace_path(spec.storage_class, spec.namespace)
    parent.mkdir(parents=True)
    stale = parent / f".staging-{spec.artifact_id}-stale"
    stale.mkdir()
    builder = store.writer(spec)

    def fail_cleanup(path: Path) -> None:
        raise OSError(f"injected cleanup failure: {path}")

    monkeypatch.setattr(store_module.shutil, "rmtree", fail_cleanup)
    with pytest.raises(OSError, match="injected"):
        builder.__enter__()
    assert builder._lock is None

    stale.rmdir()
    entered = store.writer(spec)
    entered.__enter__()
    entered.write_text("payload", "partial")
    body_error = RuntimeError("body failure")
    result = entered.__exit__(RuntimeError, body_error, body_error.__traceback__)
    assert result is False
    assert entered._lock is None
    assert any("cleanup also failed" in note for note in getattr(body_error, "__notes__", []))


def test_lock_is_released_when_implicit_commit_fails(tmp_path: Path) -> None:
    store = _store(tmp_path)
    builder = store.writer(_spec())
    with pytest.raises(ArtifactIntegrityError, match="payload"):
        with builder:
            pass
    assert builder._lock is None


def test_nested_artifact_directories_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sanos.artifacts.store as store_module

    calls: list[Path] = []
    original = store_module._fsync_directory

    def record(path: Path) -> None:
        calls.append(path)
        original(path)

    monkeypatch.setattr(store_module, "_fsync_directory", record)
    _publish(_store(tmp_path), _spec())
    assert any(path.name == "data" for path in calls)


def test_artifact_pointer_namespace_is_bounded() -> None:
    with pytest.raises(ValueError, match="namespace"):
        ArtifactPointer(StorageClass.INTERMEDIATE, "x" * 1025, "0" * 64)
