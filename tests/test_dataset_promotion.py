from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from click.testing import CliRunner

import sanos.datasets.promotion as promotion_module
from sanos.artifacts import (
    ArtifactError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    DatasetDistribution,
    DistributionStatus,
    StorageClass,
    WildcardReuseMissError,
    atomic_write_json,
)
from sanos.cli import main
from sanos.config import StorageLocations, canonical_sha256
from sanos.datasets.promotion import (
    PROMOTION_SCHEMA,
    CanonicalReferencePromotionConfig,
    CanonicalReferencePromotionResult,
    ReferencePromotionError,
    create_promotion_config,
    create_review_attestation,
    make_review_attestation,
    plan_reference_promotion,
    promote_reference_dataset,
    validate_reference_promotion,
)
from sanos.production import PHASE11_EVIDENCE_INDEX_SCHEMA
from sanos.reference import (
    EXPERIMENT_REFERENCE_REPORT_SCHEMA,
    REFERENCE_REPORT_SCHEMA,
)
from sanos.runtime import runtime_fingerprint


def _locations(root: Path) -> StorageLocations:
    return StorageLocations(
        raw=root / "raw",
        intermediate=root / "intermediate",
        datasets=root / "datasets",
        experiments=root / "experiments",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_source_dataset(store: ArtifactStore) -> object:
    raw = ArtifactSpec.create(
        storage_class=StorageClass.RAW,
        namespace="wrds/reference-test",
        configuration={"source": "fixture"},
        schema_version="raw-test-v1",
        producer_version="test-v1",
        source_generation="test-v1",
    )
    with store.writer(raw, metadata={}) as builder:
        builder.write_text("data.parquet", "fixture")
    dataset = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace="widb/internal/name-reference-test",
        configuration={"name": "reference-test"},
        schema_version="dataset-test-v1",
        producer_version="test-v1",
        upstream=(raw.pointer,),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )
    with store.writer(dataset, metadata={"source_revision": "a" * 40}) as builder:
        builder.write_text("data/part.parquet", "fixture")
    return store.validate(dataset.pointer)


def _report(value: dict[str, object]) -> dict[str, object]:
    return {"report_id": canonical_sha256(value), **value}


def _write_evidence_index(
    path: Path,
    *,
    source_revision: str,
    artifact_binding: dict[str, object],
) -> dict[str, object]:
    runtime = runtime_fingerprint()
    runtime_path = path.parent / "runtime-fingerprint.json"
    atomic_write_json(runtime_path, runtime)
    public_pointer = ArtifactPointer(
        StorageClass.DATASET,
        "sanos/public-candidate/name-reference-test",
        "7" * 64,
    )
    public_manifest_sha256 = "8" * 64
    artifact_binding = {
        **artifact_binding,
        "public_candidate_binding": {
            "status": "passed",
            "public_candidate": public_pointer.as_dict(),
            "public_candidate_manifest_sha256": public_manifest_sha256,
        },
    }
    evidence_names = (
        "raw_preflight",
        "raw_validation",
        "dataset_validation",
        "daily_comparison",
        "downstream_configs",
        "public_candidate_validation",
        "experiment_validation",
        "experiment_comparison",
    )
    body: dict[str, object] = {
        "schema": PHASE11_EVIDENCE_INDEX_SCHEMA,
        "run_id": "test-run",
        "source_revision": source_revision,
        "network_accessed_during_validation": False,
        "source_generation": "default-v1",
        "resolved_versions": {
            "public_plan": "test-public-plan-v1",
            "experiment_plan": "test-experiment-plan-v1",
            "runtime_fingerprint": runtime["fingerprint_id"],
        },
        "coverage": {},
        "artifacts": {
            "public_candidate": {
                **public_pointer.as_dict(),
                "manifest_sha256": public_manifest_sha256,
            }
        },
        "artifact_binding": artifact_binding,
        "evidence": {
            name: {"file": f"{name}.json", "sha256": "9" * 64, "bytes": 1}
            for name in evidence_names
        }
        | {
            "runtime_fingerprint": {
                "file": runtime_path.name,
                "sha256": _sha256(runtime_path),
                "bytes": runtime_path.stat().st_size,
                "identity": runtime["fingerprint_id"],
            }
        },
    }
    index = {"evidence_id": canonical_sha256(body), **body}
    atomic_write_json(path, index)
    return index


class _PassedSummary:
    def as_dict(self) -> dict[str, object]:
        return {"status": "passed"}


def _promotion_build_fixture(
    tmp_path: Path,
) -> tuple[
    StorageLocations,
    CanonicalReferencePromotionConfig,
    dict[str, Path],
]:
    locations = _locations(tmp_path / "build-data")
    store = ArtifactStore(locations)
    raw = ArtifactSpec.create(
        storage_class=StorageClass.RAW,
        namespace="wrds/promotion-build-test",
        configuration={"source": "fixture"},
        schema_version="raw-test-v1",
        producer_version="test-v1",
        source_generation="test-v1",
    )
    with store.writer(raw, metadata={}) as builder:
        builder.write_text("data.parquet", "fixture")
    dates: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    for index in range(918):
        label = f"session-{index:04d}"
        data_sha = hashlib.sha256(label.encode()).hexdigest()
        fitted = ArtifactSpec.create(
            storage_class=StorageClass.INTERMEDIATE,
            namespace=f"widb/fitted-day/{label}",
            configuration={"date": label},
            schema_version="fitted-test-v1",
            producer_version="test-v1",
            upstream=(raw.pointer,),
        ).pointer
        partitions.append(
            {
                "trading_date": label,
                "fitted_artifact": fitted.as_dict(),
                "fitted_source_sha256": data_sha,
            }
        )
        dates.append(
            {
                "date": label,
                "classification": "match",
                "fresh": {"artifact": fitted.as_dict(), "data_sha256": data_sha},
            }
        )
    dataset_spec = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace="widb/internal/name-promotion-build-test",
        configuration={"name": "promotion-build-test"},
        schema_version="dataset-test-v1",
        producer_version="test-v1",
        upstream=(raw.pointer,),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    )
    with store.writer(dataset_spec, metadata={"source_revision": "1" * 40}) as builder:
        builder.write_text("data/part.parquet", "fixture")
        builder.write_json("partitions.json", {"partitions": partitions})
    dataset = store.validate(dataset_spec.pointer)
    experiment_spec = ArtifactSpec.create(
        storage_class=StorageClass.EXPERIMENT,
        namespace="dysanos/daily-front-market/name-promotion-build-test",
        configuration={"name": "promotion-build-test"},
        schema_version="experiment-test-v1",
        producer_version="test-v1",
        upstream=(dataset.pointer,),
    )
    with store.writer(experiment_spec, metadata={}) as builder:
        builder.write_text("data.h5", "fixture")
    experiment = store.validate(experiment_spec.pointer)

    evidence = tmp_path / "build-evidence"
    evidence.mkdir()
    daily_value: dict[str, object] = {
        "schema": REFERENCE_REPORT_SCHEMA,
        "outcome": "passed",
        "production_contract": {"passed": True},
        "classification_counts": {"match": 891, "oracle_gap": 26, "policy_empty": 1},
        "approval_document": None,
        "dates": dates,
    }
    daily = _report(daily_value)
    daily_path = evidence / "daily.json"
    atomic_write_json(daily_path, daily)
    experiment_value: dict[str, object] = {
        "schema": EXPERIMENT_REFERENCE_REPORT_SCHEMA,
        "outcome": "passed",
        "phase11_contract": {"passed": True},
        "experiment": {
            "artifact": experiment.pointer.as_dict(),
            "manifest_sha256": _sha256(experiment.path / "manifest.json"),
            "source_dataset": dataset.pointer.as_dict(),
        },
    }
    experiment_report = _report(experiment_value)
    experiment_path = evidence / "experiment.json"
    atomic_write_json(experiment_path, experiment_report)
    binding = promotion_module.validate_reference_evidence_binding(
        store,
        dataset=dataset.pointer,
        daily_report_path=daily_path,
        experiment_report_path=experiment_path,
    )
    evidence_path = evidence / "evidence-index.json"
    evidence_index = _write_evidence_index(
        evidence_path,
        source_revision="1" * 40,
        artifact_binding=binding,
    )
    paths = {
        "daily": daily_path,
        "experiment": experiment_path,
        "evidence": evidence_path,
        "code": evidence / "code.json",
        "quant": evidence / "quant.json",
    }
    for role, output in (("code", paths["code"]), ("quantitative", paths["quant"])):
        attestation = create_review_attestation(
            store,
            role=cast(Literal["code", "quantitative"], role),
            reviewer=f"{role}-reviewer",
            review_date=date(2026, 10, 13),
            review_reference=f"review://{role}/build-test",
            dataset=dataset.pointer,
            daily_report_path=daily_path,
            experiment_report_path=experiment_path,
            evidence_index_path=evidence_path,
        )
        atomic_write_json(output, attestation)
    config = create_promotion_config(
        store,
        name="promotion-build-test",
        dataset=dataset.pointer,
        daily_report_path=daily_path,
        experiment_report_path=experiment_path,
        evidence_index_path=evidence_path,
        code_review_path=paths["code"],
        quantitative_review_path=paths["quant"],
    )
    assert evidence_index["evidence_id"] == config.evidence_index_id
    return locations, config, paths


def test_promotion_is_output_first_self_contained_and_wildcard_reusable(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path / "data")
    store = ArtifactStore(locations)
    dataset = _publish_source_dataset(store)
    dataset_manifest_sha = _sha256(dataset.path / "manifest.json")
    dataset_configuration_id = str(dataset.manifest["configuration_id"])
    experiment_pointer = ArtifactSpec.create(
        storage_class=StorageClass.EXPERIMENT,
        namespace="dysanos/daily-front-market/name-reference-test",
        configuration={"name": "reference-test"},
        schema_version="experiment-test-v1",
        producer_version="test-v1",
        upstream=(dataset.pointer,),
    ).pointer
    experiment_manifest_sha = "b" * 64

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    daily_value: dict[str, object] = {
        "schema": REFERENCE_REPORT_SCHEMA,
        "outcome": "passed",
        "production_contract": {"passed": True},
        "classification_counts": {"match": 891, "oracle_gap": 26, "policy_empty": 1},
        "approval_document": None,
    }
    daily = _report(daily_value)
    daily_path = evidence / "daily.json"
    atomic_write_json(daily_path, daily)
    experiment_value: dict[str, object] = {
        "schema": EXPERIMENT_REFERENCE_REPORT_SCHEMA,
        "outcome": "passed",
        "phase11_contract": {"passed": True},
        "experiment": {
            "artifact": experiment_pointer.as_dict(),
            "manifest_sha256": experiment_manifest_sha,
            "source_dataset": dataset.pointer.as_dict(),
        },
    }
    experiment_report = _report(experiment_value)
    experiment_path = evidence / "experiment.json"
    atomic_write_json(experiment_path, experiment_report)

    partition_binding = {
        "status": "passed",
        "partition_count": 918,
        "bound_fitted_artifacts": 918,
        "binding_sha256": "f" * 64,
    }
    evidence_path = evidence / "evidence-index.json"
    evidence_index = _write_evidence_index(
        evidence_path,
        source_revision="a" * 40,
        artifact_binding={
            "status": "passed",
            "dataset": dataset.pointer.as_dict(),
            "dataset_manifest_sha256": dataset_manifest_sha,
            "dataset_configuration_id": dataset_configuration_id,
            "dataset_source_revision": "a" * 40,
            "daily_report_id": str(daily["report_id"]),
            "daily_report_sha256": _sha256(daily_path),
            "daily_approvals_sha256": None,
            "partition_binding": partition_binding,
            "experiment": experiment_pointer.as_dict(),
            "experiment_manifest_sha256": experiment_manifest_sha,
            "experiment_report_id": str(experiment_report["report_id"]),
            "experiment_report_sha256": _sha256(experiment_path),
        },
    )

    review_identity = {
        "dataset": dataset.pointer,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "daily_report_id": str(daily["report_id"]),
        "daily_report_sha256": _sha256(daily_path),
        "experiment": experiment_pointer,
        "experiment_report_id": str(experiment_report["report_id"]),
        "experiment_report_sha256": _sha256(experiment_path),
        "evidence_index_id": str(evidence_index["evidence_id"]),
        "evidence_index_sha256": _sha256(evidence_path),
    }
    code_review = make_review_attestation(
        role="code",
        reviewer="code-reviewer",
        review_date=date(2026, 10, 13),
        review_reference="review://code/1",
        **review_identity,
    )
    quantitative_review = make_review_attestation(
        role="quantitative",
        reviewer="quant-reviewer",
        review_date=date(2026, 10, 13),
        review_reference="review://quant/1",
        **review_identity,
    )
    code_path = evidence / "code.json"
    quant_path = evidence / "quant.json"
    atomic_write_json(code_path, code_review)
    atomic_write_json(quant_path, quantitative_review)
    config = CanonicalReferencePromotionConfig(
        name="reference-test",
        dataset=dataset.pointer,
        dataset_manifest_sha256=dataset_manifest_sha,
        dataset_configuration_id=dataset_configuration_id,
        dataset_source_revision="a" * 40,
        daily_report_id=str(daily["report_id"]),
        daily_report_sha256=_sha256(daily_path),
        daily_approvals_sha256=None,
        experiment=experiment_pointer,
        experiment_manifest_sha256=experiment_manifest_sha,
        experiment_report_id=str(experiment_report["report_id"]),
        experiment_report_sha256=_sha256(experiment_path),
        evidence_index_id=str(evidence_index["evidence_id"]),
        evidence_index_sha256=_sha256(evidence_path),
        code_review_sha256=_sha256(code_path),
        quantitative_review_sha256=_sha256(quant_path),
    )
    plan = plan_reference_promotion(config)
    old_spec = plan.spec.with_producer_version("old-reviewed-producer-v1")
    evidence_records = cast(dict[str, Any], evidence_index["evidence"])
    runtime_record = cast(dict[str, Any], evidence_records["runtime_fingerprint"])
    promotion = {
        "schema": PROMOTION_SCHEMA,
        "canonical_promotion_ready": True,
        **{key: value for key, value in config.to_document().items() if key != "name"},
        "runtime_fingerprint_id": runtime_record["identity"],
        "runtime_fingerprint_sha256": runtime_record["sha256"],
        "validation": {
            "dataset_shallow_semantic": "passed",
            "dataset_deep_lineage": "passed",
            "dataset_summary": {"status": "passed"},
            "experiment_semantic": "passed",
            "experiment_summary": {"status": "passed"},
            "partition_binding": partition_binding,
        },
        "distribution_effect": "none; designation does not grant redistribution rights",
    }
    with store.writer(
        old_spec,
        metadata={"canonical_promotion_ready": True},
    ) as builder:
        builder.write_text("README.md", "test promotion\n")
        builder.write_json("promotion.json", promotion)
        builder.copy_file("daily-comparison.json", daily_path)
        builder.copy_file("experiment-comparison.json", experiment_path)
        builder.copy_file("evidence-index.json", evidence_path)
        builder.copy_file(
            "runtime-fingerprint.json", evidence / "runtime-fingerprint.json"
        )
        builder.copy_file("code-review.json", code_path)
        builder.copy_file("quantitative-review.json", quant_path)
    source_path = dataset.path
    shutil.rmtree(source_path)

    with pytest.warns(RuntimeWarning, match="version='\\*'"):
        result = promote_reference_dataset(
            locations,
            config,
            version_override="*",
        )
    assert result.built is False
    assert result.reuse is not None and result.reuse.used_wildcard
    assert result.validation is not None
    assert result.validation["canonical_promotion_ready"] is True
    assert not source_path.exists()
    assert validate_reference_promotion(store, result.artifact)["status"] == "passed"
    cli = CliRunner().invoke(
        main,
        [
            "--root",
            str(locations.raw.parent),
            "--log-format",
            "json",
            "dataset",
            "reference",
            "validate",
            "--name",
            config.name,
            "--artifact-id",
            result.artifact.artifact_id,
        ],
        env={"WRDS_OFF": "1"},
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["validation"]["status"] == "passed"

    changed = replace(config, name="missing-reference")
    with pytest.raises(WildcardReuseMissError):
        promote_reference_dataset(locations, changed, version_override="*")


def test_promotion_build_is_concurrent_atomic_and_rejects_stale_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, config, paths = _promotion_build_fixture(tmp_path)
    store = ArtifactStore(locations)
    def validate_dataset(*_args: object, **_kwargs: object) -> _PassedSummary:
        return _PassedSummary()

    def validate_experiment(*_args: object, **_kwargs: object) -> _PassedSummary:
        return _PassedSummary()

    monkeypatch.setattr(promotion_module, "validate_internal_dataset", validate_dataset)
    monkeypatch.setattr(promotion_module, "validate_experiment", validate_experiment)

    def publish() -> CanonicalReferencePromotionResult:
        return promote_reference_dataset(
            locations,
            config,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=paths["evidence"],
            code_review_path=paths["code"],
            quantitative_review_path=paths["quant"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert sorted(result.built for result in results) == [False, True]
    assert {result.artifact for result in results} == {
        plan_reference_promotion(config).spec.pointer
    }
    assert validate_reference_promotion(store, results[0].artifact)["status"] == "passed"

    missing_review = replace(config, name="promotion-missing-review")
    with pytest.raises(ReferencePromotionError, match="code review"):
        promote_reference_dataset(
            locations,
            missing_review,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=paths["evidence"],
            quantitative_review_path=paths["quant"],
        )

    stale_index = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    stale_index["artifact_binding"]["daily_report_id"] = "8" * 64
    stale_body = {key: value for key, value in stale_index.items() if key != "evidence_id"}
    stale_index["evidence_id"] = canonical_sha256(stale_body)
    stale_path = paths["evidence"].with_name("stale-evidence-index.json")
    atomic_write_json(stale_path, stale_index)
    with pytest.raises(ReferencePromotionError, match="daily_report_id"):
        create_review_attestation(
            store,
            role="code",
            reviewer="code-reviewer",
            review_date=date(2026, 10, 13),
            review_reference="review://code/stale",
            dataset=config.dataset,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=stale_path,
        )

    legacy_index = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    legacy_index["schema"] = "sanos-phase11-evidence-index-v2"
    legacy_body = {key: value for key, value in legacy_index.items() if key != "evidence_id"}
    legacy_index["evidence_id"] = canonical_sha256(legacy_body)
    legacy_path = paths["evidence"].with_name("legacy-evidence-index.json")
    atomic_write_json(legacy_path, legacy_index)
    with pytest.raises(ReferencePromotionError, match="schema changed"):
        create_review_attestation(
            store,
            role="code",
            reviewer="code-reviewer",
            review_date=date(2026, 10, 13),
            review_reference="review://code/legacy",
            dataset=config.dataset,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=legacy_path,
        )

    mismatched_runtime_index = json.loads(
        paths["evidence"].read_text(encoding="utf-8")
    )
    mismatched_runtime_index["resolved_versions"]["runtime_fingerprint"] = "0" * 64
    mismatched_body = {
        key: value
        for key, value in mismatched_runtime_index.items()
        if key != "evidence_id"
    }
    mismatched_runtime_index["evidence_id"] = canonical_sha256(mismatched_body)
    mismatched_path = paths["evidence"].with_name("mismatched-runtime-index.json")
    atomic_write_json(mismatched_path, mismatched_runtime_index)
    with pytest.raises(ReferencePromotionError, match="runtime identity differs"):
        create_review_attestation(
            store,
            role="code",
            reviewer="code-reviewer",
            review_date=date(2026, 10, 13),
            review_reference="review://code/runtime-mismatch",
            dataset=config.dataset,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=mismatched_path,
        )

    runtime_path = paths["evidence"].with_name("runtime-fingerprint.json")
    runtime_bytes = runtime_path.read_bytes()
    runtime_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReferencePromotionError, match="Runtime fingerprint is invalid"):
        create_review_attestation(
            store,
            role="code",
            reviewer="code-reviewer",
            review_date=date(2026, 10, 13),
            review_reference="review://code/runtime-tamper",
            dataset=config.dataset,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=paths["evidence"],
        )
    with pytest.raises(ReferencePromotionError, match="Runtime fingerprint is invalid"):
        create_promotion_config(
            store,
            name=config.name,
            dataset=config.dataset,
            daily_report_path=paths["daily"],
            experiment_report_path=paths["experiment"],
            evidence_index_path=paths["evidence"],
            code_review_path=paths["code"],
            quantitative_review_path=paths["quant"],
        )
    runtime_path.write_bytes(runtime_bytes)

    promoted_index = results[0].path / "evidence-index.json"
    promoted_index_bytes = promoted_index.read_bytes()
    promoted_index.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError):
        validate_reference_promotion(store, results[0].artifact)
    promoted_index.write_bytes(promoted_index_bytes)

    promoted_runtime = results[1].path / "runtime-fingerprint.json"
    promoted_runtime.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError):
        validate_reference_promotion(store, results[1].artifact)


def test_review_attestation_is_exactly_evidence_bound() -> None:
    dataset = ArtifactSpec.create(
        storage_class=StorageClass.DATASET,
        namespace="widb/internal/name-test",
        configuration={"test": True},
        schema_version="test",
        producer_version="test",
        upstream=(
            ArtifactSpec.create(
                storage_class=StorageClass.RAW,
                namespace="wrds/test",
                configuration={"test": True},
                schema_version="test",
                producer_version="test",
                source_generation="test",
            ).pointer,
        ),
        distribution=DatasetDistribution(DistributionStatus.INTERNAL),
    ).pointer
    experiment = ArtifactSpec.create(
        storage_class=StorageClass.EXPERIMENT,
        namespace="dysanos/test",
        configuration={"test": True},
        schema_version="test",
        producer_version="test",
        upstream=(dataset,),
    ).pointer
    attestation = make_review_attestation(
        role="code",
        reviewer="reviewer",
        review_date=date(2026, 10, 13),
        review_reference="review://test",
        dataset=dataset,
        dataset_manifest_sha256="a" * 64,
        daily_report_id="b" * 64,
        daily_report_sha256="c" * 64,
        experiment=experiment,
        experiment_report_id="d" * 64,
        experiment_report_sha256="e" * 64,
        evidence_index_id="f" * 64,
        evidence_index_sha256="0" * 64,
    )
    assert attestation["outcome"] == "approved"
    assert attestation["dataset"]["artifact_id"] == dataset.artifact_id
    with pytest.raises(ValueError, match="role"):
        make_review_attestation(
            role="wrong",  # type: ignore[arg-type]
            reviewer="reviewer",
            review_date=date(2026, 10, 13),
            review_reference="review://test",
            dataset=dataset,
            dataset_manifest_sha256="a" * 64,
            daily_report_id="b" * 64,
            daily_report_sha256="c" * 64,
            experiment=experiment,
            experiment_report_id="d" * 64,
            experiment_report_sha256="e" * 64,
            evidence_index_id="f" * 64,
            evidence_index_sha256="0" * 64,
        )


def test_promotion_versions_bind_runtime_evidence() -> None:
    for function in (promote_reference_dataset, validate_reference_promotion):
        versioned = cast(Any, function)
        assert "_verified_runtime_evidence" in dict(versioned.version.dependencies[1])
