from __future__ import annotations

import hashlib
import json
import re
import tomllib
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

from sanos.artifacts import ArtifactPointer, StorageClass
from sanos.config import canonical_json, canonical_sha256, canonical_value
from sanos.datasets import PublicProjectionConfig, plan_public_candidate
from sanos.models.dysanos import ExperimentConfig, plan_experiment
from sanos.production import (
    PHASE11_CONFIG_BUNDLE_SCHEMA,
    PHASE11_EVIDENCE_INDEX_SCHEMA,
    Phase11EvidenceError,
    attach_phase11_artifact_binding,
    build_phase11_evidence_index,
    finalize_phase11_downstream_configs,
    validate_phase11_public_candidate_binding,
    write_phase11_downstream_configs,
    write_phase11_evidence_index,
)
from sanos.reference.compare import (
    PHASE11_PRODUCTION_ORACLE_GAPS,
    REFERENCE_REPORT_SCHEMA,
)
from sanos.reference.experiment_compare import EXPERIMENT_REFERENCE_REPORT_SCHEMA
from sanos.runtime import runtime_fingerprint
from sanos.widb.calendar import business_dates

ROOT = Path(__file__).resolve().parents[1]


def _requirement_name(requirement: str) -> str:
    match = re.match(r"^[A-Za-z0-9_.-]+", requirement)
    assert match is not None
    return match.group(0).lower().replace("_", "-")


def test_phase11_tested_constraints_cover_dev_operator_requirements() -> None:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    required = {
        _requirement_name(requirement)
        for group in (
            project["dependencies"],
            project["optional-dependencies"]["dev"],
            project["optional-dependencies"]["operator"],
        )
        for requirement in group
    }
    constraints: dict[str, str] = {}
    for line in (
        ROOT / "constraints" / "phase11-tested-py312.txt"
    ).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert stripped.count("==") == 1
        name, pinned_version = stripped.split("==")
        normalized = _requirement_name(name)
        assert normalized not in constraints
        assert pinned_version
        constraints[normalized] = pinned_version

    assert required <= constraints.keys()
    assert {"clarabel", "ipython", "jaxlib"} <= constraints.keys()


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text("".join(canonical_json(value) + "\n" for value in values), encoding="utf-8")


def _dataset_pointer() -> ArtifactPointer:
    return ArtifactPointer(
        StorageClass.DATASET,
        "widb/internal/name-widb-spx-2022-2025-reference",
        "a" * 64,
    )


def _bundle() -> dict[str, object]:
    return finalize_phase11_downstream_configs(
        _dataset_pointer(),
        public_template=_read(ROOT / "configs" / "sanos-public-candidate-reference.json"),
        experiment_template=_read(ROOT / "configs" / "dysanos-daily-front-reference.json"),
    )


def test_phase11_downstream_config_finalization_is_exact_and_immutable(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    assert bundle["schema"] == PHASE11_CONFIG_BUNDLE_SCHEMA
    assert bundle["source_dataset"] == _dataset_pointer().as_dict()
    assert bundle["bundle_id"] == canonical_sha256(
        {key: value for key, value in bundle.items() if key != "bundle_id"}
    )
    output = tmp_path / "final-configs"
    write_phase11_downstream_configs(output, bundle)
    assert _read(output / "bundle.json") == bundle
    for section, file_name in (
        ("public_candidate", "public-candidate.json"),
        ("experiment", "experiment.json"),
    ):
        value = bundle[section]
        assert isinstance(value, dict)
        assert _read(output / file_name) == value["configuration"]
    with pytest.raises(Phase11EvidenceError, match="already exists"):
        write_phase11_downstream_configs(output, bundle)


def test_phase11_downstream_config_finalization_rejects_policy_edits() -> None:
    public = _read(ROOT / "configs" / "sanos-public-candidate-reference.json")
    section = public["public_dataset"]
    assert isinstance(section, dict)
    section["smoothness"] = 0.5
    with pytest.raises(ValueError, match="frozen Phase 11 profile"):
        finalize_phase11_downstream_configs(
            _dataset_pointer(),
            public_template=public,
            experiment_template=_read(
                ROOT / "configs" / "dysanos-daily-front-reference.json"
            ),
        )


def _evidence_files(tmp_path: Path) -> dict[str, Path]:
    revision = "d" * 40
    dataset = _dataset_pointer()
    bundle = _bundle()
    public_config = PublicProjectionConfig.from_document(
        bundle["public_candidate"]["configuration"]  # type: ignore[index]
    )
    experiment_config = ExperimentConfig.from_document(
        bundle["experiment"]["configuration"]  # type: ignore[index]
    )
    public = plan_public_candidate(public_config).dataset_spec.pointer
    experiment = plan_experiment(experiment_config).spec.pointer
    boundary = [
        {
            "table": table,
            "ticker": "SPX",
            "observation_date": "2021-12-31",
            "roles": ["boundary"],
        }
        for table in ("forwards", "spot", "zero_curve")
    ]
    raw_common = {
        "source": "optionmetrics",
        "source_generation": "default-v1",
        "ticker": "SPX",
        "start_date": "2022-01-03",
        "end_date": "2025-08-29",
        "planned": 3676,
        "network_accessed": False,
    }
    paths = {
        name: tmp_path / name
        for name in (
            "runtime-fingerprint.json",
            "raw-preflight.jsonl",
            "raw-validation.jsonl",
            "dataset-validation.jsonl",
            "daily-report.json",
            "bundle.json",
            "public-validation.jsonl",
            "experiment-validation.jsonl",
            "experiment-report.json",
        )
    }
    _write_json(paths["runtime-fingerprint.json"], runtime_fingerprint())
    _write_jsonl(
        paths["raw-preflight.jsonl"],
        [
            {
                "event": "raw_plan",
                **raw_common,
                "reused": 0,
                "missing": 3676,
                "boundary_partitions": 3,
                "boundary_dependencies": boundary,
                "dry_run": True,
            },
            {
                "event": "raw_result",
                **raw_common,
                "reused": 0,
                "downloaded": 0,
                "missing": 3676,
                "dry_run": True,
            },
        ],
    )
    sessions = [
        value.isoformat()
        for value in business_dates(date(2022, 1, 3), date(2025, 8, 29))
    ]
    table_observations: list[tuple[str, str | None]] = [("security", None)]
    table_observations.extend(("options", value) for value in sessions)
    for table in ("spot", "forwards", "zero_curve"):
        table_observations.append((table, "2021-12-31"))
        table_observations.extend((table, value) for value in sessions)
    assert len(table_observations) == 3676
    raw_records = []
    for index, (table, observation_date) in enumerate(table_observations):
        raw_records.append(
            {
                "event": "raw_artifact_validated",
                "artifact": {
                    "storage_class": "raw",
                    "namespace": "wrds/optionmetrics/test",
                    "artifact_id": hashlib.sha256(str(index).encode()).hexdigest(),
                },
                "logical_table": table,
                "observation_date": observation_date,
                "ticker": "SPX",
                "source_generation": "default-v1",
            }
        )
    _write_jsonl(
        paths["raw-validation.jsonl"],
        [
            {
                "event": "raw_plan",
                **raw_common,
                "reused": 3676,
                "missing": 0,
                "boundary_partitions": 3,
                "boundary_dependencies": boundary,
                "dry_run": False,
            },
            *raw_records,
            {
                "event": "raw_result",
                **raw_common,
                "reused": 3676,
                "downloaded": 0,
                "missing": 0,
                "dry_run": False,
            },
        ],
    )
    _write_jsonl(
        paths["dataset-validation.jsonl"],
        [
            {
                "event": "dataset_validation_result",
                "status": "passed",
                "artifact": dataset,
                "manifest_sha256": "e" * 64,
                "configuration_id": "f" * 64,
                "source_revision": revision,
                "semantic": True,
                "deep_lineage": True,
                "network_accessed": False,
                "filesystem_modified": False,
                "summary": {
                    "requested_dates": 918,
                    "populated_dates": 917,
                    "typed_empty_dates": 1,
                },
            }
        ],
    )
    daily_body = {
        "schema": REFERENCE_REPORT_SCHEMA,
        "outcome": "passed",
        "source_revision": revision,
        "request": {
            "ticker": "SPX",
            "start_date": "2022-01-03",
            "end_date": "2025-08-29",
            "source": "optionmetrics",
            "source_generation": "default-v1",
            "declared_oracle_gaps": list(PHASE11_PRODUCTION_ORACLE_GAPS),
            "enforce_phase11_production": True,
        },
        "classification_counts": {"match": 891, "oracle_gap": 26, "policy_empty": 1},
        "production_contract": {"passed": True},
        "network_accessed": False,
        "artifacts_built": False,
    }
    _write_json(
        paths["daily-report.json"],
        {"report_id": canonical_sha256(daily_body), **daily_body},
    )
    _write_json(paths["bundle.json"], bundle)
    _write_jsonl(
        paths["public-validation.jsonl"],
        [
            {
                "event": "public_dataset_validation_result",
                "status": "passed",
                "distribution": "public_candidate",
                "artifact": public,
                "manifest_sha256": "2" * 64,
                "semantic": True,
                "deep_lineage": True,
                "network_accessed": False,
                "filesystem_modified": False,
                "config": canonical_value(public_config),
            }
        ],
    )
    _write_jsonl(
        paths["experiment-validation.jsonl"],
        [
            {
                "event": "experiment_validation_result",
                "status": "passed",
                "artifact": experiment,
                "semantic": True,
                "deep_lineage": True,
                "network_accessed": False,
                "filesystem_modified": False,
                "config": experiment_config.to_document(),
            }
        ],
    )
    experiment_body = {
        "schema": EXPERIMENT_REFERENCE_REPORT_SCHEMA,
        "outcome": "passed",
        "source_revision": revision,
        "experiment": {
            "artifact": experiment,
            "manifest_sha256": "3" * 64,
            "source_dataset": dataset,
        },
        "phase11_contract": {"passed": True},
        "network_accessed": False,
        "artifacts_built": False,
    }
    _write_json(
        paths["experiment-report.json"],
        {"report_id": canonical_sha256(experiment_body), **experiment_body},
    )
    return paths


def _build_index(paths: dict[str, Path]) -> dict[str, object]:
    index = build_phase11_evidence_index(
        run_id="fresh-reference-20261013",
        source_revision="d" * 40,
        runtime_fingerprint_path=paths["runtime-fingerprint.json"],
        raw_preflight_path=paths["raw-preflight.jsonl"],
        raw_validation_path=paths["raw-validation.jsonl"],
        dataset_validation_path=paths["dataset-validation.jsonl"],
        daily_report_path=paths["daily-report.json"],
        downstream_bundle_path=paths["bundle.json"],
        public_validation_path=paths["public-validation.jsonl"],
        experiment_validation_path=paths["experiment-validation.jsonl"],
        experiment_report_path=paths["experiment-report.json"],
    )
    daily = _read(paths["daily-report.json"])
    experiment = _read(paths["experiment-report.json"])
    experiment_pointer = experiment["experiment"]
    assert isinstance(experiment_pointer, dict)
    experiment_pointer = experiment_pointer["artifact"]
    binding = {
        "status": "passed",
        "dataset": _dataset_pointer().as_dict(),
        "dataset_manifest_sha256": "e" * 64,
        "dataset_configuration_id": "f" * 64,
        "dataset_source_revision": "d" * 40,
        "daily_report_id": daily["report_id"],
        "daily_report_sha256": hashlib.sha256(
            paths["daily-report.json"].read_bytes()
        ).hexdigest(),
        "daily_approvals_sha256": None,
        "partition_binding": {
            "status": "passed",
            "partition_count": 918,
            "bound_fitted_artifacts": 918,
            "binding_sha256": "2" * 64,
        },
        "public_candidate_binding": {
            "status": "passed",
            "public_candidate": {
                key: index["artifacts"]["public_candidate"][key]  # type: ignore[index]
                for key in ("storage_class", "namespace", "artifact_id")
            },
            "public_candidate_manifest_sha256": "2" * 64,
        },
        "experiment": experiment_pointer,
        "experiment_manifest_sha256": "3" * 64,
        "experiment_report_id": experiment["report_id"],
        "experiment_report_sha256": hashlib.sha256(
            paths["experiment-report.json"].read_bytes()
        ).hexdigest(),
    }
    return attach_phase11_artifact_binding(index, binding)


def test_phase11_evidence_index_binds_complete_offline_chain(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    index = _build_index(paths)
    assert index["schema"] == PHASE11_EVIDENCE_INDEX_SCHEMA
    assert index["resolved_versions"]["runtime_fingerprint"] == (  # type: ignore[index]
        runtime_fingerprint()["fingerprint_id"]
    )
    assert index["artifacts"] == {
        "internal_dataset": {
            **_dataset_pointer().as_dict(),
            "manifest_sha256": "e" * 64,
            "configuration_id": "f" * 64,
            "source_revision": "d" * 40,
        },
        "public_candidate": {
            **plan_public_candidate(
                PublicProjectionConfig.from_document(
                    _bundle()["public_candidate"]["configuration"]  # type: ignore[index]
                )
            ).dataset_spec.pointer.as_dict(),
            "manifest_sha256": "2" * 64,
        },
        "experiment": {
            **plan_experiment(
                ExperimentConfig.from_document(
                    _bundle()["experiment"]["configuration"]  # type: ignore[index]
                )
            ).spec.pointer.as_dict(),
            "manifest_sha256": "3" * 64,
        },
    }
    output = tmp_path / "evidence-index.json"
    assert write_phase11_evidence_index(output, index) is True
    assert write_phase11_evidence_index(output, index) is False
    changed = {**index, "run_id": "other"}
    with pytest.raises(Phase11EvidenceError, match="other content"):
        write_phase11_evidence_index(output, changed)

    versioned = cast(Any, build_phase11_evidence_index)
    dependencies = set(dict(versioned.version.dependencies[1]))
    assert {"canonical_json", "load_runtime_fingerprint", "runtime_fingerprint"} <= (
        dependencies
    )
    assert str(versioned.version)


def test_phase11_public_candidate_binding_rejects_stale_manifest(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    index = build_phase11_evidence_index(
        run_id="fresh-reference-20261013",
        source_revision="d" * 40,
        runtime_fingerprint_path=paths["runtime-fingerprint.json"],
        raw_preflight_path=paths["raw-preflight.jsonl"],
        raw_validation_path=paths["raw-validation.jsonl"],
        dataset_validation_path=paths["dataset-validation.jsonl"],
        daily_report_path=paths["daily-report.json"],
        downstream_bundle_path=paths["bundle.json"],
        public_validation_path=paths["public-validation.jsonl"],
        experiment_validation_path=paths["experiment-validation.jsonl"],
        experiment_report_path=paths["experiment-report.json"],
    )
    public = index["artifacts"]["public_candidate"]  # type: ignore[index]
    pointer = ArtifactPointer(
        StorageClass.DATASET,
        public["namespace"],  # type: ignore[index]
        public["artifact_id"],  # type: ignore[index]
    )
    assert validate_phase11_public_candidate_binding(
        index,
        public_candidate=pointer,
        manifest_sha256="2" * 64,
    )["status"] == "passed"
    with pytest.raises(Phase11EvidenceError, match="differs"):
        validate_phase11_public_candidate_binding(
            index,
            public_candidate=pointer,
            manifest_sha256="7" * 64,
        )


def test_phase11_evidence_rejects_free_form_new_oracle_gap(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    report = _read(paths["daily-report.json"])
    changed = deepcopy(report)
    request = changed["request"]
    assert isinstance(request, dict)
    request["declared_oracle_gaps"] = [
        *PHASE11_PRODUCTION_ORACLE_GAPS,
        "2025-08-28",
    ]
    body = {key: value for key, value in changed.items() if key != "report_id"}
    changed["report_id"] = canonical_sha256(body)
    _write_json(paths["daily-report.json"], changed)
    with pytest.raises(Phase11EvidenceError, match="Daily comparison request"):
        _build_index(paths)


def test_phase11_evidence_rejects_raw_reuse_in_fresh_preflight(tmp_path: Path) -> None:
    paths = _evidence_files(tmp_path)
    events = [
        json.loads(line)
        for line in paths["raw-preflight.jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    events[0]["reused"] = 1
    events[0]["missing"] = 3675
    _write_jsonl(paths["raw-preflight.jsonl"], events)
    with pytest.raises(Phase11EvidenceError, match="Raw preflight plan"):
        _build_index(paths)


def test_phase11_evidence_rejects_missing_malformed_and_stale_runtime(
    tmp_path: Path,
) -> None:
    paths = _evidence_files(tmp_path)
    runtime_path = paths["runtime-fingerprint.json"]
    runtime_path.unlink()
    with pytest.raises(Phase11EvidenceError, match="Runtime fingerprint is invalid"):
        _build_index(paths)

    _write_json(runtime_path, {})
    with pytest.raises(Phase11EvidenceError, match="Runtime fingerprint is invalid"):
        _build_index(paths)

    stale = runtime_fingerprint()
    stale["python"]["compiler"] = "stale-compiler"  # type: ignore[index]
    stale_body = {key: value for key, value in stale.items() if key != "fingerprint_id"}
    stale["fingerprint_id"] = canonical_sha256(stale_body)
    _write_json(runtime_path, stale)
    with pytest.raises(Phase11EvidenceError, match="differs from the active"):
        _build_index(paths)

    stale["python"]["compiler"] = "tampered-compiler"  # type: ignore[index]
    _write_json(runtime_path, stale)
    with pytest.raises(Phase11EvidenceError, match="Runtime fingerprint is invalid"):
        _build_index(paths)
