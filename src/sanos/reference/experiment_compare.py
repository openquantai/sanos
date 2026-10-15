"""Read-only parity comparison for the full DySANOS Experiment materialization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np
from cdxcore.version import version

from sanos.artifacts import ArtifactPointer, ArtifactStore, ValidatedArtifact
from sanos.config import StorageLocations, canonical_sha256, canonical_value
from sanos.models.dysanos import ExperimentConfig, validate_experiment
from sanos.reference._experiment_oracle_worker import (
    PINNED_SOURCE_SHA256,
    experiment_oracle_worker_version,
)
from sanos.reference.compare import (
    PHASE11_PRODUCTION_ORACLE_GAPS,
    PINNED_CATALOG_SHA256,
    PINNED_COLLECTION_SHA256,
    ReferenceComparisonError,
    load_reference_catalog,
    write_comparison_report,
)

EXPERIMENT_REFERENCE_REPORT_SCHEMA = "sanos-legacy-experiment-comparison-v1"
PHASE11_EXPERIMENT_ARTIFACT_ID = (
    "b920d2c75975d1bdecb8c16a4e0f6eb4f9b9030c6353be446215b9d9c518e5b5"
)
PHASE11_DATASET_ARTIFACT_ID = (
    "d09a55ee4806916f0fc355f56c76401e9e21de09cfb6de99d5630e59439f1fc1"
)
PHASE11_DATASET_NAMESPACE = (
    "sanos/public-candidate/name-sanos-spx-2022-2025-public-candidate"
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT_VALIDATION_VERSION = str(validate_experiment.version)
_EXPERIMENT_VALIDATION_VERSION_SHA256 = hashlib.sha256(
    _EXPERIMENT_VALIDATION_VERSION.encode("utf-8")
).hexdigest()


@version("phase11-legacy-experiment-tolerances-v1")
@dataclass(frozen=True, slots=True)
class ExperimentReferenceTolerances:
    """Frozen logical and consumer-tensor tolerances for migration parity."""

    float64_atol: float = 1.0e-12
    float32_atol: float = 0.0

    def __post_init__(self) -> None:
        expected = {"float64_atol": 1.0e-12, "float32_atol": 0.0}
        for name, required in expected.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
                or float(value) != required
            ):
                raise ValueError(f"{name} is frozen at {required!r}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@version("1.0", dependencies=[ExperimentReferenceTolerances])
@dataclass(frozen=True, slots=True)
class ExperimentComparisonRequest:
    """Exact existing Experiment and pinned legacy-oracle comparison request."""

    experiment: ArtifactPointer
    enforce_phase11_production: bool = False
    tolerances: ExperimentReferenceTolerances = ExperimentReferenceTolerances()
    consumer_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.experiment.storage_class.value != "experiment":
            raise ValueError("Experiment comparison requires an Experiment pointer")
        if not isinstance(self.enforce_phase11_production, bool):
            raise TypeError("enforce_phase11_production must be boolean")
        if (
            isinstance(self.consumer_batch_size, bool)
            or not isinstance(self.consumer_batch_size, int)
            or not 1 <= self.consumer_batch_size <= 256
        ):
            raise ValueError("consumer_batch_size must lie in [1,256]")


@version("1.0")
def _experiment_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@version("1.0", dependencies=[ExperimentConfig])
def _manifest_config(artifact: ValidatedArtifact) -> ExperimentConfig:
    value = artifact.manifest.get("configuration")
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "experiment"}
        or not isinstance(value.get("experiment"), Mapping)
    ):
        raise ReferenceComparisonError("Experiment manifest configuration is malformed")
    try:
        return ExperimentConfig.from_document(cast(Mapping[str, Any], value["experiment"]))
    except (TypeError, ValueError) as exc:
        raise ReferenceComparisonError(
            f"Experiment manifest configuration is invalid: {exc}"
        ) from exc


ProgressCallback = Callable[[Mapping[str, Any]], None]


@version("1.0")
def _worker_environment(legacy_code_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2]
    worker_root = Path(os.getenv("TMPDIR", "/tmp")) / "sanos" / (
        f"legacy-oracle-env-{os.getpid()}"
    )
    existing = environment.get("PYTHONPATH")
    paths = [str(source_root), str(legacy_code_root)]
    if existing:
        paths.append(existing)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WRDS_OFF": "1",
            "QUANTDATA_TEST_MODE": "0",
            "QUANTDATA_CACHE_MODE": "off",
            "QUANTDATA_RAW_DATA": str(worker_root / "raw"),
            "QUANTDATA_PROCESSING_DATA": str(worker_root / "intermediate"),
            "QUANTDATA_TRAINING_DATA": str(worker_root / "datasets"),
            "QUANTDATA_EXPERIMENTS": str(worker_root / "experiments"),
            "QUANTDATA_TEMP_DATA": str(worker_root / "temp"),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment


@version("1.0")
def _terminate_oracle_worker(process: subprocess.Popen[str]) -> None:
    """Stop and reap an owned oracle worker after any parent-side failure."""

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for stream in (process.stdin, process.stdout):
        if stream is not None and not stream.closed:
            stream.close()


@version(
    "1.2",
    dependencies=[
        _terminate_oracle_worker,
        _worker_environment,
        experiment_oracle_worker_version,
    ],
)
def _run_oracle_worker(
    request: Mapping[str, Any],
    *,
    legacy_code_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "sanos.reference._experiment_oracle_worker"]
    process = subprocess.Popen(  # noqa: S603 - exact interpreter/module, no shell
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[3],
        env=_worker_environment(legacy_code_root),
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(canonical_value(request), separators=(",", ":")))
        process.stdin.close()
        result: dict[str, Any] | None = None
        worker_error: str | None = None
        diagnostics: list[str] = []
        for line in process.stdout:
            value = line.strip()
            if not value:
                continue
            try:
                message = json.loads(value)
            except json.JSONDecodeError:
                diagnostics.append(value)
                continue
            if not isinstance(message, Mapping):
                diagnostics.append(value)
                continue
            kind = message.get("kind")
            if kind == "progress":
                if progress is not None:
                    progress(
                        cast(
                            dict[str, Any],
                            canonical_value(
                                {
                                    key: item
                                    for key, item in message.items()
                                    if key != "kind"
                                }
                            ),
                        )
                    )
            elif kind == "result" and isinstance(message.get("result"), Mapping):
                result = cast(dict[str, Any], message["result"])
            elif kind == "error":
                worker_error = f"{message.get('error_type')}: {message.get('message')}"
            else:
                diagnostics.append(value)
        return_code = process.wait()
        if return_code != 0 or result is None:
            details = worker_error or "pinned worker returned no result"
            if diagnostics:
                details += "; diagnostics: " + " | ".join(diagnostics[-5:])
            raise ReferenceComparisonError(f"Experiment oracle worker failed: {details}")
        return result
    except BaseException:
        _terminate_oracle_worker(process)
        raise
    finally:
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()


@version("1.1")
def _phase11_contract(
    request: ExperimentComparisonRequest,
    *,
    artifact: ValidatedArtifact,
    config: ExperimentConfig,
    validation: Mapping[str, Any],
    worker: Mapping[str, Any],
    catalog_sha256: str,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    summary = worker.get("summary")
    if not isinstance(summary, Mapping):
        raise ReferenceComparisonError("Experiment oracle worker summary is malformed")
    expected_source = ArtifactPointer(
        "dataset", PHASE11_DATASET_NAMESPACE, PHASE11_DATASET_ARTIFACT_ID
    )
    if artifact.pointer.artifact_id != PHASE11_EXPERIMENT_ARTIFACT_ID:
        failures.append("Experiment artifact identity differs from the frozen Phase 11 plan")
    if config.source_dataset.pointer != expected_source:
        failures.append("Experiment source Dataset differs from the frozen Phase 11 plan")
    if catalog_sha256 != PINNED_CATALOG_SHA256:
        failures.append("legacy catalog checksum differs from the pinned oracle")
    if catalog.get("collection_sha256") != PINNED_COLLECTION_SHA256:
        failures.append("legacy collection identity differs from the pinned oracle")
    if worker.get("source_hashes") != PINNED_SOURCE_SHA256:
        failures.append("legacy consumer source closure differs from the pinned oracle")
    runtime = worker.get("runtime")
    if not isinstance(runtime, Mapping) or any(
        not isinstance(runtime.get(name), str) or not str(runtime[name]).strip()
        for name in ("python", "numpy", "torch")
    ):
        failures.append("legacy consumer runtime identities are missing")
    expected_summary = {
        "fresh_dates": 917,
        "common_dates": 891,
        "legacy_books": 3_566_258,
        "legacy_calls": 1_783_129,
        "legacy_puts": 1_783_129,
        "legacy_qmax": 7_898,
        "legacy_pmax": 3_949,
        "legacy_emax": 36,
        "legacy_pre_2025_dates": 729,
        "legacy_from_2025_dates": 162,
        "legacy_weekly_pairs": 861,
        "consumer_batch_mismatches": 0,
    }
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            failures.append(f"{name}={summary.get(name)!r}, expected {expected!r}")
    fresh_only = tuple(summary.get("fresh_only_dates", ()))
    if fresh_only != PHASE11_PRODUCTION_ORACLE_GAPS:
        failures.append("fresh-only Experiment dates differ from the exact 26 oracle gaps")
    mismatch_dates = summary.get("mismatch_dates")
    if mismatch_dates != []:
        failures.append(f"common-period Experiment mismatches remain: {mismatch_dates!r}")
    validation_dates = validation.get("dates")
    if (
        not isinstance(validation_dates, Mapping)
        or validation_dates.get("count") != 917
        or validation_dates.get("first") != "2022-01-03"
        or validation_dates.get("last") != "2025-08-29"
    ):
        failures.append("semantic Experiment validation does not report 917 dates")
    day_records = cast(list[dict[str, Any]], worker.get("dates", []))
    expected_common_dates = [
        record["trading_date"]
        for record in cast(list[dict[str, Any]], catalog.get("files", []))
        if record.get("populated") is True
    ]
    if [record.get("date") for record in day_records] != expected_common_dates:
        failures.append("per-day Experiment evidence does not cover the ordered 891 dates")
    if any(record.get("classification") != "match" for record in day_records):
        failures.append("per-day Experiment evidence contains a non-match classification")
    populations = [
        cast(Mapping[str, Any], cast(Mapping[str, Any], day["logical"])["population"])
        for day in day_records
    ]
    common_books = sum(int(population["quotes"]) for population in populations)
    if common_books != summary.get("legacy_books"):
        failures.append("per-day common book counts differ from the aggregate")
    population_checks = {
        "legacy_calls": sum(int(item["calls"]) for item in populations),
        "legacy_puts": sum(int(item["puts"]) for item in populations),
        "legacy_qmax": max((int(item["quotes"]) for item in populations), default=0),
        "legacy_pmax": max((int(item["prices"]) for item in populations), default=0),
        "legacy_emax": max((int(item["expiries"]) for item in populations), default=0),
    }
    for name, observed in population_checks.items():
        if summary.get(name) != observed:
            failures.append(f"per-day {name} differs from its aggregate")
    batches = worker.get("consumer_batches")
    if (
        not isinstance(batches, list)
        or len(batches) != ceil(891 / request.consumer_batch_size)
        or any(
            not isinstance(item, Mapping) or item.get("matched") is not True
            for item in batches
        )
    ):
        failures.append("consumer batch evidence is incomplete or contains a mismatch")
    validation_books = validation.get("books")
    if (
        isinstance(validation_books, bool)
        or not isinstance(validation_books, int)
        or validation_books <= common_books
    ):
        failures.append(
            "fresh Experiment must contain additional books on its 26 fresh-only dates"
        )
    if failures:
        return {"enabled": True, "passed": False, "failures": failures}
    return {
        "enabled": True,
        "passed": True,
        "expected_experiment": artifact.pointer,
        "expected_source_dataset": expected_source,
        "expected_common_dates": 891,
        "expected_fresh_only_dates": PHASE11_PRODUCTION_ORACLE_GAPS,
        "legacy_population": expected_summary,
    }


@version(
    f"1.1-validation-{_EXPERIMENT_VALIDATION_VERSION_SHA256}",
    dependencies=[
        ArtifactStore,
        ExperimentComparisonRequest,
        ExperimentConfig,
        ExperimentReferenceTolerances,
        _manifest_config,
        _phase11_contract,
        _run_oracle_worker,
        _experiment_sha256_file,
        load_reference_catalog,
    ],
)
def compare_experiment_artifact(
    locations: StorageLocations,
    request: ExperimentComparisonRequest,
    *,
    legacy_root: Path,
    catalog_path: Path,
    legacy_code_root: Path,
    source_revision: str,
    progress: ProgressCallback | None = None,
    progress_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Compare a complete fresh HDF5 Experiment with all 891 oracle days."""

    if _HEX_40.fullmatch(source_revision) is None:
        raise ReferenceComparisonError("source_revision must be exact lowercase 40-hex")
    if not 0.0 < progress_interval_seconds <= 300.0:
        raise ValueError("progress_interval_seconds must lie in (0,300]")
    for label, path in (
        ("legacy root", legacy_root),
        ("legacy code root", legacy_code_root),
    ):
        if path.is_symlink() or not path.is_dir():
            raise ReferenceComparisonError(f"{label} is missing or unsafe: {path}")
    catalog = load_reference_catalog(catalog_path)
    catalog_sha256 = _experiment_sha256_file(catalog_path)
    store = ArtifactStore(locations)
    artifact = store.validate(request.experiment)
    config = _manifest_config(artifact)
    validation_summary = validate_experiment(
        store,
        artifact,
        semantic=True,
        deep_lineage=False,
        progress=progress,
    )
    if validation_summary is None:
        raise ReferenceComparisonError("Experiment semantic validation returned no summary")
    validation = validation_summary.as_dict()
    worker_request = {
        "experiment_hdf": str(artifact.path / "data.h5"),
        "experiment_config": config.to_document(),
        "ticker": "SPX",
        "legacy_root": str(legacy_root),
        "catalog_path": str(catalog_path),
        "legacy_code_root": str(legacy_code_root),
        "float64_atol": request.tolerances.float64_atol,
        "float32_atol": request.tolerances.float32_atol,
        "consumer_batch_size": request.consumer_batch_size,
        "progress_interval_seconds": progress_interval_seconds,
    }
    worker = _run_oracle_worker(
        worker_request,
        legacy_code_root=legacy_code_root,
        progress=progress,
    )
    contract = (
        _phase11_contract(
            request,
            artifact=artifact,
            config=config,
            validation=validation,
            worker=worker,
            catalog_sha256=catalog_sha256,
            catalog=catalog,
        )
        if request.enforce_phase11_production
        else {"enabled": False, "passed": True}
    )
    summary = cast(Mapping[str, Any], worker["summary"])
    outcome = (
        "passed"
        if not summary.get("mismatch_dates")
        and summary.get("consumer_batch_mismatches") == 0
        and contract["passed"]
        else "blocked"
    )
    manifest_path = artifact.path / "manifest.json"
    report_without_id = cast(
        dict[str, Any],
        canonical_value(
            {
                "schema": EXPERIMENT_REFERENCE_REPORT_SCHEMA,
                "outcome": outcome,
                "source_revision": source_revision,
                "request": {
                    "experiment": request.experiment,
                    "enforce_phase11_production": request.enforce_phase11_production,
                    "consumer_batch_size": request.consumer_batch_size,
                },
                "tolerances": request.tolerances.to_dict(),
                "resolved_versions": {
                    "comparison": str(compare_experiment_artifact.version),
                    "worker": experiment_oracle_worker_version(),
                    "experiment_validation": _EXPERIMENT_VALIDATION_VERSION,
                    "request": str(ExperimentComparisonRequest.version),
                    "tolerances": str(ExperimentReferenceTolerances.version),
                },
                "experiment": {
                    "artifact": artifact.pointer,
                    "path": str(artifact.path),
                    "manifest_sha256": _experiment_sha256_file(manifest_path),
                    "configuration_id": artifact.manifest.get("configuration_id"),
                    "producer_version": artifact.manifest.get("producer_version"),
                    "source_dataset": config.source_dataset.pointer,
                    "semantic_validation": validation,
                },
                "legacy": {
                    "root": str(legacy_root),
                    "code_root": str(legacy_code_root),
                    "catalog": str(catalog_path),
                    "catalog_sha256": catalog_sha256,
                    "collection_sha256": catalog.get("collection_sha256"),
                    "source_revision": catalog.get("source_revision"),
                    "pinned_source_sha256": worker.get("source_hashes"),
                    "runtime": worker.get("runtime"),
                },
                "phase11_contract": contract,
                "comparison": worker,
                "network_accessed": False,
                "artifacts_built": False,
            }
        ),
    )
    return {"report_id": canonical_sha256(report_without_id), **report_without_id}


@version("1.0", dependencies=[write_comparison_report])
def write_experiment_comparison_report(
    output: Path,
    report: Mapping[str, Any],
    *,
    legacy_root: Path,
) -> None:
    """Publish immutable Experiment parity evidence outside Git and oracle roots."""

    write_comparison_report(output, report, legacy_root=legacy_root)


__all__ = [
    "EXPERIMENT_REFERENCE_REPORT_SCHEMA",
    "ExperimentComparisonRequest",
    "ExperimentReferenceTolerances",
    "PHASE11_DATASET_ARTIFACT_ID",
    "PHASE11_DATASET_NAMESPACE",
    "PHASE11_EXPERIMENT_ARTIFACT_ID",
    "compare_experiment_artifact",
    "write_experiment_comparison_report",
]
