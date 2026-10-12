"""Output-first construction and validation of immutable Experiment artifacts."""

from __future__ import annotations

import json
import multiprocessing
import shutil
import time
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import polars as pl
from cdxcore.version import version
from filelock import FileLock

from sanos.artifacts import (
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    DistributionStatus,
    ReuseResult,
    StorageClass,
    ValidatedArtifact,
    atomic_write_json,
)
from sanos.config import StorageLocations, canonical_json, canonical_sha256, canonical_value
from sanos.datasets import (
    DatasetValidationSummary,
    open_internal_dataset,
    validate_internal_dataset,
)
from sanos.datasets.internal import INTERNAL_DATASET_SCHEMA_VERSION
from sanos.experiments.config import ExperimentBuildOptions, ExperimentConfig
from sanos.experiments.daily_front import (
    daily_quote_objective,
    project_daily_front,
    recompute_market_fingerprint,
)
from sanos.experiments.hdf5 import (
    EXPERIMENT_HDF5_NAME,
    EXPERIMENT_HDF5_SCHEMA,
    DailyFrontExperiment,
    ExperimentValidationError,
    experiment_hdf5_schema_document,
    read_day_shard,
    validate_experiment_hdf5,
    write_day_shard,
    write_experiment_hdf5,
)
from sanos.pipeline import RunJournal
from sanos.widb.bad import bad_day_reason
from sanos.widb.surface import FITTED_SCHEMA

EXPERIMENT_CONFIG_SCHEMA = "sanos-experiment-config-v1"
EXPERIMENT_SOURCE_SCHEMA = "sanos-experiment-source-v1"
EXPERIMENT_SCHEMA_DOCUMENT = "sanos-dysanos-daily-front-schema-v1"
EXPERIMENT_SPLITS_SCHEMA = "sanos-experiment-splits-v1"
EXPERIMENT_VALIDATION_SCHEMA = "sanos-experiment-validation-v1"

ProgressCallback = Callable[[Mapping[str, Any]], None]


@version("1.0")
@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    config: ExperimentConfig
    spec: ArtifactSpec
    plan_id: str


@version("1.0")
@dataclass(frozen=True, slots=True)
class ExperimentValidationSummary:
    dates: int
    books: int
    calls: int
    puts: int
    expiries_max: int
    quotes_max: int
    price_locations_max: int
    model_nodes: int
    weekly_pairs: int
    first_date: date
    last_date: date
    logical_content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], canonical_value(self))


@version(
    "1.0",
    dependencies=[ExperimentPlan, ExperimentValidationSummary, ReuseResult],
)
@dataclass(frozen=True, slots=True)
class ExperimentBuildResult:
    plan: ExperimentPlan
    artifact: ArtifactPointer
    path: Path
    dry_run: bool
    built: bool
    reuse: ReuseResult | None
    validation: ExperimentValidationSummary | None
    source_accessed: bool
    workspace: Path
    journal_path: Path


def _emit(progress: ProgressCallback | None, event: str, **fields: Any) -> None:
    if progress is not None:
        progress(cast(dict[str, Any], canonical_value({"event": event, **fields})))


@version("1.0", dependencies=[ExperimentConfig])
def _parse_manifest_config(value: object) -> ExperimentConfig:
    if not isinstance(value, Mapping) or set(value) != {"schema", "experiment"}:
        raise ExperimentValidationError("Experiment manifest configuration changed")
    if value["schema"] != EXPERIMENT_CONFIG_SCHEMA or not isinstance(value["experiment"], Mapping):
        raise ExperimentValidationError("Unsupported Experiment configuration schema")
    return ExperimentConfig.from_document(cast(Mapping[str, Any], value["experiment"]))


@version(
    "1.0",
    dependencies=[ExperimentConfig, validate_internal_dataset, open_internal_dataset],
)
def _source_gate(
    store: ArtifactStore,
    config: ExperimentConfig,
    *,
    progress: ProgressCallback | None,
    progress_interval_seconds: float,
) -> tuple[ValidatedArtifact, DatasetValidationSummary]:
    source = store.validate(config.source_dataset.pointer)
    if source.pointer.storage_class is not StorageClass.DATASET:
        raise ExperimentValidationError("Reference Experiment source is not a Dataset")
    distribution = source.manifest.get("distribution")
    if not isinstance(distribution, Mapping) or distribution.get("status") != (
        DistributionStatus.INTERNAL.value
    ):
        raise ExperimentValidationError(
            "dysanos-daily-front-market-v1 requires an internal Dataset source"
        )
    source_metadata = source.manifest.get("metadata")
    if not isinstance(source_metadata, Mapping) or source_metadata.get(
        "dataset_schema_version"
    ) != (INTERNAL_DATASET_SCHEMA_VERSION):
        raise ExperimentValidationError(
            "Reference Experiment source has an unsupported internal Dataset schema"
        )
    summary = validate_internal_dataset(
        store,
        source,
        semantic=True,
        deep_lineage=False,
        progress=progress,
        progress_interval_seconds=progress_interval_seconds,
    )
    if summary is None:
        raise ExperimentValidationError("Internal Dataset semantic validation produced no result")
    dataset = open_internal_dataset(store, source)
    if dataset.config.daily.clock.overnight_variance_fraction != (
        config.market.overnight_variance_fraction
    ):
        raise ExperimentValidationError(
            "Experiment overnight_variance_fraction differs from its source Dataset"
        )
    if dataset.config.surface.smooth_sanos.smoothness != config.market.smoothness:
        raise ExperimentValidationError("Experiment smoothness differs from its source Dataset")
    if dataset.config.surface.options_filter.max_num_options != config.market.max_num_options:
        raise ExperimentValidationError(
            "Experiment max_num_options differs from the source Dataset population"
        )
    return source, summary


@version("1.0")
def _shard_path(workspace: Path, trading_date: date) -> Path:
    return workspace / "shards" / f"{trading_date.isoformat()}.npz"


@version(
    "1.0",
    dependencies=[
        ExperimentConfig,
        project_daily_front,
        read_day_shard,
        write_day_shard,
    ],
)
def _project_partition(
    partition: Path,
    shard: Path,
    *,
    source: ArtifactPointer,
    config: ExperimentConfig,
) -> Path:
    frame = pl.read_parquet(partition, hive_partitioning=False)
    if frame.schema != pl.Schema(FITTED_SCHEMA):
        raise ExperimentValidationError(f"Source partition schema changed: {partition}")
    day = project_daily_front(
        frame,
        source_dataset=source,
        market=config.market,
        geometry=config.geometry,
    )
    write_day_shard(shard, day)
    read_day_shard(shard)
    return shard


@version(
    "1.0",
    dependencies=[
        ExperimentBuildOptions,
        ExperimentConfig,
        ExperimentPlan,
        _project_partition,
        _shard_path,
        bad_day_reason,
        read_day_shard,
        recompute_market_fingerprint,
    ],
)
def _prepare_shards(
    dataset: Any,
    *,
    plan: ExperimentPlan,
    workspace: Path,
    config: ExperimentConfig,
    options: ExperimentBuildOptions,
    progress: ProgressCallback | None,
    journal: RunJournal,
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
    requested: list[tuple[date, Path, Path]] = []
    exclusions: list[dict[str, Any]] = []
    for record in dataset.partitions:
        trading_date = date.fromisoformat(str(record["trading_date"]))
        source_path = dataset.artifact.path / str(record["path"])
        if record.get("typed_empty") is True:
            reason = record.get("typed_empty_reason")
            expected = bad_day_reason(dataset.config.ticker, trading_date)
            if not isinstance(reason, str) or expected != reason:
                raise ExperimentValidationError(
                    f"Unexpected typed-empty source partition {trading_date}: {reason!r}"
                )
            exclusions.append({"trading_date": trading_date, "reason": reason})
            continue
        requested.append((trading_date, source_path, _shard_path(workspace, trading_date)))
    if not requested:
        raise ExperimentValidationError("Source Dataset has no populated partitions")
    workspace.mkdir(parents=True, exist_ok=True)
    if workspace.is_symlink() or not workspace.is_dir():
        raise ExperimentValidationError(f"Unsafe Experiment workspace: {workspace}")
    shards_directory = workspace / "shards"
    shards_directory.mkdir(exist_ok=True)
    if shards_directory.is_symlink() or not shards_directory.is_dir():
        raise ExperimentValidationError(f"Unsafe Experiment shard directory: {shards_directory}")
    plan_path = workspace / "plan.json"
    plan_value = {
        "schema": "sanos-experiment-workspace-v1",
        "plan_id": plan.plan_id,
        "artifact": plan.spec.pointer,
        "dates": [item[0] for item in requested],
    }
    if plan_path.is_symlink():
        raise ExperimentValidationError(f"Unsafe Experiment workspace plan: {plan_path}")
    if plan_path.exists():
        try:
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExperimentValidationError(f"Cannot resume malformed workspace: {exc}") from exc
        if canonical_json(existing) != canonical_json(plan_value):
            raise ExperimentValidationError("Experiment workspace belongs to another plan")
    else:
        atomic_write_json(plan_path, plan_value)

    pending: list[tuple[date, Path, Path]] = []
    reused = 0
    for trading_date, source_path, shard in requested:
        if shard.exists() or shard.is_symlink():
            try:
                day = read_day_shard(shard)
                if day.trading_date != trading_date:
                    raise ExperimentValidationError("Shard date changed")
                if day.market_fingerprint != recompute_market_fingerprint(
                    day, config.source_dataset.pointer
                ):
                    raise ExperimentValidationError("Shard market fingerprint changed")
            except ExperimentValidationError:
                shard.unlink()
            else:
                reused += 1
                journal.emit(
                    "unit_reused",
                    key=trading_date.isoformat(),
                    shard=shard.name,
                )
                continue
        pending.append((trading_date, source_path, shard))
    total = len(requested)
    completed = reused
    started = time.monotonic()
    _emit(
        progress,
        "experiment_preparation_started",
        completed=completed,
        total=total,
        reused=reused,
        failures=0,
    )
    if pending:
        executor: ProcessPoolExecutor | None = None
        futures: dict[Future[Path], date] = {}
        try:
            executor = ProcessPoolExecutor(
                max_workers=min(options.max_workers, len(pending)),
                mp_context=multiprocessing.get_context("spawn"),
            )
            for trading_date, source_path, shard in pending:
                future = executor.submit(
                    _project_partition,
                    source_path,
                    shard,
                    source=config.source_dataset.pointer,
                    config=config,
                )
                futures[future] = trading_date
            remaining = set(futures)
            while remaining:
                done, remaining = wait(
                    remaining,
                    timeout=options.progress_interval_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    completed_path = future.result()
                    completed += 1
                    journal.emit(
                        "unit_completed",
                        key=futures[future].isoformat(),
                        shard=completed_path.name,
                    )
                elapsed = time.monotonic() - started
                rate = (completed - reused) / elapsed if elapsed > 0.0 else 0.0
                eta = (total - completed) / rate if rate > 0.0 else None
                _emit(
                    progress,
                    "experiment_preparation_progress",
                    completed=completed,
                    total=total,
                    reused=reused,
                    failures=0,
                    elapsed_seconds=round(elapsed, 6),
                    eta_seconds=None if eta is None else round(eta, 6),
                )
        except BaseException:
            for future in futures:
                future.cancel()
            if executor is not None:
                processes = tuple(getattr(executor, "_processes", {}).values())
                _, unfinished = wait(tuple(futures), timeout=1.0)
                if unfinished:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes:
                        process.join(timeout=1.0)
                    for process in processes:
                        if process.is_alive():
                            process.kill()
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        assert executor is not None
        executor.shutdown(wait=True)
    shards = tuple(item[2] for item in requested)
    for shard in shards:
        read_day_shard(shard)
    _emit(
        progress,
        "experiment_preparation_completed",
        completed=total,
        total=total,
        reused=reused,
        failures=0,
        elapsed_seconds=round(time.monotonic() - started, 6),
    )
    return shards, tuple(exclusions)


def _restart_workspace(workspace: Path, experiments_root: Path) -> None:
    """Remove one exact plan-owned restart workspace after explicit request."""

    root = experiments_root.resolve(strict=False)
    target = workspace.resolve(strict=False)
    if target.parent != root / ".sanos-runs" or target.name != workspace.name:
        raise ExperimentValidationError(f"Refusing unsafe Experiment workspace removal: {target}")
    if workspace.is_symlink():
        raise ExperimentValidationError(f"Refusing symlink Experiment workspace: {workspace}")
    if workspace.exists():
        if not workspace.is_dir():
            raise ExperimentValidationError(f"Experiment workspace is not a directory: {workspace}")
        shutil.rmtree(workspace)


@version("2.0", dependencies=[experiment_hdf5_schema_document])
def _experiment_schema_document() -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_SCHEMA_DOCUMENT,
        "hdf5_schema": EXPERIMENT_HDF5_SCHEMA,
        "profile": "dysanos-daily-front-market-v1",
        "layout": {
            "day": "one row per populated retained trading date",
            "quote": "global-max padded separate call/put books with prefix mask",
            "price": "first-occurrence unique (expiry_index,K/F) locations",
            "expiry": "retained market expiry geometry",
            "model": "fixed Direct-200 interpreting geometry",
        },
        "units": {
            "strike": "K/F",
            "prices": "option cash value divided by DF*F",
            "time": "NYSE business sessions divided by 252",
            "spot_ref": "absolute source spot; internal-only",
        },
        "padding": "global_max_with_masks_v1; masks/counts are authoritative",
        "hdf5": experiment_hdf5_schema_document(),
    }


@version("1.0")
def _experiment_readme(
    config: ExperimentConfig,
    summary: ExperimentValidationSummary,
) -> str:
    return f"""# SANOS internal daily-front Experiment

This immutable, disposable HDF5 materialization adapts one completed internal
SANOS Dataset to the recovered DySANOS daily-front `FlexMarketDay` input
contract. It contains `{summary.dates}` dates and `{summary.books}` separate
call/put books. Prices are normalized by discounted forward and strikes are
`K/F`.

This artifact is **internal** and is not redistributable: it contains exact
source bids/asks, liquidity fields, vendor expiry identities, and absolute spot
references. It is not the public SANOS projection and does not use legacy
`widb.timeseries`.

Source Dataset: `{config.source_dataset.namespace}/`
`{config.source_dataset.artifact_id}`.

Validate with `sanos experiment validate --config CONFIG.json`. Rebuilding or
deleting this Experiment does not change its immutable source Dataset.
"""


@version("1.0", dependencies=[ExperimentValidationSummary])
def _experiment_summary(value: Mapping[str, Any]) -> ExperimentValidationSummary:
    dimensions = cast(Mapping[str, Any], value["dimensions"])
    dates = cast(Mapping[str, Any], value["dates"])
    return ExperimentValidationSummary(
        dates=int(dates["count"]),
        books=int(value["books"]),
        calls=int(value["calls"]),
        puts=int(value["puts"]),
        expiries_max=int(dimensions["E"]),
        quotes_max=int(dimensions["Q"]),
        price_locations_max=int(dimensions["P"]),
        model_nodes=int(dimensions["M"]),
        weekly_pairs=int(value["weekly_pairs"]),
        first_date=date.fromisoformat(str(dates["first"])),
        last_date=date.fromisoformat(str(dates["last"])),
        logical_content_sha256=str(value["logical_content_sha256"]),
    )


@version("1.0", dependencies=[DatasetValidationSummary, ValidatedArtifact])
def _experiment_source_document(
    source: ValidatedArtifact,
    source_summary: DatasetValidationSummary,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_SOURCE_SCHEMA,
        "dataset": source.pointer,
        "manifest_sha256": manifest_sha256,
        "manifest_configuration_id": source.manifest["configuration_id"],
        "manifest_schema_version": source.manifest["schema_version"],
        "distribution": source.manifest["distribution"],
        "semantic_validation": source_summary.as_dict(),
    }


@version("1.0", dependencies=[ExperimentConfig, ExperimentValidationSummary])
def _experiment_split_document(
    config: ExperimentConfig,
    summary: ExperimentValidationSummary,
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_SPLITS_SCHEMA,
        "mode": config.split.mode,
        "training_dates": summary.dates,
        "validation_dates": 0,
        "test_dates": 0,
        "diagnostic_boundary": config.split.diagnostic_boundary,
        "weekly_pairs": summary.weekly_pairs,
    }


@version("1.0", dependencies=[ExperimentValidationSummary])
def _experiment_validation_document(
    summary: ExperimentValidationSummary,
    hdf5_validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_VALIDATION_SCHEMA,
        "status": "passed",
        "summary": summary.as_dict(),
        "hdf5": hdf5_validation,
    }


@version("1.0")
def _read_experiment_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentValidationError(f"Cannot read Experiment {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"Experiment {label} must be a JSON object")
    return value


@version(
    "1.1",
    dependencies=[
        ExperimentConfig,
        daily_quote_objective,
        project_daily_front,
        write_experiment_hdf5,
        validate_experiment_hdf5,
        _parse_manifest_config,
        _source_gate,
        _experiment_schema_document,
        _experiment_readme,
        _experiment_summary,
        _experiment_source_document,
        _experiment_split_document,
        _experiment_validation_document,
        _prepare_shards,
        _project_partition,
        _read_experiment_json,
        read_day_shard,
        write_day_shard,
    ],
)
def experiment_producer_version() -> str:
    """Return the resolved content-producing and publication-gate version closure."""

    return str(experiment_producer_version.version)


@version("1.0", dependencies=[ExperimentConfig, ArtifactSpec, experiment_producer_version])
def plan_experiment(config: ExperimentConfig) -> ExperimentPlan:
    """Plan one Experiment identity without opening storage or its source."""

    spec = ArtifactSpec.create(
        storage_class=StorageClass.EXPERIMENT,
        namespace=f"dysanos/daily-front-market/name-{config.name}",
        configuration={
            "schema": EXPERIMENT_CONFIG_SCHEMA,
            "experiment": config.to_document(),
        },
        schema_version=EXPERIMENT_HDF5_SCHEMA,
        producer_version=experiment_producer_version(),
        upstream=(config.source_dataset.pointer,),
    )
    return ExperimentPlan(
        config=config,
        spec=spec,
        plan_id=canonical_sha256(
            {
                "schema": "sanos-experiment-plan-v1",
                "artifact": spec.pointer,
                "configuration": config,
            }
        ),
    )


@version(
    "1.0",
    dependencies=[
        _parse_manifest_config,
        _experiment_summary,
        _experiment_schema_document,
        _experiment_source_document,
        _read_experiment_json,
        _source_gate,
        validate_experiment_hdf5,
        experiment_producer_version,
    ],
)
def validate_experiment(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
    *,
    semantic: bool = True,
    deep_lineage: bool = False,
    progress: ProgressCallback | None = None,
) -> ExperimentValidationSummary | None:
    """Validate an Experiment shallowly and optionally traverse its Dataset source."""

    pointer = artifact.pointer if isinstance(artifact, ValidatedArtifact) else artifact
    validated = store.validate(pointer, deep=deep_lineage)
    if validated.pointer.storage_class is not StorageClass.EXPERIMENT:
        raise ExperimentValidationError("Requested artifact is not an Experiment")
    config = _parse_manifest_config(validated.manifest.get("configuration"))
    config_file = _read_experiment_json(validated.path / "config.json", "config.json")
    if canonical_json(config_file) != canonical_json(validated.manifest["configuration"]):
        raise ExperimentValidationError("Experiment config.json differs from its manifest")
    if validated.manifest.get("schema_version") != EXPERIMENT_HDF5_SCHEMA:
        raise ExperimentValidationError("Unsupported Experiment artifact schema")
    upstream = validated.manifest.get("upstream")
    if not isinstance(upstream, list) or upstream != [config.source_dataset.pointer.as_dict()]:
        raise ExperimentValidationError("Experiment source pointer differs from its manifest")
    if not semantic:
        return None
    result = validate_experiment_hdf5(
        validated.path / EXPERIMENT_HDF5_NAME,
        config=config,
        producer_version=str(validated.manifest["producer_version"]),
        progress=progress,
    )
    summary = _experiment_summary(result)
    schema_document = _read_experiment_json(validated.path / "schema.json", "schema.json")
    if canonical_json(schema_document) != canonical_json(_experiment_schema_document()):
        raise ExperimentValidationError("Experiment schema.json changed")
    source_document = _read_experiment_json(validated.path / "source.json", "source.json")
    required_source = {
        "schema",
        "dataset",
        "manifest_sha256",
        "manifest_configuration_id",
        "manifest_schema_version",
        "distribution",
        "semantic_validation",
    }
    source_distribution = source_document.get("distribution")
    if (
        set(source_document) != required_source
        or source_document.get("schema") != EXPERIMENT_SOURCE_SCHEMA
        or source_document.get("dataset") != config.source_dataset.pointer.as_dict()
        or not isinstance(source_document.get("manifest_sha256"), str)
        or len(str(source_document["manifest_sha256"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(source_document["manifest_sha256"])
        )
        or not isinstance(source_distribution, Mapping)
        or source_distribution.get("status") != DistributionStatus.INTERNAL.value
    ):
        raise ExperimentValidationError("Experiment source.json changed")
    if deep_lineage:
        source, source_summary = _source_gate(
            store,
            config,
            progress=progress,
            progress_interval_seconds=30.0,
        )
        source_completion = _read_experiment_json(
            source.path / "COMPLETE.json",
            "source Dataset COMPLETE.json",
        )
        source_manifest_sha256 = source_completion.get("manifest_sha256")
        if not isinstance(source_manifest_sha256, str):
            raise ExperimentValidationError(
                "Source Dataset completion marker has no manifest checksum"
            )
        expected_source_document = _experiment_source_document(
            source,
            source_summary,
            manifest_sha256=source_manifest_sha256,
        )
        if canonical_json(source_document) != canonical_json(expected_source_document):
            raise ExperimentValidationError(
                "Experiment source evidence differs from deep Dataset validation"
            )
    split_document = _read_experiment_json(validated.path / "splits.json", "splits.json")
    expected_split = _experiment_split_document(config, summary)
    if canonical_json(split_document) != canonical_json(expected_split):
        raise ExperimentValidationError("Experiment splits.json changed")
    evidence = _read_experiment_json(validated.path / "validation.json", "validation.json")
    if not isinstance(evidence, Mapping) or canonical_json(evidence) != canonical_json(
        _experiment_validation_document(summary, result)
    ):
        raise ExperimentValidationError("Experiment validation evidence changed")
    return summary


@version("1.0", dependencies=[_parse_manifest_config, DailyFrontExperiment])
def open_experiment(
    store: ArtifactStore,
    artifact: ArtifactPointer | ValidatedArtifact,
) -> DailyFrontExperiment:
    """Open a complete Experiment without requiring its Dataset source."""

    validated = store.validate(artifact) if isinstance(artifact, ArtifactPointer) else artifact
    config = _parse_manifest_config(validated.manifest.get("configuration"))
    return DailyFrontExperiment(
        path=validated.path / EXPERIMENT_HDF5_NAME,
        config=config,
        producer_version=str(validated.manifest["producer_version"]),
    )


@version(
    "1.0",
    dependencies=[
        plan_experiment,
        _source_gate,
        _prepare_shards,
        _experiment_schema_document,
        _experiment_readme,
        _experiment_source_document,
        _experiment_split_document,
        _experiment_validation_document,
        validate_experiment,
        write_experiment_hdf5,
    ],
)
def build_experiment(
    locations: StorageLocations,
    config: ExperimentConfig,
    options: ExperimentBuildOptions = ExperimentBuildOptions(),
    *,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> ExperimentBuildResult:
    """Build or output-first reuse one self-contained Experiment artifact."""

    plan = plan_experiment(config)
    store = ArtifactStore(locations)
    workspace = locations.experiments / ".sanos-runs" / plan.plan_id
    journal_path = locations.experiments / ".runs" / "dysanos" / f"{plan.plan_id}.jsonl"
    _emit(
        progress,
        "experiment_plan",
        artifact=plan.spec.pointer,
        plan_id=plan.plan_id,
        dry_run=dry_run,
        source_accessed=False,
        network_accessed=False,
    )
    reuse = store.resolve(plan.spec, version=options.version_override)
    if reuse is not None:
        summary = validate_experiment(store, reuse.artifact, semantic=True, progress=progress)
        return ExperimentBuildResult(
            plan=plan,
            artifact=reuse.artifact.pointer,
            path=reuse.artifact.path,
            dry_run=dry_run,
            built=False,
            reuse=reuse,
            validation=summary,
            source_accessed=False,
            workspace=workspace,
            journal_path=journal_path,
        )
    if dry_run:
        return ExperimentBuildResult(
            plan=plan,
            artifact=plan.spec.pointer,
            path=store.path_for(plan.spec.pointer),
            dry_run=True,
            built=False,
            reuse=None,
            validation=None,
            source_accessed=False,
            workspace=workspace,
            journal_path=journal_path,
        )

    runs_root = workspace.parent
    runs_root.mkdir(parents=True, exist_ok=True)
    if runs_root.is_symlink() or workspace.is_symlink():
        raise ExperimentValidationError("Experiment workspace path is unsafe")
    lock_path = runs_root / f"{plan.plan_id}.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ExperimentValidationError(f"Experiment workspace lock is unsafe: {lock_path}")
    lock = FileLock(str(lock_path), timeout=30.0)
    with lock:
        winner = store.resolve(plan.spec)
        if winner is not None:
            summary = validate_experiment(store, winner.artifact, semantic=True, progress=progress)
            return ExperimentBuildResult(
                plan=plan,
                artifact=winner.artifact.pointer,
                path=winner.artifact.path,
                dry_run=False,
                built=False,
                reuse=winner,
                validation=summary,
                source_accessed=False,
                workspace=workspace,
                journal_path=journal_path,
            )
        journal = RunJournal(journal_path)
        journal.require_plan(plan.plan_id)
        journal.emit(
            "run_started",
            plan_id=plan.plan_id,
            artifact=plan.spec.pointer,
            workspace=workspace,
        )
        try:
            if options.restart_work:
                _restart_workspace(workspace, locations.experiments)
                journal.emit("workspace_restarted", workspace=workspace)
            elif not options.resume and workspace.exists():
                raise ExperimentValidationError(
                    "Experiment restart workspace exists while --no-resume is active; "
                    "pass --restart-work to remove that exact workspace"
                )
            source, source_summary = _source_gate(
                store,
                config,
                progress=progress,
                progress_interval_seconds=options.progress_interval_seconds,
            )
            dataset = open_internal_dataset(store, source)
            shards, exclusions = _prepare_shards(
                dataset,
                plan=plan,
                workspace=workspace,
                config=config,
                options=options,
                progress=progress,
                journal=journal,
            )
            days = tuple(read_day_shard(path) for path in shards)
            metadata = {
                "profile": config.profile,
                "source_dataset": source.pointer,
                "source_distribution": "internal",
                "source_validation": source_summary.as_dict(),
                "typed_empty_exclusions": exclusions,
                "network_accessed": False,
            }
            with store.writer(plan.spec, metadata=metadata) as builder:
                hdf_path = builder.staging_path / EXPERIMENT_HDF5_NAME
                write_experiment_hdf5(
                    hdf_path,
                    days,
                    config=config,
                    producer_version=plan.spec.producer_version,
                    progress=progress,
                    progress_interval_seconds=options.progress_interval_seconds,
                )
                validation_value = validate_experiment_hdf5(
                    hdf_path,
                    config=config,
                    producer_version=plan.spec.producer_version,
                    progress=progress,
                )
                summary = _experiment_summary(validation_value)
                source_completion = _read_experiment_json(
                    source.path / "COMPLETE.json",
                    "source Dataset COMPLETE.json",
                )
                source_manifest_sha256 = source_completion.get("manifest_sha256")
                if not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64:
                    raise ExperimentValidationError(
                        "Source Dataset completion marker has no manifest checksum"
                    )
                source_document = _experiment_source_document(
                    source,
                    source_summary,
                    manifest_sha256=source_manifest_sha256,
                )
                split_document = _experiment_split_document(config, summary)
                validation_document = _experiment_validation_document(
                    summary,
                    validation_value,
                )
                builder.write_json("config.json", plan.spec.configuration)
                builder.write_json("schema.json", _experiment_schema_document())
                builder.write_json("source.json", source_document)
                builder.write_json("splits.json", split_document)
                builder.write_json("validation.json", validation_document)
                builder.write_text("README.md", _experiment_readme(config, summary))
            validated = store.validate(plan.spec.pointer, expected=plan.spec)
            final_summary = validate_experiment(
                store,
                validated,
                semantic=True,
                progress=progress,
            )
            journal.emit("artifact_published", artifact=validated.pointer)
            if not options.keep_work:
                try:
                    _restart_workspace(workspace, locations.experiments)
                except (OSError, ExperimentValidationError) as cleanup_error:
                    warnings.warn(
                        f"Published Experiment but could not remove restart workspace "
                        f"{workspace}: {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    journal.emit(
                        "workspace_cleanup_failed",
                        workspace=workspace,
                        error_type=type(cleanup_error).__name__,
                        message=str(cleanup_error),
                    )
            journal.emit(
                "run_completed",
                plan_id=plan.plan_id,
                artifact=validated.pointer,
                dates=summary.dates,
                books=summary.books,
            )
        except BaseException as exc:
            journal.emit(
                "run_interrupted" if isinstance(exc, KeyboardInterrupt) else "run_failed",
                plan_id=plan.plan_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        _emit(
            progress,
            "experiment_build_completed",
            artifact=validated.pointer,
            path=validated.path,
            dates=summary.dates,
            books=summary.books,
            source_accessed=True,
            network_accessed=False,
        )
        return ExperimentBuildResult(
            plan=plan,
            artifact=validated.pointer,
            path=validated.path,
            dry_run=False,
            built=True,
            reuse=None,
            validation=final_summary,
            source_accessed=True,
            workspace=workspace,
            journal_path=journal_path,
        )


__all__ = [
    "ExperimentBuildResult",
    "ExperimentPlan",
    "ExperimentValidationSummary",
    "build_experiment",
    "experiment_producer_version",
    "open_experiment",
    "plan_experiment",
    "validate_experiment",
]
