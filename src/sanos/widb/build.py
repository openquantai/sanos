"""Output-first construction of immutable per-day WIDB intermediates."""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl
from cdxcore.version import version

from sanos.artifacts import (
    ArtifactExistsError,
    ArtifactNotFoundError,
    ArtifactPointer,
    ArtifactSpec,
    ArtifactStore,
    ReuseResult,
    StorageClass,
    ValidatedArtifact,
)
from sanos.config import StorageLocations, canonical_value
from sanos.pipeline import (
    DailyPlan,
    DailyRunSummary,
    DailyWorkItem,
    plan_daily_dates,
    run_daily_plan,
)
from sanos.widb.bad import bad_data_policy, bad_day_reason
from sanos.widb.calendar import business_dates, previous_business_day
from sanos.widb.config import DailySurfaceConfig, DailyWidbConfig
from sanos.widb.daily import (
    ENRICHED_SCHEMA,
    ENRICHED_SCHEMA_VERSION,
    PAIRED_SCHEMA,
    PAIRED_SCHEMA_VERSION,
    DailyDataError,
    DailyProvenance,
    DailyRawInputs,
    enrich_daily_frame,
    pair_daily_frame,
)
from sanos.widb.surface import (
    FILTERED_SCHEMA,
    FILTERED_SCHEMA_VERSION,
    FITTED_SCHEMA,
    FITTED_SCHEMA_VERSION,
    NO_ARB_SCHEMA,
    NO_ARB_SCHEMA_VERSION,
    apply_options_filter,
    fit_initial_no_arbitrage,
    fit_smooth_sanos,
)
from sanos.wrds import (
    RawPartition,
    RawTable,
    raw_artifact_request,
    raw_content_producer_version,
)

IntermediateStage = Literal[
    "paired-day",
    "enriched-day",
    "no-arb-day",
    "filtered-day",
    "fitted-day",
]
INTERMEDIATE_STAGES: tuple[IntermediateStage, ...] = (
    "paired-day",
    "enriched-day",
    "no-arb-day",
    "filtered-day",
    "fitted-day",
)
_SAFE_DAILY_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


@version("1.0")
@dataclass(frozen=True, slots=True)
class RawDayPointers:
    """The exact six Raw partitions required to assemble one trading day."""

    security: ArtifactPointer
    options: ArtifactPointer
    spot: ArtifactPointer
    previous_spot: ArtifactPointer
    forwards: ArtifactPointer
    zero_curve: ArtifactPointer

    def all(self) -> tuple[ArtifactPointer, ...]:
        """Return deterministic lineage pointers."""

        return tuple(
            sorted(
                (
                    self.security,
                    self.options,
                    self.spot,
                    self.previous_spot,
                    self.forwards,
                    self.zero_curve,
                )
            )
        )


@version("1.0", dependencies=[RawDayPointers, ArtifactSpec])
@dataclass(frozen=True, slots=True)
class DailyArtifactPlan:
    """Exact Raw and daily-stage identities for one planned trading session."""

    trading_date: date
    raw: RawDayPointers
    paired: ArtifactSpec
    enriched: ArtifactSpec
    no_arb: ArtifactSpec
    filtered: ArtifactSpec
    fitted: ArtifactSpec

    def stage_spec(self, stage: IntermediateStage) -> ArtifactSpec:
        """Return the immutable specification for a named daily stage."""

        return {
            "paired-day": self.paired,
            "enriched-day": self.enriched,
            "no-arb-day": self.no_arb,
            "filtered-day": self.filtered,
            "fitted-day": self.fitted,
        }[stage]


@version("1.0")
@dataclass(frozen=True, slots=True)
class IntermediateBuildRequest:
    """Selection and executor settings that do not alter daily content."""

    ticker: str
    start_date: date
    end_date: date
    through: IntermediateStage = "enriched-day"
    include_half_days: bool = True
    source: str = "optionmetrics"
    source_generation: str = "default-v1"
    max_workers: int = 1
    progress_interval_seconds: float = 30.0
    version_override: str | None = None

    def __post_init__(self) -> None:
        validated_ticker = RawPartition(RawTable.SECURITY, self.ticker, None).ticker
        object.__setattr__(self, "ticker", validated_ticker)
        if self.end_date < self.start_date:
            raise ValueError("Intermediate end_date cannot precede start_date")
        if self.through not in INTERMEDIATE_STAGES:
            raise ValueError(f"through must be one of {INTERMEDIATE_STAGES}")
        if not isinstance(self.include_half_days, bool):
            raise TypeError("include_half_days must be a boolean")
        if self.source != "optionmetrics":
            raise ValueError(f"Unsupported Raw source: {self.source!r}")
        if not self.source_generation.strip():
            raise ValueError("source_generation cannot be empty")
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int):
            raise TypeError("max_workers must be an integer")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.through in {"no-arb-day", "filtered-day", "fitted-day"}:
            if self.max_workers > 4:
                raise ValueError("Surface fitting supports at most four local workers")
        if self.version_override not in {None, "*"}:
            raise ValueError("version_override must be None or '*'")


@version(
    "1.0",
    dependencies=[DailyPlan, ArtifactPointer, RawDayPointers, ReuseResult],
)
@dataclass(frozen=True, slots=True)
class IntermediateBuildResult:
    """Dry-run inspection or material execution result."""

    plan: DailyPlan
    expected: tuple[ArtifactPointer, ...]
    terminal: tuple[ArtifactPointer, ...]
    existing: tuple[ArtifactPointer, ...]
    reuses: tuple[ReuseResult, ...]
    raw_dependencies: tuple[RawDayPointers, ...]
    dry_run: bool
    summary: DailyRunSummary | None = None


@version("1.0", dependencies=[RawDayPointers, previous_business_day, raw_artifact_request])
def plan_raw_day_pointers(
    *,
    ticker: str,
    trading_date: date,
    source: str,
    source_generation: str,
    raw_producer_version: str,
    previous_spot_date: date | None = None,
) -> RawDayPointers:
    """Plan the six location-independent Raw identities without reading files."""

    previous = (
        previous_business_day(trading_date)
        if previous_spot_date is None
        else previous_spot_date
    )
    if previous >= trading_date:
        raise ValueError("previous_spot_date must precede trading_date")

    def pointer(table: RawTable, observation_date: date | None) -> ArtifactPointer:
        return raw_artifact_request(
            RawPartition(table, ticker, observation_date),
            source=source,
            source_generation=source_generation,
            producer_version=raw_producer_version,
        ).spec.pointer

    return RawDayPointers(
        security=pointer(RawTable.SECURITY, None),
        options=pointer(RawTable.OPTIONS, trading_date),
        spot=pointer(RawTable.SPOT, trading_date),
        previous_spot=pointer(RawTable.SPOT, previous),
        forwards=pointer(RawTable.FORWARDS, trading_date),
        zero_curve=pointer(RawTable.ZERO_CURVE, trading_date),
    )


@version("1.0")
def _safe_daily_ticker_label(ticker: str) -> str:
    """Return a collision-resistant runner/namespace label for a vendor ticker."""

    normalized = RawPartition(RawTable.SECURITY, ticker, None).ticker
    label = _SAFE_DAILY_LABEL.sub("-", normalized).strip("-._") or "ticker"
    if label != normalized:
        suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
        label = f"{label}-{suffix}"
    return label


@version("1.0", dependencies=[_safe_daily_ticker_label])
def _daily_namespace(stage: IntermediateStage, ticker: str, trading_date: date) -> str:
    ticker_label = _safe_daily_ticker_label(ticker)
    return (
        f"widb/{stage}/ticker-{ticker_label}/year-{trading_date.year}/"
        f"date-{trading_date.isoformat()}"
    )


@version("1.0")
def _artifact_configuration(
    stage: IntermediateStage,
    ticker: str,
    trading_date: date,
    config: DailyWidbConfig,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "ticker": ticker,
        "trading_date": trading_date,
        "daily": config,
    }


@version("1.0")
def _surface_artifact_configuration(
    stage: IntermediateStage,
    ticker: str,
    trading_date: date,
    config: DailySurfaceConfig,
) -> dict[str, Any]:
    if stage == "no-arb-day":
        settings: dict[str, Any] = {"no_arbitrage": config.no_arbitrage}
    elif stage == "filtered-day":
        settings = {"options_filter": config.options_filter}
    elif stage == "fitted-day":
        settings = {
            "no_arbitrage": config.no_arbitrage,
            "smooth_sanos": config.smooth_sanos,
            "discrete_dlv": config.discrete_dlv,
        }
    else:
        raise ValueError(f"Surface configuration is not defined for stage {stage!r}")
    return {
        "stage": stage,
        "ticker": ticker,
        "trading_date": trading_date,
        "surface": settings,
    }


@version("1.0", dependencies=[pair_daily_frame])
def paired_day_producer_version() -> str:
    """Return the paired-stage content version closure."""

    return str(paired_day_producer_version.version)


@version("1.0", dependencies=[enrich_daily_frame])
def enriched_day_producer_version() -> str:
    """Return the enriched-stage content version closure."""

    return str(enriched_day_producer_version.version)


@version("1.0", dependencies=[fit_initial_no_arbitrage])
def no_arb_day_producer_version() -> str:
    """Return the initial no-arbitrage content version closure."""

    return str(no_arb_day_producer_version.version)


@version("1.0", dependencies=[apply_options_filter])
def filtered_day_producer_version() -> str:
    """Return the option-filter content version closure."""

    return str(filtered_day_producer_version.version)


@version("1.0", dependencies=[fit_smooth_sanos])
def fitted_day_producer_version() -> str:
    """Return the smooth-SANOS plus discrete-DLV content version closure."""

    return str(fitted_day_producer_version.version)


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        RawDayPointers,
        _artifact_configuration,
        _daily_namespace,
        paired_day_producer_version,
    ],
)
def paired_day_spec(
    *,
    ticker: str,
    trading_date: date,
    config: DailyWidbConfig,
    raw: RawDayPointers,
) -> ArtifactSpec:
    """Create the complete paired-day identity."""

    return ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=_daily_namespace("paired-day", ticker, trading_date),
        configuration=_artifact_configuration("paired-day", ticker, trading_date, config),
        schema_version=PAIRED_SCHEMA_VERSION,
        producer_version=paired_day_producer_version(),
        upstream=raw.all(),
    )


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        _artifact_configuration,
        _daily_namespace,
        enriched_day_producer_version,
    ],
)
def enriched_day_spec(
    *,
    ticker: str,
    trading_date: date,
    config: DailyWidbConfig,
    paired: ArtifactPointer,
) -> ArtifactSpec:
    """Create the complete enriched-day identity."""

    return ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=_daily_namespace("enriched-day", ticker, trading_date),
        configuration=_artifact_configuration("enriched-day", ticker, trading_date, config),
        schema_version=ENRICHED_SCHEMA_VERSION,
        producer_version=enriched_day_producer_version(),
        upstream=(paired,),
    )


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        _daily_namespace,
        _surface_artifact_configuration,
        no_arb_day_producer_version,
    ],
)
def no_arb_day_spec(
    *,
    ticker: str,
    trading_date: date,
    config: DailySurfaceConfig,
    enriched: ArtifactPointer,
) -> ArtifactSpec:
    return ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=_daily_namespace("no-arb-day", ticker, trading_date),
        configuration=_surface_artifact_configuration(
            "no-arb-day", ticker, trading_date, config
        ),
        schema_version=NO_ARB_SCHEMA_VERSION,
        producer_version=no_arb_day_producer_version(),
        upstream=(enriched,),
    )


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        _daily_namespace,
        _surface_artifact_configuration,
        filtered_day_producer_version,
    ],
)
def filtered_day_spec(
    *,
    ticker: str,
    trading_date: date,
    config: DailySurfaceConfig,
    no_arb: ArtifactPointer,
) -> ArtifactSpec:
    return ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=_daily_namespace("filtered-day", ticker, trading_date),
        configuration=_surface_artifact_configuration(
            "filtered-day", ticker, trading_date, config
        ),
        schema_version=FILTERED_SCHEMA_VERSION,
        producer_version=filtered_day_producer_version(),
        upstream=(no_arb,),
    )


@version(
    "1.0",
    dependencies=[
        ArtifactSpec,
        _daily_namespace,
        _surface_artifact_configuration,
        fitted_day_producer_version,
    ],
)
def fitted_day_spec(
    *,
    ticker: str,
    trading_date: date,
    config: DailySurfaceConfig,
    filtered: ArtifactPointer,
) -> ArtifactSpec:
    return ArtifactSpec.create(
        storage_class=StorageClass.INTERMEDIATE,
        namespace=_daily_namespace("fitted-day", ticker, trading_date),
        configuration=_surface_artifact_configuration(
            "fitted-day", ticker, trading_date, config
        ),
        schema_version=FITTED_SCHEMA_VERSION,
        producer_version=fitted_day_producer_version(),
        upstream=(filtered,),
    )


@version(
    "1.0",
    dependencies=[
        DailyArtifactPlan,
        plan_raw_day_pointers,
        paired_day_spec,
        enriched_day_spec,
        no_arb_day_spec,
        filtered_day_spec,
        fitted_day_spec,
    ],
)
def plan_daily_artifact(
    *,
    ticker: str,
    trading_date: date,
    source: str,
    source_generation: str,
    raw_producer_version: str,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig,
    previous_spot_date: date | None = None,
) -> DailyArtifactPlan:
    """Plan the complete immutable identity chain for one trading session."""

    raw = plan_raw_day_pointers(
        ticker=ticker,
        trading_date=trading_date,
        source=source,
        source_generation=source_generation,
        raw_producer_version=raw_producer_version,
        previous_spot_date=previous_spot_date,
    )
    paired = paired_day_spec(
        ticker=ticker,
        trading_date=trading_date,
        config=daily_config,
        raw=raw,
    )
    enriched = enriched_day_spec(
        ticker=ticker,
        trading_date=trading_date,
        config=daily_config,
        paired=paired.pointer,
    )
    no_arb = no_arb_day_spec(
        ticker=ticker,
        trading_date=trading_date,
        config=surface_config,
        enriched=enriched.pointer,
    )
    filtered = filtered_day_spec(
        ticker=ticker,
        trading_date=trading_date,
        config=surface_config,
        no_arb=no_arb.pointer,
    )
    fitted = fitted_day_spec(
        ticker=ticker,
        trading_date=trading_date,
        config=surface_config,
        filtered=filtered.pointer,
    )
    return DailyArtifactPlan(
        trading_date=trading_date,
        raw=raw,
        paired=paired,
        enriched=enriched,
        no_arb=no_arb,
        filtered=filtered,
        fitted=fitted,
    )


@version(
    "1.0",
    dependencies=[
        IntermediateBuildRequest,
        DailyArtifactPlan,
        business_dates,
        plan_daily_artifact,
        raw_content_producer_version,
    ],
)
def plan_daily_artifacts(
    request: IntermediateBuildRequest,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig = DailySurfaceConfig(),
) -> tuple[DailyArtifactPlan, ...]:
    """Plan every date and artifact identity without reading configured roots."""

    dates = business_dates(
        request.start_date,
        request.end_date,
        exclude_half_days=not request.include_half_days,
    )
    if not dates:
        raise ValueError("The intermediate request contains no eligible NYSE sessions")
    complete_schedule = business_dates(request.start_date - timedelta(days=14), request.end_date)
    positions = {trading_date: index for index, trading_date in enumerate(complete_schedule)}
    previous_dates: dict[date, date] = {}
    for trading_date in dates:
        position = positions.get(trading_date)
        if position is None or position == 0:
            raise RuntimeError(f"Cannot find the prior NYSE session for {trading_date}")
        previous_dates[trading_date] = complete_schedule[position - 1]
    raw_version = raw_content_producer_version()
    return tuple(
        plan_daily_artifact(
            ticker=request.ticker,
            trading_date=trading_date,
            source=request.source,
            source_generation=request.source_generation,
            raw_producer_version=raw_version,
            daily_config=daily_config,
            surface_config=surface_config,
            previous_spot_date=previous_dates[trading_date],
        )
        for trading_date in dates
    )


@version("1.0")
def _write_parquet_durable(frame: pl.DataFrame, path: Path) -> None:
    frame.write_parquet(path, compression="zstd")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


@version("1.0")
def _validate_daily_frame(
    frame: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    ticker: str,
    trading_date: date,
) -> None:
    expected = pl.Schema(schema)
    if frame.schema != expected:
        raise DailyDataError(
            f"Daily frame schema mismatch: expected={expected}, observed={frame.schema}"
        )
    if frame.is_empty():
        return
    if frame.get_column("ticker").unique().to_list() != [ticker]:
        raise DailyDataError(f"Daily frame does not contain only ticker {ticker}")
    if frame.get_column("trading_date").unique().to_list() != [trading_date]:
        raise DailyDataError(
            f"Daily frame does not contain only trading date {trading_date.isoformat()}"
        )


@version("1.0", dependencies=[ArtifactStore, _validate_daily_frame])
def read_daily_frame(
    store: ArtifactStore,
    artifact: ValidatedArtifact,
    *,
    schema: Mapping[str, pl.DataType],
    ticker: str,
    trading_date: date,
) -> pl.DataFrame:
    """Read and semantically validate one complete daily Parquet artifact."""

    path = artifact.path / "data.parquet"
    if path.is_symlink() or not path.is_file():
        raise ArtifactNotFoundError(f"Daily artifact has no safe data.parquet: {path}")
    frame = pl.read_parquet(path)
    _validate_daily_frame(
        frame,
        schema=schema,
        ticker=ticker,
        trading_date=trading_date,
    )
    return frame


@version(
    "1.1",
    dependencies=[
        ArtifactStore,
        _write_parquet_durable,
        _validate_daily_frame,
        bad_data_policy,
        bad_day_reason,
    ],
)
def publish_daily_frame(
    store: ArtifactStore,
    spec: ArtifactSpec,
    frame: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    ticker: str,
    trading_date: date,
) -> ValidatedArtifact:
    """Atomically publish one frame with its portable configuration and description."""

    _validate_daily_frame(
        frame,
        schema=schema,
        ticker=ticker,
        trading_date=trading_date,
    )
    exclusion_reason = bad_day_reason(ticker, trading_date) if frame.is_empty() else None
    metadata = {
        "ticker": ticker,
        "trading_date": trading_date,
        "row_count": frame.height,
        "column_count": frame.width,
        "typed_empty": frame.is_empty(),
        "typed_empty_reason": exclusion_reason,
        "bad_data_policy_version": (
            str(bad_data_policy.version) if exclusion_reason is not None else None
        ),
        "persisted_schema": {name: str(dtype) for name, dtype in frame.schema.items()},
    }
    builder = store.writer(spec, metadata=metadata)
    try:
        with builder as staging:
            _write_parquet_durable(frame, staging.staging_path / "data.parquet")
            staging.write_json("config.json", spec.configuration)
            staging.write_text(
                "README.md",
                f"# {spec.namespace}\n\n"
                "One immutable Polars/Parquet trading-day frame. The manifest records "
                "all direct upstream artifact identities; `config.json` records the exact "
                "canonical content configuration.\n",
            )
    except ArtifactExistsError:
        return store.validate(spec.pointer, expected=spec)
    return builder.artifact


def _locations_from_payload(payload: Mapping[str, Any]) -> StorageLocations:
    roots = payload["locations"]
    if not isinstance(roots, Mapping):
        raise ValueError("Daily worker locations must be a mapping")
    return StorageLocations(
        raw=Path(str(roots["raw"])),
        intermediate=Path(str(roots["intermediate"])),
        datasets=Path(str(roots["datasets"])),
        experiments=Path(str(roots["experiments"])),
    )


def _config_from_payload(payload: Mapping[str, Any]) -> DailyWidbConfig:
    value = payload["daily_config"]
    if not isinstance(value, Mapping):
        raise ValueError("daily_config must be a mapping")
    return DailyWidbConfig.from_mapping(value)


def _surface_config_from_payload(payload: Mapping[str, Any]) -> DailySurfaceConfig:
    value = payload.get("surface_config", {})
    if not isinstance(value, Mapping):
        raise ValueError("surface_config must be a mapping")
    return DailySurfaceConfig.from_mapping(value)


def _actual_ticker(item: DailyWorkItem) -> str:
    payload = item.payload
    if not isinstance(payload, Mapping):
        raise ValueError("Daily worker payload must be a mapping")
    ticker = payload.get("ticker")
    if not isinstance(ticker, str):
        raise ValueError("Daily worker payload ticker must be a string")
    return RawPartition(RawTable.SECURITY, ticker, None).ticker


def _specs_from_item(
    item: DailyWorkItem,
) -> tuple[
    ArtifactStore,
    DailyWidbConfig,
    DailySurfaceConfig,
    RawDayPointers,
    ArtifactSpec,
    ArtifactSpec,
    ArtifactSpec,
    ArtifactSpec,
    ArtifactSpec,
]:
    payload = item.payload
    if not isinstance(payload, Mapping):
        raise ValueError("Daily worker payload must be a mapping")
    store = ArtifactStore(_locations_from_payload(payload))
    config = _config_from_payload(payload)
    surface_config = _surface_config_from_payload(payload)
    ticker = _actual_ticker(item)
    planned = plan_daily_artifact(
        ticker=ticker,
        trading_date=item.trading_date,
        source=str(payload["source"]),
        source_generation=str(payload["source_generation"]),
        raw_producer_version=str(payload["raw_producer_version"]),
        daily_config=config,
        surface_config=surface_config,
    )
    return (
        store,
        config,
        surface_config,
        planned.raw,
        planned.paired,
        planned.enriched,
        planned.no_arb,
        planned.filtered,
        planned.fitted,
    )


def _target_spec(
    item: DailyWorkItem,
    paired: ArtifactSpec,
    enriched: ArtifactSpec,
    no_arb: ArtifactSpec,
    filtered: ArtifactSpec,
    fitted: ArtifactSpec,
) -> ArtifactSpec:
    by_stage = {
        "paired-day": paired,
        "enriched-day": enriched,
        "no-arb-day": no_arb,
        "filtered-day": filtered,
        "fitted-day": fitted,
    }
    return by_stage[item.stage]


def probe_daily_target(item: DailyWorkItem) -> ArtifactPointer | None:
    """Shallowly resolve the requested terminal artifact without reading upstreams."""

    store, _, _, _, paired, enriched, no_arb, filtered, fitted = _specs_from_item(item)
    payload = item.payload
    target = _target_spec(item, paired, enriched, no_arb, filtered, fitted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reuse = store.resolve(target, version=payload.get("version_override"))
    return None if reuse is None else reuse.artifact.pointer


def _validated_raw(store: ArtifactStore, pointer: ArtifactPointer) -> ValidatedArtifact:
    try:
        return store.validate(pointer)
    except ArtifactNotFoundError as exc:
        raise ArtifactNotFoundError(
            f"Required Raw artifact is absent; run `sanos raw download` first: {pointer.namespace}"
        ) from exc


def _build_paired(
    store: ArtifactStore,
    config: DailyWidbConfig,
    raw: RawDayPointers,
    spec: ArtifactSpec,
    *,
    ticker: str,
    trading_date: date,
) -> ValidatedArtifact:
    reuse = store.resolve(spec)
    if reuse is not None:
        return reuse.artifact
    artifacts = {
        name: _validated_raw(store, pointer)
        for name, pointer in (
            ("security", raw.security),
            ("options", raw.options),
            ("spot", raw.spot),
            ("previous_spot", raw.previous_spot),
            ("forwards", raw.forwards),
            ("zero_curve", raw.zero_curve),
        )
    }
    inputs = DailyRawInputs(
        security=pl.read_parquet(artifacts["security"].path / "data.parquet"),
        options=pl.read_parquet(artifacts["options"].path / "data.parquet"),
        spot=pl.read_parquet(artifacts["spot"].path / "data.parquet"),
        previous_spot=pl.read_parquet(artifacts["previous_spot"].path / "data.parquet"),
        forwards=pl.read_parquet(artifacts["forwards"].path / "data.parquet"),
        zero_curve=pl.read_parquet(artifacts["zero_curve"].path / "data.parquet"),
    )
    frame = pair_daily_frame(
        ticker=ticker,
        trading_date=trading_date,
        raw=inputs,
        config=config,
        provenance=DailyProvenance(raw.options.artifact_id),
    )
    return publish_daily_frame(
        store,
        spec,
        frame,
        schema=PAIRED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )


def _build_enriched(
    store: ArtifactStore,
    daily_config: DailyWidbConfig,
    raw: RawDayPointers,
    paired_specification: ArtifactSpec,
    enriched_specification: ArtifactSpec,
    *,
    ticker: str,
    trading_date: date,
) -> ValidatedArtifact:
    reuse = store.resolve(enriched_specification)
    if reuse is not None:
        return reuse.artifact
    paired = _build_paired(
        store,
        daily_config,
        raw,
        paired_specification,
        ticker=ticker,
        trading_date=trading_date,
    )
    paired_frame = read_daily_frame(
        store,
        paired,
        schema=PAIRED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )
    return publish_daily_frame(
        store,
        enriched_specification,
        enrich_daily_frame(paired_frame, config=daily_config),
        schema=ENRICHED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )


def _build_no_arb(
    store: ArtifactStore,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig,
    raw: RawDayPointers,
    paired_specification: ArtifactSpec,
    enriched_specification: ArtifactSpec,
    no_arb_specification: ArtifactSpec,
    *,
    ticker: str,
    trading_date: date,
) -> ValidatedArtifact:
    reuse = store.resolve(no_arb_specification)
    if reuse is not None:
        return reuse.artifact
    enriched = _build_enriched(
        store,
        daily_config,
        raw,
        paired_specification,
        enriched_specification,
        ticker=ticker,
        trading_date=trading_date,
    )
    frame = read_daily_frame(
        store,
        enriched,
        schema=ENRICHED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )
    return publish_daily_frame(
        store,
        no_arb_specification,
        fit_initial_no_arbitrage(frame, config=surface_config),
        schema=NO_ARB_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )


def _build_filtered(
    store: ArtifactStore,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig,
    raw: RawDayPointers,
    paired_specification: ArtifactSpec,
    enriched_specification: ArtifactSpec,
    no_arb_specification: ArtifactSpec,
    filtered_specification: ArtifactSpec,
    *,
    ticker: str,
    trading_date: date,
) -> ValidatedArtifact:
    reuse = store.resolve(filtered_specification)
    if reuse is not None:
        return reuse.artifact
    no_arb = _build_no_arb(
        store,
        daily_config,
        surface_config,
        raw,
        paired_specification,
        enriched_specification,
        no_arb_specification,
        ticker=ticker,
        trading_date=trading_date,
    )
    frame = read_daily_frame(
        store,
        no_arb,
        schema=NO_ARB_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )
    return publish_daily_frame(
        store,
        filtered_specification,
        apply_options_filter(frame, config=surface_config),
        schema=FILTERED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )


def _build_fitted(
    store: ArtifactStore,
    daily_config: DailyWidbConfig,
    surface_config: DailySurfaceConfig,
    raw: RawDayPointers,
    paired_specification: ArtifactSpec,
    enriched_specification: ArtifactSpec,
    no_arb_specification: ArtifactSpec,
    filtered_specification: ArtifactSpec,
    fitted_specification: ArtifactSpec,
    *,
    ticker: str,
    trading_date: date,
) -> ValidatedArtifact:
    reuse = store.resolve(fitted_specification)
    if reuse is not None:
        return reuse.artifact
    filtered = _build_filtered(
        store,
        daily_config,
        surface_config,
        raw,
        paired_specification,
        enriched_specification,
        no_arb_specification,
        filtered_specification,
        ticker=ticker,
        trading_date=trading_date,
    )
    frame = read_daily_frame(
        store,
        filtered,
        schema=FILTERED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )
    return publish_daily_frame(
        store,
        fitted_specification,
        fit_smooth_sanos(frame, config=surface_config),
        schema=FITTED_SCHEMA,
        ticker=ticker,
        trading_date=trading_date,
    )


def build_daily_target(item: DailyWorkItem) -> ArtifactPointer:
    """Spawn-safe worker that recursively builds one terminal daily stage."""

    (
        store,
        daily_config,
        surface_config,
        raw,
        paired_specification,
        enriched_specification,
        no_arb_specification,
        filtered_specification,
        fitted_specification,
    ) = _specs_from_item(item)
    ticker = _actual_ticker(item)
    target = _target_spec(
        item,
        paired_specification,
        enriched_specification,
        no_arb_specification,
        filtered_specification,
        fitted_specification,
    )
    payload = item.payload
    if payload.get("version_override") == "*":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reuse = store.resolve(target, version="*")
        return reuse.artifact.pointer
    reuse = store.resolve(target)
    if reuse is not None:
        return reuse.artifact.pointer
    if item.stage == "paired-day":
        return _build_paired(
            store,
            daily_config,
            raw,
            paired_specification,
            ticker=ticker,
            trading_date=item.trading_date,
        ).pointer
    if item.stage == "enriched-day":
        return _build_enriched(
            store,
            daily_config,
            raw,
            paired_specification,
            enriched_specification,
            ticker=ticker,
            trading_date=item.trading_date,
        ).pointer
    if item.stage == "no-arb-day":
        return _build_no_arb(
            store,
            daily_config,
            surface_config,
            raw,
            paired_specification,
            enriched_specification,
            no_arb_specification,
            ticker=ticker,
            trading_date=item.trading_date,
        ).pointer
    if item.stage == "filtered-day":
        return _build_filtered(
            store,
            daily_config,
            surface_config,
            raw,
            paired_specification,
            enriched_specification,
            no_arb_specification,
            filtered_specification,
            ticker=ticker,
            trading_date=item.trading_date,
        ).pointer
    return _build_fitted(
        store,
        daily_config,
        surface_config,
        raw,
        paired_specification,
        enriched_specification,
        no_arb_specification,
        filtered_specification,
        fitted_specification,
        ticker=ticker,
        trading_date=item.trading_date,
    ).pointer


@version(
    "1.0",
    dependencies=[
        IntermediateBuildRequest,
        DailySurfaceConfig,
        DailyWidbConfig,
        plan_daily_artifacts,
        filtered_day_producer_version,
        fitted_day_producer_version,
        no_arb_day_producer_version,
        paired_day_producer_version,
        enriched_day_producer_version,
        raw_content_producer_version,
        plan_daily_dates,
    ],
)
def plan_intermediate_build(
    locations: StorageLocations,
    request: IntermediateBuildRequest,
    config: DailyWidbConfig,
    surface_config: DailySurfaceConfig = DailySurfaceConfig(),
) -> DailyPlan:
    """Create the deterministic per-session plan without touching artifact roots."""

    artifacts = plan_daily_artifacts(request, config, surface_config)
    dates = tuple(item.trading_date for item in artifacts)
    producer_by_stage = {
        "paired-day": paired_day_producer_version(),
        "enriched-day": enriched_day_producer_version(),
        "no-arb-day": no_arb_day_producer_version(),
        "filtered-day": filtered_day_producer_version(),
        "fitted-day": fitted_day_producer_version(),
    }
    schema_by_stage = {
        "paired-day": PAIRED_SCHEMA_VERSION,
        "enriched-day": ENRICHED_SCHEMA_VERSION,
        "no-arb-day": NO_ARB_SCHEMA_VERSION,
        "filtered-day": FILTERED_SCHEMA_VERSION,
        "fitted-day": FITTED_SCHEMA_VERSION,
    }
    payload = {
        "locations": locations.as_dict(),
        "daily_config": canonical_value(config),
        "ticker": request.ticker,
        "source": request.source,
        "source_generation": request.source_generation,
        "raw_producer_version": raw_content_producer_version(),
        "version_override": request.version_override,
    }
    if request.through in {"no-arb-day", "filtered-day", "fitted-day"}:
        payload["surface_config"] = canonical_value(surface_config)
    return plan_daily_dates(
        ticker=_safe_daily_ticker_label(request.ticker),
        dates=dates,
        stage=request.through,
        payload=payload,
        schema_version=schema_by_stage[request.through],
        producer_version=producer_by_stage[request.through],
        max_workers=request.max_workers,
        progress_interval_seconds=request.progress_interval_seconds,
    )


def build_intermediates(
    locations: StorageLocations,
    request: IntermediateBuildRequest,
    config: DailyWidbConfig,
    surface_config: DailySurfaceConfig = DailySurfaceConfig(),
    *,
    dry_run: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> IntermediateBuildResult:
    """Inspect or execute a restartable output-first daily build."""

    plan = plan_intermediate_build(locations, request, config, surface_config)
    expected: list[ArtifactPointer] = []
    existing: list[ArtifactPointer] = []
    reuses: list[ReuseResult] = []
    resolved_by_key: dict[str, ArtifactPointer] = {}
    raw_dependencies: list[RawDayPointers] = []
    for item in plan.items:
        _, _, _, raw, paired, enriched, no_arb, filtered, fitted = _specs_from_item(item)
        raw_dependencies.append(raw)
        target = _target_spec(item, paired, enriched, no_arb, filtered, fitted)
        expected.append(target.pointer)
        payload = item.payload
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reuse = ArtifactStore(_locations_from_payload(payload)).resolve(
                target,
                version=payload.get("version_override"),
            )
        if reuse is not None:
            existing.append(reuse.artifact.pointer)
            reuses.append(reuse)
            resolved_by_key[item.key] = reuse.artifact.pointer
    if dry_run:
        terminal = tuple(
            resolved_by_key.get(item.key, expected[index]) for index, item in enumerate(plan.items)
        )
        return IntermediateBuildResult(
            plan=plan,
            expected=tuple(expected),
            terminal=terminal,
            existing=tuple(existing),
            reuses=tuple(reuses),
            raw_dependencies=tuple(raw_dependencies),
            dry_run=True,
        )
    journal = locations.intermediate / ".runs" / "widb" / f"{plan.plan_id}.jsonl"
    summary = run_daily_plan(
        plan,
        build_daily_target,
        journal_path=journal,
        is_complete=probe_daily_target,
        progress=progress,
    )
    results = dict(summary.results)
    missing_results = tuple(item.key for item in plan.items if results.get(item.key) is None)
    if missing_results:
        raise RuntimeError(
            "Daily intermediate workers returned no terminal artifact for: "
            + ", ".join(missing_results)
        )
    terminal = tuple(results[item.key] for item in plan.items if results[item.key] is not None)
    return IntermediateBuildResult(
        plan=plan,
        expected=tuple(expected),
        terminal=tuple(terminal),
        existing=tuple(existing),
        reuses=tuple(reuses),
        raw_dependencies=tuple(raw_dependencies),
        dry_run=False,
        summary=summary,
    )


__all__ = [
    "IntermediateBuildRequest",
    "IntermediateBuildResult",
    "DailyArtifactPlan",
    "RawDayPointers",
    "build_daily_target",
    "build_intermediates",
    "enriched_day_producer_version",
    "enriched_day_spec",
    "filtered_day_producer_version",
    "filtered_day_spec",
    "fitted_day_producer_version",
    "fitted_day_spec",
    "no_arb_day_producer_version",
    "no_arb_day_spec",
    "paired_day_producer_version",
    "paired_day_spec",
    "plan_intermediate_build",
    "plan_daily_artifact",
    "plan_daily_artifacts",
    "plan_raw_day_pointers",
    "probe_daily_target",
    "publish_daily_frame",
    "read_daily_frame",
]
