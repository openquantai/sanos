"""Human- and automation-facing SANOS command hierarchy."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from sanos import __version__
from sanos.artifacts import ArtifactError, ArtifactPointer, ArtifactStore, StorageClass
from sanos.config import (
    LocationConfigurationError,
    LocationOverrides,
    LocationResolution,
    canonical_sha256,
    canonical_value,
    resolve_locations,
)
from sanos.datasets import (
    DatasetBuildOptions,
    InternalDatasetConfig,
    build_internal_dataset,
    plan_internal_dataset,
    validate_internal_dataset,
)
from sanos.pipeline import DailyExecutionError
from sanos.widb.build import IntermediateBuildRequest, build_intermediates
from sanos.widb.config import DailySurfaceConfig, DailyWidbConfig
from sanos.wrds import (
    ALL_RAW_TABLES,
    RawDownloadConfig,
    RawDownloadError,
    RawTable,
    WrdsError,
    WrdsLeaseUnavailableError,
    download_raw,
)


class ConfigurationCliError(click.ClickException):
    """Configuration failure with a stable process exit code."""

    exit_code = 3

    def __init__(self, message: str, *, log_format: str | None = None) -> None:
        super().__init__(message)
        self.log_format = log_format

    def show(self, file: Any = None) -> None:
        _show_cli_error(self, log_format=self.log_format, file=file)


class PhaseNotImplementedError(click.ClickException):
    """Known command whose implementation belongs to a later program phase."""

    exit_code = 7

    def __init__(self, message: str, *, log_format: str | None = None) -> None:
        super().__init__(message)
        self.log_format = log_format

    def show(self, file: Any = None) -> None:
        _show_cli_error(self, log_format=self.log_format, file=file)


class MaterialCliError(click.ClickException):
    """A material command failed safely with an actionable message."""

    exit_code = 4

    def __init__(self, message: str, *, log_format: str | None = None) -> None:
        super().__init__(message)
        self.log_format = log_format

    def show(self, file: Any = None) -> None:
        _show_cli_error(self, log_format=self.log_format, file=file)


@dataclass(frozen=True)
class CliContext:
    """Global presentation and resolved-location settings."""

    verbose: int
    log_format: str
    dry_run: bool
    locations: LocationResolution


def _json_requested(arguments: Sequence[str]) -> bool:
    return "--log-format=json" in arguments or any(
        arguments[index] == "--log-format" and arguments[index + 1] == "json"
        for index in range(len(arguments) - 1)
    )


def _show_cli_error(
    error: click.ClickException,
    *,
    log_format: str | None,
    file: Any = None,
) -> None:
    if log_format == "json":
        click.echo(
            json.dumps(
                {
                    "event": "error",
                    "error_type": type(error).__name__,
                    "exit_code": error.exit_code,
                    "message": error.format_message(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=file,
            err=True,
        )
        return
    click.ClickException.show(error, file=file)


class SanosGroup(click.Group):
    """Root command that preserves structured errors across Click parsing."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        arguments = list(sys.argv[1:] if args is None else args)
        log_format = "json" if _json_requested(arguments) else None
        try:
            result = super().main(
                args=arguments,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )
        except click.ClickException as exc:
            if not standalone_mode:
                raise
            _show_cli_error(exc, log_format=log_format)
            raise SystemExit(exc.exit_code) from exc
        except click.Abort as exc:
            if not standalone_mode:
                raise
            error = click.ClickException("Aborted")
            _show_cli_error(error, log_format=log_format)
            raise SystemExit(1) from exc
        if standalone_mode and isinstance(result, int):
            raise SystemExit(result)
        return result


def _emit(context: CliContext, event: str, **fields: Any) -> None:
    """Emit one stable event in text or line-delimited JSON form."""

    payload = canonical_value({"event": event, **fields})
    if context.log_format == "json":
        click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    details = " ".join(f"{key}={value}" for key, value in payload.items() if key != "event")
    click.echo(f"{event}{' ' if details else ''}{details}")


def _phase_not_implemented(command: str, phase: str) -> None:
    context = click.get_current_context().find_root().obj
    raise PhaseNotImplementedError(
        f"'{command}' is planned for {phase} and is not implemented yet; "
        "see PROGRAM_PLAN.md for its acceptance gate",
        log_format=context.log_format,
    )


def _location_payload(resolution: LocationResolution) -> dict[str, Any]:
    return {
        name: {
            "path": None if path is None else str(path),
            "source": resolution.source_for(name),
            "exists": False if path is None else path.exists(),
        }
        for name, path in resolution.as_dict().items()
    }


def _read_dataset_config(path: Path) -> InternalDatasetConfig:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Dataset configuration {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Dataset configuration must contain a JSON object")
    return InternalDatasetConfig.from_document(document)


@click.group(cls=SanosGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sanos")
@click.option("-v", "--verbose", count=True, help="Increase diagnostic output.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Plan and validate without material writes or network access.",
)
@click.option(
    "--log-format",
    type=click.Choice(("text", "json"), case_sensitive=False),
    default="text",
    show_default=True,
    help="Select human-readable or structured line output.",
)
@click.option("--root", type=click.Path(path_type=Path), help="Override all four data roots.")
@click.option("--raw-root", type=click.Path(path_type=Path), help="Override the Raw root.")
@click.option(
    "--intermediate-root",
    type=click.Path(path_type=Path),
    help="Override the Intermediate root.",
)
@click.option("--datasets-root", type=click.Path(path_type=Path), help="Override Dataset root.")
@click.option(
    "--experiments-root", type=click.Path(path_type=Path), help="Override Experiment root."
)
@click.option(
    "--repository-config",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Use this repository-local TOML configuration explicitly.",
)
@click.option(
    "--user-config",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Use this user TOML configuration explicitly.",
)
@click.pass_context
def main(
    context: click.Context,
    verbose: int,
    dry_run: bool,
    log_format: str,
    root: Path | None,
    raw_root: Path | None,
    intermediate_root: Path | None,
    datasets_root: Path | None,
    experiments_root: Path | None,
    repository_config: Path | None,
    user_config: Path | None,
) -> None:
    """Build, validate, and inspect SANOS data products."""

    try:
        locations = resolve_locations(
            LocationOverrides(
                root=root,
                raw=raw_root,
                intermediate=intermediate_root,
                datasets=datasets_root,
                experiments=experiments_root,
            ),
            repository_config=repository_config,
            user_config=user_config,
        )
    except LocationConfigurationError as exc:
        raise ConfigurationCliError(str(exc), log_format=log_format.lower()) from exc
    context.obj = CliContext(
        verbose=verbose,
        log_format=log_format.lower(),
        dry_run=dry_run,
        locations=locations,
    )


@main.command()
@click.option(
    "--strict-locations",
    is_flag=True,
    help="Fail unless all four storage roots resolve.",
)
@click.pass_obj
def doctor(context: CliContext, strict_locations: bool) -> None:
    """Inspect installation and configuration without network access or writes."""

    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "cdxcore",
            "click",
            "cvxpy",
            "filelock",
            "highspy",
            "numpy",
            "pandas_market_calendars",
            "polars",
            "pyarrow",
        )
    }
    optional_modules = {"wrds": importlib.util.find_spec("wrds") is not None}
    locations_ok = not context.locations.missing
    ok = all(modules.values()) and sys.version_info >= (3, 12)
    if strict_locations:
        ok = ok and locations_ok
    _emit(
        context,
        "doctor",
        status="ok" if ok else "error",
        sanos_version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        network_accessed=False,
        filesystem_modified=False,
        dry_run=context.dry_run,
        verbosity=context.verbose,
        modules=modules,
        optional_modules=optional_modules,
        locations=_location_payload(context.locations),
        missing_locations=context.locations.missing,
        warnings=context.locations.warnings,
    )
    if not ok:
        raise click.exceptions.Exit(1)


@main.command()
@click.pass_obj
def status(context: CliContext) -> None:
    """Report implemented capabilities without network access or writes."""

    _emit(
        context,
        "status",
        phase="daily-widb-intermediate",
        pipeline_available=True,
        material_pipeline_available=True,
        raw_download_available=True,
        intermediate_build_available=True,
        dataset_build_available=True,
        dataset_validate_available=True,
        live_wrds_verified=False,
        network_accessed=False,
        dry_run=context.dry_run,
        verbosity=context.verbose,
        locations=_location_payload(context.locations),
        warnings=context.locations.warnings,
    )


@main.group(name="config")
def config_group() -> None:
    """Inspect resolved local configuration."""


@config_group.command(name="show")
@click.option("--require-complete", is_flag=True, help="Fail if any storage root is unresolved.")
@click.pass_obj
def config_show(context: CliContext, require_complete: bool) -> None:
    """Show location precedence results without creating directories."""

    if require_complete:
        try:
            context.locations.require_complete()
        except LocationConfigurationError as exc:
            raise ConfigurationCliError(str(exc), log_format=context.log_format) from exc
    _emit(
        context,
        "config",
        locations=_location_payload(context.locations),
        missing_locations=context.locations.missing,
        repository_config=context.locations.repository_config,
        user_config=context.locations.user_config,
        warnings=context.locations.warnings,
        filesystem_modified=False,
        dry_run=context.dry_run,
        verbosity=context.verbose,
    )


@main.group()
def raw() -> None:
    """Plan, download, and validate durable vendor extracts."""


@raw.command("download")
@click.option("--source", type=click.Choice(("optionmetrics",)), default="optionmetrics")
@click.option("--ticker", required=True, help="OptionMetrics ticker, for example SPX.")
@click.option(
    "--start-date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First requested NYSE session (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last requested NYSE session (YYYY-MM-DD).",
)
@click.option(
    "--table",
    "tables",
    multiple=True,
    type=click.Choice(tuple(table.value for table in ALL_RAW_TABLES)),
    default=tuple(table.value for table in ALL_RAW_TABLES),
    show_default=True,
    help="Logical raw table; repeat to select a subset. Security identity is always included.",
)
@click.option(
    "--source-generation",
    default="default-v1",
    show_default=True,
    help="Immutable vendor extraction generation included in artifact identity.",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Download a new explicitly named source generation; never replace an artifact.",
)
@click.option(
    "--cache-only",
    is_flag=True,
    help="Validate requested raw artifacts and fail on a miss without opening WRDS.",
)
@click.option(
    "--version",
    "version_request",
    type=click.Choice(("current", "*")),
    default="current",
    show_default=True,
    help="Use '*' only to reuse an existing complete artifact across producer versions.",
)
@click.option(
    "--max-wrds-connections",
    type=click.IntRange(1, 4),
    default=2,
    show_default=True,
    help="Bound concurrent account connections; the host-wide maximum is four.",
)
@click.option(
    "--retry-attempts",
    type=click.IntRange(1, 5),
    default=3,
    show_default=True,
    help="Attempts for recognized transient queries; authentication is never retried.",
)
@click.option(
    "--confirm-2fa-ready",
    is_flag=True,
    help="Required for non-interactive live access; confirms immediate WRDS 2FA readiness.",
)
@click.pass_obj
def raw_download(
    context: CliContext,
    source: str,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    tables: tuple[str, ...],
    source_generation: str,
    refresh: bool,
    cache_only: bool,
    version_request: str,
    max_wrds_connections: int,
    retry_attempts: int,
    confirm_2fa_ready: bool,
) -> None:
    """Plan, validate, or download durable raw OptionMetrics partitions."""

    try:
        locations = context.locations.require_complete()
        config = RawDownloadConfig(
            source=source,
            ticker=ticker,
            start_date=start_date.date(),
            end_date=end_date.date(),
            tables=tuple(RawTable(value) for value in tables),
            source_generation=source_generation,
            max_connections=max_wrds_connections,
            retry_attempts=retry_attempts,
        )

        def authorize_network() -> None:
            if confirm_2fa_ready:
                _emit(context, "wrds_2fa_confirmed", interactive=False)
                return
            if not sys.stdin.isatty():
                raise MaterialCliError(
                    "Live WRDS access from a non-interactive caller requires --confirm-2fa-ready",
                    log_format=context.log_format,
                )
            if not click.confirm(
                "A raw cache miss requires WRDS. Are you ready to complete 2FA now?",
                default=False,
            ):
                raise MaterialCliError(
                    "WRDS access cancelled before connecting",
                    log_format=context.log_format,
                )
            _emit(context, "wrds_2fa_confirmed", interactive=True)

        def progress(payload: Any) -> None:
            event = str(payload["event"])
            fields = {key: value for key, value in payload.items() if key != "event"}
            _emit(context, event, **fields)

        result = download_raw(
            ArtifactStore(locations),
            config,
            dry_run=context.dry_run,
            cache_only=cache_only,
            refresh=refresh,
            version_override="*" if version_request == "*" else None,
            authorize_network=authorize_network,
            progress=progress,
        )
    except LocationConfigurationError as exc:
        raise ConfigurationCliError(str(exc), log_format=context.log_format) from exc
    except (
        ArtifactError,
        RawDownloadError,
        WrdsError,
        WrdsLeaseUnavailableError,
        ValueError,
    ) as exc:
        if isinstance(exc, click.ClickException):
            raise
        raise MaterialCliError(str(exc), log_format=context.log_format) from exc
    _emit(
        context,
        "raw_result",
        planned=result.planned,
        reused=result.reused,
        downloaded=result.downloaded,
        missing=result.missing,
        network_accessed=result.network_accessed,
        dry_run=result.dry_run,
        artifact_paths=tuple(artifact.path for artifact in result.artifacts),
    )


@main.group()
def intermediate() -> None:
    """Build and validate reproducible per-stage artifacts."""


@intermediate.command("build")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    help=(
        "Strict portable JSON containing 'daily' and an optional 'surface' "
        "configuration object."
    ),
)
@click.option("--ticker", required=True, help="OptionMetrics ticker, for example SPX.")
@click.option(
    "--start-date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First requested NYSE session (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last requested NYSE session (YYYY-MM-DD).",
)
@click.option(
    "--through",
    type=click.Choice(
        ("paired-day", "enriched-day", "no-arb-day", "filtered-day", "fitted-day")
    ),
    default="enriched-day",
    show_default=True,
)
@click.option(
    "--include-half-days/--exclude-half-days",
    default=True,
    show_default=True,
    help="Include NYSE early-close sessions in the requested date range.",
)
@click.option("--source", type=click.Choice(("optionmetrics",)), default="optionmetrics")
@click.option(
    "--source-generation",
    default="default-v1",
    show_default=True,
    help="Raw source generation that the intermediates must consume.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(1),
    default=1,
    show_default=True,
    help="Bound local per-day worker processes.",
)
@click.option(
    "--version",
    "version_request",
    type=click.Choice(("current", "*")),
    default="current",
    show_default=True,
    help="Use '*' only to reuse an existing complete terminal artifact.",
)
@click.pass_obj
def intermediate_build(
    context: CliContext,
    config_path: Path,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    through: str,
    include_half_days: bool,
    source: str,
    source_generation: str,
    max_workers: int,
    version_request: str,
) -> None:
    """Build a restartable daily WIDB/filter/SANOS stage offline from Raw."""

    try:
        locations = context.locations.require_complete()
        try:
            document = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read intermediate configuration {config_path}: {exc}"
            ) from exc
        if not isinstance(document, dict) or "daily" not in document:
            raise ValueError("Intermediate configuration must contain a 'daily' object")
        unknown = set(document) - {"dataset", "daily", "surface"}
        if unknown:
            raise ValueError(f"Unknown intermediate configuration sections: {sorted(unknown)}")
        if not isinstance(document["daily"], dict):
            raise ValueError("Intermediate configuration 'daily' must be an object")
        daily = DailyWidbConfig.from_mapping(document["daily"])
        surface_payload = document.get("surface", {})
        if not isinstance(surface_payload, dict):
            raise ValueError("Intermediate configuration 'surface' must be an object")
        surface = DailySurfaceConfig.from_mapping(surface_payload)
        request = IntermediateBuildRequest(
            ticker=ticker,
            start_date=start_date.date(),
            end_date=end_date.date(),
            through=through,
            include_half_days=include_half_days,
            source=source,
            source_generation=source_generation,
            max_workers=max_workers,
            version_override="*" if version_request == "*" else None,
        )

        def progress(payload: Any) -> None:
            _emit(context, "intermediate_progress", **dict(payload))

        result = build_intermediates(
            locations,
            request,
            daily,
            surface,
            dry_run=context.dry_run,
            progress=progress,
        )
    except LocationConfigurationError as exc:
        raise ConfigurationCliError(str(exc), log_format=context.log_format) from exc
    except DailyExecutionError as exc:
        details = "; ".join(
            f"{failure.key}: {failure.error_type}: {failure.message}"
            for failure in exc.summary.failures
        )
        raise MaterialCliError(
            f"{exc}{'; ' if details else ''}{details}",
            log_format=context.log_format,
        ) from exc
    except (ArtifactError, OSError, ValueError) as exc:
        raise MaterialCliError(str(exc), log_format=context.log_format) from exc
    store = ArtifactStore(locations)
    summary = result.summary
    planned_journal = locations.intermediate / ".runs" / "widb" / f"{result.plan.plan_id}.jsonl"
    raw_paths = tuple(
        tuple(store.path_for(pointer) for pointer in dependencies.all())
        for dependencies in result.raw_dependencies
    )
    reported_configuration: Any
    if through in {"paired-day", "enriched-day"}:
        reported_configuration = daily
    else:
        reported_configuration = {"daily": daily, "surface": surface}
    _emit(
        context,
        "intermediate_result",
        stage=through,
        config_id=canonical_sha256(reported_configuration),
        configuration=canonical_value(reported_configuration),
        plan_id=result.plan.plan_id,
        total=len(result.expected),
        existing=len(result.existing),
        built=0 if summary is None else summary.completed,
        reused=len(result.existing) if summary is None else summary.reused,
        failed=0 if summary is None else summary.failed,
        dry_run=result.dry_run,
        network_accessed=False,
        version_override=version_request == "*",
        journal=planned_journal if summary is None else summary.journal_path,
        previous_spot_paths=tuple(
            store.path_for(dependencies.previous_spot) for dependencies in result.raw_dependencies
        ),
        raw_dependency_count=sum(len(dependencies) for dependencies in raw_paths),
        raw_dependency_paths=raw_paths if context.verbose else (),
        artifact_paths=tuple(store.path_for(pointer) for pointer in result.terminal),
        expected_artifact_paths=tuple(store.path_for(pointer) for pointer in result.expected),
        reuse_records=tuple(reuse.as_log_record() for reuse in result.reuses),
        warnings=tuple(reuse.warning for reuse in result.reuses if reuse.warning),
    )


@main.group()
def dataset() -> None:
    """Build, publish, and validate immutable research datasets."""


@dataset.command("build")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    help="Strict portable JSON containing dataset, daily, and surface sections.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(1, 4),
    default=1,
    show_default=True,
    help="Bound local per-day fitting workers; no WRDS connections are opened.",
)
@click.option(
    "--require-existing-intermediates",
    is_flag=True,
    help="Fail on a missing fitted day instead of rebuilding it from existing Raw data.",
)
@click.option(
    "--version",
    "version_request",
    type=click.Choice(("current", "*")),
    default="current",
    show_default=True,
    help="Use '*' only to reuse an existing complete Dataset across producer versions.",
)
@click.pass_obj
def dataset_build(
    context: CliContext,
    config_path: Path,
    max_workers: int,
    require_existing_intermediates: bool,
    version_request: str,
) -> None:
    """Build and atomically publish a self-contained internal Dataset."""

    try:
        locations = context.locations.require_complete()
        config = _read_dataset_config(config_path)

        def progress(payload: Any) -> None:
            event = str(payload.get("event", "dataset_progress"))
            _emit(
                context,
                event,
                **{key: value for key, value in payload.items() if key != "event"},
            )

        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            result = build_internal_dataset(
                locations,
                config,
                DatasetBuildOptions(
                    max_workers=max_workers,
                    require_existing_intermediates=require_existing_intermediates,
                    version_override="*" if version_request == "*" else None,
                ),
                dry_run=context.dry_run,
                progress=progress,
            )
    except LocationConfigurationError as exc:
        raise ConfigurationCliError(str(exc), log_format=context.log_format) from exc
    except DailyExecutionError as exc:
        details = "; ".join(
            f"{failure.key}: {failure.error_type}: {failure.message}"
            for failure in exc.summary.failures
        )
        raise MaterialCliError(
            f"{exc}{'; ' if details else ''}{details}",
            log_format=context.log_format,
        ) from exc
    except (ArtifactError, OSError, RuntimeError, ValueError) as exc:
        raise MaterialCliError(str(exc), log_format=context.log_format) from exc
    for warning_record in warning_records:
        _emit(
            context,
            "warning",
            category=warning_record.category.__name__,
            message=str(warning_record.message),
        )
    _emit(
        context,
        "dataset_result",
        artifact=result.artifact,
        path=result.path,
        identity=result.artifact.artifact_id,
        plan_id=result.plan.plan_id,
        planned_dates=len(result.plan.expected_dates),
        built=result.built,
        reused=result.reuse is not None,
        reuse_record=None if result.reuse is None else result.reuse.as_log_record(),
        dry_run=result.dry_run,
        network_accessed=False,
        journal=result.journal_path,
        validation=None if result.validation is None else result.validation.as_dict(),
    )


@dataset.command("validate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    help="Resolve the Dataset identity from a strict portable configuration.",
)
@click.option("--name", help="Dataset name for exact-ID validation.")
@click.option("--artifact-id", help="Exact 64-hex Dataset artifact identity.")
@click.option(
    "--version",
    "version_request",
    type=click.Choice(("current", "*")),
    default="current",
    show_default=True,
    help="Use '*' only with --config to find an existing compatible Dataset.",
)
@click.option(
    "--deep-lineage",
    is_flag=True,
    help="Also require the recorded fitted-day, Raw catalog, and Raw lineage.",
)
@click.option(
    "--integrity-only",
    is_flag=True,
    help="Check immutable files and manifest without recomputing quantitative invariants.",
)
@click.pass_obj
def dataset_validate(
    context: CliContext,
    config_path: Path | None,
    name: str | None,
    artifact_id: str | None,
    version_request: str,
    deep_lineage: bool,
    integrity_only: bool,
) -> None:
    """Validate an existing Dataset without rebuilding or network access."""

    try:
        locations = context.locations.require_complete()
        store = ArtifactStore(locations)
        warning_records: Sequence[warnings.WarningMessage] = ()
        if config_path is not None:
            if name is not None or artifact_id is not None:
                raise ValueError("Use either --config or --name with --artifact-id, not both")
            config = _read_dataset_config(config_path)
            plan = plan_internal_dataset(config)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                reuse = store.resolve(
                    plan.dataset_spec,
                    version="*" if version_request == "*" else None,
                )
            warning_records = tuple(caught_warnings)
            if reuse is None:
                raise ValueError(
                    "Requested Dataset is absent; validation never builds or publishes it"
                )
            pointer = reuse.artifact.pointer
            reuse_record: Any = reuse.as_log_record()
        else:
            if name is None or artifact_id is None:
                raise ValueError("Provide --config or both --name and --artifact-id")
            if version_request == "*":
                raise ValueError("--version '*' requires --config; exact-ID validation is concrete")
            config = None
            pointer = ArtifactPointer(
                StorageClass.DATASET,
                f"widb/internal/name-{name}",
                artifact_id,
            )
            reuse_record = None

        def progress(payload: Any) -> None:
            event = str(payload.get("event", "dataset_validation_progress"))
            _emit(
                context,
                event,
                **{key: value for key, value in payload.items() if key != "event"},
            )

        summary = validate_internal_dataset(
            store,
            pointer,
            semantic=not integrity_only,
            deep_lineage=deep_lineage,
            progress=progress,
        )
    except LocationConfigurationError as exc:
        raise ConfigurationCliError(str(exc), log_format=context.log_format) from exc
    except (ArtifactError, OSError, RuntimeError, ValueError) as exc:
        raise MaterialCliError(str(exc), log_format=context.log_format) from exc
    for warning_record in warning_records:
        _emit(
            context,
            "warning",
            category=warning_record.category.__name__,
            message=str(warning_record.message),
        )
    _emit(
        context,
        "dataset_validation_result",
        status="passed",
        artifact=pointer,
        path=store.path_for(pointer),
        identity=pointer.artifact_id,
        semantic=not integrity_only,
        deep_lineage=deep_lineage,
        network_accessed=False,
        filesystem_modified=False,
        config=None if config is None else canonical_value(config),
        reuse_record=reuse_record,
        summary=None if summary is None else summary.as_dict(),
    )


@main.group()
def experiment() -> None:
    """Build and validate disposable experiment materializations."""


@experiment.command("build")
def experiment_build() -> None:
    """Build an experiment materialization."""

    _phase_not_implemented("experiment build", "Phase 9")
