"""Human- and automation-facing SANOS command hierarchy."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from sanos import __version__
from sanos.config import (
    LocationConfigurationError,
    LocationOverrides,
    LocationResolution,
    canonical_value,
    resolve_locations,
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
            "polars",
            "pyarrow",
        )
    }
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
        phase="configuration-artifacts",
        pipeline_available=True,
        material_pipeline_available=False,
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
def raw_download() -> None:
    """Download raw data (enabled after the guarded WRDS phase)."""

    _phase_not_implemented("raw download", "Phase 4")


@main.group()
def intermediate() -> None:
    """Build and validate reproducible per-stage artifacts."""


@intermediate.command("build")
def intermediate_build() -> None:
    """Build intermediate artifacts."""

    _phase_not_implemented("intermediate build", "Phases 5-6")


@main.group()
def dataset() -> None:
    """Build, publish, and validate immutable research datasets."""


@dataset.command("build")
def dataset_build() -> None:
    """Build and publish a dataset."""

    _phase_not_implemented("dataset build", "Phase 7")


@dataset.command("validate")
def dataset_validate() -> None:
    """Validate an existing dataset."""

    _phase_not_implemented("dataset validate", "Phase 7")


@main.group()
def experiment() -> None:
    """Build and validate disposable experiment materializations."""


@experiment.command("build")
def experiment_build() -> None:
    """Build an experiment materialization."""

    _phase_not_implemented("experiment build", "Phase 9")
