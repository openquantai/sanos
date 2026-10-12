"""Human- and automation-facing SANOS command hierarchy."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from typing import Any

import click

from sanos import __version__


@dataclass(frozen=True)
class CliContext:
    """Global presentation settings shared by subcommands."""

    verbose: int
    log_format: str


def _emit(context: CliContext, event: str, **fields: Any) -> None:
    """Emit one stable event in text or line-delimited JSON form."""

    payload = {"event": event, **fields}
    if context.log_format == "json":
        click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    click.echo(f"{event}{' ' if details else ''}{details}")


def _phase_not_implemented(command: str) -> None:
    raise click.ClickException(
        f"'{command}' is not implemented in the Phase 0 bootstrap; "
        "see PROGRAM_PLAN.md for its delivery phase"
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sanos")
@click.option("-v", "--verbose", count=True, help="Increase diagnostic output.")
@click.option(
    "--log-format",
    type=click.Choice(("text", "json"), case_sensitive=False),
    default="text",
    show_default=True,
    help="Select human-readable or structured output.",
)
@click.pass_context
def main(context: click.Context, verbose: int, log_format: str) -> None:
    """Build, validate, and inspect SANOS data products."""

    context.obj = CliContext(verbose=verbose, log_format=log_format.lower())


@main.command()
@click.pass_obj
def doctor(context: CliContext) -> None:
    """Inspect the installation without opening network connections."""

    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("cdxcore", "click", "cvxpy", "highspy", "numpy", "polars", "pyarrow")
    }
    ok = all(modules.values()) and sys.version_info >= (3, 12)
    _emit(
        context,
        "doctor",
        status="ok" if ok else "error",
        sanos_version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        network_accessed=False,
        modules=modules,
    )
    if not ok:
        raise click.exceptions.Exit(1)


@main.command()
@click.pass_obj
def status(context: CliContext) -> None:
    """Report the implemented repository phase without network access."""

    _emit(
        context,
        "status",
        phase="bootstrap",
        pipeline_available=False,
        network_accessed=False,
    )


@main.group()
def raw() -> None:
    """Plan, download, and validate durable vendor extracts."""


@raw.command("download")
def raw_download() -> None:
    """Download raw data (enabled after the guarded WRDS phase)."""

    _phase_not_implemented("raw download")


@main.group()
def intermediate() -> None:
    """Build and validate reproducible per-stage artifacts."""


@intermediate.command("build")
def intermediate_build() -> None:
    """Build intermediate artifacts."""

    _phase_not_implemented("intermediate build")


@main.group()
def dataset() -> None:
    """Build, publish, and validate immutable research datasets."""


@dataset.command("build")
def dataset_build() -> None:
    """Build and publish a dataset."""

    _phase_not_implemented("dataset build")


@dataset.command("validate")
def dataset_validate() -> None:
    """Validate an existing dataset."""

    _phase_not_implemented("dataset validate")


@main.group()
def experiment() -> None:
    """Build and validate disposable experiment materializations."""


@experiment.command("build")
def experiment_build() -> None:
    """Build an experiment materialization."""

    _phase_not_implemented("experiment build")
