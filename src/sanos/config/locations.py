"""Resolve the four SANOS storage roots without creating directories."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from cdxcore.version import version


class LocationConfigurationError(ValueError):
    """Raised when location configuration is invalid or incomplete."""


LOCATION_NAMES: Final = ("raw", "intermediate", "datasets", "experiments")
_NEW_ENV: Final = {
    "raw": "QUANT_DATA_RAW",
    "intermediate": "QUANT_DATA_INTERMEDIATE",
    "datasets": "QUANT_DATA_DATASETS",
    "experiments": "QUANT_DATA_EXPERIMENTS",
}
_LEGACY_ENV: Final = {
    "raw": "QUANTDATA_RAW_DATA",
    "intermediate": "QUANTDATA_PROCESSING_DATA",
    "datasets": "QUANTDATA_TRAINING_DATA",
    "experiments": "QUANTDATA_EXPERIMENTS",
}
_NEW_ROOT_ENV: Final = "QUANT_DATA_ROOT"
_LEGACY_ROOT_ENV: Final = "QUANTDATA_ROOT"
_NEW_ROOT_CHILD: Final = {
    "raw": "raw",
    "intermediate": "intermediate",
    "datasets": "datasets",
    "experiments": "experiments",
}
_LEGACY_ROOT_CHILD: Final = {
    "raw": "raw",
    "intermediate": "processing",
    "datasets": "training",
    "experiments": "experiments",
}
_CONFIG_KEYS: Final = frozenset({"root", *LOCATION_NAMES})


@version("1.0")
def _path(value: str | Path, *, base: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise LocationConfigurationError("A configured data location cannot be empty")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


@version("1.0", dependencies=[_path])
def _load_location_table(path: Path | None, *, required: bool) -> dict[str, Path]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.exists():
        if required:
            raise LocationConfigurationError(f"Configuration file does not exist: {resolved}")
        return {}
    if not resolved.is_file():
        raise LocationConfigurationError(f"Configuration path is not a file: {resolved}")
    try:
        document = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise LocationConfigurationError(f"Cannot read configuration {resolved}: {exc}") from exc
    table = document.get("locations", {})
    if not isinstance(table, dict):
        raise LocationConfigurationError(f"[locations] must be a TOML table in {resolved}")
    unknown = set(table) - _CONFIG_KEYS
    if unknown:
        raise LocationConfigurationError(
            f"Unknown [locations] keys in {resolved}: {', '.join(sorted(unknown))}"
        )
    result: dict[str, Path] = {}
    for key, value in table.items():
        if not isinstance(value, str):
            raise LocationConfigurationError(
                f"[locations].{key} must be a string path in {resolved}"
            )
        result[key] = _path(value, base=resolved.parent)
    return result


@version("1.0")
def _discover_repository_config(start: Path) -> Path | None:
    current = start.expanduser().resolve(strict=False)
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / ".sanos.toml"
        if candidate.is_file():
            return candidate.resolve(strict=False)
    return None


@version("1.0")
@dataclass(frozen=True, slots=True)
class LocationOverrides:
    """Explicit CLI location options, all of which outrank other sources."""

    root: Path | None = None
    raw: Path | None = None
    intermediate: Path | None = None
    datasets: Path | None = None
    experiments: Path | None = None


@version("1.0")
@dataclass(frozen=True, slots=True)
class StorageLocations:
    """Complete resolved roots required by artifact-producing commands."""

    raw: Path
    intermediate: Path
    datasets: Path
    experiments: Path

    def as_dict(self) -> dict[str, Path]:
        """Return roots keyed by the four canonical storage names."""

        return {name: getattr(self, name) for name in LOCATION_NAMES}


@version("1.0", dependencies=[StorageLocations])
@dataclass(frozen=True, slots=True)
class LocationResolution:
    """Possibly partial resolution used by diagnostics and status commands."""

    raw: Path | None
    intermediate: Path | None
    datasets: Path | None
    experiments: Path | None
    sources: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]
    repository_config: Path | None
    user_config: Path | None

    @property
    def missing(self) -> tuple[str, ...]:
        """Return canonical location names that could not be resolved."""

        return tuple(name for name in LOCATION_NAMES if getattr(self, name) is None)

    def source_for(self, name: str) -> str | None:
        """Return the winning source label for a canonical location name."""

        return dict(self.sources).get(name)

    def as_dict(self) -> dict[str, Path | None]:
        """Return all four roots in canonical order."""

        return {name: getattr(self, name) for name in LOCATION_NAMES}

    def require_complete(self) -> StorageLocations:
        """Return strict locations or raise an actionable configuration error."""

        if self.missing:
            variables = ", ".join(_NEW_ENV[name] for name in self.missing)
            raise LocationConfigurationError(
                "Missing data locations for "
                f"{', '.join(self.missing)}. Set explicit CLI paths, {variables}, "
                "a [locations] table, or QUANT_DATA_ROOT."
            )
        return StorageLocations(
            raw=cast(Path, self.raw),
            intermediate=cast(Path, self.intermediate),
            datasets=cast(Path, self.datasets),
            experiments=cast(Path, self.experiments),
        )


@version("1.0")
def _configured_candidate(
    values: Mapping[str, Path], name: str, *, source: str
) -> tuple[Path, str] | None:
    if name in values:
        return values[name], f"{source}:[locations].{name}"
    if "root" in values:
        return values["root"] / _NEW_ROOT_CHILD[name], f"{source}:[locations].root"
    return None


@version(
    "1.0",
    dependencies=[
        LocationOverrides,
        LocationResolution,
        _path,
        _load_location_table,
        _discover_repository_config,
        _configured_candidate,
    ],
)
def resolve_locations(
    overrides: LocationOverrides = LocationOverrides(),
    *,
    environment: Mapping[str, str] | None = None,
    repository_start: Path | None = None,
    repository_config: Path | None = None,
    user_config: Path | None = None,
    home: Path | None = None,
    cwd: Path | None = None,
) -> LocationResolution:
    """Resolve roots in the documented precedence order without creating them."""

    env = os.environ if environment is None else environment
    working_directory = Path.cwd() if cwd is None else cwd
    working_directory = working_directory.expanduser().resolve(strict=False)
    start = working_directory if repository_start is None else repository_start
    explicit_repository_config = repository_config is not None
    repo_path = (
        repository_config.expanduser().resolve(strict=False)
        if repository_config is not None
        else _discover_repository_config(start)
    )
    home_path = Path.home() if home is None else home
    user_path = (
        user_config.expanduser().resolve(strict=False)
        if user_config is not None
        else (home_path / ".config" / "quant_data" / "sanos.toml").resolve(strict=False)
    )
    repository_values = _load_location_table(repo_path, required=explicit_repository_config)
    user_values = _load_location_table(user_path, required=user_config is not None)

    warnings: list[str] = []
    for name in LOCATION_NAMES:
        new_name = _NEW_ENV[name]
        legacy_name = _LEGACY_ENV[name]
        if new_name in env and legacy_name in env:
            new_value = _path(env[new_name], base=working_directory)
            legacy_value = _path(env[legacy_name], base=working_directory)
            if new_value != legacy_value:
                warnings.append(
                    f"{new_name}={new_value} overrides conflicting legacy "
                    f"{legacy_name}={legacy_value}"
                )
    if _NEW_ROOT_ENV in env and _LEGACY_ROOT_ENV in env:
        new_root = _path(env[_NEW_ROOT_ENV], base=working_directory)
        legacy_root = _path(env[_LEGACY_ROOT_ENV], base=working_directory)
        if new_root != legacy_root:
            warnings.append(
                f"{_NEW_ROOT_ENV}={new_root} overrides conflicting legacy "
                f"{_LEGACY_ROOT_ENV}={legacy_root}"
            )

    explicit_root = (
        None if overrides.root is None else _path(overrides.root, base=working_directory)
    )
    resolved: dict[str, Path | None] = {}
    sources: list[tuple[str, str]] = []
    for name in LOCATION_NAMES:
        explicit_value = getattr(overrides, name)
        candidates: list[tuple[Path, str]] = []
        if explicit_value is not None:
            candidates.append((_path(explicit_value, base=working_directory), f"cli:{name}"))
        if explicit_root is not None:
            candidates.append((explicit_root / _NEW_ROOT_CHILD[name], "cli:root"))

        new_name = _NEW_ENV[name]
        legacy_name = _LEGACY_ENV[name]
        if new_name in env:
            candidates.append((_path(env[new_name], base=working_directory), f"env:{new_name}"))
        elif legacy_name in env:
            candidates.append(
                (_path(env[legacy_name], base=working_directory), f"legacy-env:{legacy_name}")
            )

        repository_candidate = _configured_candidate(
            repository_values,
            name,
            source=(
                f"repository-config:{repo_path}" if repo_path is not None else "repository-config"
            ),
        )
        if repository_candidate is not None:
            candidates.append(repository_candidate)
        user_candidate = _configured_candidate(
            user_values,
            name,
            source=f"user-config:{user_path}",
        )
        if user_candidate is not None:
            candidates.append(user_candidate)

        if _NEW_ROOT_ENV in env:
            root = _path(env[_NEW_ROOT_ENV], base=working_directory)
            candidates.append((root / _NEW_ROOT_CHILD[name], f"env:{_NEW_ROOT_ENV}"))
        elif _LEGACY_ROOT_ENV in env:
            root = _path(env[_LEGACY_ROOT_ENV], base=working_directory)
            candidates.append((root / _LEGACY_ROOT_CHILD[name], f"legacy-env:{_LEGACY_ROOT_ENV}"))

        if candidates:
            value, source = candidates[0]
            resolved[name] = value.resolve(strict=False)
            sources.append((name, source))
            if source == f"legacy-env:{legacy_name}":
                warnings.append(f"Using legacy location alias {legacy_name}; prefer {new_name}")
            elif source == f"legacy-env:{_LEGACY_ROOT_ENV}":
                warnings.append(
                    f"Using legacy location alias {_LEGACY_ROOT_ENV}; prefer {_NEW_ROOT_ENV}"
                )
        else:
            resolved[name] = None

    return LocationResolution(
        raw=resolved["raw"],
        intermediate=resolved["intermediate"],
        datasets=resolved["datasets"],
        experiments=resolved["experiments"],
        sources=tuple(sources),
        warnings=tuple(dict.fromkeys(warnings)),
        repository_config=repo_path if repo_path is not None and repo_path.exists() else None,
        user_config=user_path if user_path.exists() else None,
    )
