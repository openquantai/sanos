from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from sanos.config import (
    CanonicalizationError,
    LocationConfigurationError,
    LocationOverrides,
    canonical_json,
    canonical_sha256,
    resolve_locations,
)


@dataclass(frozen=True)
class ExampleConfig:
    day: date
    weights: tuple[float, ...]


def test_canonical_serialization_is_stable_and_typed() -> None:
    left = {
        "z": np.array([1, 2], dtype=np.int64),
        "a": ExampleConfig(date(2024, 1, 2), (0.2, 0.8)),
        "when": datetime(2024, 1, 2, 3, 4, tzinfo=UTC),
    }
    right = {"when": left["when"], "a": left["a"], "z": [1, 2]}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "bad"}, {1, 2}, b"bad"])
def test_canonical_serialization_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json(value)


def test_canonical_serialization_rejects_naive_datetime() -> None:
    with pytest.raises(CanonicalizationError, match="timezone"):
        canonical_json(datetime(2024, 1, 1))


def test_location_precedence_and_relative_config_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".sanos.toml").write_text(
        '[locations]\nroot = "repo-data"\nexperiments = "repo-experiments"\n',
        encoding="utf-8",
    )
    user_config = tmp_path / "user" / "sanos.toml"
    user_config.parent.mkdir()
    user_config.write_text('[locations]\nroot = "user-data"\n', encoding="utf-8")
    explicit_root = tmp_path / "explicit"
    explicit_raw = tmp_path / "explicit-raw"
    environment = {
        "QUANT_DATA_RAW": str(tmp_path / "new-raw"),
        "QUANTDATA_RAW_DATA": str(tmp_path / "legacy-raw"),
        "QUANT_DATA_INTERMEDIATE": str(tmp_path / "new-intermediate"),
        "QUANT_DATA_ROOT": str(tmp_path / "general"),
    }

    resolution = resolve_locations(
        LocationOverrides(root=explicit_root, raw=explicit_raw),
        environment=environment,
        repository_start=repository,
        user_config=user_config,
        cwd=tmp_path,
    )

    assert resolution.raw == explicit_raw.resolve()
    assert resolution.intermediate == (explicit_root / "intermediate").resolve()
    assert resolution.datasets == (explicit_root / "datasets").resolve()
    assert resolution.experiments == (explicit_root / "experiments").resolve()
    assert resolution.source_for("raw") == "cli:raw"
    assert any("overrides conflicting legacy" in warning for warning in resolution.warnings)
    assert not explicit_root.exists()
    assert not explicit_raw.exists()


def test_specific_environment_then_repository_then_user_then_general_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".sanos.toml").write_text(
        '[locations]\nintermediate = "repo-intermediate"\n', encoding="utf-8"
    )
    user_config = tmp_path / "user.toml"
    user_config.write_text('[locations]\ndatasets = "user-datasets"\n', encoding="utf-8")
    resolution = resolve_locations(
        environment={
            "QUANT_DATA_RAW": str(tmp_path / "env-raw"),
            "QUANT_DATA_ROOT": str(tmp_path / "general"),
        },
        repository_start=repository,
        user_config=user_config,
        cwd=tmp_path,
    )

    assert resolution.raw == (tmp_path / "env-raw").resolve()
    assert resolution.intermediate == (repository / "repo-intermediate").resolve()
    assert resolution.datasets == (tmp_path / "user-datasets").resolve()
    assert resolution.experiments == (tmp_path / "general" / "experiments").resolve()


def test_legacy_root_maps_old_processing_and_training_names(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    resolution = resolve_locations(
        environment={"QUANTDATA_ROOT": str(root)}, cwd=tmp_path, home=tmp_path
    )

    assert resolution.raw == root / "raw"
    assert resolution.intermediate == root / "processing"
    assert resolution.datasets == root / "training"
    assert resolution.experiments == root / "experiments"
    assert any("QUANTDATA_ROOT" in warning for warning in resolution.warnings)


def test_invalid_or_missing_explicit_config_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(LocationConfigurationError, match="does not exist"):
        resolve_locations(repository_config=missing, environment={}, home=tmp_path, cwd=tmp_path)

    invalid = tmp_path / "invalid.toml"
    invalid.write_text('[locations]\nderived = "x"\n', encoding="utf-8")
    with pytest.raises(LocationConfigurationError, match="Unknown"):
        resolve_locations(repository_config=invalid, environment={}, home=tmp_path, cwd=tmp_path)


def test_incomplete_resolution_is_diagnostic_until_required(tmp_path: Path) -> None:
    resolution = resolve_locations(environment={}, home=tmp_path, cwd=tmp_path)

    assert resolution.missing == ("raw", "intermediate", "datasets", "experiments")
    with pytest.raises(LocationConfigurationError, match="Missing data locations"):
        resolution.require_complete()


def test_foundation_objects_expose_cdxcore_versions() -> None:
    assert str(canonical_json.version).startswith("1.0")
    assert str(resolve_locations.version).startswith("1.0")
