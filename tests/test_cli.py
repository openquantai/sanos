from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from click.testing import CliRunner

from sanos.artifacts import ArtifactPointer, StorageClass
from sanos.cli import main
from sanos.config import canonical_sha256, canonical_value
from sanos.datasets.public import (
    PublicApproval,
    PublicProjectionConfig,
    PublicReleaseConfig,
)
from sanos.widb import DailyWidbConfig


def test_help_lists_stable_command_groups() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "config",
        "doctor",
        "status",
        "raw",
        "intermediate",
        "dataset",
        "experiment",
    ):
        assert command in result.output


def test_public_dataset_help_lists_human_release_workflow() -> None:
    result = CliRunner().invoke(main, ["dataset", "public", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("build", "validate", "approval-template", "approve"):
        assert command in result.output


def test_doctor_json_is_offline_structured_and_non_mutating(tmp_path: Path) -> None:
    root = tmp_path / "data"
    result = CliRunner().invoke(
        main,
        ["--root", str(root), "--log-format", "json", "doctor", "--strict-locations"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "doctor"
    assert payload["status"] == "ok"
    assert payload["network_accessed"] is False
    assert payload["filesystem_modified"] is False
    assert payload["dry_run"] is False
    assert payload["modules"]["cdxcore"] is True
    assert payload["modules"]["filelock"] is True
    assert payload["modules"]["pandas_market_calendars"] is True
    assert payload["modules"]["polars"] is True
    assert isinstance(payload["optional_modules"]["wrds"], bool)
    assert payload["locations"]["raw"]["source"] == "cli:root"
    assert not root.exists()


def test_status_json_reports_daily_intermediate_capability(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["--root", str(tmp_path / "data"), "--log-format", "json", "status"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "status"
    assert payload["network_accessed"] is False
    assert payload["phase"] == "internal-experiment-materialization"
    assert payload["pipeline_available"] is True
    assert payload["material_pipeline_available"] is True
    assert payload["raw_download_available"] is True
    assert payload["intermediate_build_available"] is True
    assert payload["dataset_build_available"] is True
    assert payload["dataset_validate_available"] is True
    assert payload["public_candidate_build_available"] is True
    assert payload["public_candidate_validate_available"] is True
    assert payload["public_release_approval_available"] is True
    assert payload["experiment_build_available"] is True
    assert payload["experiment_validate_available"] is True
    assert payload["live_wrds_verified"] is False


def test_config_show_reports_precedence_without_creating_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    raw = tmp_path / "raw-override"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--raw-root",
            str(raw),
            "--log-format",
            "json",
            "config",
            "show",
            "--require-complete",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["locations"]["raw"]["path"] == str(raw)
    assert payload["locations"]["raw"]["source"] == "cli:raw"
    assert payload["locations"]["datasets"]["path"] == str(root / "datasets")
    assert payload["filesystem_modified"] is False
    assert not root.exists()
    assert not raw.exists()


def test_explicit_bad_config_uses_stable_configuration_exit(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["--repository-config", str(tmp_path / "missing.toml"), "config", "show"],
    )

    assert result.exit_code == 3
    assert "does not exist" in result.output


def test_structured_errors_and_global_dry_run_are_machine_readable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    result = CliRunner().invoke(
        main,
        [
            "--log-format",
            "json",
            "--dry-run",
            "--repository-config",
            str(missing),
            "config",
            "show",
        ],
    )

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["event"] == "error"
    assert payload["error_type"] == "ConfigurationCliError"
    assert payload["exit_code"] == 3

    status = CliRunner().invoke(
        main,
        ["--root", str(tmp_path / "data"), "--dry-run", "--log-format", "json", "status"],
    )
    assert status.exit_code == 0
    assert json.loads(status.output)["dry_run"] is True


def test_intermediate_build_dry_run_is_structured_offline_and_non_mutating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    config = Path(__file__).parents[1] / "configs" / "widb-spx-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "intermediate",
            "build",
            "--config",
            str(config),
            "--ticker",
            "SPX",
            "--start-date",
            "2024-01-03",
            "--end-date",
            "2024-01-03",
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "intermediate_result"
    assert payload["stage"] == "enriched-day"
    assert payload["total"] == 1
    assert payload["existing"] == 0
    assert payload["config_id"] == canonical_sha256(DailyWidbConfig())
    assert payload["configuration"] == canonical_value(DailyWidbConfig())
    assert "surface" not in payload["configuration"]
    assert payload["network_accessed"] is False
    assert payload["dry_run"] is True
    assert not root.exists()


def test_fitted_intermediate_dry_run_exposes_surface_configuration(tmp_path: Path) -> None:
    root = tmp_path / "data"
    config = Path(__file__).parents[1] / "configs" / "widb-spx-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "intermediate",
            "build",
            "--config",
            str(config),
            "--ticker",
            "SPX",
            "--start-date",
            "2024-01-03",
            "--end-date",
            "2024-01-03",
            "--through",
            "fitted-day",
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["stage"] == "fitted-day"
    assert payload["configuration"]["surface"]["smooth_sanos"]["smoothness"] == 0.25
    assert payload["network_accessed"] is False
    assert payload["dry_run"] is True
    assert not root.exists()


def test_dataset_build_dry_run_is_structured_offline_and_non_mutating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    config = Path(__file__).parents[1] / "configs" / "widb-spx-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "dataset",
            "build",
            "--config",
            str(config),
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "dataset_result"
    assert payload["planned_dates"] == 918
    assert payload["built"] is False
    assert payload["reused"] is False
    assert payload["dry_run"] is True
    assert payload["network_accessed"] is False
    assert len(payload["identity"]) == 64
    assert not root.exists()


def test_dataset_validate_never_builds_a_missing_dataset(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "widb-spx-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "dataset",
            "validate",
            "--config",
            str(config),
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 4
    payload = json.loads(result.output)
    assert payload["event"] == "error"
    assert "never builds or publishes" in payload["message"]
    assert not (tmp_path / "data").exists()


def test_public_candidate_build_dry_run_is_structured_offline_and_non_mutating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    config = Path(__file__).parents[1] / "configs" / "sanos-public-candidate-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "dataset",
            "public",
            "build",
            "--config",
            str(config),
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "public_candidate_result"
    assert payload["distribution"] == "public_candidate"
    assert payload["built"] is False
    assert payload["reused"] is False
    assert payload["dry_run"] is True
    assert payload["source_accessed"] is False
    assert payload["network_accessed"] is False
    assert len(payload["identity"]) == 64
    assert not root.exists()


def test_public_candidate_validate_never_builds_a_missing_candidate(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "sanos-public-candidate-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "dataset",
            "public",
            "validate",
            "--config",
            str(config),
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 4
    payload = json.loads(result.output)
    assert payload["event"] == "error"
    assert "validation never builds" in payload["message"]
    assert not (tmp_path / "data").exists()


def test_public_release_approve_dry_run_is_offline_and_needs_no_license_file(
    tmp_path: Path,
) -> None:
    projection = PublicProjectionConfig.from_document(
        json.loads(
            (
                Path(__file__).parents[1]
                / "configs"
                / "sanos-public-candidate-reference.json"
            ).read_text(encoding="utf-8")
        )
    )
    release = PublicReleaseConfig(
        projection=projection,
        approval=PublicApproval(
            candidate_dataset=ArtifactPointer(
                StorageClass.DATASET,
                "sanos/public-candidate/name-test",
                "a" * 64,
            ),
            candidate_manifest_sha256="b" * 64,
            candidate_logical_content_sha256="c" * 64,
            release_license="TEST-ONLY",
            redistribution_basis="Dry-run fixture",
            approver="pytest",
            approval_date=date(2026, 10, 12),
            approval_reference="test://dry-run",
            license_sha256="d" * 64,
        ),
    )
    config_path = tmp_path / "release.json"
    config_path.write_text(json.dumps(release.to_document()), encoding="utf-8")
    root = tmp_path / "data"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "dataset",
            "public",
            "approve",
            "--config",
            str(config_path),
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "public_release_result"
    assert payload["distribution"] == "public"
    assert payload["dry_run"] is True
    assert payload["source_accessed"] is False
    assert payload["network_accessed"] is False
    assert not root.exists()


def test_experiment_build_dry_run_is_structured_offline_and_non_mutating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    config = Path(__file__).parents[1] / "configs" / "dysanos-daily-front-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "experiment",
            "build",
            "--config",
            str(config),
            "--max-workers",
            "2",
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 0, result.output
    payloads = [json.loads(line) for line in result.output.splitlines()]
    assert [payload["event"] for payload in payloads] == [
        "experiment_plan",
        "experiment_result",
    ]
    result_payload = payloads[-1]
    assert result_payload["built"] is False
    assert result_payload["reused"] is False
    assert result_payload["dry_run"] is True
    assert result_payload["source_accessed"] is False
    assert result_payload["network_accessed"] is False
    assert len(result_payload["identity"]) == 64
    assert not root.exists()


def test_experiment_validate_never_builds_a_missing_experiment(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "dysanos-daily-front-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "experiment",
            "validate",
            "--config",
            str(config),
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 4
    payload = json.loads(result.output)
    assert payload["event"] == "error"
    assert "never builds or publishes" in payload["message"]
    assert not (tmp_path / "data").exists()


def test_intermediate_raw_miss_is_actionable_and_never_connects(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "widb-spx-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "intermediate",
            "build",
            "--config",
            str(config),
            "--ticker",
            "SPX",
            "--start-date",
            "2024-01-03",
            "--end-date",
            "2024-01-03",
        ],
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"},
    )

    assert result.exit_code == 4
    payloads = [json.loads(line) for line in result.output.splitlines()]
    assert payloads[-1]["event"] == "error"
    assert "Required Raw artifact is absent" in payloads[-1]["message"]
    assert "sanos raw download" in payloads[-1]["message"]


def test_raw_download_dry_run_is_structured_and_never_connects(tmp_path: Path) -> None:
    root = tmp_path / "data"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(root),
            "--dry-run",
            "--log-format",
            "json",
            "raw",
            "download",
            "--ticker",
            "SPX",
            "--start-date",
            "2022-01-03",
            "--end-date",
            "2022-01-03",
        ],
        env={"WRDS_USERNAME": "must-not-be-read", "WRDS_OFF": "1"},
    )

    assert result.exit_code == 0, result.output
    payloads = [json.loads(line) for line in result.output.splitlines()]
    assert [payload["event"] for payload in payloads] == ["raw_plan", "raw_result"]
    assert payloads[0]["missing"] == 8
    assert payloads[0]["boundary_dependencies"] == [
        {
            "observation_date": "2021-12-31",
            "roles": ["boundary"],
            "table": "forwards",
            "ticker": "SPX",
        },
        {
            "observation_date": "2021-12-31",
            "roles": ["boundary"],
            "table": "spot",
            "ticker": "SPX",
        },
        {
            "observation_date": "2021-12-31",
            "roles": ["boundary"],
            "table": "zero_curve",
            "ticker": "SPX",
        },
    ]
    assert payloads[-1]["network_accessed"] is False
    assert not root.exists()


def test_noninteractive_raw_miss_requires_explicit_2fa_confirmation(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "raw",
            "download",
            "--ticker",
            "SPX",
            "--start-date",
            "2024-01-03",
            "--end-date",
            "2024-01-03",
        ],
        env={"WRDS_USERNAME": "must-not-connect"},
    )

    assert result.exit_code == 4
    payloads = [json.loads(line) for line in result.output.splitlines()]
    assert payloads[0]["event"] == "raw_plan"
    assert payloads[-1]["event"] == "error"
    assert "--confirm-2fa-ready" in payloads[-1]["message"]


def test_raw_connection_limit_above_four_is_rejected_by_click(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "raw",
            "download",
            "--ticker",
            "SPX",
            "--start-date",
            "2024-01-03",
            "--end-date",
            "2024-01-03",
            "--max-wrds-connections",
            "5",
        ],
    )
    assert result.exit_code == 2
    assert "range 1<=x<=4" in result.output


def test_click_usage_errors_are_structured_when_json_is_requested() -> None:
    root_error = CliRunner().invoke(main, ["--log-format", "json", "--unknown"])
    assert root_error.exit_code == 2
    root_payload = json.loads(root_error.output)
    assert root_payload["event"] == "error"
    assert root_payload["error_type"] == "NoSuchOption"
    assert root_payload["exit_code"] == 2

    nested_error = CliRunner().invoke(main, ["--log-format=json", "config", "show", "--unknown"])
    assert nested_error.exit_code == 2
    nested_payload = json.loads(nested_error.output)
    assert nested_payload["event"] == "error"
    assert nested_payload["error_type"] == "NoSuchOption"

    missing_value = CliRunner().invoke(main, ["--log-format", "json", "--root"])
    assert missing_value.exit_code == 2
    assert json.loads(missing_value.output)["error_type"] == "BadOptionUsage"
