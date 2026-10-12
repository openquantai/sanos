from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sanos.cli import main


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


def test_status_json_reports_guarded_raw_capability(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["--root", str(tmp_path / "data"), "--log-format", "json", "status"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "status"
    assert payload["network_accessed"] is False
    assert payload["phase"] == "guarded-wrds-raw"
    assert payload["pipeline_available"] is True
    assert payload["material_pipeline_available"] is False
    assert payload["raw_download_available"] is True
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


def test_material_commands_fail_actionably_until_implemented() -> None:
    result = CliRunner().invoke(main, ["intermediate", "build"])

    assert result.exit_code == 7
    assert "planned for Phases 5-6" in result.output
    assert "PROGRAM_PLAN.md" in result.output


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
