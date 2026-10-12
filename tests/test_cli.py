from __future__ import annotations

import json

from click.testing import CliRunner

from sanos.cli import main


def test_help_lists_stable_command_groups() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("doctor", "status", "raw", "intermediate", "dataset", "experiment"):
        assert command in result.output


def test_doctor_json_is_offline_and_structured() -> None:
    result = CliRunner().invoke(main, ["--log-format", "json", "doctor"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "doctor"
    assert payload["status"] == "ok"
    assert payload["network_accessed"] is False
    assert payload["modules"]["cdxcore"] is True
    assert payload["modules"]["polars"] is True


def test_status_json_reports_bootstrap() -> None:
    result = CliRunner().invoke(main, ["--log-format", "json", "status"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "event": "status",
        "network_accessed": False,
        "phase": "bootstrap",
        "pipeline_available": False,
    }


def test_material_commands_fail_actionably_until_implemented() -> None:
    result = CliRunner().invoke(main, ["raw", "download"])

    assert result.exit_code != 0
    assert "not implemented in the Phase 0 bootstrap" in result.output
    assert "PROGRAM_PLAN.md" in result.output

