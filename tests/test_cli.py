from __future__ import annotations

import importlib
import json
from datetime import date
from pathlib import Path

import pytest
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
        "dysanos",
        "experiment",
        "production",
    ):
        assert command in result.output


def test_public_dataset_help_lists_human_release_workflow() -> None:
    result = CliRunner().invoke(main, ["dataset", "public", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("build", "validate", "approval-template", "approve"):
        assert command in result.output


def test_canonical_reference_help_lists_review_and_promotion_workflow() -> None:
    result = CliRunner().invoke(main, ["dataset", "reference", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("review-template", "promotion-template", "promote", "validate"):
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
    assert payload["phase"] == "offline-production-readiness"
    assert payload["pipeline_available"] is True
    assert payload["material_pipeline_available"] is True
    assert payload["raw_download_available"] is True
    assert payload["intermediate_build_available"] is True
    assert payload["dataset_build_available"] is True
    assert payload["dataset_validate_available"] is True
    assert payload["canonical_reference_promotion_available"] is True
    assert payload["public_candidate_build_available"] is True
    assert payload["public_candidate_validate_available"] is True
    assert payload["public_release_approval_available"] is True
    assert payload["experiment_build_available"] is True
    assert payload["experiment_validate_available"] is True
    assert payload["phase11_config_finalization_available"] is True
    assert payload["phase11_runtime_fingerprint_available"] is True
    assert payload["phase11_evidence_index_available"] is True
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


def test_config_resources_reports_packaged_operator_assets() -> None:
    result = CliRunner().invoke(main, ["--log-format", "json", "config", "resources"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "config_resources"
    assert payload["network_accessed"] is False
    assert payload["filesystem_modified"] is False
    assert Path(payload["resources"]["reference_catalog"]).is_file()
    assert payload["resources"]["source_revision"] == "checkout"


def test_production_help_lists_frozen_operator_contract() -> None:
    result = CliRunner().invoke(main, ["production", "--help"])

    assert result.exit_code == 0, result.output
    assert "runtime-fingerprint" in result.output
    assert "finalize-configs" in result.output
    assert "evidence-index" in result.output


def _finalize_arguments(tmp_path: Path) -> list[str]:
    root = Path(__file__).parents[1]
    return [
        "--root",
        str(tmp_path / "data"),
        "--log-format",
        "json",
        "production",
        "finalize-configs",
        "--dataset-config",
        str(root / "configs" / "widb-spx-reference.json"),
        "--public-template",
        str(root / "configs" / "sanos-public-candidate-reference.json"),
        "--experiment-template",
        str(root / "configs" / "dysanos-daily-front-reference.json"),
        "--output-directory",
        str(tmp_path / "final-configs"),
    ]


def test_finalize_configs_resolves_dataset_graph_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = importlib.import_module("sanos.cli.main")
    monkeypatch.setattr(cli_module.ArtifactStore, "validate", lambda *_args, **_kwargs: None)
    result = CliRunner().invoke(
        main,
        ["--dry-run", *_finalize_arguments(tmp_path)],
        env={"WRDS_OFF": "1"},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event"] == "phase11_configs_finalized"
    assert payload["dry_run"] is True
    assert not (tmp_path / "final-configs").exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "Provide --dataset-config"),
        (
            ["--dataset-name", "widb-spx-2022-2025-reference", "--dataset-artifact-id", "a" * 64],
            "Use --dataset-config",
        ),
    ],
)
def test_finalize_configs_rejects_ambiguous_dataset_selection(
    tmp_path: Path,
    extra: list[str],
    message: str,
) -> None:
    arguments = _finalize_arguments(tmp_path)
    if not extra:
        index = arguments.index("--dataset-config")
        del arguments[index : index + 2]
    else:
        arguments.extend(extra)
    result = CliRunner().invoke(main, arguments, env={"WRDS_OFF": "1"})

    assert result.exit_code == 4, result.output
    assert message in json.loads(result.output)["message"]


def test_finalize_configs_requires_the_planned_dataset_to_exist(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        _finalize_arguments(tmp_path),
        env={"WRDS_OFF": "1"},
    )

    assert result.exit_code == 4, result.output
    assert "does not exist" in json.loads(result.output)["message"].lower()


def test_runtime_fingerprint_cli_is_offline_rootless_reusable_and_dry_run_safe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime-fingerprint.json"
    arguments = [
        "--log-format",
        "json",
        "production",
        "runtime-fingerprint",
        "--output",
        str(output),
    ]
    runner = CliRunner()
    first = runner.invoke(
        main,
        arguments,
        env={"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-appear"},
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert first_payload["created"] is True
    assert first_payload["reused"] is False
    assert first_payload["network_accessed"] is False
    assert first_payload["fingerprint_id"] == json.loads(
        output.read_text(encoding="utf-8")
    )["fingerprint_id"]
    assert "must-not-appear" not in output.read_text(encoding="utf-8")

    second = runner.invoke(main, arguments, env={"WRDS_OFF": "1"})
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["created"] is False
    assert second_payload["reused"] is True

    dry_output = tmp_path / "dry-runtime-fingerprint.json"
    dry = runner.invoke(
        main,
        [
            "--dry-run",
            "--log-format",
            "json",
            "production",
            "runtime-fingerprint",
            "--output",
            str(dry_output),
        ],
        env={"WRDS_OFF": "1"},
    )
    assert dry.exit_code == 0, dry.output
    dry_payload = json.loads(dry.output)
    assert dry_payload["status"] == "planned"
    assert dry_payload["filesystem_modified"] is False
    assert not dry_output.exists()


def test_runtime_fingerprint_cli_rejects_conflicting_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime-fingerprint.json"
    runner = CliRunner()
    arguments = [
        "--log-format",
        "json",
        "production",
        "runtime-fingerprint",
        "--output",
        str(output),
    ]
    first = runner.invoke(main, arguments, env={"WRDS_OFF": "1"})
    assert first.exit_code == 0, first.output
    conflicting = json.loads(output.read_text(encoding="utf-8"))
    conflicting["python"]["compiler"] = "conflicting-compiler"
    body = {key: value for key, value in conflicting.items() if key != "fingerprint_id"}
    conflicting["fingerprint_id"] = canonical_sha256(body)
    output.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")

    result = runner.invoke(main, arguments, env={"WRDS_OFF": "1"})
    assert result.exit_code == 4, result.output
    payload = json.loads(result.output)
    assert "already exists with other content" in payload["message"]


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
    payloads = [json.loads(line) for line in result.output.splitlines()]
    assert all(item["event"] == "warning" for item in payloads[:-1])
    payload = payloads[-1]
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


@pytest.mark.parametrize("command_group", ["dysanos", "experiment"])
def test_dysanos_build_dry_run_and_legacy_alias_are_identical(
    tmp_path: Path,
    command_group: str,
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
            command_group,
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


def test_dysanos_validate_never_builds_a_missing_experiment(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "dysanos-daily-front-reference.json"
    result = CliRunner().invoke(
        main,
        [
            "--root",
            str(tmp_path / "data"),
            "--log-format",
            "json",
            "dysanos",
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


def test_raw_download_has_no_manual_2fa_gate_and_caps_query_attempts() -> None:
    result = CliRunner().invoke(
        main,
        ["raw", "download", "--help"],
    )

    assert result.exit_code == 0, result.output
    assert "--confirm-2fa-ready" not in result.output
    assert "--retry-attempts INTEGER RANGE" in result.output
    assert "1<=x<=3" in result.output


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


def test_raw_query_attempt_limit_above_three_is_rejected_by_click(tmp_path: Path) -> None:
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
            "--retry-attempts",
            "4",
        ],
    )
    assert result.exit_code == 2
    assert "range 1<=x<=3" in result.output


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
