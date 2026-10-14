from __future__ import annotations

import copy
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from sanos.artifacts import (
    ArtifactIntegrityError,
    ArtifactPointer,
    ArtifactStore,
    StorageClass,
)
from sanos.cli import main
from sanos.config import StorageLocations, canonical_json, canonical_sha256
from sanos.reference import (
    DailyComparisonRequest,
    ReferenceComparisonError,
    build_comparison_approval_document,
    compare_daily_artifacts,
    load_comparison_approvals,
    load_comparison_report_for_approval,
    preflight_comparison_approval_output,
    validate_comparison_approval_document,
    validate_comparison_report_for_approval,
    write_comparison_approval_document,
    write_comparison_report,
)
from sanos.widb import DailySurfaceConfig, DailyWidbConfig
from sanos.widb.build import (
    IntermediateBuildRequest,
    build_intermediates,
    read_daily_frame,
)
from sanos.widb.surface import FITTED_SCHEMA
from tests.test_reference_compare import _catalog, _legacy_payload
from tests.test_reference_legacy_market import _write_legacy
from tests.test_widb_build import _locations, _publish_raw_inputs

_EXPLANATION = (
    "Owner accepts the exact validated solver-runtime discrepancies; no model, "
    "tolerance, population, or legacy-value substitution is approved."
)


@dataclass(frozen=True)
class _ApprovalCase:
    locations: StorageLocations
    legacy_root: Path
    catalog_path: Path
    report_path: Path
    report: dict[str, Any]
    request: DailyComparisonRequest


def _case(base: Path) -> _ApprovalCase:
    locations = _locations(base / "data")
    store = ArtifactStore(locations)
    trading_dates = (date(2024, 1, 3), date(2024, 1, 4))
    for trading_date in trading_dates:
        _publish_raw_inputs(store, trading_date)
    result = build_intermediates(
        locations,
        IntermediateBuildRequest(
            ticker="SPX",
            start_date=trading_dates[0],
            end_date=trading_dates[-1],
            through="fitted-day",
        ),
        DailyWidbConfig(),
        DailySurfaceConfig(),
    )
    assert len(result.terminal) == 2
    legacy_root = base / "legacy"
    legacy_root.mkdir()
    for pointer, trading_date in zip(result.terminal, trading_dates, strict=True):
        artifact = store.validate(pointer)
        frame = read_daily_frame(
            store,
            artifact,
            schema=FITTED_SCHEMA,
            ticker="SPX",
            trading_date=trading_date,
        )
        payload = _legacy_payload(frame)
        fit_result = cast(dict[str, object], payload["fit_result"])
        fit_result["mean_err"] = float(fit_result["mean_err"]) + 2.0e-12
        _write_legacy(legacy_root / f"{trading_date.isoformat()}.h5", payload)
    catalog_path = base / "catalog.json"
    _catalog(legacy_root, catalog_path)
    request = DailyComparisonRequest(
        ticker="SPX",
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        source="optionmetrics",
        source_generation="default-v1",
    )
    report = compare_daily_artifacts(
        locations,
        request,
        daily_config=DailyWidbConfig(),
        surface_config=DailySurfaceConfig(),
        legacy_root=legacy_root,
        catalog_path=catalog_path,
        source_revision="e" * 40,
        require_pinned_catalog=False,
    )
    assert report["classification_counts"] == {"unexplained_discrepancy": 2}
    evidence = base / "evidence"
    evidence.mkdir()
    report_path = evidence / "comparison.json"
    write_comparison_report(report_path, report, legacy_root=legacy_root)
    return _ApprovalCase(
        locations=locations,
        legacy_root=legacy_root,
        catalog_path=catalog_path,
        report_path=report_path,
        report=report,
        request=request,
    )


@pytest.fixture(scope="module")
def approval_case(tmp_path_factory: pytest.TempPathFactory) -> _ApprovalCase:
    return _case(tmp_path_factory.mktemp("reference-approval"))


def _build(case: _ApprovalCase, dates: tuple[date, ...]) -> dict[str, Any]:
    return build_comparison_approval_document(
        case.locations,
        comparison_report=case.report_path,
        legacy_root=case.legacy_root,
        approval_dates=dates,
        explanation=_EXPLANATION,
        approver="Test Owner",
        approval_date=date(2026, 10, 14),
        approval_reference="decision://test/exact-migration-discrepancy",
    )


def _resigned_report(
    original: dict[str, Any],
    output: Path,
    mutation: Any,
) -> Path:
    changed = copy.deepcopy(original)
    mutation(changed)
    body = {key: value for key, value in changed.items() if key != "report_id"}
    changed["report_id"] = canonical_sha256(body)
    output.write_text(canonical_json(changed) + "\n", encoding="utf-8")
    return output


def _comparison_arguments(case: _ApprovalCase) -> dict[str, Any]:
    return {
        "daily_config": DailyWidbConfig(),
        "surface_config": DailySurfaceConfig(),
        "legacy_root": case.legacy_root,
        "catalog_path": case.catalog_path,
        "source_revision": "e" * 40,
        "require_pinned_catalog": False,
    }


def test_approval_functions_have_resolvable_explicit_versions() -> None:
    expected_dependencies: dict[Any, set[str]] = {
        load_comparison_report_for_approval: {
            "_read_canonical_json_object",
            "validate_comparison_report_for_approval",
        },
        validate_comparison_report_for_approval: {
            "_iso_date",
            "_passing_checks",
            "canonical_sha256",
            "canonical_value",
        },
        build_comparison_approval_document: {
            "ArtifactPointer",
            "ArtifactStore",
            "DailyLegacyComparisonTolerances",
            "DailySurfaceConfig",
            "DailyWidbConfig",
            "IntermediateBuildRequest",
            "InternalDatasetConfig",
            "_approval_sha256_file",
            "_assert_approval_path_has_no_symlinks",
            "_authority_text",
            "_exact_pointer",
            "_iso_date",
            "_manifest_digest",
            "_passing_checks",
            "canonical_value",
            "compare_populated_day",
            "load_comparison_report_for_approval",
            "plan_daily_artifacts",
            "read_daily_frame",
            "read_legacy_pytree_hdf5",
            "validate_fitted_partition",
            "validate_comparison_approval_document",
        },
        preflight_comparison_approval_output: {
            "_assert_approval_path_has_no_symlinks"
        },
        write_comparison_approval_document: {
            "atomic_create_json",
            "preflight_comparison_approval_output",
            "validate_comparison_approval_document",
        },
    }
    for function, expected in expected_dependencies.items():
        versioned = cast(Any, function)
        assert str(versioned.version)
        assert set(dict(versioned.version.dependencies[1])) == expected

    assert validate_comparison_approval_document(
        {
            "schema": "sanos-legacy-daily-comparison-approvals-v1",
            "approvals": [
                {
                    "date": "2024-01-03",
                    "fields": ["fit_result.mean_err"],
                    "fresh_artifact_id": "a" * 64,
                    "fresh_data_sha256": "b" * 64,
                    "legacy_file_sha256": "c" * 64,
                    "explanation": "Exact reviewed discrepancy",
                    "approver": "Test Owner",
                    "approval_date": "2026-10-14",
                    "approval_reference": "decision://test/validation",
                }
            ],
        }
    )["approvals"][0]["date"] == "2024-01-03"


def test_generated_approval_applies_only_to_exact_selected_dates(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    partial_document = _build(approval_case, (date(2024, 1, 3),))
    partial_path = tmp_path / "partial.json"
    write_comparison_approval_document(
        partial_path,
        partial_document,
        legacy_root=approval_case.legacy_root,
    )
    assert load_comparison_approvals(partial_path) == {
        "2024-01-03": partial_document["approvals"][0]
    }
    partial = compare_daily_artifacts(
        approval_case.locations,
        approval_case.request,
        approvals_path=partial_path,
        **_comparison_arguments(approval_case),
    )
    assert partial["outcome"] == "blocked"
    assert partial["classification_counts"] == {
        "approved_discrepancy": 1,
        "unexplained_discrepancy": 1,
    }

    complete_document = _build(
        approval_case,
        (date(2024, 1, 3), date(2024, 1, 4)),
    )
    complete_path = tmp_path / "complete.json"
    write_comparison_approval_document(
        complete_path,
        complete_document,
        legacy_root=approval_case.legacy_root,
    )
    complete = compare_daily_artifacts(
        approval_case.locations,
        approval_case.request,
        approvals_path=complete_path,
        **_comparison_arguments(approval_case),
    )
    assert complete["outcome"] == "passed"
    assert complete["classification_counts"] == {"approved_discrepancy": 2}


def test_report_loader_rejects_resigned_internal_contradictions(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    inconsistent_counts = _resigned_report(
        approval_case.report,
        tmp_path / "counts.json",
        lambda value: value.__setitem__("classification_counts", {"match": 2}),
    )
    with pytest.raises(ReferenceComparisonError, match="counts are inconsistent"):
        load_comparison_report_for_approval(inconsistent_counts)

    def duplicate(value: dict[str, Any]) -> None:
        value["dates"].append(copy.deepcopy(value["dates"][-1]))
        value["classification_counts"]["unexplained_discrepancy"] += 1

    duplicate_dates = _resigned_report(
        approval_case.report,
        tmp_path / "duplicate.json",
        duplicate,
    )
    with pytest.raises(ReferenceComparisonError, match="sorted and unique"):
        load_comparison_report_for_approval(duplicate_dates)

    def failed_check(value: dict[str, Any]) -> None:
        value["dates"][0]["pilot_checks"] = [{"name": "tampered", "passed": False}]

    failed_checks = _resigned_report(
        approval_case.report,
        tmp_path / "failed-check.json",
        failed_check,
    )
    with pytest.raises(ReferenceComparisonError, match="only passing checks"):
        load_comparison_report_for_approval(failed_checks)


def test_builder_rejects_resigned_widened_or_missing_evidence(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    def widen(value: dict[str, Any]) -> None:
        fields = value["dates"][0]["comparison"]["mismatch_fields"]
        fields.append("market.strikes[0]")
        fields.sort()

    widened = _resigned_report(
        approval_case.report,
        tmp_path / "widened.json",
        widen,
    )
    with pytest.raises(ReferenceComparisonError, match="does not reproduce exactly"):
        build_comparison_approval_document(
            approval_case.locations,
            comparison_report=widened,
            legacy_root=approval_case.legacy_root,
            approval_dates=(date(2024, 1, 3),),
            explanation=_EXPLANATION,
            approver="Test Owner",
            approval_date=date(2026, 10, 14),
            approval_reference="decision://test/widened",
        )

    def remove_legacy(value: dict[str, Any]) -> None:
        value["dates"][0]["legacy"] = None

    missing_legacy = _resigned_report(
        approval_case.report,
        tmp_path / "missing-legacy.json",
        remove_legacy,
    )
    with pytest.raises(ReferenceComparisonError, match="legacy evidence is malformed"):
        build_comparison_approval_document(
            approval_case.locations,
            comparison_report=missing_legacy,
            legacy_root=approval_case.legacy_root,
            approval_dates=(date(2024, 1, 3),),
            explanation=_EXPLANATION,
            approver="Test Owner",
            approval_date=date(2026, 10, 14),
            approval_reference="decision://test/missing",
        )


def test_builder_reopens_fresh_and_legacy_bytes(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    stale_legacy = tmp_path / "stale-legacy"
    shutil.copytree(approval_case.legacy_root, stale_legacy)
    with (stale_legacy / "2024-01-03.h5").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ReferenceComparisonError, match="size is stale"):
        build_comparison_approval_document(
            approval_case.locations,
            comparison_report=approval_case.report_path,
            legacy_root=stale_legacy,
            approval_dates=(date(2024, 1, 3),),
            explanation=_EXPLANATION,
            approver="Test Owner",
            approval_date=date(2026, 10, 14),
            approval_reference="decision://test/stale-legacy",
        )

    copied_base = tmp_path / "copied-data"
    copied_locations = StorageLocations(
        raw=copied_base / "raw",
        intermediate=copied_base / "intermediate",
        datasets=copied_base / "datasets",
        experiments=copied_base / "experiments",
    )
    shutil.copytree(approval_case.locations.intermediate, copied_locations.intermediate)
    copied_locations.raw.mkdir(parents=True)
    copied_locations.datasets.mkdir(parents=True)
    copied_locations.experiments.mkdir(parents=True)
    first = approval_case.report["dates"][0]["fresh"]["artifact"]
    pointer = ArtifactPointer(
        StorageClass(first["storage_class"]),
        first["namespace"],
        first["artifact_id"],
    )
    copied_data = ArtifactStore(copied_locations).path_for(pointer) / "data.parquet"
    with copied_data.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ArtifactIntegrityError, match="byte size|checksum|integrity|digest"):
        build_comparison_approval_document(
            copied_locations,
            comparison_report=approval_case.report_path,
            legacy_root=approval_case.legacy_root,
            approval_dates=(date(2024, 1, 3),),
            explanation=_EXPLANATION,
            approver="Test Owner",
            approval_date=date(2026, 10, 14),
            approval_reference="decision://test/stale-fresh",
        )


def test_report_and_output_paths_reject_leaf_and_parent_symlinks(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    report_link = tmp_path / "report-link.json"
    report_link.symlink_to(approval_case.report_path)
    with pytest.raises(ReferenceComparisonError, match="symlink"):
        load_comparison_report_for_approval(report_link)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ReferenceComparisonError, match="symlink"):
        preflight_comparison_approval_output(
            linked_parent / "approval.json",
            legacy_root=approval_case.legacy_root,
        )

    redirected = tmp_path / "redirected.json"
    redirected.write_text("sentinel", encoding="utf-8")
    output_link = tmp_path / "output-link.json"
    output_link.symlink_to(redirected)
    with pytest.raises(ReferenceComparisonError, match="symlink"):
        preflight_comparison_approval_output(
            output_link,
            legacy_root=approval_case.legacy_root,
        )
    assert redirected.read_text(encoding="utf-8") == "sentinel"


def test_writer_is_atomic_no_replace_under_concurrency(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    document = _build(approval_case, (date(2024, 1, 3),))
    output = tmp_path / "approval.json"
    barrier = threading.Barrier(2)

    def publish() -> str:
        barrier.wait()
        try:
            write_comparison_approval_document(
                output,
                document,
                legacy_root=approval_case.legacy_root,
            )
        except ReferenceComparisonError:
            return "conflict"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _ in range(2)]
        outcomes = sorted(future.result() for future in futures)
    assert outcomes == ["conflict", "created"]
    assert load_comparison_approvals(output) == {
        "2024-01-03": document["approvals"][0]
    }
    with pytest.raises(ReferenceComparisonError, match="already exists"):
        write_comparison_approval_document(
            output,
            document,
            legacy_root=approval_case.legacy_root,
        )


def _location_arguments(locations: StorageLocations) -> list[str]:
    return [
        "--raw-root",
        str(locations.raw),
        "--intermediate-root",
        str(locations.intermediate),
        "--datasets-root",
        str(locations.datasets),
        "--experiments-root",
        str(locations.experiments),
    ]


def _approval_cli_arguments(case: _ApprovalCase, output: Path) -> list[str]:
    return [
        "reference",
        "approval-document",
        "--comparison-report",
        str(case.report_path),
        "--legacy-root",
        str(case.legacy_root),
        "--date",
        "2024-01-03",
        "--explanation",
        _EXPLANATION,
        "--approver",
        "Test Owner",
        "--approval-date",
        "2026-10-14",
        "--approval-reference",
        "decision://test/cli",
        "--output",
        str(output),
    ]


def test_cli_requires_owner_attestation_and_dry_run_writes_nothing(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    output = tmp_path / "approval.json"
    base = [*_location_arguments(approval_case.locations), "--log-format", "json"]
    missing = CliRunner().invoke(
        main,
        [*base, *_approval_cli_arguments(approval_case, output)],
    )
    assert missing.exit_code != 0
    assert "acknowledge-owner-approved-exact-discrepancies" in missing.output
    assert not output.exists()

    dry_run = CliRunner().invoke(
        main,
        [
            *base,
            "--dry-run",
            *_approval_cli_arguments(approval_case, output),
            "--acknowledge-owner-approved-exact-discrepancies",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    payload = json.loads(dry_run.output)
    assert payload == {
        "approval_count": 1,
        "artifacts_built": False,
        "authority": "owner-approved-exact-migration-discrepancies",
        "dry_run": True,
        "event": "reference_approval_document_result",
        "filesystem_modified": False,
        "network_accessed": False,
        "output": None,
        "redistribution_approval": False,
        "status": "passed",
    }
    assert _EXPLANATION not in dry_run.output
    assert "decision://test/cli" not in dry_run.output
    assert not output.exists()


def test_dry_run_preflights_existing_output(
    approval_case: _ApprovalCase,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("sentinel", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            *_location_arguments(approval_case.locations),
            "--dry-run",
            *_approval_cli_arguments(approval_case, output),
            "--acknowledge-owner-approved-exact-discrepancies",
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == "sentinel"
