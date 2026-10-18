from __future__ import annotations

import importlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
from cdxcore.verbose import Context
from click.testing import CliRunner

import sanos.widb.recovery as recovery_module
from sanos.artifacts import ArtifactIntegrityError, ArtifactPointer, ArtifactStore, StorageClass
from sanos.cli.main import main
from sanos.config import StorageLocations, canonical_sha256, canonical_value
from sanos.datasets import InternalDatasetConfig, plan_internal_dataset
from sanos.datasets.internal import DatasetValidationError, _solver_calls
from sanos.model.arbfree import MkArbFree
from sanos.widb import DailySurfaceConfig, DailyWidbConfig
from sanos.widb.build import IntermediateBuildRequest, plan_daily_artifacts
from sanos.widb.recovery import (
    FITTED_SOURCE_SELECTION_SCHEMA,
    ORIGINAL_FITTED_PLAN_SCHEMA,
    FittedSourceBinding,
    FittedSourceSelection,
    OriginalFittedPlan,
    OriginalFittedSource,
    build_fitted_recovery,
    feasible_candidate_recovery_policy_id,
    fitted_recovery_spec,
    read_original_fitted_plan,
    recover_failed_fitted_sources,
    resume_fitted_sources,
    write_selection,
)


def _pointer(label: str, *, raw: bool = False) -> ArtifactPointer:
    return ArtifactPointer(
        StorageClass.RAW if raw else StorageClass.INTERMEDIATE,
        f"test/{label}",
        canonical_sha256(label),
    )


def _source(trading_date: date, label: str) -> OriginalFittedSource:
    return OriginalFittedSource(
        trading_date=trading_date,
        raw_security=_pointer(f"{label}-security", raw=True),
        raw_options=_pointer(f"{label}-options", raw=True),
        raw_spot=_pointer(f"{label}-spot", raw=True),
        raw_previous_spot=_pointer(f"{label}-previous-spot", raw=True),
        raw_forwards=_pointer(f"{label}-forwards", raw=True),
        raw_zero_curve=_pointer(f"{label}-zero", raw=True),
        paired=_pointer(f"{label}-paired"),
        enriched=_pointer(f"{label}-enriched"),
        no_arb=_pointer(f"{label}-no-arb"),
        filtered=_pointer(f"{label}-filtered"),
        expected_fitted=_pointer(f"{label}-fitted"),
    )


def _original_plan(*sources: OriginalFittedSource) -> OriginalFittedPlan:
    unsigned = {
        "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
        "ticker": "SPX",
        "source_revision": "0" * 40,
        "configuration_sha256": "c" * 64,
        "sources": [item.as_dict() for item in sources],
    }
    return OriginalFittedPlan(
        ticker="SPX",
        source_revision="0" * 40,
        configuration_sha256="c" * 64,
        sources=tuple(sources),
        evidence_sha256=canonical_sha256(unsigned),
    )


def _locations(root: Path) -> StorageLocations:
    return StorageLocations(
        raw=root / "raw",
        intermediate=root / "intermediate",
        datasets=root / "datasets",
        experiments=root / "experiments",
    )


def _failure_evidence(trading_date: date) -> tuple[dict[str, str], ...]:
    return ({"event": "unit_failed", "key": f"SPX/{trading_date.isoformat()}"},)


def _plan_sha(
    sources: tuple[OriginalFittedSource, ...],
    *,
    configuration_sha256: str = "c" * 64,
) -> str:
    return canonical_sha256(
        {
            "schema": ORIGINAL_FITTED_PLAN_SCHEMA,
            "ticker": "SPX",
            "source_revision": "0" * 40,
            "configuration_sha256": configuration_sha256,
            "sources": [item.as_dict() for item in sources],
        }
    )


def test_selection_round_trip_preserves_complete_original_chain() -> None:
    source = _source(date(2024, 1, 3), "old")
    recovered = _pointer("recovered")
    evidence = _failure_evidence(source.trading_date)
    selection = FittedSourceSelection(
        ticker="SPX",
        original_plan_sha256=_plan_sha((source,)),
        original_source_revision="0" * 40,
        original_configuration_sha256="c" * 64,
        bindings=(
            FittedSourceBinding(
                trading_date=source.trading_date,
                original_source=source,
                selected=recovered,
                recovery_policy_id=feasible_candidate_recovery_policy_id(),
                failure_evidence_sha256=canonical_sha256(evidence),
                failure_evidence=evidence,
            ),
        ),
    )

    assert selection.as_dict()["schema"] == FITTED_SOURCE_SELECTION_SCHEMA
    assert FittedSourceSelection.from_document(selection.as_dict()) == selection
    binding = selection.bindings[0]
    assert binding.original_expected == source.expected_fitted
    assert binding.filtered == source.filtered
    assert binding.selected == recovered


def test_resume_preserves_original_success_and_plans_every_missing_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily = DailyWidbConfig()
    surface = DailySurfaceConfig()
    request = IntermediateBuildRequest(
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 4),
        through="fitted-day",
    )
    current = plan_daily_artifacts(request, daily, surface)
    current_sources = tuple(recovery_module._source_from_daily_plan(item) for item in current)
    originals = tuple(
        replace(source, expected_fitted=_pointer(f"old-{source.trading_date}-fitted"))
        for source in current_sources
    )
    original_plan = _original_plan(*originals)
    original_validate = ArtifactStore.validate

    def validate_first_only(store: ArtifactStore, pointer, *args, **kwargs):
        if pointer == originals[0].expected_fitted:
            return object()
        return original_validate(store, pointer, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "validate", validate_first_only)
    selection, result = resume_fitted_sources(
        _locations(tmp_path),
        original_plan=original_plan,
        daily_config=daily,
        surface_config=surface,
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        max_workers=1,
        dry_run=True,
    )

    assert result.dry_run
    assert selection.bindings[0].selected == originals[0].expected_fitted
    assert selection.bindings[0].replacement_source is None
    assert selection.bindings[1].selected == current_sources[1].expected_fitted
    assert selection.bindings[1].replacement_source == current_sources[1]
    assert FittedSourceSelection.from_document(selection.as_dict()) == selection


def test_selection_rejects_coherent_original_source_substitution() -> None:
    source = _source(date(2024, 1, 3), "old")
    selection = FittedSourceSelection(
        ticker="SPX",
        original_plan_sha256=_plan_sha((source,)),
        original_source_revision="0" * 40,
        original_configuration_sha256="c" * 64,
        bindings=(
            FittedSourceBinding(
                trading_date=source.trading_date,
                original_source=source,
                selected=source.expected_fitted,
            ),
        ),
    ).as_dict()
    replacement = _source(source.trading_date, "substitute")
    selection["bindings"][0]["original_source"] = replacement.as_dict()
    selection["bindings"][0]["selected"] = replacement.expected_fitted.as_dict()

    with pytest.raises(ValueError, match="original plan digest differs"):
        FittedSourceSelection.from_document(selection)


def test_recovery_inputs_reject_symlinks_and_selection_is_no_replace(tmp_path: Path) -> None:
    plan = _original_plan(_source(date(2024, 1, 3), "old"))
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    plan_link = tmp_path / "plan-link.json"
    plan_link.symlink_to(plan_file)
    with pytest.raises(ValueError, match="symlink"):
        read_original_fitted_plan(plan_link)
    evidence = tmp_path / "failure.jsonl"
    evidence.write_text(
        json.dumps({"event": "unit_failed", "key": "SPX/2024-01-03"}) + "\n",
        encoding="utf-8",
    )
    evidence_link = tmp_path / "failure-link.jsonl"
    evidence_link.symlink_to(evidence)
    with pytest.raises(ValueError, match="symlink"):
        recovery_module._journal_failed_records(evidence_link, ticker="SPX")

    selection = FittedSourceSelection(
        ticker="SPX",
        original_plan_sha256=plan.evidence_sha256,
        original_source_revision=plan.source_revision,
        original_configuration_sha256=plan.configuration_sha256,
        bindings=(
            FittedSourceBinding(
                trading_date=plan.sources[0].trading_date,
                original_source=plan.sources[0],
                selected=plan.sources[0].expected_fitted,
            ),
        ),
    )
    output = tmp_path / "selection.json"
    write_selection(output, selection)
    with pytest.raises(FileExistsError):
        write_selection(output, selection)
    output_link = tmp_path / "selection-link.json"
    output_link.symlink_to(output)
    with pytest.raises((ArtifactIntegrityError, FileExistsError, OSError)):
        write_selection(output_link, selection)


def test_recovery_identity_binds_original_policy_filtered_and_failure() -> None:
    source = _source(date(2024, 1, 3), "old")
    evidence = _failure_evidence(source.trading_date)
    arguments = {
        "ticker": "SPX",
        "trading_date": source.trading_date,
        "surface_config": DailySurfaceConfig(),
        "filtered": source.filtered,
        "original_expected": source.expected_fitted,
        "original_plan_sha256": "a" * 64,
        "failure_evidence_sha256": canonical_sha256(evidence),
        "failure_evidence": evidence,
    }
    base = fitted_recovery_spec(**arguments)
    changed_evidence = ({"event": "unit_failed", "key": "SPX/2024-01-03", "message": "changed"},)
    changed_failure = fitted_recovery_spec(
        **{
            **arguments,
            "failure_evidence_sha256": canonical_sha256(changed_evidence),
            "failure_evidence": changed_evidence,
        }
    )
    changed_original = fitted_recovery_spec(
        **{**arguments, "original_expected": _pointer("other-fitted")}
    )

    assert base.pointer not in {changed_failure.pointer, changed_original.pointer}
    assert base.pointer.namespace.startswith("widb/fitted-recovery/")
    assert base.pointer != source.expected_fitted


def test_recovery_resolved_version_owns_direct_policy_and_evidence_dependencies() -> None:
    resolved = str(recover_failed_fitted_sources.version)
    for dependency in (
        "_journal_failed_records",
        "_validate_original_filtered_chain",
        "business_dates",
        "canonical_sha256",
        "feasible_candidate_recovery_policy_id",
        "fitted_recovery_spec",
    ):
        assert f"{dependency}:" in resolved
    build_resolved = str(build_fitted_recovery.version)
    assert "read_daily_frame:" in build_resolved
    assert "fit_smooth_sanos:" in build_resolved
    assert "canonical_value:" in str(FittedSourceSelection.version)
    assert "_recovery_pointer:" in str(recovery_module._require_upstream.version)


def test_recovery_refuses_a_successful_original_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    config = InternalDatasetConfig(
        name="recovery-test",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )
    item = plan_internal_dataset(config).daily_artifacts[0]
    evidence = _failure_evidence(item.trading_date)
    monkeypatch.setattr(
        ArtifactStore,
        "validate",
        lambda self, pointer, **kwargs: (
            object()
            if pointer == item.fitted.pointer
            else pytest.fail(f"unexpected validation: {pointer}")
        ),
    )

    with pytest.raises(ValueError, match="Refusing recovery for successful"):
        build_fitted_recovery(
            store,
            ticker="SPX",
            trading_date=item.trading_date,
            surface_config=config.surface,
            filtered=item.filtered.pointer,
            original_expected=item.fitted.pointer,
            original_plan_sha256="a" * 64,
            failure_evidence_sha256=canonical_sha256(evidence),
            failure_evidence=evidence,
        )


def test_dry_run_retries_only_recorded_failure_without_requiring_pending_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _source(date(2024, 1, 3), "failed")
    pending = _source(date(2024, 1, 4), "pending")
    original = _original_plan(failed, pending)
    evidence = tmp_path / "run.jsonl"
    evidence.write_text(
        json.dumps({"event": "unit_failed", "key": "SPX/2024-01-03"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recovery_module,
        "_validate_original_filtered_chain",
        lambda *args, **kwargs: None,
    )

    selection = recover_failed_fitted_sources(
        _locations(tmp_path),
        original_plan=original,
        failure_evidence=evidence,
        failed_dates=(failed.trading_date,),
        surface_config=DailySurfaceConfig(),
        include_half_days=True,
        dry_run=True,
    )

    assert selection.bindings[0].selected != failed.expected_fitted
    assert selection.bindings[1].selected == pending.expected_fitted
    assert not ArtifactStore(_locations(tmp_path)).path_for(pending.expected_fitted).exists()


def test_dataset_plan_binds_mixed_selection_without_replanning_old_chain() -> None:
    config = InternalDatasetConfig(
        name="recovery-test",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )
    current = plan_internal_dataset(config).daily_artifacts[0]
    old = _source(current.trading_date, "old")
    evidence = _failure_evidence(old.trading_date)
    recovered = fitted_recovery_spec(
        ticker="SPX",
        trading_date=old.trading_date,
        surface_config=config.surface,
        filtered=old.filtered,
        original_expected=old.expected_fitted,
        original_plan_sha256="a" * 64,
        failure_evidence_sha256=canonical_sha256(evidence),
        failure_evidence=evidence,
    ).pointer
    selection = FittedSourceSelection(
        ticker="SPX",
        original_plan_sha256=_plan_sha(
            (old,),
            configuration_sha256=canonical_sha256(
                {
                    "dataset": {
                        "name": config.name,
                        "ticker": config.ticker,
                        "start_date": config.start_date,
                        "end_date": config.end_date,
                        "include_half_days": config.include_half_days,
                        "source": config.source,
                        "source_generation": config.source_generation,
                    },
                    "daily": config.daily,
                    "surface": config.surface,
                }
            ),
        ),
        original_source_revision="0" * 40,
        original_configuration_sha256=canonical_sha256(
            {
                "dataset": {
                    "name": config.name,
                    "ticker": config.ticker,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "include_half_days": config.include_half_days,
                    "source": config.source,
                    "source_generation": config.source_generation,
                },
                "daily": config.daily,
                "surface": config.surface,
            }
        ),
        bindings=(
            FittedSourceBinding(
                trading_date=old.trading_date,
                original_source=old,
                selected=recovered,
                recovery_policy_id=feasible_candidate_recovery_policy_id(),
                failure_evidence_sha256=canonical_sha256(evidence),
                failure_evidence=evidence,
            ),
        ),
    )
    recovered_config = replace(config, fitted_selection=selection)
    plan = plan_internal_dataset(recovered_config)

    assert plan.fitted_bindings == selection.bindings
    assert plan.raw_catalog_spec.upstream == tuple(sorted(old.raw_roles.values()))
    assert plan.dataset_spec.configuration["fitted_selection"] == selection.as_dict()


def test_export_fitted_plan_rejects_claimed_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = InternalDatasetConfig(
        name="export-test",
        ticker="SPX",
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 3),
        include_half_days=True,
        source="optionmetrics",
        source_generation="default-v1",
        daily=DailyWidbConfig(),
        surface=DailySurfaceConfig(),
    )
    document = {
        "dataset": {
            "name": config.name,
            "ticker": config.ticker,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "include_half_days": config.include_half_days,
            "source": config.source,
            "source_generation": config.source_generation,
        },
        "daily": canonical_value(config.daily),
        "surface": canonical_value(config.surface),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    cli_module = importlib.import_module("sanos.cli.main")
    monkeypatch.setattr(cli_module, "repository_source_revision", lambda: "1" * 40)

    result = CliRunner().invoke(
        main,
        [
            "--log-format",
            "json",
            "--dry-run",
            "intermediate",
            "export-fitted-plan",
            "--config",
            str(config_path),
            "--source-revision",
            "2" * 40,
            "--output",
            str(tmp_path / "plan.json"),
        ],
        env={"WRDS_OFF": "1"},
    )

    assert result.exit_code == 4, result.output
    assert "does not match" in result.output
    assert not (tmp_path / "plan.json").exists()

    overridden = CliRunner().invoke(
        main,
        [
            "--log-format",
            "json",
            "--dry-run",
            "intermediate",
            "export-fitted-plan",
            "--config",
            str(config_path),
            "--source-revision",
            "1" * 40,
            "--output",
            str(tmp_path / "overridden-plan.json"),
        ],
        env={"WRDS_OFF": "1", "SANOS_SOURCE_REVISION": "1" * 40},
    )
    assert overridden.exit_code == 4, overridden.output
    assert "refuses SANOS_SOURCE_REVISION" in overridden.output
    assert not (tmp_path / "overridden-plan.json").exists()

    configuration_sha256 = canonical_sha256(
        {
            "dataset": {
                "name": config.name,
                "ticker": config.ticker,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "include_half_days": config.include_half_days,
                "source": config.source,
                "source_generation": config.source_generation,
            },
            "daily": config.daily,
            "surface": config.surface,
        }
    )
    source = _source(config.start_date, "prior-recovery")
    prior_selection = FittedSourceSelection(
        ticker=config.ticker,
        original_plan_sha256=_plan_sha((source,), configuration_sha256=configuration_sha256),
        original_source_revision="0" * 40,
        original_configuration_sha256=configuration_sha256,
        bindings=(
            FittedSourceBinding(
                trading_date=source.trading_date,
                original_source=source,
                selected=source.expected_fitted,
            ),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_read_dataset_config",
        lambda _: replace(config, fitted_selection=prior_selection),
    )
    selected = CliRunner().invoke(
        main,
        [
            "--log-format",
            "json",
            "--dry-run",
            "intermediate",
            "export-fitted-plan",
            "--config",
            str(config_path),
            "--source-revision",
            "1" * 40,
            "--output",
            str(tmp_path / "selected-plan.json"),
        ],
        env={"WRDS_OFF": "1"},
    )
    assert selected.exit_code == 4, selected.output
    assert "without fitted_selection" in selected.output
    assert not (tmp_path / "selected-plan.json").exists()


def test_solver_grammar_authenticates_recovered_and_published_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solve = cp.Problem.solve

    def return_finite_limit(problem: cp.Problem, *args: object, **kwargs: object) -> object:
        result = original_solve(problem, *args, **kwargs)
        problem._status = "user_limit"
        return result

    monkeypatch.setattr(cp.Problem, "solve", return_finite_limit)
    events: list[dict[str, object]] = []
    fitter = MkArbFree(
        smoothness=0.25,
        min_dlv=None,
        max_dlv=None,
        _accept_feasible_nonoptimal=True,
        _solver_diagnostic_sink=lambda event: events.append(dict(event)),
    )
    with pytest.warns(UserWarning):
        fitter.surface_densities(
            expiries=np.array([0.1]),
            variances=np.array([0.004]),
            strikes=[np.array([0.9, 1.0, 1.1])],
            bids=[np.array([0.105, 0.025, 0.005])],
            asks=[np.array([0.115, 0.035, 0.015])],
            masks=None,
            xstrikes=[np.array([0.5, 0.9, 1.0, 1.1, 1.5])],
            xguessed_p=[np.array([0.0, 0.1, 0.8, 0.1, 0.0])],
            verbose=Context.quiet,
        )
    calls = _solver_calls(events, "smooth")
    assert calls[-1].terminal_status == "feasible_nonoptimal"
    assert calls[-1].published_state_density_digest == events[-1]["published_state_density_digest"]

    shifted = [dict(event) for event in events]
    shifted[-1]["objective"] = float(shifted[-1]["objective"]) + 1.0e-8
    shifted[-1]["published_state_contracted_objective_recomputed"] = shifted[-1]["objective"]
    shifted_calls = _solver_calls(shifted, "smooth")
    assert shifted_calls[-1].terminal_status == "feasible_nonoptimal"

    tampered = [dict(event) for event in shifted]
    tampered[-1]["published_state_density_digest"] = "0" * 64
    # A syntactically valid but unlinked digest is rejected later against the
    # persisted state; objective linkage is rejected directly by the grammar.
    tampered[-1]["objective"] = float(tampered[-1]["objective"]) + 1.0
    with pytest.raises(DatasetValidationError, match="objective linkage"):
        _solver_calls(tampered, "smooth")
