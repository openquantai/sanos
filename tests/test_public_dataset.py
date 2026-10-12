from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from test_widb_build import _dataset_config, _locations, _publish_raw_inputs

import sanos.datasets.internal as internal_dataset_module
import sanos.datasets.public.publish as public_publish_module
from sanos.artifacts import (
    COMPLETION_NAME,
    MANIFEST_NAME,
    ArtifactNotFoundError,
    ArtifactStore,
    WildcardReuseMissError,
)
from sanos.cli import main
from sanos.config import canonical_json
from sanos.datasets import DatasetBuildOptions, build_internal_dataset, open_internal_dataset
from sanos.datasets.public import (
    PublicApproval,
    PublicBuildOptions,
    PublicDatasetValidationError,
    PublicExpectedOmission,
    PublicProjectionConfig,
    PublicProjectionError,
    PublicReleaseConfig,
    build_public_candidate,
    build_public_release,
    open_public_dataset,
    plan_public_candidate,
    plan_public_release,
    project_public_day,
    public_capabilities,
    public_polars_schema,
    validate_public_dataset,
    validate_public_frame,
)
from sanos.datasets.public.project import (
    _daily_front_mask,
    _log_curve_value,
    _source_carry_curve,
)
from sanos.datasets.public.validate import (
    _state_pricer,
    public_coverage_sha256,
    public_frame_logical_sha256,
    public_return_values_sha256,
    validate_public_payload,
)
from sanos.widb.calendar import NMS, ONE_MINUTE, business_clock_policy
from sanos.widb.surface import compatibility_dlv_pricer, smooth_sanos_pricer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resign_files(artifact_path: Path, *relative_paths: str) -> None:
    manifest_path = artifact_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {record["path"]: record for record in manifest["files"]}
    for relative_path in relative_paths:
        payload = artifact_path / relative_path
        record = records[relative_path]
        record["byte_size"] = payload.stat().st_size
        record["sha256"] = _sha256(payload)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    completion_path = artifact_path / COMPLETION_NAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = _sha256(manifest_path)
    completion_path.write_text(canonical_json(completion) + "\n", encoding="utf-8")


def _publish_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    start: date = date(2024, 1, 3),
    end: date = date(2024, 1, 4),
):
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    current = start
    while current <= end:
        if current.weekday() < 5:
            _publish_raw_inputs(store, current)
        current = date.fromordinal(current.toordinal() + 1)
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    config = replace(
        _dataset_config(start),
        name="public-source",
        start_date=start,
        end_date=end,
    )
    return locations, build_internal_dataset(
        locations,
        config,
        DatasetBuildOptions(max_workers=1),
    )


def _config(
    source,
    *,
    omissions: tuple[PublicExpectedOmission, ...] = (),
) -> PublicProjectionConfig:
    return PublicProjectionConfig(
        name="test-public-candidate",
        source_dataset=source.artifact,
        coverage_start_date=source.plan.expected_dates[0],
        coverage_end_date=source.plan.expected_dates[-1],
        expected_typed_empty_omissions=omissions,
    )


def _release_config(
    locations,
    projection: PublicProjectionConfig,
    candidate,
    license_bytes: bytes,
) -> PublicReleaseConfig:
    return PublicReleaseConfig(
        projection=projection,
        approval=PublicApproval(
            candidate_dataset=candidate.artifact,
            candidate_manifest_sha256=_sha256(candidate.path / MANIFEST_NAME),
            candidate_logical_content_sha256=public_frame_logical_sha256(
                open_public_dataset(ArtifactStore(locations), candidate.artifact).load()
            ),
            release_license="TEST-ONLY",
            redistribution_basis="Synthetic test fixture only",
            approver="pytest",
            approval_date=date(2026, 10, 12),
            approval_reference="test://phase-8-release",
            license_sha256=hashlib.sha256(license_bytes).hexdigest(),
        ),
    )


def test_public_configuration_is_strict_and_identity_sensitive() -> None:
    document = json.loads(
        (Path(__file__).parents[1] / "configs" / "sanos-public-candidate-reference.json").read_text(
            encoding="utf-8"
        )
    )
    config = PublicProjectionConfig.from_document(document)
    assert PublicProjectionConfig.from_document(config.to_document()) == config
    original = plan_public_candidate(config).dataset_spec.artifact_id
    changed = replace(config, exclusion_tolerance=2.0e-4)
    assert plan_public_candidate(changed).dataset_spec.artifact_id != original
    assert plan_public_candidate(config).dataset_spec.artifact_id == original

    malformed = json.loads(json.dumps(document))
    malformed["machine_path"] = "/private/data"
    with pytest.raises(ValueError, match="only 'public_dataset'"):
        PublicProjectionConfig.from_document(malformed)
    malformed = json.loads(json.dumps(document))
    malformed["public_dataset"]["european_only"] = 1
    with pytest.raises(ValueError, match="booleans"):
        PublicProjectionConfig.from_document(malformed)


def test_candidate_build_validate_and_output_first_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(tmp_path, monkeypatch)
    config = _config(source)
    dry_run = build_public_candidate(locations, config, dry_run=True)
    assert dry_run.dry_run is True
    assert dry_run.source_accessed is False
    assert not dry_run.path.exists()

    built = build_public_candidate(locations, config)
    assert built.built is True
    assert built.source_accessed is True
    assert built.validation is not None
    assert built.validation.date_count == 2
    assert built.validation.book_count > 0
    assert built.path.joinpath("data.parquet").is_file()
    frame = open_public_dataset(ArtifactStore(locations), built.artifact).load()
    assert frame.schema == public_polars_schema()
    assert frame.get_column("forward_adjusted_spot_return").null_count() == 1
    assert frame.item(0, "forward_adjusted_spot_return") is None
    assert frame.item(1, "forward_adjusted_spot_return") < 0.0
    source_dataset = open_internal_dataset(ArtifactStore(locations), source.artifact)
    previous_day = source_dataset.load_day(date(2024, 1, 3))
    current_day = source_dataset.load_day(date(2024, 1, 4))
    business_dte = float(
        previous_day.get_column("business_dte").drop_nulls().unique().item()
    )
    previous_forward = float(
        previous_day.get_column("implied_forward").drop_nulls().unique().item()
    )
    previous_spot = float(previous_day.get_column("spot_reference").item(0))
    carry_at_one_session = (previous_forward / previous_spot) ** (1.0 / business_dte)
    expected_return = (
        float(current_day.get_column("spot_reference").item(0))
        / float(previous_day.get_column("spot_reference").item(0))
    ) / carry_at_one_session - 1.0
    assert frame.item(1, "forward_adjusted_spot_return") == pytest.approx(
        expected_return,
        abs=1.0e-14,
    )
    expiry = frame.item(0, "expiries")[0]
    assert expiry["discounted_forward_over_spot"] == pytest.approx(
        float(previous_day.get_column("discount_factor").item(0))
        * float(previous_day.get_column("spot_normalized_implied_forward").item(0)),
        abs=1.0e-14,
    )
    assert frame.columns == [
        "trading_date",
        "is_half_day",
        "forward_adjusted_spot_return",
        "nms_epsilon_atm_annual_volatility",
        "expiries",
        "books",
        "smooth_sanos",
        "discrete_dlv",
    ]
    assert built.path.joinpath("capabilities.json").read_text(encoding="utf-8") == (
        json.dumps(public_capabilities(), sort_keys=True, separators=(",", ":")) + "\n"
    )

    hidden = tmp_path / "hidden-source"
    source.path.rename(hidden)
    reused = build_public_candidate(
        locations,
        config,
        PublicBuildOptions(version_override="*"),
    )
    assert reused.built is False
    assert reused.source_accessed is False
    assert reused.reuse is not None and reused.reuse.used_wildcard
    assert validate_public_dataset(ArtifactStore(locations), reused.artifact) is not None
    with pytest.raises(ArtifactNotFoundError):
        validate_public_dataset(
            ArtifactStore(locations),
            reused.artifact,
            deep_lineage=True,
        )
    hidden.rename(source.path)


def test_candidate_wildcard_miss_never_reads_or_builds_source(tmp_path: Path) -> None:
    pointer = PublicProjectionConfig.from_document(
        json.loads(
            (
                Path(__file__).parents[1]
                / "configs"
                / "sanos-public-candidate-reference.json"
            ).read_text(encoding="utf-8")
        )
    ).source_dataset
    config = PublicProjectionConfig(
        name="missing-candidate",
        source_dataset=pointer,
        coverage_start_date=date(2022, 1, 3),
        coverage_end_date=date(2025, 8, 29),
        expected_typed_empty_omissions=(
            PublicExpectedOmission(
                date(2025, 1, 8), "known_bad_day_all_tickers"
            ),
        ),
    )
    locations = _locations(tmp_path)
    with pytest.raises(WildcardReuseMissError):
        build_public_candidate(
            locations,
            config,
            PublicBuildOptions(version_override="*"),
        )
    assert not locations.datasets.exists()


def test_approved_log_curve_interpolation_and_no_extrapolation() -> None:
    assert _log_curve_value((0.0, 2.0), (1.0, 1.21), 1.0, "forward") == pytest.approx(
        1.1,
        abs=1.0e-15,
    )
    assert _log_curve_value((0.0, 2.0), (1.0, 0.81), 2.0, "discount") == 0.81
    with pytest.raises(PublicProjectionError, match="does not span"):
        _log_curve_value((0.0, 0.5), (1.0, 1.01), 1.0, "forward")


def test_recursive_allowlist_rejects_unexpected_or_nonfinite_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    config = _config(source)
    built = build_public_candidate(locations, config)
    frame = open_public_dataset(ArtifactStore(locations), built.artifact).load()
    with pytest.raises(PublicDatasetValidationError, match="recursive schema changed"):
        validate_public_frame(frame.with_columns(pl.lit(None).alias("spot")), config)
    corrupted = frame.with_columns(
        pl.lit(float("inf"), dtype=pl.Float64).alias(
            "nms_epsilon_atm_annual_volatility"
        )
    )
    with pytest.raises(PublicDatasetValidationError, match="non-finite"):
        validate_public_frame(corrupted, config)


def test_candidate_contains_no_forbidden_top_level_or_nested_names() -> None:
    schema_text = str(public_polars_schema()).lower()
    for forbidden in (
        "secid",
        "optionid",
        "symbol",
        "volume",
        "open_interest",
        "market_bid",
        "market_ask",
        "spot_reference",
        "implied_forward",
        "absolute_strike",
    ):
        assert forbidden not in schema_text
    capabilities = public_capabilities()
    assert capabilities["normalized_smooth_sanos_pricing"] is True
    assert capabilities["normalized_discrete_dlv_pricing"] is True
    assert capabilities["exact_vendor_books"] is False
    assert capabilities["contract_or_security_replay"] is False


def test_projection_is_invariant_to_identifiers_and_absolute_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    internal = open_internal_dataset(ArtifactStore(locations), source.artifact)
    frame = internal.load_day(date(2024, 1, 3))
    config = _config(source)
    baseline = project_public_day(frame, config)
    changed = frame.with_columns(
        pl.lit(999999, dtype=pl.Int64).alias("source_secid"),
        pl.lit("FORBIDDEN-CALL", dtype=pl.String).alias("call_symbol"),
        pl.lit("FORBIDDEN-PUT", dtype=pl.String).alias("put_symbol"),
        (pl.col("call_option_id") + 100000).cast(pl.Int32),
        (pl.col("put_option_id") + 100000).cast(pl.Int32),
        (pl.col("market_call_log_volume") + 3.0).cast(pl.Float32),
        (pl.col("market_put_log_open_interest") + 3.0).cast(pl.Float32),
    )
    absolute_float32 = (
        "spot",
        "previous_spot",
        "spot_reference",
        "strike",
        "vendor_forward",
        "implied_forward",
        "market_call_ask",
        "market_call_bid",
        "market_put_ask",
        "market_put_bid",
    )
    changed = changed.with_columns(
        *((pl.col(name) * 10.0).cast(pl.Float32).alias(name) for name in absolute_float32)
    )
    projected = project_public_day(changed, config)
    assert projected.row == baseline.row
    assert projected.carry.forward_over_spot == baseline.carry.forward_over_spot
    assert projected.carry.spot == baseline.carry.spot * 10.0


def test_typed_empty_known_bad_date_is_omitted_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    for trading_date in (date(2025, 1, 7), date(2025, 1, 8)):
        _publish_raw_inputs(
            store,
            trading_date,
            expiry=date(2025, 1, 17),
            force_enrichment_rejection=trading_date == date(2025, 1, 7),
        )
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    source_config = replace(
        _dataset_config(date(2025, 1, 7)),
        name="typed-empty-public-source",
        start_date=date(2025, 1, 7),
        end_date=date(2025, 1, 8),
    )
    source = build_internal_dataset(locations, source_config)
    result = build_public_candidate(
        locations,
        _config(
            source,
            omissions=(
                PublicExpectedOmission(
                    date(2025, 1, 8), "known_bad_day_all_tickers"
                ),
            ),
        ),
    )
    assert result.validation is not None
    assert result.validation.date_count == 1
    assert result.validation.omitted_date_count == 1
    evidence = json.loads(
        result.path.joinpath("validation.json").read_text(encoding="utf-8")
    )
    assert evidence["omitted_dates"] == [
        {
            "reason": "known_bad_day_all_tickers",
            "trading_date": "2025-01-08",
        }
    ]


def test_trading_session_half_day_is_distinct_from_expiry_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    trading_date = date(2024, 11, 29)
    _publish_raw_inputs(
        store,
        trading_date,
        expiry=date(2024, 12, 20),
        force_enrichment_rejection=True,
    )
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    source = build_internal_dataset(
        locations,
        replace(
            _dataset_config(trading_date),
            name="half-day-public-source",
        ),
    )
    candidate = build_public_candidate(locations, _config(source))
    row = open_public_dataset(store, candidate.artifact).load().row(0, named=True)
    assert row["is_half_day"] is True
    assert all(expiry["is_am"] is False for expiry in row["expiries"])


def test_semantic_validation_rejects_resigned_parquet_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    result = build_public_candidate(locations, _config(source))
    data_path = result.path / "data.parquet"
    frame = pl.read_parquet(data_path, hive_partitioning=False).with_columns(
        pl.lit(0.0, dtype=pl.Float64).alias("forward_adjusted_spot_return")
    )
    frame.write_parquet(data_path, compression="zstd", compression_level=3, statistics=True)

    _resign_files(result.path, "data.parquet")

    with pytest.raises(PublicDatasetValidationError, match="return vector"):
        validate_public_dataset(ArtifactStore(locations), result.artifact)


def test_missing_internal_source_cannot_build_concrete_candidate(tmp_path: Path) -> None:
    config = PublicProjectionConfig.from_document(
        json.loads(
            (
                Path(__file__).parents[1]
                / "configs"
                / "sanos-public-candidate-reference.json"
            ).read_text(encoding="utf-8")
        )
    )
    with pytest.raises(ArtifactNotFoundError):
        build_public_candidate(_locations(tmp_path), config)


def test_resigned_finite_return_mutation_is_bound_to_projection_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(tmp_path, monkeypatch)
    result = build_public_candidate(locations, _config(source))
    data_path = result.path / "data.parquet"
    frame = pl.read_parquet(data_path, hive_partitioning=False)
    frame[1, "forward_adjusted_spot_return"] = 42.0
    frame.write_parquet(data_path, compression="zstd", compression_level=3, statistics=True)
    _resign_files(result.path, "data.parquet")

    with pytest.raises(PublicDatasetValidationError, match="return vector"):
        validate_public_dataset(ArtifactStore(locations), result.artifact)


def test_resigned_fabricated_omission_is_rejected_by_identity_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    result = build_public_candidate(locations, _config(source))
    validation_path = result.path / "validation.json"
    evidence = json.loads(validation_path.read_text(encoding="utf-8"))
    evidence["omitted_dates"] = [
        {"trading_date": "2023-12-29", "reason": "fabricated"}
    ]
    evidence["summary"]["omitted_date_count"] = 1
    validation_path.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
    _resign_files(result.path, "validation.json")

    with pytest.raises(PublicDatasetValidationError, match="omission policy"):
        validate_public_dataset(ArtifactStore(locations), result.artifact)


def test_extra_checksummed_payload_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    result = build_public_candidate(locations, _config(source))
    extra = result.path / "vendor-raw.csv"
    extra.write_text("bid,ask\n1,2\n", encoding="utf-8")
    manifest_path = result.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append(
        {
            "path": extra.name,
            "byte_size": extra.stat().st_size,
            "sha256": _sha256(extra),
        }
    )
    manifest["files"] = sorted(manifest["files"], key=lambda item: item["path"])
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    completion_path = result.path / COMPLETION_NAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = _sha256(manifest_path)
    completion_path.write_text(canonical_json(completion) + "\n", encoding="utf-8")

    with pytest.raises(PublicDatasetValidationError, match="payload files changed"):
        validate_public_dataset(ArtifactStore(locations), result.artifact)


def test_deep_validation_reprojects_source_bound_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    result = build_public_candidate(locations, _config(source))
    data_path = result.path / "data.parquet"
    frame = pl.read_parquet(data_path, hive_partitioning=False)
    frame = frame.with_columns(
        pl.col("expiries").list.eval(
            pl.element().struct.with_fields(
                (
                    pl.element().struct.field("discounted_forward_over_spot") * 2.0
                ).alias("discounted_forward_over_spot")
            )
        )
    )
    frame.write_parquet(data_path, compression="zstd", compression_level=3, statistics=True)
    validation_path = result.path / "validation.json"
    evidence = json.loads(validation_path.read_text(encoding="utf-8"))
    evidence["logical_content_sha256"] = public_frame_logical_sha256(frame)
    evidence["return_values_sha256"] = public_return_values_sha256(frame)
    validation_path.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
    manifest_path = result.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["logical_content_sha256"] = evidence[
        "logical_content_sha256"
    ]
    manifest["metadata"]["return_values_sha256"] = evidence["return_values_sha256"]
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    _resign_files(result.path, "data.parquet", "validation.json")

    assert validate_public_dataset(ArtifactStore(locations), result.artifact) is not None
    with pytest.raises(PublicDatasetValidationError, match="source re-projection"):
        validate_public_dataset(
            ArtifactStore(locations), result.artifact, deep_lineage=True
        )


def test_daily_front_boundary_mask_and_clock_endpoints() -> None:
    pointer = PublicProjectionConfig.from_document(
        json.loads(
            (
                Path(__file__).parents[1]
                / "configs"
                / "sanos-public-candidate-reference.json"
            ).read_text(encoding="utf-8")
        )
    ).source_dataset
    config = PublicProjectionConfig(
        name="boundary",
        source_dataset=pointer,
        coverage_start_date=date(2024, 1, 3),
        coverage_end_date=date(2024, 1, 3),
        overnight_variance_fraction=0.0,
    )
    assert config.overnight_variance_fraction == 0.0
    assert replace(config, overnight_variance_fraction=1.0).overnight_variance_fraction == 1.0
    time_years = 10.0 / business_clock_policy()[0]
    atm = 0.4
    epsilon = 0.4
    bounds = np.asarray(
        NMS.from_nms(
            np.asarray(config.nms_range),
            time_years,
            volatility=atm,
            epsilon_volatility=epsilon,
        ),
        dtype=np.float64,
    )
    lower_inside = np.nextafter(np.float32(bounds[0]), np.float32(np.inf))
    upper_inside = np.nextafter(np.float32(bounds[1]), np.float32(-np.inf))
    lower_outside = np.nextafter(lower_inside, np.float32(-np.inf))
    while float(lower_outside) >= bounds[0]:
        lower_outside = np.nextafter(lower_outside, np.float32(-np.inf))
    upper_outside = np.nextafter(upper_inside, np.float32(np.inf))
    while float(upper_outside) <= bounds[1]:
        upper_outside = np.nextafter(upper_outside, np.float32(np.inf))
    strikes = np.asarray(
        [lower_outside, lower_inside, 1.0, upper_inside, upper_outside],
        dtype=np.float32,
    )
    common = {
        "time_years": time_years,
        "strikes": strikes,
        "active": np.ones(5, dtype=bool),
        "is_american": np.zeros(5, dtype=bool),
        "smooth_available": np.ones(5, dtype=bool),
        "smooth_fit": np.ones(5, dtype=np.float64),
        "atm_volatility": atm,
        "epsilon_atm_volatility": epsilon,
        "config": config,
    }
    assert _daily_front_mask(business_dte=70.0, **common).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert not _daily_front_mask(business_dte=70.0001, **common).any()
    assert not _daily_front_mask(business_dte=0.2, **common).any()
    assert not _daily_front_mask(business_dte=0.20009, **common).any()
    assert _daily_front_mask(business_dte=0.20011, **common).any()
    inactive = dict(common)
    inactive["active"] = np.zeros(5, dtype=bool)
    assert not _daily_front_mask(business_dte=10.0, **inactive).any()
    american = dict(common)
    american["is_american"] = np.ones(5, dtype=bool)
    assert not _daily_front_mask(business_dte=10.0, **american).any()
    unavailable = dict(common)
    unavailable["smooth_available"] = np.zeros(5, dtype=bool)
    assert not _daily_front_mask(business_dte=10.0, **unavailable).any()
    nonfinite = dict(common)
    nonfinite["smooth_fit"] = np.full(5, np.nan)
    assert not _daily_front_mask(business_dte=10.0, **nonfinite).any()


def test_carry_curve_uses_non_model_source_nodes() -> None:
    frame = pl.DataFrame(
        {
            "business_dte": [1.0, 2.0, 2.0],
            "implied_forward": [101.0, 104.0, 104.0],
            "discount_factor": [0.99, 0.97, 0.97],
        }
    )
    curve = _source_carry_curve(date(2024, 1, 3), 100.0, frame)
    assert curve.business_dtes == (0.0, 1.0, 2.0)
    assert curve.forward_over_spot == (1.0, 1.01, 1.04)
    assert curve.discount_factors == (1.0, 0.99, 0.97)


def test_approved_release_is_distinct_reprojected_and_output_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(tmp_path, monkeypatch)
    projection = _config(source)
    candidate = build_public_candidate(locations, projection)
    license_bytes = b"Synthetic test license; not a real redistribution approval.\n"
    release_config = _release_config(locations, projection, candidate, license_bytes)
    with pytest.raises(ValueError, match="license bytes"):
        build_public_release(
            locations,
            replace(
                release_config,
                approval=replace(release_config.approval, license_sha256="e" * 64),
            ),
            license_bytes,
        )
    released = build_public_release(locations, release_config, license_bytes)
    assert released.built is True
    assert released.artifact != candidate.artifact
    assert released.artifact.namespace.startswith("sanos/public/name-")
    release_dataset = open_public_dataset(ArtifactStore(locations), released.artifact)
    assert release_dataset.distribution.value == "public"
    assert release_dataset.load().equals(
        open_public_dataset(ArtifactStore(locations), candidate.artifact).load(),
        null_equal=True,
    )
    assert (released.path / "LICENSE.txt").read_bytes() == license_bytes

    hidden_source = tmp_path / "hidden-release-source"
    hidden_candidate = tmp_path / "hidden-release-candidate"
    source.path.rename(hidden_source)
    candidate.path.rename(hidden_candidate)
    try:
        reused = build_public_release(
            locations,
            release_config,
            None,
            PublicBuildOptions(version_override="*"),
        )
        assert reused.built is False
        assert reused.source_accessed is False
        assert validate_public_dataset(ArtifactStore(locations), reused.artifact) is not None
    finally:
        hidden_source.rename(source.path)
        hidden_candidate.rename(candidate.path)

    release_frame = release_dataset.load().with_columns(
        pl.when(pl.col("trading_date") == pl.col("trading_date").max())
        .then(pl.lit(42.0))
        .otherwise(pl.col("forward_adjusted_spot_return"))
        .alias("forward_adjusted_spot_return")
    )
    release_frame.write_parquet(
        released.path / "data.parquet",
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    release_validation_path = released.path / "validation.json"
    release_evidence = json.loads(
        release_validation_path.read_text(encoding="utf-8")
    )
    release_evidence["logical_content_sha256"] = public_frame_logical_sha256(
        release_frame
    )
    release_evidence["return_values_sha256"] = public_return_values_sha256(
        release_frame
    )
    release_validation_path.write_text(
        canonical_json(release_evidence) + "\n", encoding="utf-8"
    )
    release_manifest_path = released.path / MANIFEST_NAME
    release_manifest = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    release_manifest["metadata"]["logical_content_sha256"] = release_evidence[
        "logical_content_sha256"
    ]
    release_manifest["metadata"]["return_values_sha256"] = release_evidence[
        "return_values_sha256"
    ]
    release_manifest_path.write_text(
        canonical_json(release_manifest) + "\n", encoding="utf-8"
    )
    _resign_files(released.path, "data.parquet", "validation.json")
    with pytest.raises(
        PublicDatasetValidationError,
        match="release logical content differs from the approved candidate",
    ):
        validate_public_dataset(ArtifactStore(locations), released.artifact)


def test_public_validation_version_dependencies_are_exact() -> None:
    payload_dependencies = set(dict(validate_public_payload.version.dependencies[1]))
    assert payload_dependencies == {
        "DistributionStatus",
        "PublicDatasetValidationSummary",
        "PublicProjectionConfig",
        "PublicReleaseConfig",
        "_payload_names",
        "_read_json",
        "_validate_coverage_evidence",
        "candidate_readme",
        "public_capabilities",
        "public_coverage_sha256",
        "public_frame_logical_sha256",
        "public_manifest_metadata",
        "public_return_values_sha256",
        "public_schema_document",
        "release_readme",
        "validate_public_frame",
    }
    assert "public_schema_document" in dict(
        public_frame_logical_sha256.version.dependencies[1]
    )
    assert "PublicBuildOptions" not in dict(
        public_publish_module._project_source_dataset.version.dependencies[1]
    )
    assert set(dict(public_coverage_sha256.version.dependencies[1])) == {
        "PublicProjectionConfig",
        "canonical_sha256",
    }


def test_release_staging_failure_cleans_up_and_concurrent_publishers_converge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    projection = _config(source)
    candidate = build_public_candidate(locations, projection)
    license_bytes = b"Synthetic concurrent release test license.\n"
    release_config = _release_config(
        locations, projection, candidate, license_bytes
    )
    plan = plan_public_release(release_config)

    def fail_staged(*args, **kwargs):
        raise PublicDatasetValidationError("injected release staged failure")

    with monkeypatch.context() as staged_patch:
        staged_patch.setattr(
            public_publish_module, "validate_public_payload", fail_staged
        )
        with pytest.raises(
            PublicDatasetValidationError, match="injected release staged failure"
        ):
            build_public_release(locations, release_config, license_bytes)
    final_path = ArtifactStore(locations).path_for(plan.dataset_spec.pointer)
    assert not final_path.exists()
    assert not list(final_path.parent.glob(f".staging-{plan.dataset_spec.artifact_id}-*"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda _: build_public_release(
                    locations, release_config, license_bytes
                ),
                range(2),
            )
        )
    assert sum(result.built for result in results) == 1
    assert results[0].artifact == results[1].artifact
    assert all(result.validation is not None for result in results)


def test_public_release_cli_template_approve_and_validate_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    projection = _config(source)
    candidate = build_public_candidate(locations, projection)
    license_path = tmp_path / "test-license.txt"
    license_path.write_bytes(b"Synthetic CLI release test license.\n")
    release_config_path = tmp_path / "release-config.json"
    runner = CliRunner()
    common = ["--root", str(tmp_path), "--log-format", "json"]
    environment = {"WRDS_OFF": "1", "WRDS_USERNAME": "must-not-be-read"}

    template = runner.invoke(
        main,
        [
            *common,
            "dataset",
            "public",
            "approval-template",
            "--candidate-name",
            projection.name,
            "--candidate-artifact-id",
            candidate.artifact.artifact_id,
            "--release-license",
            "TEST-ONLY",
            "--redistribution-basis",
            "Synthetic CLI fixture only",
            "--approver",
            "pytest",
            "--approval-date",
            "2026-10-12",
            "--approval-reference",
            "test://phase-8-cli-release",
            "--license-file",
            str(license_path),
            "--output",
            str(release_config_path),
        ],
        env=environment,
    )
    assert template.exit_code == 0, template.output
    template_events = [json.loads(line) for line in template.output.splitlines()]
    assert template_events[-1]["event"] == "public_release_approval_template"
    assert template_events[-1]["status"] == "written"

    approved = runner.invoke(
        main,
        [
            *common,
            "dataset",
            "public",
            "approve",
            "--config",
            str(release_config_path),
            "--license-file",
            str(license_path),
        ],
        env=environment,
    )
    assert approved.exit_code == 0, approved.output
    approval_events = [json.loads(line) for line in approved.output.splitlines()]
    event_names = [item["event"] for item in approval_events]
    assert "public_release_projection_progress" in event_names
    assert "public_candidate_projection_progress" not in event_names
    assert approval_events[-1]["event"] == "public_release_result"
    assert approval_events[-1]["distribution"] == "public"
    assert approval_events[-1]["built"] is True
    assert approval_events[-1]["network_accessed"] is False
    release_identity = approval_events[-1]["identity"]
    release_path = Path(approval_events[-1]["path"])
    documented_command = (
        "sanos dataset public validate --distribution public "
        + "\\"
        + "\n"
        f"  --name {projection.name} --artifact-id {release_identity}"
    )
    assert documented_command in release_path.joinpath("README.md").read_text(
        encoding="utf-8"
    )

    validated = runner.invoke(
        main,
        [
            *common,
            "dataset",
            "public",
            "validate",
            "--distribution",
            "public",
            "--name",
            projection.name,
            "--artifact-id",
            release_identity,
        ],
        env=environment,
    )
    assert validated.exit_code == 0, validated.output
    validation_events = [json.loads(line) for line in validated.output.splitlines()]
    progress_events = [
        item for item in validation_events if item["event"] == "public_validation_progress"
    ]
    assert progress_events
    assert {
        "elapsed_seconds",
        "completed",
        "total",
        "failed",
        "estimated_remaining_seconds",
    } <= set(progress_events[-1])
    assert validation_events[-1]["event"] == "public_dataset_validation_result"
    assert validation_events[-1]["distribution"] == "public"
    assert validation_events[-1]["network_accessed"] is False


def test_typed_empty_gap_uses_previous_published_date_and_two_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    for trading_date in (date(2025, 1, 7), date(2025, 1, 8), date(2025, 1, 10)):
        _publish_raw_inputs(
            store,
            trading_date,
            expiry=date(2025, 1, 17),
            force_enrichment_rejection=trading_date != date(2025, 1, 8),
        )
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    source = build_internal_dataset(
        locations,
        replace(
            _dataset_config(date(2025, 1, 7)),
            name="typed-empty-gap-source",
            start_date=date(2025, 1, 7),
            end_date=date(2025, 1, 10),
        ),
    )
    config = _config(
        source,
        omissions=(
            PublicExpectedOmission(date(2025, 1, 8), "known_bad_day_all_tickers"),
        ),
    )
    candidate = build_public_candidate(locations, config)
    frame = open_public_dataset(store, candidate.artifact).load()
    assert frame["trading_date"].to_list() == [date(2025, 1, 7), date(2025, 1, 10)]
    evidence = json.loads(
        candidate.path.joinpath("validation.json").read_text(encoding="utf-8")
    )
    assert evidence["return_evidence"] == [
        {
            "business_sessions": 2,
            "discount_curve_validated": True,
            "previous_trading_date": "2025-01-07",
            "trading_date": "2025-01-10",
        }
    ]
    internal = open_internal_dataset(store, source.artifact)
    previous = internal.load_day(date(2025, 1, 7))
    current = internal.load_day(date(2025, 1, 10))
    dte = float(previous["business_dte"].drop_nulls().unique().item())
    carry = (
        float(previous["implied_forward"].drop_nulls().unique().item())
        / float(previous["spot_reference"].item(0))
    ) ** (2.0 / dte)
    expected = (
        float(current["spot_reference"].item(0))
        / float(previous["spot_reference"].item(0))
    ) / carry - 1.0
    assert frame[1, "forward_adjusted_spot_return"] == pytest.approx(expected)


def test_public_state_reconstructs_source_pricers_across_cd014_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    candidate = build_public_candidate(locations, _config(source))
    public_row = open_public_dataset(ArtifactStore(locations), candidate.artifact).load().row(
        0, named=True
    )
    internal_frame = open_internal_dataset(
        ArtifactStore(locations), source.artifact
    ).load_day(date(2024, 1, 3))
    source_dlv = compatibility_dlv_pricer(internal_frame)
    source_smooth = smooth_sanos_pricer(internal_frame)
    public_dlv, *_ = _state_pricer(
        public_row["discrete_dlv"], public_row["expiries"], smooth=False
    )
    public_smooth, *_ = _state_pricer(
        public_row["smooth_sanos"], public_row["expiries"], smooth=True
    )
    first = float(source_dlv.expiries[0])
    query_times = (0.0, ONE_MINUTE / 2.0, ONE_MINUTE, first / 2.0, first, first * 2.0)
    strikes = np.asarray([0.83, 1.0, 1.17], dtype=np.float64)
    for model_time in query_times:
        np.testing.assert_allclose(
            public_dlv(model_time, strikes),
            source_dlv(model_time, strikes),
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            public_smooth(model_time, strikes),
            source_smooth(model_time, strikes),
            rtol=0.0,
            atol=1.0e-13,
        )
        calls = np.asarray(public_dlv(model_time, strikes))
        puts = calls - 1.0 + strikes
        np.testing.assert_allclose(calls - puts, 1.0 - strikes, atol=1.0e-15)
    _, public_atm = public_dlv(first, np.asarray([1.0]), and_vols=True)
    _, source_atm = source_dlv(first, np.asarray([1.0]), and_vols=True)
    np.testing.assert_allclose(public_atm, source_atm, rtol=0.0, atol=1.0e-13)


def test_staged_validation_failure_exposes_no_final_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    config = _config(source)
    plan = plan_public_candidate(config)

    def fail_staged(*args, **kwargs):
        raise PublicDatasetValidationError("injected staged failure")

    monkeypatch.setattr(public_publish_module, "validate_public_payload", fail_staged)
    with pytest.raises(PublicDatasetValidationError, match="injected staged failure"):
        build_public_candidate(locations, config)
    final_path = ArtifactStore(locations).path_for(plan.dataset_spec.pointer)
    assert not final_path.exists()
    assert not list(final_path.parent.glob(f".staging-{plan.dataset_spec.artifact_id}-*"))


def test_zero_overnight_fraction_builds_when_source_clock_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trading_date = date(2024, 1, 3)
    locations = _locations(tmp_path)
    store = ArtifactStore(locations)
    _publish_raw_inputs(store, trading_date)
    monkeypatch.setattr(internal_dataset_module, "_source_revision", lambda: "f" * 40)
    base = _dataset_config(trading_date)
    source = build_internal_dataset(
        locations,
        replace(
            base,
            name="zero-overnight-source",
            daily=replace(
                base.daily,
                clock=replace(
                    base.daily.clock,
                    overnight_variance_fraction=0.0,
                ),
            ),
        ),
    )
    config = replace(_config(source), overnight_variance_fraction=0.0)
    candidate = build_public_candidate(locations, config)
    assert candidate.validation is not None
    assert candidate.validation.date_count == 1


def test_duplicate_public_book_key_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    config = _config(source)
    candidate = build_public_candidate(locations, config)
    frame = open_public_dataset(ArtifactStore(locations), candidate.artifact).load()
    rows = frame.to_dicts()
    rows[0]["books"].insert(1, dict(rows[0]["books"][0]))
    corrupted = pl.DataFrame(rows, schema=public_polars_schema(), strict=True)
    with pytest.raises(PublicDatasetValidationError, match="duplicate"):
        validate_public_frame(corrupted, config)


def test_concurrent_identical_candidate_publishers_converge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    config = _config(source)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(lambda _: build_public_candidate(locations, config), range(2))
        )
    assert sum(result.built for result in results) == 1
    assert results[0].artifact == results[1].artifact
    assert all(result.validation is not None for result in results)


def test_concurrent_public_validation_preserves_absent_wrds_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations, source = _publish_source(
        tmp_path,
        monkeypatch,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
    )
    candidate = build_public_candidate(locations, _config(source))
    monkeypatch.delenv("WRDS_OFF", raising=False)
    store = ArtifactStore(locations)
    with ThreadPoolExecutor(max_workers=2) as pool:
        summaries = tuple(
            pool.map(
                lambda _: validate_public_dataset(store, candidate.artifact),
                range(2),
            )
        )
    assert all(summary is not None for summary in summaries)
    assert "WRDS_OFF" not in os.environ
