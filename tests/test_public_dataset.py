from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

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
    public_schema_document,
    validate_public_dataset,
    validate_public_frame,
)
from sanos.datasets.public.documents import candidate_readme
from sanos.datasets.public.project import (
    _book_payload,
    _daily_front_mask,
    _inflate_public_day,
    _log_curve_value,
    _model_iv_fields,
    _source_carry_curve,
)
from sanos.datasets.public.validate import (
    _state_pricer,
    _validate_calendar_calls,
    public_coverage_sha256,
    public_frame_logical_sha256,
    public_return_values_sha256,
    public_validation_document,
    validate_public_payload,
)
from sanos.model.tools import SanosPricer
from sanos.widb.calendar import (
    NMS,
    ONE_MINUTE,
    business_clock_policy,
    count_business_days_between,
)
from sanos.widb.surface import initial_no_arb_pricer, smooth_sanos_pricer
from tests.test_widb_build import _dataset_config, _locations, _publish_raw_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_days(frame: pl.DataFrame) -> list[dict[str, object]]:
    dates = frame.filter(pl.col("record_kind") == "day")["trading_date"].to_list()
    return [
        _inflate_public_day(frame.filter(pl.col("trading_date") == trading_date))
        for trading_date in dates
    ]


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


@pytest.fixture(scope="module")
def public_frame_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[PublicProjectionConfig, pl.DataFrame]:
    """One immutable two-day candidate frame cloned by semantic tests."""

    root = tmp_path_factory.mktemp("public-frame")
    patch = pytest.MonkeyPatch()
    try:
        locations, source = _publish_source(root, patch)
    finally:
        patch.undo()
    config = _config(source)
    built = build_public_candidate(locations, config)
    frame = open_public_dataset(ArtifactStore(locations), built.artifact).load()
    return config, frame


def test_public_configuration_is_strict_and_identity_sensitive() -> None:
    document = json.loads(
        (Path(__file__).parents[1] / "configs" / "sanos-public-candidate-reference.json").read_text(
            encoding="utf-8"
        )
    )
    config = PublicProjectionConfig.from_document(document)
    assert PublicProjectionConfig.from_document(config.to_document()) == config
    original = plan_public_candidate(config).dataset_spec.artifact_id
    experiment_document = json.loads(
        (
            Path(__file__).parents[1]
            / "configs"
            / "dysanos-daily-front-reference.json"
        ).read_text(encoding="utf-8")
    )
    assert experiment_document["experiment"]["source_dataset"]["artifact_id"] == original
    changed = replace(config, exclusion_tolerance=2.0e-4)
    assert plan_public_candidate(changed).dataset_spec.artifact_id != original
    assert plan_public_candidate(config).dataset_spec.artifact_id == original
    assert config.excluded_business_dtes == (0.0, 0.2)
    include_zero = replace(config, excluded_business_dtes=(0.2,))
    assert plan_public_candidate(include_zero).dataset_spec.artifact_id != original

    malformed = json.loads(json.dumps(document))
    malformed["machine_path"] = "/private/data"
    with pytest.raises(ValueError, match="only 'public_dataset'"):
        PublicProjectionConfig.from_document(malformed)

    malformed = json.loads(json.dumps(document))
    malformed["public_dataset"]["european_only"] = 1
    with pytest.raises(ValueError, match="booleans"):
        PublicProjectionConfig.from_document(malformed)

    for field, legacy_value in (
        ("return_convention", "previous-retained-log-carry-v1"),
        ("unit_convention", "discounted-forward-over-spot-v1"),
    ):
        malformed = json.loads(json.dumps(document))
        malformed["public_dataset"][field] = legacy_value
        with pytest.raises(ValueError, match="Unsupported public"):
            PublicProjectionConfig.from_document(malformed)


def test_zero_time_public_book_has_six_null_model_ivs() -> None:
    fields = {
        **_model_iv_fields("sanos", 1.0, 0.0, 0.01, 0.0),
        **_model_iv_fields("dlv", 1.0, 0.0, 0.01, 0.0),
    }
    assert set(fields) == {
        "sanos_fit_annual_iv",
        "sanos_synthetic_bid_annual_iv",
        "sanos_synthetic_ask_annual_iv",
        "dlv_fit_annual_iv",
        "dlv_synthetic_bid_annual_iv",
        "dlv_synthetic_ask_annual_iv",
    }
    assert all(value is None for value in fields.values())


def test_public_full_support_calendar_violation_is_rejected() -> None:
    support = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)
    pricer = SanosPricer(
        smoothness=None,
        expiries=np.asarray([1.0 / 252.0, 2.0 / 252.0]),
        variances=np.asarray([0.04, 0.08]),
        xstrikes=[support, support],
        xdensities=[
            np.asarray([0.25, 0.5, 0.25]),
            np.asarray([0.0, 1.0, 0.0]),
        ],
    )
    with pytest.raises(PublicDatasetValidationError, match="full-support calendar"):
        _validate_calendar_calls(pricer, "test DLV")


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
    days = _public_days(frame)
    assert days[0]["driftless_spot_return"] is None
    assert days[1]["driftless_spot_return"] < 0.0
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
    discount_at_one_session = float(
        previous_day.get_column("discount_factor").drop_nulls().unique().item()
    ) ** (1.0 / business_dte)
    expected_return = (
        float(current_day.get_column("spot_reference").item(0))
        / float(previous_day.get_column("spot_reference").item(0))
    ) * discount_at_one_session / carry_at_one_session - 1.0
    assert days[1]["driftless_spot_return"] == pytest.approx(
        expected_return,
        abs=1.0e-14,
    )
    assert "discounted_forward_over_spot" not in frame.columns
    assert all(not isinstance(dtype, pl.List | pl.Struct) for dtype in frame.dtypes)
    assert set(frame["record_kind"]) == {
        "day", "expiry", "book", "smooth_node", "dlv_node"
    }
    first_public = days[0]
    first_book = first_public["books"][0]
    first_expiry = first_public["expiries"][first_book["expiry_index"]]
    source_book = previous_day.filter(
        (pl.col("expiry_date") == first_expiry["expiry_date"])
        & (pl.col("expiry_is_am") == first_expiry["is_am"])
        & (pl.col("normalized_strike") == first_book["normalized_strike"])
    )
    assert source_book.height == 1
    side = "call" if first_book["is_call"] else "put"
    expected_half_spread = 0.5 * (
        float(source_book[f"pure_market_{side}_ask"].item())
        - float(source_book[f"pure_market_{side}_bid"].item())
    )
    assert first_book["half_spread"] == pytest.approx(expected_half_spread, abs=0.0)
    assert first_book["sanos_fit_annual_iv"] > 0.0
    assert first_book["dlv_fit_annual_iv"] > 0.0
    assert all(
        first_book[field] is None or first_book[field] > 0.0
        for field in (
            "sanos_fit_annual_iv",
            "sanos_synthetic_bid_annual_iv",
            "sanos_synthetic_ask_annual_iv",
            "dlv_fit_annual_iv",
            "dlv_synthetic_bid_annual_iv",
            "dlv_synthetic_ask_annual_iv",
        )
    )
    public_sanos, *_ = _state_pricer(
        first_public["smooth_sanos"], first_public["expiries"], smooth=True
    )
    public_dlv, *_ = _state_pricer(
        first_public["dlv"], first_public["expiries"], smooth=False
    )
    source_excesses = {"sanos": [], "dlv": []}
    for book in first_public["books"]:
        expiry = first_public["expiries"][book["expiry_index"]]
        source_row = previous_day.filter(
            (pl.col("expiry_date") == expiry["expiry_date"])
            & (pl.col("expiry_is_am") == expiry["is_am"])
            & (pl.col("normalized_strike") == book["normalized_strike"])
        )
        side = "call" if book["is_call"] else "put"
        bid = float(source_row[f"pure_market_{side}_bid"].item())
        ask = float(source_row[f"pure_market_{side}_ask"].item())
        for label, pricer in (("sanos", public_sanos), ("dlv", public_dlv)):
            call = float(
                pricer(
                    expiry["time_to_expiry_years"],
                    np.asarray([book["normalized_strike"]]),
                )[0]
            )
            model = call if book["is_call"] else call - 1.0 + book["normalized_strike"]
            source_excesses[label].append(
                (max(model - ask, 0.0) + max(bid - model, 0.0))
                / book["half_spread"]
            )
    for label, excesses in source_excesses.items():
        positive = [value for value in excesses if value > 0.0]
        assert first_public[f"{label}_market_percent_books_outside"] == pytest.approx(
            100.0 * len(positive) / len(excesses), abs=0.0
        )
        assert first_public[
            f"{label}_market_conditional_mean_excess_halfspreads"
        ] == pytest.approx(float(np.mean(positive)) if positive else 0.0, abs=0.0)
        assert first_public[
            f"{label}_market_conditional_median_excess_halfspreads"
        ] == pytest.approx(float(np.median(positive)) if positive else 0.0, abs=0.0)
    assert built.path.joinpath("capabilities.json").read_text(encoding="utf-8") == (
        json.dumps(public_capabilities(), sort_keys=True, separators=(",", ":")) + "\n"
    )

    hidden = tmp_path / "hidden-source"
    source.path.rename(hidden)
    monkeypatch.delenv("WRDS_OFF", raising=False)
    reused = build_public_candidate(
        locations,
        config,
        PublicBuildOptions(version_override="*"),
    )
    assert reused.built is False
    assert reused.source_accessed is False
    assert reused.reuse is not None and reused.reuse.used_wildcard
    store = ArtifactStore(locations)
    with ThreadPoolExecutor(max_workers=2) as pool:
        summaries = tuple(
            pool.map(lambda _: validate_public_dataset(store, reused.artifact), range(2))
        )
    assert all(summary is not None for summary in summaries)
    with pytest.raises(ArtifactNotFoundError):
        validate_public_dataset(
            ArtifactStore(locations),
            reused.artifact,
            deep_lineage=True,
        )
    assert "WRDS_OFF" not in os.environ
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
    public_frame_fixture: tuple[PublicProjectionConfig, pl.DataFrame],
) -> None:
    config, shared_frame = public_frame_fixture
    frame = shared_frame.clone()
    with pytest.raises(PublicDatasetValidationError, match="flat scalar schema changed"):
        validate_public_frame(frame.with_columns(pl.lit(None).alias("spot")), config)
    corrupted = frame.with_columns(
        pl.when(pl.col("record_kind") == "day")
        .then(pl.lit(float("inf"), dtype=pl.Float64))
        .otherwise(pl.col("nms_epsilon_atm_annual_volatility"))
        .alias("nms_epsilon_atm_annual_volatility")
    )
    with pytest.raises(PublicDatasetValidationError, match="non-finite"):
        validate_public_frame(corrupted, config)


def test_public_validation_allows_independent_model_expiry_populations(
    public_frame_fixture: tuple[PublicProjectionConfig, pl.DataFrame],
) -> None:
    config, shared_frame = public_frame_fixture
    frame = shared_frame.clone()
    row = _public_days(frame)[0]

    extra_expiry_row = copy.deepcopy(row)
    last = extra_expiry_row["expiries"][-1]
    expiry_date = last["expiry_date"] + timedelta(days=7)
    business_dte = float(count_business_days_between(row["trading_date"], expiry_date))
    extra_expiry_row["expiries"].append(
        {
            **last,
            "expiry_index": len(extra_expiry_row["expiries"]),
            "expiry_date": expiry_date,
            "calendar_dte": (expiry_date - row["trading_date"]).days,
            "business_dte": business_dte,
            "is_am": False,
            "time_to_expiry_years": business_dte / business_clock_policy()[0],
        }
    )
    from sanos.datasets.public.project import _flatten_public_day

    uncovered = _flatten_public_day(extra_expiry_row)
    with pytest.raises(
        PublicDatasetValidationError,
        match="complete union",
    ):
        validate_public_frame(uncovered, config)

    previous_dlv = extra_expiry_row["dlv"]["model_expiries"][-1]
    extra_expiry_row["dlv"]["model_expiries"].append(
        {
            **copy.deepcopy(previous_dlv),
            "expiry_index": len(extra_expiry_row["expiries"]) - 1,
            "time_increment_years": (
                extra_expiry_row["expiries"][-1]["time_to_expiry_years"]
                - row["expiries"][-1]["time_to_expiry_years"]
            ),
            "input_total_variance": 2.0 * previous_dlv["input_total_variance"],
        }
    )
    independent = _flatten_public_day(extra_expiry_row)
    summary = validate_public_frame(independent, config)
    assert summary.smooth_state_count + 1 == summary.dlv_state_count


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
    assert capabilities["normalized_dlv_pricing"] is True
    assert capabilities["independent_model_expiry_populations"] is True
    assert capabilities["complete_initial_dlv_expiry_chain"] is True
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
    row = _public_days(open_public_dataset(store, candidate.artifact).load())[0]
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
        pl.lit(0.0, dtype=pl.Float64).alias("driftless_spot_return")
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
    day_indices = frame.with_row_index().filter(pl.col("record_kind") == "day")["index"]
    frame[int(day_indices[1]), "driftless_spot_return"] = 42.0
    frame.write_parquet(data_path, compression="zstd", compression_level=3, statistics=True)
    _resign_files(result.path, "data.parquet")

    with pytest.raises(PublicDatasetValidationError, match="return vector"):
        validate_public_dataset(ArtifactStore(locations), result.artifact)


@pytest.mark.parametrize("invalid_return", [-1.0, -1.000001])
def test_public_semantics_reject_nonpositive_driftless_path_increment(
    public_frame_fixture: tuple[PublicProjectionConfig, pl.DataFrame],
    invalid_return: float,
) -> None:
    config, shared_frame = public_frame_fixture
    day_indices = shared_frame.with_row_index().filter(pl.col("record_kind") == "day")[
        "index"
    ]
    frame = shared_frame.clone()
    frame[int(day_indices[1]), "driftless_spot_return"] = invalid_return
    with pytest.raises(PublicDatasetValidationError, match="greater than -1"):
        validate_public_frame(frame, config)


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


def test_deep_validation_reprojects_source_bound_half_spread(
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
        pl.when(pl.col("record_kind") == "day")
        .then(pl.lit(value, dtype=pl.Float64))
        .otherwise(pl.col(column))
        .alias(column)
        for column, value in (
            ("sanos_market_percent_books_outside", 99.9),
            ("sanos_market_conditional_mean_excess_halfspreads", 1.234),
            ("sanos_market_conditional_median_excess_halfspreads", 1.345),
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
    assert not _daily_front_mask(business_dte=0.0, **common).any()
    assert not _daily_front_mask(business_dte=0.2, **common).any()
    assert not _daily_front_mask(business_dte=0.20009, **common).any()
    assert _daily_front_mask(business_dte=0.20011, **common).any()
    include_zero = dict(common)
    include_zero["config"] = replace(config, excluded_business_dtes=(0.2,))
    assert _daily_front_mask(business_dte=0.0, **include_zero).any()
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


def test_public_book_projection_excludes_zero_dte_but_preserves_later_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {
            "options_filter_eligible": [True, True],
            "expiry_id": [202401030, 202401100],
            "time_to_expiry_years": [0.0, 1.0 / 252.0],
            "normalized_strike": [1.0, 1.0],
            "call_active": [True, True],
            "put_active": [True, True],
            "is_american": [False, False],
            "smooth_mid_available": [True, True],
            "smooth_call_mid": [0.01, 0.02],
            "smooth_put_mid": [0.01, 0.02],
            "pure_market_call_bid": [0.005, 0.015],
            "pure_market_call_ask": [0.015, 0.025],
            "pure_market_put_bid": [0.005, 0.015],
            "pure_market_put_ask": [0.015, 0.025],
        }
    )
    records = (
        {"expiry_id": 202401030, "business_dte": 0.0, "time_to_expiry_years": 0.0},
        {
            "expiry_id": 202401100,
            "business_dte": 1.0,
            "time_to_expiry_years": 1.0 / 252.0,
        },
    )

    class _Pricer:
        def __call__(self, _time: float, strikes: np.ndarray) -> np.ndarray:
            return np.full(strikes.shape, 0.02, dtype=np.float64)

    monkeypatch.setattr(
        "sanos.datasets.public.project._atm_volatility",
        lambda _time, _pricer: np.asarray(0.2),
    )
    books, _ = _book_payload(
        frame,
        expiry_records=records,
        public_index_by_expiry_id={202401030: 0, 202401100: 1},
        surface_pricer=_Pricer(),  # type: ignore[arg-type]
        dlv_pricer=_Pricer(),  # type: ignore[arg-type]
        epsilon_atm_vol=0.2,
        config=PublicProjectionConfig(
            name="zero-dte-filter",
            source_dataset=PublicProjectionConfig.from_document(
                json.loads(
                    (
                        Path(__file__).parents[1]
                        / "configs"
                        / "sanos-public-candidate-reference.json"
                    ).read_text(encoding="utf-8")
                )
            ).source_dataset,
            coverage_start_date=date(2024, 1, 3),
            coverage_end_date=date(2024, 1, 3),
        ),
    )
    assert books and {book["expiry_index"] for book in books} == {1}

    with pytest.raises(PublicProjectionError, match="no public daily-front books"):
        _book_payload(
            frame.filter(pl.col("expiry_id") == 202401030),
            expiry_records=records[:1],
            public_index_by_expiry_id={202401030: 0},
            surface_pricer=_Pricer(),  # type: ignore[arg-type]
            dlv_pricer=_Pricer(),  # type: ignore[arg-type]
            epsilon_atm_vol=0.2,
            config=PublicProjectionConfig(
                name="only-zero-dte",
                source_dataset=PublicProjectionConfig.from_document(
                    json.loads(
                        (
                            Path(__file__).parents[1]
                            / "configs"
                            / "sanos-public-candidate-reference.json"
                        ).read_text(encoding="utf-8")
                    )
                ).source_dataset,
                coverage_start_date=date(2024, 1, 3),
                coverage_end_date=date(2024, 1, 3),
            ),
        )


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
        .otherwise(pl.col("driftless_spot_return"))
        .alias("driftless_spot_return")
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
    assert str(_state_pricer.version).startswith("4.0 ")
    assert str(public_schema_document.version).startswith("4.0 ")
    assert str(public_capabilities.version) == "4.0"
    assert str(validate_public_frame.version).startswith("3.0 ")
    assert str(public_validation_document.version).startswith("2.0 ")
    assert str(candidate_readme.version).startswith("2.0 ")
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
    assert set(dict(_state_pricer.version.dependencies[1])) == {
        "SanosPricer",
        "_validate_calendar_calls",
        "_validate_density",
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
    days = _public_days(frame)
    assert [row["trading_date"] for row in days] == [
        date(2025, 1, 7), date(2025, 1, 10)
    ]
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
    discount = float(previous["discount_factor"].drop_nulls().unique().item()) ** (
        2.0 / dte
    )
    expected = (
        float(current["spot_reference"].item(0))
        / float(previous["spot_reference"].item(0))
    ) * discount / carry - 1.0
    assert days[1]["driftless_spot_return"] == pytest.approx(expected)


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
    public_row = _public_days(
        open_public_dataset(ArtifactStore(locations), candidate.artifact).load()
    )[0]
    internal_frame = open_internal_dataset(
        ArtifactStore(locations), source.artifact
    ).load_day(date(2024, 1, 3))
    source_dlv = initial_no_arb_pricer(internal_frame)
    source_smooth = smooth_sanos_pricer(internal_frame)
    public_dlv, *_ = _state_pricer(
        public_row["dlv"], public_row["expiries"], smooth=False
    )
    public_smooth, *_ = _state_pricer(
        public_row["smooth_sanos"], public_row["expiries"], smooth=True
    )
    source_dlv_expiry_dates = (
        internal_frame.filter(pl.col("initial_no_arb_input_expiry_index").is_not_null())
        .sort("initial_no_arb_input_expiry_index")
        .group_by("initial_no_arb_input_expiry_index", maintain_order=True)
        .agg(pl.col("expiry_date").first())
        .get_column("expiry_date")
        .to_list()
    )
    public_dlv_expiry_dates = [
        public_row["expiries"][item["expiry_index"]]["expiry_date"]
        for item in public_row["dlv"]["model_expiries"]
    ]
    assert public_dlv_expiry_dates == source_dlv_expiry_dates
    assert len(public_dlv.xstrikes) == len(source_dlv.xstrikes)
    for public_support, source_support, public_density, source_density in zip(
        public_dlv.xstrikes,
        source_dlv.xstrikes,
        public_dlv.xdensities,
        source_dlv.xdensities,
        strict=True,
    ):
        np.testing.assert_array_equal(public_support, source_support)
        np.testing.assert_array_equal(public_density, source_density)
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

    zero_selected = internal_frame.with_columns(
        pl.lit(False).alias("options_filter_eligible")
    )
    with pytest.raises(PublicProjectionError, match="zero filter-selected rows"):
        project_public_day(zero_selected, _config(source))


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
    public_frame_fixture: tuple[PublicProjectionConfig, pl.DataFrame],
) -> None:
    config, shared_frame = public_frame_fixture
    frame = shared_frame.clone()
    rows = frame.to_dicts()
    first_book = next(index for index, row in enumerate(rows) if row["record_kind"] == "book")
    duplicate = dict(rows[first_book])
    for row in rows[first_book:]:
        if (
            row["record_kind"] == "book"
            and row["trading_date"] == duplicate["trading_date"]
            and row["expiry_index"] == duplicate["expiry_index"]
        ):
            row["book_index"] += 1
    duplicate["book_index"] = 0
    rows.insert(first_book, duplicate)
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
