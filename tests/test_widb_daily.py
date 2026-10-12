from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from cdxcore.bs import bs

from sanos.config import canonical_sha256
from sanos.widb import (
    ENRICHED_SCHEMA,
    NMS5,
    PAIRED_SCHEMA,
    ClockConfig,
    DailyDataError,
    DailyProvenance,
    DailyRawInputs,
    DailyWidbConfig,
    ExpiryEnrichmentConfig,
    count_business_days_between,
    enrich_daily_frame,
    interpolate_zero_curve,
    is_half_day,
    pair_daily_frame,
)


def _raw_inputs(
    *,
    trading_date: date = date(2024, 1, 3),
    expiry: date = date(2024, 1, 19),
    is_am: bool = False,
    strikes: np.ndarray | None = None,
    reject_last_put: bool = False,
) -> DailyRawInputs:
    strike_values = (
        np.arange(90.0, 115.0, 2.0) if strikes is None else np.asarray(strikes, dtype=float)
    )
    spot = 100.0
    forward = 101.0
    calendar_dte = (expiry - trading_date).days
    discount = float(np.exp(-0.05 * calendar_dte / 365.0))
    business_dte = float(count_business_days_between(trading_date, expiry))
    if is_am:
        business_dte -= 0.8
    time_years = business_dte / 252.0
    calls = (
        np.asarray(
            bs(
                k=strike_values / forward,
                vol=0.4,
                sqrtT=np.sqrt(time_years),
            )
        )
        * discount
        * forward
    )
    puts = calls - discount * (forward - strike_values)
    rows: list[dict[str, object]] = []
    for index, (strike, call, put) in enumerate(zip(strike_values, calls, puts, strict=True)):
        for side, price in (("C", call), ("P", put)):
            rows.append(
                {
                    "secid": 108105,
                    "date": trading_date.isoformat(),
                    "symbol": f"SPX-{expiry}-{side}-{index}",
                    "optionid": index * 2 + int(side == "P"),
                    "am_settlement": int(is_am),
                    "cp_flag": side,
                    "strike_price": int(strike * 1000),
                    "exdate": expiry.isoformat(),
                    "best_bid": float(
                        0.02
                        if reject_last_put and side == "P" and index == len(strike_values) - 1
                        else price - 0.05
                    ),
                    "best_offer": float(price + 0.05),
                    "volume": 10,
                    "open_interest": 100,
                    "impl_volatility": 0.4,
                    "ss_flag": "0",
                    "delta": 0.5,
                    "gamma": 0.01,
                    "vega": 1.0,
                    "theta": -0.5,
                    "expiry_indicator": "w",
                    "contract_size": 100,
                }
            )
    previous = date(2024, 1, 2)
    return DailyRawInputs(
        security=pl.DataFrame({"ticker": ["SPX"], "secid": [108105]}),
        options=pl.DataFrame(rows),
        spot=pl.DataFrame({"secid": [108105], "date": [trading_date.isoformat()], "close": [spot]}),
        previous_spot=pl.DataFrame(
            {"secid": [108105], "date": [previous.isoformat()], "close": [99.0]}
        ),
        forwards=pl.DataFrame(
            {
                "secid": [108105],
                "date": [trading_date.isoformat()],
                "expiration": [expiry.isoformat()],
                "amsettlement": [int(is_am)],
                "forwardprice": [forward],
            }
        ),
        zero_curve=pl.DataFrame(
            {
                "date": [trading_date.isoformat(), trading_date.isoformat()],
                "days": [1, 30],
                "rate": [5.0, 5.0],
            }
        ),
    )


def test_zero_curve_uses_percent_continuous_act_365_rates() -> None:
    curve = _raw_inputs().zero_curve
    observed = interpolate_zero_curve(curve, np.asarray([1.0, 16.0, 30.0]))
    expected = np.exp(-0.05 * np.asarray([1.0, 16.0, 30.0]) / 365.0)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-15)
    assert np.all(observed > 0.0)


def test_normalized_and_enriched_frames_preserve_reference_dtypes_and_units() -> None:
    config = DailyWidbConfig()
    normalized = pair_daily_frame(
        ticker="spx",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(reject_last_put=True),
        config=config,
        provenance=DailyProvenance("a" * 64),
    )

    assert normalized.shape == (13, len(PAIRED_SCHEMA))
    assert normalized.schema == pl.Schema(PAIRED_SCHEMA)
    assert normalized.get_column("strike").to_list() == list(np.arange(90.0, 115.0, 2.0))
    assert normalized.get_column("business_dte").unique().item() == 11.0
    assert normalized.get_column("calendar_dte").unique().item() == 16
    assert normalized.get_column("source_secid").unique().item() == 108105
    assert normalized.get_column("raw_options_artifact_id").item(0) == "a" * 64
    assert normalized.get_column("spot_return").unique().item() == pytest.approx(
        np.float32(np.float32(100.0) / np.float32(99.0) - np.float32(1.0))
    )

    enriched = enrich_daily_frame(normalized, config=config)
    assert enriched.shape == (12, len(ENRICHED_SCHEMA))
    assert enriched.schema == pl.Schema(ENRICHED_SCHEMA)
    assert enriched.get_column("implied_forward").n_unique() == 1
    assert enriched.get_column("implied_forward").item(0) > 0.0
    np.testing.assert_allclose(
        enriched.get_column("normalized_strike").to_numpy(),
        enriched.get_column("strike").to_numpy()
        / enriched.get_column("implied_forward").to_numpy(),
        rtol=2.0e-7,
        atol=0.0,
    )
    assert np.all(
        enriched.get_column("pure_ask_price").to_numpy()
        > enriched.get_column("pure_bid_price").to_numpy()
    )


@pytest.mark.parametrize(
    ("fraction", "expected_nms", "expected_atm_nms"),
    [
        (0.1, 1.4749082, 1.2809552),
        (0.2, 1.4606576, 1.1409388),
        (0.5, 1.4202629, 0.8964080),
    ],
)
def test_overnight_fraction_is_a_clock_input_and_changes_identity(
    fraction: float,
    expected_nms: float,
    expected_atm_nms: float,
) -> None:
    config = DailyWidbConfig(
        clock=ClockConfig(overnight_variance_fraction=fraction),
        enrichment=ExpiryEnrichmentConfig(
            num_min_either_traded=1,
            num_min_both_traded=1,
        ),
    )
    am = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(
            expiry=date(2024, 1, 4),
            is_am=True,
            strikes=np.asarray([90.0, 101.0, 112.0]),
            reject_last_put=True,
        ),
        config=config,
    )
    pm = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(
            expiry=date(2024, 1, 4),
            is_am=False,
            strikes=np.asarray([90.0, 101.0, 112.0]),
            reject_last_put=True,
        ),
        config=config,
    )

    assert am.get_column("business_dte").item() == pytest.approx(np.float32(fraction))
    assert am.get_column("time_to_expiry_years").item() == pytest.approx(
        np.float32(fraction / 252.0)
    )
    assert pm.get_column("business_dte").item() == 1.0
    assert pm.get_column("time_to_expiry_years").item() == pytest.approx(np.float32(1.0 / 252.0))
    am_time = np.float32(fraction / 252.0)
    assert NMS5.to_nms(1.02, am_time) == pytest.approx(expected_nms, abs=1.5e-7)
    assert NMS5.to_nms(1.02, am_time, volatility=0.4) == pytest.approx(
        expected_atm_nms,
        abs=1.5e-7,
    )
    pm_time = np.float32(1.0 / 252.0)
    pm_vector = np.asarray([NMS5.to_nms(1.02, pm_time), NMS5.to_nms(1.02, pm_time, volatility=0.4)])
    np.testing.assert_allclose(
        pm_vector,
        np.asarray([1.3597982, 0.7004350]),
        rtol=0.0,
        atol=1.5e-7,
    )
    assert canonical_sha256(config) != canonical_sha256(
        replace(
            config,
            clock=replace(
                config.clock,
                overnight_variance_fraction=0.2 + fraction / 10,
            ),
        )
    )
    with pytest.raises(DailyDataError, match="CD-020"):
        enrich_daily_frame(am, config=config)


def test_overnight_fraction_endpoints_are_explicit() -> None:
    paired: dict[float, pl.DataFrame] = {}
    for fraction in (0.0, 1.0):
        config = DailyWidbConfig(
            clock=ClockConfig(overnight_variance_fraction=fraction),
            enrichment=ExpiryEnrichmentConfig(
                num_min_either_traded=1,
                num_min_both_traded=1,
            ),
        )
        paired[fraction] = pair_daily_frame(
            ticker="SPX",
            trading_date=date(2024, 1, 3),
            raw=_raw_inputs(
                expiry=date(2024, 1, 4),
                is_am=True,
                strikes=np.asarray([90.0, 101.0, 112.0]),
                reject_last_put=True,
            ),
            config=config,
        )
    assert paired[0.0].get_column("business_dte").item() == 0.0
    assert paired[1.0].get_column("business_dte").item() == 1.0
    with pytest.raises(DailyDataError, match="implied-volatility inversion"):
        enrich_daily_frame(
            paired[0.0],
            config=DailyWidbConfig(
                clock=ClockConfig(overnight_variance_fraction=0.0),
                enrichment=ExpiryEnrichmentConfig(
                    num_min_either_traded=1,
                    num_min_both_traded=1,
                ),
            ),
        )
    with pytest.raises(DailyDataError, match="CD-020"):
        enrich_daily_frame(
            paired[1.0],
            config=DailyWidbConfig(
                clock=ClockConfig(overnight_variance_fraction=1.0),
                enrichment=ExpiryEnrichmentConfig(
                    num_min_either_traded=1,
                    num_min_both_traded=1,
                ),
            ),
        )


def test_default_clock_and_calendar_match_reference_conventions() -> None:
    assert count_business_days_between(date(2024, 1, 3), date(2024, 1, 19)) == 11
    assert is_half_day(date(2023, 11, 24)) is True
    assert is_half_day(date(2024, 1, 3)) is False
    am = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(
            expiry=date(2024, 1, 4),
            is_am=True,
            strikes=np.asarray([101.0]),
        ),
    )
    assert am.get_column("business_dte").item() == pytest.approx(np.float32(0.2))


def test_known_bad_day_returns_typed_empty_compatibility_marker() -> None:
    result = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2025, 1, 8),
        raw=_raw_inputs(trading_date=date(2025, 1, 8), expiry=date(2025, 1, 17)),
    )
    assert result.is_empty()
    assert result.schema == pl.Schema(PAIRED_SCHEMA)
    enriched = enrich_daily_frame(pl.DataFrame())
    assert enriched.is_empty()
    assert enriched.schema == pl.Schema(ENRICHED_SCHEMA)


def test_latest_valid_date_guard_is_fatal_before_raw_access() -> None:
    with pytest.raises(ValueError, match="LAST_LIKELY_VALID_DATE"):
        pair_daily_frame(
            ticker="SPX",
            trading_date=date(2025, 9, 1),
            raw=_raw_inputs(trading_date=date(2025, 9, 1), expiry=date(2025, 9, 19)),
        )


def test_zero_business_day_is_limited_to_historical_saturday_am_books() -> None:
    base = _raw_inputs(
        trading_date=date(2024, 1, 5),
        expiry=date(2024, 1, 8),
        is_am=True,
    )
    saturday = replace(
        base,
        options=base.options.with_columns(pl.lit("2024-01-06").alias("exdate")),
        forwards=base.forwards.with_columns(pl.lit("2024-01-06").alias("expiration")),
    )
    allowed = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 5),
        raw=saturday,
    )
    assert allowed.get_column("business_dte").unique().item() == 0.0
    sunday = replace(
        base,
        options=base.options.with_columns(
            pl.lit("2024-01-07").alias("exdate"),
            pl.lit(0).alias("am_settlement"),
        ),
        forwards=base.forwards.with_columns(
            pl.lit("2024-01-07").alias("expiration"),
            pl.lit(0).alias("amsettlement"),
        ),
    )
    with pytest.raises(DailyDataError, match="Zero-business-day"):
        pair_daily_frame(
            ticker="SPX",
            trading_date=date(2024, 1, 5),
            raw=sunday,
        )


def test_reference_duplicate_and_first_spot_semantics_are_explicit() -> None:
    raw = _raw_inputs()
    duplicate_call = (
        raw.options.filter(pl.col("cp_flag") == "C")
        .head(1)
        .with_columns(
            pl.lit(999999).cast(raw.options.schema["optionid"]).alias("optionid"),
            pl.lit("SPX-DUPLICATE-CALL").alias("symbol"),
        )
    )
    duplicated = replace(raw, options=pl.concat((raw.options, duplicate_call)))
    paired = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=duplicated,
    )
    assert paired.height == 14  # Legacy inner join preserves the Cartesian duplicate.

    two_spots = replace(
        raw,
        spot=pl.concat(
            (
                raw.spot,
                pl.DataFrame({"secid": [108105], "date": ["2024-01-03"], "close": [120.0]}),
            )
        ),
    )
    first = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=two_spots,
    )
    assert first.get_column("spot").unique().item() == 100.0


def test_legacy_enrichment_normalize_flag_is_identity_only() -> None:
    paired = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(reject_last_put=True),
    )
    natural = enrich_daily_frame(paired, config=DailyWidbConfig())
    flagged = enrich_daily_frame(
        paired,
        config=DailyWidbConfig(enrichment=ExpiryEnrichmentConfig(normalize=True)),
    )
    numeric = [name for name, dtype in ENRICHED_SCHEMA.items() if dtype == pl.Float32]
    np.testing.assert_array_equal(
        natural.select(numeric).to_numpy(),
        flagged.select(numeric).to_numpy(),
    )


def test_nonempty_day_cannot_silently_become_empty_during_enrichment() -> None:
    paired = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(strikes=np.asarray([101.0])),
    )
    with pytest.raises(DailyDataError, match="No valid option rows"):
        enrich_daily_frame(paired, config=DailyWidbConfig())


def test_all_valid_day_preserves_pinned_reference_failure_cd020() -> None:
    paired = pair_daily_frame(
        ticker="SPX",
        trading_date=date(2024, 1, 3),
        raw=_raw_inputs(strikes=np.asarray([101.0])),
    )
    config = DailyWidbConfig(
        enrichment=ExpiryEnrichmentConfig(
            num_min_either_traded=1,
            num_min_both_traded=1,
        )
    )
    with pytest.raises(DailyDataError, match="CD-020"):
        enrich_daily_frame(paired, config=config)


def test_config_rejects_invalid_clock_and_unknown_mapping_fields() -> None:
    for value in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="overnight_variance_fraction"):
            ClockConfig(overnight_variance_fraction=value)
    with pytest.raises(ValueError, match="Unknown"):
        DailyWidbConfig.from_mapping({"extra": 1})


def test_daily_transform_versions_close_over_clock_and_source_contracts() -> None:
    normalized = str(pair_daily_frame.version)
    enriched = str(enrich_daily_frame.version)
    for dependency in (
        "DailyWidbConfig",
        "RawDayConfig",
        "ClockConfig",
        "bad_data_policy",
        "business_clock_policy",
        "count_business_days_between",
        "interpolate_zero_curve",
        "_paired_schema",
        "raw_table_spec",
    ):
        assert dependency in normalized
    for dependency in (
        "ExpiryEnrichmentConfig",
        "NormalizedStrikes",
        "normalized_strike_policy",
        "_black_implied",
        "_black_values",
        "_numerical_tolerances",
        "_enriched_only_schema",
    ):
        assert dependency in enriched
