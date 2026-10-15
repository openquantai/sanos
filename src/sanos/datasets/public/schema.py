"""Exact flat scalar schema and capability contract for public SANOS data."""

from __future__ import annotations

from typing import Any, Final, cast

import polars as pl
from cdxcore.version import version

PUBLIC_DATASET_SCHEMA_VERSION: Final = "sanos-public-flat-v3"
PUBLIC_SCHEMA_DOCUMENT_VERSION: Final = "sanos-public-schema-document-v4"
PUBLIC_CAPABILITIES_VERSION: Final = "sanos-public-capabilities-v4"
PUBLIC_VALIDATION_VERSION: Final = "sanos-public-validation-v5"

PUBLIC_RECORD_KINDS: Final = (
    "day",
    "expiry",
    "book",
    "smooth_node",
    "dlv_node",
)


@version("sanos-public-flat-polars-schema-v3")
def public_polars_schema() -> pl.Schema:
    """Return the only accepted flat public Parquet schema.

    Every column is scalar. Fields which do not apply to a row's ``record_kind``
    are null; the schema document owns the exact population rules.
    """

    return pl.Schema(
        {
            "trading_date": pl.Date,
            "record_kind": pl.String,
            "is_half_day": pl.Boolean,
            "driftless_spot_return": pl.Float64,
            "nms_epsilon_atm_annual_volatility": pl.Float64,
            "smoothness": pl.Float64,
            "sanos_market_percent_books_outside": pl.Float64,
            "sanos_market_conditional_mean_excess_halfspreads": pl.Float64,
            "sanos_market_conditional_median_excess_halfspreads": pl.Float64,
            "dlv_market_percent_books_outside": pl.Float64,
            "dlv_market_conditional_mean_excess_halfspreads": pl.Float64,
            "dlv_market_conditional_median_excess_halfspreads": pl.Float64,
            "expiry_index": pl.Int32,
            "expiry_date": pl.Date,
            "business_dte": pl.Float32,
            "calendar_dte": pl.Int32,
            "is_am": pl.Boolean,
            "time_to_expiry_years": pl.Float32,
            "nms_atm_annual_volatility": pl.Float64,
            "is_call": pl.Boolean,
            "normalized_strike": pl.Float32,
            "half_spread": pl.Float64,
            "sanos_fit_annual_iv": pl.Float64,
            "sanos_synthetic_bid_annual_iv": pl.Float64,
            "sanos_synthetic_ask_annual_iv": pl.Float64,
            "dlv_fit_annual_iv": pl.Float64,
            "dlv_synthetic_bid_annual_iv": pl.Float64,
            "dlv_synthetic_ask_annual_iv": pl.Float64,
            "book_index": pl.Int32,
            "node_index": pl.Int32,
            "time_increment_years": pl.Float64,
            "kernel_input_total_variance": pl.Float64,
            "input_total_variance": pl.Float64,
            "normalized_support": pl.Float64,
            "density_mass": pl.Float64,
            "squared_dlv": pl.Float64,
        }
    )


@version("3.0", dependencies=[public_polars_schema])
def public_schema_document() -> dict[str, Any]:
    """Return the human- and machine-readable immutable schema contract."""

    scalar_types = {
        "trading_date": "Date non-null",
        "record_kind": f"String one-of {list(PUBLIC_RECORD_KINDS)}",
        "is_half_day": "Boolean day-only",
        "driftless_spot_return": "Float64 day-only nullable-first-day",
        "nms_epsilon_atm_annual_volatility": "Float64 day-only",
        "smoothness": "Float64 day-only",
        "sanos_market_percent_books_outside": "Float64 day-only percent",
        "sanos_market_conditional_mean_excess_halfspreads": "Float64 day-only",
        "sanos_market_conditional_median_excess_halfspreads": "Float64 day-only",
        "dlv_market_percent_books_outside": "Float64 day-only percent",
        "dlv_market_conditional_mean_excess_halfspreads": "Float64 day-only",
        "dlv_market_conditional_median_excess_halfspreads": "Float64 day-only",
        "expiry_index": "Int32 expiry/book/model-node",
        "expiry_date": "Date expiry-only",
        "business_dte": "Float32 expiry-only",
        "calendar_dte": "Int32 expiry-only",
        "is_am": "Boolean expiry-only",
        "time_to_expiry_years": "Float32 expiry-only",
        "nms_atm_annual_volatility": "Float64 expiry-only",
        "is_call": "Boolean book-only",
        "normalized_strike": "Float32 book-only",
        "half_spread": "Float64 book-only",
        "sanos_fit_annual_iv": "Float64 nullable book-only (null at zero model time)",
        "sanos_synthetic_bid_annual_iv": "Float64 nullable book-only",
        "sanos_synthetic_ask_annual_iv": "Float64 nullable book-only",
        "dlv_fit_annual_iv": "Float64 nullable book-only (null at zero model time)",
        "dlv_synthetic_bid_annual_iv": "Float64 nullable book-only",
        "dlv_synthetic_ask_annual_iv": "Float64 nullable book-only",
        "book_index": "Int32 book-only",
        "node_index": "Int32 model-node-only",
        "time_increment_years": "Float64 model-node-only, repeated within expiry",
        "kernel_input_total_variance": "Float64 smooth-node-only, repeated within expiry",
        "input_total_variance": "Float64 dlv-node-only, repeated within expiry",
        "normalized_support": "Float64 model-node-only",
        "density_mass": "Float64 model-node-only",
        "squared_dlv": "Float64 dlv-node-only",
    }
    return {
        "schema": PUBLIC_SCHEMA_DOCUMENT_VERSION,
        "integrated_contract": "DySANOS 1.1",
        "physical_schema": {
            "layout": "single flat scalar Polars frame",
            "columns": scalar_types,
            "row_keys": {
                "day": ["trading_date", "record_kind"],
                "expiry": ["trading_date", "record_kind", "expiry_index"],
                "book": [
                    "trading_date",
                    "record_kind",
                    "expiry_index",
                    "is_call",
                    "normalized_strike",
                    "book_index",
                ],
                "smooth_node": [
                    "trading_date",
                    "record_kind",
                    "expiry_index",
                    "node_index",
                ],
                "dlv_node": [
                    "trading_date",
                    "record_kind",
                    "expiry_index",
                    "node_index",
                ],
            },
            "ordering": "date, then day/expiry/book/smooth_node/dlv_node canonical order",
            "nested_columns": False,
        },
        "units": {
            "normalized_strike": "K/F",
            "model_call_price": "C/(DF*F), reconstructed only",
            "half_spread": "(pure_ask-pure_bid)/2 in C/(DF*F)",
            "book_implied_volatility": "annualized Black volatility",
            "market_excess": "outside-market distance divided by market half-spread",
            "squared_dlv": "annualized 1/year with exact zero boundaries",
            "model_time": "NYSE business sessions/252",
        },
        "return": {
            "field": "driftless_spot_return",
            "formula": "DF(prev,now)*S_now/F_prev(now)-1",
            "normalized_path": "starts at 1; cumulative product of (1+r)",
            "log_path": "starts at 0; cumulative sum of log1p(r)",
            "curve_interpolation": "log-linear in business DTE with (0,1) anchor",
            "date_pair": "consecutive populated retained public days",
            "first_day": "null",
            "extrapolation": "forbidden",
        },
        "pricing": {
            "business_days_per_year": 252.0,
            "trading_hours_per_day": 6.5,
            "one_minute_year_fraction": 1.0 / (60.0 * 6.5 * 252.0),
            "pre_first_expiry": "first fitted density outside intrinsic tolerance",
            "between_expiries": "linear interpolation of normalized call prices",
            "after_last_expiry": "final fitted density",
            "put_call_parity": "p=c-1+k",
            "state_mode": "common surface",
            "dlv_source": "finite projection of the completed smooth SANOS surface",
            "dlv_projection_scope": (
                "retained expiries and inclusive normalized-moneyness domain [-3,2]"
            ),
            "dlv_projection_fit_quality": "measured, not guaranteed",
            "arbitrary_time_smooth_dlv_equivalence": False,
        },
    }


@version("3.0")
def public_capabilities() -> dict[str, bool | str]:
    """Return the closed advertised capability and non-capability object."""

    return cast(
        dict[str, bool | str],
        {
            "schema": PUBLIC_CAPABILITIES_VERSION,
            "integrated_contract": "DySANOS 1.1",
            "flat_scalar_polars": True,
            "normalized_smooth_sanos_pricing": True,
            "normalized_discrete_dlv_pricing": True,
            "discrete_dlv_projection_scope": "retained-expiries-inclusive-nms-[-3,2]",
            "discrete_dlv_projection_fit_quality": "measured-not-guaranteed",
            "arbitrary_time_smooth_dlv_equivalence": False,
            "model_atm_implied_volatility": True,
            "daily_front_normalized_books": True,
            "driftless_spread_experiments": True,
            "driftless_return_dynamics": True,
            "exact_vendor_books": False,
            "quote_containment_replay": False,
            "vendor_implied_volatility_or_greeks": False,
            "volume_open_interest_or_liquidity_models": False,
            "absolute_levels_or_dollar_tick_effects": False,
            "contract_or_security_replay": False,
            "expiry_fit_mode_pricing": False,
        },
    )


__all__ = [
    "PUBLIC_CAPABILITIES_VERSION",
    "PUBLIC_DATASET_SCHEMA_VERSION",
    "PUBLIC_RECORD_KINDS",
    "PUBLIC_SCHEMA_DOCUMENT_VERSION",
    "PUBLIC_VALIDATION_VERSION",
    "public_capabilities",
    "public_polars_schema",
    "public_schema_document",
]
