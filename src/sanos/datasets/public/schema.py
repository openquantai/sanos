"""Exact recursive schema and capability contract for public SANOS data."""

from __future__ import annotations

from typing import Any, Final, cast

import polars as pl
from cdxcore.version import version

PUBLIC_DATASET_SCHEMA_VERSION: Final = "sanos-public-daily-v2"
PUBLIC_SCHEMA_DOCUMENT_VERSION: Final = "sanos-public-schema-document-v3"
PUBLIC_CAPABILITIES_VERSION: Final = "sanos-public-capabilities-v3"
PUBLIC_VALIDATION_VERSION: Final = "sanos-public-validation-v4"

EXPIRY_DTYPE: Final = pl.Struct(
    [
        pl.Field("expiry_index", pl.Int32),
        pl.Field("expiry_date", pl.Date),
        pl.Field("business_dte", pl.Float32),
        pl.Field("calendar_dte", pl.Int32),
        pl.Field("is_am", pl.Boolean),
        pl.Field("time_to_expiry_years", pl.Float32),
        pl.Field("discounted_forward_over_spot", pl.Float64),
        pl.Field("nms_atm_annual_volatility", pl.Float64),
    ]
)

BOOK_DTYPE: Final = pl.Struct(
    [
        pl.Field("expiry_index", pl.Int32),
        pl.Field("is_call", pl.Boolean),
        pl.Field("normalized_strike", pl.Float32),
        pl.Field("half_spread_spot", pl.Float64),
    ]
)

SMOOTH_EXPIRY_DTYPE: Final = pl.Struct(
    [
        pl.Field("expiry_index", pl.Int32),
        pl.Field("time_increment_years", pl.Float64),
        pl.Field("kernel_input_total_variance", pl.Float64),
        pl.Field("normalized_support", pl.List(pl.Float64)),
        pl.Field("density_mass", pl.List(pl.Float64)),
    ]
)

SMOOTH_STATE_DTYPE: Final = pl.Struct(
    [
        pl.Field("smoothness", pl.Float64),
        pl.Field("model_expiries", pl.List(SMOOTH_EXPIRY_DTYPE)),
    ]
)

DLV_EXPIRY_DTYPE: Final = pl.Struct(
    [
        pl.Field("expiry_index", pl.Int32),
        pl.Field("time_increment_years", pl.Float64),
        pl.Field("input_total_variance", pl.Float64),
        pl.Field("normalized_support", pl.List(pl.Float64)),
        pl.Field("density_mass", pl.List(pl.Float64)),
        pl.Field("squared_dlv", pl.List(pl.Float64)),
    ]
)

DLV_STATE_DTYPE: Final = pl.Struct(
    [pl.Field("model_expiries", pl.List(DLV_EXPIRY_DTYPE))]
)


@version("sanos-public-recursive-polars-schema-v2")
def public_polars_schema() -> pl.Schema:
    """Return the only accepted version-2 public Parquet schema."""

    return pl.Schema(
        {
            "trading_date": pl.Date,
            "is_half_day": pl.Boolean,
            "forward_adjusted_spot_return": pl.Float64,
            "nms_epsilon_atm_annual_volatility": pl.Float64,
            "expiries": pl.List(EXPIRY_DTYPE),
            "books": pl.List(BOOK_DTYPE),
            "smooth_sanos": SMOOTH_STATE_DTYPE,
            "discrete_dlv": DLV_STATE_DTYPE,
        }
    )


@version("2.1", dependencies=[public_polars_schema])
def public_schema_document() -> dict[str, Any]:
    """Return the human- and machine-readable immutable schema contract."""

    return {
        "schema": PUBLIC_SCHEMA_DOCUMENT_VERSION,
        "physical_schema": {
            "trading_date": "Date",
            "is_half_day": "Boolean",
            "forward_adjusted_spot_return": "Float64 nullable-first-row-only",
            "nms_epsilon_atm_annual_volatility": "Float64",
            "expiries": {
                "type": "List[Struct]",
                "fields": {
                    "expiry_index": "Int32",
                    "expiry_date": "Date",
                    "business_dte": "Float32",
                    "calendar_dte": "Int32",
                    "is_am": "Boolean",
                    "time_to_expiry_years": "Float32",
                    "discounted_forward_over_spot": "Float64",
                    "nms_atm_annual_volatility": "Float64",
                },
            },
            "books": {
                "type": "List[Struct]",
                "fields": {
                    "expiry_index": "Int32",
                    "is_call": "Boolean",
                    "normalized_strike": "Float32",
                    "half_spread_spot": "Float64",
                },
            },
            "smooth_sanos": {
                "smoothness": "Float64",
                "model_expiries": {
                    "type": "List[Struct]",
                    "fields": {
                        "expiry_index": "Int32",
                        "time_increment_years": "Float64",
                        "kernel_input_total_variance": "Float64",
                        "normalized_support": "List[Float64]",
                        "density_mass": "List[Float64]",
                    },
                },
            },
            "discrete_dlv": {
                "model_expiries": {
                    "type": "List[Struct]",
                    "fields": {
                        "expiry_index": "Int32",
                        "time_increment_years": "Float64",
                        "input_total_variance": "Float64",
                        "normalized_support": "List[Float64]",
                        "density_mass": "List[Float64]",
                        "squared_dlv": "List[Float64]",
                    },
                }
            },
        },
        "units": {
            "normalized_strike": "K/F",
            "model_call_price": "C/(DF*F), reconstructed only",
            "half_spread_spot": "(ask-bid)/(2*S)",
            "discounted_forward_over_spot": "DF*(F/S)",
            "squared_dlv": "annualized 1/year with exact zero boundaries",
            "model_time": "NYSE business sessions/252",
        },
        "return": {
            "formula": "(S_now/S_prev)/(F_prev(target)/S_prev)-1",
            "curve_interpolation": "log-linear in business DTE with (0,1) anchor",
            "date_pair": "consecutive populated retained public rows",
            "first_row": "null",
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


@version("2.1")
def public_capabilities() -> dict[str, bool | str]:
    """Return the closed advertised capability and non-capability object."""

    return cast(
        dict[str, bool | str],
        {
            "schema": PUBLIC_CAPABILITIES_VERSION,
            "normalized_smooth_sanos_pricing": True,
            "normalized_discrete_dlv_pricing": True,
            "discrete_dlv_projection_scope": "retained-expiries-inclusive-nms-[-3,2]",
            "discrete_dlv_projection_fit_quality": "measured-not-guaranteed",
            "arbitrary_time_smooth_dlv_equivalence": False,
            "model_atm_implied_volatility": True,
            "daily_front_normalized_books": True,
            "spot_one_spread_experiments": True,
            "forward_adjusted_return_dynamics": True,
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
    "BOOK_DTYPE",
    "DLV_STATE_DTYPE",
    "EXPIRY_DTYPE",
    "PUBLIC_CAPABILITIES_VERSION",
    "PUBLIC_DATASET_SCHEMA_VERSION",
    "PUBLIC_SCHEMA_DOCUMENT_VERSION",
    "PUBLIC_VALIDATION_VERSION",
    "SMOOTH_STATE_DTYPE",
    "public_capabilities",
    "public_polars_schema",
    "public_schema_document",
]
