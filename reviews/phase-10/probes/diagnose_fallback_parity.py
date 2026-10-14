from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from cdxcore.verbose import Context

from quantdata.sanos.arbfree import MkArbFree as LegacyMkArbFree
from sanos.reference import read_legacy_pytree_hdf5


DATA_ROOT = Path("/home/hans/quantdata_data/sanos/phase-10-f8e546c/data")
LEGACY_ROOT = Path(
    "/home/hans/quantdata_data/training/dysanos/market/"
    "bid=0.011, nat,eth=10,bth=4,mps=1e-06,m ca250824/"
    "sm_0.25 6b930b59/SPX"
)
FITTED = {
    "2025-01-02": (
        "widb/fitted-day/ticker-SPX/year-2025/date-2025-01-02",
        "4de1d1094a51abeb7d46949049f41b807901c7454c1b2b93bffc76fbc9c76a4e",
    ),
    "2025-01-03": (
        "widb/fitted-day/ticker-SPX/year-2025/date-2025-01-03",
        "e39309c855fd53ae1dbd4f989787582b126d7d2a6814dd99748c87937fe8542c",
    ),
}


def artifact_path(namespace: str, artifact_id: str) -> Path:
    return DATA_ROOT / "intermediate" / namespace / artifact_id


def max_nested(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    assert len(left) == len(right)
    return max(
        float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
        for a, b in zip(left, right, strict=True)
    )


def fitter(cls: type):
    kwargs = dict(
        no_arb_mode="surface",
        smoothness=0.25,
        surface_expiry_linear=True,
        min_half_spread=1.0e-4,
        mid_fit_weight=0.1,
        min_atm_vol=0.05,
        min_dlv=None,
        max_dlv=None,
        timeout_seconds=300.0,
        min_strikes_per_expiry=5,
        min_density=1.0e-4,
        min_dC_eps=1.0e-3,
    )
    return cls(**kwargs)


for day, (namespace, artifact_id) in FITTED.items():
    fitted_path = artifact_path(namespace, artifact_id)
    fitted_frame = pl.read_parquet(fitted_path / "data.parquet")
    manifest = json.loads((fitted_path / "manifest.json").read_text(encoding="utf-8"))
    upstream = manifest["upstream"][0]
    filtered_path = artifact_path(upstream["namespace"], upstream["artifact_id"])
    frame = pl.read_parquet(filtered_path / "data.parquet")
    selected = frame["options_filter_eligible"].to_numpy().astype(bool)
    expiry_ids = frame["expiry_id"].to_numpy()
    groups = [
        frame.filter(pl.Series(selected & (expiry_ids == expiry_id)))
        for expiry_id in np.unique(expiry_ids[selected])
    ]
    expiries = np.asarray(
        [group["time_to_expiry_years"].to_numpy().astype(np.float32)[0] for group in groups],
        dtype=np.float32,
    )
    strikes = [group["normalized_strike"].to_numpy().astype(np.float32) for group in groups]
    bids = [group["pure_bid_price"].to_numpy().astype(np.float32) for group in groups]
    asks = [group["pure_ask_price"].to_numpy().astype(np.float32) for group in groups]

    legacy = fitter(LegacyMkArbFree)(
        expiries=expiries,
        strikes=strikes,
        bids=bids,
        asks=asks,
        verbose=Context.quiet,
    )
    oracle = read_legacy_pytree_hdf5(LEGACY_ROOT / f"{day}.h5")
    oracle_pricer = oracle["af_pricer"]
    fresh_source_indices = tuple(
        int(value)
        for value in fitted_frame["smooth_input_expiry_index"]
        .drop_nulls()
        .unique(maintain_order=True)
    )
    fresh_fits = [
        fitted_frame.filter(pl.col("smooth_input_expiry_index") == source_index)[
            "smooth_surface_fit_price"
        ].to_numpy()
        for source_index in fresh_source_indices
    ]
    fresh_densities = [
        np.asarray(
            fitted_frame.filter(pl.col("smooth_input_expiry_index") == source_index)[
                "smooth_surface_model_density"
            ][0]
        )
        for source_index in fresh_source_indices
    ]
    print(
        json.dumps(
            {
                "date": day,
                "input_index_equal": bool(
                    np.array_equal(np.asarray(fresh_source_indices), legacy.input_index)
                ),
                "fresh_vs_legacy_code_fitsX_max_abs": max_nested(
                    fresh_fits,
                    legacy.fitsX,
                ),
                "fresh_vs_legacy_code_mdensity_max_abs": max_nested(
                    fresh_densities,
                    legacy.mdensities,
                ),
                "legacy_code_vs_historical_fitsX_max_abs": max_nested(
                    legacy.fitsX,
                    oracle_pricer["fitsX"],
                ),
                "legacy_code_vs_historical_mdensity_max_abs": max_nested(
                    legacy.mdensities,
                    oracle_pricer["mdensities"],
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
