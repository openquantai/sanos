from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from sanos.reference import read_legacy_pytree_hdf5


REPORT = Path(
    "/home/hans/quantdata_data/sanos/phase-10-f8e546c/evidence/"
    "20261014T034740Z-clean-closure/daily-comparison-restart-2.json"
)
LEGACY_ROOT = Path(
    "/home/hans/quantdata_data/training/dysanos/market/"
    "bid=0.011, nat,eth=10,bth=4,mps=1e-06,m ca250824/"
    "sm_0.25 6b930b59/SPX"
)


def objective(
    bids: list[np.ndarray],
    asks: list[np.ndarray],
    masks: list[np.ndarray],
    fits: list[np.ndarray],
) -> float:
    average_half_spread = sum(
        float(np.sum((ask[mask] - bid[mask]) / 2.0))
        for bid, ask, mask in zip(bids, asks, masks, strict=True)
    ) / sum(int(np.sum(mask)) for mask in masks)
    result = 0.0
    for bid, ask, mask, fit in zip(bids, asks, masks, fits, strict=True):
        bid = bid[mask]
        ask = ask[mask]
        fit = fit[mask]
        half_spread = (ask - bid) / 2.0
        weighted = np.maximum(half_spread, 1.0e-4) / average_half_spread
        lower = bid + half_spread * 0.99
        upper = ask - half_spread * 0.99
        box = np.maximum(np.maximum(fit - upper, lower - fit), 0.0) / weighted
        mid = (bid + ask) / 2.0
        z_mid = float(np.max(np.abs(fit - mid) / weighted))
        result += 0.9 * float(np.sum(box)) + 0.1 * z_mid
    return result


report = json.loads(REPORT.read_text(encoding="utf-8"))
for day in report["dates"]:
    if day["classification"] != "unexplained_discrepancy":
        continue
    date = day["date"]
    frame = pl.read_parquet(Path(day["fresh"]["path"]) / "data.parquet")
    selected = frame["options_filter_eligible"].to_numpy().astype(bool)
    expiry_ids = frame["expiry_id"].to_numpy()
    groups = [
        frame.filter(pl.Series(selected & (expiry_ids == expiry_id)))
        for expiry_id in np.unique(expiry_ids[selected])
    ]
    bids = [group["pure_bid_price"].to_numpy().astype(np.float64) for group in groups]
    asks = [group["pure_ask_price"].to_numpy().astype(np.float64) for group in groups]
    fresh_masks = [group["smooth_fit_mask"].to_numpy() for group in groups]
    fresh_fits = [group["smooth_surface_fit_price"].to_numpy() for group in groups]
    legacy = read_legacy_pytree_hdf5(LEGACY_ROOT / f"{date}.h5")["af_pricer"]
    print(
        json.dumps(
            {
                "date": date,
                "fresh_objective_recomputed": objective(
                    bids,
                    asks,
                    fresh_masks,
                    fresh_fits,
                ),
                "legacy_objective_recomputed": objective(
                    bids,
                    asks,
                    legacy["masks"],
                    legacy["fitsX"],
                ),
                "mask_equal": all(
                    np.array_equal(a, b)
                    for a, b in zip(fresh_masks, legacy["masks"], strict=True)
                ),
            },
            sort_keys=True,
        )
    )
