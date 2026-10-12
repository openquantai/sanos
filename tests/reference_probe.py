"""Emit deterministic numerical outputs from the pinned reference snapshot."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
from cdxcore.verbose import Context
from quantdata.sanos.sanos import SanosFitter
from quantdata.sanos.sanos_tools import SanosPricer, repair_density_nonnegative


def main() -> None:
    expiries = np.array([5.0 / 252.0, 10.0 / 252.0], dtype=np.float64)
    strikes = [
        np.array([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
        np.array([0.4, 0.7, 1.0, 1.3, 1.6], dtype=np.float64),
    ]
    dummy_densities = [
        np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    ]
    input_dlvsqs = [
        np.array([0.04, 0.09, 0.16], dtype=np.float64),
        np.array([0.05, 0.10, 0.15], dtype=np.float64),
    ]
    template = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=dummy_densities,
    )
    densities = template.densities_from_dlvsqs(input_dlvsqs)
    pricer = SanosPricer(
        smoothness=None,
        expiries=expiries,
        variances=expiries * 0.04,
        xstrikes=strikes,
        xdensities=densities,
    )
    prices = pricer(
        expiries[:, None],
        np.array([[0.6, 0.8, 1.0, 1.2, 1.4]], dtype=np.float64),
    )
    repaired = repair_density_nonnegative(
        p=np.array([-0.08, 0.52, 0.41, 0.16], dtype=np.float64),
        x=np.array([0.45, 0.8, 1.1, 1.65], dtype=np.float64),
    )
    market_strikes = [
        np.array([0.75, 1.0, 1.25], dtype=np.float64),
        np.array([0.7, 1.0, 1.3], dtype=np.float64),
    ]
    query_strikes = np.array([0.6, 0.8, 1.0, 1.2, 1.4], dtype=np.float64)
    fit_parity = {}
    for smoothness in (0.0, 0.25):
        source = SanosPricer(
            smoothness=smoothness,
            expiries=expiries,
            variances=expiries * 0.04,
            xstrikes=strikes,
            xdensities=densities,
        )
        mids = [
            np.asarray(source(float(expiry), strike), dtype=np.float64)
            for expiry, strike in zip(expiries, market_strikes, strict=True)
        ]
        fitted = SanosFitter(smoothness=smoothness, timeout_seconds=20.0)(
            expiries,
            market_strikes,
            mids,
            strikes,
            vols=0.2,
            verbose=cast(Any, Context).quiet,
        )
        fit_parity[str(smoothness)] = {
            "market_mids": [values.tolist() for values in mids],
            "xdensities": [values.tolist() for values in fitted.xdensities],
            "dlvsqs": [values.tolist() for values in fitted.dlvsqs()],
            "query_prices": np.asarray(
                fitted(expiries[:, None], query_strikes[None, :]), dtype=np.float64
            ).tolist(),
            "variances": fitted.variances.tolist(),
        }
    print(
        json.dumps(
            {
                "densities": [values.tolist() for values in densities],
                "dlvsqs": [values.tolist() for values in pricer.dlvsqs()],
                "prices": prices.tolist(),
                "repair": repaired.tolist(),
                "fit_parity": fit_parity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
