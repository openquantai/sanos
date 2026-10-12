"""Run the pinned quant_data MkArbFree implementation on JSON arrays."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from cdxcore.bs import bs
from cdxcore.verbose import Context
from quantdata.sanos.arbfree import MkArbFree
from quantdata.widb.day import _GREEK_CALC_EPS, _IMPLIED_VOL_CALC_EPS


def _lists(values: Any) -> list[Any] | None:
    if values is None:
        return None
    return [None if value is None else np.asarray(value).tolist() for value in values]


request = json.load(sys.stdin)
input_dtype = np.float32 if request["input_dtype"] == "float32" else np.float64
expiries = np.asarray(request["expiries"], dtype=input_dtype)
strikes = [np.asarray(value, dtype=input_dtype) for value in request["strikes"]]
bids = [np.asarray(value, dtype=input_dtype) for value in request["bids"]]
asks = [np.asarray(value, dtype=input_dtype) for value in request["asks"]]
pricer = MkArbFree(
    smoothness=request["smoothness"],
    surface_expiry_linear=True,
)(
    expiries=expiries,
    strikes=strikes,
    bids=bids,
    asks=asks,
    verbose=Context.quiet,
)
fit = pricer.compute_market_fit(
    expiries=expiries,
    strikes=strikes,
    bids=bids,
    asks=asks,
    and_vols=False,
)
quote_fields = []
for output_index, source_index in enumerate(pricer.input_index):
    local_strikes = strikes[int(source_index)]
    sqrt_time = np.sqrt(float(pricer.expiries[output_index]))
    fields = {}
    for prefix, values in (
        ("fit1", pricer.fits1[output_index]),
        ("fitX", pricer.fitsX[output_index]),
    ):
        volatility = bs.implied(
            local_strikes,
            values,
            sqrtT=sqrt_time,
            eps=_IMPLIED_VOL_CALC_EPS,
        )
        greeks = bs(
            k=local_strikes,
            vol=volatility,
            sqrtT=sqrt_time,
            eps=_GREEK_CALC_EPS,
            what=bs.PRICE | bs.DELTA | bs.GAMMA | bs.VEGA | bs.THETA | bs.DK,
        )
        fields[f"{prefix}_volatility"] = np.asarray(
            volatility, dtype=np.float32
        ).tolist()
        for name, value in greeks.items():
            fields[f"pure_{prefix}_{name}"] = np.asarray(
                value, dtype=np.float32
            ).tolist()
    quote_fields.append(fields)
json.dump(
    {
        "expiries": np.asarray(pricer.expiries).tolist(),
        "variances": np.asarray(pricer.variances).tolist(),
        "input_index": np.asarray(pricer.input_index).tolist(),
        "masks": _lists(pricer.masks),
        "strikes": _lists(pricer.strikes),
        "fits1": _lists(pricer.fits1),
        "fitsX": _lists(pricer.fitsX),
        "xstrikes": _lists(pricer.xstrikes),
        "xdensities": _lists(pricer.xdensities),
        "xix_to_mix": _lists(pricer.xix_to_mix),
        "mstrikes": _lists(pricer.mstrikes),
        "mdensities": _lists(pricer.mdensities),
        "quote_fields": quote_fields,
        "fit": {
            "fits": _lists(fit.fits),
            "fit_errs": _lists(fit.fit_errs),
            "pcnt_fit": float(fit.pcnt_fit),
            "median_err": float(fit.median_err),
            "mean_err": float(fit.mean_err),
            "median_total_err": float(fit.median_total_err),
            "mean_total_err": float(fit.mean_total_err),
        },
    },
    sys.stdout,
    allow_nan=False,
    sort_keys=True,
)
