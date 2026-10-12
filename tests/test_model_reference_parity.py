from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from cdxcore.verbose import Context

from sanos.model import SanosFitter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = REPOSITORY_ROOT / "reference" / "quant_data"


def _pinned_reference_output() -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REFERENCE_SOURCE),
            "QUANTDATA_ROOT": "/tmp/sanos/phase-3-reference/data",
            "QUANTDATA_TEMP_DATA": "/tmp/sanos/phase-3-reference/temp",
            "WRDS_OFF": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tests" / "reference_probe.py")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_lp_smoothness_zero_and_production_quarter_match_pinned_reference() -> None:
    reference = _pinned_reference_output()["fit_parity"]
    expiries = np.array([5.0 / 252.0, 10.0 / 252.0], dtype=np.float64)
    xstrikes = [
        np.array([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64),
        np.array([0.4, 0.7, 1.0, 1.3, 1.6], dtype=np.float64),
    ]
    market_strikes = [
        np.array([0.75, 1.0, 1.25], dtype=np.float64),
        np.array([0.7, 1.0, 1.3], dtype=np.float64),
    ]
    query_strikes = np.array([0.6, 0.8, 1.0, 1.2, 1.4], dtype=np.float64)

    for smoothness in (0.0, 0.25):
        expected = reference[str(smoothness)]
        mids = [np.asarray(values, dtype=np.float64) for values in expected["market_mids"]]
        fitted = SanosFitter(smoothness=smoothness, timeout_seconds=20.0)(
            expiries,
            market_strikes,
            mids,
            xstrikes,
            vols=0.2,
            verbose=cast(Any, Context).quiet,
        )

        for observed, wanted in zip(
            fitted.xdensities, expected["xdensities"], strict=True
        ):
            np.testing.assert_allclose(observed, wanted, rtol=0.0, atol=1.0e-12)
        for observed, wanted in zip(fitted.dlvsqs(), expected["dlvsqs"], strict=True):
            np.testing.assert_allclose(observed, wanted, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            fitted(expiries[:, None], query_strikes[None, :]),
            expected["query_prices"],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            fitted.variances,
            expected["variances"],
            rtol=0.0,
            atol=0.0,
        )
