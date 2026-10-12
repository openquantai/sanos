from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = REPOSITORY_ROOT / "reference" / "quant_data"
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "reference_numerics_v1.json"


def test_pinned_quant_data_snapshot_reproduces_numerical_fixture() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REFERENCE_SOURCE),
            "QUANTDATA_ROOT": "/tmp/sanos/reference-baseline/data",
            "QUANTDATA_TEMP_DATA": "/tmp/sanos/reference-baseline/temp",
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
        timeout=30,
    )
    observed = json.loads(completed.stdout)
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))

    for field in ("densities", "dlvsqs", "prices", "repair"):
        np.testing.assert_allclose(
            np.asarray(observed[field], dtype=np.float64),
            np.asarray(expected[field], dtype=np.float64),
            rtol=1.0e-13,
            atol=1.0e-14,
        )

    assert expected["metadata"]["quant_data_revision"] == (
        "de15be96d2f283b6c31e60d3ea47325a0c093ff0"
    )
    assert expected["metadata"]["evidence_runtime_baseline"] == (
        "reviews/phase-1/reference-runtime-baseline.json"
    )
