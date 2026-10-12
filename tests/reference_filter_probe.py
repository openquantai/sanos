"""Run the pinned quant_data OptionsFilter on portable JSON rows."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from quantdata.widb.day import SCHEMA_ENRICHED_DAY
from quantdata.widb.options_filter import OptionsFilter

request = json.load(sys.stdin)
rows = request["rows"]
count = len(rows)
columns: dict[str, Any] = {}
for name, dtype in SCHEMA_ENRICHED_DAY.items():
    normalized = np.dtype(dtype)
    if normalized.kind == "O":
        columns[name] = np.asarray([f"value-{index}" for index in range(count)], dtype=object)
    elif normalized.kind == "M":
        columns[name] = np.asarray(["2024-01-03"] * count, dtype=normalized)
    elif normalized.kind == "b":
        columns[name] = np.zeros(count, dtype=normalized)
    elif normalized.kind in {"i", "u"}:
        columns[name] = np.arange(1, count + 1, dtype=normalized)
    else:
        columns[name] = np.ones(count, dtype=normalized)

columns["row_id"] = np.arange(count, dtype=np.int32)
for name in rows[0]:
    if name == "row_id":
        continue
    dtype = np.dtype(SCHEMA_ENRICHED_DAY[name])
    columns[name] = np.asarray([row[name] for row in rows], dtype=dtype)

data = pd.DataFrame(columns)
config = dict(request["config"])
if isinstance(config["max_nms"], list):
    config["max_nms"] = tuple(config["max_nms"])
result = OptionsFilter(**config)._filter("SPX", np.datetime64("2024-01-03").astype(object), data)
json.dump(
    {
        "row_ids": result["row_id"].astype(int).tolist(),
        "spot_ref": result["spot_ref"].astype(float).tolist(),
        "spot_prev": result["spot_prev"].astype(float).tolist(),
        "strike": result["strike"].astype(float).tolist(),
        "fwd": result["fwd"].astype(float).tolist(),
        "mkt_call_bid": result["mkt_call_bid"].astype(float).tolist(),
        "mkt_call_ask": result["mkt_call_ask"].astype(float).tolist(),
        "mkt_put_bid": result["mkt_put_bid"].astype(float).tolist(),
        "mkt_put_ask": result["mkt_put_ask"].astype(float).tolist(),
    },
    sys.stdout,
    allow_nan=False,
    sort_keys=True,
)
