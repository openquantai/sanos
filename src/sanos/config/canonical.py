"""Canonical serialization used by configuration and artifact identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from cdxcore.version import version


class CanonicalizationError(TypeError):
    """Raised when a value cannot participate in a deterministic identity."""


@version("1.0")
def canonical_value(value: Any) -> Any:
    """Convert supported Python values to a deterministic JSON value."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("Canonical configuration contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return canonical_value(value.item())
    if isinstance(value, np.ndarray):
        return canonical_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise CanonicalizationError("Canonical datetimes must carry an explicit timezone")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return canonical_value(asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"Canonical mapping keys must be strings, found {type(key).__name__}"
                )
            normalized[key] = canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [canonical_value(item) for item in value]
    raise CanonicalizationError(
        f"Unsupported canonical configuration value of type {type(value).__name__}"
    )


@version("1.0", dependencies=[canonical_value])
def canonical_json(value: Any) -> str:
    """Return compact canonical JSON without a trailing newline."""

    return json.dumps(
        canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@version("1.0", dependencies=[canonical_json])
def canonical_sha256(value: Any) -> str:
    """Hash a value's canonical UTF-8 JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
