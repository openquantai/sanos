"""Paths and provenance embedded in installed SANOS distributions."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_NAMES = frozenset(
    {
        "dysanos-daily-front-reference.json",
        "sanos-public-candidate-reference.json",
        "spx-2022-2025-oracle-gaps.json",
        "widb-spx-reference.json",
    }
)


def _checkout_root() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").is_file() and (root / "src" / "sanos").is_dir():
        return root
    return None


def _installed_path(name: str) -> Path:
    resource = files(__package__).joinpath(name)
    path = Path(str(resource))
    if not path.is_file():
        raise RuntimeError(f"Installed SANOS runtime resource is missing: {name}")
    return path


def reference_catalog_path() -> Path:
    """Return the immutable legacy catalog in a checkout or installed wheel."""

    root = _checkout_root()
    if root is not None:
        candidate = root / "reviews" / "phase-1" / "reference-market-catalog.json"
        if candidate.is_file():
            return candidate
    return _installed_path("reference-market-catalog.json")


def reference_config_path(name: str) -> Path:
    """Return a checked portable reference configuration by exact file name."""

    if name not in _CONFIG_NAMES:
        raise ValueError(f"Unknown packaged reference configuration: {name!r}")
    root = _checkout_root()
    if root is not None:
        candidate = root / "configs" / name
        if candidate.is_file():
            return candidate
    return _installed_path(name)


def packaged_source_revision() -> str | None:
    """Return exact build provenance when running from a built distribution."""

    try:
        value = _installed_path("source-revision.txt").read_text(encoding="ascii").strip()
    except (OSError, RuntimeError, UnicodeError):
        return None
    if _HEX_40.fullmatch(value) is None:
        raise RuntimeError("Packaged SANOS source revision is malformed")
    return value


def runtime_resource_inventory() -> dict[str, str]:
    """Return stable paths for the runtime resources used by operator tooling."""

    return {
        "reference_catalog": str(reference_catalog_path()),
        **{
            f"config:{name}": str(reference_config_path(name))
            for name in sorted(_CONFIG_NAMES)
        },
        "source_revision": packaged_source_revision() or "checkout",
    }


__all__ = [
    "packaged_source_revision",
    "reference_catalog_path",
    "reference_config_path",
    "runtime_resource_inventory",
]
