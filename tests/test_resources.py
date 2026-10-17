from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanos.resources import (
    packaged_source_revision,
    reference_catalog_path,
    reference_config_path,
    runtime_resource_inventory,
)


def test_checkout_runtime_resources_are_complete() -> None:
    catalog = reference_catalog_path()
    root = Path(__file__).resolve().parents[1]
    assert catalog.name == "reference-market-catalog.json"
    assert catalog.is_relative_to(root)
    assert json.loads(catalog.read_text(encoding="utf-8"))["source_revision"]
    for name in (
        "dysanos-daily-front-reference.json",
        "sanos-public-candidate-reference.json",
        "spx-2022-2025-oracle-gaps.json",
        "widb-spx-reference.json",
    ):
        assert json.loads(reference_config_path(name).read_text(encoding="utf-8"))
    assert packaged_source_revision() is None
    assert runtime_resource_inventory()["source_revision"] == "checkout"


def test_unknown_reference_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown packaged"):
        reference_config_path("machine-local.json")
