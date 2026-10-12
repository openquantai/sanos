from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from sanos.reference import (
    LegacyMarketReference,
    ReferenceFormatError,
    catalog_legacy_market,
    read_legacy_pytree_hdf5,
)


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    text_dtype = h5py.string_dtype(encoding="utf-8")
    if isinstance(value, np.ndarray):
        group.create_dataset(key, data=value)
    elif isinstance(value, dt.date):
        group.create_dataset(f"_1_D{key}", data=np.array(value.isoformat(), dtype=text_dtype))
    elif isinstance(value, str):
        group.create_dataset(f"_1_S{key}", data=np.array(value, dtype=text_dtype))
    elif isinstance(value, (int, float, np.integer, np.floating)):
        group.create_dataset(f"_1_N{key}", data=np.array(value))
    elif isinstance(value, list):
        child = group.create_group(f"_1_L{key}")
        for index, item in enumerate(value):
            _write_value(child, str(index), item)
    elif isinstance(value, dict):
        child = group.create_group(key)
        for child_key, item in value.items():
            _write_value(child, child_key, item)
    else:  # pragma: no cover - fixture helper guard
        raise TypeError(type(value))


def _payload(day: dt.date) -> dict[str, object]:
    masks = [
        np.array([True, False, True], dtype=bool),
        np.array([True, True], dtype=bool),
    ]
    return {
        "schema_version": 4,
        "ticker": "SPX",
        "d": day,
        "spot_ref": 4_000.0,
        "spot_ret": 0.01,
        "market": {
            "expiries": np.array([0.2 / 252.0, 5.0 / 252.0]),
            "expiry_dates": ["2025-01-02", "2025-01-08"],
            "expiry_ids": np.array([202501020, 202501081], dtype=np.int64),
            "is_am": np.array([True, False], dtype=bool),
            "strikes": [np.array([0.9, 1.0, 1.1]), np.array([0.95, 1.05])],
            "call_active": masks,
            "put_active": masks,
        },
        "af_pricer": {},
        "fit_result": {},
    }


def _write_legacy(path: Path, payload: dict[str, object]) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["version"] = "reference-test-v1"
        root = handle.create_group("root")
        for key, value in payload.items():
            _write_value(root, key, value)


def test_read_only_adapter_decodes_populated_and_empty_days(tmp_path: Path) -> None:
    populated = tmp_path / "2025-01-02.h5"
    empty = tmp_path / "2025-01-03.h5"
    _write_legacy(populated, _payload(dt.date(2025, 1, 2)))
    _write_legacy(empty, {})
    before = populated.stat().st_mtime_ns

    reference = LegacyMarketReference(tmp_path)
    loaded = reference.load_day(dt.date(2025, 1, 2))

    assert loaded is not None
    assert loaded["d"] == dt.date(2025, 1, 2)
    assert loaded["ticker"] == "SPX"
    assert reference.load_day(dt.date(2025, 1, 3)) is None
    assert [day for day, _ in reference.iter_populated_days()] == [dt.date(2025, 1, 2)]
    assert populated.stat().st_mtime_ns == before


def test_catalog_is_deterministic_and_counts_physical_empty_marker(tmp_path: Path) -> None:
    _write_legacy(tmp_path / "2025-01-02.h5", _payload(dt.date(2025, 1, 2)))
    _write_legacy(tmp_path / "2025-01-03.h5", {})

    first = catalog_legacy_market(
        tmp_path,
        source_revision="abc123",
        relative_path="dysanos/market/example/SPX",
    )
    second = catalog_legacy_market(
        tmp_path,
        source_revision="abc123",
        relative_path="dysanos/market/example/SPX",
    )
    rendered = first.to_dict()

    assert first.collection_sha256 == second.collection_sha256
    assert rendered["physical_file_count"] == 2
    assert rendered["populated_day_count"] == 1
    assert rendered["empty_marker_dates"] == ["2025-01-03"]
    assert rendered["first_populated_date"] == "2025-01-02"
    assert rendered["last_populated_date"] == "2025-01-02"
    assert rendered["aggregate"] == {
        "active_books": 8,
        "active_call_books": 4,
        "active_point_two_day_books": 4,
        "active_put_books": 4,
        "expiries": 2,
        "strike_nodes": 5,
    }
    assert [record["file"] for record in rendered["files"]] == [
        "2025-01-02.h5",
        "2025-01-03.h5",
    ]


def test_reader_rejects_an_unexpected_root(tmp_path: Path) -> None:
    artifact = tmp_path / "2025-01-02.h5"
    with h5py.File(artifact, "w") as handle:
        handle.create_group("unexpected")

    with pytest.raises(ReferenceFormatError, match="sole HDF5 root group"):
        read_legacy_pytree_hdf5(artifact)


def test_reference_loader_rejects_a_date_only_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "2025-01-02.h5"
    _write_legacy(artifact, {"d": dt.date(2025, 1, 2)})

    with pytest.raises(ReferenceFormatError, match="missing payload keys"):
        LegacyMarketReference(tmp_path).load_day(dt.date(2025, 1, 2))


@pytest.mark.parametrize("malformation", ["metadata_length", "mask_dtype", "strikes"])
def test_reference_loader_rejects_malformed_market_geometry(
    tmp_path: Path,
    malformation: str,
) -> None:
    payload = _payload(dt.date(2025, 1, 2))
    market = payload["market"]
    assert isinstance(market, dict)
    if malformation == "metadata_length":
        market["expiry_dates"] = ["2025-01-02"]
        expected = "inconsistent expiry lengths"
    elif malformation == "mask_dtype":
        calls = market["call_active"]
        assert isinstance(calls, list)
        calls[0] = np.array([1, 0, 1], dtype=np.int8)
        expected = "dtype kind"
    else:
        strikes = market["strikes"]
        assert isinstance(strikes, list)
        strikes[0] = np.array([0.9, np.nan, 1.1], dtype=np.float64)
        expected = "finite, positive and strictly increasing"
    artifact = tmp_path / "2025-01-02.h5"
    _write_legacy(artifact, payload)

    with pytest.raises(ReferenceFormatError, match=expected):
        LegacyMarketReference(tmp_path).load_day(dt.date(2025, 1, 2))


def test_reader_wraps_malformed_date_encoding(tmp_path: Path) -> None:
    artifact = tmp_path / "2025-01-02.h5"
    with h5py.File(artifact, "w") as handle:
        root = handle.create_group("root")
        root.create_dataset("_1_Dd", data=np.bytes_("not-a-date"))

    with pytest.raises(ReferenceFormatError, match="Malformed legacy HDF5 encoding"):
        read_legacy_pytree_hdf5(artifact)


def _run_catalog_cli(root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    return subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "catalog_reference_market.py"),
            str(root),
            str(output),
            "--source-revision",
            "test",
            "--relative-path",
            "test/SPX",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_catalog_cli_refuses_an_input_file_as_output(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    source = root / "2025-01-02.h5"
    source.write_bytes(b"immutable-reference")

    completed = _run_catalog_cli(root, source)

    assert completed.returncode != 0
    assert "outside the read-only reference root" in completed.stderr
    assert source.read_bytes() == b"immutable-reference"


def test_catalog_cli_refuses_every_descendant_output(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    output = root / "catalog.json"

    completed = _run_catalog_cli(root, output)

    assert completed.returncode != 0
    assert "outside the read-only reference root" in completed.stderr
    assert not output.exists()


def test_catalog_cli_refuses_a_symlink_alias_of_an_input(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    source = root / "2025-01-02.h5"
    source.write_bytes(b"immutable-reference")
    output = tmp_path / "catalog.json"
    output.symlink_to(source)

    completed = _run_catalog_cli(root, output)

    assert completed.returncode != 0
    assert "outside the read-only reference root" in completed.stderr
    assert output.is_symlink()
    assert source.read_bytes() == b"immutable-reference"


def test_reference_transform_versions_include_material_helpers() -> None:
    assert "_read_group: 1.0" in read_legacy_pytree_hdf5.version.full
    assert "_record: 1.1" in catalog_legacy_market.version.full
    assert "_LegacyReferenceSchema: 1.0" in catalog_legacy_market.version.full
