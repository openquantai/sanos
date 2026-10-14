from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

import sanos.runtime as runtime_module
from sanos.config import canonical_json, canonical_sha256
from sanos.runtime import (
    RUNTIME_FINGERPRINT_SCHEMA,
    RuntimeFingerprintError,
    load_runtime_fingerprint,
    runtime_fingerprint,
    validate_runtime_fingerprint,
    write_runtime_fingerprint,
)


def _with_identity(value: dict[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "fingerprint_id"}
    return {"fingerprint_id": canonical_sha256(body), **body}


def test_runtime_fingerprint_is_closed_canonical_and_path_free() -> None:
    fingerprint = runtime_fingerprint()
    assert fingerprint["schema"] == RUNTIME_FINGERPRINT_SCHEMA
    assert validate_runtime_fingerprint(fingerprint) == fingerprint
    assert {"CLARABEL", "HIGHS"}.issubset(fingerprint["cvxpy_solvers"])
    serialized = canonical_json(fingerprint)
    for forbidden in (
        str(Path.home()),
        str(Path.cwd()),
        platform_node(),
        '"filepath"',
        '"include directory"',
        '"lib directory"',
    ):
        if forbidden:
            assert forbidden not in serialized


def test_runtime_functions_have_resolvable_explicit_versions() -> None:
    expected_dependencies: dict[Any, set[str]] = {
        runtime_fingerprint: {
            "_numpy_identity",
            "_package_identities",
            "_safe_text",
            "_thread_environment",
            "canonical_sha256",
            "canonical_value",
        },
        validate_runtime_fingerprint: {
            "_safe_text",
            "_valid_thread_value",
            "canonical_json",
            "canonical_sha256",
            "canonical_value",
        },
        load_runtime_fingerprint: {
            "_assert_runtime_path_has_no_symlinks",
            "validate_runtime_fingerprint",
        },
        write_runtime_fingerprint: {
            "_assert_runtime_path_has_no_symlinks",
            "atomic_create_json",
            "canonical_json",
            "load_runtime_fingerprint",
            "validate_runtime_fingerprint",
        },
    }
    for function, expected in expected_dependencies.items():
        versioned = cast(Any, function)
        assert str(versioned.version)
        assert set(dict(versioned.version.dependencies[1])) == expected


def platform_node() -> str:
    import platform

    return platform.node()


def test_runtime_fingerprint_mutation_requires_a_new_identity() -> None:
    original = runtime_fingerprint()
    changed = deepcopy(original)
    changed["platform"]["release"] += "-test"  # type: ignore[index]
    with pytest.raises(RuntimeFingerprintError, match="does not bind"):
        validate_runtime_fingerprint(changed)
    changed = _with_identity(changed)
    assert validate_runtime_fingerprint(changed) == changed
    assert changed["fingerprint_id"] != original["fingerprint_id"]


def test_runtime_validation_rejects_path_text_and_wrong_python_version_types() -> None:
    fingerprint = runtime_fingerprint()
    path_release = deepcopy(fingerprint)
    path_release["python"]["version_info"][3] = "/secret/release"  # type: ignore[index]
    path_release = _with_identity(path_release)
    with pytest.raises(RuntimeFingerprintError, match="version_info"):
        validate_runtime_fingerprint(path_release)

    string_serial = deepcopy(fingerprint)
    string_serial["python"]["version_info"][4] = "0"  # type: ignore[index]
    string_serial = _with_identity(string_serial)
    with pytest.raises(RuntimeFingerprintError, match="version_info"):
        validate_runtime_fingerprint(string_serial)


def test_runtime_collection_canonicalizes_solver_and_package_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_solvers = runtime_module.cp.installed_solvers()
    monkeypatch.setattr(
        runtime_module.cp,
        "installed_solvers",
        lambda: [*reversed(original_solvers), *original_solvers],
    )
    collected = runtime_fingerprint()
    assert collected["cvxpy_solvers"] == sorted(set(original_solvers))

    reordered = deepcopy(collected)
    packages = reordered["packages"]
    reordered["packages"] = dict(reversed(list(packages.items())))  # type: ignore[union-attr]
    assert validate_runtime_fingerprint(reordered) == collected


def test_thread_environment_unset_set_and_unsafe_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in runtime_module._THREAD_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    unset = runtime_fingerprint()
    assert all(value is None for value in unset["thread_environment"].values())
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    configured = runtime_fingerprint()
    assert configured["thread_environment"]["OMP_NUM_THREADS"] == "2"
    assert configured["fingerprint_id"] != unset["fingerprint_id"]

    monkeypatch.setenv("OMP_NUM_THREADS", "/secret/sentinel")
    with pytest.raises(RuntimeFingerprintError) as caught:
        runtime_fingerprint()
    assert "/secret/sentinel" not in str(caught.value)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("MKL_DYNAMIC", "FALSE"),
        ("OMP_DYNAMIC", "true"),
        ("RAYON_NUM_THREADS", "3"),
    ),
)
def test_runtime_fingerprint_binds_solver_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    for variable in runtime_module._THREAD_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    baseline = runtime_fingerprint()
    monkeypatch.setenv(name, value)
    configured = runtime_fingerprint()
    assert configured["thread_environment"][name] == value
    assert configured["fingerprint_id"] != baseline["fingerprint_id"]


def test_runtime_fingerprint_binds_cpu_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(
        runtime_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(4)),
        raising=False,
    )
    four_cpus = runtime_fingerprint()
    assert four_cpus["platform"]["logical_cpu_count"] == 32
    assert four_cpus["platform"]["affinity_cpu_count"] == 4
    assert four_cpus["platform"]["available_cpu_count"] == 4

    monkeypatch.setattr(
        runtime_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(3)),
    )
    three_cpus = runtime_fingerprint()
    assert three_cpus["fingerprint_id"] != four_cpus["fingerprint_id"]


def test_numpy_build_capture_discards_path_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "/secret/build/location"
    monkeypatch.setattr(
        runtime_module.np.__config__,
        "CONFIG",
        {
            "Build Dependencies": {
                "blas": {
                    "name": "safe-blas",
                    "version": "1.0",
                    "found": True,
                    "detection method": "pkgconfig",
                    "openblas configuration": "pthreads architecture effective-threads-2",
                    "include directory": sentinel,
                    "lib directory": sentinel,
                },
                "lapack": {},
            },
            "SIMD Extensions": {
                "baseline": ["B", "A", "A"],
                "found": [],
                "not found": [],
            },
        },
    )
    identity = runtime_module._numpy_identity()
    assert sentinel not in canonical_json(identity)
    assert identity["simd"]["baseline"] == ["A", "B"]  # type: ignore[index]


def test_package_identity_exposes_distribution_runtime_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_installed_distribution_version", lambda name: "1")
    monkeypatch.setattr(
        runtime_module,
        "_imported_runtime_version",
        lambda distribution, module: "2" if distribution == "highspy" else "1",
    )
    identities = runtime_module._package_identities()
    assert identities["highspy"] == {
        "distribution_version": "1",
        "runtime_version": "2",
    }


def test_runtime_validation_rejects_missing_required_and_runtime_blas() -> None:
    fingerprint = runtime_fingerprint()
    required_missing = deepcopy(fingerprint)
    required_missing["packages"]["highspy"] = {  # type: ignore[index]
        "distribution_version": None,
        "runtime_version": None,
    }
    required_missing = _with_identity(required_missing)
    with pytest.raises(RuntimeFingerprintError, match="Required runtime package highspy"):
        validate_runtime_fingerprint(required_missing)

    threadpool_missing = deepcopy(fingerprint)
    threadpool_missing["packages"]["threadpoolctl"] = {  # type: ignore[index]
        "distribution_version": None,
        "runtime_version": None,
    }
    threadpool_missing = _with_identity(threadpool_missing)
    with pytest.raises(RuntimeFingerprintError, match="threadpoolctl"):
        validate_runtime_fingerprint(threadpool_missing)

    no_runtime_blas = deepcopy(fingerprint)
    no_runtime_blas["numpy"]["runtime_libraries"] = []  # type: ignore[index]
    no_runtime_blas = _with_identity(no_runtime_blas)
    with pytest.raises(RuntimeFingerprintError, match="runtime library"):
        validate_runtime_fingerprint(no_runtime_blas)

    optional_missing = deepcopy(fingerprint)
    optional_missing["packages"]["jax"] = {  # type: ignore[index]
        "distribution_version": None,
        "runtime_version": None,
    }
    optional_missing = _with_identity(optional_missing)
    assert validate_runtime_fingerprint(optional_missing) == optional_missing


def test_runtime_writer_reuses_identical_and_rejects_conflict(tmp_path: Path) -> None:
    output = tmp_path / "runtime-fingerprint.json"
    fingerprint = runtime_fingerprint()
    assert write_runtime_fingerprint(output, fingerprint) is True
    assert write_runtime_fingerprint(output, fingerprint) is False
    assert load_runtime_fingerprint(output) == fingerprint

    changed = deepcopy(fingerprint)
    changed["platform"]["release"] += "-other"  # type: ignore[index]
    changed = _with_identity(changed)
    with pytest.raises(RuntimeFingerprintError, match="other content"):
        write_runtime_fingerprint(output, changed)


def test_runtime_writer_rejects_symlinked_leaf_and_parent(tmp_path: Path) -> None:
    fingerprint = runtime_fingerprint()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    leaf = tmp_path / "runtime-fingerprint.json"
    leaf.symlink_to(target)
    with pytest.raises(RuntimeFingerprintError, match="symlink"):
        write_runtime_fingerprint(leaf, fingerprint)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeFingerprintError, match="symlink"):
        write_runtime_fingerprint(linked_parent / "runtime.json", fingerprint)


def test_runtime_writer_does_not_follow_legacy_lock_symlink(tmp_path: Path) -> None:
    output = tmp_path / "runtime-fingerprint.json"
    redirected = tmp_path / "redirected-lock-target"
    redirected.write_text("sentinel", encoding="utf-8")
    legacy_lock = tmp_path / f".{output.name}.lock"
    legacy_lock.symlink_to(redirected)

    assert write_runtime_fingerprint(output, runtime_fingerprint()) is True
    assert redirected.read_text(encoding="utf-8") == "sentinel"
    assert legacy_lock.is_symlink()


def test_runtime_writer_serializes_concurrent_identical_writers(tmp_path: Path) -> None:
    output = tmp_path / "runtime-fingerprint.json"
    fingerprint = runtime_fingerprint()
    barrier = threading.Barrier(2)
    results: list[bool] = []
    failures: list[BaseException] = []

    def write() -> None:
        try:
            barrier.wait()
            results.append(write_runtime_fingerprint(output, fingerprint))
        except BaseException as exc:  # pragma: no cover - failure captured for assertion
            failures.append(exc)

    threads = [threading.Thread(target=write) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not any(thread.is_alive() for thread in threads)
    assert not failures
    assert sorted(results) == [False, True]
    assert json.loads(output.read_text(encoding="utf-8")) == fingerprint


def test_runtime_writer_serializes_concurrent_conflicting_writers(tmp_path: Path) -> None:
    output = tmp_path / "runtime-fingerprint.json"
    first = runtime_fingerprint()
    second = deepcopy(first)
    second["platform"]["release"] += "-other"  # type: ignore[index]
    second = _with_identity(second)
    barrier = threading.Barrier(2)
    created: list[bool] = []
    failures: list[BaseException] = []

    def write(value: dict[str, object]) -> None:
        try:
            barrier.wait()
            created.append(write_runtime_fingerprint(output, value))
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=write, args=(first,)),
        threading.Thread(target=write, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not any(thread.is_alive() for thread in threads)
    assert created == [True]
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeFingerprintError)
    assert load_runtime_fingerprint(output) in (first, second)
