"""Closed, path-free runtime provenance for production evidence."""

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import struct
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast

import cvxpy as cp
import numpy as np
from cdxcore.version import version

from sanos.artifacts import atomic_create_json
from sanos.config import canonical_json, canonical_sha256, canonical_value

RUNTIME_FINGERPRINT_SCHEMA = "sanos-runtime-fingerprint-v1"

_PACKAGE_MODULES = {
    "cdxcore": "cdxcore",
    "clarabel": "clarabel",
    "cvxpy": "cvxpy",
    "h5py": "h5py",
    "highspy": "highspy",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "numpy": "numpy",
    "pandas-market-calendars": "pandas_market_calendars",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "threadpoolctl": "threadpoolctl",
}
_OPTIONAL_PACKAGES = frozenset({"jax", "jaxlib"})
_REQUIRED_SOLVERS = frozenset({"CLARABEL", "HIGHS"})
_THREAD_ENVIRONMENT = (
    "BLIS_NUM_THREADS",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_DYNAMIC_THREAD_ENVIRONMENT = frozenset({"MKL_DYNAMIC", "OMP_DYNAMIC"})
_THREAD_COUNT_VALUE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
_THREAD_DYNAMIC_VALUE = re.compile(r"^(?:0|1|false|true)$", re.IGNORECASE)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class RuntimeFingerprintError(RuntimeError):
    """Raised when runtime provenance is unsafe, malformed, or inconsistent."""


@version("1.0")
def _safe_text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RuntimeFingerprintError(f"{label} must be a safe string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise RuntimeFingerprintError(f"{label} contains unsafe text")
    if "/" in value or "\\" in value:
        raise RuntimeFingerprintError(f"{label} contains path-like text")
    return value


@version("1.0")
def _installed_distribution_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


@version("1.0")
def _imported_runtime_version(distribution: str, module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    if distribution == "highspy":
        highs = module.Highs()
        return str(highs.version())
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


@version("1.0", dependencies=[_imported_runtime_version, _installed_distribution_version])
def _package_identities() -> dict[str, dict[str, str | None]]:
    return {
        distribution: {
            "distribution_version": _installed_distribution_version(distribution),
            "runtime_version": _imported_runtime_version(distribution, module),
        }
        for distribution, module in sorted(_PACKAGE_MODULES.items())
    }


@version("1.0")
def _valid_thread_value(name: str, value: object) -> bool:
    if not isinstance(value, str) or len(value) > 128:
        return False
    pattern = (
        _THREAD_DYNAMIC_VALUE
        if name in _DYNAMIC_THREAD_ENVIRONMENT
        else _THREAD_COUNT_VALUE
    )
    return pattern.fullmatch(value) is not None


@version("1.0", dependencies=[_safe_text, _valid_thread_value])
def _thread_environment() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in _THREAD_ENVIRONMENT:
        value = os.environ.get(name)
        if value is not None:
            if not _valid_thread_value(name, value):
                raise RuntimeFingerprintError(
                    f"Thread environment variable {name} has an unsafe value"
                )
            _safe_text(value, f"thread_environment.{name}")
        values[name] = value
    return values


@version("1.0", dependencies=[_safe_text])
def _optional_safe_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _safe_text(value, label)


@version("1.0", dependencies=[_optional_safe_text])
def _numpy_build_library(name: str) -> dict[str, object]:
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {}) if isinstance(config, Mapping) else {}
    value = dependencies.get(name, {}) if isinstance(dependencies, Mapping) else {}
    if not isinstance(value, Mapping):
        value = {}
    return {
        "configuration": _optional_safe_text(
            value.get("openblas configuration") or value.get("configuration"),
            f"numpy.build.{name}.configuration",
        ),
        "detection_method": _optional_safe_text(
            value.get("detection method"), f"numpy.build.{name}.detection_method"
        ),
        "found": value.get("found") if isinstance(value.get("found"), bool) else None,
        "name": _optional_safe_text(value.get("name"), f"numpy.build.{name}.name"),
        "version": _optional_safe_text(
            value.get("version"), f"numpy.build.{name}.version"
        ),
    }


@version("1.0", dependencies=[_safe_text])
def _numpy_simd() -> dict[str, list[str]]:
    config = getattr(np.__config__, "CONFIG", {})
    raw = config.get("SIMD Extensions", {}) if isinstance(config, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    result: dict[str, list[str]] = {}
    for output_name, source_name in (
        ("baseline", "baseline"),
        ("found", "found"),
        ("not_found", "not found"),
    ):
        values = raw.get(source_name, [])
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RuntimeFingerprintError(f"NumPy SIMD {source_name} must be a sequence")
        result[output_name] = sorted(
            {_safe_text(value, f"numpy.simd.{output_name}") for value in values}
        )
    return result


@version("1.0", dependencies=[_optional_safe_text])
def _numpy_runtime_libraries() -> list[dict[str, object]]:
    try:
        module = importlib.import_module("threadpoolctl")
    except ModuleNotFoundError as exc:
        raise RuntimeFingerprintError(
            "Required runtime package threadpoolctl is unavailable"
        ) from exc
    raw = module.threadpool_info()
    if not isinstance(raw, list):
        raise RuntimeFingerprintError("threadpoolctl returned malformed runtime data")
    result: list[dict[str, object]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise RuntimeFingerprintError("threadpoolctl returned malformed library data")
        threads = value.get("num_threads")
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            threads = None
        result.append(
            {
                "architecture": _optional_safe_text(
                    value.get("architecture"), f"numpy.runtime_libraries.{index}.architecture"
                ),
                "internal_api": _optional_safe_text(
                    value.get("internal_api"), f"numpy.runtime_libraries.{index}.internal_api"
                ),
                "num_threads": threads,
                "threading_layer": _optional_safe_text(
                    value.get("threading_layer"),
                    f"numpy.runtime_libraries.{index}.threading_layer",
                ),
                "user_api": _optional_safe_text(
                    value.get("user_api"), f"numpy.runtime_libraries.{index}.user_api"
                ),
                "version": _optional_safe_text(
                    value.get("version"), f"numpy.runtime_libraries.{index}.version"
                ),
            }
        )
    return sorted(
        result,
        key=lambda item: tuple("" if value is None else str(value) for value in item.values()),
    )


@version(
    "1.0",
    dependencies=[_numpy_build_library, _numpy_runtime_libraries, _numpy_simd],
)
def _numpy_identity() -> dict[str, object]:
    return {
        "build": {
            "blas": _numpy_build_library("blas"),
            "lapack": _numpy_build_library("lapack"),
        },
        "runtime_libraries": _numpy_runtime_libraries(),
        "simd": _numpy_simd(),
    }


@version(
    "1.0",
    dependencies=[
        _numpy_identity,
        _package_identities,
        _safe_text,
        _thread_environment,
        canonical_sha256,
        canonical_value,
    ],
)
def runtime_fingerprint() -> dict[str, Any]:
    """Return canonical, secret-free identities for the active numerical runtime."""

    libc_name, libc_version = platform.libc_ver()
    logical_cpu_count = os.cpu_count()
    if not isinstance(logical_cpu_count, int) or logical_cpu_count < 1:
        raise RuntimeFingerprintError("Logical CPU count is unavailable")
    affinity_cpu_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity_cpu_count = len(os.sched_getaffinity(0))
        if affinity_cpu_count < 1:
            raise RuntimeFingerprintError("Process CPU affinity is empty")
    body = cast(
        dict[str, Any],
        canonical_value(
            {
                "schema": RUNTIME_FINGERPRINT_SCHEMA,
                "python": {
                    "abiflags": sys.abiflags,
                    "byteorder": sys.byteorder,
                    "cache_tag": sys.implementation.cache_tag,
                    "compiler": platform.python_compiler(),
                    "implementation": platform.python_implementation(),
                    "soabi": sysconfig.get_config_var("SOABI"),
                    "version": sys.version.replace("\n", " "),
                    "version_info": list(sys.version_info[:5]),
                },
                "platform": {
                    "affinity_cpu_count": affinity_cpu_count,
                    "available_cpu_count": affinity_cpu_count or logical_cpu_count,
                    "libc_name": libc_name or None,
                    "libc_version": libc_version or None,
                    "logical_cpu_count": logical_cpu_count,
                    "machine": platform.machine(),
                    "pointer_bits": struct.calcsize("P") * 8,
                    "release": platform.release(),
                    "system": platform.system(),
                },
                "packages": _package_identities(),
                "cvxpy_solvers": sorted(set(cp.installed_solvers())),
                "thread_environment": _thread_environment(),
                "numpy": _numpy_identity(),
            }
        ),
    )
    for section in ("python", "platform"):
        for key, value in body[section].items():
            if isinstance(value, str):
                _safe_text(value, f"{section}.{key}", allow_empty=key == "abiflags")
    return {"fingerprint_id": canonical_sha256(body), **body}


@version(
    "1.0",
    dependencies=[
        _safe_text,
        _valid_thread_value,
        canonical_json,
        canonical_sha256,
        canonical_value,
    ],
)
def validate_runtime_fingerprint(value: object) -> dict[str, Any]:
    """Strictly validate and return one canonical runtime fingerprint."""

    if not isinstance(value, Mapping) or set(value) != {
        "fingerprint_id",
        "schema",
        "python",
        "platform",
        "packages",
        "cvxpy_solvers",
        "thread_environment",
        "numpy",
    }:
        raise RuntimeFingerprintError("Runtime fingerprint fields are missing or unexpected")
    document = cast(dict[str, Any], canonical_value(value))
    if document.get("schema") != RUNTIME_FINGERPRINT_SCHEMA:
        raise RuntimeFingerprintError("Runtime fingerprint schema changed")
    python = document.get("python")
    if not isinstance(python, Mapping) or set(python) != {
        "abiflags",
        "byteorder",
        "cache_tag",
        "compiler",
        "implementation",
        "soabi",
        "version",
        "version_info",
    }:
        raise RuntimeFingerprintError("Runtime fingerprint Python identity is malformed")
    for key in ("byteorder", "cache_tag", "compiler", "implementation", "soabi", "version"):
        _safe_text(python.get(key), f"python.{key}")
    _safe_text(python.get("abiflags"), "python.abiflags", allow_empty=True)
    version_info = python.get("version_info")
    if (
        not isinstance(version_info, list)
        or len(version_info) != 5
        or any(
            not isinstance(version_info[index], int)
            or isinstance(version_info[index], bool)
            for index in (0, 1, 2, 4)
        )
        or version_info[3] not in {"alpha", "beta", "candidate", "final"}
    ):
        raise RuntimeFingerprintError("Runtime fingerprint Python version_info is malformed")
    _safe_text(version_info[3], "python.version_info.releaselevel")

    platform_value = document.get("platform")
    if not isinstance(platform_value, Mapping) or set(platform_value) != {
        "affinity_cpu_count",
        "available_cpu_count",
        "libc_name",
        "libc_version",
        "logical_cpu_count",
        "machine",
        "pointer_bits",
        "release",
        "system",
    }:
        raise RuntimeFingerprintError("Runtime fingerprint platform identity is malformed")
    for key in ("machine", "release", "system"):
        _safe_text(platform_value.get(key), f"platform.{key}")
    for key in ("libc_name", "libc_version"):
        if platform_value.get(key) is not None:
            _safe_text(platform_value.get(key), f"platform.{key}")
    if platform_value.get("pointer_bits") not in {32, 64}:
        raise RuntimeFingerprintError("Runtime fingerprint pointer width is invalid")
    logical_cpu_count = platform_value.get("logical_cpu_count")
    affinity_cpu_count = platform_value.get("affinity_cpu_count")
    available_cpu_count = platform_value.get("available_cpu_count")
    if (
        not isinstance(logical_cpu_count, int)
        or isinstance(logical_cpu_count, bool)
        or logical_cpu_count < 1
        or (
            affinity_cpu_count is not None
            and (
                not isinstance(affinity_cpu_count, int)
                or isinstance(affinity_cpu_count, bool)
                or affinity_cpu_count < 1
            )
        )
        or not isinstance(available_cpu_count, int)
        or isinstance(available_cpu_count, bool)
        or available_cpu_count < 1
        or available_cpu_count != (affinity_cpu_count or logical_cpu_count)
    ):
        raise RuntimeFingerprintError("Runtime fingerprint CPU inventory is malformed")

    packages = document.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(_PACKAGE_MODULES):
        raise RuntimeFingerprintError("Runtime fingerprint package set changed")
    for name, identity in packages.items():
        if not isinstance(identity, Mapping) or set(identity) != {
            "distribution_version",
            "runtime_version",
        }:
            raise RuntimeFingerprintError(f"Runtime fingerprint package {name} is malformed")
        distribution = identity.get("distribution_version")
        imported = identity.get("runtime_version")
        if name not in _OPTIONAL_PACKAGES and (distribution is None or imported is None):
            raise RuntimeFingerprintError(f"Required runtime package {name} is unavailable")
        if (distribution is None) != (imported is None):
            raise RuntimeFingerprintError(
                f"Runtime package {name} has inconsistent installed/imported state"
            )
        if distribution is not None:
            _safe_text(distribution, f"packages.{name}.distribution_version")
            _safe_text(imported, f"packages.{name}.runtime_version")

    solvers = document.get("cvxpy_solvers")
    if (
        not isinstance(solvers, list)
        or any(not isinstance(item, str) for item in solvers)
        or solvers != sorted(set(solvers))
        or not _REQUIRED_SOLVERS.issubset(solvers)
    ):
        raise RuntimeFingerprintError("CVXPY solver inventory is malformed or incomplete")
    for solver in solvers:
        _safe_text(solver, "cvxpy_solvers")

    threads = document.get("thread_environment")
    if not isinstance(threads, Mapping) or tuple(sorted(threads)) != _THREAD_ENVIRONMENT:
        raise RuntimeFingerprintError("Thread environment allowlist changed")
    for name, thread_value in threads.items():
        if thread_value is not None and not _valid_thread_value(name, thread_value):
            raise RuntimeFingerprintError(
                f"Thread environment variable {name} has an unsafe value"
            )

    numpy_value = document.get("numpy")
    if not isinstance(numpy_value, Mapping) or set(numpy_value) != {
        "build",
        "runtime_libraries",
        "simd",
    }:
        raise RuntimeFingerprintError("NumPy runtime identity is malformed")
    build = numpy_value.get("build")
    if not isinstance(build, Mapping) or set(build) != {"blas", "lapack"}:
        raise RuntimeFingerprintError("NumPy build identity is malformed")
    library_fields = {"configuration", "detection_method", "found", "name", "version"}
    for name, identity in build.items():
        if not isinstance(identity, Mapping) or set(identity) != library_fields:
            raise RuntimeFingerprintError(f"NumPy {name} build identity is malformed")
        if identity.get("found") not in {True, False, None}:
            raise RuntimeFingerprintError(f"NumPy {name} found flag is malformed")
        for key in library_fields - {"found"}:
            if identity.get(key) is not None:
                _safe_text(identity.get(key), f"numpy.build.{name}.{key}")
    simd = numpy_value.get("simd")
    if not isinstance(simd, Mapping) or set(simd) != {"baseline", "found", "not_found"}:
        raise RuntimeFingerprintError("NumPy SIMD identity is malformed")
    for name, items in simd.items():
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) for item in items)
            or items != sorted(set(items))
        ):
            raise RuntimeFingerprintError(f"NumPy SIMD {name} identity is malformed")
        for item in items:
            _safe_text(item, f"numpy.simd.{name}")
    runtime_libraries = numpy_value.get("runtime_libraries")
    runtime_fields = {
        "architecture",
        "internal_api",
        "num_threads",
        "threading_layer",
        "user_api",
        "version",
    }
    if not isinstance(runtime_libraries, list) or not runtime_libraries:
        raise RuntimeFingerprintError("NumPy runtime library identity is malformed")
    canonical_libraries: list[dict[str, object]] = []
    for index, identity in enumerate(runtime_libraries):
        if not isinstance(identity, Mapping) or set(identity) != runtime_fields:
            raise RuntimeFingerprintError("NumPy runtime library fields changed")
        for key in runtime_fields - {"num_threads"}:
            if identity.get(key) is not None:
                _safe_text(identity.get(key), f"numpy.runtime_libraries.{index}.{key}")
        threads_value = identity.get("num_threads")
        if threads_value is not None and (
            not isinstance(threads_value, int)
            or isinstance(threads_value, bool)
            or threads_value < 1
        ):
            raise RuntimeFingerprintError("NumPy runtime library thread count is malformed")
        canonical_libraries.append(dict(identity))
    expected_libraries = sorted(
        canonical_libraries,
        key=lambda item: tuple("" if value is None else str(value) for value in item.values()),
    )
    if canonical_libraries != expected_libraries or len(
        {canonical_json(item) for item in canonical_libraries}
    ) != len(canonical_libraries):
        raise RuntimeFingerprintError("NumPy runtime libraries are not canonical")
    blas_libraries = [
        item for item in canonical_libraries if item.get("user_api") == "blas"
    ]
    if not blas_libraries or any(
        item.get("internal_api") is None
        or item.get("version") is None
        or item.get("num_threads") is None
        for item in blas_libraries
    ):
        raise RuntimeFingerprintError("BLAS runtime identity is incomplete")

    fingerprint_id = document.get("fingerprint_id")
    if not isinstance(fingerprint_id, str) or _HEX_64.fullmatch(fingerprint_id) is None:
        raise RuntimeFingerprintError("Runtime fingerprint identity is invalid")
    body = {key: item for key, item in document.items() if key != "fingerprint_id"}
    if canonical_sha256(body) != fingerprint_id:
        raise RuntimeFingerprintError("Runtime fingerprint identity does not bind its content")
    return document


@version("1.0")
def _assert_runtime_path_has_no_symlinks(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise RuntimeFingerprintError("Runtime fingerprint path contains a symlink")
    return absolute


@version(
    "1.0",
    dependencies=[_assert_runtime_path_has_no_symlinks, validate_runtime_fingerprint],
)
def load_runtime_fingerprint(path: Path) -> dict[str, Any]:
    """Load a strict fingerprint without following a symlinked path component."""

    safe_path = _assert_runtime_path_has_no_symlinks(path)
    if not safe_path.is_file():
        raise RuntimeFingerprintError(f"Runtime fingerprint is missing: {path}")
    try:
        value = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeFingerprintError("Cannot decode runtime fingerprint") from exc
    return validate_runtime_fingerprint(value)


@version(
    "1.0",
    dependencies=[
        _assert_runtime_path_has_no_symlinks,
        atomic_create_json,
        canonical_json,
        load_runtime_fingerprint,
        validate_runtime_fingerprint,
    ],
)
def write_runtime_fingerprint(path: Path, fingerprint: Mapping[str, Any]) -> bool:
    """Create immutable provenance, or reuse only canonical-identical content."""

    document = validate_runtime_fingerprint(fingerprint)
    safe_path = _assert_runtime_path_has_no_symlinks(path)
    _assert_runtime_path_has_no_symlinks(safe_path.parent)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_runtime_path_has_no_symlinks(safe_path.parent)
    if safe_path.exists() or safe_path.is_symlink():
        existing = load_runtime_fingerprint(safe_path)
        if canonical_json(existing) != canonical_json(document):
            raise RuntimeFingerprintError(
                f"Runtime fingerprint already exists with other content: {path}"
            )
        return False
    try:
        atomic_create_json(safe_path, document)
    except FileExistsError:
        existing = load_runtime_fingerprint(safe_path)
        if canonical_json(existing) != canonical_json(document):
            raise RuntimeFingerprintError(
                f"Runtime fingerprint already exists with other content: {path}"
            )
        return False
    return True


__all__ = [
    "RUNTIME_FINGERPRINT_SCHEMA",
    "RuntimeFingerprintError",
    "load_runtime_fingerprint",
    "runtime_fingerprint",
    "validate_runtime_fingerprint",
    "write_runtime_fingerprint",
]
