"""Verify SANOS wheel and sdist contracts from outside the source checkout."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

HEX_40 = re.compile(r"^[0-9a-f]{40}$")
WHEEL_RESOURCES = {
    "sanos/resources/reference-market-catalog.json",
    "sanos/resources/dysanos-daily-front-reference.json",
    "sanos/resources/sanos-public-candidate-reference.json",
    "sanos/resources/spx-2022-2025-oracle-gaps.json",
    "sanos/resources/widb-spx-reference.json",
    "sanos/resources/source-revision.txt",
}
SDIST_PREFIXES = (
    ".github/",
    "configs/",
    "constraints/",
    "docs/",
    "reference/",
    "reviews/",
    "scripts/",
    "tests/",
    "tex/",
)


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _wheel_inventory(path: Path, expected_revision: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(WHEEL_RESOURCES - names)
        if missing:
            raise RuntimeError(f"Wheel is missing runtime resources: {missing}")
        revision = archive.read("sanos/resources/source-revision.txt").decode("ascii").strip()
    if revision != expected_revision:
        raise RuntimeError(
            f"Wheel provenance {revision!r} differs from expected {expected_revision!r}"
        )


def _extract_sdist(path: Path, destination: Path) -> Path:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise RuntimeError(f"sdist must have exactly one archive root, found {sorted(roots)}")
        root = next(iter(roots))
        relative = {name.removeprefix(root + "/") for name in names}
        for prefix in SDIST_PREFIXES:
            if not any(name.startswith(prefix) for name in relative):
                raise RuntimeError(f"sdist contains no files below {prefix}")
        required = {
            "_source-revision.txt",
            "AGENTS.md",
            "PROGRAM_PLAN.md",
            "PORTING.md",
            "README.md",
            "constraints/phase11-tested-py312.txt",
            "pyproject.toml",
            "setup.py",
            "tests/fixtures/reference_numerics_v1.json",
            "tests/reference_probe.py",
            "reviews/phase-1/reference-market-catalog.json",
        }
        missing = sorted(required - relative)
        if missing:
            raise RuntimeError(f"sdist is missing required files: {missing}")
        archive.extractall(destination, filter="data")
    return destination / root


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("SANOS_SOURCE_REVISION", None)
    environment["WRDS_OFF"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _json_event(output: str) -> dict[str, Any]:
    values = [line for line in output.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"Expected one JSON event, got {len(values)}: {output!r}")
    value = json.loads(values[0])
    if not isinstance(value, dict):
        raise RuntimeError("Structured CLI event must be a JSON object")
    return value


def verify(dist_dir: Path, work_root: Path, expected_revision: str) -> None:
    if HEX_40.fullmatch(expected_revision) is None:
        raise ValueError("--expected-revision must be exact lowercase 40-hex")
    wheels = sorted(dist_dir.glob("sanos-*.whl"))
    sdists = sorted(dist_dir.glob("sanos-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Distribution directory must contain exactly one SANOS wheel and sdist")
    wheel, sdist = wheels[0], sdists[0]
    _wheel_inventory(wheel, expected_revision)

    if work_root.exists():
        raise RuntimeError(f"Verification work root already exists: {work_root}")
    work_root.mkdir(parents=True)
    extracted = _extract_sdist(sdist, work_root / "sdist")
    recorded = (extracted / "_source-revision.txt").read_text(encoding="ascii").strip()
    if recorded != expected_revision:
        raise RuntimeError("sdist source provenance differs from expected revision")

    environment = _clean_environment()
    rebuilt_dir = work_root / "rebuilt"
    rebuilt_dir.mkdir()
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--skip-dependency-check",
            "--outdir",
            str(rebuilt_dir),
        ],
        cwd=extracted,
        environment=environment,
    )
    rebuilt = sorted(rebuilt_dir.glob("sanos-*.whl"))
    if len(rebuilt) != 1:
        raise RuntimeError("Building the sdist did not create exactly one wheel")
    _wheel_inventory(rebuilt[0], expected_revision)

    installed = work_root / "installed"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=work_root,
        environment=environment,
    )
    environment["PYTHONPATH"] = str(installed)
    outside = work_root / "outside"
    outside.mkdir()
    _run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from sanos.reference.experiment_compare import _run_oracle_worker; "
            "result = _run_oracle_worker({'operation': 'preflight'}, "
            "legacy_code_root=Path.cwd(), progress=None); "
            "assert result['protocol'] == "
            "'sanos-experiment-oracle-worker-preflight-v1'; "
            "assert result['torch']; assert result['h5py']; "
            "assert 'quantdata/dysanos/flex/metrics.py' in "
            "result['pinned_source_files']",
        ],
        cwd=outside,
        environment=environment,
    )
    _run([str(installed / "bin" / "sanos"), "--help"], cwd=outside, environment=environment)
    _run([sys.executable, "-m", "sanos", "--help"], cwd=outside, environment=environment)
    event = _json_event(
        _run(
            [sys.executable, "-m", "sanos", "--log-format", "json", "config", "resources"],
            cwd=outside,
            environment=environment,
        )
    )
    resources = event.get("resources")
    if not isinstance(resources, dict) or resources.get("source_revision") != expected_revision:
        raise RuntimeError("Installed config resources did not expose exact provenance")
    for key, value in resources.items():
        if key == "source_revision":
            continue
        if not isinstance(value, str) or not Path(value).is_file():
            raise RuntimeError(f"Installed runtime resource is unavailable: {key}={value!r}")
    legacy = outside / "legacy"
    legacy.mkdir()
    comparison = subprocess.run(
        [
            sys.executable,
            "-m",
            "sanos",
            "--root",
            str(outside / "data"),
            "--log-format",
            "json",
            "reference",
            "compare-daily",
            "--config",
            str(resources["config:widb-spx-reference.json"]),
            "--ticker",
            "SPX",
            "--start-date",
            "2022-01-03",
            "--end-date",
            "2022-01-03",
            "--source-generation",
            "default-v1",
            "--legacy-root",
            str(legacy),
            "--output",
            str(outside / "comparison.json"),
        ],
        cwd=outside,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if comparison.returncode != 4 or "reference-market-catalog.json does not exist" in (
        comparison.stdout + comparison.stderr
    ):
        raise RuntimeError(
            "Installed compare-daily did not resolve its packaged default catalog:\n"
            f"stdout:\n{comparison.stdout}\nstderr:\n{comparison.stderr}"
        )
    _run(
        [
            sys.executable,
            "-c",
            "from sanos.reference import repository_source_revision; "
            "from sanos.resources import reference_config_path; "
            "from sanos.datasets import InternalDatasetConfig; "
            "from sanos.models.dysanos import ExperimentConfig; "
            "from sanos.datasets.public import PublicProjectionConfig; "
            "import json; "
            "assert repository_source_revision(); "
            "InternalDatasetConfig.from_document(json.loads(reference_config_path(" 
            "'widb-spx-reference.json').read_text())); "
            "ExperimentConfig.from_document(json.loads(reference_config_path(" 
            "'dysanos-daily-front-reference.json').read_text())); "
            "PublicProjectionConfig.from_document(json.loads(reference_config_path(" 
            "'sanos-public-candidate-reference.json').read_text()))",
        ],
        cwd=outside,
        environment=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    arguments = parser.parse_args()
    try:
        verify(
            arguments.dist_dir.resolve(),
            arguments.work_root.resolve(),
            arguments.expected_revision,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"distribution verification failed: {exc}", file=sys.stderr)
        return 1
    print("distribution verification passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
