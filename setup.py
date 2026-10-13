"""Setuptools hooks for immutable runtime resources and source provenance."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_RESOURCES = {
    ROOT / "reviews" / "phase-1" / "reference-market-catalog.json": (
        "reference-market-catalog.json"
    ),
    **{
        ROOT / "configs" / name: name
        for name in (
            "dysanos-daily-front-reference.json",
            "sanos-public-candidate-reference.json",
            "spx-2022-2025-oracle-gaps.json",
            "widb-spx-reference.json",
        )
    },
}


def source_revision() -> str:
    explicit = os.environ.get("SANOS_SOURCE_REVISION")
    if explicit is not None:
        if HEX_40.fullmatch(explicit) is None:
            raise RuntimeError("SANOS_SOURCE_REVISION must be exact lowercase 40-hex")
        return explicit
    recorded = ROOT / "_source-revision.txt"
    if recorded.is_file():
        value = recorded.read_text(encoding="ascii").strip()
        if HEX_40.fullmatch(value) is None:
            raise RuntimeError("_source-revision.txt must contain exact lowercase 40-hex")
        return value
    git_dir = ROOT / ".git-real"
    if not git_dir.is_dir():
        git_dir = ROOT / ".git"
    if not git_dir.exists():
        raise RuntimeError(
            "Building SANOS requires a Git checkout, source sdist, or "
            "SANOS_SOURCE_REVISION"
        )
    status = subprocess.run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={ROOT}", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "Building a release distribution requires a clean checkout or an explicit "
            "SANOS_SOURCE_REVISION"
        )
    revision = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if HEX_40.fullmatch(revision) is None:
        raise RuntimeError("Git did not return an exact lowercase 40-hex revision")
    return revision


class SanosBuildPy(build_py):
    """Copy release resources into the wheel without duplicating source files."""

    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "sanos" / "resources"
        destination.mkdir(parents=True, exist_ok=True)
        for source, name in RUNTIME_RESOURCES.items():
            if not source.is_file():
                raise RuntimeError(f"Required SANOS runtime resource is missing: {source}")
            shutil.copy2(source, destination / name)
        (destination / "source-revision.txt").write_text(
            source_revision() + "\n", encoding="ascii"
        )


class SanosSdist(sdist):
    """Embed revision provenance so wheels built from the sdist remain exact."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        revision = source_revision()
        super().make_release_tree(base_dir, files)
        Path(base_dir, "_source-revision.txt").write_text(revision + "\n", encoding="ascii")


setup(cmdclass={"build_py": SanosBuildPy, "sdist": SanosSdist})
