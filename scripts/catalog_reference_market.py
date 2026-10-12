#!/usr/bin/env python3
"""Build a deterministic, read-only catalog of the legacy market reference."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from sanos.reference import catalog_legacy_market


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="legacy SPX HDF5 directory")
    parser.add_argument("output", type=Path, help="catalog JSON destination")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="emit progress after this many files (default: 50)",
    )
    return parser


def _validated_catalog_paths(root: Path, output: Path) -> tuple[Path, Path]:
    try:
        resolved_root = root.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Cannot resolve reference root {root}: {error}") from error
    if not resolved_root.is_dir():
        raise ValueError(f"Reference root is not a directory: {resolved_root}")
    try:
        resolved_output = output.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot resolve catalog output {output}: {error}") from error
    if resolved_output == resolved_root or resolved_output.is_relative_to(resolved_root):
        raise ValueError(
            "Catalog output must be outside the read-only reference root: "
            f"output={resolved_output}, root={resolved_root}"
        )
    return resolved_root, output.expanduser()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    try:
        reference_root, output = _validated_catalog_paths(args.root, args.output)
    except ValueError as error:
        parser.error(str(error))

    started = time.monotonic()

    def progress(completed: int, total: int, path: Path) -> None:
        if completed % args.progress_every and completed != total:
            return
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        remaining = (total - completed) / rate if rate else float("nan")
        print(
            f"catalogued {completed}/{total} files; elapsed={elapsed:.1f}s; "
            f"eta={remaining:.1f}s; current={path.name}",
            file=sys.stderr,
            flush=True,
        )

    catalog = catalog_legacy_market(
        reference_root,
        source_revision=args.source_revision,
        relative_path=args.relative_path,
        progress=progress,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(catalog.to_json())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    print(
        f"wrote {output}: files={len(catalog.records)}; "
        f"populated={len(catalog.populated_records)}; "
        f"sha256={catalog.collection_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
