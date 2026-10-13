"""Check repository-local Markdown links without network access."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
SCHEMES = ("http://", "https://", "mailto:", "data:")


def markdown_files(root: Path) -> tuple[Path, ...]:
    selected = [root / name for name in ("AGENTS.md", "PORTING.md", "PROGRAM_PLAN.md", "README.md")]
    selected.extend((root / "configs").rglob("*.md"))
    selected.extend((root / "docs").rglob("*.md"))
    selected.extend((root / "reviews").rglob("*.md"))
    return tuple(sorted(path for path in selected if path.is_file()))


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for source in markdown_files(root):
        text = source.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            raw = match.group(1).strip()
            target = raw.split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith("#") or target.startswith(SCHEMES):
                continue
            path_text = unquote(target.split("#", 1)[0])
            destination = (source.parent / path_text).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                failures.append(f"{source.relative_to(root)}: link escapes repository: {raw}")
                continue
            if not destination.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{source.relative_to(root)}:{line}: missing local target: {raw}"
                )
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = check(root)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"checked local links in {len(markdown_files(root))} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
