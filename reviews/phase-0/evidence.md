# Phase 0 evidence bundle

Status: implementation review candidate; the normal `.git` mount handoff is
tracked explicitly below.

## Source anchors and Git

- Paper source: `15e07d4ab0652f837883550d555f601836685db9`.
- Python reference: `de15be96d2f283b6c31e60d3ea47325a0c093ff0`.
- Bootstrap commit: `182cc5e` on `main`, descended from the complete paper
  history.
- A 55-path `quant_data` history filtered from the exact reviewed commit was
  merged as a second parent. Its filtered head is `ab9743926bd53e98f160e33223bc5bab8c1e92fe`;
  every selected HEAD blob was verified against the source commit.
- `git diff --cached --summary` detected every paper asset as a 99--100%
  history-preserving move into `tex/`; the only source edit is the documented
  case correction from `Comments` to `comments`.
- The execution environment mounts an empty read-only tmpfs at `.git`. The real
  repository database is staged as ignored `.git-real` and commands use
  `git --git-dir=.git-real`. It must become the ordinary `.git` after the
  protected mount is removed.

## Python verification

Commands were run from the repository root unless stated otherwise.

```text
python -m pytest -q -p no:cacheprovider
.....                                                                    [100%]
5 passed in 0.02s

ruff check --no-cache src tests
All checks passed!

mypy
Success: no issues found in 14 source files

PYTHONPYCACHEPREFIX=/tmp/sanos/phase0-verify/pycache \
  python -m compileall -q src tests
# exit 0
```

The current source was copied to `/tmp/sanos/phase0-verify/package-source-2`
and built there. `python -m build --no-isolation --skip-dependency-check`
successfully produced both an sdist and wheel. The dependency check was skipped
only because the host interpreter exposes setuptools' vendored `wheel` without
installed wheel metadata; the wheel was built successfully.

The wheel was installed with all base dependencies into a new environment at
`/tmp/sanos/phase0-verify/venv-4`. `pip check`, `sanos --log-format json
doctor`, and `python -m sanos --help` all exited zero. The doctor event reported
all required modules present and `network_accessed=false`.

## Paper verification

Before the documented case-only source adaptation, `diff -qr` against the
paper checkout at the reviewed commit exited zero. The preserved upstream PDF
has SHA-256
`ec1659de0bef34d3e79b299ad9ee0a20a2ace30a5b9064e0f6d2f187fc89b0c9`.

The host TeX installation lacked `bbm` and `fontawesome`. Their official CTAN
archives were downloaded under `/tmp/sanos/phase0-verify/ctan`, not into the
repository. With those packages on TeX's search path, this out-of-tree command
completed successfully:

```text
latexmk -g -pdf -interaction=nonstopmode -halt-on-error sanos.tex
Output written on sanos.pdf (30 pages, 4438838 bytes).
```

No generated PDF replaced the preserved `tex/sanos.pdf`.

## CLI contract sample

```json
{"event":"doctor","modules":{"cdxcore":true,"click":true,"cvxpy":true,"highspy":true,"numpy":true,"polars":true,"pyarrow":true},"network_accessed":false,"python":"3.12.3","sanos_version":"0.1.0","status":"ok"}
```

The command hierarchy exposes `doctor`, `status`, `raw`, `intermediate`,
`dataset`, and `experiment`. Material commands fail explicitly during the
bootstrap instead of opening WRDS or pretending to produce artifacts.

## Artifact and schema impact

Phase 0 creates no data artifact or schema. It adds only package, CLI,
documentation, paper, test, and CI files. All verification outputs are under
the project-specific `/tmp/sanos/` directory.

## Open risks and gates

1. The protected `.git` mount must be removed so `.git-real` can become the
   ordinary Git metadata location before release.
2. The paper repository has no explicit license. No project-wide license has
   been invented.
3. Live WRDS access has not occurred and is not part of Phase 0.
