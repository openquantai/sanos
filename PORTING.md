# Migration provenance

This file records the source anchors used to create the standalone repository.
It does not grant permission to alter either reference repository.

## Paper repository

- Upstream: `git@github.com:hansbuehler/sanos.git`
- Reviewed/imported commit: `15e07d4ab0652f837883550d555f601836685db9`
- Commit subject: `Updates from Overleaf`
- Imported destination: `tex/`

All files tracked at that revision were first copied without content changes
beneath `tex/`; an exact tree comparison against the reviewed checkout passed.
The upstream history remains authoritative for the pre-migration paper. One
post-import portability adaptation changes `\input{Comments}` to
`\input{comments}` because the tracked filename is lowercase and Linux TeX is
case-sensitive. This does not change the paper's content.
The execution workspace exposes a protected empty `.git` mount at the requested
working directory, so this session cannot attach or write Git metadata there.
The complete upstream Git database and migration commits are therefore staged
inside this directory as `.git-real`, configured with the repository root as
its work tree. Until the protected mount is removed, commands use
`git --git-dir=.git-real ...`. Moving this database into the ordinary `.git`
location after that mount is removed remains an explicit Phase 0 handoff and
must be completed before a release.

## Python reference implementation

- Read-only source: `~/python/quant_data`
- Reviewed clean commit: `de15be96d2f283b6c31e60d3ea47325a0c093ff0`
- Relevant SANOS/WIDB paths were clean at the start of migration.

A filtered history containing 55 code, documentation, and offline-test paths
was imported beneath `reference/quant_data/` before refactoring. It was made
from a fresh local clone at the reviewed commit using `git-filter-repo 2.47.0`
with `--paths-from-file` and `--to-subdirectory-filter reference/quant_data`.
The resulting filtered head is
`ab9743926bd53e98f160e33223bc5bab8c1e92fe`; the reviewed source commit itself
became empty in the filtered history because it changed only unrelated files.
Every imported HEAD blob was compared with the corresponding blob at
`de15be96d2f283b6c31e60d3ea47325a0c093ff0` and matched exactly. The committed
path list is `docs/quant-data-import-paths.txt`.

The imported snapshot is not installed as part of the SANOS package and is not
a data-pipeline input. Phase 1 will map functions and tests before moving the
required implementations into `src/sanos/`; files found to be historical or
unexercised will remain provenance only and can be removed from the working
tree after their history is retained.

Existing `quant_data` raw, processing, training, and experiment artifacts are
not build inputs. The existing 891-date fitted dataset is a read-only parity
oracle only.

### Phase 3 numerical-core copy

The following files were copied byte-for-byte from the pinned snapshot before
imports, formatting, `@version` dependencies, standalone constants, and the
two explicit compatibility gates were adapted. The pre-adaptation copy was
verified against these source SHA-256 values:

| Source | SHA-256 | Standalone target |
|---|---|---|
| `quantdata/utils/backbone.py` | `8b61f5b6d3e36ac8b957436fbdf8f0ab71d655ab0212d26475d79836f875a535` | `src/sanos/model/backbone.py` |
| `quantdata/sanos/tridiag.py` | `97333b68e7f726b2e7273b360d1e8b4834f7c11d32212526ef98a75f8f90bf2b` | `src/sanos/model/tridiag.py` |
| `quantdata/sanos/sanos_tools.py` | `e250df0db749197f272c8807d2b5f342b07ae303a20096401e377d31da005b43` | `src/sanos/model/tools.py` |
| `quantdata/sanos/sanos.py` | `15e885d7af93485cbaec5ec11b417b4451dd2597cf9ffe8e8b859b11741792` | `src/sanos/model/fitter.py` |
| `quantdata/sanos/arbfree.py` | `d26dd8236160786dc880f30631217d1a8183503019b748e4a6f9bfa0c02f27ca` | `src/sanos/model/arbfree.py` |
| `quantdata/sanos/sanos_jax.py` | `ba9ec3f65295f4e4b4f1a59f77c4105de592fdd2df63b4c09141dd3dd39a485f` | `src/sanos/model/jax_fit.py` |

The authoritative CVXPY/HiGHS fitter is retained. During the copy, the optional
JAX fitter was found to append to `weights=None` and to evaluate the initial
state instead of its optimizer result. It is therefore an explicit blocked
stub under CD-015, while the shared JAX transition and gradient path is tested.
The already registered broken ndarray market-fit diagnostic is likewise an
explicit CD-013 gate. Neither gate changes the 891-day producer path.

Other adaptations are intended to be behavior-neutral: imports now use
`sanos.model`; the 252-day and one-minute constants live in
`sanos.model.constants`; files are Ruff-formatted with LF endings; deterministic
classes/functions and array helpers have resolved `@version` dependencies; and
dead local storage/deletions plus an overwritten duplicate NumPy
`stop_gradient` definition were removed. Solver settings, tolerances,
objectives, strike construction, interpolation, warning thresholds, and the
observable production interfaces were not redesigned.

The resolved public model versions close over the one-minute branch convention,
the density-floor policy where applicable, and the versioned market-fit
diagnostic (including its CD-013 gate). An isolated pinned-reference subprocess
provides the committed smoothness `0` and `0.25` LP parity oracle; standalone
code is not used to generate its expected state.

The dynamic array/CVXPY implementation is excluded only by exact module name
from strict mypy checking; it remains covered by reference-value, invariant,
solver, NumPy/JAX, and serialization tests. New model gates, artifact code,
workflow code, and future WIDB code remain under repository-wide strict mypy.

## Licensing inventory

The paper repository does not contain an explicit license at the reviewed
revision. The `quant_data` package metadata declares MIT, but copied source and
the paper must not be relicensed implicitly. Project-wide license metadata will
be added only after the owner confirms the intended treatment of code and paper.

## Phase 0 verification notes

- The preserved source tree was compared against paper commit
  `15e07d4ab0652f837883550d555f601836685db9` before the filename adaptation.
- The checked-in `tex/sanos.pdf` is the preserved upstream generated PDF.
- Building the paper requires the TeX Live `bbm` and `bbm-macros` packages;
  Debian and Ubuntu provide them through `texlive-fonts-extra`.
- Python build and paper-build scratch files belong below `/tmp/sanos/`, never
  in the repository.
