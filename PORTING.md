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
