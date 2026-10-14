# Phase 11 isolated-environment readiness evidence

Date: 2026-10-14
Baseline: `acd97ba235d43d4977d3341fefa497ea95742230`
Network/WRDS access: none (`WRDS_OFF=1`)

## Scope

An isolated Python 3.12 environment was created outside Git at
`/home/hans/quantdata_data/sanos/environments/phase11-acd97ba`. The shared
`/home/hans/vqdata` environment was not modified. The operator and reference
extras now include scikit-learn because the pinned legacy Flex consumer imports
it transitively; this does not change a SANOS or WIDB transformation.

The CLI JSONL regression now accepts structured warning records before the
terminal Dataset result while requiring every preceding record to be a warning
and exactly one terminal `dataset_result`.

## Offline checks

- `python -m pip check`: no broken requirements.
- `sanos doctor --strict-locations`: status `ok`, no network access, no
  filesystem mutation, and all four explicit lifecycle locations resolved.
- focused operator/parity suite: **109 passed**.
- complete repository suite: **429 passed** in 774.81 seconds.
- settled CLI regression after review remediation: **1 passed**.
- `ruff check --no-cache src tests scripts`: passed.
- `git diff --check`: passed.
- every periodic `gpuctl resources --json` result came from the daemon and all
  RAM, disk, and temporary-storage checks were `OK`.

The offline full-range Raw dry-run for SPX from 2022-01-03 through 2025-08-29
reported 3,676 planned and missing artifacts, zero reused artifacts, and the
three required 2021-12-31 boundary partitions. It reported
`network_accessed=false`. The proposed Phase 11 production root remained
nonexistent after doctor and dry-run.

## Gate retained

No production root was created and no WRDS connection was opened. Phase 11
remains blocked by the documented owner classification of the exact 2025-01-02
and 2025-01-03 pilot discrepancies. This readiness work neither creates that
approval nor authorizes redistribution.
