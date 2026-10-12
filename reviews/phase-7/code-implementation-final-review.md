# Phase 7 final code implementation review

**Verdict: CHANGES REQUIRED**

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the current uncommitted
Phase 7 worktree  
Network/WRDS access: none (`WRDS_OFF=1`)

## Blocking finding

### High — NaN values bypass the Dataset's numerical integrity checks

`_max_abs` returns `NaN` if either input contains one
(`src/sanos/datasets/internal.py:245-249`). Callers then use comparisons such as
`residual > tolerance`; IEEE `NaN` makes every such comparison false. The smooth-fit
summary checks at `src/sanos/datasets/internal.py:1385-1395` have the same issue:
`abs(NaN - expected) > tolerance` is false.

An independent offline probe built the reviewed two-expiry fitted fixture, replaced
each named column by `NaN`, and called `validate_fitted_partition`. All three were
accepted:

```text
pure_bid_delta ACCEPTED
pure_fitX_gamma ACCEPTED
smooth_pcnt_fit ACCEPTED
```

This defeats the claimed immutable numerical validation for source Greeks, fitted
Greeks, and daily fit diagnostics. The same comparison pattern is used by several
other residual checks, so fixing only these three columns is insufficient.

Require finite values for every numerical scalar/vector wherever its schema presence
condition says it is present, before computing residuals. `_max_abs` should also fail
closed on non-finite inputs (or return positive infinity), and every summary comparison
must reject non-finite observed and recomputed values explicitly. Add parameterized
NaN and infinity mutation tests covering enrichment Black vectors, fitted Black
vectors, smooth summaries, prices, unit bridges, state arrays, and residual evidence.

## Remaining acceptance-coverage gaps

These are smaller than the implementation defect above but still need closing before
the Phase 7 gate can claim its full acceptance matrix:

1. The row-multiplicity assertion at `tests/test_widb_build.py:983-988` selects
   `(trading_date, expiry_id, time_to_expiry_years)`. Those values are naturally
   repeated for every strike and therefore do not test the promised equal ordering key
   `(expiry_id, normalized_strike)`. Construct a valid fitted fixture with an actual
   equal strike key, publish it, and assert exact byte/row/order/multiplicity round
   trip—or narrow the documented contract if such keys are invalid at the fitted-day
   boundary.
2. The re-signed matrix now covers catalog date-role mapping, stage ancestry, fitted
   manifest checksum, and flattened fitted identity. It still lacks the explicitly
   requested re-signed mutations for `metadata.raw_artifacts` and
   `fitted_source_sha256`. The implementation appears to reject both, but the tests
   should prove those exact relations.

## Remediation confirmed

- The validator's explicit `@version` closure now owns its schema, parsing, evidence,
  hashing, policy, canonicalization, and invariant dependencies. Fresh-process daily
  and Dataset closures resolve deterministically.
- The complete OptionsFilter projection is recomputed from `NO_ARB_SCHEMA`; retained
  expiry supports bind both no-arbitrage masks to exact fitted support geometry; the
  prior self-consistent mask mutations are rejected.
- Stored enrichment and fitted Black IV/Greek vectors are recomputed with the
  version-owned production kernels and cast boundaries. Ordinary finite corruption
  tests are present; the blocking gap is their non-finite failure mode.
- Two simultaneous identical Dataset publishers converge on one validated immutable
  artifact and leave no staging directories. Mid-copy interruption exposes no partial
  Dataset and restart succeeds.
- The re-signed evidence matrix rejects missing/duplicate/reordered dates, altered
  schema and validation reports, false stage ancestry, fitted-manifest evidence,
  flattened fitted identity, and Raw-catalog date-role changes.
- Full-range planning is frozen at 918 included sessions, 911 excluded-half-day
  sessions, the exact seven half days, one policy-owned typed-empty date, and 3,674
  unique Raw identities.
- Output-first shallow reuse and `version="*"` semantics remain correct and do not
  require Raw, Intermediate, or catalog artifacts after publication. Deep validation
  deliberately traverses and reconciles retained lineage.
- Published partitions are byte-copied, checksummed, catalog ordered, and loaded with
  Hive inference disabled. Dataset publication remains atomic and immutable.
- The CLI remains offline for Dataset build/validate, exposes human and JSON results,
  reports progress, supports dry-run/exact identity/wildcard selection, and does not
  initialize WRDS.

## Canonical-reference semantics

The wording is now consistent with the governing rule. A generic Phase 7 publication
is an unpromoted internal candidate with `canonical_promotion_ready=false`. The fresh
SANOS publication becomes the authoritative reference only after the production
validation/comparison, exact-artifact code and quantitative reviews, discrepancy
resolution, and explicit promotion. The old 891-date `quant_data` artifact is only a
read-only migration/parity oracle; it is neither a build input, runtime dependency,
nor co-equal reference.

This review does not approve a live production Dataset identity, manifest digest, or
canonical promotion.

## Commands and evidence

```text
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/test_widb_build.py::test_reference_profile_freezes_full_calendar_and_policy_structure \
  tests/test_widb_build.py::test_internal_dataset_validator_version_owns_all_semantic_helpers \
  tests/test_widb_build.py::test_internal_dataset_concurrent_identical_publishers_converge \
  tests/test_widb_build.py::test_internal_dataset_rejects_resigned_catalog_lineage_and_evidence_mutations \
  tests/test_widb_surface.py::test_dataset_quantitative_boundary_rejects_state_config_and_clock_corruption \
  tests/test_widb_surface.py::test_natural_zero_selected_day_passes_dataset_boundary
# 6 passed in 22.89s

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m pytest -q -p no:cacheprovider
# 187 passed in 121.21s

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 ruff check src tests scripts
# All checks passed!

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m compileall -q src tests scripts
# passed

git --git-dir=.git-real --work-tree=. diff --check
# passed

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m mypy src tests scripts
# unavailable in the active interpreter: No module named mypy
```

The NaN probe used only the stored synthetic Phase 6 fixture in memory. No network,
WRDS access, live credentials, or `quant_data` mutation occurred.

Signed: `code` — independent Phase 7 code reviewer
