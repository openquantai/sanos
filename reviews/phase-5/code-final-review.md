# Phase 5 final code and architecture review

**Verdict: PASS**

Reviewer: independent code agent (`/root/code`)
Date: 2026-10-12
Network/WRDS access: none

Phase 5 meets its code and architecture acceptance criteria and may be
committed. The settled implementation preserves the pinned CD-020 failure by
default; the documented repair remains unapproved and outside this verdict.

## Resolved finding

### C5-001 — The unapproved all-valid enrichment repair is no longer active

The pinned reference leaves its internal `is_valid` column in an enriched frame
when every row survives, and then fails schema validation. The standalone
implementation now preserves that observable failure boundary with a targeted
`DailyDataError` carrying the CD-020 identifier. It does not leak the temporary
control column into the Polars schema.

The behavior is pinned by
`test_all_valid_day_preserves_pinned_reference_failure_cd020`, and
`enrich_daily_frame` was bumped to version `1.2`. CD-020 remains open in
`docs/compatibility-decisions.md`; changing the failure into a successful frame
still requires explicit user approval, a further version change, and population
impact evidence from the fresh build. This Phase 5 verdict does not approve that
future repair.

No code-architecture blocker remains in the reviewed tree.

## Verified implementation properties

- The two stages remain one Polars frame per day. `paired-day` directly pins six
  immutable Raw pointers; `enriched-day` pins the paired artifact. Publication
  uses the existing staging-directory writer, durable Parquet write, upstream
  validation, atomic directory publication, immutable-race validation, and
  checksummed completion records.
- Terminal resolution is output-first and shallow. A valid terminal artifact
  is reused without opening or requiring Raw ancestors. Missing outputs build
  only through the declared lineage, and spawned workers return pointers rather
  than frames.
- `version="*"` is reuse-only. A miss fails before material execution; a hit
  preserves the actual `ReuseResult`, reports the stored producer version and
  warning in structured output, and returns the actual reused artifact path
  rather than the nonexistent current-version path.
- Actual vendor ticker text remains in canonical configuration and Raw
  identities, while a collision-resistant safe label is used for runner keys
  and Intermediate namespaces. This resolves the `^SPX` mismatch without
  changing vendor identity.
- The bounded spawn runner, parent-owned durable journal, restart probe, and
  actionable per-day worker failures are reused correctly. Dry-run remains
  non-mutating and reports effective configuration, expected outputs, previous
  spot dependencies, and optional verbose Raw paths. JSON output does not leak
  Python warning prose.
- Configuration parsing is strict at the CLI boundary; the overnight fraction
  is canonical, defaults to `0.2`, changes artifact identity, and reaches AM
  clocks while PM clocks remain unchanged.
- Daily schemas, exclusion registries, business-clock/NMS policies, Black
  adapters, numerical tolerances, transforms, I/O validators, and stage
  producers now have resolvable `cdxcore.version` closures. An independent
  enumeration of every versioned Phase 5 object found no unresolved closure.
- Schema/order/dtype checks, known-bad reason metadata, scalar/vector calendar
  behavior, exact Float32 spot-return/NMS boundaries, wildcard behavior,
  safe-ticker behavior, output-first reuse, CLI errors, and a frozen synthetic
  pinned-reference vector are covered offline.
- CD-003 and CD-016 through CD-019 remain explicitly registered rather than
  being silently redesigned. CD-004/public-return, no-arbitrage, SANOS/DLV,
  Dataset publication, and Experiment materialization remain outside Phase 5.
- `sanos status` now advertises the implemented daily Intermediate capability;
  Dataset and Experiment commands remain gated to later phases.

## Commands and results

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider tests/test_widb_daily.py tests/test_widb_build.py \
  tests/test_widb_reference_parity.py
# 26 passed in 14.56s

WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
# 145 passed in 28.70s

ruff check src tests scripts
# pass

python -m compileall -q src tests scripts
# pass

git --git-dir=.git-real --work-tree=. diff --check
# pass

# Imported and resolved every `.version` object owned by
# sanos.widb.{bad,build,calendar,config,daily}
# 57 versioned objects; no unresolved versions
```

`python -m mypy src tests scripts` could not be run because mypy is absent from
the active project interpreter. This is a tooling gap, not a claimed pass, and
should be run in the configured development/CI environment before release.

## Residual risks and nonblocking follow-ups

- The independent quantitative final re-review is PASS. Its pinned-reference
  and threshold/adversarial conclusions remain the quantitative authority; this
  code review independently verified execution and architecture.
- No live WRDS query was needed or made. The later fresh-data pilot remains
  separately gated on immediate 2FA readiness and the Phase 4 connection cap.
- Before any future CD-020 repair is approved, the fresh build must report
  whether any real all-valid dates exist. That measurement affects the repair's
  population impact but does not itself authorize the change.

Signed: `code agent /root/code`
