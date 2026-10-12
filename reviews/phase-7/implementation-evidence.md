# Phase 7 implementation evidence

Date: 2026-10-12  
Scope: internal WIDB/SANOS Dataset assembly, publication, loading and validation  
Network/WRDS access: none

## Implemented boundary

- Strict `InternalDatasetConfig` and operational `DatasetBuildOptions`.
- Shared one-date/all-date identity planning used by both daily builds and
  Dataset publication.
- Exact date/role Raw mapping plus one rebuildable Raw-lineage-catalog
  Intermediate whose upstream is the complete deduplicated Raw closure.
- One immutable internal Dataset with the Raw catalog as its sole direct
  upstream and the complete paired/enriched/no-arbitrage/filter/fitted code
  closure in its producer version.
- Lossless byte-copy of one complete fitted-day Parquet frame per planned
  session; no row filtering, deduplication, padding, or HDF5 conversion.
- Self-contained `config.json`, `schema.json`, `partitions.json`,
  `lineage.json`, `validation.json`, `legacy-comparison.json`, and `README.md`.
- Output-first current-version and wildcard reuse, shallow default validation,
  explicit deep-lineage validation, atomic publication, durable interruption
  journal, and exact-ID validation.
- Human/JSON CLI commands for `dataset build` and `dataset validate`.

The generated SANOS Dataset is designated as the reference Dataset after the
fresh production validation/review gate. The old 891-populated-date artifact is
recorded only as a migration/parity oracle and is not a build or runtime input.

## Quantitative checks

For every populated partition the validator recomputes exact schema and
ordering; daily/expiry constancy and state/index null alignment; cumulative
filter masks; selected/model/active masks; all configuration identities;
normalization bridges and put/call parity; diagnostic counts and terminal
solver states; density non-negativity, mass and first moment; squared-DLV
boundaries; strike/calendar no-arbitrage; masked quote containment; stored
fitted values; and corrected DLV density/off-grid-call round trips.

Typed-empty partitions are accepted only for an exact date/reason in the
versioned bad-data policy. A nonempty selected-zero day remains a populated
audit partition and is tested separately.

The checked-in reference profile deterministically plans 918 sessions from
2022-01-03 through 2025-08-29, including the seven NYSE half days. The policy
therefore expects 917 populated partitions and the typed-empty 2025-01-08
partition during the fresh run. These production counts have not yet been
claimed because no live WRDS access has occurred.

## Offline results

```text
ruff check src tests
All checks passed!

python -m pytest -q -p no:cacheprovider
188 passed in 146.13s
```

The suite includes clean-process producer-version resolution, a complete
fitted-state adversarial mutation matrix, a two-expiry fitted-state validation,
a natural zero-selected populated day, an OptionMetrics `^SPX` path round trip,
exact support-to-mask and Black-IV/Greek recomputation, concurrent identical
publishers, re-signed lineage/catalog/evidence mutation rejection, exact
row/order/multiplicity preservation for repeated partition keys, and a compact
multi-outcome publication spanning an ordinary session, a half day,
previous-session linkage, and the policy-owned typed-empty date. The full-range
plan is frozen at 918 inclusive sessions, 911 sessions when half days are
excluded, all seven expected half days, and the one policy-owned typed-empty
date. Every present Float32/Float64 scalar and float-list element is covered by
a fail-closed finite-value gate and an exhaustive NaN/positive-infinity/
negative-infinity mutation test. Exact `(expiry_id, normalized_strike)` keys
are required to be unique because the SANOS support geometry is strictly
increasing; exact publication equality is tested on valid repeated
session/expiry geometry.

Representative non-mutating reference plan:

```json
{"event":"dataset_result","planned_dates":918,"built":false,"reused":false,"dry_run":true,"network_accessed":false,"identity":"2ea20498735adf1e625b84f99c812bae78c90e9aeb81b36c9e0a330a21bd4585"}
```

The synthetic end-to-end integration test proves:

1. daily fitted output and the Raw catalog deep-validate before publication;
2. the published partition has byte identity with its fitted source;
3. semantic shallow and deep validation pass while lineage exists;
4. after deleting copied Raw and Intermediate trees, loading, scanning, shallow
   validation, current-version reuse and wildcard reuse still pass;
5. deep validation then fails because missing lineage is explicit;
6. an interruption after an actual partition copy exposes no partial Dataset
   and a rerun completes from existing daily/cache evidence;
7. a genuinely historical producer and validator are accepted only through
   explicit `version="*"` reuse;
8. partition tampering is rejected by the artifact checksum boundary;
9. two simultaneous identical Dataset publishers converge on the same complete
   artifact without leaving staging directories; and
10. even re-signed edits to partition coverage/order, validation or schema
    evidence, flattened metadata, stage ancestry, fitted checksums, and Raw
    date/role mappings are rejected by semantic or deep-lineage validation.

The implementation reviews requested changes in multiple rounds. Their
identity, ticker, manifest-lineage, pre-promotion wording, repeated diagnostics,
wildcard, progress, quantitative-state, clock, schema, multi-outcome,
non-finiteness and adversarial-evidence findings are addressed in the reviewed
worktree. The append-only final re-reviews record independent code and
quantitative/data PASS verdicts for this implementation commit.

## Deliberate gates

- No live WRDS pilot or production download was attempted. The user must confirm
  immediate 2FA readiness before Phase 10.
- `legacy-comparison.json` is structurally complete for generic publications but
  marks full fresh-run numerical comparison and canonical production promotion
  pending. Before the production run, the versioned comparison implementation
  must be completed so the final reference Dataset is published atomically with
  its full evidence under a new identity; an existing artifact is never edited.
- Public projection and experiment materialization remain Phase 8/9 work and
  are not exposed as options on the internal publisher.
