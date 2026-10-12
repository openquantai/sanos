# Phase 7 code implementation re-review

**Verdict: CHANGES REQUIRED**

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the current uncommitted
Phase 7 worktree  
Network/WRDS access: none (`WRDS_OFF=1`)

## Remaining blocking findings

### 1. High — the Dataset validator's resolved version omits content-affecting helpers

`validate_internal_dataset` directly applies the immutable schema renderer, JSON and
pointer parsers, partition-statistics parser, migration-comparison renderer, file
hasher, bad-data policy, and canonicalization helpers throughout
`src/sanos/datasets/internal.py:2190-2667`. Its `@version` declaration at
`src/sanos/datasets/internal.py:2180-2188` declares only
`open_internal_dataset`, `validate_fitted_partition`, `aggregate_statistics`,
`_scheduled_exclusions`, and `_safe_partition_ticker_label`.

This is material because `validation.json` persists
`str(validate_internal_dataset.version)` and the Dataset producer declares the
validator as part of its content closure. A change to `_schema_document`,
`_pointer_from_value`, `_read_json_file`, `_partition_record_statistics`,
`_legacy_comparison_document`, `_sha256_file`, or their relevant policy/
canonicalization dependencies can change what is accepted or the evidence contract
without changing the recorded validator version or Dataset producer identity. That
violates the binding version-closure rule and the Phase 7 design gate.

Declare the complete content-affecting validator dependency graph (excluding only
pure progress/logging), then add an exact direct-dependency regression analogous to
the Phase 6 closure tests. The regression should demonstrate that changing each
schema/evidence/invariant dependency changes the resolved validator and Dataset
producer versions.

### 2. High — prior finding 8's Dataset-boundary acceptance matrix remains incomplete

The remediation added valuable fresh-process, caret-ticker, historical-wildcard,
mid-copy interruption, multi-outcome, typed-empty, zero-selected, diagnostics, clock,
and numerical-state tests. It did not add the remaining tests explicitly required by
the prior code finding and both Phase 7 design audits:

- no Dataset-level concurrent identical-publisher test exists (the only concurrent
  publication test is the generic ArtifactStore test at
  `tests/test_artifact_store.py:264`);
- no mutation matrix exercises Raw-catalog/date-role records, paired-to-fitted
  ancestry, flattened manifest Raw/fitted identities, or retained fitted
  manifest/source checksums under deep validation;
- no Dataset catalog tests reject missing, extra, duplicate, or reordered planned
  dates, and no publication test proves preservation of equal-key/duplicate-key row
  order;
- the full-range regression checks 918 included sessions but does not freeze the
  complementary 911-session no-half-day plan, seven scheduled exclusions, and the
  single policy-owned typed-empty date; and
- immutable-evidence tamper coverage remains limited to a partition-byte append and
  the deliberately rewritten historical `validation.json`, rather than the promised
  config/schema/partition/lineage/report/manifest relation matrix.

These are not merely optional test polish: concurrent-winner behavior, exact lineage,
catalog completeness, duplicate preservation, and immutable evidence are core Phase 7
publication acceptance criteria. Add focused, compact tests. They may use tiny
synthetic publications and intentionally re-signed test artifacts; no WRDS data is
needed.

## Prior finding disposition

1. **Resolved — deterministic versions.** The ambiguous daily/surface `_empty`
   helpers are now uniquely named, a fresh process resolves all daily and Dataset
   versions identically, and the former order-sensitive full-suite failures are gone.
2. **Resolved — ticker/path round trip.** The accepted OptionMetrics ticker grammar is
   aligned with Raw planning, partition paths use a collision-resistant safe label,
   and `^SPX` publishes, loads, and validates while retaining the vendor ticker.
3. **Resolved — manifest lineage/exclusions.** Manifest metadata now contains exact
   flattened Raw pointers, fitted pointers, scheduled exclusions, and typed-empty
   evidence. Deep validation reconciles catalog upstream/payload records with the
   daily Raw set and checks every recorded stage ancestry and retained-source digest.
4. **Resolved — canonical authority wording.** Phase 7 artifacts consistently remain
   unpromoted candidates with `canonical_promotion_ready=false`. The freshly rebuilt
   SANOS Dataset becomes authoritative only after guarded production comparison,
   exact-artifact code and quantitative reviews, discrepancy resolution, and explicit
   promotion. The legacy 891-date `quant_data` object is described only as a
   migration/parity oracle and never as a co-equal reference or runtime input.
5. **Resolved — repeated and quantitative state validation.** All three repeated
   diagnostics are checked for daily constancy, and the validator now recomputes the
   configuration/clock/mask/state/mapping/fit relationships covered by the earlier
   adversarial probe. The added mutation test rejects every formerly accepted case.
6. **Resolved — wildcard historical provenance.** A genuinely historical Dataset
   producer and validator version can be reused through `version="*"` while current
   schema, configuration, integrity, and substantive semantic checks still run.
7. **Resolved — progress.** Standalone validation, build reuse, post-publication
   validation, and deep traversal accept the common progress callback and emit
   bounded semantic/lineage progress plus terminal completion.
8. **Partially resolved.** The principal scenario tests were strengthened, but the
   publication/concurrency/catalog/lineage acceptance gaps in finding 2 above remain.

## Confirmed behavior and canonical-reference semantics

- Dataset lookup remains output-first and shallow by default; wildcard is reuse-only.
- A published Dataset is self-contained after Raw and Intermediate deletion; explicit
  deep validation then fails clearly when disposable lineage is absent.
- One complete fitted Polars frame is copied byte-for-byte per day. Dataset assembly
  does not add padded arrays, HDF5, Redun, or a fifth storage class.
- Publication retains one Raw-lineage-catalog direct upstream while the immutable
  evidence records every fitted chain and flattened Raw identity.
- No live Dataset identity, manifest digest, migration comparison, or promotion is
  approved by this code review. The legacy 891-date object remains only the migration
  oracle; authority transfers to the fresh SANOS Dataset only after the stated
  validation/review/promotion gate.

## Commands and evidence

```text
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -c 'from sanos.datasets.internal import internal_dataset_producer_version; \
             print(internal_dataset_producer_version())'
# passed in a clean process; resolved closure printed

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m pytest -q -p no:cacheprovider tests/test_widb_build.py tests/test_cli.py
# 37 passed in 73.91s

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m pytest -q -p no:cacheprovider
# 183 passed in 122.79s

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

All tests and inspection were offline. No WRDS credentials, network connection, or
`quant_data` mutation was attempted.

Signed: `code` — independent Phase 7 code reviewer
