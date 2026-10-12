# Phase 7 code implementation audit

**Verdict: CHANGES REQUIRED**

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the uncommitted Phase 7 worktree  
Network/WRDS access: none (`WRDS_OFF=1`)

## Findings

### 1. High — the Dataset producer-version closure is call-order dependent and fails in a fresh process

`internal_dataset_producer_version` declares all daily producer closures directly at
`src/sanos/datasets/internal.py:969`, but resolving it before those closures have
already been resolved raises a `cdxcore.version` recursion error on the two distinct
helpers named `_empty`.

Reproduction:

```text
PYTHONPATH=src WRDS_OFF=1 python -c \
  'from sanos.datasets.internal import internal_dataset_producer_version; print(internal_dataset_producer_version())'
# RecursionError: ... recursive dependency on function '_empty'
```

The ordinary suite misses this because Dataset planning resolves the individual
daily-stage versions first. A resolved content identity cannot depend on import or
call order. Refactor the version owner so it resolves successfully and identically in
a fresh interpreter, and add an isolated-subprocess regression that calls this public
helper first. The regression should also prove that changing each daily-stage version
changes the Dataset producer identity.

### 2. High — a valid OptionMetrics ticker can be atomically published with a path the Dataset rejects

`InternalDatasetConfig` accepts any nonempty ticker of at most 32 characters
(`src/sanos/datasets/config.py:39`), and the Raw/Intermediate contracts explicitly
support tickers such as `^SPX`. `_partition_relative_path` embeds the vendor ticker
unchanged (`src/sanos/datasets/internal.py:1113`), while `_PARTITION_PATH` excludes
`^` (`src/sanos/datasets/internal.py:69`). A direct probe produced:

```text
^SPX
data/ticker=^SPX/year=2024/date=2024-01-03/part-00000.parquet
validator_accepts_path False
```

Because post-publication validation happens after the completion marker is committed,
this can leave a complete but permanently unusable immutable Dataset at its expected
identity. Reuse the collision-resistant safe ticker-label contract already used by the
daily pipeline, retain the actual vendor ticker in configuration/frame data, and add an
end-to-end `^SPX` publication/load/validation regression. Configuration should also use
the same `RawPartition` ticker validation as the upstream layers.

### 3. High — `manifest.json` does not contain the required Raw provenance and exact exclusions

The binding Phase 7 contract requires the manifest itself to contain upstream Raw
identities and exact exclusions (`PROGRAM_PLAN.md:417`; `AGENTS.md:179`). The metadata
assembled at `src/sanos/datasets/internal.py:1812` contains only the Raw-catalog pointer,
mapping digest, and exclusion count. The flattened Raw pointers, exact fitted pointers,
and date/reason exclusion records exist only indirectly in checksummed payload files.
This also falls short of the agreed catalog contract in both Phase 7 design audits.

Keep the Raw-lineage catalog as the Dataset's sole direct `ArtifactSpec.upstream`, but
add the sorted flattened Raw identities, exact fitted-day identities, and exact
exclusions to manifest metadata. Validate those fields against `lineage.json`,
`partitions.json`, the catalog identity/digest, and recomputed summaries. Deep
validation must additionally prove that the catalog's actual Raw upstream set equals
the flattened daily-chain set, rather than merely validating both independently.

### 4. High — an unpromoted generic publication claims to be the reference Dataset

The safety gate itself is correct: both the comparison report and manifest metadata
record `canonical_promotion_ready: false`. However, the generated Dataset README says
the publication *is* the reference Dataset (`src/sanos/datasets/internal.py:1156`), and
`legacy-comparison.json` records `sanos_dataset_is_reference_after_publication: true`
while simultaneously saying the guarded comparison is pending
(`src/sanos/datasets/internal.py:1188`). Publication and canonical promotion are
separate operations under the quantitative design and Phase 11.

Change immutable payload wording/fields to say that this is a valid internal Dataset
and a canonical-reference candidate until the fresh comparison, exact-artifact dual
reviews, discrepancy approvals, and external promotion complete. Keeping
`canonical_promotion_ready=false` in this offline implementation is appropriate; the
legacy 891-date object must remain described only as a migration oracle.

### 5. High — daily semantic validation accepts inconsistent repeated diagnostics

`validate_fitted_partition` checks several repeated daily values for constancy at
`src/sanos/datasets/internal.py:429`, but omits `no_arb_diagnostics_json`,
`filter_diagnostics_json`, and `smooth_diagnostics_json`; it then parses only row zero
at line 621. An adversarial one-day fitted frame whose final
`smooth_diagnostics_json` was replaced by `{}` was accepted with all 13 rows.

Require all three diagnostic documents to be identical across the day before parsing.
Add mutation tests for each repeated diagnostic and the other promised repeated
expiry/model-state fields. Also enforce that every Raw role is `Raw`, every fitted
pointer is `Intermediate`, and, during deep validation, compare recorded source
Parquet/manifest checksums with the validated fitted artifacts rather than checking
only checksum syntax (`src/sanos/datasets/internal.py:1565`).

### 6. Medium — wildcard reuse is coupled to the current validator/report producer version

Semantic validation reconstructs an exact expected `validation.json` using the current
`str(validate_internal_dataset.version)` and current legacy-comparison renderer
(`src/sanos/datasets/internal.py:1595`). Consequently, a historical complete Dataset
found through `version="*"` can be rejected solely because the validator or report
renderer version changed, even when configuration, schema, lineage, payload checksums,
and all current quantitative invariants pass. That defeats the documented producer-
version override.

Treat stored validator/comparison versions as immutable provenance: validate their
types/schema and recompute the substantive summary/invariants with the current
validator, but do not require the stored producer string to equal the current one for
wildcard or exact-ID validation. Add a Dataset-level test with a genuinely different
stored producer/validator version; the existing wildcard test reuses the current
version and does not exercise this case.

### 7. Medium — standalone and reuse validation can be long-running without progress

The build's source-validation and copy loops now emit bounded progress, which is a good
improvement. `validate_internal_dataset` itself has no progress callback or interval
(`src/sanos/datasets/internal.py:1441`), so `dataset validate`, output-first build
reuse, and the final post-publication validation can process all 918 partitions and
reconstruct every surface without emitting a heartbeat (`src/sanos/cli/main.py:904`,
`src/sanos/datasets/internal.py:1989`). This violates the CLI's long-running progress
contract. Thread the same structured progress interface through semantic/deep
validation and cover human and JSON modes.

### 8. Medium — Phase 7 acceptance coverage is incomplete

The current end-to-end test is valuable but covers one ordinary synthetic day. Its
publication interruption is injected before the Dataset writer is entered, and there
is no Dataset-level concurrent-publisher test. The final gate still needs focused
tests for a published half-day, published known-bad typed empty, published natural
zero-selected day, equal-key order preservation, missing/extra/duplicate catalog
dates, mid-copy interruption and restart, concurrent winner validation, and semantic
tampering of every evidence document/metadata relation. Add a fresh-process version
test and the adversarial checks from findings 1–6 so the full-suite pass cannot mask
identity or validator defects.

## Confirmed strengths

- Dataset lookup is output-first and shallow by default. The existing integration test
  demonstrates loading and semantic validation after Raw and Intermediate trees are
  removed, while explicit deep validation then fails as intended.
- The Dataset has exactly one direct Raw-catalog upstream; fitted caches remain
  disposable assembly evidence. The catalog commits the deduplicated Raw closure and
  date/role mapping digest.
- Final files are staged through `ArtifactBuilder`, daily Parquet files are byte-copied,
  payload checksums are verified, and the completion marker is published last.
- `version="*"` is reuse-only at the store/build boundary, emits a visible warning, and
  cannot authorize a miss/build.
- Dataset build and validation remain offline and do not invoke WRDS. The CLI has
  stable text/JSON result records and exact-ID validation.
- The implementation keeps one complete Polars frame per day and does not introduce
  HDF5, padded arrays, Redun, or a fifth storage class.

## Commands and evidence

```text
PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
# 178 passed in 67.11s

PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider \
  tests/test_widb_build.py::test_internal_dataset_publication_is_self_contained_and_output_first
# 1 passed in 8.69s

PYTHONPATH=src WRDS_OFF=1 ruff check src tests scripts
# All checks passed!

PYTHONPATH=src WRDS_OFF=1 python -m compileall -q src tests scripts
# passed

git --git-dir=.git-real --work-tree=. diff --check
# passed

PYTHONPATH=src WRDS_OFF=1 python -m mypy src tests scripts
# unavailable in the active interpreter: No module named mypy
```

Additional read-only/adversarial probes confirmed the fresh-process version failure,
the `^SPX` path mismatch, and acceptance of row-inconsistent smooth diagnostics. No
network or WRDS access occurred. Temporary probe artifacts were confined to
`/tmp/sanos/phase7-code-review/`.

## Residual production gate

No fresh WRDS Dataset exists yet, so this review does not approve a Dataset identity,
manifest digest, numerical legacy comparison, or canonical promotion. A structurally
valid publication may remain explicitly unpromoted. The target 2022–2025 Dataset may
become authoritative only after the guarded fresh run, full comparison, exact-artifact
code and quantitative reviews, discrepancy resolution/approval, and explicit external
promotion required by the program plan.
