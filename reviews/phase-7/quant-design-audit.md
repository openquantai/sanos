# Phase 7 quantitative/data design audit

**Status: DESIGN PASS; IMPLEMENTATION AND CANONICAL PROMOTION REMAIN GATED**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Reviewed SANOS revision: `675d171685954089b49c566459eb570b019c9c8d`  
Legacy migration oracle revision: `de15be96d2f283b6c31e60d3ea47325a0c093ff0`

This is a design review only. I did not access WRDS or the network, modify the
read-only migration implementation, or edit implementation code. The legacy
891-date collection is comparison evidence, not a build input or a second
canonical dataset.

## 1. Dataset population and physical layout

The internal dataset must copy each completed `fitted-day` frame losslessly. It
must not re-run a fit, select only eligible rows, deduplicate Cartesian option
pairs, or add padded arrays. The logical row remains one paired call/put strike
book from `FITTED_SCHEMA`; side populations are represented by `call_active`
and `put_active`, not by interpreting every row as two active books.

The preferred physical contract is one immutable Parquet file per planned
session:

```text
data/ticker=SPX/year=YYYY/date=YYYY-MM-DD/data.parquet
```

Each file retains the exact `FITTED_SCHEMA`, dtypes and input row order. The
assembler should byte-copy a validated daily `data.parquet` when possible.
Partition directory values are indexing metadata, not replacement columns;
the loader must read files in the checked partition-catalog order with Hive
column inference disabled, then concatenate without reordering rows.

The daily order inherited from Phase 5/6 is nondecreasing
`(expiry_id, normalized_strike)`. Equal keys are legal under CD-017 and retain
their stable source order. Publication must reject an out-of-order partition,
but must not impose uniqueness or choose a tie-break after the fact. The exact
legacy-selected view is `options_filter_eligible`, stably ordered by expiry and
normalized strike as implemented by `selected_reference_view`.

Publish all audit rows, including filter and no-arbitrage rejections. At least
the following populations must be counted separately per day and in aggregate:

- frame rows and distinct input expiries;
- initial retained expiries and `mask_arb_free` rows;
- `options_filter_eligible` rows and selected expiries;
- smooth retained expiries, `smooth_fit_mask` and `smooth_model_mask` rows;
- active calls, active puts and their sum as active side books;
- each ordered first-rejection reason at every stage; and
- active side books on the `0.2`-business-day AM node.

These names prevent the full audit-row count, selected strike-node count and
active side-book count from being conflated.

## 2. Date coverage, half-days, empty markers and exclusions

The canonical configuration must contain ticker, inclusive start/end dates,
`include_half_days`, source generation, daily and surface configuration, and
the exact planned date sequence. For 2022-01-03 through 2025-08-29 the current
NYSE calendar yields:

- 918 sessions when half-days are included;
- 911 sessions when they are excluded; and
- seven half-days: `2022-11-25`, `2023-07-03`, `2023-11-24`, `2024-07-03`,
  `2024-11-29`, `2024-12-24`, and `2025-07-03`.

The implemented general profile defaults to including half-days. Therefore a
fresh default build has 918 expected daily partitions; the known-bad policy
turns `2025-01-08` into one exact-schema typed-empty partition, so the current
policy expects exactly 917 populated days and one typed-empty day. An
`include_half_days=false` compatibility profile instead has 911 expected
partitions, 910 populated days and the same typed-empty day. Any other result
blocks publication unless a separately reviewed policy/configuration explains
it. The chosen half-day value is content identity. `is_half_day` remains a
trading-session field and must never be
derived from or merged with `expiry_is_am`.

There are only three permitted date outcomes:

1. **populated**: one exact-schema, nonempty fitted partition;
2. **known-bad typed empty**: one exact-schema empty Parquet partition, with an
   exact `bad_day_reason`, bad-policy version and zero counts; or
3. **configured schedule exclusion**: no partition, but an exact date and
   reason in the exclusion report (currently only excluded half-days).

Every planned date must have exactly one of the first two outcomes. An unknown
empty, absent partition, failed solver, or failed daily worker blocks
publication. A day whose full frame is nonempty but whose selected population
is zero is not a typed-empty day: it remains a populated audit partition and
must report zero selected rows and the stage reasons.

The legacy oracle has 892 physical files: 891 populated and the sole
`2025-01-08` empty marker. It omits all seven half-days and these 19 other NYSE
sessions:

```text
2022-03-22  2022-08-11  2022-10-28
2023-05-17  2023-06-02  2023-07-06  2023-10-02  2023-10-06  2023-11-02
2024-01-10  2024-01-26  2024-02-13  2024-02-28  2024-03-04
2024-07-11  2024-07-18  2024-07-24  2024-08-01
2025-05-16
```

The seven half-day differences are a declared configuration difference for the
fresh default profile. The other 19 are not an exclusion policy. A fresh
success on one of them is recorded as an oracle-only gap and reviewed; a fresh
failure requires stage evidence. Neither outcome may be hidden by making this
list a bad-day registry.

## 3. Assembly and lineage contract

Before a first publication, the assembler must:

1. calculate the expected date sequence independently from the supplied daily
   pointers and reject missing, duplicate, extra, wrong-ticker or wrong-date
   inputs before `ArtifactSpec` sorting can conceal them;
2. require a concrete producer version; `version="*"` may reuse a complete
   matching Dataset but may neither assemble nor publish one;
3. shallow-validate every fitted-day artifact and semantically validate its
   Parquet frame, then deep-validate all daily lineages while upstreams still
   exist;
4. confirm that each daily manifest's configuration IDs, schema version,
   producer version, source generation and ticker/date agree with the dataset
   configuration;
5. build a date-keyed partition catalog rather than infer date order or role
   multiplicity from an `ArtifactSpec.upstream` tuple, which is deliberately
   sorted/deduplicated;
6. copy every daily payload into private staging, validate the copied Dataset
   from its own files, and publish the completion marker last; and
7. fail rather than publish if any quantitative, population, comparison or
   documentation report is incomplete.

The Dataset's **sole direct upstream** should be one versioned
`raw-lineage-catalog` Intermediate, not the disposable fitted-day cache set.
That catalog's upstream tuple is the exact deduplicated Raw closure. Its
canonical configuration also includes a digest of the sorted per-date,
per-role mapping (security, options, current/previous spot, forwards and zero
curve), so tuple deduplication cannot erase boundary roles or multiplicity.
The Dataset identity then commits to Raw evidence, full Dataset configuration,
schema and the complete daily producer-version closure without coupling the
permanent boundary to the current Intermediate topology.

The Dataset payload and metadata must still record every fitted-day pointer,
source Parquet checksum and paired/enriched/no-arb/filtered chain used for the
actual copy. Before publication, the publisher proves that those chains equal
the planned producer/configuration identities and close to exactly the Raw
catalog, with no missing or extra Raw pointer. The custom deep-lineage
validator traverses both the catalog-to-Raw lineage and these recorded fitted
chains while they exist. Thus fitted pointers remain mandatory assembly
evidence, but are not direct Dataset identity inputs.

After publication, default loading and validation are shallow and use only the
Dataset payload. Deleting a *copied test tree* of Raw and Intermediate artifacts
must not change loading, reconstruction or shallow validation. Explicit deep
validation is expected to fail when that lineage has been removed. No command
that loads, validates or resolves a known Dataset identity may rebuild it when
the directory is absent. Wildcard reuse retains all ordinary schema,
configuration, completion and checksum checks.

## 4. Schema and quantitative invariants

Every populated partition must have exact `FITTED_SCHEMA` names, order and
dtypes. Daily fields are constant; expiry fields are constant within
`expiry_id`; ticker/date match the partition catalog; all three stage-specific
configuration IDs are constant and equal the hash of their appropriate
canonical configuration projection; they are not required to equal one
another or the full Dataset configuration ID. Typed empty partitions must have
the same schema exactly.

The Dataset validator must recompute, not merely trust diagnostic JSON:

- ordered cumulative filter masks and
  `options_filter_eligible == filter_cap_eligible`;
- `smooth_model_mask == options_filter_eligible & mask_arb_free &
  smooth_fit_mask`;
- call/put activity from selection, the initial mask and the valid direct side
  book, without substituting the smooth model mask;
- `smooth_mid_available` from selected finite call/put mids and
  `smooth_put_mid = smooth_call_mid - 1 + normalized_strike` exactly;
- pure and direct-book units (`K/F`, `C/(DF*F)`) and positive finite spot,
  forward and discount bridges;
- nullable-state/index alignment, strictly increasing nonnegative retained
  input indices, positive strictly increasing expiry times and support grids;
- finite density and variance state, density minimum at least `-1e-7`, mass and
  first moment within `1e-4`, finite nonnegative DLV interiors and exact zero
  DLV boundaries;
- normalized call bounds, decreasing/convex strike shape, calendar monotonicity
  within `1e-7`, and masked quote containment within `1e-6`; and
- DLV decoding back to stored densities and off-grid calls with absolute error
  at most `1e-12` on the corrected Phase 6 implementation (the implementation
  regressions currently achieve about `2e-14`).

Model state list columns are repeated on the rows of an expiry. The validator
must prove that repeats are identical before taking one representative value;
it must not silently select the first inconsistent copy. Diagnostic JSON must
parse canonically, agree with recomputed counts, and contain a terminal
accepted solve for every successful required fit. Intermediate fallback errors
are allowed only when followed by an accepted terminal solve. A terminal
failure blocks the daily artifact and therefore publication.

The validator must reconstruct both `compatibility_dlv_pricer` and
`smooth_sanos_pricer` from each nonempty day. This proves that the complete
pre-filter unsmoothed DLV and post-filter smooth SANOS state survived Dataset
assembly; a few fitted price columns alone are insufficient.

## 5. Required immutable evidence

The published directory must contain, in addition to Parquet and the store's
completion record:

- `README.md`: source tables, coverage, units, masks, exclusions, known-bad
  handling, migration-oracle role and exact build/validate/load commands;
- `config.json`: complete canonical content configuration, with no local paths
  or credentials;
- `schema.json`: exact fields, dtypes, nullability, cardinality level and units;
- `partitions.json`: one sorted record per expected session, including path,
  fitted-day pointer and checksum, row/expiry/mask/active counts, half-day flag,
  typed-empty status and reason;
- `lineage.json`: the canonical flattened stage/Raw identity closure and its
  digest;
- `validation.json`: validator/producer versions, aggregate counts, invariant
  maxima, failure count (zero), exclusion classes and report outcome; and
- `legacy-comparison.json`: oracle identities/checksums, overlap and date-gap
  classification, per-field tolerances, aggregate and per-day discrepancies,
  and the first divergent stage where evidence permits it.

The artifact manifest remains the authority for every payload byte checksum.
Its immutable `distribution` is exactly `internal`. Manifest metadata must
also expose the Git revision, dataset/schema/producer versions, configuration
ID, planned/partition/populated/empty/excluded counts, full-frame and selected
row counts, active side-book counts, upstream-catalog digests and the outcome
and checksum of each evidence report. `README.md`, `config.json`, the reports
and every Parquet file must therefore appear in the manifest file list.

Publication requires a clean, identifiable Git revision and complete resolved
`@version` closure for the assembler, schema, validator, invariant checks,
catalog/report serializers and loader. Runtime, machine paths and credentials
are not content metadata.

## 6. Migration-oracle comparison contract

The comparison report must keep these legacy populations distinct:

| Oracle | Dates/population | Phase 7 meaning |
|---|---:|---|
| smooth daily cache | 891 populated, 26,732 expiries, 2,414,341 strike nodes, 4,651,416 active side books, including 8,090 at 0.2D | direct daily smooth-market comparison |
| daily-front DySANOS | 891 dates, 3,566,258 books | downstream adapter check only |
| Daily25 SDLV | 891 dates, 2,192,458 books | separate Experiment oracle, not an internal-Dataset count |

The full internal frame row count is not expected to equal 2,414,341 because
the new frame deliberately retains audit rows. Compare the legacy strike-node
count to the exact selected/smooth-market projection and active-book counts to
`sum(call_active) + sum(put_active)`.

For dates present in both datasets, compare exact date/expiry/settlement order,
geometry, masks and populations. Integer, Boolean, string and index fields are
exact. Persisted Float32 geometry and quote-aligned outputs are bitwise exact
when source values are identical. Float64 supports, densities, variances,
prices and diagnostics use `rtol=0`, `atol=1e-12`; reconstructed calls on the
canonical comparison grid use `rtol=0`, `atol=5e-8` to cover the documented
Float32-query boundary. Validity tolerances (`1e-7` density/calendar,
`1e-6` quote containment, `1e-4` moments) are not parity tolerances and cannot
be used to excuse an oracle mismatch.

The original environment that produced the HDF5 oracle is unknown. Therefore
the report must always record maximum absolute/relative/ULP errors and mismatch
counts, even below tolerance. A mask, date, order or population difference is
never accepted numerically. A suspected vendor revision must be demonstrated
from fresh Raw identities/values and the first divergent daily stage; it cannot
be declared merely because a final fit differs. Any remaining unexplained
difference blocks canonical promotion and requires the prescribed reviews and
user approval.

The legacy HDF5 cache does not provide the newly named complete pre-filter DLV
state. Full-history initial-DLV correctness therefore comes from the fresh
lineage, Phase 6 pinned source probes and the invariant/reconstruction report;
the Phase 7 comparison must not claim a nonexistent persisted DLV oracle.

## 7. Canonical-reference promotion

Publication and canonical designation are separate operations:

1. build and atomically publish one immutable internal Dataset with a concrete
   identity;
2. run shallow validation from the Dataset alone and the full quantitative
   validation/comparison while lineage is available;
3. obtain the required code-agent review of the exact implementation/revision
   and quantagent review of the exact Dataset identity, manifest SHA-256 and
   evidence reports;
4. resolve every unexplained discrepancy or record explicit user approval for
   the documented difference; and
5. only then atomically update an external canonical-reference registry to the
   immutable Dataset pointer and manifest digest.

The registry is an index, not a fifth storage class, Dataset payload or mutable
manifest field. Promotion must not modify the published artifact, accept
`version="*"`, or use a path alias in place of an artifact identity. A later
canonical replacement is another explicit reviewed promotion. A valid but
unpromoted internal Dataset remains usable by explicit identity.

The actual fresh 2022--2025 promotion necessarily occurs only after guarded
WRDS acquisition and the end-to-end reproduction phase. This offline design
review is not approval of an as-yet-unbuilt Dataset and does not authorize WRDS
access.

## 8. Minimum tests and phase acceptance

Phase 7 implementation is ready for final dual review only after offline tests
prove:

1. lossless assembly of populated, half-day, zero-selected and known-bad typed
   empty partitions, including exact schema/dtype/order and duplicate-key
   preservation;
2. rejection of missing/extra/duplicate dates, unknown empties, wrong
   ticker/date/config/schema, corrupt checksums, inconsistent repeated state,
   malformed diagnostics and each numerical invariant failure;
3. exact recomputation of daily and aggregate population counts and all legacy
   oracle constants/date-gap classes above;
4. interruption leaves no visible Dataset, restart succeeds, and a concurrent
   publisher either wins atomically or reuses the validated winner;
5. a published Dataset loads and shallow-validates after copied Raw,
   fitted-day/other Intermediate, and raw-lineage-catalog trees are removed,
   while explicit deep validation then reports missing lineage;
6. wildcard reuse emits its warning and reuses only a complete matching
   Dataset; loading/validating a missing published identity never invokes a
   builder;
7. tampering with any Parquet, report, README, config, manifest or completion
   record is detected;
8. canonical promotion rejects a wrong artifact/review digest, unsigned or
   failed review, unresolved comparison, dirty/unidentified producer revision,
   wildcard identity and incomplete validation;
9. `WRDS_OFF=1` throughout; Dataset load/validate/promotion performs no source
   access; and
10. code-agent and quantagent final reviews each name the same artifact ID and
    manifest digest. Neither review may be inferred from this design PASS.

No new quantitative semantic choice is needed to implement this contract. The
19 unexplained oracle gaps, any fresh vendor revisions, and every open CD item
remain evidence/promotion gates rather than permission to alter daily semantics
or hard-code exclusions.

## Offline evidence run

```text
git --git-dir=.git-real --work-tree=. rev-parse HEAD
# 675d171685954089b49c566459eb570b019c9c8d

NYSE/calendar versus checked legacy catalog:
# include-half-days sessions=918; exclude-half-days sessions=911
# half-days=7; legacy physical files=892; populated=891
# missing NYSE sessions=26: 7 half-days + 19 other dates

WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_reference_inventory.py tests/test_reference_legacy_market.py \
  tests/test_widb_surface.py tests/test_artifact_store.py
# 50 passed in 12.58s
```

Signed: `quantagent` — independent quantitative/data reviewer
