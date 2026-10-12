# Phase 8 code design audit

**Reviewer:** independent code agent (`/root/code`)  
**Date:** 2026-10-12  
**Scope:** design only; no WRDS/network access and no implementation changes  
**Verdict:** **DESIGN READY, SEMANTIC IMPLEMENTATION BLOCKED** pending explicit
user approval of CD-004 and CD-009. The additional retained-date/typed-empty
rule below must also be made normative before the producer version is frozen.

## Sources reviewed

- `AGENTS.md`, especially the distribution, format, identity, output-first and
  CLI requirements;
- `PROGRAM_PLAN.md`, Phase 8 and its Phase 7/10 dependencies;
- `docs/data-contracts.md`, including the internal and provisional public
  contracts;
- `docs/compatibility-decisions.md`, specifically open CD-004 and CD-009;
- the Phase 7 internal Dataset implementation and reviews;
- `src/sanos/artifacts/store.py`, including immutable publication, shallow
  resolution and current distribution checks;
- `src/sanos/cli/main.py`; and
- the fitted-day schemas/reconstructors and pinned daily-front extraction path.

No live data or credentials were accessed. This audit does not approve either
open quantitative convention.

## Blocking decisions

1. **CD-004 remains a user decision.** Code must not publish a field named
   `forward_adjusted_spot_return` until the user approves the stated formula,
   exact choice of the forward curve, interpolation boundaries, and treatment
   of missing/typed-empty source dates. The internal `spot_return` remains the
   legacy simple close-to-close return and must never be relabelled.
2. **CD-009 remains a user decision.** Code must not make
   `discounted_forward_over_spot`, `half_spread_spot`, or the stated bridge
   normative until the user approves the exact units. The proposed bridge is
   dimensionally consistent, but review is not approval.
3. **The retained-date rule needs a normative statement before coding.** The
   current prose alternates between “one row per trading day” and “consecutive
   retained observations”, while the internal Dataset deliberately includes
   typed-empty known-bad partitions. The recommended contract is: public data
   contains exactly one row for each *populated retained* internal date,
   preserves their order, records omitted typed-empty/scheduled dates in
   manifest coverage evidence, and computes a return between consecutive
   published observations only when the approved previous-date curve spans the
   target; otherwise the return is null and the manifest records the gap count.
   There must be no silent extrapolation. If the intended behavior is instead
   one empty public row for every planned session, that is a different schema
   and identity and must be selected explicitly.
4. Actual transition from `public_candidate` to `public` additionally requires
   release-specific user/legal input: license, redistribution-rights basis,
   approver, approval date, durable approval reference, and the exact license
   text/checksum. Numeric/leakage validation cannot supply that authority.

Implementation may add tests and non-semantic scaffolding before items 1--3 are
resolved, but it must not publish a Phase 8 Dataset with provisional meanings.

## Recommended implementation boundary

Use a dedicated package rather than extending the already large internal
module:

```text
src/sanos/datasets/public/
    __init__.py       exported API and result types
    config.py         strict projection and approval documents
    schema.py         exact recursive Polars schema and schema document
    project.py        pure one-day projection and sequential return pass
    validate.py       self-contained semantic/leakage/capability validators
    publish.py        plans, output-first builds, approval/copy publication
```

`project.py` may import the Phase 6 pricer reconstruction APIs and versioned
calendar/NMS policy, but must not import WRDS code or resolve Raw/Intermediate
locations. `publish.py` is the only layer that opens the source internal
Dataset or the artifact store. Keep legal approval and byte-copy publication
out of the quantitative projection function.

Suggested public API:

```python
plan_public_candidate(config) -> PublicDatasetPlan
build_public_candidate(locations, config, options, *, dry_run, progress) \
    -> PublicDatasetBuildResult
open_public_dataset(store, pointer_or_validated) -> PublicDataset
validate_public_dataset(store, pointer_or_validated, *, semantic=True,
                        deep_lineage=False, progress=None) \
    -> PublicDatasetValidationSummary | None
plan_public_release(candidate_pointer, approval) -> PublicReleasePlan
approve_public_release(locations, candidate_pointer, approval, options, *,
                       dry_run, progress) -> PublicDatasetBuildResult
```

Planning must be pure and location independent. Opening/scanning a public
Dataset must consult only that Dataset's own files after the generic shallow
artifact check.

## Canonical configuration

Add versioned strict dataclasses, rejecting unknown fields and booleans passed
as numbers:

- `PublicProjectionConfig`
  - portable `name`;
  - exact source `ArtifactPointer`, restricted to `StorageClass.DATASET`;
  - immutable profile enum, initially `daily-front-v1`;
  - return convention enum and version (blocked by CD-004);
  - unit convention enum and version (blocked by CD-009);
  - inclusive NMS bounds `[-3.0, 2.0]`;
  - maximum business DTE `70.0`;
  - explicit excluded business DTE `[0.2]` and versioned comparison tolerance;
  - `european_only=true`, separate-side-book behavior, required finite smooth
    price behavior, and state scope; and
  - retained-date/gap policy once approved.
- `PublicBuildOptions` contains only operational values: progress interval and
  `version_override`. It must not enter content identity.
- `PublicApproval` contains the five required distribution fields plus a
  SHA-256 of an exact supplied license file. Machine paths, credentials and
  free-form command context are excluded from identity.

The projection config must not copy the internal config wholesale: its ticker,
source-generation details and paths are not public fields. Exact provenance is
the upstream Dataset pointer. All population-changing public selections are
present in canonical config even when they equal the default. `config.json`
must equal the manifest configuration canonically.

## Version-1 data schema

Use one physical `data.parquet` containing one Polars row per retained date.
The history is small, and a single frame best matches the user contract. Sort
rows strictly by `trading_date`. Use an exact, recursively checked Polars
schema; do not store opaque JSON or arbitrary maps in any data column.

Recommended top-level fields are:

```text
trading_date: Date
is_half_day: Boolean
forward_adjusted_spot_return: Float64 nullable       # CD-004
dlv_state: Struct
sanos_state: Struct
books: List(Struct)
```

The two half-day concepts remain separate: `is_half_day` is a trading-session
flag; each model expiry node has its own `expiry_is_am` settlement flag.

### `dlv_state`

This is the complete initial/pre-filter compatibility-DLV state, not only the
public book subset. It contains ordered expiry nodes with:

- public-local contiguous expiry index, expiry date, business/calendar DTE,
  `expiry_is_am`, model time in years and model time increment;
- per-expiry and surface `K/F` supports and probability masses;
- exact per-expiry-to-surface support index mapping;
- full-support annualized squared-DLV values in `1/year`, with the boundary
  encoding stated explicitly; and
- any input total variance needed to interpret the persisted state.

Pricing mode, one-minute intrinsic boundary, positive pre-first-expiry rule,
surface/expiry selection, interpolation rule, mass/first-moment tolerances and
density encoding are fixed in `schema.json` and canonical config rather than
hidden in a consumer. Do not serialize internal `expiry_id`.

### `sanos_state`

This is the complete smooth surface state for every smooth-retained expiry,
not only expiries with public books. It contains:

- `smoothness`, pricing/expiry-linear mode and ordered model time increments;
- public-local expiry geometry as above;
- separately named **smoothing-kernel input total variance** for each expiry;
- per-expiry and surface `K/F` supports, probability masses, and their exact
  index mapping.

Do not call the input variance fitted ATM IV. The annualized ATM curve used for
the `[-3,2]` NMS conversion must be reproducible as
`sqrt(interp(total_variance)/time)` from the stored inputs; if materialized for
audit, name it `kernel_input_atm_annual_volatility`, place it once per expiry,
and test exact derivability. Do not persist per-book model price or IV arrays.

### `books`

Each entry has exactly:

```text
sanos_expiry_index: UInt32
side: Enum("call", "put")
normalized_strike: Float64       # K/F
half_spread_spot: Float64        # (ask-bid)/(2S), pending CD-009
```

The referenced SANOS expiry node also carries the proposed positive finite
`discounted_forward_over_spot = DF*(F/S)` under CD-009. Books are ordered by
SANOS expiry index, normalized strike, then call before put. Calls and puts are
distinct entries and multiplicity is preserved.

The selection predicate is a direct, versioned transcription of the recovered
daily-front profile: side-specific active valid finite book after the
pre-smooth no-arbitrage mask; European; corresponding smooth fit finite;
business DTE at most 70 inclusive; 0.2D explicitly excluded; and K/F within
the inclusive AF-ATM NMS interval `[-3,2]`. It is not the Phase 6
`options_filter_eligible` predicate and must not silently inherit an experiment
hedge screen. The source frame must first match the exact Phase 7 fitted schema.

## Artifact layout and identity

Candidate namespace:
`sanos/public-candidate/name-<safe-name>`.
Release namespace: `sanos/public/name-<safe-name>`. Required candidate files:

```text
data.parquet
README.md
config.json
schema.json
capabilities.json
validation.json
```

The generic `manifest.json` and completion marker remain store-owned. Manifest
metadata should contain only recomputable exact values: source pointer, source
revision if already present in the source manifest, schema/producer/config IDs,
date/book/expiry counts, omitted-date summary, validation outcome, evidence
file checksums and exact capability flags. Do not flatten internal Raw lineage,
vendor identifiers, fitted intermediates, source filenames or source values
into the public artifact.

The candidate `ArtifactSpec` is a Dataset with
`DistributionStatus.PUBLIC_CANDIDATE` and the exact internal Dataset pointer as
its sole upstream. Identity includes canonical projection config, schema
version, resolved producer version and that pointer.

A public release is a **new immutable materialization**, not an alias or an
in-place manifest edit. It re-runs the exact deterministic projection from the
candidate's internal Dataset source, proves exact logical equality to the
approved candidate, and creates release-specific
README/config/validation/license evidence. Its `ArtifactSpec` uses
`DistributionStatus.PUBLIC`, the same internal Dataset pointer as its sole
upstream, and the exact approval fields. Put the approved-candidate evidence
and license payload checksum in release config so different approvals or
license bytes cannot share an artifact identity.

### Required artifact-store behavior

The current store's binding Phase 2 transition table remains:

```text
public_candidate <- internal
public           <- internal
```

Reject all other transitions, including candidate/public from candidate/public.
The current `ArtifactSpec`, publication validation and deep validation already
enforce this rule and need no Phase 8 change. The approved candidate is
self-contained approval evidence, not the public artifact's direct lineage.

## Output-first, deletion and wildcard behavior

`plan_public_candidate` can construct the final spec using only the exact
source pointer in config. `build_public_candidate` performs:

1. `store.resolve(candidate_spec, version=...)` before opening the source;
2. on a hit, self-contained public semantic validation and return;
3. on a concrete-version miss, shallow plus semantic validation of the exact
   internal source, then projection and atomic publication; and
4. on a `version="*"` miss, fail without opening/building/publishing anything.

Thus a complete candidate remains loadable/reusable after the internal Dataset,
Intermediate and Raw artifacts are unavailable. A complete public release is
likewise reusable after its candidate and internal source are unavailable.
Building a missing public release requires both the candidate evidence and the
internal source. Neither build nor approval deletes its source; deletion is an
explicit external lifecycle action. Shallow validation succeeds without
sources. `--deep-lineage` deliberately fails if the recorded internal source
was later deleted. Wildcard reuse emits the existing visible warning/result
record and never changes the stored manifest.

Do not reconstruct or replace a missing public artifact under an old identity.
An exact source pointer whose payload is missing is sufficient to plan/reuse an
already complete output but insufficient to build a new one.

## Atomic publication and restart behavior

Use `ArtifactBuilder`; never write inside the final path. Write Parquet to a
regular staging file, flush/fsync it and its directory, then write the evidence
documents. Run the pure schema, semantic, leakage and capability validators on
the staged payload **before** leaving the builder context. The completion
marker remains last and the final directory rename is atomic.

On interruption, no final artifact may exist and stale staging is removed on
the next locked attempt. Projection is deterministic and cheap enough to
restart from the source Dataset; no separate public Intermediate cache is
needed. For identical concurrent publishers, one commits and the loser catches
`ArtifactExistsError`, validates the winning exact artifact semantically, and
returns it. A different identity never replaces the winner.

Release approval first semantically validates the candidate, copies with the
no-follow streaming API, verifies staged byte hashes against the candidate,
adds approval/license evidence, validates the staged release, then commits.

## Leakage and semantic validation

The allowlist is positive and recursive. Validation must compare the exact
top-level and nested field names, order, dtypes and nullability. A field is
rejected even when all its values are null. No binary/object/opaque JSON column
is permitted. Name-token scanning is useful defense in depth, not a substitute
for exact schema equality.

Explicitly forbidden anywhere in the Parquet schema are:

- absolute spot, previous spot, absolute strike/forward/price and contract
  multiplier;
- raw/pure/normalized bid, ask or midpoint values (only the one allowed
  half-spread survives);
- ticker, secid, vendor/contract/option/expiry identifiers and symbols;
- volume/open interest and liquidity diagnostics;
- vendor IVs or Greeks;
- raw artifact IDs, source paths, source diagnostic JSON and row-level
  rejection strings; and
- stored fitted model prices, model IVs or quote-containment arrays.

Before projection, require an exact current `FITTED_SCHEMA`; do not silently
ignore a new internal column. Extraction selects named source columns into new
arrays and never performs a broad struct/list conversion or pass-through.

Semantic validation must independently re-derive:

- unique ascending dates and exact retained-date coverage;
- half-day flags from the versioned calendar;
- positive finite state supports/times/variances/bridges, strictly ordered
  model geometry, list-length/index consistency, non-negative finite masses,
  unit mass and first moment one within versioned tolerances;
- DLV boundary/length/time-increment invariants and density round trip;
- SANOS reconstruction and off-grid pricing from stored state;
- book sort, index association, side values, finite positive normalized strike
  and half-spread;
- inclusive/exclusive filter edges and the NMS bounds reconstructed from the
  kernel-input variance curve;
- approved return/carry formula, null/gap policy and positivity of its boundary
  inputs; and
- CD-009 price/spread conversions after approval, without persisting the
  derived prices.

Reject NaN and both infinities recursively in every non-null float and float
list. Re-signing manifest checksums must not make a semantic mutation pass.

`capabilities.json` and identical manifest metadata flags are a closed object:

```text
normalized_sanos_dynamics: true
normalized_dlv_dynamics: true
normalized_spread_experiments: true
exact_vendor_books: false
quote_containment_replay: false
liquidity_models: false
absolute_dollar_tick_effects: false
contract_level_replay: false
```

Validation must execute a small versioned smoke consumer for every `true`
capability using only the public artifact with `WRDS_OFF=1`. It must assert that
the allowed combination cannot recover an absolute level or original bid/ask
pair: a normalized spread plus state/bridge permits normalized experiments but
has neither an absolute scale nor a vendor midpoint.

## CLI contract

Keep existing internal `sanos dataset build/validate` behavior unchanged. Add a
separate subgroup so distribution state is never inferred:

```console
sanos dataset public build --config FILE [--version current|*]
sanos dataset public validate (--config FILE | --name NAME --artifact-id ID) \
    [--version current|*] [--integrity-only] [--deep-lineage]
sanos dataset public approve --candidate-name NAME --candidate-artifact-id ID \
    --approval FILE --license-file FILE [--version current|*]
```

The global `--dry-run`, `-v/--verbose`, and `--log-format text|json` continue to
apply. Dry-run reports the exact candidate/release identity, source pointer,
distribution transition and planned counts available without reading the
source; it writes nothing. Config-based wildcard is reuse-only. Exact-ID
validation does not accept wildcard. Approval never guesses an artifact by
“latest” and never accepts an unvalidated candidate or missing license bytes.

Human output ends with identity and path. JSON events/results use stable keys
and include `distribution`, `built`, `reused`, `dry_run`, `network_accessed`
(always false), `source_accessed`, validation summary and wildcard reuse record.
Map configuration errors and material/artifact errors to existing stable Click
exit categories. All commands expose useful `--help` and require no agent.

Emit flushed progress at start, source validation, per-date projection,
staged leakage validation, capability validation, publication and completion.
Use the existing monotonic 30-second default, never over 300 seconds, reporting
elapsed, completed/total, failures and ETA where meaningful. Approval emits
copy bytes/files and validation progress. Progress/logging is operational and
must not enter artifact identity.

## Required offline test matrix

1. Strict config parsing, canonicalization and identity sensitivity for every
   content field; operational settings do not change identity.
2. Exact nested schema/dtype/nullability and sorted one-row-per-retained-date
   behavior, including a half-day and the final approved typed-empty/gap rule.
3. Known synthetic CD-004 calculation: first null, exact carry interpolation,
   anchor, boundary, gaps and no unintended use of legacy `spot_return`.
4. Known synthetic CD-009 bridge in both directions and exact half-spread/S.
5. Edge population tests: 70D included, over-70D excluded, 0.2D excluded,
   NMS -3 and 2 included, just-outside excluded, American excluded, missing
   smooth fit excluded, call/put separate, stable multiplicity/order.
6. Complete DLV/SANOS state reconstruction, irregular support, density/DLV
   round trips, off-grid prices, ATM IV derived in memory and NMS re-derivation.
7. Every forbidden field at each nesting level, including all-null fields;
   unexpected FITTED source schema; opaque types; NaN/+inf/-inf in every float
   and list boundary; re-signed semantically corrupt payloads.
8. Self-contained open/semantic validation/capability execution with
   `WRDS_OFF=1` after moving the internal source and all lower stores away.
   Restore fixtures in `finally` blocks.
9. Output-first current/wildcard hits do not touch the source; wildcard miss
   cannot build; concrete miss requires the exact source; source deletion
   affects only deep validation.
10. Interrupted writes, staging cleanup, injected fsync/copy/validation errors,
    and concurrent identical candidate and release publishers.
11. Full distribution transition matrix, missing/malformed approval fields,
    license checksum mismatch, distinct candidate/public identities, immutable
    candidate after approval, and no direct internal-to-public route.
12. Artifact/file/manifest tampering, source-pointer mismatch, advertised
    capability mismatch and approval-record mismatch.
13. Text and JSON CLI success/errors/help/dry-run/progress, Click usage errors,
    warning capture, exact-ID validation and no implicit network access.
14. Fresh-process resolved-version and exact direct-dependency regression tests;
    Ruff, strict mypy for new code, compileall, full offline pytest and
    `git diff --check`.

## `@version` closure

Every content-affecting config/schema/projection/validator/README/capability and
publication helper must carry `@version`. At minimum:

- projection producer closure: strict config, recursive schema, source fitted
  schema/version, calendar/NMS conversion, model-state reconstruction,
  DLV/SANOS state encoders, daily-front predicate, approved return/carry
  interpolation, approved unit conversion, sorting/finite validators, Parquet
  writer settings and deterministic README/evidence generators;
- semantic validator closure: all recursive schema, numerical, leakage,
  capability and source-coverage validators;
- approval producer closure: candidate validator, exact-copy/checksum helpers,
  approval/release config, license handling, release documents and updated
  distribution-transition validation; and
- reader closure: exact schema/config parsers and public Dataset wrapper.

Do not include wall-clock timestamps, progress, source availability or runtime
paths in the producer closure/config. Continue the Phase 7 fresh-process test
that independently computes planned IDs and worker-produced IDs.

## Acceptance and sequencing

Recommended sequence after user decisions:

1. freeze CD-004/CD-009 and retained-date wording in docs and fixtures;
2. implement config/schema and transition-table tests;
3. implement pure per-day state/book projection, then the sequential return
   pass and exhaustive semantic/leakage tests;
4. implement candidate plan/build/open/validate with output-first and atomic
   behavior;
5. implement separate release approval/copy path and legal-evidence validation;
6. add CLI, docs and isolated capability consumers; and
7. run the full offline/static gates and independent code and quant reviews.

Phase 8 is complete only when both reviewers pass, forbidden/reconstruction
tests pass, candidate use succeeds with WRDS/internal/Raw/Intermediate absent,
every advertised capability runs from the public artifact alone, and approval
produces a distinct immutable `public` Dataset without mutating or aliasing the
candidate.

## Residual risks

- A transformed/fitted value being dimensionless does not establish legal
  redistribution rights; `public_candidate` must remain non-public until the
  separate approval record exists.
- The source contains Float32 boundaries while public computations may use
  Float64. Cast points are content semantics and must be frozen/tested, not
  “cleaned up” during implementation.
- A field-name blacklist alone is not a leakage boundary. Exact recursive
  schema plus value/derivability tests are mandatory.
- The public source pointer preserves reproducibility but may reveal an
  internal namespace label. Public names should be portable and non-sensitive;
  do not put credentials, filesystem paths, vendor security IDs or contract
  symbols in Dataset names.

## Addendum — binding Phase 2 lineage correction (2026-10-12)

**Correction:** the earlier recommendation that `public` directly derive from
`public_candidate` is withdrawn. It conflicts with the reviewed and accepted
Phase 2 artifact contract. I found no newer binding user instruction that
authorizes overturning that contract.

The binding direct-lineage rule is:

```text
public_candidate <- internal
public           <- internal
```

For both non-internal statuses, the **entire** direct upstream tuple is exactly
one shallow-validated Dataset whose manifest distribution is `internal`.
Candidate-on-candidate, public-on-candidate, candidate-on-public and
public-on-public remain invalid. The current `ArtifactSpec`, publication-time
upstream validation and deep validation implement this rule consistently and
must not be changed in Phase 8. This addendum supersedes the report's “Required
artifact-store change” transition table and any earlier statement that the
public release should copy from, or directly derive from, the candidate.

The separate approval action is instead implemented as a deterministic second
materialization from the same internal Dataset:

1. The approval record identifies an exact, semantically validated
   `public_candidate`, its manifest checksum, public logical-content digest,
   projection configuration ID, schema ID and exact internal source pointer.
2. The planned `public` spec has that same internal pointer as its sole direct
   upstream. Its canonical configuration contains the unchanged projection
   configuration plus the approved-candidate evidence and release/license
   record. Distribution is `public` with all mandatory approval fields.
3. Output-first resolution of the planned `public` artifact occurs before
   opening either source. The approval document must carry the exact evidence
   above so planning/reuse does not require the candidate or internal Dataset
   to remain present.
4. On a concrete-version miss, both the candidate and internal Dataset must be
   present and shallow plus semantically valid. The publisher re-runs the same
   versioned deterministic projection from the internal Dataset; it does not
   copy candidate data as its production source.
5. Before publication, compare the new frame to the approved candidate with
   exact schema, row/order/null/value equality and a versioned type-aware
   logical-content digest. Floating-point comparison is exact because this is
   the same producer/config/source, not a numerical-parity comparison. Record
   both candidate and re-projection digests in immutable approval evidence.
   Optionally require an exact `data.parquet` byte checksum when the same
   versioned writer is guaranteed, but logical equality is the normative
   content test and must not depend on incidental Parquet metadata.
6. Run the full leakage, semantic and advertised-capability validation on the
   newly staged public materialization, then atomically publish it. A mismatch
   fails without exposing a final artifact and cannot be overridden by
   `version="*"`.

Accordingly, adjust the earlier API/layout recommendations as follows:

- `plan_public_release` consumes a self-contained approved-candidate evidence
  document and produces a public spec with the internal Dataset as sole
  upstream.
- `approve_public_release` validates the candidate as approval evidence, opens
  the internal Dataset as the actual projection source, reprojects, proves
  exact candidate equality, and publishes. It does not use
  `ArtifactBuilder.copy_file` for `data.parquet`.
- The public artifact's approval evidence records the candidate pointer and
  checksums, but that candidate is not a direct manifest upstream. Its absence
  after publication does not break shallow reuse. Generic `--deep-lineage`
  traverses the sole internal upstream as required by Phase 2; a separate
  optional approval-evidence audit may revalidate the candidate when retained.
- Neither approval nor build deletes the internal Dataset or candidate.
  Existing complete public output remains shallow-valid and capability-usable
  if either later becomes unavailable.

Replace the transition tests proposed earlier with the binding matrix: positive
internal-to-candidate and internal-to-public cases; rejection of every
candidate/public-on-non-internal edge; public publication failure when the
approved candidate and fresh re-projection differ in schema, order, nulls or
any nested value; and output-first public reuse after both evidence/source
artifacts are moved away. Retain tests that the public and candidate are
distinct immutable identities and that approval never mutates or aliases the
candidate.
