# Phase 7 code and architecture design audit

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Scope: internal Dataset assembly, publication, loading, and validation  
Baseline: committed Phase 6 tree `675d171685954089b49c566459eb570b019c9c8d`  
Network/WRDS access: none

## Recommendation

Implement Phase 7 as an explicit raw-to-Dataset publication boundary backed by
the existing `ArtifactStore`. Do not add Redun, HDF5, a second artifact store,
or a database. The published object should contain byte-copied fitted-day
Parquet partitions plus portable index, schema, lineage, validation, config,
and README payloads. It must load and validate with `WRDS_OFF=1` after every
Intermediate has been removed.

Treat fitted-day artifacts as rebuildable assembly caches, not as the logical
source identity of the permanent Dataset. Add one small
`raw-lineage-catalog` Intermediate whose upstream tuple is the complete sorted,
deduplicated Raw closure for the requested dates. The internal Dataset has that
catalog as its sole direct upstream; its producer closure contains the complete
daily pipeline. This has four useful properties:

1. the Dataset identity contains canonical configuration, schema, pipeline
   version, and exact Raw lineage without making an immutable Dataset depend on
   retained daily caches;
2. a completed Dataset can be resolved before the catalog, fitted days, or Raw
   roots are read;
3. final publication validates one small catalog rather than re-hashing every
   Raw payload after the assembly pass; and
4. a Phase 5/6 code change is represented by the Dataset producer version,
   while the source-data lineage remains independently auditable.

The catalog is an Intermediate and may be deleted after publication. Its
absence must not affect shallow Dataset validation or loading. Deep lineage
validation explicitly requires it and the Raw artifacts.

## Modules and public APIs

Add the following boundaries under the existing empty `sanos.datasets`
package.

### `sanos.datasets.config`

```text
InternalDatasetConfig
  name: str
  ticker: str
  start_date: date
  end_date: date
  include_half_days: bool = True
  source: Literal["optionmetrics"] = "optionmetrics"
  source_generation: str = "default-v1"
  daily: DailyWidbConfig
  surface: DailySurfaceConfig

DatasetBuildOptions                 # operational, excluded from content identity
  max_workers: int = 1
  progress_interval_seconds: float = 30
  require_existing_intermediates: bool = False
  version_override: Literal["*"] | None = None
```

Both are strict frozen dataclasses. `InternalDatasetConfig` is versioned and
rejects unknown keys, unsafe/empty names, invalid dates, unsupported sources,
and non-mapping `daily`/`surface` sections. Parse one checked-in document with
top-level `dataset`, `daily`, and `surface` objects. Keep paths, credentials,
worker counts, verbosity, journals, and version overrides out of the canonical
content configuration.

Extend `configs/widb-spx-reference.json` or add a clearly named complete
profile with an exact `dataset` section. The intermediate CLI may accept and
ignore that known section when using the same file, but must continue rejecting
all other unknown sections. Existing two-section configurations remain valid.

`distribution=internal` is fixed by the internal publisher and is not a CLI or
configuration choice. Phase 7 must not expose public/public-candidate switches.

### `sanos.widb.build`

Expose the existing private daily identity construction through one reviewed
pure API rather than duplicating it in the Dataset layer:

```text
DailyArtifactPlan
  trading_date
  raw: RawDayPointers
  paired, enriched, no_arb, filtered, fitted: ArtifactSpec

plan_daily_artifacts(request, daily, surface) -> tuple[DailyArtifactPlan, ...]
```

`plan_intermediate_build`, the spawned worker, and Phase 7 must consume this
same API. The function is versioned and deterministically orders dates. This
prevents Dataset planning from drifting from the identities actually built by
Phase 6.

### `sanos.datasets.internal`

Use typed plan/result objects and small functions with no CLI dependency:

```text
InternalDatasetPlan
  configuration
  expected_dates
  daily_artifacts
  raw_catalog_spec
  dataset_spec
  plan_id

InternalDatasetBuildResult
  expected, terminal, reuse
  dry_run
  daily_summary
  validation_summary
  journal_path

DatasetValidationSummary
  requested/populated/excluded date counts
  row, expiry, eligible, model, active-call, active-put counts
  coverage and exclusions
  invariant residual maxima

plan_internal_dataset(config) -> InternalDatasetPlan
build_internal_dataset(locations, config, options, *, dry_run, progress)
validate_internal_dataset(store, artifact, *, semantic=True, deep_lineage=False)
open_internal_dataset(store, artifact) -> InternalDataset
```

`InternalDataset.scan()` returns a Polars `LazyFrame` over the exact partition
paths listed in the index with hive inference disabled; `load_day(date)` reads
one partition. Neither API accepts a glob or consults upstream storage.

## Identity and artifact layout

Use collision-resistant, versioned label normalization for `name` and `ticker`.
Suggested namespaces are:

```text
Intermediate: widb/raw-lineage-catalog/name-<safe-name>
Dataset:      widb/internal/name-<safe-name>
```

The raw catalog spec has all unique expected Raw pointers as upstreams and a
small schema such as `widb-raw-lineage-catalog-v1`. Its payload records, for
every Raw artifact, the pointer, manifest checksum, payload checksums, table,
ticker, observation date, and acquisition generation. It also records the
per-day six-pointer mapping, including the previous-business-day boundary.

The internal Dataset spec has:

- `StorageClass.DATASET`;
- `DatasetDistribution(INTERNAL)`;
- the complete canonical `InternalDatasetConfig`;
- a schema version resolving the partition-layout version and
  `FITTED_SCHEMA_VERSION`;
- a producer version resolving the entire paired/enriched/no-arbitrage/filter/
  fitted chain plus Dataset validation/rendering/copy helpers; and
- the raw catalog pointer as its sole direct upstream.

Do not put Git revision, timestamps, host paths, workers, or progress tuning in
the semantic pipeline version. Record the exact 40-hex source revision in
manifest metadata. Resolve it from packaged build provenance first and a clean
Git checkout second; fail publication with an actionable message when an exact
revision is unavailable. Validation of an existing Dataset uses its stored
revision and does not require Git.

Recommended immutable payload layout:

```text
README.md
config.json
schema.json
index.json
lineage.json
validation.json
data/ticker=<safe>/year=<YYYY>/date=<YYYY-MM-DD>/part-00000.parquet
```

Copy each validated fitted-day `data.parquet` byte for byte with
`ArtifactBuilder.copy_file`; do not concatenate, pad, re-encode, or split the
daily frame. Keep a typed-empty partition for every approved empty day. The
index is sorted by trading date and records the partition path, fitted artifact
pointer, source and copied checksum, rows, expiries, eligibility/model/activity
counts, typed-empty state, and exclusion reason.

The generated artifact-store manifest already records checksums and sizes for
every payload. Add Dataset metadata containing:

- `distribution: internal` through the existing distribution object;
- Git revision and resolved pipeline/schema versions;
- requested, populated, and excluded coverage;
- aggregate row/expiry/filter/model/active-side counts;
- the complete sorted Raw pointer list and raw-catalog identity;
- exact fitted-day pointers used for assembly;
- exclusion records and bad-data policy version; and
- checksums of `index.json`, `lineage.json`, and `validation.json`.

This satisfies the requirement that `manifest.json` itself carries upstream
Raw identities, while `lineage.json` supplies the more detailed per-day map.
The legacy 891-date artifacts and paths must never appear as upstreams.

## Planning, build, and restart sequence

The build order is binding:

1. Purely plan expected NYSE sessions, Raw pointers, fitted-day specs, raw
   catalog spec, and Dataset spec. Dry-run stops here and creates no roots.
2. Resolve the requested Dataset shallowly before reading any upstream. A hit
   returns immediately. `version="*"` emits/returns the existing warning and is
   reuse-only; a miss fails before any Intermediate or catalog is built.
3. On a concrete miss, run/reuse the exact current `fitted-day` plan. By
   default `dataset build` completes missing Intermediates from existing Raw;
   it never downloads. `--require-existing-intermediates` instead reports all
   missing dates and exits without building them.
4. Validate every fitted artifact and its Raw lineage, then atomically publish
   or reuse the raw catalog. The observed lineage must equal the planned Raw
   closure exactly.
5. Enter the Dataset writer, copy and validate each partition, write the
   portable metadata payloads, and publish only after all semantic gates pass.
   Catch an `ArtifactExistsError` from a concurrent publisher only by validating
   and returning the immutable winner.
6. Re-open the published Dataset through the public loader and run the same
   self-contained validation before reporting success.

Journal Dataset runs under
`QUANT_DATA_DATASETS/.runs/widb/<plan-id>.jsonl` using the existing durable
`RunJournal` format and canonical plan check. Emit phase, elapsed time,
completed/total work, failures, reuse count, and ETA at most every 30 seconds.
Do not retain frames in the journal or parent result.

An interruption during daily work resumes through the existing Phase 6
journal. An interruption during final copy leaves no visible Dataset;
`ArtifactBuilder` cleans or replaces only its own stale identity-specific
staging directory, the raw catalog is reused, and the explicit next
`dataset build` safely repeats the copy. Incremental final-copy resume is not
worth weakening atomicity for this data volume.

Only `sanos dataset build` may construct a missing Dataset. Load, validate,
status, public export, and experiment commands must never invoke this builder.
A corrupt or incomplete directory at the expected identity is an integrity
error and is never overwritten. Rebuilding an absent Dataset is therefore an
explicit user action, never silent dependency resolution.

## Completeness and semantic validation

Publication must enforce all of the following without reading a migration
cache:

1. Expected sessions come only from the configured calendar/range/half-day
   policy. There is exactly one partition and one fitted pointer per expected
   date, with no extras or duplicates.
2. Every partition has exact `FITTED_SCHEMA`, ticker and trading date, and the
   deterministic expiry/normalized-strike order. Row keys are unique.
3. A typed-empty day is accepted only when the versioned bad-data policy names
   that date and the artifact metadata carries the matching reason/version.
   An unexpected empty or nonempty day with no eligible/model population blocks
   publication pending review; it is not silently dropped.
4. Configuration-ID columns are constant and equal the planned no-arbitrage,
   filter, and smooth projections. Stage/schema versions and all source artifact
   IDs are consistent with the plan.
5. Eligibility predicates and first-rejection reasons are internally
   consistent. Active calls/puts, smooth-fit masks, and model masks retain their
   distinct Phase 6 meanings.
6. Numeric fields used by selected books are finite and have valid bid/ask,
   forward, discount, expiry, and normalization relationships. Reconstruct both
   model states; enforce ordered support, density positivity/tolerances, unit
   mass/first moment, DLV boundary zeros, price monotonicity/convexity, calendar
   consistency, and stored fitted-price tolerances using version-owned Phase 6
   constants.
7. Recomputed per-partition and aggregate counts, exclusion records, residuals,
   schema, and checksums exactly match the index, validation payload, and
   manifest metadata.

Default `dataset validate` performs complete self-contained artifact and
semantic validation but does not traverse lineage. `--deep-lineage` additionally
validates the raw catalog, all Raw artifacts, and the recorded fitted-day chains;
it is expected to fail after those optional Intermediates are removed. The
default remains successful in that state.

The 891-day legacy population is a migration oracle, not a hard-coded Dataset
schema rule or upstream. Produce a separate comparison report during the fresh
run and block canonical publication on unexplained discrepancies, but do not
modify fresh rows to force legacy counts. Generic Phase 7 loading/publication
must work without legacy files.

## CLI contract

Implement these human- and automation-facing commands behind the existing
global location, verbosity, dry-run, and log-format options:

```text
sanos dataset build --config FILE [--max-workers N]
                    [--require-existing-intermediates]
                    [--version current|*]

sanos dataset validate --name NAME --artifact-id SHA256
                       [--deep-lineage] [--integrity-only]

sanos dataset validate --config FILE [--version current|*]
                       [--deep-lineage] [--integrity-only]
```

Exact `--artifact-id` validation must remain available after code versions
change. Config-based validation resolves current or wildcard identity without
building. Reject ambiguous selector combinations at Click parsing time.

Structured build progress uses `dataset_progress`; the terminal
`dataset_result` includes distribution, artifact pointer/ID/path, config ID,
schema and producer versions, counts, exclusions, build/reuse status, wildcard
record/warning, daily build summary, journal path, `dry_run`, and
`network_accessed=false`. `dataset_validation` reports shallow/deep mode,
integrity/semantic status, stored Git revision, counts, and actionable failures.
Text output contains the same decisions in readable form. Dataset commands
never initialize WRDS or request 2FA.

Update `sanos status`, README, architecture, artifact, and data-contract docs so
a non-AI user can run: raw download, fitted Intermediate build, Dataset build,
Dataset validate, optional deep audit, then safe Intermediate cleanup.

## Version closure

Apply `@version` to every configuration, schema, planner, transform, renderer,
validator, loader, and path-layout helper. At minimum:

- the raw-catalog producer depends on Raw pointer/manifest fingerprinting,
  boundary planning, catalog schema, and canonical serialization;
- the internal Dataset schema depends directly on the fitted schema and
  partition/index layout;
- the Dataset producer depends on all five daily producer closures,
  `InternalDatasetConfig`, semantic validation, partition path/copy logic,
  index/lineage/README/schema/validation rendering, bad-data policy, and source
  revision handling; and
- load/validate versions depend on strict index/schema parsing and every
  invariant helper they invoke.

Test the direct dependency names and resolved strings. Changing
`overnight_variance_fraction`, filter/SANOS settings, dates, half-day policy,
source generation, Raw identity, schema, or content-affecting code must change
the Dataset identity. Executor tuning and prose-only logging must not.

## Offline acceptance matrix

Phase 7 is ready to commit only when all of these pass with `WRDS_OFF=1`:

1. A synthetic multi-day build includes a populated day, half-day, boundary
   previous close, and approved typed-empty day; all exact schema, partition,
   coverage, count, exclusion, lineage, and invariant assertions pass.
2. Output-first reuse succeeds after copied Raw/Intermediate/catalog trees are
   removed. Default validation and `scan`/`load_day` still work; explicit deep
   lineage validation fails clearly.
3. `version="*"` reuses an existing complete Dataset with a visible structured
   warning and never builds on a miss. Exact-ID validation works independently
   of the current producer version.
4. Missing, duplicate, reordered, extra, corrupt, wrong-date, wrong-ticker,
   wrong-schema, count-mismatched, unexpected-empty, and numerically invalid
   partitions are rejected before publication or by self-contained validation.
5. Injected interruption during Intermediate completion and during Dataset copy
   publishes no partial Dataset; rerun reuses completed work and succeeds.
   Concurrent identical publishers return one validated immutable winner.
6. Artifact corruption, symlink/path attacks, an altered index/lineage/config,
   and a Raw/fitted lineage mismatch are rejected. No replacement path exists.
7. CLI help, text, JSON, dry-run, exact-ID/config validation, progress,
   actionable errors, and final identity/path are tested. Dry-run and all
   validation modes are offline and non-mutating.
8. After publication, deleting Intermediates has no effect on Dataset loading,
   checksums, schema/invariant validation, or an advertised internal consumer.
9. Full pytest, Ruff, strict mypy, compileall, build/install smoke test, and diff
   checks pass. Both independent code and quantitative/data final reviews record
   PASS.

## Risks and decisions

- Do not use direct fitted-day pointers as the permanent Dataset's only logical
  upstream. Besides coupling permanent identity to disposable cache topology,
  that would make final commit re-hash hundreds of Parquet sources and would
  unnecessarily narrow published-boundary wildcard reuse. The raw-lineage
  catalog keeps the source identity explicit and small at publication time.
- Do not weaken `ArtifactStore.validate` to make large publication faster. Use
  the catalog boundary and retain strict checksum validation.
- Do not implement resumable writes inside the final atomic staging directory.
  Daily computation and lineage validation are already cached; repeating a
  byte-copy after interruption is simpler and safer.
- A source-revision mechanism that can emit `unknown` does not satisfy the
  Phase 7 manifest contract. Packaged builds must embed an exact revision, and
  publication must fail rather than invent provenance.
- Fresh full-history counts and legacy discrepancies cannot be accepted in an
  offline design phase. They remain a later dual-review publication gate and
  must be reported, not normalized away.

Signed: `code agent /root/code`
