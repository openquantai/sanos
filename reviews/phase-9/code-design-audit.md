# Phase 9 code/design audit

**Reviewer:** independent code agent (`/root/code`)  
**Date:** 2026-10-12  
**Scope:** design only; no implementation changes and no WRDS/network access  
**Verdict:** **IMPLEMENTATION READY for the internal daily-front input
materialization.** A public-normalized Experiment profile remains blocked by
Phase 8/CD-004/CD-009. Model fitting and learned Direct-200 checkpoints are not
part of Phase 9 unless the user explicitly expands the scope.

## Evidence and conclusions

I reviewed the repository instructions and plan, the Phase 2 artifact
contract, Phase 7 internal Dataset reader/validator, current CLI and artifact
store, the pinned/reference inventories, and the read-only `quant_data`
materialization/consumer paths. The important conclusions are:

1. The production reference path is **not** legacy `widb.timeseries`. That
   function is an older generic padded `PYTREE_HDF5` cache. The pinned
   daily-front path is `market_day_from_payload` followed by the FLEX training
   consumer (`FlexMarketDay`, `materialize_day`, `materialize_batch`).
2. The exact daily-front consumer needs normalized bids, asks, fit prices,
   liquidity features, quote identities and an absolute spot reference. Those
   are deliberately forbidden from the Phase 8 public Dataset. Therefore the
   exact reference Experiment must source the **internal Dataset**, not a
   public candidate/public projection.
3. Phase 9 materializes training inputs; it does not fit the neural model. The
   legacy `joint-direct-200-targets.pt` states and learned checkpoint are
   experiment-output parity oracles, not input fields to copy into a new HDF5.
4. The exact all-history Direct-200 reference run states that all usable dates
   enter training and that there is no temporal validation holdout. Preserve
   that explicitly for the reference profile. A pre-2025/2025 holdout is a
   separate configured Experiment identity, not a silent change.
5. Direct h5py with an explicit closed schema is preferable to copying generic
   `SubDir.PYTREE_HDF5`. The existing `h5py>=3.10,<4` experiment extra and
   test/dev dependency are sufficient for materialization.

## Initial profile and scope

Name the first profile `dysanos-daily-front-market-v1`. It materializes the
exact market tensors consumed by the recovered all-history FLEX path:

- smoothness `0.25`;
- non-binding upstream cap `100000` paired strike rows;
- explicit exclusion of business DTE `0.2`;
- European books only;
- separate real calls and puts;
- finite corresponding smooth fit;
- inclusive canonical AF-ATM NMS `[-3,2]`;
- model business DTEs `1,2,3,4,5,10,15,...,70` (18 nodes); and
- 200 inner strike coordinates.

The filter is copied/adapted from the pinned `market_day_from_payload` path and
must operate on the Phase 7 fitted partitions. It is not the older
`widb.timeseries` selection and it must not apply a different downstream hedge
screen.

The materialized prices are the reference discounted-forward-normalized real
books (`C/(DF*F)`), with call and put fit values in their own payoff units.
This profile therefore does not need CD-004 or CD-009. It may include absolute
spot, the retained expiry identity, and the pinned consumer's deterministic
`date:expiry:strike:side` quote label because it is an internal disposable
Experiment, not a redistributed Dataset. It does not need to preserve vendor
option identifiers merely to label quotes. The manifest and README must say so.

Add future profiles rather than adding optional columns to this schema:

- `normalized-public-dynamics-v1` may source a validated public candidate or
  public Dataset after Phase 8 is complete and may contain only advertised
  public capabilities;
- fixed-grid DLV/JAX or other consumers receive separate schemas/configs; and
- learned model states/checkpoints are outputs of training, not revisions of
  this input materialization.

## Source Dataset eligibility

An Experiment spec and publisher should impose a stronger profile-level rule
than the generic store currently needs:

- the entire upstream tuple is exactly one `StorageClass.DATASET` pointer;
- for `dysanos-daily-front-market-v1`, its distribution must be `internal`;
- its manifest/schema must be the supported Phase 7 internal Dataset contract,
  and default build-time validation calls `validate_internal_dataset(...,
  semantic=True, deep_lineage=False)`;
- the source must have one exact ordered fitted partition per planned date,
  with populated/typed-empty status already validated; and
- no Raw or Intermediate artifact is opened. Deep source lineage remains an
  explicit optional validation operation.

The generic artifact store need not change. Its Experiment lineage already
permits Dataset upstreams and publication shallow-validates them. The typed
Experiment publisher must enforce exact-one and distribution/schema
eligibility before materialization.

Typed-empty source partitions do not become padded “market days”; omit them,
record their dates/reasons in coverage evidence, and require every retained day
to have at least one eligible book. Preserve strictly increasing retained
dates. This matches the 891 populated-day oracle without copying its omissions
into a fresh build.

The source Dataset pointer is sufficient immutable content provenance and is
part of the Experiment artifact identity. A materialization may be built for a
valid but not-yet-canonically-promoted source during offline/pilot work, but it
must not claim to be the authoritative reference Experiment. Record the source
promotion/reference status. Phase 11 records the authoritative Experiment only
after the fresh Dataset has passed its production gates.

## Package and APIs

Use a dedicated package:

```text
src/sanos/experiments/
    __init__.py
    config.py          strict profile/geometry/split/HDF5 configuration
    daily_front.py     fitted-day -> immutable NumPy day payload
    hdf5.py            closed-schema writer and reader
    validate.py        logical/schema/consumer validation
    build.py           plan, restart workspace, assembly and publication
```

Recommended public APIs:

```python
plan_experiment(config: ExperimentConfig) -> ExperimentPlan
build_experiment(locations, config, options, *, dry_run=False, progress=None) \
    -> ExperimentBuildResult
open_experiment(store, pointer_or_validated) -> DailyFrontExperiment
validate_experiment(store, pointer_or_validated, *, semantic=True,
                    deep_lineage=False, progress=None) \
    -> ExperimentValidationSummary | None
```

`DailyFrontExperiment` exposes dates/splits and `load_day(index_or_date)` as
NumPy arrays. Add an adapter that reconstructs the exact logical
`FlexMarketDay` field set. It should be duck-type compatible with the pinned
`materialize_day`; the core HDF5 package must not require Torch. A separate
optional Torch integration may turn an adapter day into device tensors.

## Canonical configuration

Use strict versioned dataclasses and reject unknown keys, non-finite numbers,
implicit booleans and unordered/duplicated date rules. A checked-in document
should have exact sections:

```json
{
  "experiment": {
    "name": "spx-daily-front-reference-v1",
    "profile": "dysanos-daily-front-market-v1",
    "source_dataset": {
      "storage_class": "dataset",
      "namespace": "...",
      "artifact_id": "..."
    },
    "consumer_contract": "flex-market-day-v1"
  },
  "market": {
    "smoothness": 0.25,
    "max_num_options": 100000,
    "excluded_business_dtes": [0.2]
  },
  "geometry": {
    "model_business_dtes": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "nms_range": [-3.0, 2.0],
    "num_inner_strikes": 200
  },
  "objective": {
    "fit_weight": 0.05,
    "date_weighting": "mean_quotes_then_mean_dates"
  },
  "split": {
    "mode": "all_history_no_holdout",
    "diagnostic_boundary": "2025-01-01"
  },
  "storage": {
    "float_dtype": "float64",
    "padding": "global_max_with_masks_v1",
    "compression": "gzip",
    "compression_level": 4,
    "shuffle": true,
    "fletcher32": true
  }
}
```

All fields above affect artifact identity, including physical HDF5 settings.
Worker count, progress interval, resume/keep-work controls and wildcard request
belong in a separate `ExperimentBuildOptions` and do not affect identity.

The exact no-holdout rule is deliberate reference parity, not absence of split
metadata. Store train/validation/test masks: train is all retained dates and
validation/test are explicitly empty. The pre-2025 and 2025 masks are labelled
diagnostic evaluation periods, not holdouts. A non-empty temporal holdout uses
a distinct split mode/configuration and artifact ID.

Do not put filesystem paths, cache labels, legacy artifact paths, credentials,
GPU selection or wall-clock fields in canonical config.

## HDF5 schema

Use one `experiment.h5` with a closed hierarchy. All numeric dtypes have fixed
little-endian encodings; dates are signed days since 1970-01-01 rather than
locale strings. Fixed-size ASCII is adequate for hashes/IDs. Reject object
dtypes, pickles, external links, soft links and virtual/external data sources.

Let `D` be retained dates, `E` maximum observed eligible expiries per day, `Q`
maximum separate side books, `P` maximum distinct `(expiry_index,K/F)` price
locations, and `M=18` fixed model nodes.

```text
/axes/date_days                         int32  [D]
/axes/model_business_dtes               float64[M]
/axes/model_times_years                 float64[M]

/day/is_half_day                        bool   [D]
/day/spot_reference                     float64[D]
/day/market_uid                         S64    [D]
/day/market_fingerprint                 S64    [D]

/expiry/count                           int32  [D]
/expiry/mask                            bool   [D,E]
/expiry/time_years                      float64[D,E]
/expiry/date_days                       int32  [D,E]
/expiry/source_expiry_id                int64  [D,E]
/expiry/is_am                           bool   [D,E]
/expiry/atm_annual_volatility           float64[D,E]
/expiry/strike_lower_bound              float64[D,E]
/expiry/strike_upper_bound              float64[D,E]

/model/atm_total_volatility             float64[D,M]

/quote/count                            int32  [D]
/quote/mask                             bool   [D,Q]
/quote/expiry_index                     int32  [D,Q]
/quote/normalized_strike                float64[D,Q]
/quote/is_call                          bool   [D,Q]
/quote/bid                              float64[D,Q]
/quote/ask                              float64[D,Q]
/quote/smooth_fit_price                 float64[D,Q]
/quote/log_volume                       float64[D,Q]
/quote/log_open_interest                float64[D,Q]
/quote/quote_id                         S96    [D,Q]
/quote/price_location_index             int32  [D,Q]

/price_location/count                   int32  [D]
/price_location/mask                    bool   [D,P]
/price_location/expiry_time_years       float64[D,P]
/price_location/normalized_strike       float64[D,P]

/split/train_mask                       bool   [D]
/split/validation_mask                  bool   [D]
/split/test_mask                        bool   [D]
/period/pre_2025_mask                   bool   [D]
/period/from_2025_mask                  bool   [D]

/audit/active_european_books            int64  [D]
/audit/horizon_eligible_books           int64  [D]
/audit/nms_eligible_books               int64  [D]
/audit/input_expiries                   int32  [D]
/audit/eligible_expiries                int32  [D]
```

The root has exact scalar attributes for schema/profile, source Dataset
pointer components, configuration ID, concrete producer version, units,
`D/E/Q/P/M`, and a type-aware logical-content digest. Attributes are redundant
validation anchors, not a second mutable configuration.

Real values occupy the prefix defined by each count; masks must equal those
prefixes exactly. Padding is deterministic and validated: zero times/counts,
ATM strike `1`, call side `true`, bid/fit `0`, ask `1`, empty IDs, index `-1`
until sliced by the adapter. No consumer may infer validity from padding values;
it uses masks/counts.

The `price_location` arrays and quote index are re-derived exactly from first
occurrence of each ordered `(expiry_index, normalized_strike)` key, matching
the reference consumer's de-duplication. The validator proves that every real
quote maps to the correct unique location.

All real floats must be finite. Expiry times and normalized strikes are
positive and ordered; expiry indices are in range; real books satisfy
`0 <= bid < ask`; fit prices are finite; model ATM total volatilities are
positive; bounds straddle one. Source quote IDs and fingerprints are
deterministic provenance only and are not copied from the legacy oracle.

`market_uid` should hash the fresh source's relevant daily/filter/surface
configuration identities. `market_fingerprint` should use the established
length-prefixed scalar/contiguous-array scheme over all ordered objective
fields plus the exact source Dataset identity. It intentionally does not equal
a legacy fingerprint merely to make old checkpoints appear reusable.

## Artifact layout and manifest

Use namespace
`dysanos/daily-front-market/name-<safe-name>` under the Experiment root. The
immutable artifact contains:

```text
experiment.h5
README.md
config.json
schema.json
source.json
splits.json
validation.json
manifest.json          # store-owned
COMPLETE.json          # store-owned and written last
```

`source.json` records the exact Dataset pointer, source manifest checksum,
distribution/schema/configuration IDs and source semantic-validation summary
digest. `schema.json` documents every HDF5 path, dtype, shape symbol, unit,
padding and null/mask rule. `splits.json` records exact ordered indices/date
ranges/counts and distinguishes training splits from diagnostic periods.

`validation.json` records exact dimension/population summaries, per-HDF5-
dataset logical digests, combined logical digest, consumer round-trip result
and validator version. The generic artifact manifest provides byte sizes and
SHA-256 for every payload, including the complete HDF5 file.

Do not include legacy paths, mutable “latest” aliases or learned checkpoints.
An Experiment is not a Dataset and has no distribution status; its README
inherits the source's access restrictions and must not imply that an internal
Experiment is redistributable.

## Output-first and wildcard behavior

Planning is pure: canonical config plus exact source pointer is enough to form
the Experiment spec without opening any root. Build order is:

1. `store.resolve(experiment_spec, version=requested_version)`;
2. on a hit, validate the Experiment itself and return without touching the
   source Dataset or restart workspace;
3. on a concrete miss, validate the exact source Dataset semantically but
   shallowly, then resume/materialize and publish; and
4. on a `version="*"` miss, fail before source/workspace access and require a
   concrete version.

Wildcard skips only producer comparison, emits the existing warning and result
record, and cannot build or mutate. Configuration, source pointer, schema,
completeness and checksums remain exact. Default `experiment validate` is
self-contained and shallow. `--deep-lineage` validates the source Dataset and
its selected semantic contract; it is expected to fail if that Dataset has
been deleted.

A completed Experiment remains reusable after Dataset/Intermediate/Raw removal.
If the Experiment is deleted, it can be rebuilt only while the exact Dataset
still exists. Build never deletes or rewrites its Dataset.

## Restartable build and process ownership

Do not make a partially written HDF5 file resumable in place. Use a private,
plan-owned preparation workspace under the configured Experiment root, for
example `.sanos-runs/<plan-id>/`, protected by a no-follow lock. It contains:

- canonical plan/config/source identity;
- Phase 2 append-only/fsynced run journal; and
- one atomic, checksummed, `allow_pickle=False` NPZ shard per retained date.

Daily workers receive only a validated source partition path/checksum, date and
canonical projection config. They return a bounded shard identity, never large
arrays through process IPC. Use the existing spawn runner and its interruption
ownership/termination handshake. Default to one worker, allow 1--4 CPU workers,
and open no WRDS connection.

On restart, validate plan identity and every completed shard before skipping
it. A partial tail journal is recoverable under the existing rule; an earlier
malformed record, wrong plan, symlink, missing shard or checksum mismatch is
fatal or rebuilt explicitly. Never adopt a workspace from another plan.

After all shards exist, derive bounded `D/E/Q/P/M` dimensions with checked
integer arithmetic, create HDF5 inside `ArtifactBuilder` staging, and stream
one day/chunk at a time. Close and flush h5py, fsync the regular file and staging
directory, reopen read-only, and run full semantic/consumer validation before
the builder commits. The generic manifest/completion/final rename remain last.

For concurrent identical publishers, a plan/workspace ownership lock prevents
duplicate preparation. A waiter periodically polls for the final artifact and
may reuse a valid winner. At final publication, an `ArtifactExistsError` loser
validates and returns the winner. Different identities never share a workspace
or replace one another.

Successful publication may remove its exact run workspace; cleanup failure is
a warning and does not invalidate the immutable Experiment. `--keep-work`
preserves it for diagnosis. `--restart-work` is an explicit destructive option
which prints the exact plan-owned workspace before removal and never touches a
published artifact or a different plan.

## HDF5 safety and determinism

- Use direct `h5py.File(..., "w")`; do not use generic PyTree serialization or
  Python object/pickle payloads.
- Fix group/dataset creation order, dtype byte order, chunk shape, gzip level,
  shuffle, Fletcher32 and fill values in the versioned writer contract.
- Disable object timestamps where h5py/HDF5 exposes `track_times=False`; do not
  put current time, host, PID or absolute path in HDF5.
- Write day-major chunks so a training batch can select rows without reading
  the full file. Do not require SWMR for an immutable completed file.
- Reader validation enumerates exact groups/datasets/attributes and rejects
  unknown objects, links, references, external filters and unexpected dtype or
  shape. It never executes code from the file.
- Byte-for-byte HDF5 determinism is desirable and testable in one supported
  build, but the normative content identity is the versioned type-aware logical
  digest; HDF5-library metadata must not affect numerical parity.
- Check `D*E`, `D*Q`, `D*P` and estimated bytes before allocation. Reject
  negative, excessive or overflowed dimensions with an actionable message.

## CLI contract

Replace the current placeholder and add validation:

```console
sanos experiment build --config FILE --max-workers 1 \
    [--version current|*] [--no-resume] [--keep-work] [--restart-work]
sanos experiment validate (--config FILE | --name NAME --artifact-id ID) \
    [--version current|*] [--integrity-only] [--deep-lineage]
```

The existing global `--dry-run`, `-v/--verbose`, location overrides and
`--log-format text|json` apply. Dry-run computes the exact artifact/plan/work
identities, source pointer and profile without opening HDF5 or source data and
writes nothing. `--version '*'` is config-based reuse-only; exact-ID validation
is concrete.

Human completion output prints identity, path, source Dataset, date/book counts,
split counts, built/reused/resumed status and validation outcome. JSON events
use stable keys and include `network_accessed=false`, `source_accessed`,
`filesystem_modified`, reuse/wildcard warning, journal/workspace, completed/
total, failures, elapsed and ETA.

Emit flushed progress at source validation, shard preparation, HDF5 allocation,
day/chunk writes, logical hashing, consumer validation and publication. Default
heartbeat is 30 seconds and never exceeds 300. An HDF5 compression call or
source validation that runs longer than the interval needs a parent heartbeat,
not silent blocking. This is CPU materialization, so GPU allocation is neither
needed nor permitted implicitly.

Missing `h5py` raises an actionable request to install `sanos[experiment]`.
Do not import h5py at package/CLI startup for unrelated commands. Click parsing
and material/configuration failures retain the established structured envelopes
and exit codes.

## Reader and reference-consumer adapter

The reader slices all arrays by validated counts/masks and returns immutable
NumPy values with the logical fields of `FlexMarketDay`. The adapter recomputes
and checks:

- per-quote expiry years;
- distinct price locations and quote-to-price index;
- half-spreads and call-equivalent put transformation;
- quote IDs, market fingerprint and training fingerprint; and
- model/observed ATM curves and canonical NMS bounds.

An offline integration probe in an environment containing the pinned reference
consumer passes each adapted day to `materialize_day`, then compares all tensor
values, masks, shapes, indices and dtypes against direct materialization of the
same synthetic/reference day. This probe may use Torch, but Torch must not
become a base dependency. Before adding a mandatory Torch CI extra, obtain the
user's approval for its platform/version cost; the HDF5 producer itself needs
only existing NumPy/h5py dependencies.

For the legacy oracle period, compare common-day population and every ordered
array against the read-only 891-day daily-front oracle. Do not ingest the
legacy HDF5/PT/NPZ as new build input and do not alter fresh data to force old
hashes. New source identities and fingerprints are expected; unexplained
numerical/population differences remain Phase 11 review evidence.

## `@version` closure

Every deterministic config and transformation uses `cdxcore.version.version`.
The Experiment producer closure must directly or transitively include:

- `ExperimentConfig` and all geometry/market/objective/split/storage configs;
- exact source schema/version eligibility and fitted-partition reader;
- daily-front active/European/horizon/0.2D/NMS predicates;
- ATM interpolation/NMS conversion, call/put fit conversion, quote ordering and
  ID/fingerprint functions;
- price-location deduplication and padding/fill rules;
- HDF5 schema, dtypes, chunk/compression writer and type-aware logical hashing;
- split/diagnostic-period construction;
- deterministic README/source/schema/split/validation document generators; and
- semantic and consumer adapters used as publication gates.

The reader/validator closure includes exact object enumeration, no-link checks,
manifest/config/source cross-checks, masks/counts/padding, all numerical
invariants and logical digests. Run workspace/journal functions are operational
and versioned where they affect resume correctness, but progress interval,
worker count and workspace retention never enter content identity.

Add a fresh-process regression which computes the planned producer/spec ID and
the worker/assembled ID independently. Changes to any content helper must alter
the resolved producer version; logging-only changes must not.

## Offline test matrix

1. Strict config parsing and canonical identity sensitivity for every content
   setting; worker/progress/resume controls are identity-neutral.
2. Source gate: exact one internal Dataset, supported schema/distribution and
   semantic validation; reject public candidate/public, multiple/missing
   sources and any implicit Raw/Intermediate access for the reference profile.
3. Synthetic filter edges: 0.2D excluded, 70D included, over-70 excluded, NMS
   -3/+2 included, just-outside excluded, American excluded, missing fit
   excluded, separate call/put multiplicity/order preserved.
4. Exact HDF5 object/attribute/path/dtype/shape/fill/chunk/filter contract;
   unknown group/dataset/link/object/reference rejected.
5. Count/mask/padding consistency and adversarial off-by-one/out-of-range quote,
   expiry and price-location indices.
6. NaN/+inf/-inf, nonpositive times/strikes/vols, crossed books, invalid bounds,
   unsorted dates/expiries/quotes, split overlap/gaps and content-digest
   mutations, including re-signed manifests.
7. Reader day round-trip and pinned `materialize_day` value/mask/shape/dtype
   parity on small fixtures; exact equal-date weighting retained.
8. Typed-empty omission and coverage/exclusion evidence; no silent omission of
   a populated day with zero eligible books.
9. Output-first current/wildcard reuse after source and workspace removal;
   wildcard miss cannot touch source or build; deep validation source failure
   is explicit.
10. Interrupt at shard write, journal append, HDF5 group creation, flush/fsync,
    semantic validation and final commit; no partial final cache hit.
11. Resume valid shards, reject/rebuild corrupt shards, plan mismatch, partial
    final journal tail and worker interruption/termination; immediate restart
    has no orphan writer.
12. Concurrent identical publishers, winning-artifact validation and different
    identities/workspaces isolation.
13. Text/JSON help, dry-run, success, missing h5py, usage/config/material
    failures, progress heartbeat, exact-ID/config validation and wildcard
    warning result.
14. No WRDS/network with `WRDS_OFF=1`; import isolation without h5py/Torch;
    source Dataset remains unmodified.
15. Legacy-oracle common-date counts/values/masks and the frozen facts (891
    dates, 3,566,258 books, geometry) as parity evidence, never as hard-coded
    acceptance for a fresh vendor revision.
16. Full offline pytest, Ruff, strict mypy for new modules, compileall,
    `git diff --check`, plus version-dependency assertions.

## User decisions and blockers

The internal reference input materialization has no new numerical decision:
its selection/geometry and all-history no-holdout semantics are already pinned.
It can be implemented while Phase 8 is blocked.

The following require explicit direction only if requested:

1. **Public-normalized Experiment:** blocked until CD-004/CD-009 and the Phase 8
   schema are approved/implemented. It cannot claim exact vendor-book or
   reference-consumer parity.
2. **Learned Direct-200 states/checkpoint:** outside the current Phase 9 plan.
   Expanding Phase 9 from input HDF5 materialization into GPU model fitting
   requires a separate config/artifact design and approval.
3. **Changing the reference split:** the pinned all-history producer has no
   holdout. A 2025 validation holdout is useful but is a different Experiment
   profile/identity and must not replace the parity default silently.
4. **Mandatory Torch dependency:** not needed for the materializer. Making the
   pinned Torch consumer test mandatory across ordinary installations requires
   a dependency/platform decision; keep it an explicit integration environment
   meanwhile.

## Acceptance sequence

1. Freeze this profile/config/HDF5 schema in docs and fixtures.
2. Implement strict config, pure daily projection and reader/validator first.
3. Prove small-fixture reference-consumer parity.
4. Add restart shards/journal and streaming HDF5 assembly.
5. Add output-first artifact publication and CLI.
6. Run offline/static/adversarial gates and independent code/quant reviews.
7. During Phase 11, materialize from the fresh validated internal Dataset and
   compare the common-period arrays/population to the legacy oracle before
   declaring its identity authoritative.

Phase 9 is complete when the standalone adapter feeds the reference consumer
with exact value/mask/shape parity, the artifact is self-contained and
output-first reusable without its source, interruption/concurrency tests leave
no partial final artifact, and a non-AI user can build/validate it from the
documented CLI.
