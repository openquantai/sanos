# Phase 5 code and architecture design audit

Date: 2026-10-12
Reviewer: independent code agent (`/root/code`)
Scope: pre-implementation design for daily WIDB normalization and enrichment
Network/WRDS access: none

## Recommendation

Proceed with Phase 5 as two immutable per-day Intermediate stages:

1. `paired-day`: resolve the six exact Raw inputs for one date, reproduce the
   deterministic part of legacy `collect_raw_day`, and persist one typed Polars
   frame; and
2. `enriched-day`: consume one `paired-day` frame, reproduce legacy
   `enrich_day`, and persist the same frame with enrichment columns appended.

Use the existing `ArtifactStore` and `run_daily_plan`; do not add Redun. A
module-level spawn worker should build the requested terminal stage for one day,
building/reusing that day's prerequisite stage recursively. The existing atomic
artifact writer, shallow validation, immutable upstream identities, spawn
executor, durable journal, interruption cleanup, and progress callback are the
required orchestration primitives already.

There is no architectural blocker to starting the parity implementation. The
open issues in the final section block only the affected semantics: do not fix
them silently. In particular, CD-003 must remain open until the malformed-empty
input rule is approved. Valid known-bad days can still reproduce the observable
typed-empty result.

Four untracked Phase 5 files appeared while this audit was in progress:
`src/sanos/widb/config.py`, `src/sanos/widb/calendar.py`,
`src/sanos/widb/bad.py`, and `src/sanos/widb/daily.py`, together with edits to
`src/sanos/widb/__init__.py`. They were inspected as live design inputs. The
pre-review corrections for them are listed below; this audit did not edit them.

## Evidence and compatibility boundary

The pinned authority is revision
`de15be96d2f283b6c31e60d3ea47325a0c093ff0` under
`reference/quant_data/`. The production chain reaches
`widb.day.collect_raw_day` and then `widb.day.enrich_day`; it does not reach the
stale `widb_sanos_day` implementation or the old padded `widb.timeseries`
materializer.

Port/adapt these exact sources:

| Pinned source | Phase 5 use |
|---|---|
| `quantdata/biz_calendar.py` | NYSE dates, half days, business-DTE arithmetic, `DAYS_PER_YEAR`, `ONE_MINUTE`, `NormalizedStrikes`, `NMS`, and `NMS5` |
| `quantdata/widb/bad.py` | latest-valid guard and exact known-bad day/ticker/option sets |
| `quantdata/widb/day.py` | cleaning order, forward/discount attachment, call-put pairing, float32 boundary, implied-forward projection, pure normalization, IV/Greeks, and diagnostics |
| `quantdata/widb/client.py:480-507` | linear interpolation of rates in vendor percent and `exp(-rate * days / 36500)` |
| `tests/test_biz_calendar.py` | offline calendar/date arithmetic cases |
| `tests/test_dysanos_market.py` | relevant pure-unit/parity fixture ideas only; the smooth fit itself belongs to Phase 6 |
| `reviews/phase-1/reference-runtime-baseline.json` | frozen default values, schema groups, cast points, and versions |

Do not port the legacy connection/cache decorators, `SubDir.cache`, day module
CLI, `widb.timeseries`, or pandas schema utilities. Phase 4 owns SQL, guarded
WRDS access, Raw Parquet, and table resolution. `widb.options_filter` and
`widb.arbfree` belong to Phase 6.

The Raw-to-daily ingress necessarily reads several Raw table frames. Treat that
as a typed ingress adapter, not as a daily transformation stage. Its sole output
is one `paired-day` Polars frame. From that boundary onward, every transform
accepts and returns exactly one daily Polars frame.

## Recommended modules and APIs

### `sanos.widb.config`

Use versioned immutable configuration objects with strict parsers:

```text
ClockConfig
  overnight_variance_fraction: float = 0.2

RawDayConfig
  min_bid: float = 0.011
  drop_irregular_expiries: bool = True

ExpiryEnrichmentConfig
  num_min_either_traded: int = 10
  num_min_both_traded: int = 4
  min_pure_spread: float = 1e-6
  min_implied_vol: float = 0.01
  max_implied_vol: float = 5.0
  normalize: bool = False  # retained identity-only legacy field; see issue register

DailyWidbConfig
  clock: ClockConfig
  raw: RawDayConfig
  enrichment: ExpiryEnrichmentConfig
```

The overnight fraction should not be nested under raw quote cleaning. It is a
clock input consumed by this phase and downstream filtering, variance,
calendar/DLV, and experiment logic. A separate `ClockConfig` makes dependency
closure and later config projection explicit. Validate real numbers as finite,
reject booleans as numbers, validate actual boolean fields as `bool`, reject
unknown mapping keys, and use `@version` on every deterministic helper and
configuration class with complete direct dependencies.

Separate content configuration from the build request:

```text
IntermediateBuildRequest        # selection/operation, not daily content identity
  ticker, start_date, end_date, include_half_days, target_stage
  raw_source, raw_source_generation, max_workers, progress_interval

DailyWidbConfig                  # only output-affecting transform choices
```

Changing a requested date range or worker count must not change the identity of
an otherwise identical daily artifact. Ticker and trading date should appear in
the daily artifact configuration and namespace; stage identity uses only the
relevant config projection. Machine paths and credentials never enter either
object.

`overnight_variance_fraction` must be in the canonical config of both stages
whose output contains clock-derived fields and all downstream stages that use
them. At the default, compute the AM subtraction as exactly `1.0 - 0.2`; PM
values take a literal zero subtraction. Values `0.0` and `1.0` are valid.

### `sanos.widb.calendar`

Port the reference surface needed by the pipeline rather than introducing a
second partial calendar API:

```text
business_dates(start, end, *, exclude_half_days=False)
half_days(start, end)
count_business_days_between(start, end, *, exclude_half_days=False)
add_business_days(...), previous_business_day(...), next_business_day(...)
session_open_close(date)
NormalizedStrikes.to_nms/from_nms, NMS, NMS5
```

Retain scalar and vector start/end behavior and the `(start, end]` convention.
Calendar selection includes half days by default to match the recovered
production CLI. `is_half_day` is an independent daily field and must never be
derived from or stored as the expiry `is_am` field.

The live `calendar.py` is not yet a complete port: it only accepts a scalar
start in `count_business_days_between`, omits the reference half-day toggle and
navigation/session functions, and drops the reference `backbone` hook from NMS.
That is acceptable only as an unfinished file, not as Phase 5 acceptance.

### `sanos.widb.schema`

Define exact ordered Polars schemas and validators for typed empty and populated
frames:

```text
PAIRED_DAY_SCHEMA_VERSION = "widb-paired-day-v1"
ENRICHED_DAY_SCHEMA_VERSION = "widb-enriched-day-v1"
paired_day_schema() -> pl.Schema
enriched_day_schema() -> pl.Schema
empty_paired_day() / empty_enriched_day()
validate_paired_day(frame, ticker, date)
validate_enriched_day(frame, ticker, date)
```

Use the modern names already declared in `docs/data-contracts.md` and one
explicit mapping from legacy names; do not carry two synonymous columns merely
for compatibility. Pin column order, nullability policy, `Date` fields, Boolean
flags, Int32 IDs, and every legacy Float32 cast boundary. Solvers may promote
selected arrays to Float64 at the same points as the reference.

The data-contract field `raw_artifact_id` is ambiguous because a paired day has
six direct Raw inputs. Resolve this before freezing schema v1. Recommended:
keep the complete ordered upstream pointers in the artifact manifest and, if
row-level provenance is required, name the options pointer explicitly (for
example `raw_options_artifact_id`) rather than implying that one ID covers the
day. Do not repeat all six IDs in every quote row without a demonstrated use.

The frame should carry compact typed stage diagnostics (for example a
`List(Struct)` column repeated as a per-day constant) when rows exist. Empty-day
reason/count diagnostics must also be present in manifest metadata because a
zero-row frame cannot carry a value. Do not create sibling quote, expiry,
surface, or diagnostics tables.

### `sanos.widb.raw_inputs`

Provide a typed ingress object and deterministic pointer planning:

```text
RawDayPointers
  security, options, current_spot, previous_spot, forwards, zero_curve

RawDayInputs
  pointers plus six validated Polars frames

plan_raw_day_pointers(ticker, date, source, source_generation,
                      raw_producer_version) -> RawDayPointers
load_raw_day_inputs(store, pointers) -> RawDayInputs
```

Pointer planning must be pure and must not require the files to exist. Use
Phase 4 `RawPartition` and `raw_artifact_request`; do not duplicate raw namespace
or identity construction. This lets a completion probe compute and shallowly
validate a finished Intermediate before touching upstream files. Only a cache
miss loads and validates Raw artifacts.

The exact direct inputs are security identity, current options, current and
previous-session spot, current forwards, and current zero curve. The extra
previous-session forward/zero partitions conservatively downloaded by Phase 4
for the still-unapproved public return are not Phase 5 inputs. A missing direct
input produces an actionable cache-only error naming the exact Raw pointer and
the corresponding `sanos raw download` command; Phase 5 never connects to WRDS.

Expose or centralize Phase 4's safe ticker-label conversion. Raw ticker syntax
permits `^`, while artifact namespace and `DailyWorkItem.ticker` syntax do not.
SPX is unaffected, but a generic Phase 5 CLI must not silently accept a Raw
ticker that the runner later rejects.

### `sanos.widb.day`

Keep pure transformation APIs independent of storage and CLI:

```text
interpolate_zero_curve(zero_curve, calendar_dtes) -> np.ndarray
assemble_paired_day(inputs, *, ticker, date, config) -> pl.DataFrame
enrich_day(frame, *, ticker, date, config) -> pl.DataFrame
```

`assemble_paired_day` reproduces this exact population order:

1. apply latest-valid and known-bad rules;
2. exclude pinned bad symbols;
3. remove same-day and configured irregular expiries;
4. remove null books;
5. remove `bid >= ask` books;
6. remove bids `<= min_bid`;
7. remove `ss_flag != "0"` rows;
8. interpolate discount factors in ACT/365 using percent rates;
9. attach the vendor forward by expiry and AM flag, including the legacy
   `spot / 1_000_000` repair and suspicious-forward guard;
10. add log liquidity, the synthetic European-style field, scale raw integer
    strike by 1,000, pair calls and puts, derive expiry ID, business/calendar
    DTE and TTE, add session half-day, and cast at the frozen float32 boundary.

`enrich_day` reproduces the inverse-spread-weighted `DF*F` projection, synthetic
minimal-spread call, `K/F`, `C/(DF*F)`, pre/post thresholds, implied vols,
Greeks, NMS5, Float32 output arrays, and the final valid population. Never send
Polars frames through the process pool; workers load and write them locally.

Use NumPy arrays for the numerical kernel where the reference does so and Polars
for joins, schema, and persistence. Stable ordering and tie behavior must be
tested against the pinned pandas result before selecting a Polars join/sort
strategy. Do not replace assertions that guard numerical invariants with
unchecked behavior; convert data/input failures to typed actionable exceptions
at the API boundary.

### `sanos.widb.artifacts`

Provide stage-specific identity and I/O, not a second generic store:

```text
paired_day_spec(config, day, raw_pointers) -> ArtifactSpec
enriched_day_spec(config, day, paired_pointer) -> ArtifactSpec
read_daily_frame(store, pointer, expected_schema, *, semantic=True)
publish_daily_frame(store, spec, frame, diagnostics) -> ValidatedArtifact
paired_day_producer_version() / enriched_day_producer_version()
```

Recommended namespaces are
`widb/paired-day/ticker-SPX/year-YYYY/date-YYYY-MM-DD` and
`widb/enriched-day/ticker-SPX/year-YYYY/date-YYYY-MM-DD`. Each immutable
directory contains exactly one `data.parquet` data payload plus the store's
manifest/completion files. A small `README.md` is acceptable; diagnostics belong
in the frame and manifest metadata, not another tabular artifact.

Write Parquet in the builder's private staging directory, fsync the file, and
let `ArtifactBuilder` atomically publish the directory. Reuse the durable writer
pattern in Phase 4 or move it to a common artifact helper; do not call private
Phase 4 functions across package boundaries. On an `ArtifactExistsError` race,
validate and reuse the immutable winner exactly as the Raw layer does.

Producer versions should include only code that can change content. Operational
journal/progress code is not a content dependency. The paired producer depends
on schemas, config, calendar, known-bad policy, raw normalization contracts,
zero interpolation, and the paired transform. The enriched producer depends on
its schema/config and enrichment/numerical helpers; the paired artifact pointer
already pins upstream content. Add direct version-closure regression tests as in
Phases 3 and 4.

### `sanos.widb.build`

Implement module-level, spawn-picklable functions:

```text
plan_intermediate_build(request, config) -> DailyPlan
probe_daily_target(item) -> ArtifactPointer | None
build_daily_target(item) -> ArtifactPointer
build_intermediates(...) -> DailyRunSummary
```

The canonical item payload contains resolved storage roots, actual ticker, raw
source/generation and the strict daily config. It contains no credentials. For
an `enriched-day` item, the worker:

1. computes the terminal spec without requiring upstream existence;
2. shallowly validates/reuses the terminal artifact if present;
3. otherwise validates/reuses or builds `paired-day` for that same date;
4. reads one paired Polars frame, enriches it, validates it, and publishes the
   terminal artifact; and
5. returns only the terminal `ArtifactPointer`.

An interruption after publishing `paired-day` but before `enriched-day` is safe:
the next run reuses the prerequisite. The parent alone writes the journal. Put
the default mutable journal under
`QUANT_DATA_INTERMEDIATE/.runs/widb/<plan-id>.jsonl`, not in Git and not inside
an immutable artifact directory.

The existing plan identity intentionally excludes worker/progress tuning. The
item producer version and canonical payload include every content-affecting
choice. A source/code/config change therefore changes the expected pointer even
when the operational date-range plan is otherwise the same.

`version="*"` is reuse-only: resolve a complete compatible terminal artifact,
warn visibly, record the override in result/journal, and fail on any miss. It
must never enter the worker or build a missing prerequisite. Do not add an
in-place refresh/rebuild option for immutable Intermediate identities.

## CLI contract

Replace the stub with:

```text
sanos intermediate build \
  --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2022-01-03 --end-date 2025-08-29 \
  --through enriched-day --max-workers 4 \
  [--include-half-days/--exclude-half-days] [--version current|*]
```

Defaults should include half days to reproduce the recovered production path.
`--config` is strict and portable. Explicit CLI ticker/date/selection overrides
are permitted, but the command must emit the complete effective configuration
and its ID. `--dry-run` computes sessions, direct Raw pointers, expected stage
IDs, hits/misses, boundary previous spots, journal path, and work count without
creating roots. It never imports `wrds`, asks for 2FA, or opens a connection.

Text output reports the final stage, plan/config IDs, total/reused/built/failed,
elapsed/ETA, journal, and each terminal artifact path. JSON mode emits the same
stable fields as line-delimited events. Worker failure text must identify
ticker/date/stage and preserve the exception category; no direct `print()` from
numerical code may corrupt JSON output. Existing global `-v`, `--dry-run`, and
`--log-format` behavior should be reused.

`sanos status` should report Phase 5 availability only after all acceptance
gates pass. `sanos doctor` remains offline; Polars/calendar dependencies are
already core.

## Test and acceptance matrix

All ordinary tests run with `WRDS_OFF=1`; add a socket/connection-factory trap
around Phase 5 CLI and worker integration tests.

| Test module | Required coverage |
|---|---|
| `test_widb_config.py` | strict unknown/type/nonfinite rejection; defaults; fraction endpoints; canonical JSON/ID; `0.1/0.2/0.5` identities differ; execution/date-range settings do not alter daily content ID |
| `test_widb_calendar.py` | imported scalar/vector cases, holiday and early-close cases, navigation, session indicator, NMS round trip, AM formula and PM invariance |
| `test_widb_raw_inputs.py` | exact six pointers, previous-session closure, missing/corrupt raw error, no upstream read on terminal cache hit, source-generation and producer-version identity |
| `test_widb_day_pairing.py` | strike `/1000`, exact cleaning order and threshold edges, bad symbols/irregular dates, forward key/repair/duplicate behavior, zero curve percent/ACT365, call-put pairing/order/dtypes, day/expiry invariants |
| `test_widb_day_enrichment.py` | exact `DF*F` bid/ask interval and inverse-spread projection, synthetic call selection, cash-to-pure conversion, `c-p=1-k`, threshold order, boundary IV failures, Greeks/NMS, Float32 cast points, typed empty behavior |
| `test_widb_artifacts.py` | exact identity/upstreams, atomic failure, racing publisher, Parquet schema/semantic validation, corrupted payload, empty marker, wildcard hit/miss |
| `test_widb_build.py` | spawn execution, per-day prerequisite reuse, interruption after prerequisite publication, restart, partial failures, no frame returned through worker boundary, journal/progress, no network |
| `test_cli_intermediate.py` | help, dry run no writes, text/JSON results and errors, config overrides, half-day selection, exit codes, wildcard reuse-only, actionable Raw misses |
| `test_widb_reference_parity.py` | synthetic pinned-reference oracle: ordered columns/rows/dtypes, exact default BDTE/TTE and population, numerical enrichment tolerances |

Generate the parity oracle from synthetic, redistribution-safe raw tables using
the pinned pandas implementation in an isolated explicit reference-generation
command. Check in only small expected values/dtypes/counts, not proprietary data
or cache artifacts. Ordinary tests consume the frozen fixture and do not import
legacy cache machinery or require pandas/WRDS. Independent quantitative review
must approve fixture construction and tolerances.

Minimum numerical assertions:

- default AM BDTE/TTE is bit-identical where the same Float32 cast is expected;
- changing fraction from `f1` to `f2` moves AM business DTE by exactly
  `f2 - f1` before the frozen cast and within the declared Float32 tolerance
  after it; PM is unchanged;
- zero rates remain percent, interpolation is linear in rate, discounting is
  continuous ACT/365, and every discount factor is finite/positive;
- the projected discounted forward lies in the derived parity interval when it
  is feasible and matches the inverse-spread-weighted reference calculation;
- `pure_strike=K/F`, `pure_price=C/(DF*F)`, and pure put-call parity hold at the
  declared tolerance; and
- row/expiry counts, order, and all diagnostic reason counts match the oracle.

The Phase 5 sensitivity report should use a fixed synthetic multi-expiry day and
record, for `0.1`, `0.2`, and `0.5`, expiry order, AM/PM DTE, NMS and immediate
eligibility, total/forward variance inputs, calendar time differences, DLV time
increments, and configured horizon masks. Phase 3 numerical utilities may be
used for the clock-derived calculations. Actual no-arbitrage populations, daily
SANOS fits, and experiment materializations must be rechecked in Phases 6 and 9;
Phase 5 must not claim those later integration paths have run. This is how the
cross-phase acceptance text remains traceable without fabricating unavailable
evidence.

Static gates remain:

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
ruff check src tests scripts
python -m mypy src tests scripts
python -m compileall -q src tests scripts
git diff --check
```

Both independent gates are mandatory: the code reviewer checks architecture,
identity, atomicity, restart, CLI, typing and offline isolation; the quant
reviewer checks source fidelity, units, filter/order/population parity, cast
points, clock propagation and numerical tolerances.

## Compatibility and defect register

These findings must be recorded before implementation chooses behavior:

1. **CD-003 remains blocking for malformed empty input validation.** Legacy
   `enrich_day` returns a typed empty frame before its schema check. Preserve the
   typed-empty observable result for valid known-bad input; do not move the
   malformed-input check ahead of the return until approved.

2. **The latest-valid guard is not a normal known-bad empty day.** Legacy dates
   on or after `2025-09-01` always raise `ValueError`, even when
   `raise_if_bad_day=False`; only `BAD.DAYS_ALL`/`BAD.DAYS_TICKER` may become a
   typed empty. The live `bad_day_reason()` currently represents all three as a
   reason string, so callers must not map every reason to the same empty result.

3. **Duplicate spot rows are underspecified.** Legacy code uses the first row of
   an unordered query. Before changing this to an exact-one or unique-value
   validation, register the defect and obtain approval. Until then, pin the
   behavior on synthetic duplicates and fail rather than invent a deterministic
   value if parity cannot be established.

4. **Duplicate call/put pairing keys are underspecified.** The legacy inner merge
   can create a Cartesian product for duplicate
   `(expiry, AM, style, strike)` keys and its tie order inherits pandas/source
   ordering. Do not silently deduplicate or select a contract. Add a duplicate
   synthetic oracle; a new uniqueness rule needs approval.

5. **`ExpiryFilter.normalize` is an identity-only no-op in `enrich_day`.** It is
   validated/versioned/stringified but never read by the transform. Preserve it
   as an explicitly documented compatibility field for initial parity, or add a
   compatibility decision before removing/activating it.

6. **Reference vector calendar navigation drops `count_start` in recursive array
   calls.** This does not affect the scalar daily production path. Do not copy or
   repair that behavior silently if the standalone API exposes vector
   `add_business_days(..., count_start=True)`; register a decision or omit the
   unsupported combination initially.

7. **CD-004 is outside Phase 5 output.** Preserve the internal legacy simple
   close return as `spot_return = S_t/S_previous - 1` only when the later
   normalization stage is ported. Do not compute or label a
   `forward_adjusted_spot_return` here; its formula remains approval-gated.

8. **CD-011 requires the frozen Float32 boundary.** A Float64-through-enrichment
   refactor is not a cleanup. It needs population/LP evidence, a new version, and
   approval.

9. **Eligibility-mask wording and reference row removal need reconciliation.**
   The reference physically drops rows during collect/enrich, while the current
   data contract asks for every population predicate as a named mask before
   publication. Initial Phase 5 parity should retain the exact surviving rows
   and complete per-stage reason/count diagnostics; retaining rejected vendor
   rows in the canonical paired frame would change the frame population and
   requires an approved schema/semantic decision.

## Immediate observations on the live untracked files

Before their first implementation review:

- move `overnight_variance_fraction` from `RawDayConfig` to a reusable
  `ClockConfig` or prove an equally explicit config projection into every later
  consumer;
- version `_finite` and declare it as a configuration dependency, or replace it
  with an already-versioned validation helper;
- strictly validate `drop_irregular_expiries` and `normalize` as booleans;
- complete scalar/vector and half-day calendar parity rather than freezing the
  current partial API;
- preserve the fatal latest-valid cutoff separately from typed known-bad days;
  and
- ensure producer closures depend on the known-bad/irregular data definitions,
  not merely the public lookup functions by convention;
- preserve the pinned zero-BDTE guard: the reference permits zero only for the
  historical Saturday AM case and asserts its date/settlement shape; the live
  `daily.py` currently accepts every zero-BDTE case;
- retain the implied-vol inversion invariant: the reference proves every large
  inversion error is caused by a configured volatility bound before dropping
  that row, while the live `daily.py` drops every such error without checking
  the cause. That is a silent semantic relaxation and must be restored or
  separately approved;
- after inversion filtering, preserve the reference nonempty and per-expiry
  validity assertions rather than allowing all rows to fall through to an
  unconditional typed-empty result;
- add the final per-day/per-expiry constant-field invariants present in both
  reference stages;
- rename the current `normalized` stage to `paired-day` (or another unambiguous
  pre-enrichment name). It is not legacy `normalize_day`, which is a later
  spot-normalization/filter operation, and the current name will collide with
  Phase 6 terminology; and
- complete the `@version` graphs in `daily.py`: the current decorators omit
  output-relevant local helpers, canonical serialization/hash helpers, schema
  definitions/validators, and the Black-Scholes implementation boundary.

No live WRDS pilot is needed for Phase 5 unit/integration acceptance. The later
fresh end-to-end pilot remains separately 2FA-gated and bounded by the Phase 4
lease policy.
