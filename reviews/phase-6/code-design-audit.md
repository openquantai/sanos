# Phase 6 code and architecture design audit

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Scope: pre-implementation design for daily filtering, no-arbitrage processing,
and SANOS/DLV fitting  
Network/WRDS access: none

## Recommendation

Implement Phase 6 as three chained immutable Intermediate stages:

1. `no-arb-day`: consume `enriched-day`, run the reference initial
   `MkArbFree(smoothness=None, no_arb_mode="surface")`, and align its masks,
   fits, Greeks, state, and rejection reasons to the daily frame;
2. `filtered-day`: apply the exact `OptionsFilter` sequence to the no-arbitrage
   eligible population, retaining a boolean for every predicate and a stable
   first-rejection reason; and
3. `fitted-day`: fit production smooth SANOS (`smoothness=0.25`) to the selected
   pure call book, retain separate real call/put books and active masks, and add
   the complete reconstructing model state and fit diagnostics.

Every stage still reads and writes one Polars frame for one day. Keep the
Phase 5 long-form row key and append columns; retain removed rows with false
eligibility/reason fields rather than losing the evidence. The selected subset
must exactly reproduce the legacy population at each boundary. Day- or
model-level typed structs/lists can repeat by row; Parquet compression is
preferable to a sibling state file or generic parameter blob.

Extend the existing `ArtifactStore`, `run_daily_plan`, and `sanos intermediate
build --through ...` path. Do not add Redun, HDF5, a second runner, or another
storage class. Output-first reuse, `version="*"`, atomic publication, journals,
spawn workers, and terminal pointer reporting are already suitable.

## Pinned source boundary

The authority is `quant_data` revision
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`. Adapt these exact sources; do not
copy their cache decorators, pandas persistence, CLIs, or `SubDir` ownership.

| Pinned path (Git blob) | Symbols/behavior to adapt |
|---|---|
| `quantdata/widb/arbfree.py` (`388929de...`) | `mk_arb_free_day`, aligned expiry/strike masks, Float64 solver ingress, Float32 fit/Greek egress, retained/dropped-expiry behavior |
| `quantdata/widb/options_filter.py` (`2ee6c3ac...`) | `OptionsFilter`, `_filter`, `normalize_day`, `filter_options_day`; exact predicate order, stable cap selection and minimum-per-expiry rule |
| `quantdata/dysanos/market.py` (`d8c4f550...`) | `_validated_smoothness`, `smooth_interpolated_market`; direct books in `C/(DF*F)`, final smooth fit, active/model masks, fit diagnostics |
| `quantdata/dysanos/flex/market.py` (`67f30c15...`) | `load_market_day` only as evidence for the canonical upstream profile: cap 100,000, 70 business days, non-binding NMS `[-100,100]` |
| `quantdata/dysanos/flex/config.py` (`d9d8b186...`) | `FlexMarketConfig` defaults only; experiment geometry remains Phase 8/9 |
| `tests/test_dysanos_market.py` (`942ccb8f...`) | synthetic two-expiry smooth payload, direct-book units, serialization/reconstruction expectations |
| `tests/test_arbfree.py` (`5545bb26...`) | expiry dropping, strike-mask alignment, boundary trimming, quote containment and surface invariants |
| `tests/test_flex_direct_state.py` (`9d884aa8...`) | selected market-population tests only; neural fitting is not Phase 6 |

The Phase 3 ports in `sanos.model.{arbfree,tools,fitter}` already own
`MkArbFree`, `ArbFreePricer`, `SanosPricer`, the LP solver and density/DLV
transforms. Reuse them; do not copy a second numerical core. The stale
`widb_sanos_day.fit_sanos_day`, old padded `widb.timeseries`, JAX optimizer
(CD-015), and vector market-fit diagnostic (CD-013) stay off this path.

## Modules and APIs

### Configuration

Keep `DailyWidbConfig` unchanged so adding downstream controls does not alter
the identity of already committed `paired-day` and `enriched-day` artifacts.
Add versioned strict types, preferably in `sanos.widb.config`:

```text
OptionsFilterConfig
  normalize=True, max_num_options=100000
  max_business_days_to_expiry=70, max_nms=(-100.0, 100.0)
  nms_use_atm=True, min_vega_sqrt_var=0.001  # accepted no-op, CD-012
  min_log_volume=1.0, min_log_open_interest=1.0
  remove_am=False, min_options_per_expiry=10

NoArbitrageConfig
  no_arb_mode="surface", smoothness=None, surface_expiry_linear=True
  min_half_spread=1e-4, mid_fit_weight=0.1, min_atm_vol=0.05
  min_dlv=None, max_dlv=None, min_strikes_per_expiry=5
  min_density=1e-4, min_dc_epsilon=1e-3

SmoothSanosConfig
  smoothness=0.25
  remaining numerical fields equal NoArbitrageConfig

DailySurfaceConfig
  no_arbitrage, options_filter, smooth_sanos, discrete_dlv_policy
```

The checked-in JSON should gain a separate top-level `surface` object. Stage
specs use config projections: initial no-arbitrage only for `no-arb-day`, option
filter only for `filtered-day`, and smooth/DLV choices only for `fitted-day`.
Upstream pointers already pin earlier configuration and content. Unknown keys,
booleans masquerading as numbers, non-finite values, invalid NMS intervals and
invalid caps must fail before planning.

Do not instantiate legacy classes by unpacking unchecked JSON. A versioned
adapter should construct `MkArbFree` explicitly from the reviewed fields.

### Pure daily transforms

Add these modules, independent of storage and CLI:

```text
sanos.widb.no_arb
  fit_initial_no_arbitrage(frame, config) -> pl.DataFrame
  initial_pricer_state(pricer) -> typed state value

sanos.widb.filtering
  apply_options_filter(frame, config) -> pl.DataFrame

sanos.widb.surface
  fit_smooth_sanos(frame, config) -> pl.DataFrame
  compute_direct_book_fields(frame) -> pl.DataFrame
  encode_pricer_state(pricer, ...) -> typed state value
  validate_reconstructing_state(...) -> None
```

For both LP calls, group selected rows by increasing `expiry_id`, stable-sort by
`normalized_strike`, pass `time_to_expiry_years`, strikes, bids and asks as
Float64 arrays, and map `input_index`/masks back by explicit source row index.
Never rely on a join's incidental order. Cast the legacy quote-aligned fit,
volatility and Greek outputs to Float32 at the same boundary as
`widb.arbfree.mk_arb_free_day`; keep model support/density/DLV state Float64.

The option-filter sequence is exactly: business-DTE horizon; normalization
contract; configured ATM or fixed NMS interval; volume; open interest; optional
AM removal; minimum population per expiry; then the stable per-expiry reserve
and closest-absolute-NMS cap. `min_vega_sqrt_var` remains identity-only and must
be labelled as not applied. With the production 100,000 cap it should be
non-binding, but the exact cap algorithm still needs tests.

Direct call/put books use the original cash quote divided exactly once by
`discount_factor * implied_forward`, yielding `C/(DF*F)`. Smooth inputs remain
the synthetic pure call bid/ask. `call_active` and `put_active` use the upstream
no-arbitrage/filter mask plus finite, nonnegative bid and strictly larger ask;
the final `smooth_model_mask` additionally intersects the retained smooth
expiry/strike mask. Do not collapse call and put observations.

### One-frame schema

Create ordered, versioned schemas `NO_ARB_SCHEMA`, `FILTERED_SCHEMA`, and
`FITTED_SCHEMA` as strict supersets of `ENRICHED_SCHEMA`. Recommended added
fields are:

- no-arbitrage: retained-expiry flag, `mask_arb_free`, first rejection reason,
  `fit1`/`fitX` price, volatility and Greeks, plus a typed initial-pricer state;
- filter: one sequential eligibility boolean per predicate,
  `options_filter_eligible`, and first rejection reason;
- fitted quotes: pure real call/put bid/ask, `call_active`, `put_active`,
  `smooth_model_mask`, smooth call/put fitted mids, availability, and fit error
  in half-spread units;
- fitted state: ordered expiries and `dt`, input indices, mode/boundary flags,
  smoothing input total variances, `xstrikes`/`xdensities`,
  `mstrikes`/`mdensities`, `xix_to_mix`, and the chosen unsmoothed density/DLV
  state with full-support boundary zeros; and
- diagnostics: deterministic row/expiry counts, all drop reasons, solver
  statuses/fallbacks/objectives/tolerances and invariant residuals.

Use typed `Struct` and nested `List` fields for reconstructing state, not JSON.
Existing Phase 5 JSON diagnostic columns may remain. Repeated daily state must
be identical on every row and validators must prove that. When a deterministic
population rule excludes the whole day, retain the enriched rows with false
eligibility and a reason; a Dataset assembler can later omit the day. An actual
solver error publishes no terminal artifact.

Do not persist wall-clock solve times in the canonical frame: they make a
content artifact nondeterministic. Put elapsed time in progress/journal events.
Persist deterministic solver status, fallback choice and objective. To obtain
structured per-solve evidence without changing numerics, add an optional local
diagnostic sink through `SanosLPSolver`/`MkArbFree`; prove sink-on and sink-off
produce identical arrays. Do not add diagnostic fields to
`ArbFreePricer.to_dict()`, which would break its frozen serialization contract.

### Artifacts, orchestration and CLI

Extend `sanos.widb.build` rather than creating a parallel builder:

```text
IntermediateStage += "no-arb-day" | "filtered-day" | "fitted-day"
no_arb_day_spec(enriched_pointer, config)
filtered_day_spec(no_arb_pointer, config)
fitted_day_spec(filtered_pointer, config)
*_producer_version()
```

Each stage has namespace `widb/<stage>/ticker-.../year-.../date-...`, one
`data.parquet`, canonical config, README, manifest and completion marker. The
worker first resolves the terminal shallowly; only a miss recurses through
prerequisites. Thus a completed fitted day remains usable after all earlier
Intermediates are deleted. `version="*"` remains terminal-reuse-only and cannot
build any missing stage.

Add the three values to `--through`; keep the current dry-run, text/JSON,
verbose and noninteractive contracts. Dry-run must show expected/reused
terminal pointers and must not import or access WRDS. Use the existing spawned
process runner, with default one LP day worker and a hard maximum of four to
match the recovered production guard. Parent progress heartbeats continue at
most every 30 seconds even while a solver is active. Worker exceptions should
be typed with date, stage and solver status so the durable journal is sufficient
to diagnose and restart; never convert a timeout/infeasible/error into a silent
day exclusion.

## Version closure

Use `@version` on every new config/schema/transform/adapter. Direct producer
closure must include the exact schema function, config class, Black
IV/Greek-kernel compatibility token, `MkArbFree`, `SanosPricer`/DLV transform,
density-floor policy, numerical tolerances, state encoder/validator and relevant
predicate/cap helpers. Artifact I/O closure includes semantic validation and
durable publication. Runner heartbeat/journal wording is operational and does
not belong in content versions.

Add a test that imports and resolves every `.version` object in the Phase 6
modules and asserts named critical dependencies in each stage producer. A
change to `overnight_variance_fraction` must already change the upstream
pointer and therefore every downstream identity; PM outputs remain unchanged.

## Offline test and acceptance matrix

Phase 6 is acceptable only when all of the following pass with `WRDS_OFF=1`:

1. An isolated pinned-reference fixture covers at least two expiries, one
   dropped/masked strike, a filtering rejection and both final retained/missing
   model masks. It matches ordered populations, masks, Float32 quote-aligned
   values and Float64 states within declared tolerances at smoothness `0.25`.
2. Unit tests independently cover every filter predicate, sequential
   first-rejection reason, expiry-population cascade, cap/tie stability, AM
   removal, and the CD-012 no-op.
3. Initial and smooth states satisfy positive ordered support, nonnegative
   density, unit mass/first moment, decreasing/convex calls, calendar
   monotonicity, quote-containment rules and serialization/reconstruction.
   DLV round-trip prices/densities and explicit boundary zeros are tested.
4. Direct call/put unit tests prove one and only one `DF*F` normalization and
   parity-derived smooth puts. `call_active`, `put_active`, pre-fit and final
   model masks are distinguished.
5. Sensitivity fixtures at overnight fractions `0.1`, `0.2`, `0.5` trace AM
   expiry order, NMS/filter decisions, input variances, `dt` and DLV state;
   default `0.2` matches the reference and PM values do not change.
6. Injected infeasible, timeout, fallback and unexpected solver outcomes have
   typed errors/diagnostics. No failed artifact is published. A deliberately
   interrupted multi-day fitted build restarts, reuses completed prerequisite
   stages and completes without duplicate publication.
7. Output-first and wildcard tests delete/corrupt upstream copies and verify
   the exact shallow/deep behavior. Dry-run is non-mutating; text and JSON CLI
   outputs report actual terminal identities and actionable failures.
8. Full offline pytest, Ruff, mypy, compileall and diff checks pass. Neither
   unit tests nor ordinary CLI paths connect to WRDS. Final independent code
   and quantitative reviews both record PASS.

## Decisions and blockers

1. **Unsmoothed DLV definition must be frozen before `FITTED_SCHEMA` v1.** The
   legacy smooth-market payload stores only the final `smoothness=0.25` pricer;
   its earlier unsmoothed no-arbitrage pricer is discarded before serialization.
   Three plausible states are materially different: the pre-filter initial
   no-arbitrage surface, DLVs inverted from the smooth density, or a new
   `smoothness=0` fit to the same post-filter book as SANOS. Recommend the last
   because it is genuinely nonsmooth and population-aligned, but it adds a
   solver not present in the authoritative producer. Quantitative review and
   user approval should freeze this before implementation of that state.
2. **CD-006 remains a cache-correctness blocker.** Timeout can select fallback
   or failure and therefore can affect content. Recommend including it in
   canonical fit configuration/identity and manifest despite the legacy hash
   omission. If exact cache-identity compatibility is preferred, freeze it at
   300 seconds with no override. Do not expose a configurable timeout excluded
   from identity.
3. Preserve CD-005's key `pcnt_fit` until its name is approved; add an exact
   synthetic inside/outside-count test and document its observed meaning. Do
   not silently relabel it. Preserve CD-011 Float32 boundaries, CD-012's no-op,
   CD-013's unavailable vector branch and CD-014 pricing behavior.
4. Structured solver reasons are not exposed by the current numerical return
   types. The optional diagnostic-sink change above is required to meet the
   Phase 6 inspectability contract, but it must have a no-output-change
   regression and complete version closure.

The no-arbitrage, filter, orchestration and smooth-`0.25` work can start after
this design is accepted. The two numbered semantic choices above must be
resolved before the final fitted schema/config is declared stable.

Signed: `code agent /root/code`
