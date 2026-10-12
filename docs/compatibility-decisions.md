# Compatibility decision register

The pinned source revision is
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`. A confirmed defect is not an
authorization to fix it. The affected migration remains compatibility-first
until code review, quantitative review, and explicit user approval establish a
different rule. Unaffected phases may proceed.

Status values are `open`, `provisional`, and `approved`. `Provisional` means the
evidence supports the proposed treatment but the current phase reviews have not
both accepted it. Only `approved` may alter reference behavior.

## CD-001 — Stale `widb_sanos_day` path

- **Status:** provisional; no output behavior is being changed.
- **Evidence:** `sanos.widb_sanos_day.fit_sanos_day` groups from an empty expiry
  list and references stale fields. The recovered production call graph is
  `dysanos.download -> flex.market.load_market_day ->
  smooth_interpolated_market -> filter_options_day`; no call reaches this
  function.
- **Proposed treatment:** keep it as provenance only and port
  `smooth_interpolated_market` plus the actual downstream adapter.
- **Output impact:** none for the legacy 2022–2025 migration-parity path.

## CD-002 — Option column-set assertion is reversed

- **Status:** open; confirmed legacy defect; blocks the migrated live-query
  schema check.
- **Evidence:** `WIDBClient.options` asserts
  `set(df.columns) <= OPTION_COLS` while its error describes missing required
  columns. A response missing a required column can therefore pass; an
  unexpected column fails.
- **Compatibility choice:** either reproduce the permissive check and add a
  separate non-behavior-changing diagnostic, or require
  `OPTION_COLS <= set(df.columns)` in the new raw layer.
- **Expected impact if fixed:** malformed or vendor-changed query results fail
  at ingestion rather than later; valid reference extracts are unchanged.

## CD-003 — Empty enriched-day validation is unreachable

- **Status:** open; confirmed legacy defect; blocks only empty-day validation.
- **Evidence:** `enrich_day` returns `create_empty_df(SCHEMA_ENRICHED_DAY)` and
  then has an unreachable `verify_schema(data, SCHEMA_RAW_DAY, ...)` statement.
- **Compatibility choice:** retain the observable typed-empty result while
  moving validation before return, or preserve the absence of validation.
- **Expected impact if fixed:** only malformed empty upstream frames fail
  earlier; known-bad typed empty days retain the same output.

## CD-004 — Public forward-adjusted spot return

- **Status:** open for explicit user approval; legacy behavior is frozen.
- **Evidence:** legacy `normalize_day` records `spot_ret = S_t/S_previous - 1`.
  It does not carry-adjust the return and must not be relabelled.
- **Proposed public contract:** for consecutive retained observations, compute
  the previous-date carry
  `c_t = F_(t-1)(t)/S_(t-1)`, its discount factor, and
  `forward_adjusted_spot_return = (S_t/S_(t-1))/c_t - 1`. Interpolate positive
  forward and discount curves log-linearly in business DTE with a DTE-zero
  anchor of one. Publish only the resulting return; the first retained date is
  null. Record the formula and required boundary inputs in configuration and
  provenance rather than adding carry fields to the public allowlist.
- **Expected impact:** adds a distinct public field and fresh raw boundary
  dependencies; it does not replace internal `spot_ret`.

## CD-005 — Fit-percentage label ambiguity

- **Status:** open; blocks renaming/interpreting the relevant diagnostic, not
  numerical fitting.
- **Evidence:** legacy documentation/log wording is inconsistent about whether
  the fraction is inside or outside the bid/ask spread.
- **Required evidence:** evaluate the implementation predicate and pin a
  synthetic book with known inside/outside counts before selecting a new field
  name.
- **Expected impact:** diagnostic schema/labels only unless the predicate itself
  is subsequently found wrong.

## CD-006 — Solver timeout omitted from legacy cache identity

- **Status:** open; blocks deciding whether timeout changes new artifact
  identity.
- **Evidence:** the solver timeout can change whether a fit finishes or falls
  back, but it is excluded from the current cache key.
- **Proposed treatment:** include it in canonical execution configuration and
  manifest provenance. Decide after parity runs whether it is content identity
  or operational metadata.
- **Expected impact:** possibly new artifact identities for otherwise identical
  inputs; successful deterministic outputs should remain equal.
- **Current compatibility behavior:** Phase 6 fixes the reference timeout at
  300 seconds and does not expose a timeout override. It records that fixed
  value in solver diagnostics, while artifact content identity remains driven
  by the versioned implementation. A configurable timeout remains blocked.

## CD-007 — Exact option cap and population filters

- **Status:** provisional; recovered from the production path and reference
  report.
- **Evidence:** upstream `load_market_day` uses `max_num_options=100000`, a
  deliberately non-binding `max_nms=(-100,100)`, smoothness `0.25`, and a
  70-business-day canonical horizon. The daily-front report then applies
  European-style, horizon, explicit `0.2`-DTE exclusion and AF-ATM canonical
  NMS `[-3,2]`, yielding 3,566,258 books.
- **Proposed treatment:** preserve this exact configuration as the daily-front
  parity profile; do not impose its 0.2D exclusion on the complete internal
  dataset.
- **Output impact:** none when the recovered profile is used.

## CD-008 — Overnight/AM clock conventions

- **Status:** provisional for the WIDB pipeline; FLEX interpretation remains a
  separately named geometry input.
- **Evidence:** `AM_EXIRY_SUBTRACT=0.8`, so next-session AM expiry is 0.2
  business day. The daily-front model report also records
  `first_model_business_dte=4.8` alongside explicit nodes beginning at 1.0;
  these are not evidence that WIDB's daily clock should change.
- **Proposed treatment:** introduce
  `overnight_variance_fraction: float = 0.2`, use
  `1 - overnight_variance_fraction` as the WIDB AM subtraction, include it in
  identity, and preserve the distinct FLEX geometry fields verbatim.
- **Output impact:** default 0.2 must reproduce legacy dates, ordering, masks,
  interpolation and DLV increments; non-defaults intentionally define new
  dataset identities and must not alter PM expiries.

## CD-009 — Public spread/model-price unit bridge

- **Status:** open for explicit user approval; the dimensionally consistent
  proposal is frozen for review.
- **Evidence:** legacy pure model prices use `C/(DF*F)`, while the requested
  public spreads use `half_spread/S`. They cannot be compared without carry.
- **Proposed public contract:** retain fitted prices in discounted-forward units
  and half-spreads in spot-one units, plus per-expiry
  `discounted_forward_over_spot = DF*(F/S)`. Then
  `C/S = discounted_forward_over_spot * C/(DF*F)` and
  `h/(DF*F) = (h/S)/discounted_forward_over_spot`.
- **Expected impact:** adds dimensionless interpreting geometry without an
  absolute spot field or reconstructable vendor price level.

## CD-010 — Explicit `secid` argument is ignored

- **Status:** open; confirmed legacy defect; blocks migrated ticker/security
  override semantics.
- **Evidence:** `collect_raw_day(..., secid: int | None = None)` immediately
  executes `secid = conn.lookup_secid(ticker)` and therefore ignores every
  supplied value.
- **Compatibility choice:** remove the misleading argument, or honor and
  validate it against ticker identity.
- **Expected impact if fixed:** callers using an explicit security ID may query
  a different identity than legacy code; ordinary ticker-only SPX reproduction
  is unchanged.

## CD-011 — Float32 daily boundary versus float64 refactor

- **Status:** open measurement item; the first implementation preserves legacy
  cast points.
- **Evidence:** raw/enriched/no-arbitrage schemas and many constructed arrays
  explicitly cast prices, forwards, times and Greeks to float32, while solvers
  often convert back to float64.
- **Compatibility choice:** preserve those casts through the initial parity
  build. A future float64-through-fit dataset is a semantic version change
  unless measured results prove exact acceptance equivalence.
- **Expected impact if changed:** small changes can move NMS boundaries,
  eligibility masks, LP optima and SANOS/DLV values.

## CD-012 — `min_vegaSqrtVar` is an identity-only no-op

- **Status:** open; confirmed legacy ambiguity; initial parity preserves the
  no-op and does not describe it as an applied filter.
- **Evidence:** `OptionsFilter.min_vegaSqrtVar` defaults to `0.001`, is validated,
  printed by `__str__`, and therefore participates in the legacy cache label,
  but `OptionsFilter._filter` never reads it. The implemented filter order goes
  from NMS directly to volume and open interest.
- **Compatibility choice:** preserve it as an accepted/versioned compatibility
  field with no population effect, or implement the documented
  `vega*sqrt(variance)` threshold as a new semantic version.
- **Expected impact if activated:** option populations, per-expiry eligibility,
  cap selection, no-arbitrage inputs and fitted SANOS/DLV values can change.

## CD-013 — Vector market-fit diagnostics reference undefined variables

- **Status:** open; confirmed non-production-path defect; blocks silently
  copying or repairing the vector diagnostic branch.
- **Evidence:** in the pinned
  `quantdata.sanos.sanos_tools._compute_market_fit`, the branch used when
  `strikes` is one NumPy array computes `fit_err` from undefined names `f`,
  `asks_t`, and `bids_t`. The list-of-expiries branch uses its defined local
  arrays and is the branch exercised by `smooth_interpolated_market` when it
  generated the authoritative daily `fit_result` values.
- **Compatibility choice:** either omit the unexercised vector interface from
  the first standalone public API, or repair it to use `r.fits`, `asks`, and
  `bids` with a focused regression test and a new version. Do not reproduce the
  runtime `NameError` accidentally and do not claim a repair is reference
  behavior.
- **Expected impact if fixed:** no change to the authoritative daily market
  collection; callers of the legacy vector diagnostic interface change from a
  failure to a result.

## CD-014 — Pre-first-expiry pricing jumps to the first fitted density

- **Status:** provisional parity behavior; changing it is open and blocked.
- **Evidence:** after the intrinsic branch at `ONE_MINUTE`, both
  `SanosPricer.__call__` and `ArbFreePricer.price` select density zero for every
  `0 < t < T[0]`. They do not interpolate from the time-zero Dirac mass. At an
  exact model expiry they use that expiry's density; between expiries they
  linearly interpolate call prices; after the last expiry they keep the final
  density. Surface mode uses `mstrikes`/`mdensities`, while expiry mode uses
  `xstrikes`/`xdensities`.
- **Proposed treatment:** preserve the jump and the complete legacy pricing
  rule in the initial parity implementation and public reconstruction tests.
  An interpolation from the Dirac mass would require an explicit new model
  version and user approval.
- **Expected impact if changed:** reconstructed short-expiry prices and implied
  volatilities change for every time before the first fitted expiry; downstream
  objectives or experiment horizons that query those times may also change.

## CD-015 — Optional JAX fitter does not return its optimized state

- **Status:** open; confirmed reference defects; the authoritative LP path is
  unaffected.
- **Evidence:** the pinned `quantdata.sanos.sanos_jax.fit_dlvsqs` initializes
  `jself.weights = None` and then calls `jself.weights.append(...)` when weights
  are supplied. More importantly, after computing `result_raw` with L-BFGS-B,
  it calls `jself.value_and_data(dlvsraw)` rather than
  `jself.value_and_data(result_raw)`, so a nominally successful optimization
  returns the initial state. The authoritative 891-day smooth-market producer
  uses the CVXPY/HiGHS LP fitter, not this optional function.
- **Compatibility choice:** retain JAX transition/value/gradient parity tests
  while leaving the full optimizer entry point unavailable, or repair both
  defects under a new function version and test the result against the LP path
  on deterministic markets.
- **Expected impact if fixed:** callers of the optional JAX optimizer receive
  optimized densities and prices instead of failure or the initial state; the
  authoritative LP-produced dataset is unchanged.

## CD-016 — Duplicate spot rows select the first observation

- **Status:** open; initial migration preserves the observable behavior.
- **Evidence:** `collect_raw_day` obtains the spot and previous spot through a
  helper that selects the first returned close. It does not require uniqueness
  or declare an ordering rule for a malformed duplicate response.
- **Compatibility choice:** the paired-day transform selects row zero and has a
  regression test. A future ingestion rule may reject duplicate `(secid,date)`
  closes, but that is a versioned semantic change requiring approval.
- **Expected impact if changed:** only malformed or vendor-changed partitions;
  valid reference partitions should remain unchanged.

## CD-017 — Duplicate option keys create Cartesian call/put pairs

- **Status:** open; initial migration preserves the observable behavior.
- **Evidence:** the reference performs an inner merge on expiry, AM/PM,
  strike, and exercise style without checking uniqueness or choosing a
  contract. Duplicate calls or puts therefore multiply rows.
- **Compatibility choice:** the Polars paired-day join retains the same
  Cartesian population, pinned by an offline test. Rejecting duplicates or
  introducing a tie-break requires a new version and measured reference impact.
- **Expected impact if changed:** row populations, liquidity counts, implied
  forwards, eligibility, and every downstream SANOS/DLV fit can change.

## CD-018 — `ExpiryFilter.normalize` is an identity-only no-op

- **Status:** open; initial migration retains the field without applying it.
- **Evidence:** the reference field is validated and appears in the readable
  cache label, but `enrich_day` never reads it. Numerically identical enriched
  outputs for `False` and `True` are now regression-tested.
- **Compatibility choice:** preserve the no-op in Phase 5. Remove it from the
  future public configuration or implement a meaning only after explicit
  approval and a new producer/configuration version.
- **Expected impact if activated:** units and potentially all downstream filter
  and fit values could change.

## CD-019 — Vector `add_business_days` drops `count_start`

- **Status:** open; the standalone Phase 5 API exposes the reviewed scalar
  operation only.
- **Evidence:** the reference's array branches recursively call the scalar
  function without forwarding `count_start`, so array and scalar results differ
  when that flag is true. The daily WIDB path needs only scalar navigation.
- **Compatibility choice:** do not advertise the defective vector navigation
  surface. Scalar behavior and vector start/end business-day counting are
  supported and parity-tested. Adding vector navigation requires an approved
  contract rather than silently copying or fixing the defect.
- **Expected impact if fixed:** only callers of vector navigation with
  `count_start=True`; the authoritative daily dataset is unaffected.

## CD-020 — All-valid enrichment retains its control column

- **Status:** open; initial migration preserves the observable failure.
- **Evidence:** the pinned `enrich_day` adds an internal `is_valid` column and
  drops it only when at least one row is invalid. An expiry for which every row
  survives therefore reaches schema validation with the extra column and fails
  with `has extra columns is_valid`. This was reproduced against revision
  `de15be96d2f283b6c31e60d3ea47325a0c093ff0` using an offline 14-strike
  synthetic book. The behavior appears accidental because the downstream time
  series explicitly requires enriched daily frames not to contain `is_valid`.
- **Proposed treatment:** always treat `is_valid` as an internal control mask,
  remove it before validating the enriched schema, and return the otherwise
  valid rows. The standalone Polars implementation naturally has this behavior
  because its mask is never added as a data column.
- **Expected impact if fixed:** all-valid synthetic or real days succeed rather
  than raising a schema error. Days with one or more rejected rows and all
  authoritative days that completed in the legacy producer are numerically
  unchanged. The fresh-build report must count any real all-valid days so the
  population impact is explicit.
- **Current compatibility behavior:** until the repair is explicitly approved,
  `enrich_daily_frame` raises a targeted `DailyDataError` when every input row
  survives. This preserves the reference's failure boundary without exposing
  its temporary control column in the Polars schema.

## CD-021 — Meaning of the unsmoothed DLV state

- **Status:** provisional compatibility interpretation; adding a new refit is
  open and blocked.
- **Evidence:** the production path first constructs a linear surface with
  `MkArbFree(smoothness=None)`, then applies `OptionsFilter`, and finally fits a
  smooth `0.25` SANOS surface. The legacy daily payload serializes only the
  final smooth pricer, so it does not contain a separately labelled
  post-filter DLV fit.
- **Current compatibility behavior:** Phase 6 preserves and explicitly names
  the complete pre-filter linear surface state as the compatibility
  unsmoothed DLV state. It also preserves the post-filter smooth SANOS state.
  It does not add a second post-filter `smoothness=0` optimization.
- **Expected impact if changed:** a new post-filter DLV refit would use a
  different quote population and solve a new optimization problem; its grids,
  densities, DLV values, diagnostics, runtime and failure population can all
  differ. It requires explicit approval and a separate versioned identity.

## CD-022 — Unequal-grid DLV inverse reverses spacing coefficients

- **Status:** approved correction on 2026-10-12; do not expose a legacy bug
  mode in the standalone repository.
- **Evidence:** the forward `SanosPricer.dlvsqs` calculation uses the lower and
  upper unequal-grid strike spacings in the finite-difference coefficients.
  The pinned `densities_from_dlvsqs` inverse passes those two coefficients in
  the opposite orientation to the tridiagonal solve. On an actual Phase 6
  fitted-day surface, decoding the persisted full-support DLV vector produces
  first moment `1.009062`, outside the model tolerance `0.0001`, and raises.
  Symmetric/equal-grid fixtures conceal the reversal.
- **Approved treatment:** correct the decoder's tridiagonal coefficient
  orientation, bump the owning `SanosPricer` version, and require density plus
  off-grid call-price round trips on irregular supports. Keep the forward DLV
  encoding unchanged. Per the user's decision, the defective behavior is
  documented but is not retained as an API mode in this new repository.
- **Expected impact if fixed:** newly decoded unequal-grid DLV states recover
  the fitted densities and calls rather than failing or returning distorted
  masses. The authoritative fitted density, forward DLV values, option
  population, and smooth SANOS fit are unchanged; only inverse decoding
  behavior changes.

## User decision required before affected implementation

The first decisions requiring an explicit choice are CD-002, CD-003, CD-004,
CD-006, CD-009, CD-010, CD-012, CD-013, CD-015, CD-020, and CD-021. CD-014 and
CD-016 through CD-019 need a choice only if their preserved compatibility
behavior is to change. The implementation can build
configuration, artifact storage, offline numerical kernels and parity
scaffolding while those specific behavior points remain blocked. CD-022 is
resolved by the approved corrected inverse and its irregular-grid regressions.
