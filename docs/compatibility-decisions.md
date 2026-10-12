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
- **Output impact:** none for the authoritative 2022–2025 reference path.

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

## User decision required before affected implementation

The first decisions requiring an explicit choice are CD-002, CD-003, CD-004,
CD-006, CD-009, CD-010, CD-012, CD-013 and CD-015. CD-014 needs a choice only if the
legacy pre-first-expiry jump is to change. The implementation can build
configuration, artifact storage, offline numerical kernels and parity
scaffolding while those specific behavior points remain blocked.
