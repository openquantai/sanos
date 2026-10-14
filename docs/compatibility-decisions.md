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

- **Status:** approved for the standalone public-projection contract on
  2026-10-12. The user authorized the dual-reviewed Phase 8 recommendation to
  proceed while they were away. Legacy internal behavior remains frozen.
- **Evidence:** legacy `normalize_day` records `spot_ret = S_t/S_previous - 1`.
  It does not carry-adjust the return and must not be relabelled.
- **Approved public contract:** public rows contain exactly the populated
  retained internal dates in ascending order. Policy-owned typed-empty dates
  are omitted and recorded in coverage evidence; any other empty date or a
  populated date with no retained public book fails publication. For each pair
  of consecutive public rows, let `h_t` be the number of NYSE business
  sessions in `(t-1,t]`. From the previous public date's distinct expiry curve,
  form positive nodes `q_j=F_j/S_(t-1)` and `d_j=DF_j`, add `(0,1)` anchors,
  require increasing business-DTE nodes with consistent duplicates, and
  interpolate `log(q)` and `log(d)` linearly at `h_t`. Exact nodes use the
  exact value; extrapolation beyond the final node is forbidden. Validate the
  discount interpolation as build evidence, but do not publish it. With
  `c_t=q(h_t)`, publish
  `forward_adjusted_spot_return = (S_t/S_(t-1))/c_t - 1` in Float64. The first
  public row is null. After an omitted typed-empty date, use the immediately
  preceding public row and the corresponding multi-session horizon. Weekends
  and holidays affect only the session count; a half-day counts as one
  session. Publish neither the absolute spots nor carry/discount inputs.
- **Expected impact:** adds a distinct public field and uses each preceding
  retained public day's forward curve. Because the first public return is null,
  it adds no pre-start forward/zero boundary dependency and does not replace
  internal `spot_ret`.

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

- **Status:** approved for the standalone public-projection contract on
  2026-10-12. The user authorized the dual-reviewed Phase 8 recommendation to
  proceed while they were away. This approval is technical, not a grant of
  redistribution rights.
- **Evidence:** legacy pure model prices use `C/(DF*F)`, while the requested
  public spreads use `half_spread/S`. They cannot be compared without carry.
- **Approved public contract:** model prices remain reconstructed in
  discounted-forward units. Publish only per-expiry
  `discounted_forward_over_spot = DF*(F/S)` in positive finite Float64 and
  per-book `half_spread_spot = half_spread/S` in positive finite Float64. Then
  `C/S = discounted_forward_over_spot * C/(DF*F)` and
  `h/(DF*F) = (h/S)/discounted_forward_over_spot`. Do not publish `DF`, `F/S`,
  spot, forward, absolute strikes, absolute prices, or carry separately.
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

- **Status:** resolved on 2026-10-13; the initial no-arbitrage surface is audit
  state, not the standalone DLV state.
- **Evidence:** the production path first constructs a linear surface with
  `MkArbFree(smoothness=None)`, then applies `OptionsFilter`, and finally fits a
  smooth `0.25` SANOS surface. The legacy daily payload serializes only the
  final smooth pricer. On actual fitted days the sparse initial surface can
  contain a positive call-time increment at a zero-density node. Such a
  transition has no finite DLV representation; calling `dlvsqs()` on it is not
  a valid persisted-state contract.
- **Approved treatment:** retain the complete initial linear surface, masks,
  prices, densities, and diagnostics for no-arbitrage/filter audit. Remove the
  `initial_surface_dlv_squares` field and do not expose this surface as the
  reference DLV model. The standalone DLV state is defined separately by
  CD-023.
- **Expected impact:** initial no-arbitrage prices, masks, populations, and
  smooth SANOS fits are unchanged. The defective/undefined initial DLV field
  disappears, and fitted/internal/public schema identities are versioned.

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

## CD-023 — Canonical reference DLV projection

- **Status:** approved on 2026-10-13 for the fresh SANOS reference Dataset.
- **Evidence:** neither the legacy daily payload nor the sparse initial
  no-arbitrage surface provides the required finite, independently usable DLV
  state. A deterministic projection of the completed smooth SANOS surface is
  stable on the retained pilot dates and leaves the authoritative smooth fit
  and book population unchanged.
- **Approved treatment:** after the smooth `0.25` SANOS fit, fit a separate
  `smoothness=None` `SanosFitter` at the same retained expiries. Use the source
  smooth ATM-variance geometry, uniform price weights, NMS `[-3,2]`, 200 inner
  nodes with one exact ATM node, and support boundaries equal to 0.5 times the
  lower and 1.5 times the upper inner strike. Configure DLV bounds `[0.01,3]`
  and retain the fitted values without clipping. To make the tail inequalities
  numerically meaningful, divide each dedicated DLV constraint row by its
  positive geometric box coefficient and corresponding squared-volatility
  bound, and solve this projection with HiGHS
  primal, dual, and KKT tolerances of `1e-10`; other SANOS/no-arbitrage solves
  retain their established solver settings. Impose a `9e-5` maximum absolute
  error at the 200 fitted call nodes, leaving headroom inside the independent
  `1e-4` union-grid acceptance gate. At a target node with source Black vega
  divided by `sqrt(T)` greater than or equal to `0.05`, intersect that price cap
  with the exact Black-call band corresponding to source annual implied
  volatility plus/minus `0.005` (flooring the lower volatility at zero), inset
  by the `1e-10` solver tolerance. These additional constraints remain linear
  in fitted prices and do not change the uniform L1 objective.
  Use a dedicated `1e-8` probability-mass floor and solve against DLV bounds
  inset by `1e-5` volatility units, so finite-precision inverse decoding does
  not amplify vanishing tail masses across the published `[0.01,3]` bounds.
  Persist one deterministic
  anchor row per retained expiry with support, probability masses, squared DLV,
  target/fitted calls, time increment, variance geometry, configuration ID,
  and canonical diagnostics. Before persistence, apply one deterministic
  DLV encode/decode stabilization cycle and store the decoded density with its
  re-encoded DLV vector; this removes floating-point amplification at very
  small tail masses without clipping either state. The canonical boundary
  calculation uses NumPy extended precision and fails closed when it is not
  wider than Float64, while every persisted array and the ordinary consumer
  decoder remain Float64.
- **Acceptance:** density mass and moment residuals are at most `1e-10`;
  both canonical and generic-consumer density/call inverse errors are at most
  `1e-12`; canonical DLV fixed-point error is at most `1e-9`; the mixed
  Float64-decode/canonical-re-encode residual is conditioning evidence only;
  no singular node; normalized-call projection error at most `1e-4`;
  ATM annual-IV error at most `1e-4`; and source-vega-conditioned annual-IV
  error has maximum `5e-3` and deterministic p99 `1e-3`. A `1e-9` absolute
  squared-DLV bound allowance accommodates solver-scale residuals, records
  signed shortfall/excess and saturation counts, and never clips values.
- **Capability boundary:** smooth-to-DLV equivalence is asserted only at the
  retained expiries and inside inclusive source-smooth NMS `[-3,2]`.
  Out-of-domain differences are report-only. Arbitrary-time equivalence is not
  advertised.

## CD-024 — Initial fitted-price Black round-trip tolerance

- **Status:** resolved from the fresh Phase 10 pilot measurement on 2026-10-14.
- **Evidence:** initial `fits1` and `fitsX` prices are converted to implied
  volatility and repriced before their reference-compatible Float32 fields are
  persisted. The copied implementation supplied `eps=1e-8` to the Black kernel
  but inherited cdxcore's distinct `price_tol=1e-6` implied-volatility stopping
  tolerance as an unversioned default. Across all 136,737 retained expiry rows
  in the 16-day pilot, 34 surface values and 36 per-expiry values exceeded a
  bare `1e-6` reconstruction check. Every value was within `1e-6` plus one
  Float32 ULP; the maximum observed residual was `1.023709774061743e-6` and its
  stored-value ULP was `5.960464477539063e-8`.
- **Treatment:** make `price_tol=1e-6` explicit and version-owned in the cdxcore
  adapter. Validate the direct fitted state against its persisted Black
  price using that tolerance plus one conservative Float32 ULP. Replay the
  version-owned implied-volatility and Black-vector pipeline and require exact
  Float32 equality for volatility, price, and every Greek; a merely coherent
  alternative vector is not accepted. Retain the independent Black
  price/volatility/Greek cell-envelope checks as defense in depth.
- **Expected impact:** no fitted value, state, mask, population, or schema
  changes. Validation now models the producer's actual numerical contract
  instead of rejecting valid pilot artifacts at an accidental boundary.

## CD-025 — Distinct smooth fitted-price replay contracts

- **Status:** resolved from the fresh Phase 10 pilot measurement on 2026-10-14.
- **Evidence:** the copied smooth fitter exposes three related but distinct
  Float64 observables. Per-expiry `fits1` uses a linear payoff on Float32
  supports; solver-time `fitsX` uses Float32 listed strikes/supports and the
  expression `sqrt(smoothness * variance / T)`; the later
  `compute_market_fit().fits` path uses
  `sqrt(variance / T) * sqrt(smoothness)` and the copied arbitrary-time
  interpolation rules. The supports are persisted losslessly in Float64 list
  columns. A validator that evaluated every field through the canonical
  Float64 `SanosPricer` falsely rejected 2024-12-23 at `5.798907092013916e-8`.
  Across all 15 populated pilot dates, exact producer replay has zero maximum
  error for expiry fit, surface fit, quote-aligned call mid, and fit error. The
  maximum canonical-Float64 bridge residual is `7.109712970437831e-8` for
  `fitsX` and `3.155308114788369e-8` for `smooth_call_mid`.
- **Treatment:** replay and validate `fits1`, `fitsX`, and
  `compute_market_fit` separately, preserving their Float32 inputs, operation
  ordering, arbitrary-time behavior, and direct intrinsic return below one
  model minute. Require each stored producer observable to match its replay
  within `1e-12`. Keep the canonical Float64 `SanosPricer` unchanged for
  off-grid consumers, public SANOS state, and the DLV projection. Retain the
  existing `5e-8` acceptance gate between `smooth_call_mid` and that canonical
  consumer; record the accumulated `fitsX`-to-canonical difference as a named
  diagnostic bridge rather than treating it as producer replay.
- **Expected impact:** no fitted value, support, density, variance, mask,
  population, SANOS parameter, DLV parameter, or fitted schema changes. The
  validation schema and validation producer identity change because the three
  field meanings and two bridge residuals are now explicit.

## CD-026 — Initial surface fit is a soft quote-band fit

- **Status:** resolved from the copied implementation and fresh Phase 10 pilot
  measurement on 2026-10-14.
- **Evidence:** `MkArbFree.fits1` is an accepted per-expiry fit whose producer
  accepts masked prices only within bid/ask plus `1e-6`. `MkArbFree.fitsX` is a
  globally arbitrage-free soft closest fit; the copied class contract states
  explicitly that it need not remain inside bid/ask. The first Phase 10
  validator incorrectly imposed the hard-fit tolerance on both. Across all 15
  populated pilot dates, reconstructed `fits1` has maximum quote-band
  violation `9.922384824334607e-7`, while reconstructed `fitsX` reaches
  `1.1905097704500989e-4` and is therefore behaving within its declared soft
  objective rather than a nonexistent containment contract.
- **Treatment:** enforce the version-owned `1e-6` quote-containment tolerance
  on reconstructed `fits1` only. Record the reconstructed `fitsX` value as
  `initial_surface_fit_quote_band_violation`, in absolute normalized pure-call
  units `C/(DF*F)` over `mask_arb_free`, without using it as an acceptance gate.
  Continue to require exact `fitsX` producer replay, its Black-vector contract,
  normalized call bounds, density mass/moment/positivity, strike convexity,
  and calendar no-arbitrage.
- **Expected impact:** no Raw, Intermediate, fitted, SANOS, DLV, mask, or
  population value changes. Validation schema and producer identity change to
  correct the meaning of one residual and remove the invalid surface
  containment rejection.

## CD-027 — Smooth calendar validation follows the accepted solver

- **Status:** resolved from the copied solver configuration and fresh Phase 10
  pilot measurement on 2026-10-14.
- **Evidence:** 2025-01-03's primary serial HiGHS smooth-surface solve returned
  `invalid_solution`; the recorded CLARABEL fallback returned `optimal` with
  feasibility tolerance `1e-6`. Its reconstructed calendar residual is
  `2.0046636678427987e-7`. The previous universal `1e-7` validator rejected a
  state accepted by the copied solver. Across the other populated pilot dates,
  smooth residuals are at most `8.067573398218286e-8`; initial-state residuals
  are at most `8.7691252648181e-8`; canonical DLV residuals are at most
  `4.440892098500626e-16`.
- **Treatment:** validate each initial and smooth terminal solver grammar
  first, then use its configured feasibility contract: `1e-7` for HiGHS,
  `1e-6` for an optimal CLARABEL fallback, and `5e-5` for CLARABEL
  `optimal_inaccurate` reduced termination. Record the applied values as
  `initial_surface_calendar_acceptance_tolerance` and
  `smooth_calendar_acceptance_tolerance`. Strike shape and model-free bounds
  retain `1e-7`; DLV validation retains its existing independent contracts.
- **Expected impact:** validation and legacy-comparison report identity change;
  fitted states, prices, masks, populations, SANOS parameters, and DLV
  parameters do not.

## CD-028 — Exact AM business-clock replay operation order

- **Status:** resolved from the copied daily producer and fresh Phase 10 pilot
  measurement on 2026-10-14.
- **Evidence:** the producer casts the integer business-session count to
  Float32, subtracts the configurable AM amount in Float64, clips, and then
  casts the result to Float32. The validator prematurely cast the subtraction
  to Float32. On 2025-01-16 this changed 145 rows of the front AM expiry by one
  Float32 step (`1.4901161193847656e-8`); replaying the producer order is
  bitwise exact.
- **Treatment:** own and test the producer-exact operation order in the
  validation dependency closure. Keep `overnight_variance_fraction=0.2` as the
  backward-compatible configurable default and continue requiring exact
  `business_dte` and `time_to_expiry_years` equality.
- **Expected impact:** validation identity only; no daily or fitted artifact is
  changed or rebuilt.

## User decision required before affected implementation

The first decisions requiring an explicit choice are CD-002, CD-003, CD-006,
CD-010, CD-012, CD-013, CD-015, and CD-020. CD-014 and
CD-016 through CD-019 need a choice only if their preserved compatibility
behavior is to change. The implementation can build
configuration, artifact storage, offline numerical kernels and parity
scaffolding while those specific behavior points remain blocked. CD-004,
CD-009 and CD-021 through CD-028 are resolved by their approved or measured
contracts and regressions.
