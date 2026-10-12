# Phase 6 quantitative/data design audit

**Status: READY FOR PARITY IMPLEMENTATION; DO NOT ADD A NEW POST-FILTER DLV
REFIT OR CHANGE THE OPEN POLICIES LISTED BELOW WITHOUT APPROVAL**

Reviewer: independent `quantagent`  
Date: 2026-10-12  
Pinned reference: `quant_data` revision
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`

This is a design audit, not an implementation review. I inspected the committed
Phase 5 contract, checked-in reference inventory and pinned production source.
I did not access WRDS or the network, modify `quant_data`, or edit implementation
code.

## 1. Authoritative daily sequence

The compatibility path is exactly:

```text
Phase 5 enriched-day (Float32 persisted fields)
  -> initial MkArbFree(surface, smoothness=None)
  -> append expiry/surface no-arbitrage columns
  -> OptionsFilter
  -> normalize_day inside OptionsFilter
  -> smooth MkArbFree(surface, smoothness=0.25,
                      surface_expiry_linear=True)
  -> active real-book masks, smooth fitted values, states and diagnostics
```

`SanosFitter` and stale `widb_sanos_day.fit_sanos_day` are not on this producer
path. The production smooth fit uses `MkArbFree`. No stage may be reordered.

The initial linear fit groups by ascending `expiry_id`, stable-sorts each group
by `pure_strike`, promotes `tte_y`, `pure_strike`, `pure_bid_price` and
`pure_ask_price` explicitly to Float64, and returns only retained expiries. It
keeps every input-strike row of a retained expiry, including rows whose
`mask_arb_free` is false.

The reference `OptionsFilter` does **not** filter on `mask_arb_free`. The smooth
fit therefore consumes all rows that survive `OptionsFilter`, including
pre-smooth mask-false rows. After the smooth fit:

- `pre_af_mask` is the initial linear no-arbitrage mask;
- `model_mask = pre_af_mask & smooth_fit_mask`, with false for an expiry dropped
  by the smooth fit;
- `call_active` and `put_active` use `pre_af_mask`, finite/valid real side books,
  and do **not** use `model_mask`; and
- smooth call mids are the surface-pricer values; put mids are
  `call_mid - 1 + normalized_strike`.

These distinctions are population-defining and require direct regressions.

## 2. Exact compatibility configuration and filter order

Initial no-arbitrage and smooth fits share these defaults:

| field | value |
|---|---:|
| `no_arb_mode` | `surface` |
| `min_half_spread` | `1e-4` |
| `mid_fit_weight` | `0.1` |
| `min_atm_vol` | `0.05` |
| `min_dlv`, `max_dlv` | `None`, `None` |
| `timeout_seconds` | `300` |
| `min_strikes_per_expiry` | `5` |
| `min_density` | `1e-4` |
| `min_dC_eps` | `1e-3` |
| remove no-arbitrage strikes | `true` |
| price/density/mass tolerances | `1e-6`, `1e-7`, `1e-4` |
| hard-fit spread tightening | `1%` of half-spread |

The initial fit has `smoothness=None`. The production SANOS fit has
`smoothness=0.25` and `surface_expiry_linear=True`, so its per-expiry screening
fit is linear and its final surface kernel is smooth.

For the canonical smooth-market cache, apply `OptionsFilter` in this exact
order:

1. `business_dte <= 70` (inclusive);
2. spot normalization;
3. inclusive ATM-NMS range `[-100,100]` (deliberately non-binding);
4. `log_volume >= 1`;
5. `log_open_interest >= 1`;
6. optional AM removal, disabled;
7. drop expiries with fewer than 10 remaining rows;
8. cap to 100,000 rows after reserving at least 10 per expiry.

`min_vegaSqrtVar=0.001` remains accepted and versioned but is not applied
(CD-012). The cap reserves, per expiry, the smallest positive NMS, largest
negative NMS, then the smallest absolute NMS using stable ties; remaining slots
are filled globally by smallest absolute NMS. The cap uses `nms`, not `nms_atm`,
even though the range filter uses ATM NMS. If reserved rows exceed the cap, the
reference asserts; do not invent a policy.

`normalize_day` occurs only after the horizon filter and before the NMS filter.
It preserves the legacy simple Float32 return, divides spot-denominated fields
by spot, adds `spot_ref`, and does not change the pure `K/F`, `C/(DF*F)` fields.

## 3. No-arbitrage constraints and solver semantics

For every expiry support `x` and density mass `q`, enforce:

```text
0 <= q <= 1
sum(q) = 1
x @ q = 1
C_t(k) >= C_(t-1)(k) on the current interior support
```

The first prior distribution is the unit Dirac at one. Optional DLV bounds use
`x_i^2 * dt / (x_(i+1)-x_(i-1)) * q_i`, but both are disabled in the
authoritative `MkArbFree` configurations.

Per-expiry processing is compatibility-sensitive:

- construct widening boundary strikes and trim invalid boundary slopes;
- soft-fit first; if needed hard-fit inside the spread tightened by 1%;
- containment uses bid/ask with `eps_price=1e-6`;
- if fewer than half the live strikes fit, drop the expiry;
- otherwise optionally refit after removing non-fitting strikes; if that refit
  fails, retain the prior state and expose the reduced mask; and
- carry earlier boundary ranges into later expiries.

The surface grid is built backwards by adding later-expiry strikes subject to
the current minimum spacing. The surface constraint compares call values on
the current interior grid. Density flooring is disabled; small numerical
violations are repaired by the versioned L2 non-negative mass/first-moment
projection, while large failures remain fatal.

The soft objective uses half-spread weights

```text
weighted_half_spread = max((ask-bid)/2, 1e-4) / average_half_spread
objective = 0.9 * sum(outside-spread slack) + 0.1 * max-mid-error
```

with one max-mid-error variable per expiry. Primary solve settings are HiGHS
serial dual simplex, presolve on, KKT tolerance `1e-7`, warm start, and the
configured timeout. `SolverError`, `user_limit`, or the known invalid-solution
failure triggers the CLARABEL fallback with the frozen Phase 1 tolerances.
`optimal` and `optimal_inaccurate` are accepted. Infeasible, unbounded, limit,
and other statuses are explicit failures; they must not become silent day
exclusions. Record per-solve status, fallback, objective and stage. Runtime is
operational journal metadata, not deterministic model state.

## 4. Unsmoothed DLV and smooth SANOS state

The compatibility-defined **unsmoothed DLV state** is the initial pre-filter
linear surface fit, not a new second fit. Preserve its complete surface support
`mstrikes`, masses `mdensities`, ordered positive expiries, input-index map,
per-expiry supports/masses (`xstrikes`, `xdensities`), `xix_to_mix`, masks,
input books, `fits1`, `fitsX`, inferred total variances and pricing flags.
Derive squared DLVs on `mstrikes/mdensities` with `dt=diff(T, prepend=0)`,
interior units `1/year`, and explicit zero boundaries in the stored full-vector
encoding. Reconstruction must recover the same masses and calls within the
frozen DLV tolerances.

The **smooth SANOS state** is the post-filter `smoothness=0.25` surface pricer.
Store the same complete geometry plus its separately named smoothing-kernel
input total variances. Do not call those variances fitted ATM variances. Smooth
pricing uses

```text
kernel_vol(t) = sqrt(0.25 * interp(clip(t), T, V) / clip(t))
```

with actual query horizon in the Black kernel. Preserve CD-014: intrinsic below
`ONE_MINUTE`, first density immediately for later positive times before the
first node, linear interpolation of density-implied prices between nodes, and
the final density after the last node.

Store one Polars frame per day. The preferred lossless layout is one row per
input expiry with ordered `List[Struct]` books and list-valued expiry/surface
state. It must retain input books and named stage masks/reasons even when the
parity view drops them. Day-level state/diagnostics may be a typed struct
repeated per expiry; do not create separate quote, expiry or surface artifacts.
An exact flattening adapter must reproduce the reference row view.

## 5. Dtypes, ordering and overnight propagation

- Phase 5 inputs remain Float32 at the persisted boundary.
- Initial daily no-arbitrage explicitly promotes its four numerical solver
  inputs to Float64, then writes mask as Boolean and fit price/vol/Greek columns
  as Float32.
- Spot normalization retains the reference Float32 arithmetic.
- Smooth-market strike, bid, ask, expiry and direct real-book arrays enter from
  the Float32 filtered frame; CVXPY/model state is Float64 where the reference
  naturally promotes it. Preserve the resulting mixed serialized dtypes; do
  not apply a blanket Float64 conversion under CD-011.
- Order input expiries by `expiry_id`, books by normalized strike, and retained
  state by `input_index`. Require unique strictly increasing model times and
  strikes; do not merge or jitter collisions.

For AM expiries, `T=(base_business_dte-1+overnight_variance_fraction)/252`
when not clipped. The fraction changes the initial and smooth expiry vectors,
ATM/forward-variance inference, calendar constraints and DLV `dt`, in addition
to Phase 5 NMS/filter eligibility. PM values must be unchanged before their
interaction with altered AM surface constraints. Test `0.1`, `0.2`, `0.5` end
to end. At `0`, zero-time next-session AM nodes fail the strictly-positive-time
contract; at `1`, equal AM/PM model times can fail the strictly-increasing-time
contract. Report those cases explicitly; no drop/merge/jitter policy is
approved.

## 6. Diagnostics and invariants

The daily frame/manifest must include counts and first-rejection reasons after
initial boundary trimming, per-expiry fit, expiry drop, each OptionsFilter
predicate, minimum-per-expiry, cap, smooth expiry fit, and smooth surface fit.
Retain initial mask, smooth mask, their intersection, active call/put masks,
input/retained expiry maps, solver status/fallback, and exact configuration.

Required numerical invariants are:

- support strictly increasing and positive;
- density `min >= -1e-7`, mass and first moment within `1e-4` (the reference
  repair trigger), plus the tighter post-repair checks when repair occurred;
- normalized calls bounded by intrinsic and one, decreasing and convex;
- surface calls calendar-monotone within `1e-7` on tested common grids;
- masked `fit1` inside bid/ask within `1e-6`;
- finite `fitX`, smooth fitted calls/puts and diagnostics;
- `af_put_mid = af_call_mid - 1 + k`; and
- DLV full vectors have zero boundaries, non-negative finite interiors, and
  density/call round-trip error within the frozen `1e-3` general tolerance.

Production `fit_result` uses the list-per-expiry diagnostic and counts any
strictly positive normalized half-spread error as outside. The broken ndarray
branch remains blocked by CD-013.

## 7. Parity fixtures, tolerances and population gates

Add an isolated pinned-reference subprocess fed by checked-in Phase 5 enriched
fixtures. It must freeze, stage by stage:

1. boundary trimming, expiry dropping and strike-mask refit cases;
2. initial Float32 daily additions and exact flattened ordering;
3. every OptionsFilter threshold, its order, the no-op vega field, minimum per
   expiry and a binding cap/tie case;
4. smooth `0.25` state, masks, direct real call/put books, fitted call/put mids,
   active masks and fit diagnostics;
5. default `0.2` plus `0.1/0.5` AM sensitivity and explicit `0/1` failures; and
6. solver failure/fallback plus interrupted multi-day restart.

Compare schema, dtypes, ordering, strings, indices, masks and counts exactly.
Compare persisted Float32 vectors bitwise on the pinned fixture. For the
non-degenerate pinned LP fixture compare Float64 supports, densities, fits,
DLVs and reconstructed off-grid calls at `rtol=0, atol=1e-12`, matching the
existing Phase 3 oracle. Separately enforce the implementation tolerances above
on adversarial/real-like markets; never use a loose value tolerance to excuse a
different mask or population.

Fresh-build Phase 6 gates are:

- 891 populated SPX dates from 2022-01-03 through 2025-08-29 and the explicit
  empty marker on 2025-01-08;
- 26,732 smooth-cache expiry entries and 2,414,341 strike nodes;
- 2,325,708 active calls plus 2,325,708 active puts = 4,651,416 books;
- 8,090 active books at the retained 0.2-business-day node; and
- as a downstream adapter check, 3,566,258 daily-front books after European,
  <=70D, explicit 0.2D exclusion and inclusive AF-ATM NMS `[-3,2]`.

The separate 2,192,458-book Daily25 SDLV artifact is not a Phase 6 target. A
fresh vendor revision can legitimately change source rows, but every mismatch
must be traced first to Raw fingerprints and then to the first divergent stage;
it is not accepted by numerical tolerance or a hard-coded exclusion.

## 8. Open semantic decisions

1. **CD-006:** the legacy cache identity omits solver timeout. Preserve that
   observable identity only if it is explicitly recorded as a compatibility
   policy; whether timeout enters new content identity remains open. It must
   always appear in execution provenance.
2. **CD-012:** keep `min_vegaSqrtVar` as a tested identity-only no-op. Activating
   it is a population-changing new version requiring approval.
3. **Unsmoothed DLV universe:** this audit defines the existing pre-filter
   linear surface as the compatibility DLV state. Fitting a new unsmoothed
   surface after `OptionsFilter` so DLV and smooth SANOS use identical books is
   a new quantitative pipeline and requires explicit approval and its own
   identity. Register this before exposing such a variant.
4. **Mask semantics:** changing real-book activity from `pre_af_mask` to the
   stricter `model_mask`, or filtering smooth inputs by either mask, changes the
   authoritative populations and requires approval.
5. **CD-005:** retain the legacy `pcnt_fit` field/predicate without calling it
   “inside-spread percentage.” A renamed interpreted metric requires the
   registered decision; an additional clearly defined count may be versioned
   without changing the fit.
6. **CD-011/CD-014:** Float64-through-fit and time-zero-to-first-density
   interpolation remain blocked. CD-013 and CD-015 remain outside the
   authoritative path.

The compatibility implementation may proceed while preserving these behaviors
explicitly. It must not claim that Phase 6 itself approves the optional
post-filter DLV refit, timeout identity change, vega filter, mask change, public
projection units, or public redistribution.

Signed: `quantagent` — independent quantitative/data reviewer
