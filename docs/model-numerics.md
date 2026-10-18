# SANOS, DLV, and no-arbitrage numerical contract

This document describes the standalone numerical core in `sanos.model`. It is
a compatibility port of the implementation pinned at `quant_data` revision
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`; it does not depend on WIDB,
artifact locations, or network access. The paper sources are under `tex/`.

## Units and state geometry

For expiry `T`, forward `F_T`, discount factor `DF_T`, cash strike `K`, and cash
call price `C`, the model uses

```text
k = K / F_T
c(T, k) = C(T, K) / (DF_T F_T).
```

Every expiry has its own strictly increasing normalized grid `xstrikes[t]` of
shape `(N_t,)` and density-mass vector `xdensities[t]` of the same shape. The
state constraints are

```text
q_i >= 0,       sum_i q_i = 1,       sum_i k_i q_i = 1.
```

Consequently the unsmoothed normalized call is

```text
c(T_t, k) = sum_i q[t,i] max(k[t,i] - k, 0).
```

It is bounded by intrinsic value and one, decreases and is convex in strike,
has value one at strike zero, left slope minus one, and reaches zero beyond the
right grid boundary. Calendar monotonicity is imposed on common strike points.
Expiries are positive, strictly increasing year fractions with 252 business
days per model year.

`SanosPricer.dlvsqs()` returns one interior squared-DLV vector of shape
`(N_t-2,)` per expiry. With `on_all_strikes=True`, it returns shape `(N_t,)`
and explicit zero boundary entries. Values are annualized squared local
volatilities. For `dt_t = T_t-T_(t-1)`, the interior finite-volume scale is

```text
B[t,i] = k[t,i]^2 dt_t / (k[t,i+1] - k[t,i-1])
DLV[t,i]^2 = (c_t(k[t,i]) - c_(t-1)(k[t,i])) / (B[t,i] q[t,i]).
```

The implementation handles zero mass or zero call increment with its frozen
zero/min/max conventions. `densities_from_dlvsqs()` applies the inverse
implicit tridiagonal transition and accepts either interior vectors or full
vectors with zero boundaries. Grids may differ across expiries, but later
grids must cover earlier ones as required by the reference algorithm.

A zero-density node with a positive call-time increment has no finite DLV
representation. `dlvsqs()` returns its documented sentinel convention without
running a false inverse assertion, but such a sparse state is not considered
round-trippable. This occurs in the canonical initial no-arbitrage state, so
the state is persisted as support plus density rather than requiring invented
finite coefficients. CD-034 supersedes CD-021's projection treatment.

## Canonical daily unsmoothed DLV state

The canonical unsmoothed DLV model is the initial full no-arbitrage common
surface fitted directly to the normalized market books before OptionsFilter.
Its interpreting state is the exact ordered retained expiry chain, total
variances, common `initial_surface_model_strikes` supports,
`initial_surface_model_density` probability masses, and explicit mappings from
each expiry's listed geometry. The state is serialized unchanged: validation
does not delete zero-mass nodes, compress support, alter prices, or refit it to
the later smooth SANOS surface.

The full state must retain strictly increasing positive supports, finite density
masses, the declared expiry order, unit mass and martingale first moment within
the existing tolerances, decreasing/convex/model-bound call prices, and calendar
monotonicity on the complete union of consecutive stored supports. Calendar
acceptance remains tied to the authenticated terminal surface solve: `1e-7` for
HiGHS, `1e-6` for an optimal CLARABEL fallback, and `5e-5` only for an explicit
reduced-accuracy `optimal_inaccurate` result.

A finite squared-DLV coefficient does not exist at a node with zero probability
mass and positive call-time increment. Therefore squared DLV is not a mandatory
serialization or validation field. Consumers price directly from stored
support/density state; they may derive finite coefficients only on a domain
where their own mathematical preconditions hold. Smooth SANOS is an independent
post-filter model. No SANOS/DLV price-equality or projection-error contract is
claimed.
## Smooth SANOS state

`smoothness=None` is the unsmoothed piecewise-linear DLV model.
`smoothness=0` follows the explicit zero-smoothing SANOS path and has the same
payoff kernel. A positive value replaces each point payoff in the market-price
map with its Black-smoothed value using

```text
kernel volatility = sqrt((total_variance_t / T_t) * smoothness).
```

The fitted density constraints and discrete-time DLV bounds remain on the
underlying mass state. Production compatibility uses smoothness `0.25`.
`variances[t]` is smoothing-kernel input geometry, not a claim that the fitted
surface's ATM implied variance equals that number. Explicit volatilities map to
`T_t * vol_t^2`; inferred values come from interpolated normalized ATM calls,
with forward variance floored at `0.001^2` before cumulative reconstruction.

Pricing at exactly zero uses intrinsic value. For compatibility, any positive
time before the first fitted expiry uses the first fitted density rather than
interpolating from the time-zero Dirac mass. Between expiries the pricer
interpolates call prices linearly; after the final expiry it holds the final
density. This discontinuous pre-first-expiry rule is frozen by CD-014.

## LP and no-arbitrage behavior

`SanosFitter` minimizes a weighted L1 price error subject to nonnegative unit
mass, normalized first moment, calendar-call monotonicity, and configured
minimum/maximum DLV constraints. `MkArbFree` first constructs an extended
strike grid and a per-expiry fit; surface mode then enforces the cross-expiry
constraints. Returned masks remain aligned to the original input books and
`input_index` identifies any retained subset of input expiries.

The default LP backend is HiGHS serial dual simplex (`simplex_strategy=1`),
with presolve, KKT tolerance `1e-7`, and the configured timeout. The production
smooth-SANOS surface preserves the compatibility default: simplex then
CLARABEL. Each retry uses the identical LP and incumbent values and occurs only
after `solver_error`, the recognized `invalid_solution`, or `user_limit`;
successful, infeasible, unbounded, and unexpected outcomes are not retried.
The timeout applies separately to the two attempts. CLARABEL retains its
status-dependent reference tolerances.

If every normal smooth-surface attempt fails, CD-035 permits a returned primal
state to be considered independently of its nonoptimal status. This is not
status-based acceptance. The state must come from the exact failed attempt,
map completely to the original variables, remain finite without any
candidate-specific clipping or projection, and satisfy every original
equality, inequality, and variable bound within the named `1e-7`
candidate-feasibility tolerance. This adds no
relaxation to the existing fitted-state gates. The authenticated candidate
probability masses then follow the unchanged common surface postprocessing used
for normal successes. That exact published state must also pass the existing
density minimum, unit-mass, martingale-moment, strike-shape, model-bound, and
full-support calendar gates.

Quote auxiliaries are reconstructed at their exact minima from the candidate
prices using the same one-percent inward bid/ask contraction as the LP. This
proves a complete feasible LP point and supplies the exact comparable
objective; fit quality is reported but is not an additional acceptance band.
A normal success always wins. A failed-attempt candidate enters the comparison
pool only after both its raw state and unchanged postprocessed published state
pass their complete gates. Otherwise the qualifying candidate with the lowest
exact published-state objective is retained and explicitly marked
`feasible_nonoptimal` with its original failure status and residual evidence.
Both the raw candidate and published-state digests, residuals, and exact
objectives are retained.

This numerical policy is part of the ordinary current smooth-surface producer;
it is not a manual recovery algorithm. After a policy change, `intermediate
resume-fitted` validates exact artifacts completed by the frozen original plan
and invokes the normal current Intermediate build only for original targets
that are absent. Candidate discovery, validation, ranking, and publication
therefore remain inside the versioned numerical pipeline. Neither an operator
nor an agent supplies failed dates or candidate evidence. A current result is
recorded as a typed replacement for its absent original target, while every
old successful pointer remains unchanged.

The generic numerical constructor can L2-project small density mass,
positivity, or first-moment violations onto their joint feasible set. The
canonical fitted-day serializer and validator do not invoke such a repair:
they preserve the producer's exact support, density, and prices and fail on an
invalid state. `ArbFreePricer` supports complete dictionary round trips; this
is compatibility serialization, not the published Dataset schema.

## Compatibility gates and tests

Two defective, non-production reference interfaces are intentionally blocked:

- the vector form of market-fit diagnostics raises `NotImplementedError`
  pending CD-013; the production list-per-expiry form remains available;
- the optional full JAX optimizer raises `NotImplementedError` pending CD-015.
  The shared transition kernel remains available through `BackboneJax`, with
  value and gradient parity tests against NumPy/finite differences.

The offline suite covers the deterministic pinned fixture, density/DLV/call
round trips on unequal grids, a nonrepresentable sparse audit state,
full-vector boundaries, exact bound-allowance edges, call and density
invariants, pre-first-expiry behavior, variance geometry, L2 projection,
smoothness `None`, `0`, and `0.25`, feasible fits, infeasible LP status, quote
containment, calendar monotonicity, strike trimming, serialization, resolved
`@version` dependencies, and both compatibility gates. The copied dynamic
numerical modules retain their legacy typing style and are isolated by exact
mypy module overrides; all new orchestration and data-pipeline code remains
strictly checked.

The production `0.25` assertion is not self-referential: an isolated subprocess
loads only the pinned `quant_data` snapshot, emits its deterministic LP state,
and the standalone fitter compares densities, DLVs, variance geometry, and
off-grid prices against that output. Resolved versions explicitly include the
one-minute branch convention, density-floor policy, diagnostic gate, and their
transitive numerical helpers.
