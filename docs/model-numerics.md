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
with presolve, KKT tolerance `1e-7`, and the configured timeout. Solver errors
or limit exits fall back to CLARABEL with the reference tolerances. `optimal`
and `optimal_inaccurate` are accepted; infeasible, unbounded, limit, and other
statuses are surfaced by the fitting layer as explicit failures. The timeout
identity question remains open under CD-006.

Small numerical violations in density mass, positivity, or first moment are
repaired by the L2 projection onto their joint feasible set. Larger violations
remain assertion failures instead of being silently normalized. `ArbFreePricer`
supports complete dictionary round trips; this is compatibility serialization,
not the published Dataset schema.

## Compatibility gates and tests

Two defective, non-production reference interfaces are intentionally blocked:

- the vector form of market-fit diagnostics raises `NotImplementedError`
  pending CD-013; the production list-per-expiry form remains available;
- the optional full JAX optimizer raises `NotImplementedError` pending CD-015.
  The shared transition kernel remains available through `BackboneJax`, with
  value and gradient parity tests against NumPy/finite differences.

The offline suite covers the deterministic pinned fixture, density/DLV/call
round trips on unequal grids, full-vector boundaries, call and density
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
