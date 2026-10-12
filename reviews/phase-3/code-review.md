# Phase 3 independent code review

**Verdict: PASS**

**Reviewer:** `/root/code` (independent code-agent gate)  
**Date:** 2026-10-12  
**Scope:** current uncommitted Phase 3 numerical-core port and first-review
remediation; no WRDS access and no changes to `quant_data`.

## Re-review result

Both findings from the initial `CHANGES_REQUESTED` review are resolved.

1. **Resolved — complete numerical version closure.**
   `one_minute_year_fraction` now owns the output-affecting one-minute model-time
   convention and depends on `business_days_per_year`. `SanosPricer` and
   `ArbFreePricer` depend directly on that convention and on
   `_compute_market_fit: 1.0-blocked-cd013`; `SanosFitter` and `MkArbFree` depend
   directly on the minute convention and `density_floor_enabled`. The latter two
   also retain their direct solver and density-repair dependencies. Runtime
   inspection confirms the expected resolved strings and exact direct-dependency
   sets, and `tests/test_model_core.py` asserts those sets rather than relying on
   weak substring checks. The post-class decoration of `SanosPricer` is equivalent
   to the original `@version`/`@dataclass` order and preserves the dataclass object
   while allowing the later diagnostic helper to be a real dependency.

2. **Resolved — durable production smoothness parity.**
   `tests/reference_probe.py` now runs `SanosFitter` from only the pinned
   `reference/quant_data` source in an isolated subprocess for smoothness `0` and
   production `0.25`. `tests/test_model_reference_parity.py` feeds those
   reference-derived market mids into the standalone fitter and compares fitted
   densities, DLV squares, total variances, and off-grid calls at explicit
   `1e-12` absolute tolerance. The standalone implementation is not used as its
   own oracle. The reference probe retains the earlier fixture fields, so the
   original frozen-value bridge remains active.

## Confirmed acceptance

- All six source hashes still match the pinned snapshot. Inspection of the
  normalized port delta found no unregistered change to objectives, tolerances,
  solver choices, strike geometry, interpolation, warnings, or production error
  behavior.
- Independent comparisons and the committed tests establish numerical parity for
  density/DLV/call transforms, the production smoothness path, and the retained
  no-arbitrage implementation. The invariant suite covers bounds, monotonicity,
  convexity, calendar order, mass/moment constraints, unequal grids, boundaries,
  feasible/infeasible LP behavior, quote containment, serialization, and
  NumPy/JAX values and gradients.
- CD-013 and CD-015 remain explicit, versioned `NotImplementedError` gates rather
  than silent repairs. CD-014's legacy pre-first-expiry jump remains frozen and
  tested.
- The numerical package remains independent of WRDS, filesystem artifacts, and
  WIDB. JAX is optional for runtime users but mandatory in the repository's
  test/development extras and CI, so retained JAX parity cannot silently skip.
- The exact mypy exclusions remain limited to the five directly copied dynamic
  NumPy/CVXPY modules. New constants, JAX code, compatibility gates, tests, and
  all orchestration remain checked.
- `docs/model-numerics.md`, the traceability matrix, compatibility decisions, and
  Phase 3 evidence accurately describe the inspected behavior and remaining open
  semantic choices.

## Commands and checks run

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
87 passed in 13.62s

ruff check --no-cache src tests scripts
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 38 source files

PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
(exit 0)

git --git-dir=.git-real --work-tree=. diff --check
(exit 0)

resolved-version direct-dependency inspection
all four public numerical components match their asserted dependency sets

standalone/reference ONE_MINUTE comparison
exact parity: 1.0175010175010176e-05

isolated reference probe
emitted legacy fields plus smoothness keys `0.0` and `0.25`
```

## Residual risks

- Primary-solver `SolverError`/`user_limit` fallback into CLARABEL remains
  source-faithful and was inspected, but does not yet have a focused branch-level
  regression test. This is non-blocking because Phase 3's required feasible and
  infeasible solver behavior is covered.
- The synthetic parity market is intentionally small. Full 891-day population and
  numerical parity remains a later daily-pipeline acceptance gate and is not claimed
  by this phase.

No blocking code-review finding remains for Phase 3.
