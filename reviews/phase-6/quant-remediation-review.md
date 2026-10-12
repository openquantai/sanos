# Phase 6 quantitative remediation review

**Verdict: PASS**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Scope: settled uncommitted Phase 6 implementation after remediation  
Network/WRDS access: none

## Resolution of prior findings

### 1. Reference dtype boundaries — resolved

The initial WIDB no-arbitrage adapter still promotes expiry, normalized strike,
pure bid and pure ask inputs to Float64, matching `mk_arb_free_day`. The smooth
producer now requires and passes the persisted Float32 expiry/strike/bid/ask
arrays without widening. `_float32_column` validates both the Polars and NumPy
dtypes, and it belongs to the smooth transform's version closure.

The isolated pinned surface probe now accepts an explicit input dtype. Tests
exercise initial fit with Float64 and smooth fit with Float32, and compare
supports, densities, masks, fits and diagnostics to the pinned implementation.
An independent capture test also proves that all four smooth inputs reach
`MkArbFree` as Float32. This closes the prior material density/fit discrepancy
under CD-011.

### 2. OptionsFilter and population alignment — resolved

The standalone filter retains the exact legacy sequence: retained initial
expiry, inclusive business-DTE horizon, Float32 spot normalization, configured
ATM/fixed NMS interval, volume, open interest, optional AM removal, minimum
per-expiry population, then the stable cap. `min_vega_sqrt_var` remains the
documented identity-only no-op.

The new isolated filter probe constructs the pinned enriched schema and invokes
the actual pinned `OptionsFilter._filter`. The regression compares the exact
selected row set and bitwise Float32 normalized spot, previous spot, strike,
forward and four real-book fields. Separate tests cover ordered first-rejection
reasons, binding cap reservation, no-op vega, a late-row rejection, initial
boundary masking and a later expiry dropped by both initial and smooth fits.
Input-index/null-state alignment for the dropped expiry matches the pinned
surface result.

### 3. DLV inverse and reconstructed state — resolved under CD-022

CD-022 records the user-approved correction: the unequal-grid inverse now sends
the lower-spacing coefficient to `w_minus` and upper-spacing coefficient to
`w_plus`. The forward DLV encoding, fitted density, option population and smooth
fit are unchanged. `SanosPricer` was bumped from 0.8 to 0.9 and no defective
legacy inverse mode is exposed.

The irregular-support regressions recover density masses and off-grid calls at
near machine precision. I additionally decoded the actual persisted
`initial_surface_dlv_squares` from the fitted-day fixture: maximum density
errors were `4.93e-16` and `2.05e-15`; maximum off-grid call errors were
`1.67e-16` and `5.55e-17`; stored and recomputed full DLV vectors were exactly
equal, including zero boundaries. This satisfies the reconstructing-state
contract.

### 4. Solver diagnostics and failures — resolved

The optional diagnostic sink is numerically neutral on a successful solve.
Tests now force and verify:

- primary HiGHS `SolverError` followed by successful CLARABEL fallback;
- primary `user_limit` followed by successful fallback;
- terminal fallback `user_limit` with unsuccessful status;
- terminal unexpected fallback exception;
- an infeasible non-fallback solve; and
- fitted-day failure propagation with structured stage, solver, fallback,
  status and objective evidence in the `DailyDataError`.

Successful-path behavior is unchanged. `SanosLPSolver` owns the expanded event
contract at version 1.1, and its dependency closure propagates through
`MkArbFree` and the fitted artifact identity. No solver failure becomes a
silent day exclusion.

## Quantitative contract confirmed

- The production order remains initial
  `MkArbFree(surface, smoothness=None)` -> exact `OptionsFilter` -> smooth
  `MkArbFree(surface, smoothness=0.25, surface_expiry_linear=True)`.
- The filter does not apply `mask_arb_free`. The smooth fit consumes all filter
  survivors. `smooth_model_mask` intersects selected, initial and smooth masks;
  side activity uses selected plus the initial mask and valid real side book,
  not the final smooth mask.
- Spot normalization and direct real-book conversion use Float32 arithmetic.
  Direct books are `C/(DF*F)`; synthetic pure calls remain smooth-fit inputs;
  normalized puts satisfy `put = call - 1 + K/F`.
- Initial quote-aligned fit, volatility and Greek fields are Float32 and use
  the pinned `1e-8` implied-volatility/Greek kernels. Model supports, masses,
  variances, DLV values and diagnostics retain the reviewed Float64 state.
- The compatibility unsmoothed state is the complete pre-filter linear surface,
  not an unapproved post-filter DLV refit. The post-filter state is the complete
  smooth SANOS surface.
- Default `overnight_variance_fraction=0.2` and sensitivities `0.1/0.5` reach
  filter times, both model states and DLV time increments through upstream
  identity. Endpoint failures remain explicit; no equal/zero time is silently
  merged or jittered.
- One daily Polars frame retains audit rows, books, named stage masks/reasons,
  fitted values, complete SANOS/DLV state and deterministic diagnostics.

## Dataset authority and population gates

Per the clarified project contract, the old 891-date dataset is a read-only
migration oracle only. It is never a runtime or data input to the fresh build.
Its 4,651,416 smooth-cache side books, 3,566,258 daily-front books and separate
2,192,458 Daily25 SDLV books remain correctly distinguished as migration
checks. Unexplained differences require review, but an approved correction such
as CD-022 is recorded rather than hidden.

The fresh WRDS-derived dataset becomes the sole canonical SANOS reference only
after immutable publication, validation and both reviews. The historical count
catalog therefore does not prove the future fresh population; the fresh run
must still report stage counts, raw fingerprints and the 2025-01-08 empty
marker. That is a later guarded-data/publication gate, not a Phase 6 offline
implementation blocker.

## Independent offline evidence

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
# 169 passed in 58.37s

WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_model_core.py tests/test_widb_surface.py tests/test_widb_build.py \
  tests/test_reference_inventory.py tests/test_model_reference_parity.py tests/test_runner.py
# 64 passed in 43.56s

ruff check src tests --exclude reference
# All checks passed!

git --git-dir=.git-real --work-tree=. diff --check
python -m compileall -q src tests
# clean
```

The first focused run correctly exposed a stale version assertion after
`SanosLPSolver` moved to 1.1. The assertion was updated to the intended version
ownership and both the focused and full suites then passed.

Signed: `quantagent` — independent quantitative/data reviewer
