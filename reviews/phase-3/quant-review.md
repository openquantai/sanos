# Phase 3 quantitative/data review

**Verdict: PASS**

Reviewer: independent `quantagent`  
Date: 2026-10-12  
Scope: Phase 3 numerical core only; no WRDS, network, GPU, or live/legacy data
access was used.

## Findings

1. **PASS — authoritative numerical behavior and provenance.** The six pinned
   source files reproduce the SHA-256 values recorded in `PORTING.md`, including
   `sanos_tools.py` `e250df0d...`, `sanos.py` `15e885d7...`, and `arbfree.py`
   `d26dd823...`. An AST-normalized comparison of the standalone NumPy/CVXPY
   path to the pinned sources found no output-changing production-path edits.
   The substantive differences are the explicit CD-013 gate, removal of dead
   locals/duplicate overwritten code, standalone imports/constants, formatting,
   and version metadata. Solver objectives, constraints, defaults, fallback
   rules, tolerances, strike construction, and interpolation are retained.

2. **PASS — normalized units, state geometry, and DLV inverse.** The documented
   convention `k=K/F` and `c=C/(DF*F)` matches the numerical API. Each expiry
   carries a strictly increasing normalized support and a probability-mass
   vector with non-negativity, unit mass, and normalized first moment one. The
   documented finite-volume scale, annualized squared-DLV units, interior
   `(N_t-2,)` representation, optional zero-boundary `(N_t,)` representation,
   and implicit tridiagonal inverse match the implementation. Unequal-grid
   round trips, call bounds, monotonicity, convexity, `C(0)=1`, left slope
   `-1`, zero tail, mass, and moment all pass. The newly added `None` versus
   explicit-zero payoff-kernel assertion agrees to `2e-16`.

3. **PASS — smoothing and variance geometry.** `smoothness=None` retains the
   piecewise-linear payoff; `smoothness=0` takes the SANOS smoothing branch but
   reduces to the same payoff; and positive smoothing uses
   `sqrt((total_variance/T)*smoothness)`. The production value `0.25` is tested
   in the LP fitter. Explicit variances are `T*vol^2`; inferred cumulative
   variance is reconstructed from normalized ATM calls with the reference
   `0.001^2` forward-variance floor. These inputs are correctly described as
   smoothing geometry rather than fitted ATM calibration claims.

4. **PASS — no-arbitrage and solver contract.** The fitted densities enforce
   non-negativity, mass one and mean one; call monotonicity/convexity follows
   from that representation, while surface constraints enforce calendar
   monotonicity on the constructed common geometry. `SanosFitter` preserves the
   weighted-L1 objective and DLV bounds. `MkArbFree` preserves its per-expiry
   quote fit, masking and extended-strike logic, followed by the surface
   objective. The primary HiGHS serial dual-simplex settings, KKT tolerance
   `1e-7`, configured timeout, CLARABEL fallback tolerances, accepted statuses,
   and explicit infeasible behavior match the reference. Tests verify feasible
   repricing for `None`, `0`, and `0.25`, per-expiry quote containment, surface
   calendar monotonicity, infeasibility, density repair against an independent
   CVXPY projection, and exact serialization/repricing.

5. **PASS — reference parity bridge and independent probe.** The ordinary
   offline suite runs both the pinned-reference subprocess and standalone
   checks against `reference_numerics_v1.json` at `rtol=1e-13`, `atol=1e-14`.
   I additionally compared the pinned and standalone implementations on a
   three-expiry, unequal and widening support: density inversion, prices for
   smoothing `None`, `0`, and `0.25`, the production `0.25` LP fit, and fitted
   densities all had maximum absolute difference `0.0`. This is numerical-core
   parity; full 891-date frame/population parity remains correctly assigned to
   the later daily integration and dataset phases.

6. **PASS — JAX scope and compatibility gates.** The retained JAX backbone
   evaluates the shared tridiagonal transition in x64 and its gradient agrees
   with finite differences. The full reference optimizer is not authoritative
   for the 891-day producer and contains both CD-015 defects; exposing it as an
   explicit `NotImplementedError` is the correct non-silent gate. Similarly,
   the broken ndarray diagnostic branch is outside the production
   list-per-expiry call graph and is explicitly blocked by CD-013. Neither gate
   is represented as a repair or as approved new semantics.

7. **PASS — CD-014 is preserved.** Exactly zero time returns intrinsic value;
   every positive time before the first fitted expiry uses the first density;
   interpolation between fitted expiries and final-density extrapolation are
   unchanged. The committed test pins this for `SanosPricer`. An independent
   probe also checked `ArbFreePricer` in both `expiry` and `surface` modes: the
   pre-first versus first-expiry maximum difference was `0.0`, and the zero-time
   intrinsic difference was `0.0`. Changing this jump still requires explicit
   approval and a new model version under CD-014.

8. **PASS — versioning and static boundary.** The three module-level NumPy
   backbone helpers are now individually versioned and declared in
   `BackboneNumpy`'s dependency graph; the live resolved version confirms those
   edges. `SanosPricer`, `SanosFitter`, `MkArbFree`, density repair and the
   tridiagonal kernel have resolved dependencies. The exact mypy exemptions are
   limited to the copied dynamic compatibility modules; the strict remainder,
   Ruff, compilation, and diff checks pass. Mechanical CRLF/LF normalization
   did not affect behavior.

## Commands and evidence

All commands were run from `/home/hans/python/sanos` with `WRDS_OFF=1` where a
Python process was involved.

```text
python -m pytest -q -p no:cacheprovider
86 passed in 11.01s

python -m pytest -q -ra -p no:cacheprovider tests/test_model_core.py
14 passed in 2.57s
# no skips; optional JAX coverage executed

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 37 source files

ruff check --no-cache src tests scripts
All checks passed!

python -m compileall -q src tests scripts
git --git-dir=.git-real --work-tree=. diff --check
# both exit 0

sha256sum reference/quant_data/quantdata/utils/backbone.py \
  reference/quant_data/quantdata/sanos/{tridiag.py,sanos_tools.py,sanos.py,arbfree.py,sanos_jax.py}
# all six hashes match PORTING.md

/tmp/sanos/phase3-quant-review/probe.py
{'max_density_error': 0.0,
 'price_errors': {'None': 0.0, '0.0': 0.0, '0.25': 0.0},
 'fit_price_error': 0.0,
 'fit_density_error': 0.0}
```

## Residual risks and later gates

- CD-006 remains open: the legacy short identity omits solver timeout. Phase 2's
  canonical artifact/configuration identity must govern pipeline artifacts.
- CD-013 and CD-015 remain intentionally unavailable, not approved fixes.
- The tests establish deterministic numerical parity, not real-market
  population or all-891-date output parity. Those require the Phase 5/6 daily
  normalization/filtering integration and Phase 7 dataset validation.
- A committed `ArbFreePricer` CD-014 regression mirroring the independent probe
  would strengthen future change detection, but its absence is not a Phase 3
  acceptance blocker because the implementation is reference-equivalent and
  both modes were independently exercised here.

## Remediation re-review — 2026-10-12

**Final verdict: PASS**

The live remediation resolves both code-review findings without changing the
numerical algorithms:

1. **Version ownership and dependency closure — resolved.** The one-minute
   model-year convention is now owned by versioned
   `one_minute_year_fraction`, closing transitively over the versioned 252-day
   convention. `SanosPricer` and `ArbFreePricer` directly depend on that
   convention and the versioned `1.0-blocked-cd013` market-fit diagnostic.
   `SanosFitter` and `MkArbFree` directly depend on the one-minute convention
   and versioned density-floor policy. Exact direct-dependency tests now pin all
   four closures. I independently inspected the live resolved dependency maps;
   they equal the asserted sets and expose CD-013 in both public pricer version
   strings.

2. **Durable production-smoothing oracle — resolved.** The isolated subprocess
   imports the pinned `reference/quant_data` implementation, creates market
   values and LP fits for smoothness `0` and production `0.25`, and returns
   densities, DLV squares, total variances, and off-grid calls. The standalone
   parity test consumes those reference-derived mids and outputs; it does not
   use standalone results as its oracle. Both modes agree exactly in this
   environment (maximum absolute errors `0.0` for every compared density, DLV,
   variance, and off-grid price), within the committed absolute tolerance
   `1e-12`.

3. **No semantic spillover.** The remediation changes only version identity,
   dependency tests, and reference evidence. The pricing kernels, LP
   objectives/constraints, solver settings, strike geometry, density repair,
   CD-014 behavior, and CD-015 gate are unchanged. No new WIDB, artifact,
   network, or data-source assumption was introduced.

Re-review commands:

```text
WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
87 passed in 15.66s

WRDS_OFF=1 python -m pytest -q -ra -p no:cacheprovider \
  tests/test_model_reference_parity.py tests/test_model_core.py
15 passed in 5.45s

# Independent pinned-reference/standalone error probe
smoothness 0.0: density=0, DLV=0, price=0, variance=0
smoothness 0.25: density=0, DLV=0, price=0, variance=0

ruff check --no-cache src tests scripts
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 38 source files

python -m compileall -q src tests scripts
git --git-dir=.git-real --work-tree=. diff --check
# both exit 0
```

Signed: `quantagent` — independent quantitative/data reviewer
