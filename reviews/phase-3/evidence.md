# Phase 3 evidence — SANOS and no-arbitrage numerical core

Review state: complete. After one remediation round, the code and quantitative
reviewers both recorded `PASS`. No WRDS connection, external network access,
GPU process, or reference-repository write occurred.

## Scope and provenance

Phase 3 copies the pinned numerical implementation into `sanos.model` and
separates it from WIDB, artifact, and network concerns. `PORTING.md` records the
six source/target mappings and source SHA-256 values. Each source was copied
byte-for-byte and verified before standalone imports, formatting, version
dependencies, and compatibility gates were applied.

The port includes:

- NumPy array/backbone helpers and the shared tridiagonal transition;
- `SanosPricer`, DLV/density conversion, variance geometry, density repair,
  market-fit diagnostics, and `SanosLPSolver`;
- the CVXPY/HiGHS `SanosFitter`;
- `ArbFreePricer`, `MkArbFree`, per-expiry and surface constraints, strike
  expansion, masks, diagnostics, pricing, and dictionary serialization; and
- a minimal optional JAX backbone for shared transition values and gradients.

JAX remains an optional runtime extra, but is included in the test and
development extras so CI cannot silently skip retained-path parity coverage.

All deterministic numerical components have resolved `cdxcore.version`
dependencies. Representative resolved versions are:

```text
SanosPricer 0.8 { _compute_market_fit: 1.0-blocked-cd013,
  one_minute_year_fraction: 1.0 { business_days_per_year: 1.0 },
  solve_sanos_tridiagonal: ... }
SanosFitter 0.41 { SanosLPSolver: 1.0, SanosPricer: ...,
  density_floor_enabled: 1.0, one_minute_year_fraction: ...,
  warn_and_fix_density: ... }
SanosLPSolver 1.0
ArbFreePricer 0.81 { _compute_market_fit: 1.0-blocked-cd013,
  one_minute_year_fraction: ... }
MkArbFree 0.51 { ArbFreePricer: ..., SanosLPSolver: 1.0,
  density_floor_enabled: 1.0, one_minute_year_fraction: ...,
  warn_and_fix_density: ... }
```

`docs/model-numerics.md` documents normalized units, shapes, equations,
smoothing, DLV inversion, interpolation behavior, LP choices, tolerances,
serialization, and failure modes. The architecture and migration traceability
documents link or record this contract.

## Reference parity bridge

The Phase 1 subprocess probe imports only the pinned `reference/quant_data`
snapshot and checks `tests/fixtures/reference_numerics_v1.json` at
`rtol=1e-13`, `atol=1e-14`. The new standalone model independently evaluates
the same inputs and checks the same fixture at those tolerances. Both tests run
in the ordinary offline suite, providing an executable pinned-reference →
frozen-values → standalone bridge for densities, DLV squares, calls, and
density repair.

The same isolated pinned-reference probe now emits deterministic LP results for
smoothness `0` and production `0.25`. A separate standalone test consumes the
reference mids and compares fitted densities, DLV squares, total-variance
geometry, and off-grid query prices at absolute tolerance `1e-12`. The
standalone implementation is therefore never both subject and oracle for the
positive-smoothing parity assertion.

Additional synthetic tests cover:

- unequal-grid DLV/density/call round trips, interior and full-vector DLV
  geometry, zero boundaries, and the frozen pre-first-expiry jump;
- call bounds, strike monotonicity, convexity, boundary values, density
  nonnegativity, unit mass, normalized first moment, and calendar monotonicity;
- the density repair's independent CVXPY L2 optimum with custom floor, mass,
  and moment;
- explicit and inferred total-variance geometry;
- deterministic LP repricing for smoothness `None`, `0`, and production `0.25`;
- no-arbitrage quote containment, retained geometry, infeasible status,
  reference strike trimming, and exact serialized-price round trip;
- NumPy transition equality to a dense solve and JAX value/gradient agreement
  with finite differences; and
- resolved version graphs and both explicit compatibility gates.

## Compatibility findings and gates

The port preserves CD-014: positive times before the first fitted expiry use
the first density, while exactly zero uses intrinsic value.

The already registered ndarray market-fit branch uses undefined reference
locals. It now raises an explicit `NotImplementedError` citing CD-013 instead
of accidentally exposing a `NameError`; the production list-per-expiry branch
is retained and tested.

The copy audit found a second non-production defect, CD-015: the optional JAX
fitter appends to `weights=None`, and after optimization evaluates the initial
state rather than `result_raw`. The authoritative 891-day producer uses the LP
path. The full JAX optimizer is therefore an explicit blocked stub; shared JAX
transition and gradient support remains tested. Repairing either blocked
interface requires user approval and a new version. This phase does not claim
such a repair.

CD-006 remains open: legacy unique hashes omit the solver timeout. Phase 2's
canonical artifact configuration will govern the eventual pipeline identity;
no choice is made here.

## Static-analysis boundary

The copied 3,973-line dynamic NumPy/CVXPY core produced 253 mypy findings,
principally from its array-backbone protocol, CVXPY expression overloads, and
legacy dynamic helper types. Converting it wholesale while establishing
numerical parity would add avoidable semantic risk. Exact mypy module overrides
isolate only the five copied dynamic compatibility modules; strict checking still
covers the new package API, compatibility gate, constants, all Phase 2 code,
tests, scripts, and every future WIDB module. Ruff ignores only inherited long
lines in five exact numerical files. Runtime and quantitative behavior remains
covered by the tests above.

## First-review remediation

The quantitative reviewer recorded `PASS`. The code reviewer recorded
`CHANGES_REQUESTED` for two evidence/identity gaps while independently finding
exact numerical parity on the tested positive-smoothing and no-arbitrage
markets.

- `SanosPricer` and `ArbFreePricer` now depend directly on the versioned
  market-fit diagnostic, whose version explicitly records the CD-013 gate.
  The one-minute model-time convention has versioned ownership and is a direct
  dependency of both pricers and both fitters. `SanosFitter` and `MkArbFree`
  depend directly on the versioned density-floor policy. Exact direct-key tests
  prevent those closures from regressing.
- The pinned reference subprocess now computes deterministic LP fits for
  smoothness `0` and production `0.25`. A new offline test feeds the
  reference-derived mids into the standalone fitter and compares densities,
  DLVs, variances, and off-grid calls at explicit tolerance. This supplies the
  missing durable positive-smoothing oracle; standalone self-consistency tests
  remain as separate invariant coverage.

The remediation changes version identity and test evidence only; it does not
alter the frozen pricing, fitting, or no-arbitrage algorithms.

## Verification

Run from `/home/hans/python/sanos` on 2026-10-12 with `WRDS_OFF=1`:

```text
pytest -q
87 passed in 18.30s

pytest -q tests/test_model_core.py
14 passed in 1.90s

ruff check --no-cache src tests scripts
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 38 source files

python -m compileall -q src tests scripts
(exit 0)

git --git-dir=.git-real --work-tree=. diff --check
(exit 0)
```

The mypy tool remains installed only beneath project-specific
`/tmp/sanos/phase-2-tools`; no user or project environment was mutated.

## Reviewer focus

- Confirm that formatting/import/version adaptations did not change the
  authoritative LP algorithms, tolerances, solver choices, strike geometry, or
  warning/failure behavior.
- Confirm that CD-013 and CD-015 are correctly isolated from the 891-day path
  and that explicit failure is preferable to exposing known defective APIs.
- Challenge the exact typing boundary and test sufficiency, especially
  degenerate density/LP cases and smoothness geometry.
- Confirm the documented normalization, DLV units, interpolation semantics,
  and parity tolerances before this core is integrated into daily WIDB stages.
