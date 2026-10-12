# Phase 9 quantitative/data implementation audit

**Reviewer:** `quantagent`  
**Date:** 2026-10-12  
**Scope:** settled, uncommitted internal Dataset to DySANOS daily-front
Experiment implementation; offline only, with no WRDS/network access and no
changes to `quant_data`  
**Verdict:** **PASS**

All three blockers from the first review are resolved. The implementation now
preserves the pinned nonnegative expiry boundary, has a repository-contained
hash-pinned selector oracle covering the independently important selection
boundaries, and re-establishes the self-contained population contract during
shallow HDF validation. I found no remaining quantitative or data-contract
blocker to committing Phase 9.

## Remediation verification

### 1. Zero-DTE parity — resolved

`DailyFrontDay` now admits finite `expiry_time_y == 0` and
`price_expiry_y == 0` while continuing to reject negative values. Valid-prefix
zeros remain distinct from canonical zero padding through counts and prefix
masks. The end-to-end regression constructs a genuine projected zero-DTE day,
round-trips its pickle-free NPZ shard, writes and validates HDF5, and runs both
the checked-in selector oracle and the pinned consumer. This agrees with the
pinned selector's `expiry_y >= 0` contract; it does not infer a stricter rule
from the historical oracle's zero observed 0DTE count.

### 2. Independent, hermetic selector oracle — resolved

`_run_pinned_selector_oracle` uses the checked-in
`reference/quant_data` snapshot, forces `WRDS_OFF=1`, and verifies the exact
SHA-256 values of the pinned `market.py` and `arbfree.py` sources before using
the reference `market_day_from_payload`, `_atm_curves_and_bounds`, and
`ArbFreePricer`. It compares the SANOS projection with pinned output and also
contains independent boundary assertions for:

- call and put activity independently, including single-sided survival;
- European versus American books;
- unavailable fit state and a nonfinite SANOS fitted value;
- literal `0.2` business-DTE exclusion at exact, just-inside, and just-outside
  `1e-4` tolerance values;
- zero DTE, exact 70D, and just-over-70D;
- inclusive canonical NMS endpoints and their `nextafter` outside values;
- call-before-put ordering without requiring side symmetry;
- first-occurrence `(expiry_index, K/F)` price-location mapping; and
- all five eligibility audit counts.

The boundary projection's fixed expected quote price indices are
`[0, 1, 2, 1]`, and its sides are `[call, call, put, put]`, directly exercising
duplicate call/put price locations and first-occurrence reuse. Its pinned and
SANOS audit tuple is `(8, 8, 4, 1, 1)`. This oracle no longer depends on the
external `~/python/quant_data` checkout. The separate real pinned training
consumer probe remains useful transport/consumer evidence when that larger
read-only checkout is present, but it is no longer the only selection-parity
gate.

### 3. Shallow HDF population semantics — resolved

`validate_experiment_hdf5` now validates, without reopening the source
Dataset:

- every observed expiry is at or below the configured maximum horizon using
  the same `1e-12` boundary convention;
- no observed expiry falls within `1e-4` of any configured literal excluded
  business DTE;
- every quote lies inside its stored inclusive expiry NMS bounds;
- stored NMS bounds reconstruct from the configured NMS range, observed ATM
  annual volatility, first model-node epsilon volatility, and expiry time;
- the selected book population respects the configured source cap; and
- HDF dimensions `E`, `Q`, and `P` equal the exact maxima of valid daily
  counts, rather than merely bounding them.

The tests do not rely on a bad checksum as the rejection mechanism. They
coherently recompute the source-bound market fingerprint and logical HDF
content digest for invalid excluded-DTE, over-horizon, changed-NMS,
out-of-bounds-strike, and non-maximal-dimension payloads; semantic validation
then rejects the corresponding violated contract. This proves the shallow
gate remains meaningful under output-first reuse after the source Dataset is
gone.

### 4. Overnight-fraction ownership — passes

Experiment identity is distinct for each of
`overnight_variance_fraction = 0, 0.2, 0.5, 1`. `_source_gate` also requires
exact equality with the source internal Dataset's daily clock value before
projection. The selector's downstream expiry exclusion remains the separately
configured literal `0.2` business DTE; it is not silently rewritten when the
overnight fraction changes. This is the correct Phase 9 behavior because the
clock has already affected the fitted source Dataset, while the Experiment is
a lossless model-specific materialization of that source state.

## Quantitative contract confirmed

1. Selection begins with `options_filter_eligible`, then applies side activity,
   European style, corresponding finite smooth fit, horizon/literal-exclusion,
   and inclusive NMS gates. Expiry order is increasing; within an expiry calls
   precede puts and strikes increase per side. An expiry survives when at least
   one side survives.
2. Strikes are `K/F`. Bid, ask, and smooth fit are side-specific option cash
   values divided by `DF*F`. Calls and puts remain separate observations.
3. Market-expiry ATM volatility uses clipped total-variance interpolation from
   the smooth SANOS pricer. The model geometry is 18 business-DTE nodes
   `1,2,3,4,5,10,...,70`, 200 inner strike nodes, and inclusive NMS `[-3,2]`.
4. Stored HDF values are float64 with explicit masks/counts and canonical
   padding. The pinned consumer adapter produces the expected float32 tensors,
   int64 indices, boolean side/mask arrays, and matching day/batch shapes.
5. `market_fingerprint` binds the day to the exact source Dataset pointer.
   `training_fingerprint` independently reproduces the pinned objective-only
   scalar/array formula, and validation records the ordered fingerprint digest.
6. Training includes all retained dates. Validation/test masks are empty;
   pre-2025/from-2025 masks are diagnostic only. Weekly pairs are exactly five
   NYSE sessions apart, not merely five retained rows apart.

## Lifecycle and authority

The Experiment accepts exactly one internal Dataset upstream and rejects
public/public-candidate sources. The HDF retains vendor-derived books,
liquidity, expiry identity, and an absolute spot reference, so it remains
internal and is not the public projection.

The fresh, validated, and explicitly promoted SANOS **Dataset** is the
canonical research object. This HDF5 is a self-contained, output-first,
disposable **Experiment**; deleting it cannot alter or demote its source
Dataset. The legacy 891-date Joint Direct-200 artifact is only a migration
oracle and is never a build input or the canonical future artifact.

The frozen legacy facts remain useful Phase 11 comparison evidence: 891 dates,
3,566,258 books split equally between calls and puts, `Qmax=7,898`,
`Pmax=3,949`, `Emax=36`, 729/162 pre-/from-2025 dates, and 861 exact weekly
pairs. The fresh plan's 918 sessions, seven half days, and one policy-owned
typed-empty date imply 917 expected populated Experiment dates before newly
discovered production failures. Fresh production must explain common-period
and vendor-revision differences; it must not force the fresh result back to
891.

## Evidence and commands

- Inspected the settled Experiment projection, build/source gate, HDF writer
  and validator, configuration, CLI, documentation, and the complete Phase 9
  test changes.
- Inspected the checked-in pinned selector subprocess and its exact source
  hashes; no external source or data was mutated.
- Independent focused offline gate:
  `PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider tests/test_experiments.py tests/test_cli.py`
  -> `24 passed in 39.37s`.
- An additional filtered independent run passed `9` applicable tests in
  `36.48s`.
- Coordinator's settled full offline gate: `197 passed in 167.63s`.
- `python -m compileall -q src/sanos/experiments tests/test_experiments.py tests/test_cli.py`,
  `ruff check src/sanos/experiments tests/test_experiments.py tests/test_cli.py`,
  and `git --git-dir=.git-real diff --check` all passed independently.
- `gpuctl resources --json` used its local fallback because the daemon was not
  reachable; all reported RAM/disk checks were `OK`.

No network, WRDS, GPU, or `quant_data` write was performed.

## Residual production risks (not Phase 9 blockers)

- The aggregate 891-date migration comparison and fresh 917-populated-date
  evidence require the fresh-download production phases; unit fixtures cannot
  establish vendor-history equivalence.
- The exact legacy training consumer is too large to be a required copied
  repository dependency. On this host it is hash-checked and exercised when
  present; the required selector and object-contract oracle is now hermetic.
- Shallow validation can prove self-consistency and the recorded selection
  contract, not recreate upstream WRDS eligibility from a deleted Dataset.
  Build-time source validation and optional deep lineage validation retain that
  separate responsibility.

Signed: `quantagent` — independent Phase 9 quantitative/data reviewer
