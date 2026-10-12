# Phase 1 independent quantitative/data re-review

**Verdict: PASS**

Review date: 2026-10-12
Pinned reference revision:
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`

This re-review covers the Phase 1 reference inventory, read-only adapter,
traceability, numerical contracts and offline evidence. It does not approve a
pipeline implementation, an open compatibility decision, public release, or
WRDS access. No WRDS connection was opened and `quant_data` was inspected
read-only.

## Findings

### 1. Resolved — the exact direct producer dependency trace is complete

The exact producer is correctly identified as
`quantdata/dysanos/flex/run_all_history_joint_direct.py` at Git blob
`6b1c408c3ab1d32e92115cb95c91b3472bcf6086`. An AST inventory of that exact
producer has ten direct project-module imports. `producer.direct_dependencies`,
`docs/traceability-matrix.md`, and the pinning test now record exactly those ten
paths, with no missing or extra entry. The four dependencies added after the
prior review and their independently verified pinned blobs are:

| Direct import | Pinned blob |
|---|---|
| `quantdata/dysanos/flex/config.py` | `d9d8b18680f78fb6d23bf532784984a3a86f2518` |
| `quantdata/dysanos/flex/failure_reporting.py` | `cc1bcd7ece1759bc8f544e28b62cbe6169dd4f0f` |
| `quantdata/dysanos/flex/market_encoder.py` | `fdc4457cf25c99646f5f9f1353592508e39842e8` |
| `quantdata/torch/__init__.py` | `aa4fe2a5f49c70b6ea540c2b151bb981a4404386` |

`config.py` supplies the producer's market, geometry, objective, dynamics and
decoder configuration classes; `market_encoder.py` supplies checkpoint hashing
and atomic JSON output; `failure_reporting.py` wraps the CLI; and
`quantdata.torch` supplies `BaseEnvironment`. They are therefore genuinely
direct source dependencies even though some affect execution/provenance rather
than the fitted numerical map.

The other six dependency hashes and the producer blob also match the pinned
revision. The documentation gives all ten an appropriate copy/adapt,
reference-only, or defer-experiment disposition; completing this evidence does
not add them to the standalone runtime.

### 2. Pass — fitted-state producer outputs and the three populations are exact

Independent read-only loading reconfirmed the all-history target payload:

- schema `flex-all-history-joint-direct-targets-v1`;
- 891 dates, exactly equal and identically ordered to the populated smooth-cache
  catalog, from `2022-01-03` through `2025-08-29`;
- `states`: float64, shape `(891, 200)`;
- `spot_refs`: float64, shape `(891,)`; and
- state-byte SHA-256
  `3525f0d28a7fdc9a5535c80d05c9a4ddd83dfb9c59e8b15fd6ffcfa568477cb7`.

The corresponding checkpoint is schema `flex-joint-direct-dynamics-v1`, phase
`complete`, with 891 market fingerprints and float32
`daily_state_embedding.weight` shape `(891, 200)`. The docs correctly classify
this neural state as an experiment parity oracle, not the smooth SANOS or
discrete DLV state required in a dataset.

The populations remain correctly distinct:

| Artifact | Dates | Books | Meaning |
|---|---:|---:|---|
| smooth daily market cache | 891 populated | 4,651,416 | active call/put books before downstream FLEX selection |
| DySANOS daily-front | 891 | 3,566,258 | European, finite smooth fit, non-0.2D, at most 70D, inclusive AF-ATM NMS `[-3,2]` |
| Daily25 SDLV/JAX | 891 | 2,192,458 | separate 25-expiry/NMS materialization |

The smooth collection still has 892 physical files and one empty marker,
`2025-01-08`; its populated endpoints are unchanged.

### 3. Pass — the complete SANOS/DLV reconstruction contract is now frozen

`docs/data-contracts.md`, `docs/reference-baseline.md`, and CD-014 now distinguish
all state needed for reconstruction:

- common ordered positive model expiries in year fractions, `dt` from zero for
  the first node and from the prior node thereafter, pricing mode and boundary
  flags;
- smooth surface `smoothness`, input **total** variances, and per-expiry surface
  supports/masses `mstrikes`/`mdensities`; expiry mode additionally retains its
  distinct `xstrikes`/`xdensities` and `xix_to_mix` mapping;
- discrete DLV's own `xstrikes`, probability masses and non-negative annualized
  squared-DLV values in `1/year`, with either interior encoding or explicit zero
  boundary values; and
- mass normalization and unit-first-moment invariants.

The exact legacy expiry contract is also explicit: intrinsic below
`ONE_MINUTE`; the first fitted density for every later positive query before the
first node; the exact node within tolerance; linear interpolation of
density-implied call prices between nodes; and the final density after the last
node. For smooth pricing, the interpolated total variance is evaluated at the
clipped model time and converted to annual volatility, while the Black kernel
uses the actual query horizon. This is sufficient to reconstruct normalized
model prices and IVs without redundant public price/IV arrays.

The previous contradictory instruction to store public fitted-price arrays is
gone. CD-014 correctly freezes the pre-first-density jump as initial parity
behavior and requires a new version and approval before changing it.

### 4. Pass — the broken vector market-fit branch is registered

CD-013 accurately records that the NumPy-vector branch of
`_compute_market_fit` references undefined `f`, `asks_t`, and `bids_t`, whereas
the list-of-expiries branch used by `smooth_interpolated_market` is valid. It
blocks both silent copying and silent repair, and requires a focused regression
test and versioned decision if the interface is retained. This is the correct
treatment for a confirmed defect outside the authoritative production branch.

### 5. Pass — data ordering, normalization, clock, and public boundary remain faithful

The documented path still preserves OptionMetrics identity, quotes, closes,
forwards and zero curves; the raw/clean/enrich/no-arbitrage/filter/smooth-fit
ordering; and the reference normalization

```text
k = K/F
c = C/(DF*F)
c - p = 1 - k.
```

The smooth fit retains `smoothness=0.25`. Business DTE counts sessions in
`(observation, expiry]`; AM settlement subtracts
`1 - overnight_variance_fraction`, so the compatibility default `0.2` exactly
reproduces the legacy `0.8` subtraction. The complete internal dataset retains
otherwise eligible 0.2D observations, while the daily-front profile excludes
them downstream.

The public projection remains deliberately lossy and has no absolute spot,
source identifiers, raw quotes/mids, liquidity fields, vendor analytics, or
redundant fitted-price/IV arrays. Its complete state plus dimensionless geometry
is sufficient for the advertised normalized SANOS/DLV and spread experiments,
but not vendor-book reconstruction or contract replay. CD-004 and CD-009 still
block the proposed forward-adjusted return and `DF*(F/S)` unit bridge pending
explicit user approval. Phase 1 approval would not approve either choice or
redistribution rights.

## Commands and checks run

All checks were offline and run from `/home/hans/python/sanos`.

- `WRDS_OFF=1 python -m pytest -q -p no:cacheprovider` — **21 passed**.
- `ruff check --no-cache src tests scripts` — **passed**.
- `python -m compileall -q src tests scripts` with bytecode redirected below
  `/tmp/sanos/phase1-quant-rereview/` — **passed**.
- `git --git-dir=.git-real --work-tree=. diff --check` — **passed** before this
  review-file update.
- Loaded both daily-front PyTorch artifacts on CPU with `weights_only=True` and
  independently checked schemas, phase, shapes, dtypes, exact date sequence and
  state digest.
- Parsed imports from the exact producer blob and compared them with
  `producer.direct_dependencies`; both sets contain exactly the same ten paths,
  with no missing or extra entries.
- Recomputed all ten direct project-module blob IDs at the pinned Git revision.

`mypy` was not installed in either available review interpreter, so the evidence
bundle's recorded successful run was not repeated. This is not a quantitative
blocker; the ordinary offline suite, Ruff and compilation passed.

## Gate condition and residual risks

The Phase 1 quantitative/data gate passes. This approves the reference
inventory and executable parity baseline only. It does not approve any open or
provisional compatibility decision, public redistribution, WRDS access, or a
later pipeline implementation.

Residual later-phase risks remain: fresh WRDS history can differ from the frozen
vendor snapshot; degenerate LP parity requires price/mask invariants as well as
raw density comparison; legacy float32 boundaries can move selections and LP
solutions; and mathematical transformation does not establish redistribution
rights.
