# Legacy DySANOS migration baseline

This document freezes the legacy evidence used to judge the standalone
implementation. Every referenced artifact is read-only and is a migration
oracle, never an input to a new dataset build. The fresh build must start from
newly downloaded WRDS extracts. Once published and approved, the fresh SANOS
dataset—not these artifacts—is the canonical reference dataset.

## Legacy comparison endpoint

The DySANOS reference interval is **2022-01-03 through 2025-08-29**, inclusive.
It contains **891 populated SPX trading dates**.  The smooth-market cache has
892 physical HDF5 files because `2025-01-08.h5` is an intentional empty marker
for a date in `BAD.DAYS_ALL`; it is not a populated observation.

There are three distinct legacy artifacts which must not be conflated:

| Layer | Population | Role |
|---|---:|---|
| Smooth daily market cache | 891 populated days; 4,651,416 active call/put books | Daily WIDB, no-arbitrage and smooth-SANOS parity oracle before the downstream FLEX horizon and canonical-NMS selection |
| DySANOS daily-front experiment | 891 days; 3,566,258 books | Exact all-history dataset/universe used by the daily-front DySANOS fit |
| Daily25 SDLV/JAX NPZ | 891 days; 2,192,458 books | Separate fixed-25-expiry training materialization; not the daily-front reference universe |

The complete content inventory is in
`reviews/phase-1/reference-market-catalog.json`; its collection digest is
`f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280`.
The catalog itself and the downstream files are pinned in
`reviews/phase-1/reference-artifacts.json`.

## Smooth-market collection

The symbolic root is `legacy_QUANTDATA_TRAINING_DATA`; the relative path is:

```text
dysanos/market/bid=0.011, nat,eth=10,bth=4,mps=1e-06,m ca250824/sm_0.25 6b930b59/SPX
```

All populated payloads use schema version 4 and ticker `SPX`.  Their producer
attribute is:

```text
0.15 { ExpiryFilter: 0.5, MkArbFree: 0.51 { ArbFreePric 0a91a5a8
```

The collection contains 26,732 expiry entries, 2,414,341 strike nodes,
2,325,708 active calls and 2,325,708 active puts.  Of those active books, 8,090
are on the 0.2-business-day AM node.  Removing only that node would leave
4,643,326 books, so it does not explain the 3,566,258-book downstream universe.

The checked-in adapter in `sanos.reference` opens the cdxcore
`PYTREE_HDF5` files strictly in read-only mode.  It recognizes physical empty
markers, validates the date and minimum payload/market schemas, hashes each
file, and exposes no write operation.

## Exact DySANOS experiment artifact

The symbolic root is `legacy_QUANTDATA_TRAINING_DATA`; the relative path is:

```text
flex-shared-strikes/joint-direct/dte1-5-weekly-no-am-all-history-v1
```

The completion marker declares
`flex-all-history-joint-direct-completion-v1`.  The report declares
`flex-all-history-joint-direct-v1`, `num_dates=891`, and the train range
`[2022-01-03, 2025-08-29]`.  It reports exactly 3,566,258 option books.

The selection/configuration recorded by that report is:

- smoothness `0.25`;
- a deliberately non-binding upstream cap of `100000` options per day;
- exclusion of business DTE `0.2`;
- canonical NMS interval `[-3, 2]` calculated from the arbitrage-free ATM
  curve;
- 18 model expiries at business DTEs 1, 2, 3, 4, 5, then 10 through 70 in
  five-day steps; and
- 200 inner strike coordinates.

This downstream universe keeps separate real call and put books, only European
books, expiries inside the model horizon, and strikes inside the canonical NMS
interval.  Those filters, not just the AM-node exclusion, explain why its
population is below the smooth-cache active-book count.

The report hashes are independently recomputed in
`reviews/phase-1/reference-artifacts.json`.  In particular, the fitted
checkpoint SHA-256 is
`395549f12737b8ee301c2c4b35999379f7ef55e7e2cfe0fcda6cf414068ae2ac`.

The exact producer is the pinned
`quantdata/dysanos/flex/run_all_history_joint_direct.py` blob
`6b1c408c3ab1d32e92115cb95c91b3472bcf6086`. It loads the dates and daily-front
markets certified by the native completion artifact, fits the affine
Direct-200 neural state/dynamics model, and writes the checkpoint, target
history, report and completion marker. Its direct source dependencies and blob
IDs are in `docs/traceability-matrix.md` and the machine-readable inventory.

Read-only payload inspection established that
`joint-direct-200-targets.pt` has schema
`flex-all-history-joint-direct-targets-v1`, float64 states of shape `(891,
200)`, float64 spot references of shape `(891,)`, and dates exactly equal to
the ordered 891 populated smooth-cache dates. Its state byte digest is
`3525f0d28a7fdc9a5535c80d05c9a4ddd83dfb9c59e8b15fd6ffcfa568477cb7`.
The complete `flex-joint-direct-dynamics-v1` checkpoint has a float32
`daily_state_embedding.weight` of shape `(891, 200)` and 891 market
fingerprints. This neural fitted state is an **experiment parity oracle**; it
is not the smooth SANOS or canonical finite-DLV projection required in the
internal/public dataset.

## Production path and retained semantics

The production path recovered from the pinned `quant_data` revision is:

```text
dysanos.download
  -> dysanos.flex.market.load_market_day
  -> dysanos.market.smooth_interpolated_market
  -> widb.options_filter.filter_options_day
  -> widb.arbfree.mk_arb_free_day
  -> widb.day.enrich_day
  -> widb.day.collect_raw_day
  -> widb.client.WIDBClient
```

After the daily smooth payload is loaded, `market_day_from_payload` applies the
experiment horizon, European-style condition, explicit 0.2D exclusion, and
canonical NMS interval.  `widb_sanos_day.fit_sanos_day` is not on this path and
is a stale historical implementation.  `widb.timeseries` is also not the daily
fit path; it is the older padded/HDF5 time-series materializer.

The daily stages preserve these reference defaults and ordering until parity
evidence authorizes a change:

1. Reject known bad dates/tickers/contracts.
2. Drop same-day and configured irregular expiries.
3. Drop missing or crossed books.
4. Drop bids at or below `0.011`.
5. Drop bad-settlement rows (`ss_flag != "0"`).
6. Pair calls and puts; attach ACT/365 continuously compounded discount factors
   from zero-curve rates quoted in percent.
7. Infer the forward from the inverse-spread-weighted put-call-parity interval;
   retain the WIDB forward as a reference/diagnostic.
8. Require at least four strikes with both sides traded and ten with either side
   traded, before and after price/forward validity checks.
9. Require minimum pure spread `1e-6` and implied volatility in `[0.01, 5]`.
10. Run expiry/surface no-arbitrage processing.
11. Apply the requested horizon, normalization, NMS, volume, open-interest,
    optional AM filter, minimum-per-expiry rule, and cap in `OptionsFilter`
    order. The advertised `min_vegaSqrtVar` setting is not actually applied by
    the legacy function and is tracked as CD-012.
12. Fit smooth SANOS with production smoothness `0.25`.

Until dataset/time-series assembly, the standalone implementation will retain
all of these quote fields, masks, expiry fields, fits and diagnostics in one
daily Polars frame and persist one Parquet artifact per day.

## Frozen numerical/runtime baseline

`reviews/phase-1/reference-runtime-baseline.json` is the machine-readable
baseline for later ports. It freezes:

- the Phase 1 evidence environment (CPython 3.12.3, cdxcore 0.1.96, NumPy
  2.5.4, SciPy 1.18.1, CVXPY 1.9.3, HiGHS 1.15.1, and the other inspection
  packages);
- every relevant legacy transform/schema version;
- the raw, expiry, option, no-arbitrage, smooth-market and daily-front defaults;
- `SanosPricer`, `SanosFitter`, `MkArbFree` and LP tolerances;
- the primary HiGHS serial dual-simplex configuration, the Clarabel fallback
  configuration, success statuses and fallback triggers; and
- the complete raw/enriched/expiry-fit/surface-fit daily column/dtype groups and
  float32 cast policy.

The original environment that produced the historical HDF5/PyTorch artifacts
was not recorded and is therefore explicitly marked unknown. The captured
versions describe the environment used to regenerate the numerical fixture and
inspect the artifacts; they make evidence drift diagnosable without pretending
to recover unknown provenance.

The initial standalone port preserves the reference float32 daily boundaries.
The solvers may promote selected values to float64 exactly where the legacy
code does. A global float64-through-fit refactor is blocked by CD-011 until its
selection and LP effects are measured and approved.

## Clock and units

Legacy business DTE counts trading sessions in `(observation, expiry]`.  AM
settlement subtracts `0.8`, so a next-session AM expiry has business DTE `0.2`.
The standalone parameter is `overnight_variance_fraction`; its compatibility
default is therefore `0.2`.  The particular DySANOS daily-front experiment then
excludes the 0.2D observations downstream.  These are separate behaviors.

The normalized option convention is:

```text
k = K / F_T
c(T,k) = C(T,F_T k) / (DF_T F_T)
c - p = 1 - k
```

The legacy internal `spot_ret` is only `S_t / S_previous - 1`; it is not the
public forward-adjusted return. Requested public half-spreads use
`half_spread / S`. A model price in discounted-forward units can be converted
by `C/S = DF*(F/S)*C/(DF*F)`. Approved CD-009 therefore retains the
dimensionless `DF*(F/S)` interpreting bridge without exposing absolute spot.

The exact reconstructing state and expiry-pricing policy are frozen in
`docs/data-contracts.md`. In particular, surface support/densities
(`mstrikes`/`mdensities`) are distinct from per-expiry support/densities
(`xstrikes`/`xdensities`), DLV squared values use annualized `1/year` units and
zero boundaries, and both retained pricers use the first fitted density for
positive queries before the first model expiry. That short-time jump is legacy
parity behavior under CD-014, not interpolation from the time-zero Dirac mass.

## Executable evidence

`tests/reference_probe.py` imports only the pinned snapshot and emits a stable
DLV/density/price/repair result.  `tests/test_reference_baseline.py` compares it
with `tests/fixtures/reference_numerics_v1.json` while `WRDS_OFF=1`.
`tests/test_reference_legacy_market.py` verifies decoding, deterministic
cataloguing, empty-marker handling, schema rejection, and that reads do not
change file modification times.
`tests/test_reference_inventory.py` also freezes the exact downstream producer,
payload shapes/date equality, runtime/configuration baseline, solver settings,
and daily schema cast points.

The catalog can be reproduced without modifying the legacy collection using:

```bash
PYTHONPATH=src python scripts/catalog_reference_market.py \
  LEGACY_SMOOTH_MARKET_ROOT reviews/phase-1/reference-market-catalog.json \
  --source-revision de15be96d2f283b6c31e60d3ea47325a0c093ff0 \
  --relative-path 'dysanos/market/.../SPX'
```

No Phase 1 command opens a WRDS connection.
