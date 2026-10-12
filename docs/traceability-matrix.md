# `quant_data` migration traceability

This matrix is based on immutable source revision
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`.  The exact 55-path snapshot is
under `reference/quant_data/`; it is not importable by the installed package.

Actions mean:

- **copy/adapt**: begin with the proven implementation, move it into `sanos`,
  change boundaries and names, and prove numerical parity;
- **replace**: preserve the required behavior with the standalone artifact/CLI
  architecture rather than carrying the legacy framework;
- **reference only**: retain history/evidence but do not put it on the new
  runtime path; and
- **defer experiment**: not needed to produce the curated dataset, but available
  for a separately scoped experiment adapter.

## End-to-end production call graph

| Order | Legacy symbol | Standalone disposition | Contract to preserve |
|---:|---|---|---|
| 1 | `dysanos.download.download` | replace with `sanos raw/intermediate/dataset` CLI | date planning and visible progress; replace implicit cache miss/network behavior with an explicit 2FA-gated raw command |
| 2 | `dysanos.flex.market.load_market_day` | copy/adapt into dataset selection | non-binding upstream 100,000 cap, upstream NMS `[-100,100]`, smoothness 0.25, then exact downstream horizon/NMS selection |
| 3 | `dysanos.market.smooth_interpolated_market` | copy/adapt into daily fit stage | normalized input requirement, separate real call/put books, AF masks, smooth SANOS payload and diagnostics |
| 4 | `widb.options_filter.filter_options_day` | copy/adapt to Polars daily stage | exact filter order, stable sorting/tie behavior, normalization and cap semantics |
| 5 | `widb.arbfree.mk_arb_free_day` | copy/adapt to Polars daily stage | expiry/surface modes, aligned masks/fits, retained/dropped expiry semantics |
| 6 | `widb.day.enrich_day` | copy/adapt to Polars daily stage | implied-forward projection, pure units, expiry thresholds, vols/Greeks and float32 parity boundary |
| 7 | `widb.day.collect_raw_day` | split into durable extraction plus deterministic daily cleaning | same selected result; vendor rows remain minimally transformed in raw Parquet |
| 8 | `widb.client.WIDBClient` methods | copy SQL meaning; replace connection/cache mechanics | OptionMetrics table resolution, security identity, options, spot, forward and zero curve |
| 9 | `dysanos.flex.market.market_day_from_payload` | copy/adapt into dataset/experiment projection | European books, explicit DTE exclusions, model horizon, AF-ATM canonical NMS, quote identity |
| 10 | `dysanos.flex.run_all_history_joint_direct.run` | defer experiment; retain as exact parity provenance | loads the 891 daily-front markets, fits the affine Direct-200 neural state/dynamics model, and writes the checkpoint, target-state history, report and completion marker |

`sanos.widb_sanos_day.fit_sanos_day` and `widb.timeseries` are not on this
production path.  The former is stale/broken; the latter is the older padded
HDF5 experiment materializer. Neither defines the curated dataset. The
Direct-200 neural state at step 10 is an **experiment parity oracle**, not the
smooth SANOS or discrete DLV state that the internal/public dataset must store.

### Exact downstream fitted-state producer

The producer is present at the pinned source revision even though it is outside
the 55-path runtime migration snapshot. Its Git blob is pinned so later
experiment work cannot silently substitute a similarly named workflow:

| Pinned path | Blob | Disposition |
|---|---|---|
| `quantdata/dysanos/flex/run_all_history_joint_direct.py` | `6b1c408c3ab1d32e92115cb95c91b3472bcf6086` | exact producer; defer experiment |
| `quantdata/dysanos/flex/config.py` | `d9d8b18680f78fb6d23bf532784984a3a86f2518` | exact model/market/objective configuration; selected data contracts copy/adapt, neural producer use defer experiment |
| `quantdata/dysanos/flex/failure_reporting.py` | `cc1bcd7ece1759bc8f544e28b62cbe6169dd4f0f` | guarded producer CLI wrapper; defer experiment |
| `quantdata/dysanos/flex/joint_direct.py` | `e2a1a35959cf506a2abef7b65bfa40964ab31b0b` | fitting/training dependency; defer experiment |
| `quantdata/dysanos/flex/run_shared_strikes_full_encoder.py` | `cb2fe72f6307bac0ea07c275421370906e17b68f` | market-date/loading dependency; reference only until experiment port |
| `quantdata/dysanos/flex/market.py` | `67f30c151d1e5bfb33dbd31ba3fb8d4c76c9a4f3` | market population; selected logic is copy/adapted, neural producer use is defer experiment |
| `quantdata/dysanos/flex/market_encoder.py` | `fdc4457cf25c99646f5f9f1353592508e39842e8` | checkpoint digest and atomic report writer; defer experiment |
| `quantdata/dysanos/flex/model.py` | `d5f65c8df9d8365c304cd6a04feb63b33011d595` | neural model; defer experiment |
| `quantdata/dysanos/flex/training.py` | `b4d9c0039e1d07e324ddce2626ad8b8f9ede127d` | neural training/materialization; defer experiment |
| `quantdata/dysanos/flex/weekly_validation.py` | `8756e5f9bc241dd9b42f06e01dcf680d6c4dead3` | exact-week dynamics pairing; defer experiment |
| `quantdata/torch/__init__.py` | `aa4fe2a5f49c70b6ea540c2b151bb981a4404386` | `BaseEnvironment` runtime dependency; defer experiment |

The target payload schema is
`flex-all-history-joint-direct-targets-v1`: its 891 dates exactly equal the
ordered populated smooth-cache dates, `states` is float64 shape `(891, 200)`,
and `spot_refs` is float64 shape `(891,)`. The state byte digest is
`3525f0d28a7fdc9a5535c80d05c9a4ddd83dfb9c59e8b15fd6ffcfa568477cb7`.
The complete checkpoint is `flex-joint-direct-dynamics-v1`; its float32
`daily_state_embedding.weight` has shape `(891, 200)` and it records 891 market
fingerprints. The machine-readable inventory pins these facts and every file
hash.

## Source-module inventory

| Imported path | Action | Target/responsibility |
|---|---|---|
| `quantdata/__init__.py` | replace | `sanos` package metadata; no global legacy environment object |
| `quantdata/biz_calendar.py` | copy/adapt | `sanos.calendar`; NYSE sessions, half days, NMS transforms, business-DTE clock; parameterize the AM fraction |
| `quantdata/pool.py` | replace | bounded standard-library executor and explicit progress journal |
| `quantdata/utils/__init__.py` | reference only | no compatibility namespace |
| `quantdata/utils/backbone.py` | copy/adapt | minimal NumPy/JAX array helpers used by SANOS |
| `quantdata/utils/cvxpy.py` | copy/adapt only if still needed | narrowly patches CVXPY `Sum.shape_from_args` to avoid a NumPy uninitialized-memory warning; it does not translate solver statuses or errors |
| `quantdata/utils/pd.py` | replace | Polars schemas and validation; retain observable schema rules |
| `quantdata/jax/__init__.py` | replace | lazy optional JAX imports under the `jax` extra |
| `quantdata/jax/bbjax.py` | copy/adapt | JAX backbone helpers used by retained numerical path |
| `quantdata/torch/__init__.py` | defer experiment | lazy Torch environment/model helpers, never a dataset dependency |
| `quantdata/torch/ffw.py` | defer experiment | optional neural experiment configuration |
| `quantdata/sanos/__init__.py` | copy/adapt | public numerical API and density-floor compatibility constant |
| `quantdata/sanos/tridiag.py` | copy/adapt | SANOS tridiagonal solve with NumPy/JAX parity tests |
| `quantdata/sanos/sanos_tools.py` | copy/adapt | `SanosPricer`, `SanosLPSolver`, DLV/density transforms, density repair, market-fit diagnostics |
| `quantdata/sanos/sanos.py` | copy/adapt | `SanosFitter` and production objectives/defaults |
| `quantdata/sanos/sanos_jax.py` | copy/adapt | optional JAX fitter path and gradient parity |
| `quantdata/sanos/arbfree.py` | copy/adapt | `ArbFreePricer`, `MkArbFree`, strike expansion and surface constraints |
| `quantdata/sanos/widb_sanos_day.py` | reference only | stale non-production path; do not migrate behavior |
| `quantdata/widb/__init__.py` | replace | explicit standalone exports, no `SubDir` ownership |
| `quantdata/widb/bad.py` | copy/adapt | versioned known-bad observations and latest-valid guard |
| `quantdata/widb/client.py` | copy/adapt | SQL/table semantics only; guarded connection manager and raw Parquet replace decorators |
| `quantdata/widb/day.py` | copy/adapt | deterministic clean/enrich daily Polars stages |
| `quantdata/widb/arbfree.py` | copy/adapt | aligned daily no-arbitrage stage in the same frame |
| `quantdata/widb/options_filter.py` | copy/adapt | final daily filter/normalization stage |
| `quantdata/widb/timeseries.py` | replace | partitioned curated dataset assembly; HDF5 only in experiment layer |
| `quantdata/widb/testing/__init__.py` | copy/adapt | offline fixture exports |
| `quantdata/widb/testing/generate.py` | reference only | legacy live/cache fixture regeneration harness that invokes WIDB stages; write a new offline synthetic fixture generator rather than copying its network behavior |
| `quantdata/widb/testing/scope.py` | replace | pytest markers and explicit live-WRDS scope |
| `quantdata/widb/README.md` | adapt documentation | source field meanings and filtering provenance |
| `quantdata/dysanos/__init__.py` | replace | standalone package locations/artifact store |
| `quantdata/dysanos/market.py` | copy/adapt | one-frame daily smooth-market fit |
| `quantdata/dysanos/download.py` | replace | human-operable `sanos` CLI and guarded workflow |
| `quantdata/dysanos/README.md` | adapt documentation | standalone workflow and model overview |
| `quantdata/dysanos/MARKET.md` | adapt documentation | daily payload/schema/units contract |
| `quantdata/dysanos/flex/__init__.py` | reference only | legacy aggregate import depends on modules outside the reviewed slice |
| `quantdata/dysanos/flex/config.py` | copy/adapt selected configs | geometry, market and objective contracts used by public capabilities/experiments |
| `quantdata/dysanos/flex/market.py` | copy/adapt | exact downstream option population and eligibility audit |
| `quantdata/dysanos/flex/sanos.py` | copy/adapt | interpreting geometry, support construction and time interpolation |
| `quantdata/dysanos/flex/fixed_grid_dlv.py` | defer experiment | optional fixed-grid DLV path experiment |
| `quantdata/dysanos/flex/fixed_relative_grid.py` | defer experiment | optional representative relative-strike grid |
| `quantdata/dysanos/flex/floating_grid_dlv.py` | defer experiment | optional floating-grid path experiment |
| `quantdata/sdlv/daily/__init__.py` | reference only | separate Daily25 research implementation, not core WIDB reproduction |
| `quantdata/sdlv/daily/artifacts.py` | defer experiment | source-identity conventions may inform HDF5 experiment manifests |
| `quantdata/sdlv/daily/train.py` | reference only | the NPZ it builds is a distinct 2,192,458-book materialization |
| `quantdata/sdlv/daily/README.md` | reference documentation | distinguishes the Daily25 experiment from DySANOS daily-front data |

## Required symbol/configuration map

| Legacy symbol or value | Action and acceptance oracle |
|---|---|
| `DAYS_PER_YEAR=252` | preserve in model/business-time coordinates |
| `AM_EXIRY_SUBTRACT=0.8` | express as `1 - overnight_variance_fraction`; default `0.2` must be bit/numerically compatible and PM expiries unchanged |
| `NormalizedStrikes`, `NMS`, `NMS5` | copy/adapt; test forward/back transforms and endpoint selection |
| `get_business_days`, `count_business_days_between`, `add_business_days`, `prev_business_day`, `DateRange`, `get_half_days`, `session_open_close` | copy/adapt with pinned-calendar fixtures and previous-close dependency planning |
| `WRDSCredentials` | replace with credential discovery that never serializes secrets |
| `_pick_optionm_library`, `_tables_in_library`, `_resolve_table` | copy/adapt and mock-test `optionm_all`/`optionm` plus annual/nonannual table variants |
| `WIDBClient.lookup_secid`, `options`, `zero_curve`, `forwards`, `spot_price` | preserve query fields/units; use resumable raw Parquet and guarded leases |
| `interpolate_zero_curve` | preserve linear rate interpolation and `exp(-rate*days/36500)` |
| `RawDayFilter(min_bid=0.011, drop_irregular_expiries=True)` | preserve and include canonical config in identity |
| `collect_raw_day` | split download/clean while reproducing row population, dtype, sort and diagnostics |
| `ExpiryFilter(10,4,1e-6,0.01,5.0)` | preserve thresholds and pre/post validity application |
| `enrich_day` | preserve inverse-spread weighted forward, put-call conversion, pure prices, NMS and Greeks |
| `MkArbFree`, `ArbFreePricer`, `mk_arb_free_day` | copy/adapt; parity on fits/masks/prices and invariant tests |
| `OptionsFilter`, `normalize_day`, `filter_options_day` | preserve order and legacy simple `spot_ret`; public forward-adjusted return is a separate field; `min_vegaSqrtVar` remains an explicitly registered no-op pending CD-012 |
| `SanosPricer` | copy/adapt; price, density, DLV and unequal-grid round trips |
| `SanosLPSolver` | copy/adapt; feasible/infeasible behavior, quote containment and solver selection |
| `repair_density_nonnegative`, `warn_and_fix_density` | copy/adapt; fixture and invariant tests |
| `make_variances` | copy/adapt; preserve interpolation/ATM input semantics |
| `SanosFitter`, JAX `fit_dlvsqs` | copy/adapt; NumPy/JAX value and gradient parity |
| `MARKET_SCHEMA_VERSION=4` | decode as legacy reference; new daily schema receives its own version |
| `smooth_interpolated_market(smoothness=0.25)` | preserve production fit and all direct book/diagnostic fields in one daily frame |
| `FlexMarketConfig(max_num_options=100000, smoothness=0.25)` | preserve for the daily-front reference; exclusions are explicit per dataset/experiment |
| `market_day_from_payload` | preserve active-European/horizon/canonical-NMS population and quote IDs |
| `excluded_business_dtes=(0.2,)` | preserve for the exact daily-front experiment, not as an internal-dataset default |
| daily-front geometry: DTE 1–5 then 10–70, NMS `[-3,2]`, 18 expiries, 200 inner strikes | preserve in exact experiment parity configuration |

Every new deterministic configuration class and transformation uses
`cdxcore.version.version`; canonical artifact identity does not depend on its
human-readable `__str__` value.

## Test migration map

| Imported test | Action | Coverage retained or added |
|---|---|---|
| `tests/test_biz_calendar.py` | copy/adapt in Phase 5 | sessions, half days, scalar/vector business-day arithmetic and AM fraction sensitivity |
| `tests/test_arbfree.py` | copy/adapt in Phase 3 | strike extension, LP fits, calendar and convexity rules |
| `tests/test_arbfree_density_projection.py` | copy/adapt in Phase 3 | nonnegative mass/moment projection and price invariants |
| `tests/test_sanos_utils.py` | reference only | tests only the omitted historical `sanos.old.utils.extract_atm_fwd_vols`; write new current `SanosPricer` price/density/DLV/repair tests |
| `tests/test_dysanos_download.py` | replace/adapt in Phases 2/4 | CLI planning, cache-only behavior, no-network guarantees, progress and errors |
| `tests/test_dysanos_market.py` | copy/adapt in Phases 3/5 | daily smooth payload/frame, direct books, schema and fit diagnostics |
| `tests/test_dysanos_grid_market.py` | reference only | tests vectorized business-day counting in an omitted historical grid loader; cover that behavior in new calendar tests, not as SANOS geometry evidence |
| `tests/test_flex_direct_state.py` | select/adapt in Phases 3/6 | exact market eligibility and model-state interpreting geometry; omit unrelated dynamics training |
| `tests/quantdata/dysanos/flex/test_fixed_relative_grid.py` | defer experiment | representative-grid identity and selection |
| `tests/quantdata/dysanos/flex/test_market_expiry_fit.py` | adapt diagnostic only | tests nominal-expiry bucketing and fit-excess labels; write separate current-pricer tests for expiry interpolation, pre-first jump and quote-fit semantics |

The imported `test_dysanos_download.py` cannot be collected as a standalone
snapshot because legacy `flex.__init__` imports omitted unrelated modules.  That
is a property of the filtered provenance snapshot, not a runtime dependency;
the replacement CLI tests import installed `sanos` only.

The descriptions above distinguish inherited evidence from tests that must be
newly written. In particular, none of the three narrowly scoped historical
tests is accepted as evidence for the complete current SANOS/DLV reconstruction
contract.

## Dependency disposition

| Dependency | Disposition |
|---|---|
| NumPy, SciPy | core numerical kernels |
| Polars, PyArrow | all daily/data Parquet boundaries |
| CVXPY, HiGHS | no-arbitrage and SANOS LP path; solver choice pinned in config/version |
| `cdxcore` | `@version`, Black pricing and small numerical utilities; do not reuse `SubDir.cache` |
| pandas-market-calendars | NYSE software calendar, not a market-data source |
| pandas, `wrds` | `wrds` optional extra only; raw query boundary converts to Polars/Parquet |
| JAX | optional `jax` extra for retained differentiable fitter |
| h5py | reference reader and disposable experiment HDF5 extras only |
| filelock | Phase 2 runner/atomic artifact coordination |
| Torch/Flax/Optax | not dataset dependencies; add only for a named experiment implementation |
| Click | stable human-operable CLI |
| FRED, Yahoo Finance, Massive | excluded from the core WIDB stream; Yahoo may be a separate consumer-side spot adapter |

## Decision boundary

Issues discovered during tracing are recorded in
`docs/compatibility-decisions.md`.  Confirmed legacy defects are not copied or
fixed silently.  Work that does not depend on their resolution may proceed;
the affected implementation remains blocked pending explicit approval.
