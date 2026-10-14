# Data contracts

The four and only four storage classes are **raw**, **intermediate**,
**dataset**, and **experiment**. Generated artifacts live outside Git. The
existing 891-date SPX collection from 2022-01-03 through 2025-08-29 is a
read-only migration oracle and is never an upstream input to the fresh build.
The fresh SANOS dataset becomes the canonical reference only after immutable
publication, validation, and dual review.

## Common rules

- Raw, intermediate and dataset tabular artifacts are Parquet read and written
  with Polars. HDF5 is permitted only for disposable experiment
  materializations and the read-only legacy oracle.
- Until time-series assembly, a transformation accepts and returns exactly one
  Polars `DataFrame` for one trading day. One atomic Parquet artifact contains
  all quote data, expiry geometry, masks, no-arbitrage results, SANOS/DLV state,
  fitted values and diagnostics for that day. There are no sibling
  quote/expiry/surface caches.
- A published dataset is self-contained. Successful shallow validation never
  requires raw or intermediate artifacts to exist.
- Dates use ISO calendar dates; timestamps, where unavoidable in vendor data,
  carry an explicit timezone. Monetary vendor fields retain documented vendor
  currency/units in raw data.
- Configuration, schemas and deterministic transforms use `@version` from
  `cdxcore.version`. Artifact identity includes canonical configuration,
  schema/stage version and upstream content identity.
- Every write is temporary-file plus fsync plus atomic replace. A completion
  manifest is written last.

## Raw

Raw data is durable, minimally transformed vendor data. The natural root is
`QUANT_DATA_RAW/wrds/optionmetrics/`. The fresh SPX dependency closure includes:

| Logical source | Candidate WRDS table | Natural partition |
|---|---|---|
| security identity | `securd` or `securd1` | snapshot/query identity |
| option books | `opprcdYYYY` or `opprcd` | table/year/ticker/date; secid in payload |
| underlying close | `secprd` or `secprdYYYY` | table/year/ticker/date; secid in payload |
| vendor forward | `fwdprdYYYY` or `fwdprd` | table/year/ticker/date; secid in payload |
| zero curve | `zerocd` | table/date |

Raw option rows preserve source fields such as symbol, option ID, side, strike,
expiry, bid/ask, volume, open interest, vendor implied volatility/Greeks,
settlement flags, expiry indicator and contract size. OptionMetrics stores
`strike_price` in integer thousandths; the raw layer retains that value and the
daily compatibility stage later divides by 1,000. The reference query's
`exercise_style='E'` is a synthetic constant, not a selected vendor field, so it
is introduced and documented in the later compatibility stage rather than
misrepresented as raw. The raw writer may canonicalize column names/types and
add provenance columns, but does not apply SANOS population filters. A dated
zero-row vendor response is itself a durable raw observation; later daily
stages decide whether it is an expected exclusion or a dataset error.

Each table root has `README.md` describing source/query/fields and a manifest
with query identity, selected columns, partitions, counts and checksums. It does
not contain credentials. The requested start date also plans/downloads the
fresh previous-business-day close and any earlier observations required by the
approved public return rule. CD-004 fixes that formula and the retained-date
policy; the planner exposes its boundary dependencies explicitly.

## Intermediate daily frame

Intermediate artifacts are reproducible and safely deletable. The canonical
daily frame is logically long-form at the paired call/put strike level. It
preserves the observable legacy columns below (with Polars-native dtypes) and
adds stage/provenance fields. Per-day and per-expiry values may repeat by row;
typed list/struct columns hold state whose geometry is not quote-aligned.

### Identity and daily fields

`ticker`, `trading_date`, `spot`, `previous_spot`, `is_half_day`,
`source_secid`, `stage_version`, `configuration_id`,
`raw_options_artifact_id`.

The manifest, rather than every quote row, records all six direct Raw pointers:
security identity, options, current and previous-session spot, forwards, and
zero curve. The row-level field names the options partition explicitly and does
not imply that one identifier covers the whole day.

The internal frame may contain source identifiers and absolute levels. They are
forbidden in the public projection.

### Paired book and expiry fields

`call_symbol`, `put_symbol`, `call_option_id`, `put_option_id`, `expiry_id`,
`expiry_date`, `is_american`, `expiry_is_am`, `strike`, `vendor_forward`,
`discount_factor`, `business_dte`, `calendar_dte`, `time_to_expiry_years`,
call/put bid and ask, and call/put log-volume/open-interest.

The trading-session `is_half_day` flag is distinct from `expiry_is_am`.
Business DTE counts sessions in `(trading_date, expiry_date]`; AM settlement
subtracts `1 - overnight_variance_fraction`. The compatibility default is
`overnight_variance_fraction=0.2`. Calendar DTE and business DTE are never
interchanged.

### Enrichment and filter fields

`implied_forward`, `normalized_moneyness`, `atm_normalized_moneyness`, synthetic
call book, bid/ask implied volatilities and Greeks, combined liquidity fields,
`normalized_strike = K/F`, pure bid/ask price `C/(DF*F)`, and the corresponding
pure Greeks.

Each population-changing predicate is represented by a named boolean eligibility
or rejection field before rows are physically removed at publication. The
initial compatibility ordering is documented in `docs/reference-baseline.md`.

### No-arbitrage, SANOS and DLV fields

The same frame contains no-arbitrage masks, expiry-fit and surface-fit prices,
volatilities/Greeks, the smooth SANOS fitted call/put values, active real-book
masks, fit diagnostics and the complete model state. Model-state list/struct
columns include:

- ordered model expiry times and settlement geometry;
- normalized strike support for every expiry;
- canonical finite-DLV density masses and squared-DLV values, including
  boundaries and model time increments;
- smooth SANOS parameters/density state and separately named smoothing-kernel
  input total variances; and
- solver status, objective, tolerances, quote-containment and failure
  diagnostics.

Smoothing-kernel input variances are not labelled as fitted ATM volatility.
All geometry needed to price/invert either state is explicit.

The minimum exact reconstructing state is not a generic parameter blob:

- Common geometry records ordered positive model expiries `T` in year
  fractions, `dt = diff(T, prepend=0)`, the selected pricing mode, and all
  boundary/mode flags.
- Smooth surface mode records `smoothness`, the separately named input total
  variances `V`, and, for every expiry, ordered normalized support
  `mstrikes[t]` with probability masses `mdensities[t]`. If expiry-mode pricing
  is exposed, it additionally records `xstrikes`, `xdensities`, and
  `xix_to_mix` so the per-expiry support is not confused with the surface
  support. Each mass vector is non-negative, sums to one, and has normalized
  first moment one within the versioned numerical tolerances.
- Discrete DLV mode records its own ordered normalized `xstrikes`, probability
  masses `xdensities`, and annualized squared-DLV values in `1/year` on the
  interior strikes. A full-support encoding is allowed only with explicit zero
  values at both boundaries. The time step for the first expiry is measured
  from zero and later steps from the preceding model expiry.

The legacy pricing rule is part of the state contract. For `t < ONE_MINUTE`, a
call is intrinsic, `max(1-k, 0)`. For every positive query before the first
model expiry (outside that intrinsic tolerance), the implementation uses the
**first fitted density immediately**; it does not interpolate from the
time-zero Dirac mass. A query within `ONE_MINUTE` of a model expiry uses that
node. Between model expiries, the two density-implied call prices are linearly
interpolated in time. After the last expiry, the last density is used. The
default `ArbFreePricer` mode chooses the surface support when a surface fit is
present and the expiry support otherwise. This pre-first-expiry jump is frozen
by CD-014.

For smooth pricing at query time `t`, let `tc = clip(t, T[0], T[-1])`. The
Black-mixture kernel uses annual volatility
`sqrt(smoothness * interp(tc, T, V) / tc)` and the actual query horizon
`sqrt(t)`; the resulting payoff is integrated against the selected density.
When smoothness is absent, or when the stored expiry-linear flag selects the
linear per-expiry kernel, pricing integrates the ordinary call payoff. The
input `V` curve is not a fitted ATM implied-volatility curve.

The initial parity implementation preserves legacy float32 cast boundaries even
when solvers operate in float64. A future dtype change is versioned and tested
as a semantic change.

### Phase 5 persisted stages

`paired-day` cleans and pairs call/put books, attaches the reference forward and
discount factor, and computes the settlement clock. `enriched-day` consumes
that one frame, projects an implied forward, forms the blended synthetic call
book, converts to pure units, inverts volatility, and appends Greeks and NMS.
Their namespaces are respectively
`widb/paired-day/ticker-<T>/year-<Y>/date-<D>` and
`widb/enriched-day/ticker-<T>/year-<Y>/date-<D>`.

Both stages persist exactly one tabular payload, `data.parquet`. Portable
`config.json`, explanatory `README.md`, checksummed `manifest.json`, and the
completion marker are metadata, not separate quote/expiry/surface tables.
Known-bad dates produce an exact typed empty frame. Other unexpected empty or
invalid populations fail rather than becoming silent exclusions.

### Phase 6 persisted stages

`no-arb-day` consumes `enriched-day` and applies the initial linear surface
fit. It retains all enriched rows, marks expiries dropped by the fit, preserves
the strike mask, appends reference-compatible Float32 fit/Greek fields, and
stores the complete pre-filter linear no-arbitrage audit state. Sparse audit
densities can contain a positive call-time increment at a zero-mass node, so
this state is not labelled or serialized as a finite DLV model under CD-021.

`filtered-day` consumes `no-arb-day` and applies the recovered production
sequence: inclusive 70-business-day horizon, spot normalization, inclusive
ATM-NMS `[-100,100]`, volume and open-interest thresholds, optional AM rule,
minimum 10 rows per expiry, then the stable 100,000-row cap. It retains rejected
rows with cumulative eligibility booleans and the first rejection reason.
`min_vega_sqrt_var=0.001` remains a versioned no-op under CD-012.

`fitted-day` fits the selected pure call books with smoothness 0.25, retaining
the full smooth SANOS state, quote-aligned mids/errors, distinct pre-fit,
smooth-fit and intersection masks, and separately active real call/put books.
It preserves the three legacy fitted-price meanings as distinct Float64
columns: `smooth_expiry_fit_price` is the retained per-expiry fit (`fits1`),
`smooth_surface_fit_price` is the common-surface fit (`fitsX`), and
`smooth_call_mid` is the quote-aligned result returned by
`compute_market_fit().fits`. The last two are numerically close but are not
interchangeable.
Validation replays all three producer paths separately under their original
Float32 support/strike arithmetic and operation ordering, including the direct
intrinsic-value rule below one model minute. Each persisted producer field is
checked at `1e-12`. The canonical serialized-state `SanosPricer` remains a
Float64 off-grid consumer: its bridge to `smooth_call_mid` retains a `5e-8`
acceptance gate, while the accumulated bridge to solver-time `fitsX` is a
separately named diagnostic under CD-025.
It then fits the CD-023 canonical finite-DLV projection to the completed smooth
surface at the same retained expiries. The projection has 200 inner nodes over
inclusive source-smooth NMS `[-3,2]`, one exact ATM node, support boundaries at
0.5 times the lower and 1.5 times the upper inner strike, and a uniform price
objective. One deterministic anchor row per retained expiry stores its state.
Its DLV rows are geometry-and-bound-scaled and use the configured `1e-10`
strict-solver feasibility target plus a `9e-5` fitted-node call-error cap. At
source target nodes with Black vega divided by `sqrt(T)` at least `0.05`, the
fit also obeys the exact Black-price band for an annual-IV move of at most
`0.005`, inclusive, with the lower volatility floored at zero and the band
inset by the solver tolerance. These hard linear bounds leave the uniform L1
objective unchanged. A `1e-8` support-mass floor and `1e-5` solver-only inset
from each DLV bound keep the persisted inverse well-conditioned without
post-fit clipping. Canonical DLV serialization and stabilization use wider
NumPy precision but persist Float64; validation separately proves the ordinary
Float64 consumer decoder recovers density and off-grid calls within `1e-12`.
These numerical settings are part of projection identity and do not alter the
solver policy of the preceding fits.
Real books are represented in `C/(DF*F)` units. Solver events record stage,
solver, status, fallback and objective; elapsed time remains in the run journal
rather than deterministic frame content. A failed solver or projection gate
publishes no terminal artifact.

All three artifacts keep one Polars frame per day. Model support and density
vectors are typed list columns repeated within an expiry; Parquet compression
avoids a separate state table while every rejected input row remains auditable.
The exact selected legacy row view is the subset where
`options_filter_eligible` is true, stably ordered by expiry and normalized
strike.

## Internal dataset

The internal WIDB dataset is curated, validated and immutable. It normally uses
partitioned Parquet (ticker/year/date) and includes the nearly complete filtered
option history, all eligibility masks, expiry geometry, no-arbitrage results,
smooth SANOS state, canonical finite-DLV projection, fitted values and
diagnostics. A
published identity contains:

- `README.md` with coverage, fields, units, sources, exclusions and exact CLI
  reproduction/validation commands;
- `config.json` with the complete canonical configuration;
- `manifest.json` with distribution status, schema/pipeline versions, Git
  commit, upstream identities, date/row/population counts and checksums; and
- Parquet partitions plus a completion marker written last.

The 0.2D AM observations belong in the complete internal dataset when otherwise
eligible. Their exclusion is an explicit daily-front experiment selection, not
an ingestion default.

## Public SANOS/DLV projection

Public release is a separate deliberately lossy dataset identity. The intended
physical representation has **one Polars row per trading day** and typed
list/struct columns. The exporter starts from an explicit allowlist and fails on
every unexpected input column.

The approved version-2 logical allowlist is below. CD-004 and CD-009 freeze the
exact return formula, retained-date policy, normalized units, and unit bridge:

| Level | Fields and units |
|---|---|
| day | `trading_date`, trading-session `is_half_day`, `forward_adjusted_spot_return`; the first return is null |
| expiries | expiry date, business DTE, calendar DTE, expiry `is_am`, model time, and dimensionless interpreting geometry including `discounted_forward_over_spot = DF*(F/S)` |
| books | `expiry_index`, call/put side, `normalized_strike = K/F`, `half_spread_spot = (ask-bid)/(2S)` for every book surviving the exact daily-front filter below |
| smooth SANOS | complete smooth parameter/density state, supports, expiry/time increments and separately named smoothing input variances |
| discrete DLV | complete CD-023 finite projection density and squared-DLV state, supports, boundaries and expiry/time increments |

The version-2 public book population reproduces the daily-front
reference. For
each call and put side independently, a book is initially active only when the
pre-smooth no-arbitrage mask is true, bid/ask are finite, bid is non-negative
and ask is strictly above bid. It is retained only when the option is European,
the corresponding smooth fitted call/put value is finite, the expiry is not an
explicitly excluded 0.2-business-day node, the expiry lies at or before the
70-business-day model horizon, and `K/F` is inside the **inclusive** AF-ATM NMS
interval `[-3, 2]`. Calls and puts remain separate observations. A different
filter is a different explicitly approved dataset configuration and identity.

Model prices reconstructed in memory from the retained state are expressed in
discounted-forward units `C/(DF*F)`; no redundant per-book model-price column is
stored. The proposed dimensionless conversion is:

```text
C/S = [DF*(F/S)] * [C/(DF*F)]
h/(DF*F) = (h/S) / [DF*(F/S)]
```

The exact public return formula and unit bridge are approved technical
contracts in `docs/compatibility-decisions.md`. This approval permits a
`public_candidate`; it does not establish redistribution rights.

The exporter forbids raw bids/asks/mids, absolute spot/strike/price levels,
volume, open interest, option symbols/IDs, security IDs, vendor implied
volatilities/Greeks, redundant per-book fitted model price/IV arrays and all
other unapproved source fields. It also rejects a schema containing a forbidden
field even if the value is null. Validation may derive model prices and ATM
implied volatilities from the published state in memory; those derived values
are not stored columns.

A public manifest is initially `public_candidate`. Transition to immutable
`public` additionally requires release license, rights basis, approver,
approval date and a durable governing reference. Numeric transformation alone
does not establish redistribution rights. The release is reprojected from the
same internal Dataset, must be exactly logically equal to the approved
candidate, and has its own identity, manifest, README, approval record, and
license bytes. It never aliases or mutates the candidate.

Candidate identity also fixes the coverage start/end and exact permitted
typed-empty date/reason records. Shallow validation binds the complete logical
frame and return vector to stored evidence and runs public-only capabilities;
deep validation additionally reprojects from the internal source to prove the
private-input CD-004 and CD-009 calculations.

Capability flags may advertise normalized SANOS/DLV dynamics and normalized
spread experiments. Smooth-to-DLV equivalence is claimed only at retained
expiries on the inclusive `[-3,2]` NMS projection domain, with maximum absolute
normalized-call error `1e-4`; arbitrary-time equivalence is not claimed. They
explicitly do not advertise vendor-book
reconstruction, quote-containment replay, liquidity models, absolute-dollar
tick effects or contract-level trading replay. Public validation runs all
advertised capabilities with WRDS and the internal dataset unavailable.

## Experiment

An experiment is a disposable model-specific materialization from an immutable
dataset. HDF5 may contain padding, masks, fixed arrays, splits and model-specific
selections. Beside every HDF5 file are the source dataset identity, full
experiment configuration, schema/producer version and checksums. Deleting an
experiment never deletes its source dataset.

## Output-first validation

A dataset or experiment request first validates the requested artifact's own
manifest, canonical configuration identity, schema, completion and checksums.
It does not require lineage files under shallow validation. `version="*"` may
ignore only producer-code-version comparison for an already complete matching
artifact; it still validates every other property, warns visibly, records the
override in the run result and fails rather than building on a miss.
