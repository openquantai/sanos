# Phase 5 quantitative design audit

**Status: READY FOR COMPATIBILITY IMPLEMENTATION ON THE UNAMBIGUOUS PATH; THE
EDGE CASES LISTED UNDER “blocked choices” MUST NOT BE SILENTLY RESOLVED**

Reviewer: independent `quantagent`
Date: 2026-10-12
Pinned reference revision: `de15be96d2f283b6c31e60d3ea47325a0c093ff0`

This is a pre-implementation audit, not approval of Phase 5 code. I inspected
the checked-in pinned reference only. I did not access WRDS or any external
network, modify `quant_data`, or write an implementation file.

## 1. Authoritative scope and separation from later phases

Phase 5 owns the deterministic transformation from the five Phase 4 Raw
OptionMetrics sources into a paired, enriched daily WIDB Polars frame. It owns
calendar/session metadata, raw-unit compatibility conversion, the early market
quality filters, call/put pairing, discount factors, vendor-forward attachment,
business-time geometry, the implied forward, reconciled synthetic call book,
pure normalization, implied volatilities, Greeks, and NMS coordinates.

It does **not** own expiry/surface no-arbitrage repair, the later
`OptionsFilter`, smooth SANOS fitting, the public projection, or the public
forward-adjusted return. Those remain Phases 6–8. In particular, the exact
reference order is:

```text
Phase 4 Raw partitions
  -> collect_raw_day-compatible conversion/filter/pair/clock
  -> enrich_day-compatible forward/pure-book/IV/Greek/NMS computation
  -> mk_arb_free_day                                      [Phase 6]
  -> OptionsFilter: horizon -> spot normalization -> NMS
                   -> volume -> OI -> AM -> min/expiry -> cap [Phase 6]
  -> smooth SANOS                                         [Phase 6]
```

The legacy `normalize_day` function should be ported and parity-tested in Phase
5, but its production invocation must remain at the same point inside the
Phase 6 options-filter sequence. Moving spot normalization before no-arbitrage
would be an unapproved pipeline-order change even where the pure solver inputs
happen to be scale invariant.

The reference files establishing this contract are:

- `reference/quant_data/quantdata/widb/day.py`;
- `reference/quant_data/quantdata/widb/client.py`;
- `reference/quant_data/quantdata/biz_calendar.py`;
- `reference/quant_data/quantdata/widb/bad.py`;
- `reference/quant_data/quantdata/widb/arbfree.py` and
  `widb/options_filter.py` for the downstream boundary; and
- `reference/quant_data/quantdata/dysanos/market.py` and
  `dysanos/flex/market.py` for the actual producer path.

## 2. Exact daily stage sequence

Every step below is deterministic, versioned with `cdxcore.version.version`,
and consumes/returns one Polars frame for one trading date. Configuration,
quality registries, calendar functions, schemas, and every content-affecting
helper belong in the resolved producer-version closure.

### 2.1 Session and source validation

1. Normalize ticker to the Phase 4 identity and require the requested trading
   date to be an NYSE session. Compute `is_half_day` from the same NYSE schedule;
   do not infer it from option settlement flags.
2. Apply the versioned bad-data registry before analytical work. The pinned
   registry has `LAST_LIKELY_VALID_DATE=2025-09-01`, includes `2025-01-08` in
   `DAYS_ALL`, and has the ticker-specific and option-symbol exclusions in
   `widb/bad.py`. A known-bad date must produce an explicit typed empty/status
   artifact when the dataset policy is “continue”; it must not look like an
   ordinary zero-row vendor response.
3. Resolve and validate the exact Phase 4 upstream artifacts: security identity,
   current and previous-session spot, current zero curve, current options, and
   current forward curve. `previous_session` is the immediate NYSE session,
   including an early close, not the previous retained dataset row.
4. Require one usable positive current and previous close and non-empty current
   option, zero and forward tables. Phase 4 deliberately preserves zero-row
   source evidence; Phase 5 classifies such a day as unusable with a table-
   specific reason instead of repeatedly downloading it.

The freshly captured previous-session forward and zero-curve partitions are
not used by the legacy internal daily path. They remain conservative inputs for
CD-004 and must not be used to relabel or replace the legacy simple return.

### 2.2 Raw compatibility conversion

Apply the reference SQL-level conversions explicitly, because Phase 4 correctly
stored minimally transformed vendor values:

- `strike = strike_price / 1000.0` in index/currency units;
- add synthetic `exercise_style = "E"`, hence `is_american = false` for this
  OptionMetrics compatibility stream;
- forward aliases are `expiration -> exdate`, `amsettlement -> am_settlement`,
  and `forwardprice -> forward_price`;
- preserve option IDs, symbols, quotes, volume, open interest and settlement
  fields from Raw; do not substitute vendor IVs or Greeks for the locally
  recomputed values.

The current strict Raw schema check supersedes the reversed legacy assertion
for actual SANOS artifacts. `docs/compatibility-decisions.md` still labels
CD-002 open, so that governance status should be reconciled, but Phase 5 must
not reintroduce the permissive/misdirected assertion.

### 2.3 Contract-level filter order

Order is part of parity. Each row receives a named stage mask and first-rejection
reason before the parity view removes it:

1. known bad symbol: ticker-wide union date-specific registry;
2. same-day expiry: `exdate == trading_date`;
3. configured irregular expiry: with the pinned default, `exdate` in the
   exact versioned set `{2025-01-09}`;
4. missing bid or ask;
5. crossed/locked quote: `best_bid >= best_offer`;
6. low bid: `best_bid <= min_bid`, with compatibility default `0.011`;
7. bad special settlement: `ss_flag != "0"` under the reference representation.

The retained predicate is therefore finite/non-null bid and ask, strict
`ask > bid`, strict `bid > 0.011`, acceptable settlement, and none of the
earlier date/symbol exclusions. A row matching multiple conditions is counted
at the first stage where the reference would remove it. The implementation must
not fuse predicates if doing so changes first-rejection diagnostics.

The strict string comparison for `ss_flag` requires a source-dtype probe in the
live pilot. Fixtures must use the reference string representation. Supporting
numeric zero as an equivalent adapter input is reasonable, but is a declared
schema-normalization rule, not an accidental Polars cast.

### 2.4 Discount and vendor-forward attachment

For option calendar maturity

```text
calendar_dte = (expiry_date - trading_date).days
```

linearly interpolate the continuously compounded vendor **percent** rate in
calendar days, with NumPy-compatible endpoint-constant extrapolation:

```text
r_pct(d) = interp(d, curve_days, curve_rates_pct)
DF(d)    = exp(-r_pct(d) * d / 36500)
```

This is linear interpolation of the quoted rate followed by exponentiation. It
is not interpolation of discount factors, zero rates in decimal units, or a
business-day discount clock. The curve is queried ordered by tenor. Require
finite `DF > 0`; retain Float64 working precision until the specified Float32
daily cast.

For vendor forwards:

1. treat missing `am_settlement` as zero for compatibility;
2. convert expiry to date and AM flag to numeric zero/one;
3. remove current-date forwards, matching the same-day option exclusion;
4. if `forward_price == spot / 1_000_000` exactly, replace it with `spot` and
   emit a structured correction diagnostic;
5. reject the day if the remaining table has any minimum below `spot/2` or any
   maximum above `2*spot`;
6. require at most one row per `(expiry_date, expiry_is_am)`; and
7. map by that exact key. The vendor value is retained only as
   `vendor_forward`; it does not drive the analytical fit.

### 2.5 Call/put semantics, pairing and raw-day geometry

Compute side liquidity before pairing:

```text
side_log_volume       = log10(1 + fill_null(volume, 0))
side_log_open_interest = log10(1 + fill_null(open_interest, 0))
```

Calls and puts are sorted by
`(expiry_date, expiry_is_am, strike, is_american)` and inner-joined on exactly
those four fields. Thus unpaired books do not enter the legacy paired parity
view. Call-side discount factor and vendor forward are retained; put-side
source analytics are not used. Duplicate keys have no approved deterministic
tie-break and are a blocked choice below.

The legacy expiry identifier is:

```text
expiry_id = year*100000 + month*1000 + day*10 + (0 if AM else 1)
```

Business DTE counts NYSE sessions in `(trading_date, expiry_date]`, with early
closes counted as full sessions. Define the new explicit clock as:

```text
base_business_dte = NYSE sessions in (date, expiry]
am_subtract       = 1 - overnight_variance_fraction
business_dte      = max(base_business_dte - (am_subtract if is_am else 0), 0)
time_to_expiry_years = business_dte / 252
```

`overnight_variance_fraction` must be finite, must reject Boolean inputs, must
lie in the closed interval `[0,1]`, defaults to exactly `0.2`, and participates
in canonical configuration and artifact identity. `calendar_dte` is the
unadjusted integer date difference and is never substituted for business DTE.
Discount factors also remain on calendar time and do not change with this
parameter.

The reference permits non-positive base DTE only for historical Saturday AM
expiries, asserts that condition, and clips the adjusted value at zero. Preserve
the formula and surface the case explicitly; do not silently drop it in Phase
5.

### 2.6 Expiry enrichment

Process `expiry_id` in strictly increasing order. All formulas below use
Float64 working arrays and are cast to Float32 at the daily boundary.

Define original paired-book activity from volume only:

```text
call_traded = call_log_volume > 0       # equivalent to call volume > 0
put_traded  = put_log_volume > 0
both_traded = call_traded & put_traded
either_traded = call_traded | put_traded
```

Before analytical price checks, retain an expiry only with at least four
`both_traded` strikes and at least ten `either_traded` strikes under the default
`ExpiryFilter`.

For every paired strike, construct the put-call-parity interval for discounted
forward `DFF = DF*F`:

```text
DFF_ask_i = call_ask_i - put_bid_i + DF*K_i
DFF_bid_i = call_bid_i - put_ask_i + DF*K_i
DFF_mid_i = (DFF_ask_i + DFF_bid_i)/2
w_i       = 1/max(DFF_ask_i - DFF_bid_i, 0.001) if both_traded_i else 0
DFF        = sum(DFF_mid_i*w_i) / sum(w_i)
F          = DFF/DF
```

This is inverse interval-width weighting, not inverse variance weighting. Reject
the expiry if the projected `DFF < 1e-6`. `vendor_forward` is not blended into
or used to constrain this projection.

The put-derived synthetic call interval is:

```text
syn_call_ask = put_ask + DFF - DF*K
syn_call_bid = put_bid + DFF - DF*K
```

It exactly satisfies side-by-side put-call parity before rounding. Reject a
strike when either synthetic endpoint is non-positive or its spread is below
`min_pure_spread*DFF`, with default `min_pure_spread=1e-6`.

The legacy reconciled minimum-spread call interval is then:

```text
center = [min(syn_call_ask, market_call_ask)
          + max(syn_call_bid, market_call_bid)]/2
width  = min(market_call_ask-market_call_bid,
             syn_call_ask-syn_call_bid)
new_call_ask = center + width/2
new_call_bid = center - width/2
k = K/F
c_ask = new_call_ask/DFF
c_bid = new_call_bid/DFF
```

Reject a strike if `c_ask-c_bid < 1e-6`. Then repeat the four-both/ten-either
expiry tests on the surviving strikes. The final reconciled call interval need
not satisfy parity against a particular vendor put endpoint; parity tests must
target the `syn_call_*` formulas, not assert the wrong identity on the blended
book.

Invert `c_ask` and `c_bid` independently with the pinned `cdxcore.bs` routine,
`sqrtT=sqrt(time_to_expiry_years)`, price tolerance
`min(0.1*(c_ask-c_bid), 1e-6)`, numerical epsilon `1e-8`, and annual volatility
bounds `[0.01,5.0]`. Define ATM annual volatility by one-dimensional
interpolation at `k=1` over the midpoint of bid/ask volatilities. Reprice and
compute pure delta, gamma, vega, positive optionality theta
`dc/d(time-to-expiry)`, and strike derivative `dc/dk`, using epsilon `1e-8`.
If repricing differs from the target by more than `1.1*price_tolerance`, the
reference requires the implied volatility to be on a configured boundary and
removes that strike.

The final analytical call prices are the Black-repriced values, not necessarily
the exact pre-inversion endpoints:

```text
synthetic_call_cash = pure_call_price * DFF
```

Combined liquidity is:

```text
log_volume = log10(1 + call_volume + put_volume)
log_open_interest = log10(1 + call_OI + put_OI)
```

The two legacy NMS5 coordinates are:

```text
EPS5 = 5/252
SCALE = log(1.1)
nms = log(k)*sqrt((EPS5+1)/(SCALE^2*EPS5 + SCALE^2*T))
nms_atm = log(k)*sqrt((EPS5+1)/(SCALE^2*EPS5 + atm_vol^2*T))
```

where `T=time_to_expiry_years`. Invalid rows are removed only from the legacy
parity view; the new stage should retain named validity/rejection diagnostics
until its declared materialization boundary.

## 3. Canonical Phase 5 schema and units

Canonical names should not overload a field with the legacy pre/post-
normalization meanings. A parity adapter can expose the legacy names shown in
parentheses. Numeric cast points remain reference Float32 under CD-011.

### 3.1 Paired raw-day fields

| Canonical field | Polars dtype | Unit/meaning |
|---|---|---|
| `ticker` | `String` | requested OptionMetrics ticker |
| `trading_date` (`date`) | `Date` | observation session |
| `source_secid` | `Int64` | resolved Raw security identity |
| `is_half_day` | `Boolean` | trading-session early-close flag |
| `spot` | `Float32` | current underlying close, index/currency units |
| `previous_spot` (`spot_prev`) | `Float32` | immediate previous NYSE-session close, same units |
| `call_symbol`, `put_symbol` | `String` | source option symbols |
| `call_option_id`, `put_option_id` | `Int32` | legacy-cast source option IDs |
| `expiry_id` | `Int32` | date/AM-PM encoding above |
| `is_american` | `Boolean` | false in this synthetic-European compatibility stream |
| `expiry_is_am` (`is_am`) | `Boolean` | option settlement convention, not session half-day |
| `strike` | `Float32` | cash/index strike after raw `/1000` scaling |
| `vendor_forward` (`fwd_ref`) | `Float32` | OptionMetrics forward level, diagnostic only |
| `discount_factor` (`df`) | `Float32` | positive dimensionless ACT/365 DF |
| `expiry_date` (`expiry`) | `Date` | contract expiration date |
| `calendar_dte` | `Int32` | unadjusted calendar-date difference |
| `business_dte` (`bdte`) | `Float32` | adjusted NYSE-session count in days |
| `time_to_expiry_years` (`tte_y`) | `Float32` | `business_dte/252` |
| `market_call_bid/ask`, `market_put_bid/ask` | `Float32` | cash/index units per underlying unit |
| four side log-volume/OI fields | `Float32` | `log10(1+x)` for each source side |

Artifact/configuration identity fields (`stage_version`, `configuration_id`,
and upstream Raw artifact identities) are strings or typed lists/structs and
must be recorded without credentials. An Int32 range check for option IDs and
the expiry ID must precede strict casting; silent integer wrap is not parity.

### 3.2 Enriched additions

All are `Float32` at the persisted daily boundary:

| Canonical field | Unit/meaning |
|---|---|
| `implied_forward` (`fwd`) | projected cash/index forward `DFF/DF` |
| `normalized_moneyness` (`nms`) | fixed-scale NMS5 coordinate |
| `atm_normalized_moneyness` (`nms_atm`) | ATM-volatility NMS5 coordinate |
| `synthetic_call_bid/ask` (`call_bid/ask`) | reconciled Black-repriced call in cash/index units |
| `bid_implied_vol`, `ask_implied_vol` | annual Black volatility |
| `log_volume`, `log_open_interest` | combined `log10(1+call+put)` |
| `normalized_strike` (`pure_strike`) | `K/F`, dimensionless |
| `pure_bid_price`, `pure_ask_price` | `C/(DF*F)`, dimensionless |
| `pure_*_delta` | normalized Black delta |
| `pure_*_gamma` | normalized second derivative |
| `pure_*_vega` | derivative per unit annual volatility |
| `pure_*_theta` | positive optionality derivative per year-to-expiry |
| `pure_*_dk` | derivative of pure price with respect to normalized strike |

The reference schema contains 24 enriched Float32 additions. Bid/ask ordering
is part of the field name and must not be alphabetically regenerated.

### 3.3 Legacy spot-normalization view

For exact downstream parity, derive without overloading canonical fields:

```text
simple_spot_return = spot/previous_spot - 1
previous_spot_over_spot = previous_spot/spot
implied_forward_over_spot = implied_forward/spot
strike_over_spot = strike/spot
synthetic_call_{bid,ask}_over_spot = synthetic_call_{bid,ask}/spot
market_{call,put}_{bid,ask}_over_spot = market_{call,put}_{bid,ask}/spot
```

The legacy adapter maps these to `spot_ret`, overwritten `spot_prev`, `fwd`,
`strike`, `call_*`, and `mkt_*`, and exposes absolute current spot as
`spot_ref`. It leaves `vendor_forward`, discount factors, pure fields, vols,
Greeks, NMS and liquidity unchanged. Require positive current and previous
spots. The internal daily frame may retain absolute fields; the public exporter
must not.

`simple_spot_return` is **not** forward adjusted. CD-004 remains open. No field
may be called `forward_adjusted_spot_return` until its formula, retained-date
semantics, and interpolation rule receive explicit approval.

## 4. Sorting, joins and tie-breaking

- Expiry processing order is ascending `expiry_id`; AM precedes PM on the same
  date because of the identifier suffix.
- Within the ordinary unique-key path, books are ordered by expiry date, AM/PM,
  strike and American flag. Phase 6 subsequently stable-sorts each expiry by
  normalized strike.
- The raw OptionMetrics query has no `ORDER BY`. The reference separately sorts
  calls and puts on the four join keys but supplies no secondary option-ID or
  symbol tie-break. Pandas then performs an inner merge, producing a Cartesian
  product for duplicate side keys. A fresh Polars join must not arbitrarily
  choose a row or assume its output order matches pandas.
- The Phase 5 implementation may proceed on fixtures and days where each side
  is unique per join key. If duplicate keys appear in fresh data, stop that day,
  retain a diagnostic sample and obtain the pairing-policy decision below.
- Vendor-forward duplicate `(expiry_date, expiry_is_am)` keys are already a
  hard reference error. Missing mappings should be detected explicitly rather
  than allowed to become a late NaN.

## 5. Overnight-fraction propagation contract

At the compatibility default `0.2`, every non-clipped AM DTE and TTE must match
the reference exactly after its Float32 cast; PM nodes must be bitwise unchanged.
For an AM node with base business DTE `b`:

```text
bdte(f) = max(b - 1 + f, 0)
```

Therefore `bdte(f2)-bdte(f1)=f2-f1` only when neither value is clipped. The
Phase 5 plan's “shifts exactly” acceptance sentence must be interpreted with
that qualification. Saturday-AM legacy nodes can remain at zero across a range
of fractions.

The fraction has no direct effect on strike, raw quotes, calendar DTE, discount
factor, vendor forward, or the parity-implied forward. It changes AM:

- business DTE and TTE;
- Black implied vols/Greeks and IV-boundary eligibility;
- NMS and NMS-ATM; and consequently, in later phases,
- horizon/NMS eligibility, expiry ordering constraints, total/forward variance
  interpolation, calendar constraints, DLV time increments and experiment
  horizons.

Endpoints are mathematically defined but not generally surface-buildable. At
`f=0`, a next-session AM option has `T=0`; at `f=1`, same-date AM and PM nodes
have equal TTE despite distinct expiry IDs. The pinned Phase 6 no-arbitrage path
requires strictly increasing positive expiry times. Phase 5 tests must cover
and report the endpoint geometry, but must not invent a drop/merge/jitter policy.
That downstream policy is a blocked choice.

Sensitivity tests at `0.1`, `0.2`, and `0.5` must compare the entire resulting
AM vectors, not just displayed DTE. PM rows must be identical. The resulting
artifact IDs must all differ.

## 6. Required structured diagnostics

At minimum, each daily run and manifest must expose:

- source row counts and explicit empty-source table status;
- known-bad day/ticker status and bad-registry version;
- count after every ordered raw filter plus first-rejection counts;
- forward correction count, suspicious range, duplicate keys and unmatched
  option-forward keys;
- call/put input counts, unpaired counts, duplicate pair-key samples and paired
  count;
- input expiry count and per-expiry pre-counts (`both`, `either`);
- projected DFF/F and vendor-forward difference (diagnostic only);
- per-expiry drops for pre-count, invalid forward, parity endpoints, pure
  spread, post-count and IV boundary;
- final row/expiry counts and min/max DTE, DF, forward, normalized strike,
  spread and volatility; and
- the exact overnight fraction and half-day flag.

Do not copy the reference's string/list mutation bug in empty-day logging. A
new typed diagnostic schema can correct labels and iteration without changing
market numbers, but that diagnostic-schema difference must be declared and
versioned. Empty days need manifest/status diagnostics because a zero-row
Parquet frame cannot repeat row-level metadata.

## 7. Parity and adversarial fixture plan

### 7.1 Pinned executable parity fixtures

Create small checked-in Raw-style fixtures and generate the expected legacy
views with an isolated subprocess importing only the pinned snapshot. Freeze
the input JSON/Parquet, output schema, values and hashes; do not read a legacy
Raw cache. At least these cases are required:

1. one ordinary expiry with 10+ unique paired strikes, four+ both-traded, a
   non-flat zero curve, and distinct market/put-derived call intervals;
2. two expiries, including AM and PM settlement geometry, proving ordering and
   per-expiry constancy;
3. exact threshold cases for bid `0.011`, locked/crossed quotes, null quotes,
   special settlement, volume counts, pure spread `1e-6`, forward `1e-6`, and
   IV bounds `0.01/5.0`;
4. the exact vendor-forward `spot/1_000_000` correction and a normal reference
   forward;
5. zero-curve interpolation, lower/upper extrapolation and percent scaling;
6. a Black inversion boundary failure and an entirely rejected expiry; and
7. legacy spot normalization with exact Float32 comparisons.

Comparison is exact for schema, ordering, strings, booleans, integers and
Float32 cast results where the same pinned numerical kernel is used. Formula
unit tests run in Float64 at tight tolerances before the compatibility cast.

### 7.2 Clock/calendar fixtures

- `(start,end]` business count across a weekend and holiday;
- early close counted as one business day and `is_half_day=true` independently;
- next-session AM/PM values at fractions `0`, `0.1`, `0.2`, `0.5`, and `1`;
- same-date AM/PM TTE collision at `f=1` and zero-time AM at `f=0`;
- historical Saturday-AM clipping behavior;
- calendar DTE distinct from business DTE and discount interpolation; and
- artifact identity changes with fraction while PM numerical values do not.

For a simple independent oracle, the pinned code produces business counts
`[1,2,2,3]` from 2024-01-03 to 2024-01-04/05/06/08. With curve tenors
`[1,30,90]`, percent rates `[5,4,3]`, and queried days
`[1,15,30,60,120]`, it produces DFs approximately
`[0.99986302,0.99814532,0.99671773,0.99426309,0.99018547]`.

### 7.3 Join and failure fixtures

- missing call, missing put, duplicate call key, duplicate put key, and duplicate
  vendor-forward key;
- multiple or conflicting spot rows;
- missing forward mapping and null/NaN forward;
- unsorted and duplicate zero tenors;
- option ID outside Int32 range;
- `ss_flag` represented as string and numeric zero;
- exact-schema empty options/spot/forward/zero artifacts; and
- known-bad date under both fail-fast and continue policies.

These tests should assert explicit failures/status, never whatever row ordering
a Polars join happens to return.

### 7.4 Reference-population gates

The raw date interval contains 918 NYSE sessions. The authoritative smooth
collection has 892 physical files, one typed empty marker (`2025-01-08`), and
891 populated dates. Twenty-six NYSE sessions have no physical smooth-market
file. All seven early closes in the interval are among those 26:

```text
2022-11-25
2023-07-03, 2023-11-24
2024-07-03, 2024-11-29, 2024-12-24
2025-07-03
```

Consequently, “current source code can include half days” is not evidence that
the 891-day artifact did include them. The compatibility dataset profile must
explicitly exclude half days unless the user approves a new population. A
general reusable profile may include them and must still set `is_half_day`.
The other 19 absent sessions must not be hard-coded as exclusions without
stage-by-stage failure evidence from the fresh build. Any fresh success on a
previously absent date, or failure on a reference-populated date, is a reported
population discrepancy rather than an automatic inclusion/exclusion.

Phase 5 should report its own daily populations. It cannot be required to equal
the final 891-day/smooth counts before Phase 6. The end-to-end gates remain 891
populated dates and 4,651,416 active books for the smooth cache, then 3,566,258
daily-front books after the distinct downstream FLEX selection.

## 8. Blocked choices and suspected reference defects

The ordinary unique-key, non-empty, default-0.2 path is sufficiently specified
for implementation. The following must be recorded before the affected branch
is implemented or exercised:

1. **CD-003 — empty enriched validation.** The reference returns a typed empty
   enriched frame before its schema check, making the check unreachable.
   Recommended treatment is to validate the raw empty schema first and return
   the same typed empty output. This changes only malformed-empty failure
   timing, but CD-003 is still marked open and needs approval.
2. **Half-day population default.** The 891-day oracle excludes all seven half
   days although the current `dysanos.download` CLI defaults to including them.
   Compatibility should mean exclude by default; an include-half-days profile
   is a different dataset identity. This must be explicit before declaring the
   target date universe.
3. **Duplicate option pair keys.** The reference Cartesian-merges duplicates
   and has no source-stable tie-break because SQL has no ordering. Do not pick
   an option ID, aggregate, or silently Cartesian-expand in Polars. Fail and
   surface evidence until the user approves a deterministic policy or fresh
   data proves the branch absent.
4. **Ambiguous spot rows.** `spot_price` returns `.iloc[0]` without uniqueness
   validation. The fresh implementation should reject conflicting rows, but
   that is a behavioral repair and must be approved before such a case is
   handled differently.
5. **Missing vendor-forward key.** The reference permits a missing mapping to
   become NaN and fail later. Early explicit rejection is diagnostically better
   but is a changed failure point. It must be declared; never impute the
   analytical implied forward or spot into an unmatched key.
6. **`ExpiryFilter.normalize` is an identity-only no-op.** The field appears in
   the configuration/string but is never read by `enrich_day`. Preserve the
   default `false` and do not advertise functionality. Activating it needs a
   new semantic decision/version.
7. **Endpoint overnight geometry.** `[0,1]` is the required validation domain,
   but Phase 6 cannot generally consume zero or duplicate times. Phase 5 may
   expose the geometry and diagnostics; dropping, merging or jittering nodes is
   blocked.
8. **Diagnostic defects.** The reference converts `dropped_expiries` from a
   list to a string and can iterate characters on an empty result; it also
   overwrites the input-expiry count before its final summary. Use structured
   diagnostics, but record that labels/count reporting are intentionally
   repaired while numerical eligibility remains unchanged.
9. **CD-004 remains blocked.** Internal `simple_spot_return` uses the immediate
   prior NYSE session. The proposed public return says consecutive retained
   observations and log-linear carry interpolation. Those can differ across
   excluded half days or gaps. Phase 5 must not choose between them or emit a
   public forward-adjusted field.
10. **CD-011 Float32 casts are frozen.** A Float64-through-daily refactor is not
    allowed during this port. Threshold fixtures must detect population changes
    caused by cast movement.

CD-010 remains intentionally deferred at the ticker/security boundary. CD-012
and the no-arbitrage/SANOS choices begin in Phase 6, but Phase 5 must provide
the masks and values needed to test their downstream population effects.

## 9. Offline evidence checks

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider tests/test_reference_inventory.py \
  tests/test_reference_baseline.py tests/test_reference_legacy_market.py
16 passed in 2.51s

# Pinned zero-curve/calendar/NMS numerical probe, WRDS_OFF=1
DF([1,15,30,60,120]) =
  [0.99986302, 0.99814532, 0.99671773, 0.99426309, 0.99018547]
business counts from 2024-01-03 to 2024-01-04/05/06/08 = [1,2,2,3]

# NYSE schedule versus frozen smooth-market catalog
sessions=918; physical_files=892; missing_sessions=26;
empty_markers=1; populated_days=891; early_closes=7;
all 7 early closes are among the 26 missing sessions

git --git-dir=.git-real --work-tree=. diff --check -- \
  reviews/phase-5/quant-design-audit.md
# exit 0
```

No command opened WRDS or modified the pinned reference.

## 10. Phase 5 implementation acceptance criteria

Phase 5 is ready for dual review only when:

1. every material transform/config/schema uses `@version` with complete
   dependencies and the overnight fraction changes identity;
2. every stage is one-day/one-Polars-frame, persists atomically as one Parquet
   artifact, and is independently runnable from the CLI with structured
   progress and no WRDS access on complete Raw inputs;
3. the canonical schema and the legacy parity adapter are exact and documented,
   including Float32 cast points and non-overloaded units;
4. ordered raw filters, pairing, discounting, forward projection, pure
   conversion, IV/Greeks and both NMS formulas pass pinned fixtures;
5. default `0.2` matches reference BDTE/TTE and enriched values; `0.1/0.2/0.5`
   propagation and `0/1` endpoint diagnostics are covered;
6. every population change has named row/expiry diagnostics and each empty or
   failed day has an explicit status and reason;
7. half-day session flags are distinct from expiry AM flags, the compatibility
   profile's date selection is explicit, and fresh-vs-reference date
   discrepancies are reported;
8. invalid source keys, NaNs, non-positive DF/DFF/F/spot, dtype overflows and
   ambiguous joins fail actionably rather than being imputed;
9. tests run with `WRDS_OFF=1`, use only synthetic/checked-in fixtures, and prove
   no code path opens WRDS; and
10. code-agent and quantagent reviews both pass without treating this audit as
    approval of CD-003, CD-004, duplicate-pair, ambiguous-spot, endpoint, or
    other blocked choices.

Signed: `quantagent` — independent quantitative/data reviewer
