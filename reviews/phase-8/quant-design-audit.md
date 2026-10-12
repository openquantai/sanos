# Phase 8 quantitative/data design audit

**Design verdict: READY AFTER EXPLICIT CD-004 AND CD-009 APPROVAL**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Scope: public SANOS/DLV research projection design only  
Network/WRDS access: none

This audit fixes the quantitative contract for a distinct, deliberately lossy,
one-row-per-day public projection of a completed Phase 7 internal Dataset. It
does not authorize implementation of the two still-open public formulas,
redistribution, WRDS access, or mutation of the internal Dataset. The legacy
891-date collection is comparison evidence only; it is never an input to the
new public artifact.

The public candidate is technically implementable once the user explicitly
approves CD-004 and CD-009. CD-007 and CD-014 are not new choices when their
recovered behavior is preserved exactly. Rights approval is a later, separate
gate from technical `public_candidate` validation.

## 1. Source boundary, dates and physical layout

The projection consumes exactly one complete, validated, immutable Dataset
whose distribution is `internal`. It never consumes Raw or Intermediate
artifacts directly. Building requires that internal Dataset and performs its
self-contained semantic/integrity validation; it does not require recursive
Raw/Intermediate lineage unless the operator separately requests a deep audit.
Loading and shallowly validating a completed public candidate do not require
even the internal Dataset to remain present.

The public artifact contains one `data.parquet` with one Polars row for every
**populated** internal trading date, sorted strictly by `trading_date`. A
policy-owned typed-empty internal date has no model state and is omitted from
the public rows, but its date and reason are preserved in manifest exclusions.
An unexpected empty day, a fitted-day failure, or a populated day with no
daily-front books is a publication failure, not an implicit omission. Half
days remain included and are identified by the day-level flag.

For the checked 2022-01-03 through 2025-08-29 configuration, the pre-production
expectation is 918 planned sessions, 917 populated public rows and one omitted
typed-empty date, 2025-01-08. These are calendar/policy expectations until the
fresh WRDS run; they are not claimed as observed production counts.

The public candidate has its own `README.md`, canonical `config.json`, exact
schema document, checksummed `manifest.json`, capability document and
completion marker. Its sole direct upstream is the internal Dataset pointer.
Artifact identity includes the internal Dataset identity, projection
configuration, exact schema, producer closure, CD-004/CD-009 decisions and
distribution status.

## 2. Exact version-1 logical Polars schema

No generic parameter blob or JSON state is permitted. The exact top-level
schema is:

| Field | Polars dtype | Cardinality and meaning |
|---|---|---|
| `trading_date` | `Date` | non-null unique trading date |
| `is_half_day` | `Boolean` | NYSE trading-session half-day flag; not expiry settlement |
| `forward_adjusted_spot_return` | `Float64` | nullable only on the first public row; pending CD-004 |
| `nms_epsilon_atm_annual_volatility` | `Float64` | positive DLV-state-derived ATM annual volatility at the canonical one-business-day model node used as `eps_vol` |
| `expiries` | `List(Struct)` | ordered union expiry geometry described below |
| `books` | `List(Struct)` | ordered daily-front call/put observations described below |
| `smooth_sanos` | `Struct` | complete surface-mode smooth SANOS state |
| `discrete_dlv` | `Struct` | complete surface-mode compatibility DLV state |

Every present float, including floats nested in lists/structs, must be finite.
No null is allowed except the first return. Empty `books`, model-state lists or
expiry lists are invalid on a published row.

### 2.1 `expiries`: one interpreting index shared by books and both states

`expiries` is sorted by model time and expiry identity and has contiguous
zero-based `expiry_index`. It is the ordered union needed by the complete DLV
state, smooth state and retained books; it is not pruned merely because an
expiry has no daily-front book. In the current fitted contract the DLV expiry
set should contain the smooth and book sets, and that containment is validated.

Each struct is exactly:

| Field | Dtype | Contract |
|---|---|---|
| `expiry_index` | `Int32` | contiguous index into this list |
| `expiry_date` | `Date` | option expiry calendar date |
| `business_dte` | `Float32` | Phase 5 business clock, including AM subtraction |
| `calendar_dte` | `Int32` | `expiry_date - trading_date` calendar days |
| `is_am` | `Boolean` | expiry AM/PM settlement flag |
| `time_to_expiry_years` | `Float32` | exactly `business_dte / 252` at the persisted cast boundary |
| `discounted_forward_over_spot` | `Float64` | `DF*(F/S)`, positive dimensionless bridge, pending CD-009 |
| `nms_atm_annual_volatility` | `Float64` | positive ATM annual volatility reconstructed from the unsmoothed AF/DLV state at this expiry; never vendor IV |

The day-level `is_half_day` and expiry-level `is_am` are semantically and
physically distinct. `overnight_variance_fraction=0.2` remains in canonical
configuration and determines AM `business_dte`, model time, state interpolation
and the 0.2-day book exclusion. It does not alter PM expiries.

### 2.2 `books`: deliberately lossy daily-front observations

Each `books` element is exactly:

| Field | Dtype | Contract |
|---|---|---|
| `expiry_index` | `Int32` | valid index into `expiries` |
| `is_call` | `Boolean` | call `true`, put `false`; sides remain separate observations |
| `normalized_strike` | `Float32` | `K/F`, finite and strictly positive |
| `half_spread_spot` | `Float64` | `(ask-bid)/(2*S)`, finite and strictly positive, pending CD-009 |

Order is expiry-major, calls before puts within an expiry, and stable increasing
`normalized_strike` within each side, matching the recovered daily-front
producer. No symbol, option ID, quote ID, security ID or absolute level is
stored. A strike with both active sides therefore contributes two book structs.

The canonical calculation promotes persisted Float32 source values to Float64
and uses the side-specific spot-normalized quote columns:

```text
half_spread_spot = 0.5 * (spot_normalized_ask - spot_normalized_bid)
```

Validation also checks the independent bridge

```text
half_spread_spot / discounted_forward_over_spot
    == 0.5 * (pure_ask - pure_bid)
```

within a versioned tolerance. Neither the absolute quote nor its midpoint is
published.

### 2.3 `smooth_sanos`: complete surface-mode reconstructing state

The struct contains:

```text
smoothness: Float64                         # exactly 0.25 for parity profile
model_expiries: List(Struct({
    expiry_index: Int32,
    time_increment_years: Float64,
    kernel_input_total_variance: Float64,
    normalized_support: List(Float64),
    density_mass: List(Float64),
}))
```

The model-expiry list is strictly increasing in the referenced expiry time.
The first increment is `T[0]`; later increments are `T[i]-T[i-1]` within this
state. `kernel_input_total_variance` is the persisted SANOS smoothing-kernel
input variance, not a fitted ATM implied volatility. Supports are strictly
increasing positive `K/F`; each mass vector is finite, non-negative within the
versioned tolerance, sums to one and has first moment one. The support/density
is the common surface state used by `smooth_sanos_pricer`, not the redundant
per-expiry fit state.

### 2.4 `discrete_dlv`: complete unsmoothed reconstructing state

The struct contains:

```text
model_expiries: List(Struct({
    expiry_index: Int32,
    time_increment_years: Float64,
    input_total_variance: Float64,
    normalized_support: List(Float64),
    density_mass: List(Float64),
    squared_dlv: List(Float64),
}))
```

The same expiry-time, support, density, mass and moment invariants apply.
`squared_dlv` uses annualized `1/year` units, has the same length as the full
support, has exact zero boundary values and finite non-negative interior values.
It is the approved corrected CD-022 encoding/inverse only; no defective legacy
inverse mode is advertised. Density and DLV are both retained so consumers can
validate the inverse instead of trusting either representation alone.

Per-expiry `xstrikes`/`xdensities`, x-to-surface mappings, fitted prices,
implied volatilities and Greeks are intentionally omitted. The advertised
public pricing mode is the complete common-surface mode above. Adding an
expiry-mode capability would require adding its full state under a new schema
and Dataset identity.

### 2.5 Pricing conventions belong to immutable schema/configuration

The following are not repeated as strings in every row, but are mandatory,
versioned interpreting geometry in `config.json`, the schema and README:

- price unit `C/(DF*F)` and strike unit `K/F`;
- 252 business days and 6.5 trading hours per day;
- `ONE_MINUTE = 1/(60*6.5*252)` years;
- intrinsic call `max(1-k,0)` for `t < ONE_MINUTE`;
- for every other positive `t < T[0]`, immediate use of the first fitted
  density under CD-014;
- exact node selection within `ONE_MINUTE`, linear interpolation of call prices
  between nodes, and final-density use after the last node;
- ordinary call payoff for discrete DLV and the recorded smooth-mixture kernel
  for SANOS; and
- puts reconstructed only through `p = c - 1 + k` in discounted-forward units.

CD-014 needs no new approval when this jump is preserved. Interpolation from a
time-zero Dirac mass would be a new model version and requires explicit
approval.

## 3. Exact daily-front book population (CD-007 parity profile)

Selection starts from each complete fitted internal day. For each call and put
side independently, retain a book if and only if all predicates hold:

1. the side-specific `call_active`/`put_active` field is true. In Phase 6 this
   already means `options_filter_eligible & mask_arb_free` and finite valid
   side quotes with `bid >= 0` and `ask > bid`;
2. `is_american` is false;
3. `smooth_mid_available` is true and the corresponding smooth call/put value
   is finite;
4. `business_dte` is not within `1e-4` of `0.2`;
5. `0 <= business_dte <= 70`, inclusive at 70; and
6. `normalized_strike` lies inside the inclusive AF-ATM NMS interval `[-3,2]`.

The NMS conversion uses the unsmoothed AF/DLV state, the reference
`NormalizedStrikes` epsilon `1/(252*8)`, the expiry-specific derived ATM annual
volatility, and `eps_vol` equal to the derived ATM annual volatility at the
canonical one-business-day model node. The public row retains the two volatility
inputs needed to verify the `K/F` endpoints independently. It does not store
redundant endpoint columns.

The internal 100,000-row cap, 70-day horizon and deliberately non-binding
`[-100,100]` upstream NMS filter remain part of the internal fitted identity.
Phase 8 applies no second cap after expanding call and put sides. Alternative
estimation grids, fixed-25-expiry selections and hedge screens belong to
Experiment configuration, not this public serialization.

The explicit 0.2-day rule removes books only. It must not remove the day, the
expiry geometry, or a model-state node needed to reconstruct SANOS/DLV. Other
AM expiries remain eligible. CD-007 is sufficiently recovered to implement
this parity profile; changing any predicate creates a different projection
configuration and requires review.

## 4. Proposed forward-adjusted return (CD-004: user approval required)

For public rows `t[i-1] < t[i]`, let `S_prev` and `S_now` be the internal spot
levels on those retained dates. Let

```text
h_i = NYSE business sessions in (t[i-1], t[i]]
q_j = implied_forward_j / S_prev
d_j = discount_factor_j
```

using the previous retained day's distinct expiry curve. Construct two positive
curves with anchors `(0,1)` and the previous-day `(business_dte_j,q_j)` and
`(business_dte_j,d_j)` nodes. Require increasing finite DTEs and consistent
duplicate nodes. At `h_i`, interpolate `log(q)` and `log(d)` linearly between
bracketing nodes; exact nodes use their exact values. Extrapolation beyond the
last node is forbidden. The interpolated discount factor is validated and
recorded in build evidence but is not a public field and does not enter the
return equation.

With `c_i = q(h_i) = F_(t[i-1])(t[i])/S_prev`, publish

```text
forward_adjusted_spot_return[i] = (S_now / S_prev) / c_i - 1
```

in Float64. The first public row is null even when an earlier Raw close exists.
After an omitted typed-empty date, the next row uses the immediately preceding
**retained public row** and a multi-session `h_i`; it does not substitute the
legacy simple return, chain through an absent observation, or silently fill the
gap. Weekends and holidays are handled by the NYSE session count. A half-day is
still one session; `is_half_day` does not fractionally alter the return horizon.

The internal `spot_return = S_t/S_previous-1` remains unchanged and is never
renamed or copied to the public field. The carry, discount interpolation and
absolute spots are build inputs/evidence only and are not published.

This entire formula, the retained-row gap rule, log interpolation, no-
extrapolation policy and first-row null require explicit user approval under
CD-004 before a public candidate may be published. A schema scaffold must not
populate the field with the legacy simple return or a placeholder claiming
forward adjustment.

## 5. Proposed spread/model-price bridge (CD-009: user approval required)

For every expiry, publish only

```text
discounted_forward_over_spot = DF * (F/S)
```

and for each retained book publish `half_spread_spot = half_spread/S`. The
consumer reconstructs normalized model values through

```text
C/S       = discounted_forward_over_spot * C/(DF*F)
h/(DF*F)  = half_spread_spot / discounted_forward_over_spot
```

The bridge is finite, strictly positive and constant for one day/expiry. Do not
publish `DF`, `F/S`, spot, forward, absolute strike, absolute prices or carry
separately. This supports normalized pricing/spread objectives without making
an absolute spot level a Dataset field.

The exact fields, Float64 arithmetic and equations require explicit user
approval under CD-009. Dimensional consistency and normalization do not by
themselves establish redistribution rights.

## 6. Capabilities and explicit non-capabilities

The manifest may set these capability flags true only after executable tests
using the public artifact alone:

- `normalized_smooth_sanos_pricing` — calls/puts at nodes and arbitrary valid
  normalized strikes/times under the frozen pricing convention;
- `normalized_discrete_dlv_pricing` — density/DLV inverse and normalized
  calls/puts, including irregular supports;
- `model_atm_implied_volatility` — derived in memory by pricing `K/F=1` and
  inverting with the versioned `cdxcore.bs` adapter;
- `daily_front_normalized_books` — exact side-separated CD-007 population; and
- `spot_one_spread_experiments` — spot-one spreads and conversion to
  discounted-forward units using CD-009.

The manifest must set these false explicitly:

- exact vendor bid/ask/mid or full-book reconstruction;
- quote-containment replay;
- vendor implied-volatility/Greek reconstruction;
- volume/open-interest or liquidity modelling;
- absolute spot/forward/strike/price levels and dollar tick effects;
- contract/symbol/option/security identity or trading replay; and
- expiry-fit-mode pricing not represented by the public state.

Forward-adjusted return dynamics should have their own capability flag and may
be true only after CD-004 approval and its boundary tests.

## 7. Technical leakage boundary

The exporter accepts only the exact Phase 7 internal schema and extracts
through a versioned field map. The output validator recursively enforces the
exact top-level and nested allowlists and rejects every unexpected field even
when it is null. It forbids raw bids, asks and mids; absolute spot, previous
spot, forward, strike and price; volume/open interest; symbols and option IDs;
source secid and Raw IDs; vendor or stored quote-aligned IV/Greeks; fitted
per-book prices/errors; diagnostics; and configuration/source strings in row
payloads.

Required many-to-one leakage tests include:

1. changing forbidden IDs, symbols, vendor Greeks/IVs, volume and open interest
   without changing allowed semantics leaves the public projection identical;
2. multiplying every absolute spot/forward/strike/price value by a positive
   scale while preserving normalized state and spreads produces the same public
   row, demonstrating that the absolute level is not encoded;
3. two admissible quote intervals with different midpoints but equal spot-one
   half-spread and the same fitted normalized state project identically; and
4. scanning Parquet schema, nested fields, metadata, README, config and manifest
   reveals no forbidden value or machine path.

These tests establish the declared technical capability boundary, not legal
redistribution rights. Do not claim that transformation makes vendor-derived
spreads automatically redistributable.

## 8. Distribution and rights gates

The first immutable artifact is `distribution: public_candidate`. Passing the
schema, reconstruction and leakage tests means only that it is technically
suitable for rights review. It must not be placed on a public endpoint or
described as freely redistributable.

Publishing `distribution: public` is a separate immutable Dataset identity and
materialization, never an alias or in-place manifest edit. Its manifest must
record release license, redistribution-rights basis, approver, approval date
and durable governing reference. Its sole direct upstream remains the same
completed internal Dataset, as required for every public projection; a
`public_candidate` is approval evidence, not the public artifact's lineage
source. The approved release reruns the deterministic projection from that
internal Dataset, proves byte/value equality to the approved candidate, owns
its checksums/README/license, and remains shallowly loadable without the
candidate or internal Dataset.

## 9. Numerical, parity and acceptance evidence

### 9.1 Public-only reconstruction

For every row, construct both pricers using only the public schema and frozen
configuration. Validate:

- all expiry/state indices, times and increments align;
- supports/densities satisfy finite/order/mass/moment invariants;
- DLV boundaries and corrected density/off-grid-call round trips meet the Phase
  7 `1e-12` contract;
- smooth and DLV calls match internal reconstructed calls at stored nodes and a
  fixed off-grid mesh within their Phase 7 tolerances;
- puts satisfy parity and derived ATM IV is finite and stable;
- CD-014 behavior at zero, below/at/above `ONE_MINUTE`, before the first expiry,
  at every node, between nodes and after the final node; and
- each book index, NMS predicate and half-spread bridge recomputes exactly.

Run the same capability suite after removing/mounting away Raw, Intermediate
and internal Dataset roots with `WRDS_OFF=1`.

### 9.2 Daily-front migration parity

On the 891 legacy populated dates, compare the fresh projection selector with
the pinned `market_day_from_payload` oracle date by date. Freeze ordered
side-book key/count digests and require, subject to reviewed fresh-vendor
differences:

- exactly 891 comparison dates;
- exactly 3,566,258 legacy daily-front side books in aggregate;
- identical per-date book counts, expiry dates/AM flags, side order,
  `K/F`, horizon/0.2D/NMS boundary decisions and smooth/DLV reconstructed
  prices; and
- explicit discrepancy reports for dates present only in the fresh schedule.

Do not require the fresh 917-row public candidate's total book count to equal
3,566,258: that number belongs to the legacy 891-date comparison universe. Do
not compare against the 2,192,458-book Daily25 SDLV artifact; it is a different
Experiment population. The legacy neural Direct-200 state is also not SANOS or
DLV state.

The public return has no legacy forward-adjusted oracle. Its evidence is
independent formula tests plus reconciliation to the internal fresh curves.
Likewise, the spot-one spread bridge is tested algebraically and against the
fresh internal side quotes, not inferred from the legacy experiment tensor.

### 9.3 Boundary test matrix

Offline tests must cover at least:

- first public date and a date after the typed-empty gap;
- ordinary and trading-session half days;
- AM 0.2D exactly, within/outside its `1e-4` tolerance, other AM nodes and PM
  nodes;
- 70D exactly and immediately beyond;
- NMS `-3` and `2` inclusively and one representable point outside each bound;
- call-only, put-only and two-sided active books;
- state expiries with zero retained daily-front books;
- irregular DLV supports and smooth supports differing from DLV supports;
- flat and non-flat forward/discount curves, exact/interpolated carry nodes,
  holidays, weekends, multi-session gaps and forbidden extrapolation;
- zero/negative/non-finite bridge, carry, state and nested values;
- forbidden columns at every nesting level and null forbidden fields;
- output-first reuse, wildcard reuse warning, interruption, concurrent
  publication, tampering and isolation from all upstream roots; and
- manifest capability execution and explicit non-capability assertions.

## 10. Decisions that remain blocked

Implementation may define versioned types and synthetic test scaffolding, but
must not publish a purportedly complete candidate until these choices are
recorded:

1. **CD-004 — user approval required:** approve or replace the exact previous-
   retained-date forward carry, business-session horizon, log-linear
   interpolation, no-extrapolation, typed-empty gap and first-row-null rules in
   section 4.
2. **CD-009 — user approval required:** approve or replace the
   `discounted_forward_over_spot` and `half_spread_spot` fields, Float64
   arithmetic and equations in section 5.
3. **Redistribution approval required only for `public`:** supply the license,
   rights basis, approver/date and durable reference. This is not required to
   create an internal `public_candidate`, but the candidate must remain
   non-public until granted.

Preserving CD-007's recovered population and CD-014's legacy pre-first-expiry
jump needs no additional choice. Any change to either is a new semantic/model
decision and remains blocked pending explicit approval.

Signed: `quantagent` — independent Phase 8 quantitative/data design reviewer

## Lineage reconciliation with the code design audit

The code audit proposes `public_candidate <- internal <- ...` followed by
`public <- public_candidate`, implemented by copying the candidate payload.
That transition is **not adopted**. The binding repository rule says that the
public dataset is generated as a lossy projection from the completed internal
Dataset, and the Phase 2 artifact contract deliberately enforces the completed
internal Dataset as the sole direct upstream of *both* non-internal statuses.
Publication and deep validation reject a `public_candidate` source. Changing
that rule in Phase 8 would be a lifecycle semantic change, not a necessary
consequence of separating review from release.

The reconciled contract is:

```text
public_candidate <- internal Dataset
public           <- same internal Dataset
```

The candidate is mandatory **approval evidence**, but is not the public
artifact's data-producing lineage parent. Approval therefore creates a new
immutable `public` artifact and reruns the exact deterministic projection from
the candidate's recorded internal pointer; it neither aliases nor mutates the
candidate and it does not merely relabel or byte-copy it. This preserves the
meaning of direct lineage, the existing Phase 2 validation invariant, and
shallow reuse of a complete public artifact after either evidence source is
unavailable.

Promotion must fail closed unless all of the following hold:

1. the candidate is complete, has status `public_candidate`, passes the full
   semantic/leakage/capability validation, and has exactly one completed
   `internal` Dataset upstream;
2. the public build names that identical internal pointer and uses the exact
   candidate projection configuration, schema version, producer version,
   deterministic writer settings and capability contract; no wildcard may
   build or publish it;
3. the independently regenerated public `data.parquet` is byte-for-byte equal
   to the candidate `data.parquet`, with equal SHA-256, and its canonical
   recursive schema, row ordering, row/value-content digest, `schema.json` and
   `capabilities.json` also match exactly; any mismatch requires a new
   candidate and approval rather than release; and
4. only release wrappers may differ: distribution status, public identity,
   README, license/approval documents, validation report and manifests.

The public artifact's canonical configuration and manifest must preserve the
approved candidate pointer as a non-lineage evidence reference, the candidate
payload/schema/config/capability digests, the shared internal pointer, and the
equality-validation result. They must also contain the release-license bytes'
checksum plus the required license identifier, redistribution-rights basis,
approver, approval date and durable governing reference. Those legal fields
and evidence digests participate in public identity. The release must copy the
approval and license evidence into its own self-contained files, checksum them,
and validate them before atomic commit; subsequent shallow loading must not
require the candidate, internal Dataset or an external approval server.

Accordingly, the code audit's proposed artifact-store transition-table change
is rejected unless the user first approves a repository-wide reversal of the
Phase 2 internal-only projection-source contract. No such change is needed for
a distinct, auditable candidate-to-approval workflow.

Signed reconciliation: `quantagent` — 2026-10-12
