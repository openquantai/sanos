# Phase 10 quantitative/data-contract design audit

**Reviewer:** `quantagent`
**Date:** 2026-10-13
**Scope:** fresh-download pilot design, legacy daily-oracle comparison, numerical
comparison contract, classifications, and production stop rules; offline only
**Verdict:** **PASS FOR IMPLEMENTATION; LIVE WRDS ACCESS REMAINS BLOCKED**

The date matrix and operator sequence proposed by the code design audit are
quantitatively appropriate. Implementation may proceed under the exact contract
below. No WRDS connection or 2FA request is authorized until the comparator,
Raw provenance output, restart/tamper regressions and pilot runbook required by
the code audit are implemented, reviewed, and passing offline.

**Implementation-time clarification (2026-10-13):** the original design
requirement to derive DLV from the smooth state was incorrect and is superseded
by Section 3.2 below. The current comparator's attempt to linearize the
positive-smoothness densities and call `dlvsqs()` is a blocking comparator
defect, not evidence of an invalid fitted day. Remove that operation before the
implementation gate.

## 1. Frozen representative pilot

Use one contiguous SPX request from **2024-12-23 through 2025-01-16**, with
`include_half_days=true`, source `optionmetrics`, a new immutable source
generation, default `overnight_variance_fraction=0.2`, and no more than two WRDS
connections. This is better evidence than disconnected days because it tests
boundary inputs and restart behavior under one catalog probe and one generation.

The exact 16 NYSE observation sessions are:

```text
2024-12-23  2024-12-24  2024-12-26  2024-12-27
2024-12-30  2024-12-31  2025-01-02  2025-01-03
2025-01-06  2025-01-07  2025-01-08  2025-01-10
2025-01-13  2025-01-14  2025-01-15  2025-01-16
```

The pilot cases and mandatory observations are:

| Case | Date/evidence | Required outcome |
|---|---|---|
| Ordinary | 2024-12-23 and 2025-01-06 | Populated fitted days and full daily-oracle comparison. |
| Half-day | 2024-12-24 | Populated fresh day with `is_half_day=true`; the absent legacy file is an expected `oracle_gap`, not permission to omit the fresh day. |
| Irregular expiry | 2025-01-07 quotes for expiry 2025-01-09 | Raw must retain the vendor observation; daily normalization must remove it under the pinned irregular-expiry policy and record a positive removal count. If the fresh vendor extract no longer contains it, classify the case as unexercised and stop for review. Do not substitute a different case silently. |
| Policy empty | 2025-01-08 | Exact-schema typed-empty fitted artifact with reason `known_bad_day_all_tickers`; the legacy physical empty marker must be present and hash-valid. Raw may be nonempty. |
| AM settlement | 2025-01-16, expiry 2025-01-17 AM | Retained internal expiry with business DTE `0.2` (within the clock comparison tolerance) and model time `0.2/252`; it is not removed by the internal filter because `remove_am=false`. The legacy oracle has 186 active side books on this node. |

Offline planning must continue to assert exactly **68 Raw artifacts**: one
security artifact, 16 options, 16 spot, 16 forwards, 16 zero curves, and the
three 2024-12-20 boundary artifacts for spot, forwards and zero curve. There
must be 16 terminal fitted-day artifacts. A fresh preflight must report 68
misses, zero reuse and no network access; any reuse requires a new generation.

## 2. Frozen legacy expectations

The comparator must pin the existing catalog identity
`f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280`
and every selected physical file checksum before reading values. Within the
pilot interval, the frozen legacy collection has 15 files: 14 populated files,
one empty marker on 2025-01-08, and no file on the half-day 2024-12-24.

For the 14 populated oracle days the catalog totals are:

| Quantity | Legacy expectation |
|---|---:|
| input-market expiries | 448 |
| filtered strike nodes | 40,436 |
| active call books | 38,022 |
| active put books | 38,022 |
| active side books | 76,044 |
| active books at 0.2 business DTE | 186 |

For 2025-01-16 alone the legacy expectations are 34 input expiries, 2,814
strike nodes, 2,639 active calls, 2,639 active puts, and 186 active books at
0.2D. These are migration-oracle values, not values to force onto a revised
vendor extract. If fresh market evidence changes, the observed fresh counts
must remain unchanged and the discrepancy must be adjudicated.

If all comparable data agree, the expected top-level pilot classification is
14 `match`, one `policy_empty`, and one expected `oracle_gap`.

## 3. Exact comparison population and ordering

The fresh fitted Parquet deliberately contains audit rows that the legacy HDF5
does not. Therefore the comparator must never compare full fresh row count to
legacy strike-node count. It must derive a read-only **legacy comparison view**
from `selected_reference_view(frame)`, i.e. exactly
`options_filter_eligible=true`, and group it into the legacy market geometry.

Both sides must be canonicalized and then proven already canonical:

1. input expiries in increasing `(time_to_expiry_years, expiry_id)` order;
2. within an expiry, strictly increasing `normalized_strike`, with stable source
   order retained only as diagnostic evidence;
3. unique `(expiry_id, normalized_strike)` keys;
4. smooth retained expiries in increasing `smooth_input_expiry_index`, with the
   index mapping exactly equal to legacy `af_pricer.input_index`; and
5. all list shapes, null patterns and dtypes checked before numerical values.

Sorting must not conceal an ordering defect: the report records whether each
source was already canonical and treats a noncanonical or duplicate source as
a structural discrepancy.

### 3.1 Comparable market fields

The following mappings are exact comparison fields:

| Legacy field | Fresh fitted view |
|---|---|
| `d`, `ticker`, `spot_ref`, `spot_ret` | `trading_date`, `ticker`, `spot_reference`, `spot_return` |
| `market.expiries`, `expiry_dates`, `expiry_ids`, `is_am` | unique per-expiry `time_to_expiry_years`, `expiry_date`, `expiry_id`, `expiry_is_am` |
| `market.strikes` | `normalized_strike` |
| `market.bids`, `market.asks` | `pure_bid_price`, `pure_ask_price` |
| `market.is_american` | `is_american` |
| `market.pre_af_masks` | `mask_arb_free` |
| `market.model_masks` | `smooth_model_mask` |
| normalized real call/put bid/ask arrays | `pure_market_call_bid/ask`, `pure_market_put_bid/ask` |
| side log-volume/open-interest arrays | corresponding `market_*_log_*` fields |
| `call_active`, `put_active` | exact fresh side masks |
| `af_call_mids`, `af_put_mids`, `af_mid_available` | `smooth_call_mid`, `smooth_put_mid`, `smooth_mid_available` |

The report must separately record full fresh input rows, selected rows, input
expiries, initial retained expiries, smooth retained expiries, active sides,
masked strikes and dropped expiry IDs. Equality of only aggregate counts is not
a match.

### 3.2 No-arbitrage, SANOS and DLV state

For the smooth state, compare legacy `af_pricer` expiries, input indices,
strike masks, input strikes/bids/asks, `fits1`, `fitsX`, variances, per-expiry
supports/densities, surface supports/densities and x-to-surface mappings with
the corresponding fresh smooth fields. Compare all `fit_result` values and
per-strike fit errors. Exclude wall-clock runtimes from content comparison.

Reconstruct a smooth pricer independently from each side. Compare calls on the
union of all input/support strikes and on a fixed off-grid vector at zero,
half-one-minute, one-minute, pre-first, every node, every midpoint and
post-final times. Compare parity-derived puts and ATM implied volatility.

Do **not** call `SanosPricer.dlvsqs()` on the positive-smoothness SANOS state,
and do not manufacture a `smoothness=None` pricer from its densities in order
to do so. Squared DLV is the finite-volume transition of the separate
piecewise-linear state. Positive SANOS smoothness changes the payoff kernel:
it can give positive time value at a support node with zero mass, for which the
discrete DLV denominator is zero. Consequently such a derived value is not a
well-defined invariant of the smooth surface and the inverse need not recover
the smooth densities. Smooth parity ends at the complete stored geometry,
variances, supports/densities and mappings, plus reconstructed smooth calls,
puts and ATM implied volatility across the time branches above.

The legacy smooth HDF5 stores `market.pre_af_masks` but **does not store the
complete pre-filter unsmoothed no-arbitrage supports, densities, variances or
DLV squares** now retained by SANOS. These fields are item-level `oracle_gap`,
not a failed comparison and not a claimed match. Their pilot evidence is:

- exact initial mask agreement where the mask is available;
- successful reconstruction of `compatibility_dlv_pricer`;
- support monotonicity, density non-negativity, unit mass and first moment;
- exact zero DLV boundaries and corrected density/DLV round trip on the
  separate compatibility unsmoothed state;
- quote containment and calendar no-arbitrage invariants; and
- the already pinned Phase 6 source-parity fixtures.

The per-date top-level result may still be `match` when every field the legacy
actually stores matches and these explicitly declared item-level oracle gaps
pass fresh invariants. The report must not claim legacy numerical parity for
the missing unsmoothed state.

## 4. Named numerical tolerance contract

Implement a frozen dataclass or equivalent named
`DailyLegacyComparisonTolerances`, decorated with
`@version("phase10-legacy-daily-comparison-tolerances-v1")`, and include its
resolved version and canonical values in every report and report identity.
Version 1 is:

| Category | Contract |
|---|---|
| dates, strings, IDs, indices, dtypes, shapes, order, masks, null pattern and populations | exact |
| persisted Float32 market/geometry/quote/Greek values, after validating the expected dtype | bitwise exact (`rtol=0`, `atol=0`) |
| Float64 supports, densities, variances, fitted values and comparable fit diagnostics | `rtol=0`, `atol=1e-12` |
| DLV squares and density/DLV inverse | apply only to the separately identified unsmoothed DLV state, never to the positive-smoothness SANOS state; `rtol=0`, `atol=1e-12` for a matching stored oracle and the tighter within-runtime inverse tolerance below |
| calls reconstructed through the stable public `SanosPricer` API on the canonical grid | `rtol=0`, `atol=5e-8`, solely for the documented Float32-query promotion boundary |
| density/DLV inverse performed in one runtime from the same persisted unsmoothed state | `rtol=0`, `atol=2e-14` |

The broader invariant limits (`1e-7` density/calendar, `1e-6` quote
containment, `1e-4` density mass/moment) are validity limits, **not parity
tolerances**, and may not turn an oracle mismatch into `match`.

Every floating comparison must report count, maximum absolute error, maximum
relative error where defined, maximum ULP distance for equal-width floating
dtypes, first mismatch key, nonzero-difference count and above-tolerance count.
Nonzero differences remain visible even when within tolerance. NaN/infinity and
null placement are structural failures, not values handled by `allclose`.

## 5. Classification contract

Use exactly these terminal classes per date:

- **`match`**: both sides are populated, all comparable structure/populations
  are exact, every comparable numerical field is within the named tolerance,
  and every fresh-only invariant passes.
- **`oracle_gap`**: the fresh day is valid and populated but the pinned catalog
  has no physical legacy file. This is nonblocking only for a date explicitly
  declared before the run (2024-12-24 here). A newly discovered gap stops the
  pilot. Field-level oracle gaps, such as the unsmoothed DLV state, are recorded
  separately and cannot hide a mismatch in fields that do exist.
- **`policy_empty`**: the fresh fitted artifact is exact-schema empty with the
  configured reason and the legacy artifact is the expected physical empty
  marker. An empty on any other date, a populated legacy day against a fresh
  empty, or a wrong reason is `unexplained_discrepancy`.
- **`unexplained_discrepancy`**: any other structural, population, mask or
  numerical difference, including a possible vendor revision which has not yet
  been approved.
- **`approved_discrepancy`**: an otherwise unexplained discrepancy covered by
  explicit user approval whose immutable record contains the exact date/fields,
  old/new evidence digests, explanation, approver, approval date and durable
  reference. The comparator must never infer this class from prose, a tolerance
  pass, or the phrase “vendor revision.”

Classification precedence is malformed/tampered oracle or fresh artifact
failure (abort report as failed), then `policy_empty`, declared `oracle_gap`,
populated comparison, and finally explicit approval overlay. A report with any
`unexplained_discrepancy` has outcome `blocked`. `approved_discrepancy` remains
visible and contributes to separate counts; it is never relabelled `match`.

## 6. Vendor revisions and stop rules

A difference in a current WRDS extract is evidence to investigate, not a reason
to rewrite fresh data or relax tolerances. For each suspected vendor revision,
the evidence bundle must bind the Raw artifact IDs, query digests, physical
tables, source generation, row/empty counts, persisted schema and payload
checksums; identify the first divergent daily stage and exact affected keys;
show that current SANOS/reference parity tests still pass; and record the
legacy file/catalog checksums. Because the old Raw extract was not preserved as
the authoritative oracle, this supports a **suspected vendor revision**, not
automatic proof. It remains `unexplained_discrepancy` until explicitly
approved.

Stop the pilot and do not begin Phase 11 when any of the following occurs:

1. catalog/file checksum or legacy schema differs from the pinned inventory;
2. the Raw plan is not exactly 68 fresh misses or a boundary identity differs;
3. authentication, 2FA, permission, connection-limit or unknown WRDS failure;
4. any incomplete/corrupt Raw artifact, unknown empty day, missing dependency,
   solver terminal failure, nonzero daily worker failure or non-reusable output;
5. the irregular-expiry case is not actually observed and removed;
6. the half-day is not populated and marked, the policy-empty reason differs,
   or the 0.2D AM node is absent;
7. any unexpected oracle gap or unexplained discrepancy; or
8. evidence contains credentials, OTPs, usernames or unapproved proprietary
   value samples.

An interruption is not itself a discrepancy. Preserve completed immutable
artifacts and resume the same generation only after the required renewed 2FA
readiness confirmation. Never deliberately provoke a live failure.

## 7. Required implementation and acceptance tests

Before live access, implement and independently review the comparator and Raw
provenance/restart/tamper work identified by the code audit. Quantitative
acceptance additionally requires offline tests for:

1. exact synthetic populated match, with all field mappings above;
2. one mismatch in every structural family (date/order/key/dtype/shape/mask/
   population/null) producing `unexplained_discrepancy`;
3. each numerical category at zero, exactly at tolerance, and one representable
   value beyond it; every nonzero within-tolerance difference remains reported;
4. independent smooth state reconstruction across all CD-014 time branches,
   put-call parity and ATM IV, including a zero-mass support with positive
   smooth time value; the comparator must not invoke or report smooth DLV;
5. an initial unsmoothed state recorded as item-level `oracle_gap` while its
   exact zero DLV boundaries and corrected density/call round-trip invariants
   remain mandatory;
6. the expected missing half-day, an unexpected missing oracle day, the exact
   policy empty, wrong empty reason, fresh-empty/legacy-populated and the
   reverse;
7. approval overlay accepted only when its scoped evidence digests match, and
   tampered/stale/broadened approval rejected;
8. a market-value change never auto-classified as vendor revision or approved;
9. dropped smooth expiry/input-index alignment and a state expiry with no
   active side book;
10. the 2025-01-16 AM clock/population, 2025-01-07 irregular-removal diagnostic,
    and 2024-12-24 half-day/oracle-gap contract from fixed offline fixtures;
11. report determinism, atomic publication, interruption cleanup, symlink and
    unexpected-file rejection, read-only legacy access, and binding of every
    fresh/legacy artifact checksum and version; and
12. the installed human/JSON CLI path, with no build or network action, stable
    progress, actionable exit codes, and a checksum-stable report.

The comparison report must contain no Raw bids/asks or other vendor rows. Keep
the full internal report in the external SANOS evidence root; only non-secret
counts, classifications, maxima and digests may be committed.

## 8. Offline evidence reviewed

- Read `AGENTS.md`, Phase 10 of `PROGRAM_PLAN.md`, the Phase 10 code design
  audit, the migration baseline/runtime inventory, current legacy adapter,
  fitted schemas/configuration, pinned source documentation and relevant tests.
- Independently reproduced the 16-session calendar, 2024-12-24 half-day,
  2025-01-09 irregular-expiry registry entry, 15/14/1 legacy file population,
  per-date representative counts and the 14-day aggregate above.
- Offline Raw dry-run reproduced `planned=68`, `missing=68`, `reused=0`,
  `boundary_partitions=3`, all on 2024-12-20, with `network_accessed=false`.
- `PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
  tests/test_reference_legacy_market.py tests/test_reference_inventory.py
  tests/test_widb_calendar.py` passed **19 tests** in 2.94s.
- During comparator implementation, the synthetic exact-match fixture
  reproduced the smooth-DLV domain error: an 11-business-day smooth state had
  positive call increment `0.01720725881800056` at a support with zero mass,
  and `dlvsqs()` correctly rejected the resulting transition reconstruction.
  This directly supports the Section 3.2 clarification; it does not weaken the
  mandatory DLV round trip for the separate unsmoothed state.

No network, WRDS connection, production artifact, GPU, or `quant_data` write
was performed. The legacy collection and imported reference source were read
only.

Signed: `quantagent` — independent Phase 10 quantitative/data-contract design reviewer
