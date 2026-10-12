# Phase 9 quantitative/data-contract design audit

**Reviewer:** `quantagent`
**Date:** 2026-10-12
**Scope:** experiment materialization design only; no WRDS/network access and no
implementation or `quant_data` changes
**Verdict:** **READY FOR THE INTERNAL REFERENCE EXPERIMENT; PUBLIC-SOURCE
MATERIALIZATION REMAINS BLOCKED BY PHASE 8 / CD-004 / CD-009**

## 1. Authoritative parity endpoint

The Phase 9 migration oracle is the completed all-history Joint Direct-200
DySANOS experiment at:

```text
legacy_QUANTDATA_TRAINING_DATA/
  flex-shared-strikes/joint-direct/
  dte1-5-weekly-no-am-all-history-v1
```

Its producer is pinned revision `de15be96`, file
`quantdata/dysanos/flex/run_all_history_joint_direct.py`, blob
`6b1c408c3ab1d32e92115cb95c91b3472bcf6086`. The completed report schema is
`flex-all-history-joint-direct-v1`; the checkpoint schema is
`flex-joint-direct-dynamics-v1`; and the target-history schema is
`flex-all-history-joint-direct-targets-v1`.

Two similarly named paths are not authoritative here:

- `widb.timeseries.build_time_series` is the older generic padded HDF5
  materializer. It is not on the recovered DySANOS production call graph and
  must not define Phase 9 fields or the 1,000-option default cap.
- `spx-2022-2025-daily25-shared200-fixed5-market-v1.npz` is the separate
  Daily25 SDLV/JAX population of 2,192,458 books. It is not an alternative
  serialization of the daily-front 3,566,258-book universe.

The legacy producer consumes variable-length `FlexMarketDay` objects and pads
only each Torch minibatch. Phase 9 deliberately changes the transport to one
self-contained HDF5 Experiment, but an adapter must restore the same logical
days and minibatch tensors. The trained neural state/checkpoint remains an
experiment output; it is not copied into the input HDF5 and is never treated as
curated SANOS/DLV Dataset state.

## 2. Exact source and population selection

The reference Experiment has one direct upstream: the exact immutable
**internal** SANOS Dataset pointer. It has no Raw or Intermediate direct
upstream and never consumes the legacy cache at runtime. Iterate the Dataset's
ordered dates and:

1. include trading-session half days;
2. exclude only the declared policy-owned typed-empty date; fail on any
   unexpected typed empty, malformed partition, or day with no retained
   daily-front books;
3. start with separate real call and put books for which the corresponding
   `call_active`/`put_active` flag is true;
4. require European exercise style and a finite corresponding smooth fitted
   call/put value;
5. retain business DTE in `[0, 70]`, excluding a value equal to `0.2` within
   `1e-4`; and
6. retain inclusive AF-ATM canonical NMS `[-3, 2]`, equivalently inclusive
   reconstructed strike bounds from the smooth SANOS ATM curve.

The source row order is not trusted implicitly. Canonical output order is
ascending expiry time, then calls before puts, then ascending `K/F` within each
side. A pricing location is the first occurrence of each exact
`(expiry_index, K/F)` key in that quote order. Call and put books remain
separate objective observations even when they map to the same pricing
location. Do not assume both sides survive on fresh data.

The legacy oracle has:

| Quantity | Exact value |
|---|---:|
| populated dates | 891 |
| first / final date | 2022-01-03 / 2025-08-29 |
| separate side books | 3,566,258 |
| calls / puts | 1,783,129 / 1,783,129 |
| minimum / maximum quotes per date | 2,470 / 7,898 |
| maximum unique pricing locations per date | 3,949 |
| maximum retained market expiries per date | 36 |
| pre-2025 / 2025 diagnostic dates | 729 / 162 |
| exact five-session weekly pairs | 861 |

The freshly downloaded full-range Dataset instead plans 918 sessions, 917
populated plus the one policy typed-empty date. Therefore 891 and 3,566,258
are overlap migration gates, not target fresh totals. Phase 11 must report the
fresh Experiment's new counts and every date/population discrepancy; it must
not discard fresh-only dates to force the legacy total.

## 3. Frozen reference configuration

The exact selector is `smoothness=0.25`, non-binding upstream
`max_num_options=100000`, `excluded_business_dtes=(0.2,)`, and the geometry:

- 18 model business-DTE nodes: `1,2,3,4,5,10,...,70`;
- 200 inner shared strikes and NMS range `[-3,2]`;
- `min_dlvsq=1e-4`, `max_dlvsq=9`, forward volatility `[0.01,1]`, eta
  `[0.01,0.5]`, boundary scales `0.5/1.5`, minimum log-spacing fraction
  `0.05`, and the recorded DLV-cap barrier width/std `0.1/0.35`.

The neural configuration is an affine Direct state of width 200 with 3,636
decoded raw coordinates (`18 * (200 + 2)`), runtime precision 32, objective
`fit_weight=0.05`, and equal-date quote weighting. The completed run used the
recorded 300 epochs, day batch size 128, learning rates `1e-4/3e-5/3e-4`,
dynamics/gauge/cap weights `0.1/0.1/1`, 20 warm-up epochs, 50 dynamics-ramp
epochs, gradient clip 10, seed 9281, and the remaining schedule/convergence
fields in the canonical report. All of these fields belong in experiment
configuration even though only the selector and tensor contract affect HDF5
bytes.

## 4. HDF5 logical schema

Use one atomic `data.h5` with schema
`sanos-dysanos-joint-direct-market-v1`. Let `D` be included days, `Q` the
maximum quote count, `P` the maximum unique pricing-location count, `E` the
maximum retained expiry count, and `M=18`. Derive `Q/P/E` from the selected
source, record them in the manifest, and never truncate to a configured cap.

Required arrays are:

| Group/field | Shape and dtype | Meaning / canonical pad |
|---|---|---|
| `day/date_days` | `(D,) int32` | days since Unix epoch, strictly increasing |
| `day/is_half_day` | `(D,) bool` | trading-session flag |
| `day/spot_ref` | `(D,) float64` | positive absolute spot reference |
| `day/{quote,price,expiry}_count` | `(D,) int32` | valid prefix lengths |
| `quote/mask` | `(D,Q) bool` | true exactly on each valid prefix |
| `quote/expiry_index` | `(D,Q) int64` | retained-expiry index; pad `0` |
| `quote/price_index` | `(D,Q) int64` | unique-price index; pad `0` |
| `quote/strike` | `(D,Q) float64` | `K/F`; pad `1` |
| `quote/is_call` | `(D,Q) bool` | side; pad `true` |
| `quote/{bid,ask,fit}` | `(D,Q) float64` | direct-side price in `DF*F` units; pads `0,1,0` |
| `quote/{log_volume,log_open_interest}` | `(D,Q) float64` | `log10(1+vendor count)`; pad `0` |
| `price/mask` | `(D,P) bool` | valid unique pricing locations |
| `price/expiry_y` | `(D,P) float64` | business years; pad `0` |
| `price/strike` | `(D,P) float64` | `K/F`; pad `1` |
| `expiry/mask` | `(D,E) bool` | valid retained expiries |
| `expiry/time_y` | `(D,E) float64` | business DTE / 252; pad `0` |
| `expiry/date_days` | `(D,E) int32` | expiry calendar date; pad `0` |
| `expiry/id` | `(D,E) int64` | internal vendor expiry ID; pad `0` |
| `expiry/is_am` | `(D,E) bool` | settlement flag; pad `false` |
| `expiry/atm_annual_vol` | `(D,E) float64` | smooth AF ATM annual volatility; pad `0` |
| `expiry/{strike_lower,strike_upper}` | `(D,E) float64` | inclusive canonical-NMS bounds; pad `1` |
| `model/business_dte` | `(M,) float64` | exact 18-node grid |
| `model/expiry_y` | `(M,) float64` | grid divided by 252 |
| `model/atm_total_vol` | `(D,M) float64` | smooth ATM annual vol times `sqrt(T)` |

Also store the per-day `market_uid`, exact source-day training fingerprint and
the five eligibility-audit counts. Quote IDs need not consume a 3.6-million
string dataset: reconstruct them exactly as
`date:expiry_id:{K/F:.12g}:{C|P}` and require the reconstructed fingerprint to
equal the stored fingerprint.

All valid source numerics are stored as float64 because that is the observed
`FlexMarketDay` contract. The adapter casts pricing tensors to Torch float32
only when constructing the precision-32 runtime batch. Index tensors become
Torch `long`, side/mask tensors become `bool`. Canonical padding is validated
even though masks remove it from the objective; padding may never encode a
second hidden data channel.

Use day-major chunks and deterministic, versioned HDF5 writer/compression
settings. File checksum is manifest evidence; semantic content hashes over
each logical unpadded array protect parity independently of HDF5 library
container details.

## 5. Splits and objective semantics

The authoritative all-history run has **no temporal validation holdout**. Its
explicit split arrays are therefore:

- `split/train_mask`: true for every one of the `D` dates;
- `split/validation_mask`: false for every date;
- `split/pre_2025_diagnostic_mask` and `split/2025_diagnostic_mask`: 729/162
  on the oracle, used only for reported diagnostics; and
- `dynamics/weekly_origin_index` and `weekly_endpoint_index`: exact pairs
  separated by five NYSE sessions (861 pairs on the oracle), never adjacent-row
  guesses.

For each date the loss is the mean over its unmasked separate side books; the
batch loss is then the mean over dates. For a model price `p`, bid `b`, ask
`a`, smooth target `f`, and half-spread `h=(a-b)/2`, the quote loss is

```text
[max(p-a,0) + max(b-p,0) + 0.05*abs(p-f)] / h.
```

A nonempty validation holdout, random split, changed date weighting, or padded
quote weighting is a different Experiment identity and must not be presented
as reference parity.

## 6. Units, reconstruction and overnight clock

`strike` is `K/F`. Direct-side bid, ask and smooth fit are cash option values
divided by `DF*F`. Put-to-call conversion is exactly
`C/(DF*F)=P/(DF*F)+1-K/F`; the model prices each unique location once as a call
and restores the original separate sides. No absolute strike or option price
enters these arrays. `spot_ref` is nevertheless absolute and is included only
because the observed target-history artifact records it; this fact alone makes
the reference Experiment internal-only.

The source Dataset's `overnight_variance_fraction` must be copied into the
canonical Experiment configuration and checked against the source manifest.
At the compatibility default it is `0.2`, affecting AM business DTE, expiry
order, horizon/NMS inputs and model times. The observed selector separately
contains the literal `excluded_business_dtes=(0.2,)`. Do not silently couple
these two knobs: with a nondefault overnight fraction the literal exclusion
has the literal legacy behavior. A new `exclude_same_day_am` rule that follows
the configured fraction would be sensible, but is a new selection semantic
requiring an explicit compatibility decision.

Sensitivity tests at overnight fractions `0, 0.2, 0.5, 1` must show that PM
expiries remain unchanged; AM clocks, selection, padding counts and HDF5
identity change where expected; and a change cannot reuse the `0.2` artifact.

## 7. Internal versus public eligibility

The exact Joint Direct-200 reference materialization is **internal-only**. It
requires source bid, ask and smooth fitted prices for containment/central-fit
loss, absolute `spot_ref`, vendor expiry identity, and optionally volume/open
interest fields. The public allowlist deliberately cannot reproduce exact
vendor books, quote-containment tests, liquidity models or absolute levels.
Neither `public_candidate` nor `public` therefore satisfies this HDF schema's
capability requirement.

A later public-source Experiment may materialize complete public SANOS/DLV
state, geometry, normalized half-spreads and forward-adjusted returns for the
advertised normalized capabilities. It is a different schema/producer/config,
cannot claim Direct-200 quote-objective parity, and awaits Phase 8 approval of
CD-004 and CD-009. This does not block implementation of the internal reference
HDF5 now.

## 8. Identity, files and lifecycle

The Experiment `ArtifactSpec` must include the sole source Dataset pointer,
complete canonical selection/model/split/padding/dtype/writer configuration,
schema version and resolved `@version` producer closure. The closure includes
the daily-front selector, clock/NMS rules, SANOS reconstruction, pricing-key
deduplication, split/pair builder, dtype/padding policy, HDF5 writer and
semantic validator.

Publish atomically with `data.h5`, `README.md`, `config.json`, `manifest.json`
and a completion marker written last. Record HDF5 checksum, logical array
hashes, counts/shapes/dtypes, source identity, exclusions and consumer version.
The artifact is disposable, but deleting it never affects the Dataset.
Output-first shallow reuse works after the Dataset disappears; deep validation
traverses it. `version="*"` remains warning-emitting reuse-only and cannot build
or replace an Experiment.

## 9. Parity and adversarial acceptance tests

1. On the legacy oracle, assert all exact counts/maxima in section 2 and the
   exact ordered 891 dates. Assert the daily-front selector against the pinned
   source independently, not against the new serializer itself.
2. Compare every valid float64/int/bool/string value and reconstructed
   fingerprint bit-for-bit before and after HDF5 round trip. Assert mask sums,
   contiguous prefixes, index bounds, key mappings and canonical padding.
3. For fixed minibatches, require adapter Torch tensors and masks to be bitwise
   equal to direct `materialize_day/materialize_batch` output after the same
   float32 cast. Require ragged and padded equal-date objectives to agree.
4. Reconstruct smooth SANOS prices from source state at every stored key and
   retain the Phase 7 maximum absolute `5e-8` gate; verify call/put parity and
   the corrected DLV/state geometry at its existing `1e-12` gates before
   materialization.
5. Evaluate the frozen legacy checkpoint through both input paths. Counts and
   containment classifications are exact; for float32 prices/objectives use
   `rtol=1e-6, atol=1e-7`, with report aggregates within `5e-6` relative.
6. Validate the observed target artifact without treating it as input:
   float64 states `(891,200)`, float64 spots `(891,)`, state digest
   `3525f0d28a7fdc9a5535c80d05c9a4ddd83dfb9c59e8b15fd6ffcfa568477cb7`;
   and checkpoint float32 daily embedding `(891,200)` with 891 fingerprints.
   Do not demand bitwise reproduction from a fresh stochastic/GPU retraining.
7. Reject altered source pointer, date order, mask/count, side, expiry mapping,
   normalization, split, weekly pair, overnight fraction, config identity,
   HDF5 dataset, checksum or completion evidence. Test interruption,
   concurrent identical publication, source deletion after completion,
   wildcard hit/miss and offline loading with `WRDS_OFF=1`.
8. On fresh production, compare all 891 overlapping dates and report
   fresh-only dates/vendor revisions explicitly. No legacy artifact is a build
   input, and no unexplained mismatch is waived to obtain an old checksum.

## 10. Decisions and blockers

- No new decision is needed to implement the internal HDF5 adapter and its
  offline/synthetic/legacy-oracle tests. It can proceed while Phase 8 waits.
- A temporal validation holdout would be a new experiment, because the
  observed all-history reference trains on all dates.
- Generalizing the literal `0.2` exclusion into `exclude_same_day_am` under
  nondefault overnight fractions requires an explicit compatibility decision.
- Any public-source Experiment is blocked until Phase 8 resolves CD-004 and
  CD-009 and must use a different capability/schema contract.
- Building the fresh production artifact remains a Phase 11 action requiring
  the fresh validated Dataset; it does not authorize WRDS access now.

## Evidence used

- `AGENTS.md`, `PROGRAM_PLAN.md` Phase 9, `docs/reference-baseline.md`,
  `docs/data-contracts.md`, and the Phase 1 machine-readable inventories.
- Pinned `de15be96` producer, market/config/training/joint-dynamics/weekly-pair
  sources inspected through `git show`; current `quant_data` was read-only.
- Existing completed report and target/checkpoint inventories under the
  symbolic legacy training root.
- Offline read-only scan of 891 daily HDF5 payloads through the SANOS reference
  adapter and exact pinned market selector; its total reproduced 3,566,258
  books and measured the padding maxima in section 2.
- `gpuctl resources --json` reported all checks `OK`; no GPU, WRDS or network
  operation was performed.

Signed: `quantagent` — independent Phase 9 quantitative/data-contract reviewer
