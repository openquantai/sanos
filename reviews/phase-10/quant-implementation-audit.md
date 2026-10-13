# Phase 10 quantitative/data implementation audit

**Reviewer:** `quantagent`
**Date:** 2026-10-13
**Scope:** complete uncommitted Phase 10 offline implementation relative to
`5256f9b`; no network or WRDS access
**Verdict:** **PASS — OFFLINE PREREQUISITES ONLY; LIVE PILOT NOT AUTHORIZED**

The implementation meets the amended Phase 10 quantitative and data-contract
gate. This verdict authorizes committing the offline comparator, provenance,
tests and runbook. It does **not** authorize a WRDS connection. The live pilot
still requires a clean reviewed commit, an immediately present user confirming
WRDS 2FA readiness and no other client on the account, and a healthy
`gpuctl resources --json` daemon result with every check `OK`.

## Findings

No blocking quantitative or data-contract finding remains in the reviewed
tree.

Two material issues found during review were fixed before this verdict:

1. The named v1 tolerance dataclass initially accepted arbitrary or nonfinite
   constructor values while retaining the same version. It now rejects every
   value other than the seven frozen v1 constants. A `NaN` tolerance can no
   longer make every numerical difference appear within tolerance.
2. The fixed pilot initially constrained only the 0.2 clock, irregular-expiry
   switch and AM switch. It now requires exact equality to the canonical
   `DailyWidbConfig()` and `DailySurfaceConfig()`. Changes such as
   `min_bid=0.5` or `smoothness=0.5` fail before comparison and cannot be
   concealed by a nonbinding setting or approval overlay.

The comparator also now reads and validates the immutable Raw options artifact
for 2025-01-07 and records only non-secret evidence that expiry 2025-01-09 was
actually present. This closes the distinction between exercising the named
irregular case and observing only an unidentified positive removal count.

## Population, ordering and field parity

- The fresh comparison population is exactly
  `selected_reference_view(frame)`, i.e. rows with
  `options_filter_eligible=true`. Full fresh input rows are not incorrectly
  compared with the smaller legacy population.
- Full input and selected populations are separately reported:
  `fresh_input_rows`, `fresh_selected_rows`, full `input_expiries`,
  `selected_expiries`, initial and smooth retained expiries, active sides,
  smooth masked strikes and dropped smooth expiry IDs. The independently
  recomputed fitted-partition statistics provide the corresponding full-stage
  counts and rejection categories.
- Fresh order is tested before canonical selection. Legacy expiries are proven
  increasing in `(time_to_expiry, expiry_id)`, each strike vector is strictly
  increasing, and both expiry IDs and strike keys are unique. Noncanonical or
  duplicate input is a structural mismatch; sorting cannot hide it.
- The comparator covers the complete legacy market payload used in production:
  date/ticker/spot/return, expiry geometry, strikes, pure bid/ask books,
  exercise style, pre-AF and model masks, direct call/put books, side
  liquidity vectors, active masks, smooth mids and availability. The legacy
  `uid` is configuration display metadata rather than a market value and the
  two wall-clock runtime fields are deliberately excluded.
- Smooth-state comparison covers input indices, masks, input books, expiry and
  surface supports/densities, mappings, variances, `fits1`, `fitsX`,
  smoothness, minimum half-spread, and all stored fit-result vectors and
  summaries. Mapping, state-index, Boolean-mask, integer-ID and floating dtypes
  are checked explicitly where the two formats have frozen widths.
- Shapes, null patterns, populations, masks, IDs and order are exact. Persisted
  Float32 vectors are bitwise exact. Comparable Float64 state and diagnostics
  use `rtol=0, atol=1e-12`; reconstructed smooth values use only the approved
  `5e-8` absolute boundary. Every comparison records counts, maximum absolute
  and relative error, ULP evidence where widths agree, nonzero and
  above-tolerance counts, and the first failing key.

The tolerance policy is owned by
`DailyLegacyComparisonTolerances` at
`phase10-legacy-daily-comparison-tolerances-v1`. Boundary tests cover zero,
exactly at tolerance and one representable step beyond for Float32, Float64,
smooth reconstruction, and the same-runtime density/DLV inverse.

## SANOS, no-arbitrage and DLV semantics

The positive-smoothness SANOS state is reconstructed independently on each
side. Its strike grid is the union of fixed off-grid points, input strikes, and
expiry/surface supports from both states. Calls and parity-derived puts are
compared at zero, half one-minute, one-minute, a positive pre-first point, every
expiry node, every midpoint and a post-final point. ATM implied volatility is
evaluated on the same complete time grid.

The comparator does **not** call `dlvsqs()` on the positive-smoothness state and
does not manufacture a linear pricer from smooth densities. This matches the
amended contract: positive smooth time value at a zero-mass support makes that
operation undefined. A regression proves that no smooth-DLV metric is emitted.

The separate initial compatibility state remains the unsmoothed DLV state.
Before comparison, `validate_fitted_partition` reconstructs and validates its
support geometry, density mass/first moment, call and calendar shape, quote
containment, solver terminal states, zero DLV boundaries, stored-versus-encoded
DLV, corrected density inverse and off-grid call round trip. The comparator
then independently checks stored versus recomputed DLV at `1e-12` and the
same-runtime corrected density inverse at `2e-14`. Because the legacy daily
HDF5 does not store this complete pre-filter state, it is correctly reported as
an item-level `oracle_gap`; it is never claimed as legacy numerical parity.

## Fixed pilot and classification gate

`--phase10-pilot-contract` enforces all of the following before a report can
pass:

- SPX, source OptionMetrics, exact interval 2024-12-23 through 2025-01-16,
  exactly the 16 frozen NYSE sessions, exact canonical daily/surface config,
  overnight fraction 0.2, retained AM options and the declared 2024-12-24 gap;
- pinned catalog file digest
  `9b68a9638f273271ff4dbde6365c75ac9fd749993f9284b42428682339d516b4`
  and collection digest
  `f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280`;
- 14 populated legacy files, the one 2025-01-08 empty marker, no 2024-12-24
  file, and catalog totals 448 expiries, 40,436 nodes, 38,022 calls, 38,022
  puts, 76,044 active sides and 186 active sides at 0.2 business DTE;
- a populated half-day with `is_half_day=true`, exact typed-empty reason
  `known_bad_day_all_tickers`, actual Raw 2025-01-09 expiry rows plus a positive
  irregular-removal count, and a retained fresh 0.2D AM node with positive
  activity while the pinned legacy record supplies the 186-book count.

Full field parity makes an unchanged 2025-01-16 fresh population equal the
legacy 186 books. A genuine revised-vendor population is retained as fresh
evidence and cannot be called a match.

The terminal classes are exactly `match`, `oracle_gap`, `policy_empty`,
`unexplained_discrepancy`, and `approved_discrepancy`. Only the predeclared,
populated 2024-12-24 absence is a nonblocking pilot oracle gap. Empty/populated
crosses, the wrong empty reason, a missing irregular case, a missing half-day
or AM node, and every unapproved structural/numerical difference block the
report. Approval records require sorted exact mismatch fields, fresh artifact
ID and payload digest, legacy digest, explanation, approver, date and durable
reference; stale, broadened, missing-file and out-of-range approvals fail
closed. Vendor revision is never inferred automatically.

## Provenance, leakage and operator behavior

- Each selected fitted artifact is integrity-validated against its exact
  planned identity and the full fitted semantic validator before comparison.
  The report binds fitted pointers, payload checksums, configuration and
  producer/schema versions, source generation, source revision, catalog and
  every selected legacy file checksum.
- Verbose Raw cache validation reconciles manifest, canonical query record and
  Parquet contents, then emits physical table, logical table, observation date,
  generation, query digest, row/empty count, persisted schema and payload
  digest. It emits neither SQL nor vendor rows.
- The 2025-01-07 Raw evidence adds only artifact/provenance identities, schema,
  required expiry and an aggregate row count. Comparison mismatches contain
  field names, keys and error summaries, never bids, asks or source rows.
- Report publication is immutable and atomic outside both the repository and
  oracle root. Catalog/file tamper, symlinks, unexpected output reuse and
  interrupted staging fail closed.
- Raw partial-failure tests cover both an ordinary exception and a
  `BaseException`; completed immutable partitions are reused and only misses
  are queried after restart. The copied-tree cache tamper test fails under
  `WRDS_OFF=1` before authorization or credential use.
- The runbook fixes one live request at two connections and one attempt,
  requires 68 fresh Raw misses and the three 2024-12-20 boundary partitions,
  then performs cache-only validation, offline restart, 16 fitted builds/reuse
  and comparison. Any authentication, permission, connection, unknown data,
  solver, representative-case or unexplained-parity failure stops the pilot
  and Phase 11.

## Independent offline evidence

Commands run with `WRDS_OFF=1` and no network access:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_reference_compare.py tests/test_wrds_download.py
  -> 55 passed in 37.99s

python -m pytest -q -p no:cacheprovider \
  tests/test_reference_legacy_market.py tests/test_reference_inventory.py \
  tests/test_widb_calendar.py
  -> 19 passed in 6.25s

ruff check src/sanos/reference/compare.py tests/test_reference_compare.py \
  src/sanos/wrds/download.py tests/test_wrds_download.py
  -> All checks passed

python -m compileall -q src tests
git --git-dir=.git-real diff --check
  -> passed
```

The exact CLI Raw dry-run under `/tmp/sanos/phase10-quant-impl` returned
`planned=68`, `missing=68`, `reused=0`, `boundary_partitions=3`, with the three
roles on 2024-12-20 and `network_accessed=false`. Independent adversarial
probes confirmed that huge/NaN v1 tolerances and changed daily/surface configs
are rejected. The settled focused gate also covers an expiry retained by the
initial/selected population but dropped from the smooth state, a selected
expiry with zero active books, and an expiry made wholly ineligible before
selection; their population labels and stage counts remain distinct.

The previously reported complete offline suite result was **256 passed** in
324.06 seconds. The focused suites above were rerun after the review-time
remediations.

`gpuctl resources --json` currently reports all locally measured RAM/disk
checks `OK`, but also reports **“daemon not reachable; measuring locally.”**
That is not a healthy daemon result and therefore blocks the live pilot. No
WRDS connection, credential read, 2FA request, GPU allocation, production
artifact mutation, or `quant_data` write occurred during this review.

## Residual live risks

The first fresh WRDS extraction may legitimately expose vendor revisions,
authentication/account behavior or a representative case absent from the
current vendor snapshot. Those are precisely the reasons for the bounded
pilot. They remain stop-and-review conditions and are not waived by this
offline PASS.

Signed: `quantagent` — independent Phase 10 quantitative/data implementation reviewer
