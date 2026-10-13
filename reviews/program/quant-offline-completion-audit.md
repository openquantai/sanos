# SANOS quantitative/data offline completion-gap audit

**Reviewer:** `quantagent`  
**Date:** 2026-10-13  
**Baseline:** clean commit `ffecaed34af00adc1364cf5e334daeff290d6352`  
**Scope:** `AGENTS.md` and the complete `PROGRAM_PLAN.md`, with offline-only
inspection and probes; no WRDS/network access and no production-data mutation

## Verdict

- **Phase 10 live pilot:** **OFFLINE QUANT/DATA PREREQUISITES COMPLETE.** I found
  no additional numerical, schema, source-query, or comparison implementation
  required before the already documented 16-session pilot. Live access remains
  operationally blocked until the owner gives a new immediate 2FA/readiness
  confirmation and `gpuctl resources --json` reaches a healthy daemon with every
  check `OK`.
- **Phase 11 full production build:** **OFFLINE ACCEPTANCE WORK REMAINS.** Raw,
  daily, Dataset, public-candidate, and Experiment builders exist, but the full
  run must not be authorized until the three Priority 1 contracts below are
  implemented, tested, documented, and dual-reviewed. They are acceptance and
  authority gaps, not reasons to change the reviewed WIDB/SANOS mathematics.

## Priority 1 — required before the full-range production run

### 1. Freeze a fail-closed full-range migration comparison contract

`reference compare-daily` has a strong general comparator and a frozen Phase 10
pilot contract, but there is no equivalent Phase 11 contract. A general
`DailyComparisonRequest` accepts any date interval and any declared catalog gap
(`src/sanos/reference/compare.py:118-145`); only
`enforce_phase10_pilot` invokes an exact population/configuration contract. A
successful report therefore does not by itself prove that the complete target
range and all expected legacy dates were compared.

The production contract should be a named, `@version`-owned profile and should
reject anything other than:

- the exact checked configuration and its canonical identity;
- SPX, 2022-01-03 through 2025-08-29, half days included, fraction `0.2`, and
  the selected immutable source generation;
- **918** ordered NYSE sessions, **3,676** acquired Raw artifacts, and **3,674**
  unique Raw pointers consumed by the fitted daily lineage. The acquisition
  count is one security artifact plus four dated artifacts for each session
  plus three pre-start boundary artifacts; daily chains consume only the
  previous-spot boundary, not the boundary forward and zero artifacts;
- all **892** catalogued legacy files: **891** populated and the exact
  2025-01-08 policy-empty marker;
- exactly **26** catalog gaps, comprising seven half days and nineteen other
  valid fresh sessions; and
- no unexplained classification, missing fresh partition, undeclared gap,
  altered tolerance, or unused/broadened approval.

The exact gap set recoverable offline from the pinned plan and catalog is:

```text
2022-03-22  2022-08-11  2022-10-28  2022-11-25
2023-05-17  2023-06-02  2023-07-03  2023-07-06
2023-10-02  2023-10-06  2023-11-02  2023-11-24
2024-01-10  2024-01-26  2024-02-13  2024-02-28
2024-03-04  2024-07-03  2024-07-11  2024-07-18
2024-07-24  2024-08-01  2024-11-29  2024-12-24
2025-05-16  2025-07-03
```

The seven half days are 2022-11-25, 2023-07-03, 2023-11-24, 2024-07-03,
2024-11-29, 2024-12-24, and 2025-07-03. They are retained fresh dates, not
production exclusions.

Required offline tests should reject a subrange, one missing/extra gap, the
wrong empty reason, half-day exclusion, a changed daily/surface setting, an
unpinned catalog, an unexpected catalog file, and a report with fewer than 918
terminal classifications. Tests should also prove that approved discrepancies
remain evidence-bound and cannot convert a catalog gap or policy-empty day.

### 2. Bind comparison evidence to the published Dataset and define immutable promotion

The comparator currently validates fitted **Intermediate** artifacts and records
their identities and payload hashes, but it does not accept or record the
published internal Dataset pointer. The Dataset's immutable
`legacy-comparison.json` is always emitted as
`structural_only_pending_fresh_wrds_build_and_guarded_comparison`, and both the
document and manifest set `canonical_promotion_ready=false`
(`src/sanos/datasets/internal.py:2108-2118`, `3108-3119`). Validation explicitly
rejects any other value (`2574-2575`). There is consequently no implemented path
to satisfy the plan's requirement that the reviewed fresh Dataset identity
become the canonical SANOS reference (`PROGRAM_PLAN.md:29-36`, `539-541`).

Do not rewrite the immutable Dataset or reuse its identity with changed bytes.
Add a separate immutable promotion/evidence object (or equivalently strict
canonical-reference registry entry) which binds at least:

- exact Dataset pointer, manifest digest, configuration ID, source revision,
  and shallow/deep validation evidence;
- the full-range daily comparison report ID and digest;
- proof that every fresh fitted payload hash in that report equals the
  corresponding byte-copied Dataset partition/source checksum;
- the complete approval document digest, if any discrepancy was approved;
- code-review and quantitative-review identities/references; and
- an explicit `canonical_promotion_ready=true` outcome only when every gate is
  satisfied.

The Dataset remains canonical after this promotion; neither the legacy oracle
nor the disposable Experiment becomes co-equal. Add `dataset reference promote`
and/or `dataset validate --require-promoted` (exact spelling is not material),
with output-first, atomic, checksum, concurrency, tamper, stale-report,
wrong-Dataset, missing-review, unexplained-discrepancy, and wildcard tests.

### 3. Implement the actual common-period Experiment tensor parity gate

Phase 9 proves the selector and consumer contract on synthetic/hermetic inputs,
and the new HDF validator strongly proves internal self-consistency. It does not
compare a freshly materialized all-history HDF5 artifact with the actual
891-date migration oracle. `docs/experiments.md:79-87` records the legacy facts
but supplies no comparison command. The CLI exposes only Experiment build and
validate, while `PROGRAM_PLAN.md:29-33` requires every experiment tensor to be
compared.

Add a read-only, offline Experiment comparison/report command. On the 891 common
dates it should compare the unpadded valid prefixes and consumer materialization
for:

- date/expiry arrays, AM flags, quote-expiry and price-grid geometry;
- normalized strikes, call/put sides, price-location indices and masks;
- bid, ask, fitted price, liquidity and absolute spot-reference tensors;
- ATM/model volatility arrays, NMS bounds, market/training fingerprints; and
- per-day and aggregate counts, including the legacy **3,566,258** books,
  `Qmax=7,898`, `Pmax=3,949`, `Emax=36`, 729/162 diagnostic date split, and
  861 weekly pairs.

The report must distinguish the 26 valid fresh-only dates and the one typed-empty
date from common-period discrepancies. It must bind the exact source Dataset,
fresh Experiment, legacy endpoint/checksums, tensor ordering, dtypes, masks, and
named tolerances. Tests can use small synthetic dual artifacts plus the pinned
selector/consumer oracle; only the final values require fresh data.

## Priority 2 — finish offline before claiming all plan acceptance

### 4. Close the documented overnight-fraction sensitivity chain

Current tests correctly prove the `0.1/0.2/0.5` AM clock, PM invariance, NMS,
artifact identity, and propagation into both fitted model states; the focused
gate passed seven tests. However, the plan explicitly requires the sensitivity
to trace calendar constraints, DLV time increments, and Experiment horizons
(`PROGRAM_PLAN.md:384-390`). The checked report still says those stages are not
implemented (`reviews/phase-5/overnight-sensitivity.md:24-28`), which is now
stale.

Add one end-to-end synthetic regression that, for `0.1`, `0.2`, and `0.5`,
asserts the exact stored SANOS/DLV expiry-time increments and the downstream
Experiment expiry/horizon selection while the matched PM expiry remains
unchanged. Validate no-arbitrage/calendar invariants and report population/fit
diagnostics without assuming monotonic fit improvement. Update the sensitivity
report with the Phase 6/9 results. This is an acceptance-closure test, not a
request to alter the default or production output.

### 5. Freeze the Phase 11 operator/evidence runbook and source-generation choice

The generic WRDS runbook describes full dates, while the only fail-closed
evidence protocol is the 16-date Phase 10 runbook. Before the large extraction,
write a Phase 11 protocol which freezes the commands, roots, run ID, source
generation, expected plan counts, restart proofs, redaction, reviews, and stop
rules.

The checked Dataset config uses `source_generation="default-v1"`, and its
planned Dataset ID `2ea204...4585` is correctly pinned by the public and
Experiment configs. Raw `--refresh` deliberately rejects that default
generation (`src/sanos/wrds/download.py:530-531`). Therefore choose and document
one coherent policy before downloading:

1. use `default-v1` only in a proved-empty production generation/root and block
   unless dry-run reports all **3,676 acquired Raw artifacts** missing, including
   exactly three pre-start boundary artifacts; or
2. use a named production generation, then update and review the internal,
   public-candidate, and Experiment pointers/config identities together.

Do not mix the Phase 10 pilot generation with either policy, reuse a previous
default generation, or decide this after observing vendor values. The runbook
should require a new 2FA confirmation, at most two connections unless the pilot
justifies otherwise, one fresh probe, cache-only/restart validation, resource
checks, and separate dual reviews for the Raw download, daily comparison,
Dataset promotion, public candidate, and Experiment comparison.

## Already complete offline

The audit found no need to redesign or re-test these contracts before the
pilot:

- exact WIDB sources, fields/units, previous-session dependency closure, 2FA and
  account-wide four-slot limits, retry classification, immutable Raw partitions,
  cache-only behavior, and provenance documentation;
- daily pairing/enrichment/filter/no-arbitrage/SANOS ordering, Float32/Float64
  boundaries, DLV correction, smooth SANOS state, diagnostics, typed empties,
  and solver failure behavior;
- 918/911 session planning, seven half days, the exact 2025-01-08 typed-empty
  policy, one-frame-per-day schema, the 3,676-artifact acquisition closure and
  distinct 3,674-pointer fitted daily-lineage closure, internal Dataset lossless
  partitioning and shallow/deep validation;
- public-candidate allowlist, CD-004 return, CD-009 bridge, SANOS/DLV
  reconstruction, leakage/capability checks, typed-empty omission, and the
  separate human/legal rights gate; and
- deterministic daily-front selector, HDF5 materialization, padding/masks,
  fingerprints, consumer compatibility, output-first reuse, and source-clock
  identity.

No redistribution approval is needed to build or validate the technical
`public_candidate`; lack of rights correctly continues to block only a
`public` release.

## Work that inherently requires fresh WRDS data

The following cannot be converted into a truthful offline unit test:

1. authentication/2FA and actual OptionMetrics catalog/table availability;
2. live Raw schemas, dtypes, row counts, empty partitions, vendor revision and
   the representative irregular-expiry observation;
3. the 16-date pilot's observed 14 match / one policy-empty / one oracle-gap
   outcome and its two independent reviews;
4. all 918 fresh daily outcomes, actual solver completion/failure population,
   per-stage distributions and the numerical comparison of all 891 common dates;
5. actual fresh Experiment maxima/counts/tensors and public-candidate row/state
   distributions; and
6. the final immutable Dataset, public-candidate, Experiment and promotion
   identities/checksums.

Vendor revisions remain discrepancies to explain and approve; they are never a
reason to mutate fresh data to force equality. Unknown empty days, unexplained
numerical differences, failed invariants, missing representative cases, or a
failed resource/2FA gate stop the run.

## Independent offline evidence

- Clean baseline confirmed: `ffecaed34af00adc1364cf5e334daeff290d6352` with no
  pre-existing worktree changes.
- Full test collection under `WRDS_OFF=1`: **270 tests collected**.
- Focused overnight propagation gate: **7 passed, 34 deselected in 4.99s**.
- Exact offline plan/catalog evidence: the Raw acquisition planner has 3,676
  artifacts (one security, four per session, and three boundary partitions),
  while the internal fitted-lineage plan has 3,674 unique consumed Raw pointers
  because only the boundary spot enters daily processing. The Dataset ID is
  `2ea204...4585`; the pinned catalog has 892 records, one exact empty marker,
  and the 26 gaps listed above. Both checked downstream configs pin that same
  Dataset ID.
- `gpuctl resources --json` could not reach the daemon and used its local
  fallback. Locally measured RAM and disks were `OK`, but the unavailable daemon
  remains an operational blocker for material/live work under host policy.

No network, WRDS credential, GPU, live artifact, legacy data, or `quant_data`
write was accessed during this audit.

Signed: `quantagent` — independent quantitative/data-contract reviewer
