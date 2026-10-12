# Phase 7 quantitative/data implementation review

**Verdict: CHANGES REQUIRED**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the current uncommitted
Phase 7 worktree  
Network/WRDS access: none

I reviewed the current worktree against `AGENTS.md`, `PROGRAM_PLAN.md` Phase 7,
the Phase 7 design audits, the frozen migration-oracle inventory, the Phase 6
fitted-day contract, and the current implementation/tests. I did not read or
write live WRDS data and did not modify implementation or reference files.

## Blocking findings

### 1. Critical — artifact identity is not deterministic in a fresh process

The resolved `cdxcore` dependency graph is order-dependent. In a clean Python
process,

```text
env PYTHONPATH=src WRDS_OFF=1 python -c \
  'from sanos.datasets.internal import internal_dataset_producer_version; \
   print(internal_dataset_producer_version())'
```

raises `RecursionError: ... recursive dependency on function '_empty'`.
The same defect now causes the full offline suite to fail: **6 failed, 172
passed**. Parent-process plans and spawned workers resolve different artifact
identities; for example one test planned fitted artifact `096024...` while the
worker published `2c3c44...`. Reuse, restart, wildcard matching and Dataset
publication then fail even though the numerical frame was produced.

This is an identity/data-lineage blocker, not only a test-order issue. Remove
the ambiguous `_empty` dependency collision and make every producer closure
resolve identically in a fresh interpreter and a spawned worker. Add clean
subprocess regressions for every daily producer and
`internal_dataset_producer_version()`, plus plan/worker identity equality.

### 2. Critical — fitted-partition validation does not validate the complete
quantitative state it claims to validate

An independent adversarial probe retained the exact `FITTED_SCHEMA` and found
that `validate_fitted_partition()` accepted each of the following corruptions:

- arbitrary daily `configuration_id`, `stage_version`, and
  `raw_options_artifact_id`;
- `surface_min_half_spread=0.123` instead of the configured value;
- every `pure_fit1_price` shifted by `0.01`;
- every `initial_expiry_model_density` replaced by zeros;
- every `initial_x_to_surface_indices` and
  `smooth_x_to_surface_indices` replaced by all-zero lists;
- `spot_return=999`; and
- `calendar_dte` or `business_dte` shifted by one day.

The last three state examples are especially material. The expiry-fit density
and x-to-surface maps are part of the complete SANOS/DLV state and interpreting
geometry, while `pure_fit1_price` is the stored unsmoothed expiry-fit output.
The mappings must satisfy
`surface_model_strikes[x_to_surface_indices] == expiry_model_strikes`; an
all-zero map cannot be accepted. Current reconstruction uses only the surface
supports/densities, so corrupt expiry states and maps are invisible. Initial
quote containment is also checked with the surface `pure_fitX_price`, not the
stored expiry-fit `pure_fit1_price` required by the no-arbitrage contract.

Before publication can be trusted, validation must at least:

1. match the full daily configuration ID, terminal fitted-stage version, Raw
   options identity, `surface_min_half_spread`, and smoothing fraction to the
   planned configuration/lineage;
2. validate repeated-list equality, null alignment, shape, mass/moment and
   interpreting mappings for both expiry-fit and surface states;
3. reconstruct the initial expiry fits and smooth expiry fits, compare their
   stored prices, and enforce the declared fit/quote tolerances;
4. validate initial and smooth rejection reasons and the stored smooth fit
   summary fields against their masks/diagnostics; and
5. recompute the Float32-first close return, calendar DTE, business DTE, AM/PM
   `overnight_variance_fraction` adjustment, and `T=business_dte/252` rather
   than only requiring these columns to be constant.

The existing surface density/DLV reconstruction, corrected inverse round trip,
strike shape, calendar monotonicity, selected-mask chain, direct-book activity,
unit bridge, smooth put/call parity and terminal solver checks are useful and
passed the unmodified two-expiry fixture. They do not close the omissions
above.

### 3. High — publication claims canonical reference status before the
production comparison/promotion gate

`_readme()` currently says that a freshly published object *is* the reference
Dataset, and `legacy-comparison.json` sets
`sanos_dataset_is_reference_after_publication=true`. The same artifact sets
`canonical_promotion_ready=false` and records only a structural comparison.
These statements conflict with `AGENTS.md` and Phase 11: the fresh Dataset
becomes authoritative only after the guarded full-range comparison, resolution
or approval of discrepancies, and code plus quantitative review of the actual
production evidence.

Phase 7 may publish a validated **internal candidate**. It must not declare it
canonical. Use one explicit pending-promotion state consistently in README,
manifest metadata and `legacy-comparison.json`. The comparison document should
also pin the known oracle artifact/catalog identity and keep these populations
distinct:

- smooth daily cache: 892 physical files, 891 populated dates, 26,732
  expiries, 2,414,341 strike nodes, 4,651,416 active side books, including
  8,090 at 0.2 business day;
- daily-front DySANOS: 891 dates and 3,566,258 books; and
- Daily25 SDLV: 891 dates and 2,192,458 books.

The latter two are downstream/Experiment oracles, not internal-Dataset row
counts. Full fresh numerical results correctly remain pending until the fresh
production run.

### 4. High — immutable lineage evidence and deep validation are incomplete

The reconciled direct-lineage design is sound: the Dataset's sole direct
upstream is the Raw-lineage-catalog, while date-keyed fitted chains live in
immutable evidence. The implementation does not yet enforce the complete
contract:

- manifest metadata contains only the catalog pointer and mapping digest, not
  the complete sorted Raw identities and exact fitted-day identities required
  by the Phase 7 audits and `PROGRAM_PLAN.md`;
- deep validation traverses each recorded fitted pointer, but does not prove
  that the recorded paired/enriched/no-arb/filtered pointers are the actual
  ancestry of that fitted artifact;
- it checks `fitted_manifest_sha256` only for 64-hex syntax and checks
  `fitted_source_sha256` only against the copied partition, not against the
  retained fitted source during deep validation;
- Raw role pointers are not required to have `StorageClass.RAW`, and the fitted
  pointer is not explicitly required to be Intermediate; and
- the Raw catalog's `raw-catalog.json` date/role/artifact records are not
  semantically reconciled with its manifest upstream tuple and the Dataset's
  lineage document.

Keep shallow validation self-contained and output-first. Under
`deep_lineage=True`, however, verify these recorded claims exactly and add
tamper/mismatched-chain tests. Missing Raw or Intermediate artifacts should
continue to make only deep validation fail.

### 5. High — the published schema is not a sufficient data contract

`schema.json` does record exact field names and Polars dtypes, but assigns every
field the same generic cardinality, records no nullability, and gives most
fields the unit text `see README and source field semantics`. The generated
README defines only a few headline normalizations. This is insufficient for
the required human/agent-readable immutable schema and for interpreting the
nested DLV/SANOS state after Intermediate cleanup.

Record exact cardinality (day/expiry/row/list), nullability/conditional
presence, units and normalization for every field, including business versus
calendar DTE, AM settlement, variances, density masses, squared-DLV/time,
support/mapping geometry, fitted prices, Greeks and diagnostic JSON. The
document must distinguish trading-session `is_half_day` from
`expiry_is_am`.

### 6. High — required Dataset-boundary integration/adversarial coverage is
missing

The tests cover one ordinary end-to-end day, direct validation of the known-bad
typed-empty frame, half-day planning, a natural zero-selected day, atomic
interruption, byte tampering, shallow reuse and missing-lineage failure. They do
not build one Dataset containing an ordinary day, a half-day and the policy
typed-empty day together, and therefore do not prove lossless ordering,
aggregate counts, previous-session boundaries and typed-empty catalog evidence
across those outcomes. Nor do they attempt the quantitative corruptions listed
in finding 2 or the lineage mismatches in finding 4.

Add a compact synthetic multi-day publication and corruption matrix. Keep the
checked-in full-range planning regression at exactly 918 included sessions,
911 with half days excluded, seven scheduled half days, and one policy-owned
typed-empty date. No test should turn the 19 non-half-day legacy gaps into
fresh exclusions.

### 7. Major — accepted vendor tickers and partition paths disagree

`InternalDatasetConfig` accepts a nonempty vendor ticker such as `^SPX`, in
line with the existing WIDB ticker contract, but the Dataset partition regex
allows only `[A-Za-z0-9._-]+`. Publication/validation therefore cannot round
trip a configuration the public config type accepts. Use the existing safe
path label while preserving the actual vendor ticker in configuration/frame
semantics, or reject unsupported values explicitly at configuration parsing.
This does not alter the target `SPX` population but is a real data-contract
failure.

## Confirmed correct or directionally correct

- The checked-in reference profile plans 918 ordered NYSE sessions from
  2022-01-03 through 2025-08-29, includes all seven half days, contains
  2025-01-08 exactly once, and closes over 3,674 unique Raw pointers
  (security 1, options 918, spot 919, forwards 918, zero curves 918).
- The implementation preserves the 0.2 overnight fraction in canonical daily
  configuration and changes Dataset identity when that configuration changes.
  Finding 2 concerns validation of the materialized clock columns, not planning.
- One daily fitted Parquet is byte-copied without row selection, padding or
  re-encoding. Partition order is explicit, Hive inference is disabled, and
  shallow load/validation is designed not to require Raw or Intermediate
  storage.
- Typed empty, selected-zero, full audit-row, selected-row and active-side-book
  populations are kept distinct. The known 2025-01-08 marker is not silently
  treated as a normal zero-selected day.
- The Dataset distribution is fixed to `internal`; no Phase 7 option exposes a
  public/public-candidate alias or makes a public leakage claim.
- The 891-date legacy collection is not a build/runtime input. The implementation
  correctly treats the fresh calendar as primary rather than copying its 26
  missing sessions.

These positives do not override the blockers. No fresh WRDS population or
numerical comparison was available in this offline review, and this review is
not approval to start WRDS access or promote a canonical Dataset.

## Commands/evidence

```text
env WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
6 failed, 172 passed in 56.58s

ruff check src tests
All checks passed!

python -m compileall -q src tests
passed

git --git-dir=.git-real diff --check
passed

env PYTHONPATH=src WRDS_OFF=1 python -c \
  'from sanos.datasets.internal import internal_dataset_producer_version; \
   print(internal_dataset_producer_version())'
RecursionError: recursive dependency on function '_empty'

env PYTHONPATH=src WRDS_OFF=1 python -m sanos --root \
  /tmp/sanos/phase7-quant-dry --dry-run --log-format json dataset build \
  --config configs/widb-spx-reference.json
planned_dates=918; network_accessed=false
```

The adversarial quantitative probes used only the current synthetic two-expiry
Phase 6 fixture and `WRDS_OFF=1`; every accepted mutation is enumerated in
finding 2.

## Re-review gate

Re-review requires: deterministic clean/spawned-process versions; a fully green
offline suite; rejection tests for all state/config/clock corruptions above;
deep-lineage reconciliation and manifest identities; complete schema units and
cardinalities; neutral pre-promotion wording; and the synthetic multi-outcome
Dataset test. The later live pilot and full production comparison remain Phase
10/11 gates and require an immediate user confirmation of WRDS 2FA readiness.
