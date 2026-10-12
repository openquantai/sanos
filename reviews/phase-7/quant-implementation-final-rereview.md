# Phase 7 final quantitative/data implementation re-review

**Verdict: PASS**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the settled,
uncommitted Phase 7 worktree  
Network/WRDS access: none (`WRDS_OFF=1`)

This is an append-only final re-review of the current shared tree after the
findings in the preceding Phase 7 reviews. I inspected the implementation,
tests, evidence, data contracts, compatibility decisions and the pinned
migration-oracle semantics. I modified no implementation or reference file.

Phase 7 meets its offline quantitative/data acceptance gate. This PASS approves
the internal Dataset assembly/publication implementation for commit. It does
not approve WRDS access, an as-yet-unbuilt production Dataset identity, or
canonical promotion.

## Finding closure

1. **Exact filter and fitter populations — resolved.**
   `validate_fitted_partition` recomputes the complete `OptionsFilter`
   projection from the no-arbitrage frame and canonical configuration, then
   compares every added mask, reason, eligibility and diagnostic field exactly.
   The initial and smooth fitter masks are tied exactly to the interior nodes
   of their retained expiry supports. Initial/smooth reasons, model masks,
   call/put activity and all three stage diagnostics are reconstructed rather
   than accepted as mutually self-consistent claims. The adversarial tests now
   reject the previously accepted self-consistent horizon/NMS, initial-mask and
   smooth-mask changes.

2. **Numeric completeness and Black vectors — resolved.**
   Every present scalar/list floating field is rejected when it contains NaN or
   positive/negative infinity. This closes the comparison-with-NaN bypass found
   during this re-review. Enrichment and fitted Black implied volatilities and
   Greeks are recomputed with the Phase 5 production epsilon and Float32 cast
   conventions. Mutations of bid Greeks, expiry/surface fitted IV/Greeks and
   smooth summaries are rejected. The exhaustive finiteness test covers every
   present Float32, Float64, List(Float32) and List(Float64) field in
   `FITTED_SCHEMA`.

3. **SANOS/DLV state and no-arbitrage — resolved.**
   Both expiry-fit and common-surface supports, densities, variances and
   x-to-surface mappings are reconstructed with strict ordering, finite state,
   density non-negativity tolerance, unit mass and martingale first moment.
   Stored expiry and surface prices, smooth fit errors/summaries, quote
   containment, strike convexity, calendar monotonicity and model-free bounds
   are rechecked. The corrected discrete-DLV inverse is checked through stored
   squared-DLV reconstruction, density round trip and off-grid call parity at
   `1e-12`.

4. **Duplicate semantics — resolved without changing CD-017 pairing.**
   The paired-day stage continues to preserve the reference Cartesian
   population for duplicate option keys. The pinned legacy and SANOS
   no-arbitrage fitters both require strictly increasing per-expiry strike
   supports, so an exact duplicate `(expiry_id, normalized_strike)` in a
   completed fitted frame is invalid and is now rejected rather than sorted or
   deduplicated. Valid repeated coarser expiry geometry remains byte-copied in
   its original order and multiplicity. Tests exercise both contracts.

5. **Clock, sessions and outcome classes — resolved.**
   Calendar DTE, AM/PM settlement, configured AM business-day subtraction,
   Float32 business DTE, `T=business_dte/252`, expiry ID, trading-session
   half-day flag and Float32 close return are independently reconstructed. The
   checked profile freezes 918 included sessions, 911 sessions with half days
   excluded, the exact seven half days, one policy-owned 2025-01-08 typed-empty
   date, and 3,674 unique Raw identities. The fresh default policy therefore
   expects 917 populated plus one typed-empty partition; a selected-zero day is
   correctly retained as populated audit data.

6. **Lineage and immutable evidence — resolved.**
   Shallow validation remains self-contained and output-first. Deep validation
   reconciles the Raw catalog, date/role mapping, flattened Raw identities,
   every paired/enriched/no-arbitrage/filtered/fitted ancestry edge, fitted
   source and manifest checksums, copied partition checksums and all manifest
   evidence. Re-signed tests reject missing/duplicate/reordered dates, altered
   schema/validation reports, false stage ancestry, fitted manifest and source
   digests, flattened fitted and Raw identities, and Raw catalog date/role
   changes. Concurrent identical publication, interruption recovery, wildcard
   reuse and deletion of disposable upstream artifacts are covered.

7. **Version identity — resolved.**
   The Dataset validator explicitly owns its schema, evidence, hash, bad-policy,
   partition and numerical-validation helpers. All daily and Dataset producer
   identities agree with a clean subprocess in the full settled test run.

8. **Canonical-reference semantics — resolved.**
   Phase 7 publishes only an internal, unpromoted candidate with
   `canonical_promotion_ready=false`. The fresh SANOS Dataset becomes the
   authoritative reference only after the guarded production build, numerical
   validation, comparison review and explicit immutable promotion. The old
   891-populated-date `quant_data` collection remains a migration/parity oracle
   only, never a build input, runtime dependency or co-equal canonical source.
   Its distinct 4,651,416 smooth-cache active books, 3,566,258 daily-front
   DySANOS books and 2,192,458 Daily25 SDLV books remain correctly separated.

## Independent offline evidence

```text
env PYTHONPATH=src WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/test_widb_surface.py::test_dataset_quantitative_boundary_rejects_state_config_and_clock_corruption \
  tests/test_widb_surface.py::test_dataset_finiteness_gate_covers_every_present_float_field \
  tests/test_widb_build.py::test_internal_dataset_publication_is_self_contained_and_output_first \
  tests/test_widb_build.py::test_internal_dataset_rejects_resigned_catalog_lineage_and_evidence_mutations \
  tests/test_widb_build.py::test_reference_profile_freezes_full_calendar_and_policy_structure \
  tests/test_widb_build.py::test_all_stage_and_dataset_versions_are_deterministic_in_fresh_process
6 passed in 33.27s

env PYTHONPATH=src WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider
188 passed in 162.82s

ruff check src tests scripts
All checks passed!

python -m compileall -q src tests scripts
passed

git --git-dir=.git-real diff --check
passed
```

Independent probes additionally confirmed that the formerly accepted NaN
mutations in `pure_bid_delta`, `pure_fitX_gamma` and `smooth_pcnt_fit` now raise
`DatasetValidationError`. No command contacted WRDS or the network.

## Residual production risks and later gates

- No fresh WRDS data were downloaded in Phase 7. The 918/917/1 population is a
  deterministic calendar and bad-data-policy expectation, not yet an observed
  production count. Any missing, extra, newly malformed or solver-failed day
  must block publication and be reviewed; it must not become an implicit
  exclusion.
- The full date-by-date numerical comparison with the migration oracle and the
  immutable canonical-promotion evidence remain guarded production work. The
  comparison implementation must be complete under a new reviewed identity
  before promotion; the structural Phase 7 report is deliberately insufficient.
- CD-017 duplicate vendor option keys remain preserved at pairing. Such a
  malformed day may fail the strictly increasing SANOS support requirement,
  matching current behavior. Any future deduplication, tie-break or recovery is
  a versioned semantic change requiring approval and measured impact.
- Live OptionMetrics edge cases may expose solver timeout/fallback or numerical
  tolerance behavior not represented by the compact offline fixtures. Existing
  structured diagnostics must remain blocking evidence rather than being
  converted to empty days or silent repairs.
- Before any WRDS pilot or production run, the user must confirm immediate 2FA
  readiness and the account-wide connection/process guard must remain at four
  or fewer.

Signed: `quantagent` — independent Phase 7 quantitative/data reviewer
