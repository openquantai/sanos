# Phase 7 quantitative/data implementation re-review

**Verdict: CHANGES REQUIRED**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the current uncommitted
Phase 7 worktree  
Network/WRDS access: none (`WRDS_OFF=1`)

I independently re-reviewed the current tree against `AGENTS.md`,
`PROGRAM_PLAN.md` Phase 7, both Phase 7 design audits, the original quantitative
implementation review, the pinned migration-oracle inventory, and the Phase 5/6
daily and fitted-state contracts. I did not access WRDS/network services or
modify implementation/reference files.

The remediation is substantial and resolves most prior findings. The remaining
quantitative blocker is that core no-arbitrage/filter populations are still
accepted from self-consistent stored masks rather than re-derived from the
stored state and declared rules. Two evidence/version coverage findings also
remain.

## Remaining blocking findings

### 1. Critical — the curated/no-arbitrage masks are not independently
re-derived

`validate_fitted_partition()` checks mask monotonicity, downstream Boolean
relations, rejection-label shape, and agreement with the stored diagnostic
counts. Those are internal consistency checks, not verification that the masks
are the outputs of the configured algorithms. Independent probes showed that
the validator accepts each of these semantically changed fitted frames when the
self-reported JSON counts/reasons are edited consistently:

1. one true `mask_arb_free` row changed to false, with the initial rejection,
   smooth model mask, activity fields and diagnostic counts updated; the
   accepted initial population changed from 26 to 25 rows;
2. one true `smooth_fit_mask` and `smooth_model_mask` row changed to false,
   with its rejection reason and smooth diagnostic counts updated; the accepted
   smooth model population changed from 26 to 25 rows; and
3. on a valid zero-selected fixture, an otherwise eligible row was falsely
   rejected at the horizon/NMS stages while retaining the original later-stage
   rejection reason; after changing only the diagnostic counts, validation
   passed.

These fields define the nearly complete post-no-arbitrage/filter population and
the smooth SANOS estimation population. Changing them changes Dataset contents
and later public/Experiment populations, so accepting their own diagnostics as
the oracle is not sufficient.

Remediation should:

- derive `mask_arb_free` and `smooth_fit_mask` from the retained expiry support,
  stored expiry-fit prices, bid/ask containment and the exact fitter mask
  convention, then compare exactly;
- recompute the complete `OptionsFilter` projection from the no-arbitrage
  columns and canonical configuration, including filter order, stable global
  cap/tie behavior and first-rejection labels. Reusing
  `apply_options_filter(frame.select(NO_ARB_SCHEMA), config=...)` or a shared
  pure mask helper is preferable to duplicating the rules; and
- add adversarial tests which change each mask/reason plus its diagnostics and
  still require rejection.

There is a related, lower-level completeness gap: independent mutations of
`pure_bid_delta`, `fit1_volatility`, `pure_fit1_delta`, `fitX_volatility`, and
`pure_fitX_gamma` by `+0.25` were all accepted. The state prices themselves are
now reconstructed correctly, but stored Black IV/Greek vectors are not. Either
recompute these vectors at the Phase 5 declared epsilon/cast boundaries or
narrow the immutable validation claims explicitly. Given that the internal
Dataset advertises and documents these fields as fitted WIDB values, the
preferred acceptance fix is to validate them.

### 2. High — the Dataset validator version closure is incomplete

The `validate_internal_dataset` decorator declares only
`open_internal_dataset`, `validate_fitted_partition`, `aggregate_statistics`,
`_scheduled_exclusions`, and `_safe_partition_ticker_label`. The function also
directly uses content-affecting schema/evidence helpers including
`_schema_document`, `_legacy_comparison_document`, `_pointer_from_value`,
`_read_json_file`, `_partition_record_statistics`, `_sha256_file`, canonical
hashing and the bad-data policy.

Changes to those helpers can change accepted lineage/evidence semantics without
changing `validate_internal_dataset.version`. That version is persisted in
`validation.json` and participates in the Dataset producer closure. Complete
the explicit `@version` dependency graph and add the same kind of direct closure
regression already used for the daily stages.

### 3. High — required lineage/catalog and exact population regressions remain
incomplete

The static deep-lineage implementation now checks the correct relationships,
but no adversarial test re-signs a tiny test artifact and proves rejection of:

- changed Raw-catalog date/role records or flattened Raw identities;
- a recorded paired/enriched/no-arb/filtered/fitted chain which is not the
  actual artifact ancestry;
- changed fitted-manifest/source digest evidence; or
- missing, extra, duplicate or reordered Dataset dates.

The full-range CLI regression freezes 918 included sessions, but there is no
test freezing the complementary 911-session no-half-day profile, the exact
seven scheduled half-day exclusions, and the single policy-owned 2025-01-08
typed-empty date. These were explicit requirements in the original quantitative
re-review gate. Add compact offline tests; no WRDS data is needed.

## Prior quantitative finding disposition

1. **Resolved — deterministic identities.** All daily and Dataset producer
   versions resolve identically in a fresh subprocess. The `_empty` name
   collision is removed, and the current full suite no longer exhibits the
   parent/worker identity mismatch.
2. **Partially resolved — complete fitted state.** Configuration/stage/Raw IDs,
   smoothing/min-spread configuration, daily diagnostic constancy, clock
   fields, Float32 spot return, expiry and surface density geometry, repeated
   lists, x-to-surface mappings, DLV inverse, expiry/surface fitted prices,
   smooth fit errors and summary statistics are now reconstructed and checked.
   Every corruption demonstrated in the first review is rejected. Finding 1
   above records the remaining mask and IV/Greek gaps.
3. **Resolved — canonical-reference semantics.** Phase 7 artifacts consistently
   remain unpromoted internal candidates with
   `canonical_promotion_ready=false`. The fresh, validated and explicitly
   promoted SANOS Dataset becomes the authoritative reference. The old
   891-populated-date `quant_data` artifact remains only a migration/parity
   oracle and is neither a build input, runtime dependency nor co-equal
   reference.
4. **Implementation resolved; test gate remains — lineage.** Manifest metadata
   now records exact flattened Raw and fitted identities, exclusions and
   typed-empty evidence. Deep validation reconciles Raw catalog records,
   retained checksums and every stage ancestry. Finding 3 requests the missing
   adversarial coverage.
5. **Resolved — immutable schema semantics.** `schema.json` now records exact
   dtype, day/expiry/row cardinality, conditional presence/nullability, units
   and normalization for every field, and distinguishes session half days from
   AM/PM expiry settlement.
6. **Partially resolved — Dataset scenarios.** A multi-session publication now
   covers an ordinary session, a half day, previous-session linkage and the
   policy typed-empty date. Mid-copy interruption, historical wildcard reuse,
   selected-zero behavior and progress are tested. Exact full-range and
   lineage/catalog corruption coverage remains as finding 3.
7. **Resolved — ticker round trip.** A collision-resistant safe partition label
   is used while the actual OptionMetrics ticker is retained; `^SPX` publishes,
   loads and validates end to end.

## Quantitative/data checks that passed

- The checked-in reference configuration independently plans 918 sessions;
  excluding half days produces 911. Their exact seven-date difference is
  `2022-11-25`, `2023-07-03`, `2023-11-24`, `2024-07-03`, `2024-11-29`,
  `2024-12-24`, and `2025-07-03`. The 2025-01-08 policy date appears exactly
  once.
- The included profile closes over 3,674 unique Raw identities: security 1,
  options 918, spot 919, forwards 918, and zero curves 918.
- The configured `overnight_variance_fraction=0.2` propagates through the
  Float32 AM clock; calendar DTE, business DTE, `T=business_dte/252`, expiry ID
  and close return are now verified exactly.
- Both expiry-fit and surface supports/densities are checked for ordering,
  non-negativity tolerance, unit mass, martingale first moment, repeated-state
  consistency and interpreting mappings. The corrected DLV density/call
  inverse remains checked to `1e-12`; smooth reconstruction, quote containment,
  strike convexity and calendar monotonicity retain their declared tolerances.
- Each complete fitted-day Parquet is still copied byte-for-byte and ordered by
  the explicit partition catalog. Loading and shallow validation remain
  independent of disposable Raw/Intermediate artifacts.
- The migration-oracle report pins the inventory/catalog revisions and keeps
  the 4,651,416 smooth-cache active books, 3,566,258 daily-front DySANOS books,
  and 2,192,458 Daily25 SDLV books distinct.
- Distribution remains exactly `internal`; Phase 7 introduces no public alias,
  public leakage claim, HDF5 padding, Redun dependency or fifth storage class.

## Commands and independent evidence

```text
env PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
183 passed in 125.47s

env PYTHONPATH=src WRDS_OFF=1 python -c \
  'from sanos.datasets.internal import internal_dataset_producer_version; \
   print(internal_dataset_producer_version())'
passed in a fresh process

ruff check src tests scripts
All checks passed!

python -m compileall -q src tests scripts
passed

git --git-dir=.git-real diff --check
passed

python -m mypy src tests scripts
unavailable in the active interpreter: No module named mypy
```

Independent quantitative probes used only the current synthetic Phase 6
two-expiry and zero-selected fixtures under `WRDS_OFF=1`. They performed no
filesystem writes outside pytest/temp artifacts and no network access.

## Re-review gate

Re-review requires exact reconstruction/rejection tests for the three mask
classes above; resolution of the stored IV/Greek validation claim; complete
validator version ownership; the deep-lineage/date-catalog mutation tests; and
the exact 918/911/seven-half-day/one-typed-empty regression. The later guarded
pilot and production comparison remain Phase 10/11 work and require immediate
user confirmation of WRDS 2FA readiness. This review does not approve a live
Dataset identity or canonical promotion.

Signed: `quantagent` — independent Phase 7 quantitative/data reviewer
