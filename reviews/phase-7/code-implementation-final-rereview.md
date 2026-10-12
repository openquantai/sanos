# Phase 7 final code implementation re-review

**Verdict: PASS**

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Baseline: `675d171685954089b49c566459eb570b019c9c8d` plus the current uncommitted
Phase 7 worktree  
Network/WRDS access: none (`WRDS_OFF=1`)

## Final finding disposition

No blocking or acceptance-critical code issue remains in the reviewed Phase 7
worktree.

1. **Non-finite numerical values — resolved.**
   `_validate_fitted_numeric_finiteness` examines every present `Float32`, `Float64`,
   `List(Float32)`, and `List(Float64)` value in the exact fitted schema before any
   residual comparison. It rejects NaN and either infinity in scalars and list
   elements. `validate_fitted_partition` is versioned at 3.1 and explicitly owns this
   helper. The exhaustive fixture test covers every present scalar/list float field;
   representative end-to-end mutations cover NaN, positive infinity, and negative
   infinity. I reran the previous independent probe: `pure_bid_delta`,
   `pure_fitX_gamma`, and `smooth_pcnt_fit` are now all rejected with their exact field
   named in the error.
2. **Row/order/multiplicity contract — resolved.**
   The fitted Dataset boundary now explicitly requires the SANOS model key
   `(expiry_id, normalized_strike)` to be unique and strictly ordered. The validator
   rejects a duplicated logical key, the integration test freezes uniqueness of that
   exact key, and repository documentation no longer promises unsupported duplicate
   model keys. Coarser session/expiry geometry remains repeated, and publication still
   byte-copies the valid fitted frame without sorting, grouping, re-encoding, or
   deduplication.
3. **Re-signed lineage/evidence coverage — resolved.**
   The adversarial matrix now includes both previously absent relations:
   `fitted_source_sha256` and flattened manifest `raw_artifacts`. It also retains tests
   for missing/duplicate/reordered dates, schema/report evidence, fitted manifest
   checksum, flattened fitted identities, false daily-stage ancestry, and altered Raw
   catalog date-role mapping. The tests rebuild valid manifests/checksums around the
   mutations, so they exercise semantic/deep validation rather than merely the outer
   artifact checksum.

## Confirmed Phase 7 properties

- Dataset planning and all daily/Dataset producer versions resolve deterministically
  in a fresh process. The validator version closure explicitly owns every material
  schema, parser, hash, policy, canonicalization, and quantitative helper.
- Publication validates fitted sources, byte-copies one complete Polars/Parquet frame
  per date, checks the copied bytes, and exposes the Dataset only through the atomic
  artifact writer. Mid-copy interruption leaves no visible Dataset and restart
  succeeds.
- Two simultaneous identical publishers converge on the same validated immutable
  winner and leave no staging directories.
- Manifest metadata and checksummed evidence agree on the exact fitted pointers,
  flattened Raw identities, scheduled exclusions, typed-empty policy, source/copy
  checksums, row populations, and one direct Raw-lineage-catalog upstream.
- Deep validation reconciles the Raw catalog's actual upstream/payload records with
  the flattened daily Raw closure and verifies every paired-to-fitted ancestry edge.
- Default loading and semantic validation are shallow and self-contained. They remain
  successful after disposable Raw/Intermediate/catalog trees are removed; explicit
  deep validation then fails clearly. `version="*"` remains warning-emitting,
  reuse-only, and unable to build or publish on a miss.
- Complete OptionsFilter output, no-arbitrage and smooth support masks, configuration
  identities, clock fields, Black IV/Greeks, state geometry, DLV inverse, fitted
  prices, diagnostics, counts, and finite numerical values are independently checked.
- The reference configuration is frozen at 918 included sessions, 911 sessions when
  half days are excluded, the exact seven half days, one policy-owned typed-empty
  date, and 3,674 unique Raw identities.
- Dataset CLI operations remain ordinary human/script-operable commands with help,
  dry-run, text/JSON output, progress, exact-ID/config selectors, meaningful errors,
  and no WRDS initialization.

## Canonical-reference semantics

The implementation and documentation consistently distinguish publication from
promotion. A generic Phase 7 artifact is an internal candidate and records
`canonical_promotion_ready=false`. The freshly downloaded SANOS Dataset becomes the
authoritative reference only after its production validation/comparison, exact-artifact
code and quantitative reviews, discrepancy resolution, and explicit promotion. The
old 891-date `quant_data` artifact is a read-only migration/parity oracle only; it is
not a build input, runtime dependency, or co-equal reference.

This PASS approves the Phase 7 implementation for commit. It does not approve an
as-yet-unbuilt live Dataset identity, production manifest digest, legacy comparison,
or canonical promotion.

## Commands and evidence

```text
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/test_widb_surface.py::test_dataset_quantitative_boundary_rejects_state_config_and_clock_corruption \
  tests/test_widb_surface.py::test_dataset_finiteness_gate_covers_every_present_float_field \
  tests/test_widb_build.py::test_internal_dataset_publication_is_self_contained_and_output_first \
  tests/test_widb_build.py::test_internal_dataset_rejects_resigned_catalog_lineage_and_evidence_mutations
# 4 passed in 29.44s

Independent in-memory regression probe:
# pure_bid_delta REJECTED pure_bid_delta contains a non-finite numeric value
# pure_fitX_gamma REJECTED pure_fitX_gamma contains a non-finite numeric value
# smooth_pcnt_fit REJECTED smooth_pcnt_fit contains a non-finite numeric value

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m pytest -q -p no:cacheprovider
# 188 passed in 155.01s

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 ruff check src tests scripts
# All checks passed!

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m compileall -q src tests scripts
# passed

git --git-dir=.git-real --work-tree=. diff --check
# passed

env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  python -m mypy src tests scripts
# unavailable in the active interpreter: No module named mypy
```

All executed checks were offline. No WRDS/network access, credential access, or
`quant_data` mutation occurred.

Signed: `code` — independent Phase 7 code reviewer
