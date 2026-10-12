# Phase 2 independent quantitative/data review

**Verdict: CHANGES_REQUESTED**

Reviewer: `quantagent`
Review date: 2026-10-12

This review covers the Phase 2 configuration, artifact identity/store, CLI
foundation, and daily local runner. It does not approve WRDS access, a WIDB
transformation, dataset publication, public redistribution, or an experiment
materialization. All checks were offline; `quant_data` was inspected only as
the already pinned read-only reference and was not modified.

## Findings

### 1. High — a completed `Dataset` artifact does not satisfy the binding dataset-manifest contract

`ArtifactBuilder.commit()` writes the same generic manifest for all four
storage classes. `ArtifactStore.validate()` requires that exact generic field
set. Consequently a `StorageClass.DATASET` artifact can be committed and
validated with no immutable distribution status. The ordinary artifact tests
demonstrate this path because their default `_spec()` is a Dataset and
`_publish()` finalizes it.

An independent probe published and deep/shallow validated such a Dataset; its
manifest had neither a top-level `distribution` field nor a distribution value
in metadata. This conflicts with the repository rule that every published
dataset manifest declares exactly one of `internal`, `public_candidate`, or
`public`.

Putting distribution in the current free-form `metadata` is not sufficient:
metadata is excluded from `ArtifactSpec.artifact_id` and `compatibility_id`.
The same spec requested once as internal and once as public would therefore
have the same identity and collide, whereas a public release must be a separate
materialization with its own identity. A `public` manifest also needs the
conditional license, rights basis, approver, approval date, and durable terms
reference.

Required fix: either

1. prohibit generic Dataset finalization until the typed Phase 7/8 publisher is
   present, clearly distinguishing a low-level staged artifact from a published
   Dataset; or
2. make dataset distribution a typed, validated identity input now, enforce the
   conditional `public` approval fields, and retain room for the required
   dataset README/config/count/exclusion/capability contract.

Add tests proving that distribution is mandatory, changes dataset identity,
that `public_candidate` does not imply redistribution rights, and that `public`
cannot validate without the required approval/license record. This does not
require implementing the public allowlist or leakage analysis in Phase 2.

### 2. High — lifecycle class is currently only a label; invalid lineage validates deeply

The store accepts any `ArtifactPointer` as upstream of any storage class. An
independent probe created an Experiment and then a Raw artifact pointing to the
Experiment; `validate(..., deep=True)` succeeded. Conversely an Experiment with
no Dataset input is accepted. Those graphs contradict the declared meanings:
Raw is a minimally transformed vendor extract independent of downstream work,
while an Experiment is a disposable materialization derived from a Dataset.

Required fix: define and enforce lifecycle lineage policy at the finalization
boundary, or prohibit finalization for classes whose typed publisher has not
yet landed. The policy must leave room for the intended graph: Raw collections
may aggregate Raw partitions; Intermediate derives from Raw/Intermediate; an
internal Dataset derives from the daily pipeline; the deliberately lossy public
Dataset may derive from an internal Dataset; and an Experiment must identify
its source Dataset. Dataset or Experiment state must never flow backward into
Raw or Intermediate. Add positive and negative lineage tests, including the
internal-Dataset to public-Dataset edge.

### 3. High — immutable identity has no explicit raw-source generation for `--refresh`

For deterministic stages, configuration + schema + producer + upstream
identity is sufficient. A WRDS extract is different: the same table/query/date
configuration can return corrected vendor rows at a later download. In the
current model, retrieval time and metadata do not enter identity and Raw has no
upstream pointer. Re-running a future explicit `--refresh` with the same spec
would therefore address the old directory and raise `ArtifactExistsError`, even
if WRDS returned different bytes.

Required fix: freeze the raw acquisition identity contract before Phase 4 uses
this store. It may be an explicit source snapshot/generation identity, a
content-derived identity incorporated at finalization, or another reviewed
mechanism, but it must:

- give changed vendor content a new immutable Raw identity;
- make an exact retry of the same acquisition idempotent;
- retain table/query/date/security provenance without credentials; and
- let downstream identities distinguish corrected source extracts.

Add a synthetic test in which the logical query is unchanged but the source
generation/content changes. The two Raw artifacts must not collide. Document
how `--refresh` selects or creates that identity.

### 4. Medium — the daily journal does not identify the plan/configuration it records

`DailyWorkItem.payload_json` carries canonical stage configuration, but
`run_started` records only the first item's stage, total, worker count, and
progress interval. Work keys are only `ticker/date`. Neither the journal nor
`DailyRunSummary` records a payload/configuration ID or plan ID, and a caller
may append a new run with a different payload to the same journal.

An independent probe built two otherwise identical SPX/date/stage plans with
smoothness `0.25` and `0.5`. Their `run_started` records were indistinguishable
apart from run ID/timestamp. That is inadequate provenance for diagnosing or
restarting the 891-day build, especially once
`overnight_variance_fraction`, filters, schema versions, and producer versions
affect outputs.

Required fix: give the ordered plan a canonical identity covering at least
stage, ticker/date units, payload/configuration identity, schema, and concrete
producer version; record it in `run_started`, the summary, and the completion
record. Either enforce one stage/configuration per `DailyPlan` or record those
identities per item. Reused units should record the expected artifact pointer,
not only a boolean/key, so an audit can tie reuse to immutable output.

### 5. Medium — CLI phase guidance contradicts the reviewed program plan

The not-implemented errors route Raw and Intermediate to Phase 3, Dataset to
Phase 4, and Experiment to Phase 5. The actual plan assigns the numerical core
to Phase 3, guarded WRDS Raw to Phase 4, daily WIDB/enrichment and fitting to
Phases 5/6, internal Dataset publication to Phase 7, and Experiment
materialization to Phase 9. The current messages are particularly misleading
for WRDS readiness and 2FA expectations.

Required fix: align the CLI messages and tests with the actual phase gates (or
avoid embedding phase numbers and name the required capability/gate).

## Passing quantitative/data aspects

- The implementation uses exactly the four canonical storage-class names. The
  legacy `processing` and `training` names appear only as location fallbacks;
  no fifth `derived` class is introduced.
- Location precedence, canonical/legacy aliases, conflict warning covered by
  tests, and non-mutating diagnostics match the planned foundation. No path or
  credential is included implicitly in artifact identity.
- Canonical configuration JSON is stable and rejects non-finite floats, naive
  datetimes, non-string keys, and unsupported values.
- For deterministic artifacts, the identity correctly includes storage class,
  namespace, canonical configuration identity, schema version, concrete
  producer version, and normalized upstream pointers.
- Concrete output-first reuse and `version="*"` have the required behavior:
  wildcard omits only producer version, validates the existing complete
  artifact, preserves configuration/schema/upstream/integrity checks, warns,
  returns a structured override record, does not mutate the manifest, and
  fails rather than building on a miss.
- Shallow Dataset/Experiment-style validation remains valid after upstream
  deletion; explicit deep validation traverses lineage. Atomic staging,
  completion-last publication, checksums, and symlink rejection are exercised.
- The runner keeps daily work units small, sorted and process-isolated, bounds
  CPU workers, uses spawn, and caps progress intervals at five minutes. It does
  not conflate CPU worker count with the future WRDS connection limit.
- No Phase 2 code contacts WRDS, reads the legacy artifacts as build inputs,
  hard-codes an external market-data source, or claims that normalization grants
  redistribution rights. The 891-date DySANOS parity oracle remains unchanged.
- Deferring the actual `overnight_variance_fraction` clock to the WIDB calendar
  phase is acceptable because Phase 2 produces no WIDB artifact. When those
  specs land, the typed value must default to `0.2`, enter canonical identity,
  and propagate through all clock-sensitive stages; it must not be relegated to
  free-form metadata or a display-only field.

## Offline verification

Run from `/home/hans/python/sanos` with `WRDS_OFF=1`:

- `python -m pytest -q -p no:cacheprovider` — **54 passed**.
- `ruff check --no-cache src tests` — **passed**.
- `python -m compileall -q src tests` with bytecode redirected below
  `/tmp/sanos/phase2-quant-review/` — **passed**.
- `git --git-dir=.git-real --work-tree=. diff --check` — **passed** before this
  review file was added.
- Adversarial artifact probe — confirmed that Raw depending on Experiment
  passes deep validation and that a finalized Dataset has no distribution
  status.
- Adversarial journal probe — confirmed that different canonical payloads
  produce indistinguishable `run_started` provenance fields.

## Gate condition

Phase 2 is not approved until findings 1–5 are resolved and re-reviewed. The
fixes should remain infrastructure-only: they must not introduce WIDB/SANOS
numerical behavior, authorize a public release, resolve an open compatibility
decision, or access WRDS.

## Re-review — 2026-10-12

**Verdict: PASS**

The remediation resolves all five first-review findings for the Phase 2
infrastructure scope. This verdict still does not approve WRDS access, any
WIDB/SANOS quantitative behavior, dataset publication, public redistribution,
or an open compatibility decision.

1. **Resolved — typed Dataset distribution and projection identity.**
   `DatasetDistribution` admits exactly `internal`, `public_candidate`, and
   `public`; non-public statuses reject approval fields, while `public`
   requires a non-empty license, rights basis, approver, ISO approval date, and
   durable approval reference. The full and compatibility identity payloads
   include the complete distribution object. A non-internal Dataset is
   rejected unless it has a Dataset upstream, and the positive lifecycle test
   constructs the intended completed-internal-to-public-candidate edge. The
   Phase 8 publisher must still resolve that pointer and enforce the completed
   internal source manifest, leakage checks, allowlist, capabilities, and
   approval authority; Phase 2 does not claim to implement that publisher.
2. **Resolved — lifecycle edges.** `_validate_lineage()` permits only Raw from
   Raw, Intermediate from Raw/Intermediate, Dataset from
   Raw/Intermediate/Dataset, and Experiment from Dataset. Intermediate,
   Dataset, and Experiment are source-required. Construction and manifest
   reconstruction apply the same checks, and tests reject the previously
   accepted backward and source-free graphs.
3. **Resolved — Raw acquisition generation.** Raw requires a non-empty
   `source_generation`; it is in the full and wildcard compatibility identity
   and manifest. The documented contract creates a new generation before an
   explicit refresh contacts the source, exposes it in Phase 4 dry-run/results,
   and uses the same generation for an exact retry. Synthetic tests distinguish
   refreshed generations while preserving exact-retry identity.
4. **Resolved — plan/config/schema/producer identity and pointers.** The plan
   ID hashes every ordered daily work identity, including ticker/date, stage,
   canonical payload, schema, and concrete producer version. Journals reject a
   different plan, and run-start/run-complete records carry its identity.
   Reused and completed unit records plus `DailyRunSummary.results` retain the
   bounded immutable `ArtifactPointer` returned for each unit.
5. **Resolved — CLI phase guidance.** Placeholder errors now route Raw to
   Phase 4, Intermediate to Phases 5–6, Dataset build/validation to Phase 7,
   and Experiment to Phase 9, matching `PROGRAM_PLAN.md` and preserving the
   guarded WRDS/2FA boundary.

Additional adversarial checks confirmed that `version="*"` misses when only
`source_generation` changes and also misses when only the Dataset distribution
contract changes. Thus the wildcard omits producer version only; it does not
erase either new class-specific identity field. The implementation remains
infrastructure-only: no OptionMetrics/WRDS source assumption, WIDB transform,
SANOS/DLV calculation, no-arbitrage rule, normalization, clock convention, or
redistribution claim was introduced. In particular, the required `0.2`
`overnight_variance_fraction` remains correctly deferred to the later typed
WIDB configuration.

Offline verification from `/home/hans/python/sanos` with `WRDS_OFF=1`:

- `python -m pytest -q -p no:cacheprovider` — **70 passed**.
- `ruff check --no-cache src tests` — **passed**.
- `python -m compileall -q src tests`, with bytecode redirected below
  `/tmp/sanos/phase2-quant-rereview/` — **passed**.
- `git --git-dir=.git-real --work-tree=. diff --check` — **passed** before this
  review append.
- Adversarial artifact probe — **passed** for Raw-generation and Dataset-
  distribution wildcard separation; confirmed that a non-internal Dataset
  carries a Dataset upstream and that deep validation traverses the chain.

Residual Phase 8 risk: an `ArtifactPointer` intentionally contains only class,
namespace, and identity, not the pointed Dataset's distribution status. The
public publisher must therefore load and validate the direct source manifest
as `internal` when it materializes the public projection; generic Phase 2 graph
typing alone is not public-release authorization.

## Second re-review — 2026-10-12

**Reviewer:** quantitative/data agent (`/root/quantagent`)
**Final verdict: CHANGES_REQUESTED**

The second remediation closes the earlier provenance-existence and
public-on-public gaps, preserves output-first reuse, and does not alter any
quantitative transformation. One narrow but binding public-projection edge
remains.

### High — a non-internal Dataset may still have additional Raw or Intermediate inputs

`ArtifactSpec.__post_init__()` at `src/sanos/artifacts/store.py:312-322`
requires exactly one upstream whose class is Dataset, but it does not require
that Dataset to be the complete upstream tuple. The publication check at
`src/sanos/artifacts/store.py:583-612` and the corresponding deep check at
`src/sanos/artifacts/store.py:1161-1184` repeat the same filtered-count rule.
Consequently a `public_candidate` or `public` artifact may be built from one
validated internal Dataset **plus** direct Raw and/or Intermediate artifacts.

An independent offline probe published and deep-validated:

```text
Raw-1 -> internal Dataset -> public_candidate
Raw-2 ----------------------^
```

The result printed `public_projection_with_extra_raw_published True`. This
violates the binding contract that the deliberately lossy public projection is
materialized only from the completed internal Dataset. Besides weakening the
release boundary, it leaves open a direct source-data path that the Phase 8
allowlist is intended to exclude.

Required fix: for `public_candidate` and `public`, require the entire direct
upstream tuple to contain exactly one pointer, of class Dataset, and continue
requiring its shallow-validated manifest distribution to be `internal` at
publication. Apply the same invariant during deep validation. Add negative
tests for `(internal Dataset, Raw)` and `(internal Dataset, Intermediate)` for
both non-internal statuses, alongside the existing public-on-public tests.

### Passing re-review checks

- Every declared upstream is now shallow-validated immediately before atomic
  publication. A missing or corrupt upstream therefore cannot be recorded as
  valid provenance. The completed internal source status is read from the
  validated source manifest, rather than trusted from the pointer label.
- Existing published artifacts retain the required output-first behavior.
  Shallow validation succeeds after upstream deletion, deep validation fails,
  and `version="*"` can reuse a complete matching artifact without traversing
  deleted lineage. The independent probe confirmed both shallow validation and
  wildcard reuse after removing the projection inputs.
- Raw `source_generation` and Dataset distribution/approval remain in exact
  and wildcard compatibility identity. Refresh generations remain distinct,
  and wildcard reuse still omits only producer version.
- The runner changes are execution-safety changes only. Bounded cancellation,
  termination, and journal-directory durability do not change daily item
  identity, canonical stage configuration, schema/producer identity, result
  pointers, work ordering, or any SANOS/WIDB numerical value. The active-worker
  restart test confirms no old worker performs a delayed post-return side
  effect.
- No WRDS access, OptionMetrics assumption, market normalization, clock rule,
  no-arbitrage behavior, SANOS/DLV calculation, public approval, or
  redistribution decision was introduced. The `0.2` overnight-variance default
  remains deferred to its reviewed WIDB phase.

Offline checks run with `WRDS_OFF=1`:

- `python -m pytest -q -p no:cacheprovider` — **72 passed**.
- Four focused provenance/shallow-reuse/runner tests — **4 passed**.
- `ruff check --no-cache src tests scripts` — **passed**.
- `python -m compileall -q src tests scripts`, with bytecode redirected under
  `/tmp/sanos/phase2-quant-second-rereview/` — **passed**.
- `git --git-dir=.git-real --work-tree=. diff --check` — **passed** before this
  review append.

Phase 2 should pass quantitative/data-contract review once the non-internal
Dataset's direct upstream set is restricted to the sole validated internal
Dataset and the two mixed-source rejection tests pass.

## Final follow-up — 2026-10-12

**Reviewer:** quantitative/data agent (`/root/quantagent`)
**Final verdict: PASS**

The sole remaining finding from the second re-review is resolved.
`ArtifactSpec` now requires the entire upstream tuple of a
`public_candidate` or `public` Dataset to be exactly one Dataset pointer. The
publication boundary shallow-validates that sole artifact and requires its
manifest distribution to be `internal`; deep validation mirrors both the
cardinality/class invariant and the internal-status check. The focused test
rejects Raw and Intermediate mixed with an internal Dataset for both
non-internal statuses, while retaining the valid internal-to-public path and
rejecting candidate/public-on-public paths.

This stricter build-time provenance rule does not weaken output-first reuse:
an already complete self-contained Dataset is still shallow-valid after its
upstream is removed, and wildcard reuse does not traverse deleted lineage.
Raw generation, Dataset distribution/approval, schema, configuration,
producer, and upstream identity semantics are unchanged. No runner or journal
change in this follow-up affects work identity, ordering, or quantitative
outputs.

Offline verification with `WRDS_OFF=1`:

- `python -m pytest -q -p no:cacheprovider` — **72 passed**.
- Focused public-lineage, shallow-validation, and wildcard-reuse tests —
  **3 passed**.
- `ruff check --no-cache src tests scripts` — **passed**.
- `python -m compileall -q src tests scripts`, with bytecode under
  `/tmp/sanos/phase2-quant-final-followup/` — **passed**.
- `git --git-dir=.git-real --work-tree=. diff --check` — **passed** before this
  review append.

No WRDS connection was opened and `quant_data` was not accessed or modified.
Phase 2 now passes the quantitative/data-contract gate; this remains an
infrastructure approval only and does not approve later WIDB/SANOS semantics,
dataset publication, or public redistribution.
