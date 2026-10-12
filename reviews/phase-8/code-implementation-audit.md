# Phase 8 code implementation audit

**Reviewer:** independent code agent (`/root/code`)
**Date:** 2026-10-12
**Scope:** current uncommitted Phase 8 public projection, publication, CLI,
documentation, and tests; offline only
**Verdict:** **FAIL / CHANGES REQUIRED**

The normal synthetic build follows the approved CD-004 equation and CD-009
units, produces a strict recursive Polars schema, and correctly treats the
result as a non-releasable `public_candidate`. Current and wildcard cache hits
are output-first and work with the internal source absent. The artifact store's
generic checksum, no-follow, locking, fsync, and atomic-directory publication
machinery remains sound. No reviewed text infers redistribution rights from
normalization or technical validation.

Phase 8 nevertheless does not meet its reviewed acceptance contract. The
blocking findings below must be resolved before commit/final approval.

## Findings

### 1. Critical — the required immutable `public` release transition is absent

Only `plan_public_candidate` and `build_public_candidate` exist
(`src/sanos/datasets/public/publish.py:153-179,221-397`), and the CLI exposes
only `dataset public build` and `dataset public validate`
(`src/sanos/cli/main.py:985-1181`). There is no strict approval document,
license-byte/checksum handling, release planner/publisher, candidate-equality
evidence, or `dataset public approve` command.

This is not optional acceptance work: the reviewed design says Phase 8 is
complete only when approval creates a distinct immutable `public` Dataset
without mutating or aliasing the candidate
(`reviews/phase-8/code-design-audit.md:458-476`), while the approved lineage
rule requires both candidate and release to have the same internal Dataset as
their sole direct source. The implementation can and should implement this
transition without inventing an approval: the command must require the license
bytes and all rights fields, and simply remain unusable until a human supplies
them.

Required change: implement the reviewed approval/re-projection path, exact
candidate logical-content equality proof, license/evidence validation,
independent `public` identity/namespace, output-first reuse, atomic staging, and
the complete distribution-transition/CLI test matrix. Continue to make absence
of human legal evidence a hard failure.

### 2. Critical — validation cannot establish the approved source-bound values

The producer computes the non-first CD-004 value at
`project.py:507-531`, but the persisted evidence drops the interpolated value
and keeps only dates, session count, and a Boolean assertion
(`publish.py:310-319`). `_validate_coverage_evidence` validates that assertion,
not the return (`validate.py:293-345`), and `validate_public_frame` checks only
null placement and finiteness (`validate.py:376-393`). Similarly, an expiry's
CD-009 bridge is checked only for positive finiteness on read
(`validate.py:437-443`); the source conversion at `project.py:398-412` is not
re-established. Generic `deep_lineage=True` only traverses artifact checksums
(`artifacts/store.py:1152-1162`) and does not re-project the internal Dataset.

Independent adversarial probes built valid candidates, changed a non-first
`forward_adjusted_spot_return` to `123.0` and, separately, doubled
`discounted_forward_over_spot`, then recomputed ordinary file/manifest/
completion checksums. `validate_public_dataset` accepted both artifacts. A
coherently re-signed omission or book population can fail for no more than a
mutable summary count, not because it is proven against the source.

Required change: distinguish honestly between self-contained shallow
capability validation and source-fidelity validation. Before candidate or
release publication, and for an explicit source-bound/deep validation, re-run
the exact versioned projection from the sole internal source and compare the
complete ordered public frame plus exact planned/published/omitted coverage.
Persist non-private projection/equality evidence, reconcile it across manifest
metadata and `validation.json`, and add coherent re-sign tests for non-first
returns, bridges, spreads, book removal/addition/reassignment, model state,
expiry geometry, real/fabricated omissions, and reason/date changes. Shallow
validation may remain source-independent, but its documentation and result
must not claim proof of private-input formulas it cannot perform.

### 3. Critical — leakage enforcement is not artifact-wide and a fabricated
`public` artifact is accepted by the public reader

The exact recursive Parquet schema is useful, but
`validate_public_dataset` reads selected named files and never requires the exact
payload set (`validate.py:567-614`). The generic store deliberately accepts any
checksummed payload listed in the manifest (`artifacts/store.py:1086-1139`). A
re-signed candidate can therefore add, for example, a raw/vendor payload and
remain a structurally valid public Dataset. The public validator also does not
close manifest metadata, regenerate/compare `README.md`, or scan all payloads
and public-facing documents for forbidden identifiers, absolute levels,
machine paths, or credentials.

Moreover, `open_public_dataset` accepts both `public_candidate` and `public`
(`validate.py:534-552`), although it performs none of the mandatory release
license, rights basis, approval reference, candidate-equality, or license-byte
checks. The generic distribution record merely requires five non-empty strings.

Required change: enforce an exact distribution-specific payload allowlist and
closed metadata/document contracts; validate the generated README and every
release evidence/license file; add forbidden field/token/value/path checks at
the complete artifact boundary; and reject `public` in this reader until the
full release validator exists. Test extra checksummed files, nested/all-null
forbidden data, metadata/document leakage, malformed approval evidence, and
license checksum/content mismatches.

### 4. High — content-affecting `@version` dependencies are omitted

The resolved producer graph confirms `_book_payload`, `_state_payload`, and
`_validate_books` are opaque `1.0` leaves. Their decorators declare no
dependencies (`project.py:165-212`, `validate.py:235-241`) although they depend
directly on `SanosPricer.dlvsqs`, `_atm_volatility`, NMS conversion and the
daily-front selection/unit policies. `project_public_day` directly uses `_one`,
`FITTED_SCHEMA`, `public_polars_schema`, and `DAYS_PER_YEAR`, but omits them from
its dependency list (`project.py:325-389,443-448`). Consequently changes to
these content boundaries need not change `public_dataset_producer_version()`
or the artifact ID.

Required change: version every content helper/contract and declare exact direct
dependencies through producer and validator roots, including source schema and
schema-version ownership, NMS/calendar constants and adapters, state encoding,
selection, unit conversion, Parquet settings, generated documents, and release
validation. Add fresh-process exact direct-dependency and planned-ID regression
tests.

### 5. High — full semantic validation happens after immutable publication

Inside the writer, the build validates only the staged Parquet frame and its
summary (`publish.py:345-363`). The complete artifact validator (documents,
manifest metadata, capabilities, coverage, and reader contract) runs only
after the context manager commits the final immutable directory
(`publish.py:377-378`; `artifacts/store.py:830-857`). If that validation fails,
an invalid complete artifact remains under the exact identity and blocks a
clean retry.

Required change: perform the complete applicable staged validation—including
the exact payload allowlist, canonical documents, metadata, leakage checks,
and source-bound equality—before commit, then retain post-publication
validation as a defense in depth. Add injected failures for every staged file,
validation/fsync interruption and concurrent identical candidate/release
publishers, proving no invalid final artifact is exposed.

### 6. High — CD-004 uses fitted-state expiries, not the approved distinct
source-day expiry curve

CD-004 freezes the previous retained public date's **distinct expiry curve**
(`docs/compatibility-decisions.md:56-72`). `project_public_day` instead builds
`ordered_records` only from the union of expiry records carrying
`initial_no_arb_input_expiry_index` or `smooth_input_expiry_index`
(`project.py:360-375`) and passes that reduced set to `_carry_curve`
(`project.py:448`). Initial no-arbitrage and smooth fitting can drop expiries;
their indices are explicitly nullable and their dropped IDs are recorded
(`widb/surface.py:438-478,924-965`). A dropped model expiry can still be a valid
distinct internal forward/discount curve node, including a material bracket for
the next retained-date horizon.

Required change: derive the carry/discount curves from all exact distinct
positive source-day expiry nodes, independent of state-fit retention, with
consistent-duplicate checks. Add a non-flat multi-node regression in which a
non-model expiry brackets the return horizon. If the intended approved contract
was instead the model-state union, obtain an explicit CD-004 amendment before
changing the implementation contract.

### 7. High — required boundary, mutation, and lifecycle test coverage is not
present

`tests/test_public_dataset.py` covers the happy candidate build, isolation,
wildcard miss/hit, one simple carry calculation, a top-level extra column,
top-level infinity, scale/identifier invariance, one typed-empty day, half-day
separation, and first-return null corruption. It does not implement the
reviewed matrix at `code-design-audit.md:401-456`, notably:

- independent daily-front oracle tests for exact/inside/outside 0.2D, 0/70/
  over-70D, inclusive/outside NMS edges, independent sides, American books,
  inactive/unavailable/nonfinite fits, and stable multiplicity/order;
- exhaustive recursive null/NaN/infinity and coherent re-sign mutations;
- exact source/public model calls, parity, ATM IV and CD-014 time boundaries;
- source-bound coverage/return/unit tests, source-clock variants and deep
  re-projection;
- interruption/concurrent publication and release-transition tests; or
- complete human/JSON CLI success, usage-error, progress and release cases.

Two implementation boundaries also contradict the reviewed source contract:
`PublicProjectionConfig` rejects `overnight_variance_fraction=0.0` even though
the upstream clock allows `[0,1]` (`config.py:96-115`), and `_validate_books`
widens the approved inclusive NMS interval by an undocumented `2e-6`
(`validate.py:278-287`). It also does not reject duplicate exact book keys
(`validate.py:244-290`). Resolve these semantics and own them with identity-
sensitive boundary tests; retain `0.2` as the default.

## Evidence and commands

- `env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p
  no:cacheprovider tests/test_public_dataset.py tests/test_cli.py` — **30 passed**
  in 44.17s; one expected wildcard warning.
- `env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p
  no:cacheprovider` — **210 passed** in 199.96s; the same expected wildcard
  warning was the sole warning.
- `ruff check src/sanos/datasets/public src/sanos/cli/main.py
  tests/test_public_dataset.py tests/test_cli.py` — passed.
- `python -m compileall -q src/sanos/datasets/public src/sanos/cli/main.py
  tests/test_public_dataset.py tests/test_cli.py` — passed.
- `git --git-dir=.git-real --work-tree=. diff --check` — passed.
- `python -m mypy ...` could not run because mypy is not installed in the
  project interpreter (`No module named mypy`).
- Fresh-process version inspection showed `_book_payload: 1.0`,
  `_state_payload: 1.0`, and `_validate_books: 1.0` as dependency-free leaves
  inside the resolved producer/validator graph.
- Independent candidate identity comparison matched the configured current
  internal Dataset pointer; no implicit source selection was observed.
- Independent re-signed mutations accepted a second-row return of `123.0` and
  a doubled `discounted_forward_over_spot`.

No network, WRDS connection, production data, GPU, or `quant_data` mutation was
used. Temporary probe artifacts were kept below
`/tmp/sanos/phase-8-code-review/`.

## Residual positive notes

- CD-004/CD-009 are documented as technical approvals only; release rights are
  explicitly separate.
- Candidate identity includes canonical configuration, current producer/schema
  versions, and the exact internal source pointer.
- Candidate output-first and wildcard behavior correctly avoids source access
  on a complete hit, and wildcard cannot build a miss.
- The published recursive row schema omits absolute spot and obvious vendor/
  contract identifiers, and advertised capability/non-capability flags are
  conservative.

Signed: `/root/code` — independent Phase 8 code reviewer

---

## Re-review — 2026-10-12

**Scope:** remediated Phase 8 tree; independent offline/adversarial re-review
**Verdict:** **FAIL / CHANGES REQUIRED**

Most original findings are materially resolved. The candidate and release now
have separate identities/namespaces and the same internal Dataset as their sole
direct source; approval requires exact candidate, rights, license, and checksum
evidence; concrete release publication reprojects the internal source and
requires exact candidate equality; candidate and release payload sets and
metadata are closed; staged validation precedes commit; deep validation
reprojects the source; shallow reuse remains source-independent; the carry
curve now uses all distinct source-day expiry nodes; and the clock/NMS/duplicate
book boundaries were corrected. Documentation accurately says shallow
validation cannot recreate private inputs and does not infer redistribution
rights.

Two acceptance-critical defects remain.

### R1. Critical — a public release is not shallowly bound to its approved
candidate digest

`PublicApproval.candidate_logical_content_sha256` is identity-owned
(`config.py:349-399`), and `build_public_release` correctly compares it to the
candidate before a new release is produced (`publish.py:618-650`). However,
`validate_public_payload` validates the release's newly recomputed self-digests
without comparing `evidence["logical_content_sha256"]` (or the loaded frame's
digest) with
`release_config.approval.candidate_logical_content_sha256`
(`validate.py:746-819`). Thus the immutable approval-to-release equality proof
is lost at the self-contained release boundary.

Independent reproduction:

1. build a two-day candidate and an approved release;
2. change the release's second `forward_adjusted_spot_return` to `42.0`;
3. coherently update `validation.json` return/logical digests, closed manifest
   metadata, file checksums, and completion checksum while leaving the
   identity-owned approval unchanged; and
4. run normal semantic validation.

`validate_public_dataset` returned `SHALLOW_ACCEPTED 2 42.0`. Deep validation
correctly rejected the same artifact as differing from source re-projection,
but shallow validation is supposed to verify the self-contained approval
evidence without candidate/source availability.

Required change: for `DistributionStatus.PUBLIC`, require the recomputed frame
logical digest to equal the approval's candidate logical-content digest before
returning from staged or committed payload validation. Also reconcile any
release validation/manifest copy of that value. Add a coherent re-sign
regression that updates every mutable self-digest/summary/checksum yet must fail
specifically on the immutable approved-candidate digest. Exercise the same
check with candidate and source removed.

### R2. High — the content-version dependency graph is still incomplete

The resolved graph is improved for projection, source schema, SANOS state,
calendar/NMS policy and Parquet settings, but important direct dependencies are
still absent:

- `validate_public_payload` directly calls `_payload_names`, `_read_json`,
  `_validate_coverage_evidence`, and `public_coverage_sha256`, and branches on
  `PublicReleaseConfig`/`DistributionStatus`, but none is in its decorator
  (`validate.py:681-819`); `_payload_names` itself is unversioned.
- `public_frame_logical_sha256` hashes
  `public_schema_document()["physical_schema"]` but declares only
  `canonical_sha256` and `public_polars_schema` (`validate.py:123-134`).
- `public_coverage_sha256` reads the versioned projection coverage contract but
  declares only `canonical_sha256` (`validate.py:154-169`).
- `public_manifest_metadata` consumes `PublicProjectionConfig`,
  `DistributionStatus`, and validation-document fields but omits those direct
  dependencies (`validate.py:645-678`).
- conversely, `_project_source_dataset` includes operational
  `PublicBuildOptions` even though it receives only a primitive progress
  interval (`publish.py:140-163`). A logging/progress-options change can
  therefore churn a content identity contrary to the repository rule.

Fresh-process inspection confirmed these omissions in both candidate and
release producer closures; there is still no exact direct-dependency regression
for the public roots.

Required change: version the remaining content helpers, declare every exact
direct dependency through projection/validation/document/release roots, remove
the operational-only dependency, and add fresh-process assertions for the
direct graph and planned identities. Dependency closure is part of artifact
identity, not optional cleanup.

### Remaining acceptance/test gaps

These are secondary to R1/R2 but should be closed at the same gate:

- candidate concurrency and staged-failure tests exist, but equivalent release
  concurrent-publisher and staged-failure tests do not;
- the release test proves normal build and output-first wildcard reuse but has
  no coherently re-signed approval/config/license/manifest mutation matrix;
- selector tests call the production `_daily_front_mask` directly rather than
  comparing the complete `_book_payload` population to an independent pinned
  oracle, and recursive nested null/NaN/infinity coverage remains incomplete;
- CLI tests cover help and release dry-run but not the real
  `approval-template -> approve -> validate public` human/JSON workflow; the
  validation command still describes/results as a candidate even for public
  releases (`cli/main.py:1281-1327,1358-1361,1406-1425`), and release projection
  emits `public_candidate_projection_progress` (`publish.py:214-222`);
- `public_validation_progress` contains completed/total/failed but no elapsed
  time or ETA (`validate.py:591-599`), contrary to the material-command progress
  contract; and
- `docs/public-datasets.md:79-83` says public state “and diagnostics” are in the
  frame although diagnostics are deliberately excluded by the actual
  allowlist.

### Re-review evidence

- Focused offline suite:
  `env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p
  no:cacheprovider tests/test_public_dataset.py tests/test_cli.py` — **45 passed**
  in 138.43s; two expected wildcard warnings.
- Full offline suite was started under the same isolation flags; its result is
  recorded below when complete.
- `ruff check src/sanos/datasets/public src/sanos/cli/main.py
  tests/test_public_dataset.py tests/test_cli.py` — passed.
- `python -m compileall -q src/sanos/datasets/public src/sanos/cli/main.py
  tests/test_public_dataset.py tests/test_cli.py` — passed.
- `git --git-dir=.git-real --work-tree=. diff --check` — passed.
- Checked reference candidate source pointer exactly equals the current planned
  internal Dataset pointer `2ea204…4585`; current candidate planning remains
  location-independent.
- Adversarial release probe: coherent self-re-sign accepted shallowly and
  rejected under exact deep re-projection, as detailed in R1.

No network, WRDS, production Dataset, GPU, or `quant_data` write was used.
Probe files were isolated below `/tmp/sanos/phase-8-code-rereview/`.

Signed: `/root/code` — independent Phase 8 code re-reviewer

---

## Final remediation re-review — 2026-10-13

**Scope:** final settled Phase 8 implementation after both remediation rounds;
independent offline, adversarial, concurrency, CLI, documentation, and version-
closure review
**Verdict:** **PASS**

This verdict supersedes the two earlier failure verdicts for the settled tree.
No blocking or acceptance-critical code finding remains.

### Finding closure

1. **Distinct approval and release — resolved.** A release has its own
   `sanos/public/` identity and immutable payload. Its sole direct upstream is
   the same internal Dataset as the candidate, not the candidate itself.
   `PublicApproval` binds the exact candidate pointer, candidate manifest and
   logical-content digests, license digest, license, rights basis, approver,
   date, and durable reference. A concrete release validates the candidate,
   reprojects the exact internal Dataset, requires exact logical-frame and
   deterministic-Parquet equality, validates the complete staged release, and
   only then publishes atomically. It neither aliases nor mutates the
   candidate. No repository value is treated as proof of redistribution
   rights.

2. **Output-first and approval binding — resolved.** Current and wildcard
   release hits are resolved and self-validated before candidate, source, or
   license access. Wildcard remains reuse-only and warning-emitting. A release
   miss requires concrete producer identity plus exact license bytes. Shallow
   release validation now requires the recomputed logical-content digest to
   equal the identity-owned approved-candidate digest. My prior coherent
   return/self-digest/manifest/completion re-sign probe now fails with
   `Public release logical content differs from the approved candidate`, while
   complete normal release reuse works with candidate and internal source
   moved away.

3. **Payload, leakage, source fidelity, and carry — resolved.** Candidate and
   release payloads have exact distribution-specific allowlists, canonical
   documents, a closed recursive schema, exact metadata, file checksums, finite
   nested values, public-only numerical reconstruction, coverage and return
   digests, and release evidence/license checks. Deep validation performs the
   exact internal-source reprojection and complete ordered-frame comparison;
   shallow validation honestly remains self-contained. The CD-004 carry curve
   uses every distinct positive source expiry node, including nodes absent
   from model state. CD-009 units, corrected DLV state, SANOS state, ATM/NMS
   reconstruction, side-specific books, and the approved clock/filter
   boundaries are validated and covered by quantitative review.

4. **Atomicity and concurrency — resolved.** Full staged validation precedes
   immutable commit. Injected candidate and release staged-validation failures
   leave no final artifact or staging directory. Concurrent identical candidate
   and release publishers converge on one valid winner. During this final
   review I independently reproduced a process-global `WRDS_OFF` restoration
   race in concurrent public validation; the implementation now performs no
   environment mutation. Re-running the deterministic ordered two-thread probe
   left `WRDS_OFF` absent after both validators, and a repository regression
   covers concurrent validation with the variable absent.

5. **Version and identity closure — resolved.** Remaining payload enumeration,
   JSON read, coverage, release/distribution, schema-document, metadata, and
   source-projection helpers are versioned and declared through the producer
   and validator roots. `public_coverage_sha256` directly owns
   `PublicProjectionConfig`; logical-content hashing owns the public schema
   document; metadata owns projection/distribution contracts; and operational
   build options are absent from the content projection closure. Exact direct-
   dependency assertions pass. The immutable release README changed content
   version to `1.1` when its documented exact-ID command was corrected.

6. **Human and structured CLI — resolved.** The checked workflow exercises
   `approval-template -> approve -> validate` with real synthetic artifacts in
   JSON mode. Release projection has a release-specific event, public
   validation language/results are distribution-neutral, and validation
   progress reports elapsed/completed/total/failed/ETA. The generated release
   README's copyable exact-ID command includes `--distribution public` and is
   executed successfully by the regression. Group help covers candidates and
   releases. Documentation no longer describes internal fit diagnostics as
   public fields.

### Independent final evidence

- Focused remediation/static gate:
  `env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p
  no:cacheprovider tests/test_public_dataset.py -k 'public_release_cli_template
  or concurrent_public_validation or release_staging_failure or
  public_validation_version_dependencies'` — **4 passed, 24 deselected** in
  24.71s.
- Full settled offline suite:
  `env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p
  no:cacheprovider` — **229 passed** in 305.97s. The only warnings were the two
  intentional visible `version="*"` reuse warnings.
- Deterministic env-absent ordered concurrency probe — both validators passed;
  `WRDS_OFF` was absent after the first and after the second completion.
- Coherently re-signed release-content probe — rejected shallowly on the
  immutable approved-candidate logical digest.
- `ruff check src/sanos/datasets/public src/sanos/cli/main.py
  tests/test_public_dataset.py tests/test_cli.py` — passed.
- `python -m compileall -q src/sanos/datasets/public src/sanos/cli/main.py
  tests/test_public_dataset.py tests/test_cli.py` — passed.
- `git --git-dir=.git-real --work-tree=. diff --check` — passed.
- Mypy was not available in the project interpreter (`No module named mypy`);
  this does not displace the passing configured offline and static gates.

No network, WRDS connection, production Dataset, GPU, or `quant_data` write was
used. Temporary adversarial probes remained below
`/tmp/sanos/phase-8-code-rereview/`.

### Residual production gate

Phase 8 code is ready to commit. Public release still requires genuine human
rights approval and license evidence; the repository intentionally contains
neither. The fresh 2022–2025 build must still pass the Phase 10/11 production
population, discrepancy, and promotion gates. The fresh validated/promoted
SANOS Dataset becomes authoritative only after that process; the legacy
891-date `quant_data` artifact remains a migration/parity oracle.

Signed: `/root/code` — independent Phase 8 code implementation reviewer
