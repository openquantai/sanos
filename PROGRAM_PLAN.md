# SANOS standalone program plan

Status: implementation in progress; Phases 0 through 9 are complete with dual
review. Phase 10's offline prerequisites are complete, and Phase 11/12 offline
production, packaging, and operator-readiness work passed final code and
quantitative review on 2026-10-13.
Public redistribution approval remains a separate human/legal action. The
account-scoped attempt monitor is implemented and offline-tested; the live pilot
remains pending exact-commit preflight evidence and a healthy daemon-backed
resource gate. No live WRDS access has occurred.

Initial review anchors are paper commit
`15e07d4ab0652f837883550d555f601836685db9` and clean `quant_data` commit
`de15be96d2f283b6c31e60d3ea47325a0c093ff0`. Phase 0 must reconfirm them.

## Outcome

Build a standalone, human-operable Python repository that:

- retains the paper under `tex/`;
- copies and refactors the proven SANOS and WIDB implementation from the
  read-only `~/python/quant_data` repository;
- preserves current numerical and filtering semantics unless a change is
  explicitly reviewed and approved;
- downloads fresh WRDS OptionMetrics/IvyDB source data for the reference
  2022--2025 period and never uses an old cache as an input to that release;
- produces immutable internal and separately materialized public datasets;
- produces disposable experiment-specific HDF5 files from a dataset; and
- exposes every operation through a documented `sanos` CLI with human-readable
  and structured progress output.

The existing 891-date SPX dataset from 2022-01-03 through 2025-08-29 is the
migration acceptance oracle. Phase 1 identifies its exact files, configuration,
and checksums before implementation. The fresh build must not consume it, but every
date, population decision, field, mask, SANOS/DLV value, and experiment tensor
is compared with it. An unexplained discrepancy blocks publication, while an
approved correction is recorded as an intentional difference. After the fresh
dataset passes publication validation and dual review, its manifest identity is
the canonical SANOS reference dataset; the legacy oracle is no longer normative.
The date range remains configuration; an extension through 2025-12-31 is a
separate dataset identity.

## Binding design decisions

1. The four storage classes are **raw**, **intermediate**, **dataset**, and
   **experiment**. `processing` is accepted only as a legacy alias for
   `intermediate`; `derived` is not a separate class.
2. Existing code should be copied selectively and refactored in SANOS. Existing
   data caches must not be migrated into the new release.
3. Semantic parity is the default. A suspected defect or proposed behavioral
   improvement goes into `docs/compatibility-decisions.md` and blocks the
   affected change until it has code review, quantitative review, and user
   approval.
4. Up to time-series assembly, each stage processes one trading day and returns
   one Polars `DataFrame`. Quotes, expiry metadata, masks, fitted values, smooth
   SANOS state, unsmoothed DLV state, and diagnostics remain in that frame.
5. Raw and published datasets use Parquet. HDF5 belongs only to disposable,
   model-specific experiment materializations.
6. `overnight_variance_fraction` is an explicit versioned configuration value.
   Its default is `0.2`, exactly matching the current effective behavior
   (`AM_EXIRY_SUBTRACT = 0.8`). It is included in artifact identity.
7. The core WIDB build uses WRDS OptionMetrics only. FRED, Yahoo Finance, and
   Massive are not dependencies. Yahoo may later be an optional consumer-side
   absolute-spot adapter for non-European use cases.
8. A public release is a new lossy materialization, never an alias or flag on an
   internal dataset. It is built as `public_candidate` until the manifest has
   explicit redistribution approval and license metadata.
9. The stable workflow API is the CLI and artifact contract. The first
   implementation uses a small SANOS-owned runner with atomic artifacts, file
   locks, bounded processes, and a JSONL journal. It does not use Redun: both
   reviews found that Redun would duplicate SANOS artifact identity and
   `@version` semantics without an agent-management benefit. Reconsider it only
   if remote executors, dynamic cross-project DAGs, or centralized lineage
   become concrete requirements.

## Quantitative compatibility contract

The reference pure-option normalization is

```text
k = K / F_T
c(T, k) = C(T, F_T k) / (DF_T F_T)
c - p = 1 - k
```

For every expiry, the unsmoothed state retains normalized strikes `k_i` and
density masses `q_i` with non-negative mass, `sum(q_i) = 1`, and
`sum(k_i q_i) = 1`. It must contain model time increments, interior squared-DLV
values, boundary convention, and enough grid geometry to invert DLV to density.
The smooth SANOS state uses the same density support and the reference Black
mixture. The existing production smoothing fraction is `0.25`; this is distinct
from `overnight_variance_fraction=0.2`.

These equations, current tolerances, filter ordering, and solver behavior are
the starting contract. Tests compare invariant prices as well as parameters;
they do not require identical LP density vectors when the optimum is degenerate.

## Target repository

```text
sanos/
├── AGENTS.md
├── PROGRAM_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── datasets/spx-sanos-2022-2025.toml
│   └── experiments/
├── docs/
│   ├── architecture.md
│   ├── compatibility-decisions.md
│   ├── data-contracts.md
│   └── wrds-runbook.md
├── src/sanos/
│   ├── __main__.py
│   ├── cli/
│   ├── config/
│   ├── artifacts/
│   ├── workflow/
│   ├── wrds/
│   ├── widb/
│   ├── model/
│   ├── datasets/
│   └── experiments/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── parity/
│   ├── cli/
│   └── fixtures/
└── tex/
```

Numerical kernels may continue to use NumPy, SciPy, CVXPY/HiGHS, and JAX where
appropriate. Polars is the data-frame and Parquet boundary; it is not imposed on
linear-algebra kernels.

## Artifact and execution model

```text
WRDS ──explicit guarded download──> RAW Parquet
                                      │
                                      ▼
                              daily INTERMEDIATE
                         clean → enrich → no-arb → fit
                                      │
                                      ▼
                           immutable internal DATASET
                              │                  │
                  lossy public projection       └─> EXPERIMENT HDF5
                              │
                              ▼
                 independent public DATASET
```

The artifact resolver is output-first. For `dataset build` or `experiment
build`, it first validates the requested output's manifest, checksums, schema,
configuration identity, and completeness. A valid output is returned without
requiring any ancestor to exist. Only a miss invokes the build graph.

`version="*"` may reuse a complete artifact while ignoring only the recorded
producer-code version. It still checks configuration, schema, identity,
completeness, and checksums; it emits a visible warning; and it can neither
build nor replace anything.

The runner exchanges artifact references, not large frames. Its journal and
rebuildable state live under `QUANT_DATA_INTERMEDIATE` and may be deleted. Raw
retention, dataset publication, public approval, and output-first reuse remain
explicit SANOS policies.

## Configuration and locations

Resolution precedence is:

1. explicit CLI option;
2. `QUANT_DATA_*` environment variable;
3. ignored repository-local TOML;
4. `~/.config/quant_data/sanos.toml`;
5. a documented location beneath `QUANT_DATA_ROOT`.

The canonical variables and legacy read-only fallbacks are:

| Canonical | Legacy fallback | Root default |
|---|---|---|
| `QUANT_DATA_RAW` | `QUANTDATA_RAW_DATA` | `raw/` |
| `QUANT_DATA_INTERMEDIATE` | `QUANTDATA_PROCESSING_DATA` | `intermediate/` |
| `QUANT_DATA_DATASETS` | `QUANTDATA_TRAINING_DATA` | `datasets/` |
| `QUANT_DATA_EXPERIMENTS` | `QUANTDATA_EXPERIMENTS` | `experiments/` |

`QUANT_DATA_ROOT` falls back to `QUANTDATA_ROOT`. Canonical names win, and a
conflicting canonical/legacy pair produces a warning. Source-specific data has
a natural path beneath its class root, for example `raw/wrds/optionmetrics/`;
the whole root can be overridden without embedding machine paths in dataset
configuration.

Checked-in TOML is parsed and normalized to canonical JSON. Every artifact
identity hashes canonical configuration, schema/stage versions, and upstream
artifact identities. Deterministic configurations and transformations use
`@version` from `cdxcore.version` with explicit dependencies.

## WRDS safety contract

Only an explicit live Raw cache miss opens WRDS.

- No interactive readiness prompt or special confirmation flag is required.
  Every connection open is serialized by a persistent, account-scoped monitor.
  It retains failed or interrupted attempts for 15 minutes and rejects a fourth
  attempt in that rolling window before network access. A successful connection
  clears the window; pre-network configuration failures do not consume it.
- A single connection/authentication probe succeeds before fan-out begins.
- The default is two WRDS connections. `--max-wrds-connections` is hard-limited
  to four; a value above four fails before opening a connection.
- A host-wide SANOS lease/semaphore prevents two simultaneous commands from
  exceeding four connections in aggregate for the same WRDS account. Its key
  and location are independent of configurable data roots. Leases record PID,
  host, and start time and reclaim only demonstrably stale owners.
- The foreground authentication probe either closes before fan-out or continues
  while consuming one of those four semaphore slots; it is never an uncounted
  fifth connection.
- Authentication, permission, connection-limit, unknown, and 2FA failures are
  not retried within a command. Confirmed transient query failures get bounded
  exponential-backoff retry with no more than three total attempts.
- `--dry-run`, `--cache-only`, offline tests, and validation of existing
  artifacts never open a connection.
- Downloads use resumable, atomic partitions and report elapsed time,
  completed/total partitions, failures, and ETA in prose or JSON.

The fresh reference download resolves the account's available
`optionm_all`/`optionm` tables for security identity (`securd`/`securd1`),
option quotes (`opprcdYYYY`), underlying closes (`secprd`), forward prices
(`fwdprdYYYY`), and zero curves (`zerocd`). Each table root contains generated
source/schema Markdown, query provenance, a machine-readable manifest, and
partitioned Parquet. Credentials and 2FA details are never persisted.

## Review and change-control gate

Every numbered phase below uses the same mandatory gate:

1. The implementer supplies a small evidence bundle: scoped diff, tests and
   results, representative CLI transcript/JSON events, artifact/schema changes,
   and unresolved risks.
2. The `code` reviewer independently checks architecture, maintainability,
   failure behavior, security, CLI contracts, tests, and compatibility.
3. The `quantagent` reviewer independently checks market-data meaning,
   normalization, calendar and settlement conventions, no-arbitrage and SANOS
   mathematics, statistical diagnostics, and data leakage.
4. Findings are recorded under `reviews/phase-N/`, resolved, and re-reviewed by
   both agents. The next phase cannot start until both approve.
5. If either review identifies a semantic discrepancy, record it in
   `docs/compatibility-decisions.md` and stop that affected work for user
   approval. Refactors with parity evidence may proceed without changing the
   decision.

This gate applies to each phase, including documentation, the download pilot,
and the full production run.

## Delivery phases

### Phase 0 — Repository and provenance bootstrap

- Attach the existing GitHub repository to this working directory and preserve
  its history.
- Move the existing TeX, bibliography, figures, and generated paper assets to
  `tex/` with history-preserving moves.
- Add the `src/` package layout, `pyproject.toml`, test/lint/type tooling,
  license/provenance inventory, CI, and minimal `python -m sanos`/`sanos`
  command.
- Import relevant source history from a clean, filtered view of the reviewed
  `quant_data` commit, then make an exact snapshot commit before refactoring.
  Record original paths, source revision, import method, and adaptations in
  `PORTING.md`; do not copy from an uncommitted working-tree version.
- Pin only necessary base dependencies and isolate WRDS, JAX, documentation,
  and development dependencies in extras where practical.

Acceptance: clean install in a fresh environment; `sanos --help` and offline
smoke test pass; paper still builds; no generated data is inside Git.

### Phase 1 — Reference inventory and executable parity baseline

- Map every required `quant_data` module, function, configuration, test, and
  dependency to copy, adapt, replace, or omit.
- Identify the actually used production path for the current dataset rather
  than assuming every historical module is live.
- Copy representative tests first and establish synthetic/golden numerical
  fixtures that can run without WRDS.
- Catalogue and checksum the existing 891-date legacy dataset outside Git,
  and build a read-only comparison adapter for it. It is the migration parity
  oracle but never an upstream artifact of the new build.
- Capture current versions, default values, schemas, filter order, tolerances,
  solver choices, and 2022--2025 date/eligibility counts as acceptance evidence.
- Open the compatibility decision register with all suspected defects.

Acceptance: traceability matrix covers the pipeline end-to-end; reference tests
can be run against `quant_data` read-only; no source repository or cache was
modified.

### Phase 2 — Configuration, artifacts, CLI shell, and local runner

Status: complete; code and quantitative/data-contract reviews passed on
2026-10-12. See `reviews/phase-2/`.

- Implement typed, versioned configuration and the location precedence above.
- Implement canonical serialization, content identities, checksummed manifests,
  atomic writers, completion markers, locks, and shallow/deep validation.
- Implement `doctor`, `status`, logging (`-v` and `--log-format json`), dry-run,
  and stable exit codes.
- Implement a deterministic daily planner/executor using atomic artifacts,
  locks, bounded `ProcessPoolExecutor` work, and an append-only JSONL run
  journal. Keep WRDS connection limits separate from CPU process counts.
- Test `version="*"` precisely, including the required fail-on-miss behavior.

Acceptance: artifact lifecycle and CLI contract tests pass, including legacy
environment fallback and conflicting-variable warnings.

### Phase 3 — SANOS and no-arbitrage numerical core

Status: complete; code and quantitative/data reviews passed on 2026-10-12. See
`reviews/phase-3/`.

- Copy and adapt the relevant implementations of tridiagonal operations,
  arbitrage-free pricing/filtering, discrete density and DLV transforms,
  `SanosPricer`, `SanosLPSolver`, `SanosFitter`, and the required JAX path.
- Preserve algorithms, default tolerances, objective functions, solver choices,
  strike expansion/interpolation, and warning/error semantics until parity is
  demonstrated.
- Separate pure numerical APIs from WIDB and artifact concerns.
- Document equations, units, shapes, and failure modes against the paper.

Acceptance tests include:

- call bounds, decreasing strike prices, convexity, calendar monotonicity;
- non-negative density, unit mass, and normalized first moment;
- DLV/density/call round trips and boundary behavior;
- call boundary values `C(0)=1`, left slope `-1`, and zero far tail;
- smoothness limits and smooth/unsmoothed state geometry;
- explicit smoothness `0` and production `0.25` parity;
- feasible/infeasible LP behavior, accepted per-expiry quote containment, and
  soft surface-fit quote-band diagnostics;
- unequal-grid DLV/density round trips;
- NumPy/JAX value and gradient parity where both paths are retained; and
- numerical parity with the reference on shared deterministic inputs.

### Phase 4 — Guarded WRDS raw layer

Status: complete; offline verification and independent code and quantitative
reviews passed on 2026-10-12. No live WRDS query was part of this phase's
review. See `reviews/phase-4/`.

- Implement the one-connection probe, global bounded lease, retry
  classification, progress events, and resumable atomic writes. The original
  manual 2FA gate was superseded by the owner-approved account attempt monitor.
- Keep raw extracts minimally transformed. Move filtering currently embedded in
  `collect_raw_day` into later daily stages while preserving the resulting
  processed outputs.
- Generate source/field Markdown and extraction manifests beside each raw table.
- Implement explicit selection by source, ticker, and date, plus cache-only
  validation and refresh. Keep direct secid override blocked under CD-010 until
  the legacy ignored-argument defect has an approved replacement contract.
- Compute raw dependency closure before download. For a build beginning on
  2022-01-03, freshly download the previous business-day underlying close used
  by the internal legacy-compatible return. The approved public return is null
  on its first retained row and therefore requires no pre-start forward or zero
  curve. Show the exact boundary prerequisites in dry-run output and never fill
  them from legacy data.

Acceptance: mocked failure/retry/concurrency tests pass; an offline test proves
that every non-live command refuses network access; no live WRDS query occurs in
this phase's ordinary test suite.

### Phase 5 — Daily WIDB normalization and enrichment

Status: complete; final code and quantitative reviews passed on 2026-10-12.
The initial migration preserves the CD-020 legacy all-valid failure; repairing
it remains an optional future decision. The implementation uses
`paired-day` and `enriched-day` immutable Intermediate artifacts, the existing
atomic artifact store, and the bounded spawn runner. It does not use Redun and
it performs no implicit WRDS access.

- Copy and adapt business-calendar, table resolution, zero-curve interpolation,
  forward, spot-return, quote pairing, expiry, and enrichment logic.
- Convert the daily pipeline boundary to Polars while preserving numeric dtypes,
  sorting, tie-breaking, filters, and diagnostics.
- Introduce `overnight_variance_fraction: float = 0.2` in the relevant daily
  configuration; calculate the AM adjustment as
  `1 - overnight_variance_fraction`; validate `[0, 1]`.
- Keep trading-session half-day and expiry AM/PM indicators separate.
- Test zero-curve rates in vendor percent units, continuous ACT/365 discounting,
  and positive discount factors; the implied `DF*F` bid/ask interval and its
  inverse-spread-weighted projection; cash-to-pure normalization; and pure
  put-call parity.

Acceptance: default `0.2` reproduces reference BDTE/TTE exactly on parity
fixtures. Validation covers endpoints, while sensitivity cases `0.1`, `0.2`,
and `0.5` trace AM expiry ordering, NMS/eligibility, total and forward variance
interpolation, calendar constraints, DLV time increments, and experiment
horizons. PM values remain unchanged; AM DTE shifts exactly by the change in
fraction; artifact identity changes. A report compares populations and fit
diagnostics without assuming monotonic improvement. Every day remains one
self-describing Polars frame.

### Phase 6 — Daily filtering, no-arbitrage processing, and SANOS fitting

- Copy the current filter order and eligibility rules, then integrate the
  no-arbitrage and SANOS cores.
- Retain all quote, liquidity, expiry, eligibility, fitted, SANOS/DLV, and
  diagnostic columns in one daily frame.
- Preserve the migration implementation's explicit float32 daily-frame cast
  boundaries before selected numerical arrays are promoted back to float64 by
  the solvers. A float64-through-fit variant requires measured parity evidence,
  an explicit numerical error budget, a new version, and approval under
  CD-011.
- Preserve deterministic ordering by date, expiry, settlement, normalized
  strike, and side. Prefer one row per retained expiry with nested list/struct
  columns, provided the Phase 1 schema parity review confirms that it preserves
  every reference observation and mask.
- Record stage reasons for every removed quote, expiry, and day.
- Make solver failures resumable and inspectable without converting them into
  silent exclusions.

Acceptance: per-stage row/expiry counts and numerical values match reference
fixtures within declared tolerances; all no-arbitrage invariants hold; restart
after a deliberately interrupted multi-day build succeeds.

### Phase 7 — Internal dataset assembly and publication

- Assemble daily outputs without padded arrays into partitioned Parquet.
- Publish only after completeness, schema, invariant, checksum, coverage, and
  exclusion validation.
- Write `README.md`, canonical `config.json`, and `manifest.json` containing
  Git revision, resolved transformation versions, upstream raw identities,
  date/row counts, exclusions, checksums, and `distribution: internal`.
- Make the published dataset independently loadable after intermediates are
  removed.

Acceptance: deleting a copied intermediate tree does not affect loading or
shallow validation; a missing published dataset is never silently reconstructed
under an existing identity.

### Phase 8 — Public research projection

Status: complete; final code and quantitative/data reviews passed on
2026-10-13. The repository publishes technical `public_candidate` artifacts
and provides a distinct rights-gated `public` release workflow, but records no
actual redistribution approval. See `reviews/phase-8/`.

- Materialize a distinct one-row-per-trading-day Polars dataset from the
  completed internal dataset.
- Use an explicit allowlist containing only date, trading half-day status,
  the approved `forward_adjusted_spot_return` into the date under CD-004,
  smooth SANOS state and geometry, unsmoothed DLV state and geometry, normalized
  surviving strike/side/half-spread books, and retained
  expiry/DTE/settlement geometry.
- Store each model's complete state and interpreting geometry so model prices
  and volatilities can be reconstructed, rather than publishing redundant
  fitted price or implied-volatility arrays on every retained book.
- Preserve the legacy simple close-to-close `spot_return` in the internal
  reference-compatible dataset, but do not label it forward-adjusted or use it
  as the public field. The approved public calculation for retained dates
  `t[i-1]` and `t[i]` uses previous-date carry
  `c[i]=F_(t[i-1])(t[i])/S_(t[i-1])` and
  `r_fwd[i]=(S[i]/S[i-1])/c[i]-1`. Positive forward and discount curves are
  interpolate log-linearly in business DTE with an anchor of one at
  DTE zero. The first retained date has a null return. Approved CD-004 fixes
  this formula; its inputs need not themselves be public fields.
- The approved unit bridge is `C/S=DF*(F/S)*C/(DF*F)` and
  `h/(DF*F)=(h/S)/(DF*(F/S))`. Under CD-009, the interpreting
  geometry stores `discounted_forward_over_spot=DF*(F/S)` and half-spreads in
  spot-one units. Model prices are reconstructed in memory from state and
  expressed in discounted-forward units; they are not redundant public fields.
- Freeze the other public units and geometry in Phase 1: strikes as `K/F`, SANOS
  prices in `DF*F` units, half-spreads as `half_spread/S`, annualized
  squared-DLV/time units, model time increments, expiry interpolation, and
  boundary flags.
- Exclude absolute spot and all source identifiers, raw quotes, market mids,
  volume, open interest, vendor Greeks/IVs, contract symbols, and redundant
  per-book model price/IV arrays. Leakage tests reconstruct model prices and
  implied volatilities from the allowed states in memory without adding them to
  the public schema.
- Record explicit capability flags in the manifest for supported normalized
  SANOS/DLV dynamics and spread experiments, and explicit non-capabilities for
  exact vendor-book reconstruction, quote-containment replay, liquidity models,
  absolute-dollar ticks, and contract-level replay.
- Reproduce the recovered daily-front public-book profile: separate call and
  put books, active finite valid books after the pre-smooth no-arbitrage mask,
  European style, finite smooth fitted values, at most 70 business days,
  explicit 0.2D exclusion, and inclusive AF-ATM NMS `[-3,2]`. A different
  estimation grid or experiment hedge screen is an explicit downstream
  experiment configuration, not an implicit public serialization rule.
- Keep SANOS smoothing-kernel input total variances explicitly named as inputs,
  not fitted ATM implied volatilities. Model ATM implied volatility is derived
  by pricing the retained state at `K/F=1` and inverting with `cdxcore.bs`.
  Also retain the kernel-input annualized ATM curve used to convert the chosen
  NMS population screen to `K/F`, so selection can be independently verified.
- Publish initially as `public_candidate`. A separate approval action records
  license, rights basis, approver, date, and durable reference before immutable
  `public` publication.

Acceptance: forbidden-column and reconstruction/leakage tests pass; the dataset
loads with WRDS disabled and no internal artifacts; every advertised normalized
experiment capability executes from it.

### Phase 9 — Experiment materialization

Implementation status: complete with code and quantitative review. This
Experiment is disposable. The freshly rebuilt and promoted SANOS Dataset
remains the canonical reference object.

- Convert a selected immutable dataset into model-specific HDF5 with explicit
  padding, masks, selections, and train/validation splits.
- Record source dataset identity and full canonical experiment configuration
  beside the HDF5 file.
- Keep experiment artifacts disposable and output-first reusable.

Acceptance: the reference training data consumer can ingest the new
materialization with value/mask/shape parity, or any intentional interface
change has an approved compatibility decision and adapter.

### Phase 10 — Fresh-download pilot

Offline prerequisite status: implemented, tested, and approved independently by
the code and quantitative reviewers. The operational pilot and its external
evidence review are still pending; Phase 11 remains blocked.

- Run `gpuctl resources --json` as required by host policy before material work.
- Enforce the account-scoped connection-attempt monitor immediately before each
  open; no manual readiness gate is required.
- Download a tiny, representative matrix of normal, half-day, AM-settled,
  irregular, and historically troublesome dates with at most two connections.
- Build those dates end-to-end and compare with the read-only reference outputs.
- Resolve all code and quant review findings before authorizing the full range.

Acceptance: authentication/concurrency behavior is safe; provenance and schema
docs are complete; discrepancies are explained or approved.

### Phase 11 — Fresh 2022--2025 production build

Offline contract status: the exact full-range comparator, Experiment tensor
comparator, deterministic downstream-config finalizer, indexed evidence
contract, canonical designation, and human runbook are implemented, tested, and
independently approved. Live execution remains blocked by the Phase 10 pilot
and monitored safety gates.

- Through the connection-attempt monitor, download every required raw partition
  fresh. Do not read old raw/intermediate data as an input.
- Resume safely across interruptions, never exceeding four WRDS connections;
  default to two unless pilot evidence justifies more.
- Build daily intermediates, publish the internal dataset, materialize and audit
  the public candidate, and build the reference experiment HDF5.
- Compare coverage, exclusions, distributions, fits, and all legacy-oracle values
  with the existing 891-date dataset. A vendor revision is a discrepancy to
  investigate and present for approval, never a reason to mutate fresh data to
  force equality or to waive the reference silently.

Acceptance: both reviewers approve the production evidence bundle; all manifests
and checksums validate; the documented non-AI reproduction commands work; and
the published SANOS dataset identity is recorded as the canonical reference.

### Phase 12 — Documentation and release readiness

Offline implementation status: the non-AI operator path, troubleshooting,
retention policy, aggregate operator extra, self-contained wheel/sdist contract,
installed-artifact verifier, Markdown-link check, and out-of-tree paper CI are
implemented and passed final dual review. Project licensing and the
`.git-real` metadata handoff remain explicit owner/host release gates.

- Complete the README path from installation and configuration through WRDS
  authentication, raw download, intermediate build, dataset publication,
  validation, public approval, experiment build, cleanup, and recovery.
- Document every field and normalization convention, including spot/forward/
  discounted-forward units and calendar/business DTE.
- Add troubleshooting, structured-log examples, storage cleanup policy, and a
  no-AI operator runbook.
- Build package, docs, and paper; run the full offline suite in a clean
  environment.

Acceptance: a user following only repository documentation can reproduce all
non-credentialed operations and understands the WRDS credential, attempt-monitor,
and recovery behavior.

## Initial compatibility decision register

The following item is flagged now and must not be silently copied or fixed:

- `quantdata/sanos/widb_sanos_day.py::fit_sanos_day` extracts an `expiries`
  array but initializes `market.expiries` as empty and computes unique expiries
  from that empty list. The grouping loop therefore appears unable to populate
  the SANOS input. It also appears to request stale field names. Phase 1 must
  confirm whether this path produced the legacy dataset, add a regression
  test, identify the actual production path, and request approval for the
  intended correction. Byte parity with a non-working path is not a goal.

The two reviews also flagged these decisions. They remain unresolved until
Phase 1 supplies focused evidence and the user approves any semantic change:

- `WIDBClient.options()` appears to accept missing expected columns because its
  set-inclusion test is oriented permissively, despite stricter error text.
- `enrich_day()` has schema validation after an early empty-data return and the
  validation therefore appears unreachable on that path.
- Legacy `spot_ret` is `spot / previous_spot - 1` and remains an internal
  compatibility field. The public projection uses the distinct approved
  `forward_adjusted_spot_return` contract from CD-004 and must not relabel the
  legacy field.
- Some fit-percentage documentation/log messages disagree about whether the
  reported fraction is inside or outside the bid/ask spread.
- Solver timeout is excluded from the current cache identity even though it may
  affect whether a fit completes and which output is produced.
- Saturday AM expiry behavior is guarded mainly by assertions and requires
  explicit regression fixtures.
- The generic legacy option cap defaults to 1,000, while recent DYSANOS/FLEX
  paths use a practically non-binding 100,000 before later selection. The
  legacy dataset's actual population rule must be recovered, not guessed.
- A FLEX 4.8-business-day first-node convention may encode a different 20%
  overnight clock from WIDB AM settlement. Preserve each behavior until traced.
- Public half-spreads in spot-one units and model prices reconstructed in
  discounted-forward units use the approved dimensionless `DF*(F/S)` bridge
  from CD-009. This technical approval does not grant redistribution rights.
- The ndarray branch of `_compute_market_fit` references undefined variables;
  CD-013 blocks silently copying or fixing the unexercised interface.
- Both retained legacy pricers jump from the time-zero intrinsic branch to the
  first fitted density for positive times before the first model expiry.
  CD-014 freezes this initially and blocks changing it without approval.

The configurable overnight fraction is not an unapproved semantic change at
its default: `0.2` must reproduce the existing output exactly. Any new default
requires a later, separately versioned and approved dataset configuration.

## Completion definition

The program is complete when all phases and dual reviews pass; the repository
contains no generated proprietary data; the fresh internal reference dataset,
public candidate, and experiment are complete and validated in their configured
external roots; the old `quant_data` cache is not required; and a human can run
the documented CLI workflow without an agent.
