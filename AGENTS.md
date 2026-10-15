# SANOS repository instructions

## Scope and migration

- This repository owns the standalone SANOS Python implementation, its data
  pipeline, analytics, tests, documentation, and LaTeX paper sources.
- Treat `~/python/quant_data` as a read-only migration implementation. Copy and
  adapt required code into this repository, preferring proven implementation
  over a clean-room rewrite; do not modify `quant_data` as part of SANOS work
  unless the user explicitly requests a separate compatibility change there.
- Measure numerical and data parity with the migration implementation through
  explicit parity tests before removing or redesigning behavior. Approved
  corrections are then owned directly by SANOS and need no legacy runtime mode.
- Preserve current semantics by default. Record suspected defects or desirable
  semantic changes, their evidence, and their expected output impact, and obtain
  user approval before implementing them. Do not silently preserve or fix a
  suspected defect while migrating code.
- Rebuild the target 2022--2025 dataset from newly downloaded WRDS extracts.
  Existing raw, processing, training, or experiment artifacts from
  `quant_data` must not be inputs to the new published dataset. The existing
  891-date fitted 2022--2025 dataset is the read-only migration oracle:
  catalogue it and treat unexplained differences in dates,
  populations, masks, SANOS/DLV values, or experiment tensors as blocking
  discrepancies requiring review and user approval. Once the fresh SANOS
  dataset passes publication validation and both reviews, its immutable
  identity and manifest become the authoritative reference dataset. The legacy
  oracle does not remain a co-equal reference and is never a runtime input.
- Keep generated data, caches, datasets, experiment files, logs, and temporary
  artifacts outside the Git working tree. Under a shared temporary directory,
  place all SANOS files below a project-specific `sanos/` directory.

## Human-operable software

- The framework must be fully usable without an AI agent. A user following the
  README must be able to configure locations, inspect the installation,
  download raw data, build intermediates, publish datasets, prepare experiments,
  validate outputs, and diagnose failures from ordinary CLI commands.
- Notebooks may demonstrate APIs but must never be the only way to run a
  pipeline operation.
- Do not require an MCP server, agent skill, IDE integration, or direct database
  manipulation for normal operation. Agent integrations should call the same
  documented CLI as human users.
- Keep CLI behavior stable and scriptable. Commands must have meaningful exit
  codes, non-interactive defaults, `--help`, and actionable error messages.

## Model namespace

- SANOS is the preprocessing, surface-fitting, and Dataset-producing layer.
  Learned, statistical, hedging, and other downstream research models belong
  below the plural `sanos.models` namespace, with one self-contained package
  per retained model.
- DySANOS 1.1 lives under `sanos.models.dysanos`. It owns its public-Dataset
  adapter, disposable HDF5 materialization, differentiable decoder, objective,
  initialization, dynamics, training, validation, artifacts, and CLI. Do not
  recreate the legacy `dysanos.flex` research forest or a generic Python
  `experiments` namespace around this single concrete model.
- A future retained model becomes a sibling of `dysanos`; it must declare its
  own Dataset capabilities, configuration identity, CLI, artifacts, and tests.
  Treat every model package as independently authored and self-contained. Do
  not create shared model implementation infrastructure merely because two
  packages perform similar work. Share only narrow, stable repository-level
  contracts (for example artifact pointers or Dataset interfaces) when model
  authors explicitly choose to depend on them; keep model algorithms,
  configuration, training, persistence, and CLI orchestration within each
  model package.
- The Experiment lifecycle name remains the fourth storage class. It describes
  disposable model-specific artifacts and does not imply a generic source-code
  package or framework.
- A model may produce a chain of immutable Experiment artifacts when each has a
  useful independent lifecycle. DySANOS uses one public-derived HDF5 Experiment
  and one trained-model Experiment whose sole upstream is that HDF5 artifact;
  shallow validation remains output-first and deep validation follows the
  chain to the Dataset.

## Data lifecycle terminology

Use exactly these four storage classes in code, configuration, CLI help, and
documentation:

1. **Raw**: durable, minimally transformed vendor extracts. Raw data is retained
   independently of downstream filters and should not be deleted as routine
   cache cleanup.
2. **Intermediate**: reproducible per-stage cache artifacts. Intermediate data
   may be deleted safely after a dataset has been published and must be
   rebuildable from raw data and configuration.
3. **Dataset**: curated, validated, immutable research input. For SANOS this
   includes the nearly complete filtered option history after no-arbitrage
   processing together with the smooth SANOS fit and diagnostics. A published
   dataset must not depend on intermediate artifacts remaining present.
4. **Experiment**: project- or model-specific materialization derived from a
   dataset, usually HDF5 for efficient training. Experiments are disposable once
   a project is closed, provided their dataset and configuration remain.

Do not use `derived` or `processing` as a fifth storage class. Legacy names may
be accepted as compatibility aliases, but new interfaces use `intermediate`.

## Dataset distribution classification

- Every published dataset manifest must declare one immutable distribution
  status: `internal`, `public_candidate`, or `public`.
- `public_candidate` means that the dataset has passed the repository's public
  schema and leakage checks; it does not by itself grant redistribution rights.
- A dataset may be marked `public` only when its manifest also records the
  release license, the basis for redistribution rights, the approver, approval
  date, and a durable reference to the approval or governing terms.
- Never infer redistribution rights merely because values are normalized,
  filtered, aggregated, fitted, or otherwise transformed from vendor data.
- Public release is a separate materialization with its own dataset identity,
  manifest, checksums, README, and license. Do not expose an internal dataset in
  place or make a public alias point directly at internal artifacts.
- Build the complete internal WIDB dataset first, using the same WRDS ingestion,
  filtering, no-arbitrage, and SANOS fitting semantics as the reference
  implementation. Generate the public dataset only as a deliberately lossy
  projection from that completed internal dataset.
- The public projection must not contain fields from which the original WRDS
  observations can be reconstructed. It must be independently usable without
  WRDS credentials, raw data, intermediate data, or access to the internal
  dataset.
- The public exporter uses an explicit allowlist. It must reject unexpected
  columns and must not include raw bids, asks, mid prices, spot levels, volumes,
  open interest, option identifiers or symbols, vendor Greeks, vendor implied
  volatilities, security identifiers, or other unapproved source fields.
- The intended public SANOS research projection is one canonically ordered,
  tagged long-form Polars frame with scalar columns only. Its `day`, `expiry`,
  `book`, `smooth_node`, and `dlv_node` rows contain only:

  - date and trading-session half-day indicator;
  - discounted driftless spot return into the date;
  - the complete smooth SANOS parameter state and its interpreting geometry;
  - the complete unsmoothed/discrete DLV parameter state and its interpreting
    geometry;
  - normalized strike, expiry association, call/put side, and driftless
    half-spread in `C/(DF*F)` units for every listed book surviving the declared
    filter;
  - six model-derived per-book Black IVs: SANOS and DLV fit IV plus synthetic
    fit-minus-half-spread and fit-plus-half-spread IVs; all six are null when
    model time is zero or the corresponding bound is not identifiable;
  - SANOS and DLV daily market-quality triples derived privately from the
    omitted market quotes: percentage of books outside and conditional mean
    and median excess measured in market half-spreads;
  - expiry date, business/calendar DTE as defined by the schema, and AM/PM
    settlement indicator for every retained expiry.

  Model-derived IVs are not vendor or observed-market IVs. Do not publish
  source quotes, per-book market excesses, vendor IVs, or vendor Greeks.
  Canonical public and DySANOS 1.1 book populations exclude both zero-business-
  day PM expiries and the 0.2-business-day AM node; expiry geometry and complete
  model state remain published even when that expiry contributes no books.

- Do not include an absolute spot level in the public projection. SANOS/DLV
  states, normalized strikes and spreads, and driftless returns contain
  the information needed by the advertised normalized experiments. Consumers
  needing an absolute spot series must join an independently sourced series.
- Keep the trading-session half-day indicator distinct from an option expiry's
  AM/PM settlement indicator.
- Define every normalization convention precisely in the schema and README,
  including whether strike and price fields are in spot, forward, or discounted
  forward units.
- Convert private bids and asks into driftless `C/(DF*F)` units before the
  public boundary. SANOS/DLV prices and public half-spreads then share one
  coordinate system. Do not publish or later reconstruct a forward/spot unit
  bridge; private carry may be used only while computing the published
  driftless return.
- Record capability flags in the public manifest. The intended projection
  supports normalized SANOS/DLV dynamics and normalized spread experiments but
  does not claim to reproduce exact vendor books, quote-containment tests,
  volume/open-interest models, absolute-dollar tick effects, or contract-level
  trading replay.
- Test the public-export boundary explicitly: schema allowlisting, forbidden
  column detection, absence of source identifiers and absolute levels, isolated
  loading with WRDS disabled, and successful execution of every experiment
  capability advertised by the public manifest.

## Data locations and configuration

- Use the `QUANT_DATA_` namespace for new location variables:

  - `QUANT_DATA_RAW`
  - `QUANT_DATA_INTERMEDIATE`
  - `QUANT_DATA_DATASETS`
  - `QUANT_DATA_EXPERIMENTS`

- Resolve locations in this precedence order:

  1. explicit CLI option;
  2. `QUANT_DATA_*` environment variable;
  3. repository-local ignored configuration file;
  4. user configuration under `~/.config/quant_data/`;
  5. documented default derived from a general root, if available.

- Support existing unambiguous `QUANTDATA_*` variables as read-only fallback
  aliases. New names win. Warn when new and legacy variables resolve to
  different paths.
- Keep machine paths and credentials out of checked-in dataset configurations.
- Keep WRDS credentials in the existing WRDS credential mechanism. Never write
  credentials to logs, Markdown, manifests, or generated configuration files.
- Represent the AM-settlement overnight variance assumption explicitly as
  `overnight_variance_fraction`. Its default is `0.2` for parity with the
  migration implementation. Include it in canonical configuration and artifact
  identity, and do not change the default without an approved semantic change.
- Treat the overnight fraction as a clock input, not merely a displayed DTE.
  Trace and test its effect on AM expiry ordering, NMS and eligibility, variance
  interpolation, calendar constraints, DLV time increments, and experiment
  horizons. It must not change PM expiries.

## Formats and artifact layout

- Store raw WRDS table extracts as Parquet, partitioned by natural vendor keys
  such as source table, year, security identifier, and date. Preserve source
  provenance and document columns and transformations.
- Plan and download complete boundary dependencies from WRDS. A requested start
  date includes the fresh previous-business-day close and any earlier inputs
  required by return calculations; never fill them from a legacy cache or hide
  them from dry-run output.
- Until time-series or dataset assembly, processing is per trading day. Each
  stage consumes and produces one Polars `DataFrame` for that day and persists
  it as one Parquet artifact.
- Keep market quotes, liquidity fields, expiry information, eligibility masks,
  no-arbitrage results, smooth SANOS parameters, fitted values, and diagnostics
  in the same daily Polars frame. Do not split these into separate quote,
  expiry, and surface artifacts without a demonstrated requirement.
- A published dataset should normally be partitioned Parquet plus:

  - `README.md`: human- and agent-readable description, sources, fields,
    coverage, exclusions, and reproduction commands;
  - `config.json`: the complete canonical configuration used;
  - `manifest.json`: schema and pipeline versions, Git commit, upstream artifact
    identities, date/row counts, exclusions, and checksums.

- Experiment materializations may use HDF5 and model-specific fixed-size arrays,
  padding, masks, selections, and train/validation splits. Record the source
  dataset identity and full experiment configuration beside every HDF5 output.
- Write artifacts atomically. Never expose a partially written file as a cache
  hit or published dataset.

## Versioning and identity

- Use `@version` from `cdxcore.version` for deterministic data transformations
  and configuration classes.
- Declare relevant transformation and configuration dependencies in the
  decorator so a resolved pipeline version can be recorded.
- Bump a version when a change can alter output content or schema. Do not bump a
  data version for comments, logging, or equivalent refactors.
- Artifact identity must include canonical configuration, stage/schema version,
  and upstream artifact identity. Do not use user-facing `__str__` output as the
  sole cache key.
- Publishing a dataset is an explicit validation and materialization step, not
  merely renaming or pointing at an intermediate cache entry.

## Workflow and caching design

- The stable interface is the SANOS CLI and the documented artifact contract,
  not a workflow engine's internal database or command syntax.
- Resolve requested datasets and experiments output-first. If the requested
  artifact is complete, its files and manifest are valid, its configuration and
  schema identities match, and its recorded producer version is accepted, reuse
  it without requiring raw, intermediate, internal-dataset, or other upstream
  artifacts to exist.
- Default cache validation is shallow at a published dataset or completed
  experiment boundary. Validate the requested artifact itself; do not traverse
  and validate its full lineage unless the user explicitly requests deep
  lineage validation or the artifact must be rebuilt.
- Support `version="*"` as an explicit reuse override for an existing complete
  artifact. The wildcard skips producer-code-version comparison but does not
  skip schema, configuration identity, completeness, or artifact-integrity
  checks. It must never authorize building, replacing, or publishing an
  artifact: if no matching complete artifact exists, fail and require a concrete
  version.
- Reusing an artifact with `version="*"` must emit a visible warning and record
  the override in the command result or execution log without modifying the
  artifact's original manifest.
- Prefer a small explicit artifact store and deterministic stage functions over
  copying the current `SubDir.cache` framework wholesale.
- The initial local pipeline may use ordinary Python orchestration and bounded
  process parallelism. Introduce redun only if its recorded DAG, cross-run
  provenance, resume behavior, or executor support materially simplifies the
  implementation.
- If redun is used, keep it behind the SANOS CLI. Tasks exchange file references
  rather than large in-memory frames, and redun manages only rebuildable
  intermediate work and experiment preparation. Raw retention and dataset
  publication remain explicit SANOS policies.
- If redun is used at a dataset or experiment boundary, configure or wrap it to
  provide the shallow semantics above. Do not let redun's recursive input-file
  validation make a self-contained published artifact depend on retained
  upstream files.
- A missing intermediate artifact must be rebuildable. A missing published
  dataset must never be silently reconstructed or replaced under the same
  identity.

## Finite-DLV fit and restart policy

- Fit the bounded discrete-DLV projection with the configured call-price and
  conditioned-IV bands first. Retry without only those fit-quality bands if the
  authenticated HiGHS result is `infeasible`/`infeasible_inaccurate`, or if a
  nominally successful result fails the post-solve check of those same bands.
- The fallback keeps the identical support, density, martingale, calendar, and
  DLV-bound constraints and uses the uniform-L1 objective. A timeout, solver
  error, unbounded status, malformed event, or structural validation failure is
  fatal and never authorizes a relaxed solve.
- Treat a successfully published fitted-day artifact as immutable completed
  work. A resumed range skips it; failure or fallback on another date does not
  authorize refitting it.
- Never relabel historical fitted bytes as though the current fitter produced
  them. Any explicitly reviewed historical fit retained for Dataset assembly
  keeps its original artifact identity and must be pinned by exact artifact,
  data, manifest, producer-version, stage-version, configuration, and
  diagnostics-schema fingerprints. Preserve that acceptance evidence in the
  self-contained Dataset after Intermediate cleanup.

## Smooth-SANOS solver restart policy

- Keep serial HiGHS dual simplex as the primary solver for smooth SANOS so
  every successful primary result retains the established numerical path.
- Only the smooth-SANOS stage uses the ordered fallback chain HiGHS IPM then
  CLARABEL, and only after `solver_error`, the recognized `invalid_solution`,
  or `user_limit`. Authenticated infeasible/unbounded statuses and unexpected
  runtime errors remain fatal. Initial no-arbitrage and discrete-DLV stages
  retain their existing solver policies.
- Record every attempted solver and algorithm in order. HiGHS simplex and IPM
  use the `1e-7` calendar gate; CLARABEL uses its status-dependent feasibility
  gate. The configured timeout applies separately to each attempt.

## CLI requirements

- Provide one installed `sanos` command, also invocable as `python -m sanos`.
- At minimum, expose commands equivalent to:

  - `sanos doctor`
  - `sanos raw download`
  - `sanos intermediate build`
  - `sanos dataset build`
  - `sanos dataset validate`
  - `sanos dysanos build` (with `sanos experiment build` as a temporary alias)
  - `sanos status`

- Raw download commands must support explicit source/ticker/date selection,
  cache-only validation, resumability, and an explicit refresh option.
- A live Raw cache miss may open WRDS without an interactive confirmation or
  special CLI flag. Guard every connection open with a persistent,
  account-scoped, cross-process attempt monitor: serialize opens, retain failed
  or interrupted attempts for 15 minutes, and reject a fourth attempt in that
  rolling window before network access. A successful connection clears the
  account's failure window. Never persist a username or credential. A dry run
  or cache-only validation must never read credentials, create monitor state,
  or open a WRDS connection.
- Bound WRDS access to at most four concurrent connections across SANOS
  processes on the host, use a more conservative default, and do not retry
  authentication or 2FA failures automatically. Reject a requested limit above
  four before connecting. Use an account-scoped semaphore independent of the
  configurable data roots, and close the authentication probe before fan-out or
  count it as one of the four slots.
- Permit at most three total attempts for a recognized transient query. Do not
  retry authentication, 2FA, permission, connection-limit, or unknown failures
  inside a command.
- Dataset commands must accept a checked-in configuration, support restart after
  interruption, validate completeness before publication, and print the final
  dataset identity and path.
- Every material command must support `-v`/`--verbose`. Long-running commands
  must emit flushed progress containing elapsed time, completed and total work,
  failures, and estimated remaining time when meaningful. Follow the host-wide
  maximum five-minute progress interval; prefer more frequent lightweight CLI
  updates for interactive use.
- Also provide a stable structured-output mode, such as `--log-format json`, so
  agents and automation can inspect progress and errors without scraping prose.
- Provide `--dry-run` where useful. Destructive cleanup or replacement must be
  explicit and must identify the exact affected artifacts.

## Data sources

- The reproducible WIDB-to-SANOS stream uses WRDS OptionMetrics/IvyDB tables for
  security identity, option quotes, underlying closes, forward prices, and zero
  curves.
- Do not add FRED, Yahoo Finance, or Massive as dependencies of the core WIDB
  reproduction stream unless a new, documented requirement cannot be satisfied
  from WIDB.
- Yahoo Finance may be supported through a separate optional adapter when a
  non-European use case needs an independently sourced spot series. Fetch it on
  the consumer side; do not bundle or redistribute it in the public SANOS
  dataset, and do not make public-dataset loading depend on Yahoo availability.
- The NYSE calendar library is a software dependency, not an additional market
  data source.

## Testing and documentation

- Tests run offline by default and must never contact WRDS implicitly.
- Every test must protect a relevant contract, numerical invariant, regression,
  or user-visible workflow. Keep tests concise and pertinent; do not add cases
  merely to increase test counts or nominal coverage. Prefer compact
  parameterization and shared focused fixtures over repetitive scenarios, and
  remove redundant tests once a stronger test fully subsumes them.
- Maintain small stored or synthetic fixtures for unit and integration tests.
  Put live WRDS checks behind an explicit marker and command.
- Test configuration identity, schema validation, cache reuse, interrupted-run
  recovery, atomic writes, legacy location fallback, and dataset publication.
- Add numerical and schema parity tests against the read-only `quant_data`
  reference for the migration period.
- Document a complete non-AI workflow in the README, starting from environment
  setup and WRDS authentication and ending with dataset validation and experiment
  materialization.
