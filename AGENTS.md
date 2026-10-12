# SANOS repository instructions

## Scope and migration

- This repository owns the standalone SANOS Python implementation, its data
  pipeline, analytics, tests, documentation, and LaTeX paper sources.
- Treat `~/python/quant_data` as a read-only reference implementation. Copy and
  adapt required code into this repository, preferring proven implementation
  over a clean-room rewrite; do not modify `quant_data` as part of SANOS work
  unless the user explicitly requests a separate compatibility change there.
- Preserve numerical and data parity with the reference implementation through
  explicit parity tests before removing or redesigning behavior.
- Preserve current semantics by default. Record suspected defects or desirable
  semantic changes, their evidence, and their expected output impact, and obtain
  user approval before implementing them. Do not silently preserve or fix a
  suspected defect while migrating code.
- Rebuild the target 2022--2025 dataset from newly downloaded WRDS extracts.
  Existing raw, processing, training, or experiment artifacts from
  `quant_data` must not be inputs to the new published dataset. The existing
  891-date fitted 2022--2025 dataset is the authoritative parity reference:
  catalogue it read-only and treat unexplained differences in dates,
  populations, masks, SANOS/DLV values, or experiment tensors as blocking
  discrepancies requiring review and user approval.
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
- The intended public SANOS research projection has one Polars row per trading
  day and may use typed list/struct columns. It contains only:

  - date and trading-session half-day indicator;
  - forward-adjusted spot return into the date;
  - the complete smooth SANOS parameter state and its interpreting geometry;
  - the complete unsmoothed/discrete DLV parameter state and its interpreting
    geometry;
  - normalized strike, expiry association, call/put side, and spot-normalized
    half-spread for every listed book surviving the declared filter;
  - expiry date, business/calendar DTE as defined by the schema, and AM/PM
    settlement indicator for every retained expiry.

- Do not include an absolute spot level in the public projection. SANOS/DLV
  states, normalized strikes and spreads, and forward-adjusted returns contain
  the information needed by the advertised normalized experiments. Consumers
  needing an absolute spot series must join an independently sourced series.
- Keep the trading-session half-day indicator distinct from an option expiry's
  AM/PM settlement indicator.
- Define every normalization convention precisely in the schema and README,
  including whether strike and price fields are in spot, forward, or discounted
  forward units.
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
  reference implementation. Include it in canonical configuration and artifact
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

## CLI requirements

- Provide one installed `sanos` command, also invocable as `python -m sanos`.
- At minimum, expose commands equivalent to:

  - `sanos doctor`
  - `sanos raw download`
  - `sanos intermediate build`
  - `sanos dataset build`
  - `sanos dataset validate`
  - `sanos experiment build`
  - `sanos status`

- Raw download commands must support explicit source/ticker/date selection,
  cache-only validation, resumability, and an explicit refresh option.
- Before any command opens a WRDS connection, require an interactive
  confirmation that the user is ready to complete 2FA; non-interactive callers
  must pass an explicit equivalent flag. A dry run or cache-only validation
  must never open a WRDS connection.
- Bound WRDS access to at most four concurrent connections across SANOS
  processes on the host, use a more conservative default, and do not retry
  authentication or 2FA failures automatically. Reject a requested limit above
  four before connecting. Use an account-scoped semaphore independent of the
  configurable data roots, and close the authentication probe before fan-out or
  count it as one of the four slots.
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
- Maintain small stored or synthetic fixtures for unit and integration tests.
  Put live WRDS checks behind an explicit marker and command.
- Test configuration identity, schema validation, cache reuse, interrupted-run
  recovery, atomic writes, legacy location fallback, and dataset publication.
- Add numerical and schema parity tests against the read-only `quant_data`
  reference for the migration period.
- Document a complete non-AI workflow in the README, starting from environment
  setup and WRDS authentication and ending with dataset validation and experiment
  materialization.
