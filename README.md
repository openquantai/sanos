# SANOS

SANOS is a standalone implementation of the smooth arbitrage-free normalized
option surface model, together with a reproducible WIDB/WRDS data pipeline and
research-dataset tooling.

The repository is being migrated from the read-only `quant_data` reference.
The implementation follows [PROGRAM_PLAN.md](PROGRAM_PLAN.md). The existing
891-date SPX dataset from 2022-01-03 through 2025-08-29 is the authoritative
parity reference; the new dataset will be built from a fresh WRDS extraction.

## Current status

Phases 0 through 5 are complete with code and quantitative review. CD-020's
all-valid-day failure is preserved for parity; repairing it remains an optional
future change requiring approval. All
ordinary tests, including the guarded WRDS layer, are offline; no live WRDS
access has occurred yet. The exact 2022–2025 reference endpoint and numerical
baseline are catalogued. Dataset and experiment commands still fail with an
actionable "not implemented" message until their reviewed phases land. See
[the reference baseline](docs/reference-baseline.md) for the 891-date parity
oracle and the distinction between its legacy artifact populations.

## Development installation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
sanos --help
python -m sanos doctor
pytest
```

The raw workflow is documented in the [WRDS runbook](docs/wrds-runbook.md).
Generated raw data, intermediates, datasets, experiments, logs, and temporary
files must remain outside this Git working tree.

## Repository map

- `src/sanos/`: Python package and CLI.
- `tests/`: offline tests; ordinary tests never contact WRDS.
- `tex/`: paper sources and figures imported from the original repository.
- `reference/quant_data/`: exact, non-installed migration snapshot with filtered
  source history; never a runtime or data dependency.
- `configs/`: checked-in portable dataset and experiment configurations.
- `docs/`: architecture, data contracts, compatibility decisions, and runbooks.
- `PORTING.md`: source revisions and migration provenance.

## Data locations

The CLI resolves the four storage roots in this order: explicit CLI option,
specific `QUANT_DATA_*` environment variable, ignored repository-local config,
user config under `~/.config/quant_data/`, then a documented `QUANT_DATA_ROOT`
default. The four classes are raw, intermediate, dataset, and experiment. Run
`sanos config show` to inspect the result without creating directories. See
[configuration](docs/configuration.md) and [artifact storage](docs/artifacts.md)
for the contracts implemented in Phase 2, and
[model numerics](docs/model-numerics.md) for the reviewed Phase 3 equations,
units, solver behavior, and compatibility gates.

## Offline raw planning

This command resolves trading sessions, the previous-session boundary inputs,
and existing immutable artifacts. It never reads credentials, connects to WRDS,
or creates the configured roots:

```bash
sanos --root /path/to/quant-data --dry-run raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29
```

Do not pass `--confirm-2fa-ready` until you are actually ready for an immediate
live WRDS authentication. See the runbook for cache-only validation, live
downloads, immutable refresh generations, structured logs, and connection
safety.

## Offline daily intermediates

After the required Raw partitions exist, build one immutable Polars/Parquet
frame per day. This command never opens WRDS; a Raw miss fails with the exact
download prerequisite:

```bash
sanos --root /path/to/quant-data intermediate build \
  --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2022-01-03 --end-date 2025-08-29 \
  --through enriched-day --exclude-half-days --max-workers 4
```

Use the global `--dry-run` before material work and `--log-format json` for
automation. The default includes half days, matching the recovered general
pipeline; the 891-date reference compatibility run explicitly excludes them.
Each artifact contains `data.parquet`, `config.json`, a small `README.md`, and
the checksummed manifest/completion records. Re-running is output-first and
shallow: a valid terminal artifact is reused without requiring Raw ancestors.
`--version '*'` is reuse-only and fails if no matching complete output exists.

See [AGENTS.md](AGENTS.md) for binding implementation rules and
[PROGRAM_PLAN.md](PROGRAM_PLAN.md) for phased acceptance criteria.
