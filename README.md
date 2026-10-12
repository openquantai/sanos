# SANOS

SANOS is a standalone implementation of the smooth arbitrage-free normalized
option surface model, together with a reproducible WIDB/WRDS data pipeline and
research-dataset tooling.

The repository is being migrated from the read-only `quant_data` reference.
The implementation follows [PROGRAM_PLAN.md](PROGRAM_PLAN.md). The existing
891-date SPX dataset from 2022-01-03 through 2025-08-29 is the authoritative
parity reference; the new dataset will be built from a fresh WRDS extraction.

## Current status

Phases 0, 1, and 2 are complete with code and quantitative review. Phase 3's
SANOS and no-arbitrage numerical migration is next. The exact 2022–2025
reference endpoint and numerical baseline are catalogued, but material data-pipeline
commands intentionally fail with an actionable "not implemented" message until
their reviewed phases land. No command connects to WRDS yet. See
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

The normal user and live-WRDS workflow will be documented before the associated
commands are enabled. Generated raw data, intermediates, datasets, experiments,
logs, and temporary files must remain outside this Git working tree.

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
for the contracts implemented in Phase 2.

See [AGENTS.md](AGENTS.md) for binding implementation rules and
[PROGRAM_PLAN.md](PROGRAM_PLAN.md) for phased acceptance criteria.
