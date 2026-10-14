# SANOS

SANOS is a standalone implementation of the smooth arbitrage-free normalized
option surface model, together with a reproducible WIDB/WRDS data pipeline and
research-dataset tooling.

The repository is being migrated from the read-only `quant_data` implementation.
The implementation follows [PROGRAM_PLAN.md](PROGRAM_PLAN.md). The existing
891-date SPX dataset from 2022-01-03 through 2025-08-29 is a migration oracle,
not the final authority. The newly downloaded, validated, and published SANOS
dataset will be the canonical reference dataset.

## Current status

Phases 0 through 9 are complete with code and quantitative review. Phase 10's
offline safety prerequisites and Phase 11's frozen offline production tooling
are implemented. The daily no-arbitrage, filtering, smooth-SANOS, and canonical
finite-DLV projection stages are available through the CLI, together with
output-first internal Dataset build and validation commands. The initial
no-arbitrage state remains audit state and is not presented as a finite DLV
model. CD-020's all-valid-day failure is preserved for parity; repairing it
remains an optional future change requiring approval. Ordinary tests do not
contact WRDS. The exact 2022–2025 legacy comparison endpoint and numerical
baseline are catalogued. A technically validated projection remains
`public_candidate` until separate redistribution approval is recorded. See
[the legacy baseline](docs/reference-baseline.md) for the 891-date parity oracle
and the distinction between its legacy artifact populations. Follow the
[guarded Phase 10 WRDS pilot](docs/phase-10-pilot.md) before any full-range
extraction, then use the [fresh production runbook](docs/production-runbook.md).
The [human operator guide](docs/operator-guide.md) covers installation, exit
codes, troubleshooting, restart, and retention without requiring an AI agent.

## Development installation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,operator]'
sanos --help
python -m sanos doctor
pytest
```

For the guarded Phase 11 production workflow, resolve the tested CPython 3.12
operator/developer stack rather than whatever versions happen to be newest:

```bash
python -m pip install -c constraints/phase11-tested-py312.txt -e '.[dev,operator]'
```

This constraints file is not a universal lock. The production runtime
fingerprint remains the authoritative record of its closed numerical package
set, CVXPY solvers, Python/host identity, numerical libraries, CPU availability,
and thread settings.

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
- `constraints/`: reviewed version constraints for guarded production runs.
- `docs/`: architecture, data contracts, compatibility decisions, and runbooks.
- `PORTING.md`: source revisions and migration provenance.

## Data locations

The CLI resolves the four storage roots in this order: explicit CLI option,
specific `QUANT_DATA_*` environment variable, ignored repository-local config,
user config under `~/.config/quant_data/`, then a documented `QUANT_DATA_ROOT`
default. The four classes are raw, intermediate, dataset, and experiment. Run
`sanos config show` to inspect the result without creating directories, and
`sanos config resources` to locate packaged reference assets and source
provenance. See
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

A live cache miss needs no confirmation flag. SANOS serializes connection opens
through an account-scoped monitor, records failed or interrupted attempts for
15 minutes, and blocks a fourth attempt before network access. See the runbook
for cache-only validation, live downloads, immutable refresh generations,
structured logs, and connection safety.

## Offline daily intermediates

After the required Raw partitions exist, build one immutable Polars/Parquet
frame per day. This command never opens WRDS; a Raw miss fails with the exact
download prerequisite:

```bash
sanos --root /path/to/quant-data intermediate build \
  --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2022-01-03 --end-date 2025-08-29 \
  --through fitted-day --exclude-half-days --max-workers 4
```

Use the global `--dry-run` before material work and `--log-format json` for
automation. The reference production profile includes half days; the legacy
oracle has 891 populated common dates plus 26 declared populated-date gaps and
one policy-empty day within the 918-session fresh schedule.
Each artifact contains `data.parquet`, `config.json`, a small `README.md`, and
the checksummed manifest/completion records. Re-running is output-first and
shallow: a valid terminal artifact is reused without requiring Raw ancestors.
`--version '*'` is reuse-only and fails if no matching complete output exists.
The available terminal stages are `paired-day`, `enriched-day`, `no-arb-day`,
`filtered-day`, and `fitted-day`. The final frame retains rejected rows with
eligibility/reason fields and stores the pre-filter no-arbitrage audit state,
the post-filter smooth 0.25 SANOS state, and a separately fitted 200-inner-node
finite-DLV projection of that smooth state as typed list columns. The DLV fit
tries the strict call/IV bands first and, only when HiGHS reports the strict
problem as `infeasible` or `infeasible_inaccurate`,
returns the same max-DLV-3 model's uniform-L1 closest fit with explicit quality
diagnostics.

See [AGENTS.md](AGENTS.md) for binding implementation rules and
[PROGRAM_PLAN.md](PROGRAM_PLAN.md) for phased acceptance criteria.

## Read-only legacy comparison

After a fresh fitted build, compare existing daily artifacts with the pinned
DySANOS HDF5 oracle without rebuilding or opening WRDS:

```bash
env WRDS_OFF=1 sanos -v --root /path/to/quant-data --log-format json \
  reference compare-daily --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2024-12-23 --end-date 2025-01-16 \
  --source-generation <fresh-generation> \
  --legacy-root <read-only-legacy-SPX-directory> \
  --expected-oracle-gap 2024-12-24 \
  --phase10-pilot-contract \
  --output /external/sanos-evidence/daily-comparison.json
```

The command validates fitted artifacts shallowly, checks the pinned catalog and
each selected legacy checksum, reports every structural and numerical
difference, and exits 4 for an unexplained discrepancy. Reports contain only
artifact identities, summary metrics, classifications, digests, and first
mismatch keys—not raw vendor rows or values. See the
[Phase 10 pilot runbook](docs/phase-10-pilot.md) for the mandatory live gate.

If an independently reviewed comparison is blocked only by exact populated
migration discrepancies, `sanos reference approval-document` can materialize a
separate owner decision without hand-copying field lists or digests. It reopens
both fresh and legacy evidence, recomputes each selected comparison, requires
an explicit owner-attestation flag, and is unrelated to public-redistribution
approval.

## Internal Dataset publication

Build the complete internal research Dataset offline from fresh Raw artifacts:

```bash
sanos --root /path/to/quant-data --dry-run dataset build \
  --config configs/widb-spx-reference.json
sanos --root /path/to/quant-data dataset build \
  --config configs/widb-spx-reference.json --max-workers 2
sanos --root /path/to/quant-data dataset validate \
  --config configs/widb-spx-reference.json
```

The default reference profile plans all 918 NYSE sessions from 2022-01-03
through 2025-08-29, including half days; the pinned 2025-01-08 bad-data entry is
an exact-schema typed-empty partition. Dataset publication copies each complete
daily fitted frame losslessly and adds portable schema, partition, lineage,
validation, legacy-comparison, and configuration evidence.

The resulting Dataset loads and validates without Raw or Intermediate files.
Pass `--deep-lineage` before cleanup when you also want to validate all recorded
fitted and Raw ancestors. `--version '*'` can reuse a complete compatible
Dataset but cannot build one. See the full
[internal Dataset runbook](docs/internal-datasets.md).

## Public-candidate Dataset projection

Project the complete internal Dataset into the one-row-per-day normalized
SANOS/DLV research contract:

```bash
sanos --root /path/to/quant-data --dry-run dataset public build \
  --config configs/sanos-public-candidate-reference.json
sanos --root /path/to/quant-data dataset public build \
  --config configs/sanos-public-candidate-reference.json
sanos --root /path/to/quant-data dataset public validate \
  --config configs/sanos-public-candidate-reference.json
```

The output is a self-contained `public_candidate`, not a redistribution
approval. It contains normalized returns, SANOS/DLV state and geometry, and
spot-one half-spreads, while excluding absolute levels, source identifiers,
vendor analytics, liquidity fields, and raw books. Output-first reuse and
shallow validation do not need the internal Dataset to remain present. See the
[public Dataset runbook](docs/public-datasets.md) for the exact schema, units,
capabilities, leakage boundary, wildcard behavior, and separate
`approval-template`/`approve` rights-gated release workflow. Shallow validation
checks public-only integrity and capabilities; `--deep-lineage` reprojects from
the exact internal source to re-establish source-bound values.

## Internal Experiment materialization

Once the pinned internal Dataset exists, build the disposable all-history
DySANOS daily-front HDF5 input:

```bash
sanos --root /path/to/quant-data --dry-run experiment build \
  --config configs/dysanos-daily-front-reference.json
sanos --root /path/to/quant-data experiment build \
  --config configs/dysanos-daily-front-reference.json --max-workers 2
sanos --root /path/to/quant-data experiment validate \
  --config configs/dysanos-daily-front-reference.json
```

This internal Experiment contains vendor-derived books and an absolute spot
reference; it is not the public projection and not the canonical Dataset.
Completed Experiments are output-first and self-contained, so shallow reuse and
validation do not require the source Dataset. See the
[Experiment runbook](docs/experiments.md) for recovery, wildcard reuse, schema,
and legacy-parity details.

## Canonical reference designation

After the complete fresh Dataset and Experiment pass the full-range daily and
consumer comparisons, independent code and quantitative reviews are bound into
an immutable designation with `sanos dataset reference`. The designation is
self-contained, retains the exact numerical-runtime fingerprint reviewed with
the evidence index, is shallow-validatable without lineage, and does not alter
the internal Dataset or grant redistribution rights. Capture it with `sanos
production runtime-fingerprint`; the exact operator sequence is in the [fresh
production runbook](docs/production-runbook.md).
