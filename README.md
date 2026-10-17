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

Phases 0 through 9A are complete with code and quantitative review. Phase 9B's
fresh Direct-200 numerical and training core is complete; publication of the
trained-model Experiment and its `train`/`evaluate` CLI remains pending. Phase 10's
offline safety prerequisites and Phase 11's frozen offline production tooling
are implemented. The daily initial full no-arbitrage DLV, filtering, and
independent smooth-SANOS stages are available through the CLI, together with
output-first internal Dataset build and validation commands. The unsmoothed DLV
state is stored as exact support and probability masses; finite squared-DLV
coefficients are not required. CD-020's all-valid-day failure is preserved for parity; repairing it
remains an optional future change requiring approval. Ordinary tests do not
contact WRDS. The exact 2022–2025 legacy comparison endpoint and numerical
baseline are catalogued. The 476-test offline suite and Python CI pass; paper
CI remains unresolved and is not part of the current pass claim.

As of 2026-10-17, the pinned `0116b15` full-range run has been stopped cleanly;
all completed immutable fitted-day artifacts remain available for exact reuse.
No newly regenerated complete public candidate or public Dataset exists yet.
Publication remains gated on 918/918 successful sessions, deep validation, and
code and quantitative review. An isolated one-day old/new LP comparison
reproduced the same numerical solver failure in both implementations; it did
not establish mathematical infeasibility or a regression caused by the
corrected DLV contract. The official automatic fitted-resume workflow validates
and preserves every exact old success, then sends only absent original targets
through the ordinary current Intermediate pipeline. Operators do not enumerate
failed dates or create candidate process data. Its authenticated full-range dry
run resolved the complete 918-date plan as 362 exact original successes and 556
current replacements; dry-run mode accessed no WRDS data and wrote no output
configuration. Material resume started from pinned revision `e9a4ac6` on
2026-10-17, with two workers and `WRDS_OFF=1`; detached import, location, and
runtime preflights passed. It is running, not complete. Run and
resume instructions are in the [fresh production
runbook](docs/production-runbook.md); run evidence stays outside Git under the
operator-selected evidence root.

A technically validated projection remains `public_candidate` until separate
redistribution approval is recorded. See
[the legacy baseline](docs/reference-baseline.md) for the 891-date parity oracle
and the distinction between its legacy artifact populations. Follow the
[guarded Phase 10 WRDS pilot](docs/phase-10-pilot.md) before any full-range
extraction, then use the [fresh production runbook](docs/production-runbook.md).
The [human operator guide](docs/operator-guide.md) covers installation, exit
codes, troubleshooting, restart, and retention without requiring an AI agent.
The requested repository upload is a later release step: the exact public
artifact must be complete, validated, and redistribution-approved before its
size and host limits are reviewed and that public release is deliberately added
to Git. Raw, Intermediate, internal/private Dataset, Experiment, log, cache,
and `public_candidate` artifacts must never be checked in.

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
eligibility/reason fields and stores the initial full no-arbitrage common-surface
support/density DLV state and the post-filter smooth 0.25 SANOS state as
independent models in typed list columns. The DLV state is not projected from
SANOS, and no SANOS/DLV price equality is required.

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

Project the complete internal Dataset into the flat scalar, driftless
SANOS/DLV research contract:

```bash
sanos --root /path/to/quant-data production finalize-configs \
  --dataset-config configs/widb-spx-reference.json \
  --public-template configs/sanos-public-candidate-reference.json \
  --experiment-template configs/dysanos-daily-front-reference.json \
  --output-directory /path/to/run-configs
sanos --root /path/to/quant-data --dry-run dataset public build \
  --config /path/to/run-configs/public-candidate.json
sanos --root /path/to/quant-data dataset public build \
  --config /path/to/run-configs/public-candidate.json
sanos --root /path/to/quant-data dataset public validate \
  --config /path/to/run-configs/public-candidate.json
```

The checked public and DySANOS JSON files are policy templates. The finalizer
resolves the internal Dataset identity from its config, derives the canonical
public-candidate identity, and binds DySANOS to that public Dataset; no artifact
ID is copied by hand.

The output is a self-contained `public_candidate`, not a redistribution
approval. It contains discounted driftless returns, SANOS/DLV state and
geometry, and half-spreads in the same `C/(DF*F)` space, while excluding
absolute levels, source identifiers,
vendor analytics, liquidity fields, and raw books. Output-first reuse and
shallow validation do not need the internal Dataset to remain present. See the
[public Dataset runbook](docs/public-datasets.md) for the exact schema, units,
capabilities, leakage boundary, wildcard behavior, and separate
`approval-template`/`approve` rights-gated release workflow. Shallow validation
checks public-only integrity and capabilities; `--deep-lineage` reprojects from
the exact internal source to re-establish source-bound values.

## Internal Experiment materialization

DySANOS is implemented as the downstream model package
`sanos.models.dysanos`; it is not part of the SANOS preprocessing pipeline.
Future retained models become sibling packages under `sanos.models` rather
than additions to a generic experiment framework.

Once the pinned public SANOS Dataset exists, build the disposable all-history
DySANOS daily-front HDF5 input:

```bash
sanos --root /path/to/quant-data --dry-run dysanos build \
  --config /path/to/run-configs/experiment.json
sanos --root /path/to/quant-data dysanos build \
  --config /path/to/run-configs/experiment.json --max-workers 2
sanos --root /path/to/quant-data dysanos validate \
  --config /path/to/run-configs/experiment.json
```

This Experiment reconstructs model-centred books from public smooth-SANOS mids
and driftless half-spreads. It does not contain vendor liquidity, absolute spot,
or vendor identifiers; the public Dataset remains the canonical source.
Completed Experiments are output-first and self-contained, so shallow reuse and
validation do not require the source Dataset. See the
[DySANOS runbook](docs/dysanos.md) for recovery, wildcard reuse, schema,
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
