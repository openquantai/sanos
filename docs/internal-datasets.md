# Internal Dataset publication

The SANOS internal Dataset is the immutable, self-contained boundary between
daily WIDB processing and research consumers. The fresh 2022--2025 publication
produced by this repository becomes the SANOS reference Dataset after the
production validation and review gate. The old 891-populated-date `quant_data`
collection remains a read-only migration/parity oracle; it is never a build
input, runtime dependency, or alternate reference Dataset.

## Build

First download Raw data using the separately guarded WRDS command. Dataset
assembly itself is offline and never opens WRDS:

```console
sanos --root /path/to/quant-data --dry-run dataset build \
  --config configs/widb-spx-reference.json

sanos --root /path/to/quant-data --log-format json dataset build \
  --config configs/widb-spx-reference.json --max-workers 2
```

The build resolves the requested Dataset first. A complete matching Dataset is
returned without checking whether its Raw data, fitted-day Intermediates, or Raw
lineage catalog still exist. On a miss it builds missing daily fitted frames
from existing Raw artifacts. It never downloads missing Raw data. Use
`--require-existing-intermediates` to prohibit even this offline rebuild.

`--version '*'` is reuse-only. It ignores only the producer-code version while
still requiring matching configuration, schema, lineage identity, completion
marker, and payload checksums. A wildcard miss fails and cannot build or
publish anything.

Publication requires an exact source revision from a clean Git checkout (or
packaged `SANOS_SOURCE_REVISION` provenance). Final output is staged privately,
validated, and exposed atomically. An existing Dataset is never replaced.

## Validate and load

Default validation is intentionally shallow and self-contained:

```console
sanos --root /path/to/quant-data dataset validate \
  --config configs/widb-spx-reference.json
```

It checks the artifact manifest and all checksums, then recomputes the daily
schema, population, masks, units, density/moment, no-arbitrage, canonical and
ordinary-Float64 SANOS/DLV round trips, smooth-to-DLV projection calls and
conditioned implied volatilities, strict/L1 solver mode, fitted-price-band
diagnostics, calendar,
and quote-containment invariants from the copied Dataset partitions. It does not visit upstream
storage and never rebuilds a missing Dataset.

Use deep validation only while lineage is retained:

```console
sanos --root /path/to/quant-data dataset validate \
  --config configs/widb-spx-reference.json --deep-lineage
```

This additionally validates the Raw catalog, all Raw artifacts, and every
recorded fitted-day chain. It is expected to fail after reproducible
Intermediates have been cleaned. Exact-ID validation is available without a
configuration file:

```console
sanos --root /path/to/quant-data dataset validate \
  --name widb-spx-2022-2025-reference --artifact-id <64-hex-id>
```

Python consumers can call `open_internal_dataset(store, pointer)`, then
`load_day(date)` or `scan()`. The loader follows `partitions.json` exactly,
disables Hive partition-column inference, and never uses a glob or upstream
artifact.

## Immutable contents

Each expected NYSE session has exactly one byte-copied fitted-day partition:

```text
data/ticker=SPX/year=YYYY/date=YYYY-MM-DD/part-00000.parquet
```

The frame retains every audit row and the exact `FITTED_SCHEMA`: quote and
liquidity inputs, expiry geometry, masks and reasons, no-arbitrage results,
the canonical finite-DLV projection, smooth SANOS state, fitted values, and
diagnostics all remain together. The pre-filter no-arbitrage state is retained
for audit but is not represented as a finite DLV model. The fitted boundary requires each
`(expiry_id, normalized_strike)` model key to be unique and strictly ordered,
as required by SANOS support geometry. Rows sharing coarser session/expiry
geometry retain their original multiplicity and stable source order;
publication does not sort, group, or deduplicate the fitted-day frame.

The Dataset also contains:

- `README.md`: human-readable source, unit, coverage, and operation contract;
- `config.json`: complete portable configuration, exact planned dates, and Raw
  role-mapping digest;
- `schema.json`: exact field/dtype contract and normalization notes;
- `partitions.json`: ordered date paths, source/copy checksums, fitted pointers,
  typed-empty evidence, and recomputed populations;
- `lineage.json`: every daily stage pointer and date/role Raw mapping, plus the
  exact fingerprint and acceptance policy for any reviewed fitted artifact
  reused without a numerical rerun;
- `validation.json`: aggregate populations and invariant residual maxima; and
- `legacy-comparison.json`: explicit migration-oracle status. The full guarded
  comparison is completed during the fresh production acceptance run.

The sole direct Dataset upstream is a rebuildable Raw-lineage-catalog
Intermediate. Its direct upstream tuple is the exact deduplicated Raw closure;
its configuration also commits to the date/role mapping so deduplication cannot
erase current-versus-previous spot roles.

The Phase 11 continuation directly reuses four completed, checksum-pinned
fitted-day artifacts for 2022-01-03 through 2022-01-06. They keep their
original immutable identities; SANOS neither rewrites their diagnostics nor
publishes them under a current-fit identity. `lineage.json`, the Dataset
manifest, and each partition record preserve their artifact, data, manifest,
producer, stage, configuration, and diagnostic-schema evidence. Shallow
validation therefore remains complete after those Intermediates are removed.

## Coverage policy

The checked-in reference profile includes all NYSE sessions from 2022-01-03
through 2025-08-29, including seven half days: 918 partitions in total. The
versioned bad-data policy requires 2025-01-08 to be one exact-schema typed-empty
partition, leaving 917 populated sessions. Unknown empty or absent days block
publication. Excluding half days is a different configuration and Dataset
identity with 911 partitions, 910 populated sessions, and the same typed-empty
day.

The 19 non-half-day sessions absent from the legacy oracle are not exclusions.
A fresh success is retained and reported as an oracle gap; a fresh failure must
be diagnosed and reviewed.

## Cleanup

Raw extracts are durable and are not routine cache material. Daily
Intermediates and the Raw lineage catalog are reproducible and may be deleted
after successful publication. Before cleanup, run deep validation; afterward,
run the default shallow validation to prove the Dataset remains independently
usable. Cleanup itself must be explicit and identify exact targets.
