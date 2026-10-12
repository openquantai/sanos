# Internal DySANOS Experiment materialization

The reference Dataset and an Experiment are different lifecycle objects. The
freshly downloaded, validated, and promoted SANOS **Dataset** is the canonical
research reference. This command derives a disposable, internal HDF5
**Experiment** from that Dataset for the recovered all-history DySANOS
daily-front consumer. Deleting an Experiment never changes its source Dataset.

The checked-in profile is `configs/dysanos-daily-front-reference.json`. It pins
one exact internal Dataset identity and the recovered selector and model
geometry: smoothness 0.25, a 100,000-option upstream population cap, literal
0.2-business-day exclusion, inclusive NMS `[-3,2]`, 18 business-DTE model
nodes, and 200 inner strike coordinates. Its HDF5 contains exact internal
bids, asks, fitted prices, liquidity fields, expiry identifiers, and an
absolute spot reference. It is therefore not redistributable and is not the
public SANOS Dataset. The pinned selector's lower horizon is inclusive: a
genuine zero-business-DTE book is valid when it survives all other gates;
zero-valued padding remains distinguishable through the mandatory prefix mask.

## Plan, build, and validate

Planning is read-only, does not open the source Dataset, and never contacts
WRDS:

```bash
sanos --root /path/to/quant-data --dry-run --log-format json \
  experiment build --config configs/dysanos-daily-front-reference.json
```

Build after the pinned internal Dataset exists:

```bash
sanos --root /path/to/quant-data experiment build \
  --config configs/dysanos-daily-front-reference.json \
  --max-workers 2
```

The build uses at most four local spawned processes. It writes one atomic,
pickle-free NPZ restart shard per populated date beneath the exact plan-owned
Experiment workspace, assembles and validates `data.h5`, and then publishes
the immutable artifact atomically. Progress is available as human text with
`--verbose` or structured events with `--log-format json`. This operation is
offline and never opens WRDS.

Validate the completed artifact without requiring its source:

```bash
sanos --root /path/to/quant-data experiment validate \
  --config configs/dysanos-daily-front-reference.json
```

Add `--deep-lineage` only when the source Dataset is still present and should
also be checked. `--integrity-only` checks artifact files and checksums without
reopening logical HDF5 content. Exact-ID validation is available with `--name`
and `--artifact-id`.

## Reuse and recovery

Resolution is output-first. A valid completed Experiment is reused without
requiring the source Dataset or restart workspace. `--version '*'` may reuse a
complete compatible artifact produced by another code version; it emits a
warning and can never build or replace an artifact.

Interrupted builds retain validated per-day shards and an fsynced JSONL
journal. Re-run the same command to resume. Use `--no-resume` to fail when that
workspace exists, or `--restart-work` to explicitly delete only the exact
plan-owned workspace and rebuild its shards. `--keep-work` preserves shards
after successful publication for diagnosis; otherwise they are removed.

The immutable artifact contains `data.h5`, `README.md`, `config.json`,
`schema.json`, `source.json`, `splits.json`, `validation.json`, the generic
`manifest.json`, and its completion marker. HDF5 masks/counts define valid
prefixes; all padding, dtypes, storage filters, source identity, reconstructed
market fingerprints, consumer training fingerprints, weekly pairs, and logical
content hashes are validation gates. The stored `market_fingerprint` binds a
day to the exact SANOS Dataset identity; the adapter's `training_fingerprint`
separately reproduces the pinned objective-only FLEX formula.

## Legacy parity and fresh production

The 891-date legacy Joint Direct-200 artifact is a read-only migration oracle,
not a build input. It establishes the common-period population of 3,566,258
separate call/put books and the recovered tensor contract. The newly downloaded
SANOS Dataset plans 918 sessions through 2025-08-29, with one policy-owned
typed-empty day, so fresh totals are expected to differ. Production promotion
must explain every common-period discrepancy and record the new Dataset
identity as the canonical SANOS reference.
