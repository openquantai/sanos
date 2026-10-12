# Phase 1 evidence bundle

Review scope: reference inventory and executable parity baseline. This phase
does not implement the new data pipeline and does not access WRDS.

## Source and artifact anchors

- Pinned source revision:
  `de15be96d2f283b6c31e60d3ea47325a0c093ff0`.
- Imported source blobs were proven identical to that revision during Phase 0.
- The original checkout still reports that same HEAD. Its pre-existing dirty
  files are unrelated FLEX/hedging work; no SANOS task changed them.
- Exact reference endpoint: 2022-01-03 through 2025-08-29.
- Smooth collection: 892 physical HDF5 files, 891 populated days, sole empty
  marker 2025-01-08, collection SHA-256
  `f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280`.
- Exact downstream daily-front report: 891 dates and 3,566,258 books.
- Exact daily-front producer:
  `quantdata/dysanos/flex/run_all_history_joint_direct.py`, pinned Git blob
  `6b1c408c3ab1d32e92115cb95c91b3472bcf6086`.
- Target payload: schema `flex-all-history-joint-direct-targets-v1`, float64
  state shape `(891, 200)`, float64 spot-reference shape `(891,)`, exact date
  sequence equality with the populated smooth catalog, and state SHA-256
  `3525f0d28a7fdc9a5535c80d05c9a4ddd83dfb9c59e8b15fd6ffcfa568477cb7`.
- Separate Daily25 SDLV/JAX NPZ: 891 dates and 2,192,458 books.

Machine-readable hashes, byte sizes and symbolic locators are in
`reference-artifacts.json`. The complete per-HDF5 inventory is
`reference-market-catalog.json`.

## Scoped implementation

- `src/sanos/reference/legacy_market.py` is a read-only decoder/cataloguer for
  the exact cdxcore `PYTREE_HDF5` format. It opens inputs only with mode `r`,
  applies one strict versioned schema/geometry validator for both catalog and
  day loading, recognizes empty markers, hashes every
  file, and exposes no write API.
- `scripts/catalog_reference_market.py` reproduces the catalog with flushed
  progress and an atomic output write. It rejects an output equal to or beneath
  the resolved reference root, including symlink aliases, before creating any
  output path.
- `tests/reference_probe.py` imports the pinned snapshot in a subprocess with
  `WRDS_OFF=1` and emits deterministic density/DLV/price/repair results.
- `tests/fixtures/reference_numerics_v1.json` is the frozen numerical oracle.
- `tests/test_reference_inventory.py` prevents endpoint/population/catalog
  drift and explicitly distinguishes all three legacy populations.
- `docs/reference-baseline.md` records the recovered path, defaults, filters,
  clock and units.
- `docs/traceability-matrix.md` maps every imported module/test and every
  required production symbol/configuration to copy/adapt/replace/omit.
- `docs/data-contracts.md` freezes the single-daily-Polars-frame and public
  one-row-per-day logical contracts.
- `docs/compatibility-decisions.md` records confirmed defects and proposed
  semantic choices without authorizing a change.
- `reviews/phase-1/reference-runtime-baseline.json` freezes the evidence
  runtime, transform/schema versions, exact production defaults, solver
  choices/options/tolerances, schema fields and float32 cast boundaries. It
  explicitly records that the original artifact-generation environment is
  unknown.
- `reviews/phase-1/reference-artifacts.json` now pins the exact downstream
  producer and dependencies, payload schemas/shapes/dtypes, date correspondence
  and neural-state role as an experiment parity oracle rather than dataset
  SANOS/DLV state.

## Verification transcript

All commands ran from `/home/hans/python/sanos` on 2026-10-12.

```text
env PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  TMPDIR=/tmp/sanos/phase1-final python -m pytest -q -p no:cacheprovider
.....................                                                    [100%]
21 passed in 2.27s

ruff check --no-cache src tests scripts
All checks passed!

mypy --python-executable /home/hans/vqdata/bin/python
Success: no issues found in 21 source files

python -m compileall -q src tests scripts
[exit 0]

git --git-dir=.git-real diff --check
[exit 0]
```

The pinned snapshot also remains executable offline:

```text
test_biz_calendar.py + test_arbfree_density_projection.py
13 passed in 1.26s

test_arbfree.py -k make_extended_strikes
2 passed, 2 deselected in 1.13s
```

After the strict-validator changes, full catalog generation emitted progress
every 50 files and completed as:

```text
catalogued 892/892 files; elapsed=59.0s; eta=0.0s; current=2025-08-29.h5
wrote /tmp/sanos/phase1-strict/reference-market-catalog.json: files=892;
populated=891;
sha256=f25afda76474a48f25da27be6de113b9fbee648e7ab01691c6367dafe723e280
```

The regenerated catalog is byte-for-byte equal to the checked-in catalog; both
have file SHA-256
`9b68a9638f273271ff4dbde6365c75ac9fd749993f9284b42428682339d516b4`.

## Reference write-safety

- The adapter has no writer and uses `h5py.File(path, "r")`.
- Synthetic tests assert that reading leaves source modification time unchanged.
- The catalog stores symbolic locators rather than making machine paths part of
  dataset configuration.
- No command connected to WRDS.
- The source checkout was inspected only with read commands and remains at the
  pinned revision with the same unrelated dirty paths recorded before this
  phase.

## Findings requiring decisions

The following were flagged before affected implementation:

- reversed WIDB option-column inclusion assertion;
- unreachable validation on the empty enriched-day path;
- legacy close-to-close return is not forward-adjusted;
- ambiguous fit-percentage wording;
- solver timeout omitted from legacy cache identity;
- explicit `secid` argument ignored by `collect_raw_day`;
- `OptionsFilter.min_vegaSqrtVar` is in configuration/identity but not applied;
- float32 daily cast points can affect eligibility/LP parity; and
- public spread/model-price comparison needs an explicit carry bridge;
- the ndarray/vector `_compute_market_fit` branch references undefined local
  names, while the production list branch is correct; and
- legacy pricing jumps to the first density before the first fitted expiry
  rather than interpolating from a time-zero Dirac mass.

Exact evidence and expected effects are in
`docs/compatibility-decisions.md`. Phase 2 configuration/artifact work does not
depend on resolving these semantics.

## Review questions

### Code reviewer

1. Is the reference reader demonstrably read-only, deterministic and strict
   enough to serve as a parity oracle?
2. Does the traceability matrix cover the production path and correctly isolate
   historical/experiment code, including the exact downstream fitted-state
   producer?
3. Are errors, type contracts, test isolation and generated-catalog provenance
   adequate to approve Phase 1?

### Quantitative reviewer

1. Are the 4,651,416, 3,566,258 and 2,192,458 populations correctly separated?
2. Are normalization, filter order, AM timing, NMS/horizon, smoothness, complete
   reconstructing state and expiry-pricing contracts faithfully recovered?
3. Are every unapproved formula/unit/semantic choice clearly blocked rather
   than embedded as reference behavior?

Approval means Phase 2 may begin. It does not approve any `open` compatibility
decision and does not authorize WRDS access.
