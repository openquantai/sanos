# DySANOS 1.1 model

DySANOS is a downstream model under `sanos.models.dysanos`; SANOS itself ends
at the fitted internal/public Dataset boundary. Other retained research models
belong beside DySANOS under `sanos.models`, each with its own Dataset capability
contract, configuration, CLI, artifacts, and tests.

The package deliberately contains one canonical model path rather than the
legacy `dysanos.flex` research forest. It owns the public-Dataset adapter,
model-specific HDF5, constrained differentiable SANOS decoder, synthetic-band
objective, fresh native-state calibration, Float64 affine Direct-200
initialization, stable weekly dynamics, joint training, checkpoints, and
validation. It never imports `quant_data` or consumes a legacy checkpoint at
runtime. Alternative Q/OCE, hedging, liquidity, encoder, hierarchical,
fixed-grid, plotting, and presentation experiments are not part of DySANOS
1.1.

The checked model configuration is all-history with no holdout. Its 2025 date
is a diagnostic boundary, not an out-of-sample claim.

The completed DySANOS lifecycle uses two immutable Experiment artifacts. The
currently implemented first artifact is the reusable
public-derived daily-front `data.h5` materialization documented below. The
pending second artifact is the trained Direct-200 model and will depend on that
HDF5 Experiment.
Keeping them separate permits multiple training runs to reuse the same input
materialization and preserves a valid HDF5 artifact when training is
interrupted or fails. Both remain disposable; neither replaces the public
Dataset as the SANOS reference.

The reference Dataset and an Experiment are different lifecycle objects. The
freshly downloaded, validated, and promoted SANOS **Dataset** is the canonical
research reference. DySANOS 1.1 derives a disposable HDF5 **Experiment** only
from the completed public Dataset for the all-history DySANOS
daily-front consumer. Deleting an Experiment never changes its source Dataset.

The checked-in profile is `configs/dysanos-daily-front-reference.json`. It
documents the expected public Dataset identity and the recovered selector and
model geometry: smoothness 0.25, literal zero- and
0.2-business-day exclusions, inclusive NMS `[-3,2]`, 18 business-DTE model
nodes, 200 inner strike coordinates, and a distinct 0.125-business-day
regularization time for the learned model's shared support. The public
Dataset's 1-business-day NMS field selects the ATM-volatility proxy and is not
this model support epsilon. Its HDF5 stores smooth fitted prices
and published driftless half-spreads in `C/(DF*F)` units. Synthetic bid and ask
are always derived as fit minus/plus half-spread; a negative synthetic bid is
valid and is never clipped. The HDF5 stores no absolute spot, volume, open
interest, observed vendor bid/ask, or vendor expiry identifier. Zero-valued
padding remains distinguishable through the mandatory prefix mask; it does not
represent a retained 0DTE book.

## Plan, build, and validate

Planning is read-only, does not open the source Dataset, and never contacts
WRDS:

```bash
sanos --root /path/to/quant-data --dry-run --log-format json \
  dysanos build --config configs/dysanos-daily-front-reference.json
```

Build after the pinned public Dataset exists. Raw, Intermediate, the internal
Dataset, and WRDS are not consulted:

```bash
sanos --root /path/to/quant-data dysanos build \
  --config configs/dysanos-daily-front-reference.json \
  --max-workers 2
```

For a fresh canonical production run, first use `sanos production
finalize-configs` as shown in the
[production runbook](production-runbook.md), then pass the emitted
`final-configs/experiment.json`. This binds the actual Dataset pointer and
configuration digest without manual JSON editing.

The build writes one atomic, pickle-free NPZ restart shard per populated date
beneath the exact plan-owned
Experiment workspace, assembles and validates `data.h5`, and then publishes
the immutable artifact atomically. Progress is available as human text with
`--verbose` or structured events with `--log-format json`. This operation is
offline and never opens WRDS.

Validate the completed artifact without requiring its source:

```bash
sanos --root /path/to/quant-data dysanos validate \
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
day to the exact public SANOS Dataset identity. The day return has an explicit
validity mask because the first published day has no preceding return.
`market_uid` and `market_fingerprint` are derived SHA-256 integrity digests,
not vendor identifiers or model features; training inputs are the documented
quote, price, expiry, model, return, split, and dynamics arrays.

## Legacy parity and fresh production

The internal-derived DySANOS 1.0 artifact is a read-only migration oracle,
not a build input. It establishes the common-period population of 3,566,258
separate call/put books and the recovered tensor contract. The newly downloaded
SANOS Dataset plans 918 sessions through 2025-08-29, with one policy-owned
typed-empty day, so fresh totals are expected to differ. Production promotion
must explain every common-period discrepancy and record the new Dataset
identity as the canonical SANOS reference.

Run the full logical and recovered-consumer comparison with:

```bash
env WRDS_OFF=1 sanos -v --root /path/to/quant-data --log-format json \
  reference compare-experiment \
  --config configs/dysanos-daily-front-reference.json \
  --legacy-root /read-only/legacy/SPX \
  --legacy-code-root /read-only/quant_data \
  --phase11-production-contract \
  --output /external/sanos-evidence/experiment-comparison.json
```

The command hashes the complete pinned local import closure of the legacy
selector and Torch consumer before loading it in an isolated subprocess, and
records the Python, NumPy, and Torch runtime identities. It compares all 891 common days,
including logical arrays, dtypes, geometry, quote identities, day tensors, and
padded batch tensors. It is read-only and runs with WRDS disabled.
