# Phase 9 code implementation audit

**Reviewer:** independent code agent (`/root/code`)  
**Date:** 2026-10-12  
**Scope:** current uncommitted Phase 9 SANOS-only implementation; offline, no
WRDS/network access and no changes to `quant_data`  
**Initial verdict:** **BLOCKED / CHANGES REQUIRED**  
**Final re-review verdict:** **PASS** (see the signed re-review below)

The ordinary one-day path, immutable publication, output-first reuse, and the
latest pinned consumer adapter all work. Phase 9 is not yet safe to commit,
because the implementation has one concrete selector/object contradiction,
incomplete artifact-version ownership, a reproducible corrupt-resume failure,
and missing self-contained semantic and long-operation gates.

## Blocking findings

### 1. High — zero-DTE values selected by the reference rule are rejected by the day object

`project_daily_front` deliberately preserves the pinned inclusive lower
horizon (`0.0 <= time_y`) at
`src/sanos/experiments/daily_front.py:622`. The pinned `FlexMarketDay` likewise
accepts `expiries_y >= 0.0`. However,
`DailyFrontDay.__post_init__` rejects `expiry_time_y <= 0.0` at lines 109-110
and rejects non-positive `price_expiry_y` at lines 173-179. Thus a valid
zero-business-DTE PM book is selected and then fails materialization.

Required change: preserve the recovered nonnegative contract in valid
prefixes (zero remains a valid masked-in value), or obtain an explicit
compatibility decision before changing the selector to strictly positive.
Add an exact zero-DTE boundary regression.

### 2. High — content-producing orchestration is outside the producer-version closure

`_project_partition` and `_prepare_shards`
(`src/sanos/experiments/build.py:184-365`) are unversioned. They determine the
ordered retained-day population, typed-empty exclusions stored in manifest
metadata, and the exact daily inputs handed to HDF5. Their behavior (including
the `bad_day_reason` policy at lines 219-226) can change Experiment content or
metadata without changing the artifact identity.

`experiment_producer_version` at lines 511-535 includes the projection and
shard primitives, but not these deterministic wrappers or their population
policy. Its regression at `tests/test_experiments.py:274` asserts the same
incomplete dependency set.

Required change: make the deterministic population/projection helpers
`@version`-owned, declare all content-affecting dependencies (including the
typed-empty policy), and include the resolved closure in
`experiment_producer_version`. Add a direct dependency regression that fails
if any stage capable of changing dates, exclusions, or day bytes drops from
the closure.

### 3. High — a normal corrupt restart shard is not rebuilt and aborts resume

`read_day_shard` catches only `OSError`, `ValueError`, and `KeyError`
(`src/sanos/experiments/hdf5.py:184-187`). A truncated/invalid NPZ commonly
raises `zipfile.BadZipFile`, which escapes unnormalized. `_prepare_shards`
only treats `ExperimentValidationError` as an invalid reusable shard
(`src/sanos/experiments/build.py:261-269`), so this corruption aborts the
entire rerun rather than deleting/rebuilding the shard.

Independent reproduction:

```text
read_day_shard(Path("broken.npz"))  # bytes b"PK\\x03\\x04garbage"
-> zipfile.BadZipFile: File is not a zip file
```

Required change: normalize all expected safe NPZ decode failures into
`ExperimentValidationError`, rebuild an invalid exact-plan shard, and add
resume tests for truncation, bad ZIP, wrong date/schema/fingerprint, and a
valid-shard reuse. The current Phase 9 tests contain no restart/interruption
test.

### 4. High — shallow HDF5 validation does not re-establish the advertised population contract

`validate_experiment_hdf5` validates the closed object graph, dtypes, filters,
masks, padding, indices, fingerprints, weekly pairs, and logical digest. It
does not establish that:

- every retained expiry is inside the configured horizon and outside every
  configured literal exclusion;
- every retained quote is inside its stored inclusive strike bounds; or
- root `E`, `Q`, and `P` equal the actual maxima of the valid per-day counts,
  rather than merely being upper bounds.

The relevant validation starts at
`src/sanos/experiments/hdf5.py:892`; the per-day loop at lines 935-985 only
checks `0 < count <= dimension`, masks/padding, object construction, and the
fingerprint. A coherently re-signed self-contained Experiment can therefore
pass semantic validation while violating its own selected-universe contract.

Required change: add configuration-aware self-contained relations and
re-signed mutation tests for horizon, exclusion tolerance, strike containment,
and non-maximal dimensions. Exact NMS bounds need not be re-fitted without the
source Dataset, but stored books must be proven to lie inside the stored
bounds.

### 5. High — the required selector/parity boundary is not hermetic or independently tested

The pinned-consumer subprocess in `tests/test_experiments.py:30-171` is a good
normal-day transport test after the latest fingerprint fix. It silently
returns when the external `~/python/quant_data` checkout is absent, however,
and it consumes a day already selected by the new implementation. It cannot
detect a matching selection error.

The repository has only six Phase 9 implementation tests, with no independent
matrix for inactive sides, American style, unavailable/nonfinite fits,
`0.2 +/- 1e-4`, zero/70/over-70D, inclusive/outside NMS endpoints,
single-sided survival, first-occurrence price mapping, or all audit counts.
The pinned `market.py` is already carried under `reference/quant_data`, so the
selector parity gate can be made offline and repository-contained.

Required change: add a hand-calculated or pinned-oracle selector fixture that
checks those boundaries without reusing the new predicates in its expected
values. Keep the real consumer parity test, but make required parity fail
clearly rather than silently skip when an external checkout is absent (or
carry the exact minimal consumer under the established provenance policy).

### 6. High — HDF5 assembly can violate the mandatory progress-heartbeat contract

After shard preparation, `build_experiment` reads every shard into a tuple and
calls the HDF5 writer (`src/sanos/experiments/build.py:785-806`).
`write_experiment_hdf5` has no progress callback
(`src/sanos/experiments/hdf5.py:290-296`) and its compressed per-day write loop
at lines 460-497 emits no heartbeat. On the full 917-day / multi-million-book
materialization, shard loading, allocation, compression, flushing, and fsync
can therefore be silent for more than the host-wide five-minute maximum.

Required change: thread the configured interval/progress callback through
assembly and emit flushed structured progress with completed/total, elapsed,
failures, and ETA where meaningful. Add a deterministic clock/callback test
covering a long write interval.

## Additional acceptance gaps

- The custom pool has no interruption/concurrent-publisher coverage. The plan
  lock times out after 30 seconds while held for the full build
  (`src/sanos/experiments/build.py:736-737`), rather than polling and reusing a
  long-running identical winner. Add tests for interruption during submission,
  shard write, HDF creation/validation/commit, immediate restart with no live
  worker, concurrent identical callers, and distinct-plan isolation.
- `source.json` validation checks exact top-level keys, pointer, checksum
  length, and internal status, but not the recorded configuration/schema and
  semantic-summary values against the source on `--deep-lineage`. Strengthen
  deep validation so source evidence is compared with a newly validated exact
  source Dataset; retain shallow output-first behavior when it is absent.
- The implementation loads all daily shards into memory before HDF5 assembly,
  despite the approved design's streaming plan. This is not itself a blocker
  for the stated modest dataset, but should remain a measured residual risk in
  the production pilot.

## Passing areas

- Planning, dry-run, and wildcard miss occur before source/workspace access.
  Current and wildcard hits validate and reuse the completed Experiment after
  source deletion; wildcard remains reuse-only and visibly reported.
- The exact internal Dataset pointer is the sole upstream. Build-time source
  validation checks the supported internal Dataset schema and matching
  smoothness, population cap, and overnight fraction. No WRDS path is opened.
- Final publication uses `ArtifactBuilder`; HDF5 is written in private staging,
  closed, fsynced, semantically validated, checksummed, and atomically renamed.
  Post-publication workspace cleanup failure warns/journals without
  invalidating the artifact.
- HDF5 has a closed path/dtype/filter/padding contract. Soft/external links,
  hard-link aliases, external/virtual data, and child attributes are rejected.
- The latest `DailyFrontDay` adapter exposes the pinned logical names without
  copying arrays. Its objective-only `training_fingerprint` is distinct from
  the source-bound `market_fingerprint`, and the real pinned normal-day
  `materialize_day`/`materialize_batch` values, indices, shapes, dtypes, masks,
  and fingerprint agree on this host.
- CLI planning/build/validation and human/JSON output are present, offline, and
  documented. Documentation correctly states that the freshly validated and
  promoted SANOS Dataset is canonical, the legacy 891-date artifact is only a
  migration oracle, and this HDF5 Experiment is internal and disposable.

## Commands and evidence

```text
env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
-> 196 passed in 167.78s

env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_experiments.py tests/test_cli.py
-> 23 passed in 31.44s

python -m compileall -q src tests
ruff check src/sanos/experiments src/sanos/cli/main.py \
  tests/test_experiments.py tests/test_cli.py
git --git-dir=.git-real --work-tree=. diff --check
-> passed

python -m mypy ...
-> not run: mypy is not installed in the active interpreter
```

I also inspected `AGENTS.md`, `PROGRAM_PLAN.md` Phase 9, both Phase 9 design
audits, all new Experiment modules/config/docs/tests, the CLI diff, and the
pinned read-only reference selector/consumer. No network, WRDS, GPU, or
`quant_data` write was performed. Temporary adversarial probes were kept below
`/tmp/sanos/`.

Signed: `/root/code` — independent Phase 9 code reviewer

---

## Final re-review after remediation

**Reviewer:** independent code agent (`/root/code`)  
**Date:** 2026-10-12  
**Verdict:** **PASS**

I re-reviewed the settled tree against every finding above. All six blocking
findings are resolved, the strengthened deep-lineage source-evidence check is
correct, and no new acceptance-critical code defect remains.

### Finding status

1. **Zero DTE — resolved.** `DailyFrontDay` 1.6 now permits nonnegative
   retained `expiry_time_y` and `price_expiry_y`, while masked padding remains
   governed by counts/masks. A zero-DTE day passes object construction, NPZ
   round-trip, HDF5 round-trip and validation, and the hash-pinned reference
   selector independently confirms the inclusive lower boundary.

2. **Version ownership — resolved.** `_project_partition` and
   `_prepare_shards` are versioned. The latter declares the exact plan/config,
   shard projection/read, fingerprint reconstruction, shard path and
   `bad_day_reason` policy dependencies; both are direct members of
   `experiment_producer_version`. The dependency regression and a fresh-process
   producer/artifact-ID regression pass. `DailyFrontDay` was explicitly bumped
   to 1.6 after its zero-DTE and containment invariants changed.

3. **Corrupt restart shards — resolved.** `read_day_shard` 1.1 normalizes
   `EOFError` and `zipfile.BadZipFile` in addition to the previous safe decode
   failures. Exact-plan reuse revalidates date and source-bound fingerprint.
   Tests prove valid shard reuse and rebuilding of corrupt ZIP, wrong schema,
   wrong date and wrong fingerprint shards.

4. **Self-contained HDF semantics — resolved.** The shallow validator now
   enforces the configured horizon and literal DTE exclusions, quote
   containment inside stored inclusive bounds, reconstruction of configured
   NMS bounds from stored interpreting geometry, the source population cap,
   and exact observed `E/Q/P` maxima. Coherent adversarial files with updated
   fingerprints/logical digests are rejected for excluded DTE, over-horizon,
   changed NMS geometry, out-of-bounds strikes and oversized dimensions.

5. **Hermetic selector parity — resolved.** The repository test loads the
   checked-in pinned selector from `reference/quant_data`, verifies the exact
   source hashes, and runs it in an isolated subprocess. It covers independent
   side activity, American exclusion, availability/nonfinite fit handling,
   zero DTE, the exact `0.2 +/- 1e-4` exclusion boundary, inclusive 70D and
   over-horizon behavior, inclusive/outside NMS boundaries, side ordering,
   first-occurrence price mapping and all audit counts. The normal-day pinned
   consumer probe also confirms fingerprint and materialized value/index/
   shape/dtype/mask parity on this host.

6. **Progress heartbeat — resolved.** Logical digest and compressed HDF5
   assembly now emit bounded structured progress with completed/total,
   failures, elapsed time and ETA using the configured interval. A
   deterministic clock regression separately asserts both digest and assembly
   event streams at the 301-second boundary.

7. **Deep source evidence — resolved.** Normal shallow validation remains
   output-first and self-contained. With `deep_lineage=True`, the implementation
   re-runs the exact internal Dataset semantic gate, reads its completion
   checksum, reconstructs the complete versioned `source.json` evidence, and
   requires canonical equality. The source-deletion regression proves shallow
   reuse still succeeds while explicit deep validation fails clearly when the
   source is absent.

### Final verification

```text
env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
-> 197 passed in 170.48s

env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_experiments.py
-> 7 passed in 35.04s

ruff check src/sanos/experiments src/sanos/cli/main.py \
  tests/test_experiments.py tests/test_cli.py
python -m compileall -q src tests
git --git-dir=.git-real --work-tree=. diff --check
-> passed
```

The source/config gate, output-first and wildcard behavior, immutable atomic
publication, closed HDF5 object/link/attribute contract, canonical padding,
CLI behavior, and canonical-reference wording remain correct. The fresh
validated/promoted SANOS Dataset is the authoritative research object; the
891-date legacy artifact is a migration oracle only, and this internal HDF5 is
disposable.

### Residual production-pilot risks (nonblocking)

- The full 917-populated-day scale and legacy common-period aggregate parity
  remain Phase 11 production evidence, not an offline Phase 9 fixture.
- The implementation materializes the modest daily shard set in memory before
  HDF5 assembly. Measure peak RAM and I/O duration during the fresh pilot.
- The existing bounded spawned-pool cleanup and atomic staging logic are sound,
  but production should still exercise operator interruption and concurrent
  invocation as an operational smoke test before the full-range run.

Signed: `/root/code` — independent Phase 9 final code reviewer
