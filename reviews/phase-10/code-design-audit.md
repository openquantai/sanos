# Phase 10 code/design audit — fresh-download pilot

**Verdict: CHANGES REQUIRED BEFORE REQUESTING WRDS 2FA**

**Reviewer:** `/root/code` (independent code-agent design gate)
**Date:** 2026-10-12
**Scope:** offline review of the committed Phase 4 raw layer, current CLI and
daily pipeline, Phase 10 plan, runbooks, and Phase 4 reviews. No network or
WRDS access occurred. The concurrent uncommitted Phase 8 public-projection work
was not reviewed as Phase 10 implementation.

The guarded download implementation is a sound base for a live pilot: it is
output-first, gates a real miss immediately before credentials are read, closes
the single catalog probe before fan-out, uses host-wide account leases, writes
immutable artifacts atomically, and defaults to two worker connections. A live
pilot should nevertheless not start yet. Two operator-facing capabilities and
two focused offline regressions described under **Required pre-live changes**
are needed so the pilot can satisfy its own acceptance gate without ad-hoc
Python or manual interpretation.

## Representative pilot matrix

Use one compact contiguous request, **2024-12-23 through 2025-01-16**, rather
than several isolated requests. This creates one authentication/catalog probe
and one bounded fan-out while exercising all required cases:

| Case | Trading date / evidence | Expected behavior |
|---|---|---|
| ordinary | 2024-12-23 and 2025-01-06 | populated fresh fitted days; compare to pinned legacy daily-market records |
| half day | 2024-12-24 | `is_half_day=true`; fresh output is retained; the missing legacy file is an explicit oracle gap, not a failure |
| irregular expiry | 2025-01-07 input containing expiry 2025-01-09 | raw source retains the late-announced expiry and the configured daily stage removes it; if the current vendor extract no longer contains it, record the category as unexercised and stop for review rather than silently choosing another date |
| historically troublesome | 2025-01-08 | exact-schema typed-empty fitted day with reason `known_bad_day_all_tickers`, regardless of whether fresh Raw rows exist |
| AM settlement | 2025-01-16 with expiry 2025-01-17 | next-session AM node has business DTE 0.2 under the default overnight fraction; the legacy oracle records 186 active call/put books on that node |

The NYSE planner resolves exactly 16 requested sessions in this interval.
Independent offline planning produced **68 unique Raw artifacts**: one security
snapshot, 16 option partitions, 16 current-date partitions each for spot,
forwards and zero curve, and three boundary partitions (spot, forwards and zero
curve) on 2024-12-20. The fitted-day planner produced exactly 16 terminal
artifacts. These counts and boundary identities are pre-live assertions, not
informational log text.

The pinned catalog contains populated oracle files for 2024-12-23, 2025-01-06,
2025-01-07 and 2025-01-16, an empty marker for 2025-01-08, and no 2024-12-24
file. This makes the distinction between a parity discrepancy, a policy-owned
empty day, and a legacy-oracle gap observable in the pilot.

## Required pre-live changes

### 1. Blocking — add a human-operable daily reference comparison

There is no installed command that compares freshly fitted daily Parquet with
the read-only daily HDF5 oracle. `sanos.reference.LegacyMarketReference` can
read a day, but it is only a Python API; the CLI exposes only raw download,
intermediate build, Dataset build/validate and Experiment build/validate
(`src/sanos/cli/main.py:405-878`). Re-running `intermediate build` proves cache
integrity but does not establish migration parity. Phase 10 therefore cannot
currently meet “compare with the read-only reference outputs” through the
documented non-AI interface.

Add an installed command with an interface equivalent to:

```text
sanos reference compare-daily \
  --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2024-12-23 --end-date 2025-01-16 \
  --source optionmetrics --source-generation <pilot-generation> \
  --legacy-root <read-only-smooth-market-root> \
  --output <outside-git-evidence>/daily-comparison.json
```

It must only validate/read existing fitted artifacts, never build them; open
the legacy root read-only; write its report atomically; and support JSON
progress/errors and `-v`. Its versioned report must bind the SANOS revision,
resolved transform/configuration versions, source generation, every fitted
artifact pointer/checksum, the pinned legacy collection/file checksums, and a
per-date classification (`match`, `oracle_gap`, `policy_empty`,
`unexplained_discrepancy`, or `approved_discrepancy`). Compare exact date,
expiry/strike ordering, masks, populations, AM flags, state geometry and dtypes;
report numerical maxima for market values, no-arbitrage state, DLV/density,
smooth SANOS state/fitted values and diagnostics. Structural differences are
exact failures. Numerical tolerances must be a named, `@version`-owned contract
reviewed by the quant agent; every nonzero difference remains visible even if
within tolerance. Vendor revisions are reported, never overwritten or coerced
to the legacy result.

### 2. Blocking — make live provenance evidence available from the CLI

`raw download --cache-only` correctly validates manifests and checksums, but
its final structured event contains only counts and paths
(`src/sanos/cli/main.py:554-564`). The live acceptance gate also needs the
resolved physical table, source generation, artifact ID, observation date,
query digest, row/empty count, persisted dtypes and payload checksum. Those
values exist in `manifest.json` and `query.json`, but the current runbook would
force a human to discover and merge 68 files manually.

The smallest change is to make `raw download --cache-only -v --log-format
json` emit one stable `raw_artifact_validated` record per artifact with those
fields. A separate `sanos raw validate` command is also acceptable. It must
remain output-first and network-impossible, and it must reject checksum,
schema, unexpected-file and symlink failures rather than summarize them as a
valid artifact. The resulting JSONL is the pilot's raw provenance evidence.

### 3. Blocking test gap — partial failure/restart at the Raw layer

The Phase 4 suite proves complete-generation reuse and immutable refresh, but
`test_refresh_resumes_generation_and_never_replaces_it` first completes every
partition (`tests/test_wrds_download.py:419-459`). Add an offline fake in which
a dated query fails after at least one artifact was durably published, then
rerun the identical generation. Assert that the retry opens no query for
completed partitions, downloads only misses, does not replace prior paths or
manifests, closes every connection/releases every lease, and ends with all
planned fake partitions complete. Cover a terminal failure and a
`BaseException`/interruption path. Do not deliberately interrupt the real
account merely to produce this evidence.

### 4. Blocking test/runbook gap — cache tamper and fail-closed pilot procedure

The generic artifact-store tests reject modified payloads, unexpected files
and symlinks, but the raw CLI has no focused regression proving that a corrupt
Raw hit fails `--cache-only` without invoking authorization or network. Add
that test at the CLI boundary. The Phase 10 runbook must also define the
expected cache-miss exit code, stop-on-authentication/permission/connection
limit behavior, safe resume command, external evidence directory, log
redaction check, and the explicit rule that a pilot discrepancy stops Phase 11.
The present WRDS runbook ends at generic one-day download/resume instructions
(`docs/wrds-runbook.md:24-128`) and is not an executable Phase 10 protocol.

## Exact operator sequence after remediation

The following is the intended runbook. All generated logs remain outside Git
and, for a shared root, below a SANOS-owned directory. `RUN_ID` and `GENERATION`
must be chosen once and retained for restart; use a new generation if the
preflight finds any existing pilot artifact.

```bash
set -o pipefail
export PILOT_ROOT=/data/sanos/phase-10/data
export PILOT_EVIDENCE=/data/sanos/phase-10/evidence/<run-id>
export PILOT_GENERATION=phase10-pilot-<run-id>
export LEGACY_MARKET_ROOT=<read-only-legacy-smooth-market-SPX-directory>
mkdir -p "$PILOT_EVIDENCE"
```

Record the repository revision/status and run the ordinary offline suite and
new focused restart/tamper/comparison tests. Do not copy credentials or the
legacy HDF5 files into the evidence directory.

```bash
git --git-dir=.git-real --work-tree=. rev-parse HEAD \
  > "$PILOT_EVIDENCE/sanos-revision.txt"
git --git-dir=.git-real --work-tree=. status --porcelain=v1 \
  > "$PILOT_EVIDENCE/sanos-status.txt"
env WRDS_OFF=1 python -m pytest -q -p no:cacheprovider \
  | tee "$PILOT_EVIDENCE/offline-tests.log"
env WRDS_OFF=1 sanos --root "$PILOT_ROOT" --log-format json \
  doctor --strict-locations | tee "$PILOT_EVIDENCE/doctor.jsonl"
```

Plan with networking disabled. This must report `planned=68`, `missing=68`,
`reused=0`, `boundary_partitions=3`, the three 2024-12-20 boundary records,
and `network_accessed=false`. If it reports a reuse, stop and choose a new
source generation; this pilot is explicitly a fresh extraction.

```bash
env WRDS_OFF=1 sanos --root "$PILOT_ROOT" --dry-run --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --refresh --source-generation "$PILOT_GENERATION" \
  --max-wrds-connections 2 --retry-attempts 1 \
  | tee "$PILOT_EVIDENCE/raw-plan.jsonl"
```

Run the same request with `--cache-only`; before a live run it must fail with
the stable material-command exit code 4, list missing partitions, and show no
network access. Capture and assert that expected exit rather than ignoring it.
Also run `gpuctl resources --json` into
`$PILOT_EVIDENCE/resources-before.json`. Every resource check must be `OK`; a
failed command is `WARN` and any `STOP`, `UNAVAILABLE` or `ERROR` blocks the
pilot under host policy.

Only now ask the user for immediate 2FA readiness and additionally confirm
that the same WRDS account has no non-SANOS notebook, remote host, or other
client active. The SANOS account lease cannot see external clients. If either
answer is not affirmative, do not connect. Once confirmed, remove `WRDS_OFF`
and execute exactly one live request:

```bash
env -u WRDS_OFF sanos -v --root "$PILOT_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --refresh --source-generation "$PILOT_GENERATION" \
  --max-wrds-connections 2 --retry-attempts 1 --confirm-2fa-ready \
  2>&1 | tee "$PILOT_EVIDENCE/raw-live.jsonl"
```

The probe is one leased connection and closes before fan-out
(`src/sanos/wrds/download.py:597-662`); fan-out is explicitly capped at two
(`:663-718`). Thus this invocation never intentionally has more than two WRDS
connections open concurrently. Using one retry attempt in the pilot further
limits account pressure. Authentication, 2FA, permission, connection-limit and
unknown errors are not retried. On any such error, stop; do not repeatedly
re-enter credentials or relaunch without a new user readiness confirmation.

Immediately validate and prove output-first restart with live access disabled:

```bash
env WRDS_OFF=1 sanos -v --root "$PILOT_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --source-generation "$PILOT_GENERATION" --cache-only \
  --max-wrds-connections 2 \
  | tee "$PILOT_EVIDENCE/raw-cache-validation.jsonl"

env WRDS_OFF=1 sanos -v --root "$PILOT_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --refresh --source-generation "$PILOT_GENERATION" \
  --max-wrds-connections 2 \
  | tee "$PILOT_EVIDENCE/raw-restart.jsonl"
```

Both must report 68 reused, zero downloaded/missing and
`network_accessed=false`. If the live command was interrupted, use this same
generation and command after a fresh 2FA-readiness confirmation; it may access
WRDS only for validated misses.

Plan and build all 16 daily frames through the fitted stage, including the
half-day, with WRDS forcibly disabled:

```bash
env WRDS_OFF=1 sanos --root "$PILOT_ROOT" --dry-run --log-format json \
  intermediate build --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2024-12-23 --end-date 2025-01-16 \
  --through fitted-day --include-half-days --source optionmetrics \
  --source-generation "$PILOT_GENERATION" --max-workers 2 \
  | tee "$PILOT_EVIDENCE/fitted-plan.jsonl"

env WRDS_OFF=1 sanos -v --root "$PILOT_ROOT" --log-format json \
  intermediate build --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2024-12-23 --end-date 2025-01-16 \
  --through fitted-day --include-half-days --source optionmetrics \
  --source-generation "$PILOT_GENERATION" --max-workers 2 \
  | tee "$PILOT_EVIDENCE/fitted-build.jsonl"
```

Re-run the last command under `WRDS_OFF=1` and require all 16 terminals to be
reused with zero built/failed. Then invoke the new `reference compare-daily`
command described above, writing its report and JSONL progress under the same
evidence directory.

Do not publish a Dataset in this pilot using the checked-in reference config:
its identity covers the full 2022-01-03 through 2025-08-29 range. Phase 10 ends
at validated fitted-day artifacts and a signed comparison report. Creating a
temporary Dataset config merely for the pilot would create a different curated
product and adds no evidence beyond validated per-day fitted artifacts.

## Tamper and failure checks

- **Cache miss:** the pre-live `--cache-only` request must exit 4 without
  authorization or a connection.
- **Payload tamper:** the new offline CLI test must copy a complete fake Raw
  artifact tree beneath a test root, modify one copied `data.parquet`, and prove
  cache-only validation exits 4 before authorization/network. Never corrupt the
  durable live pilot root.
- **Partial failure/restart:** the new fake-query regression must publish a
  proper subset, fail, and prove exact-generation resume as described above.
- **Completed restart:** the live generation is re-run under `WRDS_OFF=1`; all
  68 artifacts must be validated/reused with no credential read.
- **Authentication/access failure:** do not manufacture one against WRDS. If it
  occurs, preserve the redacted log, verify all connections/leases close, stop
  the pilot, and require a new user readiness confirmation before retrying.
- **Daily failure:** any unknown empty, solver failure, missing raw dependency,
  or nonzero worker failure stops the pilot. Existing completed daily artifacts
  remain immutable and the identical offline command is the restart path.

## Acceptance gate

Phase 10 passes only when all of the following are true:

1. The new comparator, verbose Raw validation evidence, partial-resume test and
   Raw CLI tamper test receive independent code and quant review.
2. Preflight records a clean/reviewed SANOS revision, successful full offline
   suite, installed WRDS extra, resolved external roots, 68-miss fresh plan,
   expected cache-only failure, and all-`OK` `gpuctl resources` output.
3. The user confirms immediate 2FA readiness and no concurrent external use;
   the sole live invocation uses at most two connections and one query attempt.
4. Every live Raw artifact validates, exposes complete non-secret physical
   table/query/schema/count/checksum provenance, and exact restart under
   `WRDS_OFF=1` performs no network access.
5. All 16 fitted days build and reuse offline. The half-day is retained, the
   irregular expiry behavior is actually exercised, 2025-01-08 is the policy
   typed-empty, and the 2025-01-16 0.2-business-day AM population is present.
6. The comparison report is checksum-bound and complete. Oracle gaps are
   labelled; every other structural/numerical discrepancy is either explained
   as a vendor revision/known compatibility item and explicitly approved, or
   blocks Phase 11. Fresh data is never mutated to force legacy equality.
7. External evidence contains revision/status, plans, resource state, raw live
   and validation JSONL, fitted plan/build/reuse JSONL, comparison JSON and
   reviewer sign-offs. A redaction scan finds no username, password, OTP or
   credential material. Only a concise non-secret summary and digests may be
   committed beneath `reviews/phase-10/`.
8. Both the independent code review and quantitative/data review sign PASS.

Phase 11 must not begin, and the user should not yet be asked for immediate
2FA, while any item above is unresolved.

## Evidence from this offline audit

```text
git --git-dir=.git-real --work-tree=. rev-parse HEAD
3d4a6ef70c3456a3ed470bbc0e77644d6c3edbf3

PYTHONPATH=src WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m sanos \
  --root /tmp/sanos/phase-10-design --dry-run --log-format json raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --source-generation phase10-design --refresh --max-wrds-connections 2
raw_plan: planned=68 reused=0 missing=68 boundary_partitions=3;
boundary date 2024-12-20 for forwards, spot and zero_curve;
network_accessed=false

PYTHONPATH=src WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m sanos \
  --root /tmp/sanos/phase-10-design --dry-run --log-format json intermediate build \
  --config configs/widb-spx-reference.json --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --through fitted-day --include-half-days --source optionmetrics \
  --source-generation phase10-design --max-workers 2
intermediate_result: total=16 existing=0 built=0 failed=0;
network_accessed=false
```

Both dry runs completed successfully and created no configured data root. No
live WRDS, network, GPU, or legacy/reference write was attempted.
