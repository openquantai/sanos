# Phase 10 guarded WRDS pilot

This is the executable, fail-closed protocol for the first live SANOS extraction.
It stops after fitted daily artifacts and a legacy comparison report. It does not
publish a Dataset and it does not authorize the full 2022--2025 build.

The pilot interval is 2024-12-23 through 2025-01-16 inclusive. With half-days
included it has 16 NYSE sessions and plans exactly 68 Raw artifacts, including
the spot, forward and zero-curve boundary partitions dated 2024-12-20. Use no
more than two WRDS connections and one query attempt.

## 1. Choose immutable run identities

Run from a clean, reviewed SANOS checkout. Keep every generated file outside
Git and beneath the SANOS-owned shared directory.

```bash
set -o pipefail
export PILOT_ROOT=/data/sanos/phase-10/data
export PILOT_RUN_ID=<UTC-date-and-operator-label>
export PILOT_EVIDENCE=/data/sanos/phase-10/evidence/$PILOT_RUN_ID
export PILOT_GENERATION=phase10-pilot-$PILOT_RUN_ID
export LEGACY_MARKET_ROOT='<legacy_QUANTDATA_TRAINING_DATA>/dysanos/market/bid=0.011, nat,eth=10,bth=4,mps=1e-06,m ca250824/sm_0.25 6b930b59/SPX'
mkdir -p "$PILOT_EVIDENCE"
```

Choose `PILOT_RUN_ID` and `PILOT_GENERATION` once. Preserve them when resuming
an interrupted extraction. If preflight finds any artifact in the proposed
generation, stop and choose a new generation; the acceptance run must begin
with 68 misses.

## 2. Offline preflight

Record exact source state, run the complete offline suite, check configuration,
and capture the host resource gate. A failed `gpuctl resources` command is a
warning and blocks this live pilot. Every reported check must be `OK`; any
`WARN`, `STOP`, `UNAVAILABLE`, or `ERROR` blocks it.

```bash
git --git-dir=.git-real --work-tree=. rev-parse HEAD \
  > "$PILOT_EVIDENCE/sanos-revision.txt"
git --git-dir=.git-real --work-tree=. status --porcelain=v1 \
  > "$PILOT_EVIDENCE/sanos-status.txt"
test ! -s "$PILOT_EVIDENCE/sanos-status.txt"

env WRDS_OFF=1 python -m pytest -q -p no:cacheprovider \
  2>&1 | tee "$PILOT_EVIDENCE/offline-tests.log"
env WRDS_OFF=1 sanos --root "$PILOT_ROOT" --log-format json \
  doctor --strict-locations | tee "$PILOT_EVIDENCE/doctor.jsonl"
gpuctl resources --json | tee "$PILOT_EVIDENCE/resources-before.json"
```

Plan with networking forcibly disabled:

```bash
env WRDS_OFF=1 sanos --root "$PILOT_ROOT" --dry-run --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --refresh --source-generation "$PILOT_GENERATION" \
  --max-wrds-connections 2 --retry-attempts 1 \
  | tee "$PILOT_EVIDENCE/raw-plan.jsonl"
```

Require `planned=68`, `missing=68`, `reused=0`, `boundary_partitions=3`, the
three 2024-12-20 boundary identities, and `network_accessed=false`. Any reuse,
different count, or different boundary blocks the pilot.

Prove a miss cannot authorize or connect. This command must exit with code 4.

```bash
set +e
env WRDS_OFF=1 sanos -v --root "$PILOT_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --source-generation "$PILOT_GENERATION" --cache-only \
  --max-wrds-connections 2 \
  2>&1 | tee "$PILOT_EVIDENCE/raw-cache-miss.jsonl"
cache_status=${PIPESTATUS[0]}
set -e
test "$cache_status" -eq 4
```

## 3. Human readiness gate and sole live request

Stop here until the repository owner is present and explicitly confirms both:

1. they are ready to complete WRDS 2FA immediately; and
2. the same WRDS account has no notebook, remote host, scheduled process, or
   other client active.

Do not infer readiness from an earlier conversation. The host-wide SANOS lease
cannot see external clients. Authentication, 2FA, permission, connection-limit
or unknown failures are not retried automatically.

After that confirmation, repeat the resource gate immediately before access:

```bash
gpuctl resources --json | tee "$PILOT_EVIDENCE/resources-pre-live.json"
```

Do not proceed unless the command reaches the daemon and every reported check
is `OK`; a local fallback, command failure, or any other health value blocks
the live pilot. Then execute exactly one live command:

```bash
env -u WRDS_OFF sanos -v --root "$PILOT_ROOT" --log-format json \
  raw download --source optionmetrics --ticker SPX \
  --start-date 2024-12-23 --end-date 2025-01-16 \
  --refresh --source-generation "$PILOT_GENERATION" \
  --max-wrds-connections 2 --retry-attempts 1 --confirm-2fa-ready \
  2>&1 | tee "$PILOT_EVIDENCE/raw-live.jsonl"
```

On an authentication, 2FA, permission, connection-limit, unknown, or resource
error, stop. Preserve completed immutable artifacts and the redacted log. Do
not repeatedly re-enter credentials or relaunch without a new immediate
readiness confirmation.

## 4. Offline validation and restart proof

Disable WRDS again. The verbose cache-only command emits one
`raw_artifact_validated` JSON record per artifact, including physical table,
generation, artifact ID, observation date, query digest, row/empty status,
persisted dtypes and payload checksum.

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

Both commands must report 68 reused, zero downloaded/missing, and
`network_accessed=false`. If the live command was interrupted, resume the same
generation only after another immediate 2FA-readiness confirmation; validated
completed partitions are reused and only misses may be queried.

Never tamper with the durable pilot root to test corruption. The automated
offline suite performs copied-tree checksum-tamper and partial-failure restart
tests.

## 5. Offline fitted build

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

env WRDS_OFF=1 sanos -v --root "$PILOT_ROOT" --log-format json \
  intermediate build --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2024-12-23 --end-date 2025-01-16 \
  --through fitted-day --include-half-days --source optionmetrics \
  --source-generation "$PILOT_GENERATION" --max-workers 2 \
  | tee "$PILOT_EVIDENCE/fitted-reuse.jsonl"
```

The plan and both executions must cover 16 terminal dates. The final run must
reuse all 16 with zero built and zero failed.

## 6. Read-only parity comparison

The comparison reads existing fitted Parquet and the pinned HDF5 oracle. It
does not build an artifact, open WRDS, or copy legacy files. The known missing
half-day is declared explicitly before the run.

```bash
env WRDS_OFF=1 sanos -v --root "$PILOT_ROOT" --log-format json \
  reference compare-daily --config configs/widb-spx-reference.json \
  --ticker SPX --start-date 2024-12-23 --end-date 2025-01-16 \
  --source optionmetrics --source-generation "$PILOT_GENERATION" \
  --legacy-root "$LEGACY_MARKET_ROOT" \
  --expected-oracle-gap 2024-12-24 \
  --phase10-pilot-contract \
  --output "$PILOT_EVIDENCE/daily-comparison.json" \
  2>&1 | tee "$PILOT_EVIDENCE/daily-comparison.jsonl"
```

Expected unchanged-oracle classifications are 14 `match`, one
`policy_empty`, and one `oracle_gap`. The report must also prove:

- 2024-12-24 is populated and `is_half_day=true`;
- 2025-01-07 Raw contains the 2025-01-09 irregular expiry and the daily stage
  records its removal; if the revised vendor extract lacks it, stop as an
  unexercised required case;
- 2025-01-08 is an exact-schema typed-empty day with reason
  `known_bad_day_all_tickers`; and
- 2025-01-16 retains the 2025-01-17 AM expiry at business DTE 0.2 and records
  186 legacy active side books on that node.

Any unexpected oracle gap or `unexplained_discrepancy` blocks Phase 11. A
suspected vendor revision stays unexplained until the owner explicitly approves
an immutable record binding the exact date, fields, fresh artifact/data
digests, legacy file digest, explanation, approver, date and durable reference.
Fresh values are never changed to force oracle equality.

## 7. Evidence redaction and review gate

Inspect logs before sharing or committing any summary. Evidence must not
contain a WRDS username, password, OTP, connection string, credential path, or
raw proprietary value sample. Do not print secrets in order to search for
them. Review authentication error text manually and use a local case-insensitive
scan for credential-related labels:

```bash
rg -n -i 'password|passwd|one[-_ ]time|otp|credential|wrds[_-]?username|dsn=' \
  "$PILOT_EVIDENCE"
find "$PILOT_EVIDENCE" -type f ! -name checksums.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "$PILOT_EVIDENCE/checksums.sha256"
gpuctl resources --json | tee "$PILOT_EVIDENCE/resources-after.json"
```

Any match must be reviewed and redacted without altering the original evidence
identity silently; create a separately named redacted copy and record both
digests. Phase 10 passes only after the code and quantitative reviewers sign
the complete external evidence bundle. Only non-secret counts, classifications,
digests and sign-offs belong under `reviews/phase-10/`.
