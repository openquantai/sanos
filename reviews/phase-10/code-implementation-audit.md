# Phase 10 code implementation audit

Date: 2026-10-13
Reviewer: `code` (independent code/engineering reviewer)
Baseline: commit `5256f9b` plus the current uncommitted Phase 10 worktree
Verdict: **PASS — offline implementation prerequisites only**

This verdict approves the reviewed offline comparator, Raw validation evidence,
failure/restart coverage, CLI wiring, and human pilot runbook for commit. It does
**not** authorize a live WRDS connection or the Phase 11 production build. The
live pilot remains blocked until the owner gives a fresh, immediate confirmation
that they are ready for WRDS 2FA and that no other client is using the account,
and until `gpuctl resources --json` reaches the daemon and reports `OK` for every
check immediately before access.

## Findings

No blocking code or acceptance finding remains in the reviewed offline scope.

The review initially found three material issues, all resolved in the current
tree:

1. The pilot could previously infer the 2025-01-07 irregular-expiry case from
   only an aggregate removal count. The comparator now shallow-validates the
   exact planned Raw options artifact, reconciles its immutable query/schema/
   checksum provenance, and proves that expiry 2025-01-09 is physically present
   before accepting the positive daily removal diagnostic. This evidence is
   included in the checksum-bound report without exposing vendor rows.
2. The comparator's `@version` graph initially omitted content-affecting
   collaborators and failed to resolve in a clean process because of a helper
   name collision. The helper is now uniquely named, the direct closure owns the
   artifact/config/planning/validation/comparison collaborators, and a clean
   subprocess regression resolves the complete version successfully.
3. Legacy canonicality and several serialized dtype boundaries were previously
   asserted rather than checked. The current comparator derives legacy order and
   uniqueness, rejects serialized expiry/index/mask/mapping dtype changes, and
   records distinct full-input, selected, retained, dropped-expiry, mask, and
   active-book populations. Adversarial tests cover both ordering/key failures
   and the dropped-expiry/zero-active-book edge cases.

## Engineering assessment

- **Read-only and network safety:** `reference compare-daily` plans identities
  deterministically and shallow-validates completed fitted artifacts; it never
  invokes an intermediate builder or WRDS client. The one Raw read under the
  frozen Phase 10 contract is an intentional validation of the already-present
  2025-01-07 options artifact. Structured results declare
  `network_accessed=false` and `artifacts_built=false`.
- **Evidence binding:** fresh fitted pointers, Parquet checksums, schema and
  producer versions, source generation, canonical configuration identities,
  pinned catalog identity, and each selected legacy file checksum are committed
  to the report. Discrepancy approvals require the exact sorted field set and
  fresh artifact/data plus legacy-file digests; stale or broadened approvals
  fail closed.
- **Parity contract:** the implementation compares the selected legacy view,
  exact masks/order/populations and declared dtypes before numerical values. It
  preserves visible nonzero differences within the frozen tolerance contract,
  reconstructs calls, parity puts, and ATM volatility on the expanded branch/
  support grids, and keeps the unavailable unsmoothed legacy state as an honest
  item-level oracle gap while enforcing the fresh DLV invariants through the
  fitted-partition validator and the tighter comparison round trip.
- **Publication:** reports are canonical JSON, created under a no-symlink
  directory policy, published with create-new hard-link semantics, directory
  fsync, winner preservation, and staging cleanup. Existing outputs, output
  locations beneath the oracle, and output locations in the Git tree are
  rejected.
- **Raw recovery:** verbose cache-only validation reconciles manifest,
  canonical query document, physical table, persisted Parquet schema/counts,
  generation, and payload digest without printing SQL or credentials. Tests
  demonstrate checksum tamper rejection before authorization and exact-
  generation continuation after both ordinary terminal failure and a
  `BaseException`, with completed immutable partitions unchanged and all owned
  connections closed.
- **Human operation:** the runbook fixes the 16-session/68-Raw-artifact pilot,
  two-connection and one-attempt limits, external SANOS-owned evidence paths,
  JSONL logs, cache-only and restart proofs, stop conditions, redaction review,
  and explicit Phase 11 block. It now repeats the daemon-backed resource gate
  immediately after the human readiness confirmation and before the sole live
  command.

## Commands and evidence

All commands below were offline; no WRDS or other network connection was made.

```text
env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider tests/test_reference_compare.py tests/test_wrds_download.py
55 passed in 38.18s

env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m compileall -q \
  src/sanos tests/test_reference_compare.py tests/test_wrds_download.py
PASS

ruff check src/sanos/reference/compare.py src/sanos/wrds/download.py \
  src/sanos/cli/main.py tests/test_reference_compare.py tests/test_wrds_download.py
All checks passed!

git --git-dir=.git-real --work-tree=. diff --check
PASS

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -c \
  'from sanos.reference.compare import compare_daily_artifacts; value=str(compare_daily_artifacts.version); print(len(value), value[:5])'
29471 1.0 {
```

The task owner also reported the pre-remediation complete offline suite as 256
passing tests with two expected wildcard warnings. I did not launch another
five-minute full-suite run after remediation because the required resource
probe returned daemon-unavailable fallback state:

```text
gpuctl resources --json
all measured RAM/disk checks OK; source=local;
gpuctl: daemon not reachable; measuring locally
```

The focused suite covers every modified implementation path and all remediation
regressions. The unavailable daemon is not a defect in this repository and does
not invalidate the offline code verdict, but under the host guardrails it is a
hard prerequisite for, and currently blocks, the live pilot.

## Residual risks and live gate

- Real WRDS authentication, 2FA timing, current vendor table contents, and
  live account throttling remain deliberately untested. No offline fake can
  authorize or substitute for the owner-present gate.
- The expected 14 match / one policy-empty / one oracle-gap result is a live
  pilot expectation, not an offline claim. Any vendor revision remains an
  unexplained discrepancy until separately investigated and explicitly approved
  with the exact evidence record.
- Phase 10 remains incomplete as an operational phase until the external pilot
  evidence bundle is produced and independently signed. This PASS permits the
  offline implementation to be committed; it does not permit Phase 11 to start.

Signed: `code` — independent Phase 10 code/engineering reviewer
