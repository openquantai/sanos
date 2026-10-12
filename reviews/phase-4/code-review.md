# Phase 4 independent code review

**Verdict: PASS**

**Reviewer:** `/root/code` (independent code-agent gate)
**Date:** 2026-10-12
**Scope:** current uncommitted Phase 4 guarded WRDS raw layer and the small
Phase 2 artifact-store concurrency fix it requires. No live WRDS connection or
write to `quant_data` occurred.

## Findings

No high-, medium-, or low-severity blocking finding remains. The final tree
meets the Phase 4 code acceptance gate.

## Review conclusions

1. **The network boundary is fail-closed and output-first.**
   `download_raw` resolves and validates every requested artifact before it
   invokes the authorization callback or reads credentials
   (`src/sanos/wrds/download.py:403-510`). Dry-run, cache-only, complete exact
   reuse, and wildcard reuse therefore stay outside the live path. Wildcard
   misses cannot build. The CLI asks for immediate 2FA readiness only after a
   real miss and requires `--confirm-2fa-ready` for non-interactive callers
   (`src/sanos/cli/main.py:464-498`). `WRDS_OFF` is enforced before the optional
   WRDS package is imported or a connection is created
   (`src/sanos/wrds/client.py:66-85`, `:136-146`).

2. **Connection concurrency and failure handling are safe.**
   The account-keyed lease exposes no username and enforces exactly four
   kernel-backed cross-process slots (`src/sanos/wrds/lease.py:37-118`). Its
   metadata-publication failure path releases the acquired lock before
   propagating (`:103-112`), and cleanup cannot mask the caller's exception.
   The one-connection catalog probe closes before bounded fan-out, so a single
   SANOS invocation and multiple SANOS invocations remain within the same
   account-wide ceiling (`src/sanos/wrds/download.py:597-714`). Authentication,
   2FA, permission, connection-limit, and unknown failures are not retried;
   recognized transient query retries are bounded at five attempts
   (`src/sanos/wrds/client.py:97-133`, `:162-191`).

3. **Raw artifacts are immutable, durable, and correctly keyed to their
   contents.**
   The planner exposes the prior-session spot/forward/zero dependencies, and
   the CLI JSON plan names each one. Query results must match the exact ticker,
   observation date, and resolved secid before publication
   (`src/sanos/wrds/download.py:141-211`). Legitimate empty dated extracts are
   retained and explicitly marked rather than repeatedly queried. Parquet is
   fsynced before the artifact builder durably publishes the staged tree
   (`:230-298`). The artifact store's shared-parent creation now handles a
   concurrent winner and then revalidates its type and symlink safety
   (`src/sanos/artifacts/store.py:486-508`). Exact-artifact publication races
   validate and reuse the winner.

4. **The security-identity race is coherent.**
   A downloader validates its local security response, publishes or reuses the
   immutable winner, and then always derives the downstream secid from that
   returned artifact (`src/sanos/wrds/download.py:624-638`). The adversarial
   two-downloader barrier test supplies conflicting local secids and proves
   that both downstream SQL streams use the single published winner
   (`tests/test_wrds_download.py:359-409`).

5. **Resume, refresh, provenance, and version identity match the repository
   contract.**
   Re-running an exact generation, including an interrupted `--refresh`, reuses
   complete partitions and downloads only misses; no path overwrites a
   published raw artifact. A genuinely new extraction requires a new source
   generation. Every artifact records the physical source, exact SQL and hash,
   persisted schema, row/empty status, checksums, and generated field README.
   The raw producer identity includes query generation, catalog selection,
   security resolution, result normalization, publication, and namespace
   helpers while excluding lease/retry/progress-only behavior
   (`src/sanos/wrds/download.py:351-383`; `src/sanos/wrds/source.py:229-263`).

6. **The CLI and documentation are usable without an agent.**
   The installed/module command supports source, ticker, date range, table
   subset, cache-only validation, immutable refresh generation, bounded
   connections/retries, dry-run, verbose mode, and line-delimited JSON. Errors
   retain meaningful exit codes and structured form. The runbook documents
   setup, no-network planning, cache validation, 2FA, live operation, resume,
   refresh, artifact contents, and the absence of any completed live pilot.
   Direct secid override remains visibly blocked by CD-010; this PASS does not
   approve or resolve that open semantic choice. `PROGRAM_PLAN.md:342-344`
   accurately reflects that gate instead of claiming the option exists.

## Independent verification

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
114 passed in 16.21s

WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_wrds_download.py
14 passed in 3.61s

ruff check src tests
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 46 source files

PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
(exit 0)

git --git-dir=.git-real --work-tree=. diff --check
(exit 0)

PYTHONPATH=src WRDS_OFF=1 python -m sanos --root \
  /tmp/sanos/code-review-phase4 --dry-run --log-format json raw download \
  --ticker SPX --start-date 2024-01-03 --end-date 2024-01-03
(exit 0; eight misses and three explicit boundary dependencies;
`network_accessed=false`; configured root not created)
```

The suite covers partition planning and row-key rejection, empty extracts,
schema/provenance generation, exact/wildcard output-first reuse, refresh
resume, atomic/concurrent publication, the security-winner race, authorization,
offline modes, retry classification, lease exhaustion/recovery, heartbeat
progress, connection bounds, structured CLI output, and resolved-version
closure.

## Residual risks

- No live WRDS connection has been attempted, by design. Actual WRDS package
  authentication prompts, account catalog shape, returned pandas dtypes, and
  vendor throttling remain Phase 10 pilot risks. The pilot must retain the
  immediate user 2FA confirmation and begin with one representative date.
- Connection-level transient messages such as a reset are retried on the same
  client object. A dead client may therefore exhaust its bounded attempts and
  fail safely rather than reconnect. Automatic reconnect would create a new
  authentication/2FA and connection-budget policy and should not be introduced
  silently.
- Retry classification is necessarily message-based. Unrecognized failures are
  classified permanent and are not retried, which favors account safety over
  availability.
- CD-010 remains open. Ticker-only SPX reproduction is implemented; direct
  secid semantics require explicit user approval before being added.

These are visible, fail-safe follow-up risks and do not block the reviewed
offline Phase 4 contract.
