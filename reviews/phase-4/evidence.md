# Phase 4 evidence — guarded WRDS raw layer

Review state: complete; independent code and quantitative/data-contract reviews
passed after remediation. No live WRDS connection, external network access,
GPU process, or reference-repository write occurred.

## Implemented contract

The raw layer now provides:

- deterministic NYSE-session planning with explicit previous-session spot,
  forward, and zero-curve boundary partitions;
- reference-compatible `optionm_all`/`optionm` library and physical-table
  candidate resolution;
- safe SQL generation and minimally transformed Polars/Parquet output in
  vendor units;
- per-artifact generated `README.md`, `query.json`, manifest, completion marker,
  checksums, and immutable source-generation identity;
- output-first exact and wildcard cache reuse before credentials or optional
  WRDS imports are read;
- dry-run and cache-only modes that cannot enter the network path;
- an immediate interactive/non-interactive 2FA-readiness gate;
- a closed single-connection access/catalog probe before bounded fan-out;
- an account-keyed, username-redacted, host-wide four-slot kernel-backed lease;
- conservative error classification that never retries authentication, 2FA,
  permission, connection-limit, or unknown failures and retries only recognized
  transient queries;
- atomic, durable partition publication and race-safe reuse if another SANOS
  process publishes the same request first;
- strict date, ticker, and secid partition validation, including authoritative
  reuse of the winning security identity during a concurrent publication race;
- durable exact-schema zero-row vendor results with explicit empty markers and
  persisted Polars dtypes in source documentation and provenance;
- immutable refresh generations and restart from completed partitions; and
- text or JSON progress on completion/failure and a 30-second heartbeat around
  blocking connection, catalog, query, and close operations.

The installed CLI exposes source, ticker, date range, logical-table subset,
source generation, refresh, cache-only validation, `version="*"` reuse,
connection/retry limits, and explicit non-interactive 2FA confirmation. Direct
secid override remains deliberately unavailable while CD-010 is unresolved.

## Offline verification

The complete ordinary suite runs with bytecode and pytest cache writes disabled:

```text
114 passed
```

Focused raw tests cover session/boundary closure, table precedence, SQL and raw
strike units, schema and partition mismatch, credential redaction, `WRDS_OFF`, transient-only
retry, four-slot lease exhaustion/recovery, dry-run, cache-only miss, 2FA gate,
probe-before-fanout lifecycle, two-worker bounds, generated documentation,
output-first reuse without credentials, resumable immutable refresh, wildcard miss,
table subsets, and CLI structured behavior.

A two-downloader race regression deliberately returns conflicting ticker-to-
secid resolutions before competing to publish the same identity artifact. Both
downloaders subsequently query using the single winning artifact's secid.

The full suite initially exposed a real concurrent namespace-parent creation
race in the Phase 2 artifact store when two raw workers targeted dates in the
same year. Directory creation now treats a concurrent `FileExistsError` as a
race, revalidates the winning path's type/symlink safety, and proceeds. A new
eight-worker/16-artifact regression covers that shared-parent case.

Static gates:

```text
ruff check src tests
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 46 source files

python -m compileall -q src tests scripts
git diff --check
```

All completed successfully. The temporary mypy package is isolated below the
project-specific `/tmp/sanos/` tree.

A module-command smoke test ran `python -m sanos` for `doctor --strict-locations` and a one-day
SPX JSON dry run under `/tmp/sanos/phase-4-smoke`. Doctor reported the
optional WRDS package available. The plan reported eight artifacts, including
three visible 2021-12-31 boundary partitions for the 2022-01-03 start, and both events recorded
`network_accessed=false`; the configured root did not exist afterward.

## Live-access boundary

No live test has been attempted. A live pilot is a Phase 10 material operation:
it first requires `gpuctl resources --json`, then an immediate explicit user
confirmation of WRDS 2FA readiness, and begins with one representative SPX date
at the default two-connection limit. Ordinary tests will remain offline.
