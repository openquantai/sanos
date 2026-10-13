# WRDS runbook

The first live extraction must follow the stricter
[Phase 10 guarded pilot](phase-10-pilot.md). The generic commands below do not
replace its fresh-generation, evidence, comparison, or review gates.

The guarded raw OptionMetrics layer is implemented. It has been exercised only
with offline connection fakes so far; no live WRDS query has been made from this
repository. A live pilot needs a separate, immediate confirmation that the user
is ready to complete 2FA.

## Install and configure

Install the optional WRDS client and configure storage outside the Git tree:

```bash
python -m pip install -e '.[wrds]'
export QUANT_DATA_ROOT=/path/to/quant-data
export WRDS_USERNAME=your_existing_wrds_username
sanos doctor --strict-locations
```

SANOS uses the existing WRDS credential mechanism and never writes the username,
password, 2FA response, or other credentials to an artifact, log, README, or
query record. `WRDS_OFF=1` disables live access while leaving planning and cache
validation available.

Configure the standard libpq password file from a private interactive terminal
before running a live SANOS command. Use the same Python environment that will
run SANOS:

```bash
read -r -p "WRDS username: " WRDS_USERNAME
export WRDS_USERNAME
python -c \
'import getpass, os, wrds; c = wrds.Connection(autoconnect=False, wrds_username=os.environ["WRDS_USERNAME"], wrds_password=getpass.getpass("New WRDS password: ")); c.create_pgpass_file()'
unset WRDS_USERNAME
stat -c 'mode=%a file=%n' "$HOME/.pgpass"
```

The password prompt is hidden. Do not put a password in chat, shell history, an
environment variable, a command argument, or a SANOS configuration file. The
final mode must be `400` or `600`. The setup command only updates `.pgpass`; it
does not connect to WRDS or trigger 2FA. If a password has been exposed, rotate
it before storing the replacement.

SANOS opens WRDS in fail-closed mode. A missing, stale, or rejected `.pgpass`
entry produces a sanitized error and never falls back to an interactive username
or password prompt inside a command whose output may be logged.

The connection URL explicitly selects the `psycopg2` driver and omits the
password field rather than inserting an empty password. This prevents current
SQLAlchemy from selecting psycopg 3 for a bare `postgresql` URL when only the
WRDS-supported psycopg2 driver is installed. It also lets libpq perform its
standard `.pgpass` lookup without SANOS reading, parsing, or holding the password
itself; an explicit empty password prevents that credential-file lookup.

## Plan without network access

Always inspect a substantial request first:

```bash
sanos --dry-run raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29
```

The dry run neither reads WRDS credentials nor creates a data root. It reports
the requested NYSE sessions, cache hits and misses, and boundary partitions.
For the first date, the boundary closure includes the fresh previous-session
close needed by the internal legacy-compatible return. CD-004 makes the first
public return null, so it requires no pre-start forward or zero curve. A planner
may report such inputs as conservative over-fetch, but not as a requirement.

Use `--table options` (repeatable) to select logical tables. Security identity
is always included. Direct secid override is deliberately unavailable while the
legacy ignored-secid defect remains open under CD-010; ticker resolution is
explicit and recorded.

## Validate cache only

```bash
sanos raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2022-01-03 --end-date 2025-08-29 \
  --cache-only
```

This validates manifests, completion markers, configuration identity, producer
version, and payload checksums. A miss exits nonzero and never opens WRDS. Use
`--version '*'` only to reuse complete artifacts while ignoring producer-code
version; it cannot build a miss.

## Perform a live download

An interactive terminal asks whether you are ready for WRDS 2FA immediately
before the first connection. The default answer is no:

```bash
sanos -v raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2024-01-03 --end-date 2024-01-03
```

Only a non-interactive operator who is genuinely ready for immediate 2FA should
pass the explicit equivalent:

```bash
sanos --log-format json raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2024-01-03 --end-date 2024-01-03 \
  --confirm-2fa-ready
```

The command first opens one leased authentication/catalog probe, then closes it
before fan-out. The default fan-out is two connections. Both the CLI and Python
configuration reject a requested limit above four before connecting, and a
host-wide account-keyed four-slot lease enforces that limit across SANOS
processes. Authentication, 2FA, connection-limit, and permission failures are
never retried; only recognized transient query failures receive bounded retry.
If authentication fails, stop the command and update `.pgpass` from a private
terminal before obtaining a fresh immediate 2FA-readiness confirmation.

Progress is flushed as partitions finish and at least every five minutes while
a connection or query is blocked. JSON mode emits one object per line with
elapsed time, completed and total partitions, failures, and ETA when meaningful.

## Resume and refresh

Every raw partition is immutable and published atomically. Re-running the same
command validates and reuses complete partitions before reading credentials;
only missing partitions enter the guarded live path. An interruption can
therefore be resumed with the identical command.

Refresh never overwrites a completed extraction. Name a new immutable source
generation explicitly:

```bash
sanos raw download \
  --source optionmetrics --ticker SPX \
  --start-date 2024-01-03 --end-date 2024-01-03 \
  --refresh --source-generation 2026-10-12-pilot-1
```

Reusing that exact generation with `--refresh` validates and reuses its complete
partitions and downloads only missing ones, so an interrupted refresh is
resumable. To force a genuinely new extraction, choose a new source generation;
an existing immutable generation is never overwritten.

## Raw artifact contents

Artifacts live below `raw/wrds/optionmetrics/` and are partitioned by logical
table, ticker, year, and observation date (security identity has no date). Each
artifact contains:

- `data.parquet`, with the selected vendor values in raw units;
- `README.md`, describing source, fields, units, date, and row count;
- `query.json`, containing non-secret physical-table and SQL provenance;
- `manifest.json` and `COMPLETE.json`, containing identity and checksums.

Option strikes remain in the vendor's stored raw units at this boundary. Strike
scaling, field aliases, filtering, normalization, and fitting belong to later
daily stages. The resolved account catalog prefers `optionm_all` over `optionm`
and preserves the reference table-candidate order for `securd`, annual
`opprcd`, `secprd`, annual `fwdprd`, and `zerocd`.
