# Phase 4 quantitative/data-logic review

**Final verdict: PASS**

Reviewer: independent `quantagent`
Date: 2026-10-12
Scope: guarded WRDS raw layer and its offline tests/documentation. No live WRDS
connection, external network access, GPU work, or write to `quant_data` was
performed.

The original findings are retained below as an audit record. The remediation
re-review at the end of this document is authoritative for the final verdict.

## Findings

### 1. High — a genuine zero-row vendor result cannot become durable Raw evidence

`normalize_raw_frame` rejects every frame with the exact expected columns but
zero rows (`src/sanos/wrds/download.py:151-161`). This moves a daily-data
eligibility decision into the Raw layer and leaves no immutable artifact proving
that WRDS returned no observation. On resume, that partition remains a cache
miss and is queried again indefinitely. A single absent options, forward, spot,
or zero-curve observation can therefore prevent the fresh 2022–2025 extraction
from completing even though absence is itself valid source evidence that Phase
5 should classify and report.

This differs from the reference layering. The pinned `WIDBClient.options`,
`zero_curve`, and `forwards` query methods return the vendor frame as received;
`collect_raw_day` subsequently interprets empty options/zero/forward frames as
day-level failures (`reference/quant_data/quantdata/widb/day.py:223-236`). Raw
capture should not silently absorb that later processing policy. The security
identity is different: zero or ambiguous ticker-to-secid rows must still fail
because no other query can be constructed safely.

Required change: accept and atomically publish exact-schema zero-row frames for
dated raw tables, with `row_count=0` and an explicit empty-result marker in
manifest/query/README provenance. Keep strict security identity resolution.
Add offline tests proving an empty dated result is a complete reusable artifact,
requires no second connection on rerun, and is surfaced explicitly to the later
daily stage rather than treated as successful market data.

### 2. High — dry-run hides the actual boundary prerequisites required by the plan

The planner itself correctly produces the prior-session spot, forward and zero
curve partitions (`src/sanos/wrds/plan.py:135-175`). For the requested reference
interval it deterministically plans 918 NYSE sessions, 3,676 raw artifacts, and
the three boundary observations on **2021-12-31**. However, the only structured
`raw_plan` field is `boundary_partitions: 3`
(`src/sanos/wrds/download.py:390-405`). The actual tables, roles and date are not
emitted. An independent JSON CLI dry run confirmed that its complete output
contains the count but no `2021-12-31` or boundary partition labels.

This does not meet `PROGRAM_PLAN.md:344-348`, which requires boundary
prerequisites to be shown in dry-run output, and makes it impossible for a
human or agent to verify dependency closure before authorizing the long live
download.

Required change: include a stable structured list (or explicit per-partition
events) containing at least table, ticker, observation date and roles for every
boundary dependency. Add a CLI assertion for the exact 2022-01-03 closure:
SPX spot, forwards and zero curve on 2021-12-31, with no boundary option query.

### 3. Medium — generated field documentation does not define the raw units/schema it claims

The generated README says merely that vendor units are retained, and most
`RawField` descriptions name a value without its unit or stored type
(`src/sanos/wrds/source.py:63-138,257-283`). The most material example is
`strike_price`: the reference query divides the stored value by 1,000
(`reference/quant_data/quantdata/widb/client.py:296-304`), while the new Raw
query intentionally retains the unscaled vendor value. The current generated
text calls it only “stored raw units”; an agent cannot safely infer the required
Phase 5 conversion. The zero rate's percent/ACT-365 convention is documented,
but equivalent field-level clarity is absent for quote/spot/forward levels,
implied volatility, Greeks, flags, and nullability/storage dtype.

There is also a contract contradiction: `docs/data-contracts.md:42-46` says Raw
preserves exercise style, but the new option projection has no such column. The
reference did not read a vendor exercise-style field; it synthesized the
constant `'E'` in SQL. Omitting that synthetic value is appropriate for a
minimally transformed Raw layer, but the frozen European assumption must be
made explicit at the later compatibility stage rather than claimed as a
preserved source field.

Required change: make the generated source/schema Markdown agent-usable by
recording exact units/scaling and persisted dtypes (or a precise vendor-schema
reference) for every selected field. Explicitly document that raw
`strike_price` is converted by `/1000` to the reference cash strike in Phase 5.
Reconcile the exercise-style wording and state where the reference's synthetic
European flag is introduced and tested.

### 4. Medium — direct secid selection remains an unresolved Phase 4 contract

Phase 4 promises explicit selection by “ticker/secid”
(`PROGRAM_PLAN.md:341-343`), while the CLI exposes only ticker and always
resolves it through `securd`; the runbook explicitly says direct secid is
unavailable (`docs/wrds-runbook.md:41-44`). Deferring is safer than silently
repairing the legacy ignored-argument defect, but CD-010 remains open and the
phase cannot claim that part of its interface as complete.

Required resolution: obtain the user's CD-010 choice before implementing the
affected behavior, or explicitly narrow/defer the Phase 4 acceptance contract.
If direct secid is approved, define ticker/secid mismatch behavior and include
the resolved security identity in artifact configuration/provenance and tests.
The authoritative ticker-only SPX path should remain unchanged by either
choice.

## Confirmed correct

- The physical-table precedence matches the pinned WIDB reference:
  `optionm_all` before `optionm`; `securd` before `securd1`; annual `opprcd`
  before base; base `secprd` before annual; annual `fwdprd` before base; and
  `zerocd` with tenor ordering.
- Raw SQL retains the material vendor columns and adds `secid`/observation date
  provenance. Strike scaling and forward/field aliases are correctly deferred
  instead of being silently applied at this boundary. The option, spot,
  forward, and zero-curve projections otherwise cover the fields used by the
  reference daily path.
- NYSE sessions use the same calendar library as the reference. The immediate
  prior session for 2022-01-03 is correctly 2021-12-31. Adding previous-date
  forward and zero curves is conservative input capture for proposed CD-004;
  it does not implement or approve the still-blocked forward-adjusted return.
- Ticker normalization and unique positive secid validation are fail-closed.
  Removing the misleading direct override from the current interface avoids
  reproducing CD-010 accidentally, although the explicit selection contract
  still needs the resolution above.
- Raw content identity closes over query construction, table/field contracts,
  normalization and durable publication, while retry/connection controls do
  not spuriously change artifact identity. Source generations are immutable;
  exact and wildcard reuse are output-first.
- The authentication probe is leased and closed before fan-out. The default is
  two workers, configuration rejects more than four, and the account-keyed
  kernel locks enforce an aggregate four-connection ceiling. The latest lease
  remediation releases its acquired lock if metadata publication fails.
- Credential/2FA/permission/connection-limit errors are not retried; only
  recognized transient queries are retried. Dry-run, cache-only and complete
  output reuse remain offline and credential-lazy.
- Atomic Parquet publication, payload/manifest checksums, completion-last
  semantics, durable fsync, race-safe publication, and the shared namespace
  parent race remediation all pass offline tests.

## Offline checks run

```text
WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
108 passed in 21.46s

WRDS_OFF=1 python -m pytest -q -ra -p no:cacheprovider \
  tests/test_wrds_plan_source.py tests/test_wrds_download.py \
  tests/test_wrds_client_lease.py tests/test_cli.py
28 passed in 5.04s

ruff check --no-cache src tests scripts
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 46 source files

python -m compileall -q src tests scripts
git --git-dir=.git-real --work-tree=. diff --check
# both exit 0

# Independent full-range planner probe
sessions=918; planned=3676; previous_session=2021-12-31
counts={forwards:919, spot:919, zero_curve:919, options:918, security:1}

# Independent exact-schema empty-frame probe
options/spot/forwards/zero_curve all raise RawDownloadError("query returned no rows")

# Independent JSON dry run
raw_plan reports planned=3676, boundary_partitions=3, but no boundary date/table list
```

## Residual live-access risk

No offline test can prove the account's current catalog, permissions, 2FA flow,
or the actual 2022–2025 vendor populations. After the findings above and both
reviews pass, Phase 10 should still begin with the documented one-date SPX live
pilot at two connections and compare physical tables, dtypes, units, row counts
and secid before authorizing the full extraction.

Signed: `quantagent` — independent quantitative/data reviewer

## Remediation re-review — 2026-10-12

**Verdict: PASS**

I independently re-read the current Phase 4 implementation, tests, plan and
contracts and reran the offline gates with `WRDS_OFF=1`. All four original
findings are resolved. This approval covers the ticker-resolved SPX Raw path;
it does not approve CD-004 or CD-010 and it does not claim that live WRDS
behavior has been verified.

### Finding resolution

1. **Resolved — durable empty dated results.** `normalize_raw_frame` now accepts
   a zero-row frame only after exact selected-column validation, while a zero or
   ambiguous security-identity result remains unusable through
   `security_secid` (`src/sanos/wrds/download.py:160-170,214-227,624-634`). Raw
   publication records `row_count`, `empty_vendor_result`, and the observed
   Polars schema in the manifest metadata and `query.json`; the generated
   README also states the empty result (`src/sanos/wrds/download.py:250-298`;
   `src/sanos/wrds/source.py:266-302`). The regression proves an empty options
   partition is complete and is reused on a second invocation without
   authorization, credentials, or a connection
   (`tests/test_wrds_download.py:219-293`). This preserves source evidence and
   leaves the day-level exclusion/error decision to Phase 5.

2. **Resolved — inspectable boundary closure.** The structured `raw_plan`
   event now includes table, ticker, date and roles for every boundary
   partition (`src/sanos/wrds/download.py:446-472`). The CLI regression fixes
   the 2022-01-03 boundary to SPX forwards, spot and zero curve on 2021-12-31,
   with no options boundary (`tests/test_cli.py:142-189`). My independent full
   reference-range probe produced 918 requested sessions and 3,676 artifacts:
   one security identity, 918 option dates, and 919 dates each for spot,
   forwards and zero curve. Its only boundary entries were the same three
   2021-12-31 partitions.

3. **Resolved — raw schema, units and compatibility wording.** Each generated
   field entry now carries the actual persisted Polars dtype. The source
   contract explicitly records vendor `strike_price` as integer thousandths
   with `/1000` deferred to the compatibility stage; underlying and forward
   levels use index/currency units, and the zero curve is continuously
   compounded ACT/365 in percent (`src/sanos/wrds/source.py:64-143,266-302`).
   `query.json` repeats the persisted schema and exact physical query
   provenance (`src/sanos/wrds/download.py:277-292`). The data contract now
   correctly says the reference `exercise_style='E'` is synthetic and will be
   introduced and tested in the later compatibility stage rather than claimed
   as a Raw vendor field (`docs/data-contracts.md:41-51`). The retained vendor
   Greeks and implied volatility deliberately remain vendor-labelled at this
   boundary; Phase 5 owns interpretation and normalization.

4. **Resolved by explicit scope decision — direct secid.** The Phase 4 contract
   now promises selection by source, ticker and date and explicitly blocks a
   direct secid override pending CD-010 (`PROGRAM_PLAN.md:342-344`). The
   runbook says the same. This is a safe deferral, not an implicit change to
   the authoritative ticker-resolved SPX path. Non-empty rows are checked
   against their date and resolved secid, and security rows against ticker
   before publication (`src/sanos/wrds/download.py:171-210`). In a competing
   security-publication race, each downloader re-reads the winning immutable
   artifact's secid before constructing downstream SQL
   (`src/sanos/wrds/download.py:629-638`); the conflicting-secid regression
   verifies both downloaders use that winner
   (`tests/test_wrds_download.py:359-409`).

### Additional quantitative/data checks

- Exact source-generation reuse remains output-first. `--refresh` requires an
  explicit non-default generation but reuses complete partitions of that same
  generation and downloads only missing ones; a new generation produces
  disjoint immutable identities (`src/sanos/wrds/download.py:403-433`). An
  independent interrupted-run probe failed after publishing two of eight
  partitions, then resumed the same refresh generation with `reused=2`,
  `downloaded=6`, and `missing=0`.
- Raw content identity includes catalog selection, table precedence, field
  contracts, date/ticker namespace helpers, security resolution,
  normalization, query construction and durable publication. My resolved
  version probe found `choose_optionmetrics_catalog`,
  `resolve_physical_table`, `raw_table_spec`, `_date_literal`,
  `_safe_ticker_label`, `security_secid`, `normalize_raw_frame`, and
  `_publish_raw_partition`; it excluded lease, retry and request-level
  operational controls. Thus operational tuning does not fork identical Raw
  content, while content-affecting changes do.
- Physical-table precedence, raw strike scaling, zero-rate convention,
  closed-probe-before-fan-out topology, account-wide four-slot bound,
  transient-only retries, atomic publication, and output-first offline reuse
  remain as confirmed in the initial review. No downstream filter, return,
  no-arbitrage, SANOS, DLV, public-projection, or overnight-variance semantics
  have leaked into Phase 4.

### Offline checks rerun

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
114 passed in 15.92s

ruff check --no-cache src tests scripts
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 46 source files

PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
git --git-dir=.git-real --work-tree=. diff --check
# both exit 0

# Independent full-range planner/version probe, with PYTHONPATH=src
planned=3676
counts={security:1, options:918, spot:919, forwards:919, zero_curve:919}
boundary=[forwards/2021-12-31, spot/2021-12-31, zero_curve/2021-12-31]
all required content-version helpers present; operational controls absent

# Independent interrupted exact-refresh-generation probe
first invocation: intentional permanent failure after two published artifacts
resume: planned=8, reused=2, downloaded=6, missing=0

# Independent one-day module CLI dry run, with WRDS_OFF=1 and PYTHONPATH=src
three explicit 2021-12-31 boundary dependencies; network_accessed=false
```

One initial smoke command without `PYTHONPATH=src` resolved an unrelated
installed `sanos` package and failed before executing repository code. It was
immediately rerun against this worktree with `PYTHONPATH=src`, producing the
results above; this is an environment-selection issue, not a Phase 4 failure.

### Residual risk

No network access was performed. Current account catalog names, actual returned
pandas/Parquet dtypes and null patterns, permissions, 2FA interaction, and live
vendor populations remain intentionally unverified until the Phase 10
one-session pilot. That pilot must still start only after immediate user 2FA
readiness confirmation and retain the account-wide connection guard. Empty
dated artifacts are source evidence, not proof that a day is analytically
usable; Phase 5 must classify them explicitly.

Signed: `quantagent` — independent quantitative/data reviewer
