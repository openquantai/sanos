# Public-candidate SANOS/DLV Dataset

The public projection is a distinct, deliberately lossy Dataset built from one
complete immutable internal Dataset. It is not an alias and never reads Raw or
Intermediate artifacts. The first materialization has distribution status
`public_candidate`: it has passed technical schema, leakage, reconstruction,
and capability checks, but it is **not approved for redistribution**.

## Build without WRDS

The checked reference configuration names an exact internal Dataset pointer:

```console
sanos --root /path/to/quant-data --dry-run dataset public build \
  --config configs/sanos-public-candidate-reference.json
sanos --root /path/to/quant-data -v dataset public build \
  --config configs/sanos-public-candidate-reference.json
```

Dry-run planning is location-independent, writes nothing, and does not open the
source. A real build first shallowly and semantically validates that exact
internal Dataset, then projects it atomically. No command in this group opens a
WRDS connection.

The candidate configuration owns `coverage_start_date`, `coverage_end_date`,
and the exact allowed typed-empty date/reason records. Published plus omitted
dates must equal every NYSE session in that range. A new or removed omission is
therefore a new reviewed identity, not an editable validation note.

For automation, add `--log-format json`. Events contain progress counts and the
final result contains the candidate identity, path, distribution, whether it
was built or reused, whether the source was accessed, and the validation
summary.

## Validate and reuse

Validate by configuration or by exact identity:

```console
sanos --root /path/to/quant-data dataset public validate \
  --config configs/sanos-public-candidate-reference.json
sanos --root /path/to/quant-data dataset public validate \
  --name sanos-spx-2022-2025-public-candidate \
  --artifact-id <sha256>
```

Normal validation is shallow with respect to lineage. It checks the candidate's
own checksums, exact payload allowlist, deterministic documents, exact recursive
Parquet schema, finite values, identity-owned coverage, model geometry, density
normalization, corrected DLV inverse, off-grid SANOS/DLV pricing, derived ATM
volatility, normalized books, exact NMS bounds, capability declarations, and
the persisted ordered-return and complete logical-content digests. It runs with
`WRDS_OFF=1`. The internal source, Intermediate, and Raw roots may therefore be
absent.

Use `--deep-lineage` only when the exact internal source and its ancestors are
expected to remain available. Deep validation re-runs the exact projection from
the internal Dataset and compares every ordered nested value and omission; this
is the mode which independently re-establishes private-input CD-004 returns and
CD-009 bridges. Shallow validation establishes artifact integrity and advertised
public-only capabilities, but cannot recreate private source inputs which are
deliberately absent. `--integrity-only` skips numerical capability execution.
`--version '*'` is an explicit warning-emitting reuse override and cannot build
a missing candidate.

## Physical and logical contract

The artifact contains:

- `data.parquet`: one Polars row per populated retained date;
- `README.md`: a portable human-readable summary;
- `config.json`: the exact projection identity;
- `schema.json`: recursive fields, units, return, and pricing conventions;
- `capabilities.json`: a closed list of supported and unsupported uses;
- `validation.json`: recomputable population evidence, typed-empty exclusions,
  return-horizon evidence, and exact return/content/coverage digests; and
- store-owned checksummed `manifest.json` and completion marker.

Each row contains only the trading date and half-day flag, the approved
forward-adjusted return, expiry geometry, normalized side-separated books,
complete common-surface smooth SANOS state, and complete common-surface
unsmoothed DLV state. The public state, interpreting geometry, and normalized
books remain in the same Polars frame; source fit diagnostics stay in the
internal Dataset and are not public fields. There are no separate per-date
quote, expiry, or surface files.

The approved units are:

```text
normalized strike             K/F
model call price              C/(DF*F), reconstructed in memory
half_spread_spot              (ask-bid)/(2*S)
discounted_forward_over_spot  DF*(F/S)
C/S                           [DF*(F/S)] * [C/(DF*F)]
h/(DF*F)                      [h/S] / [DF*(F/S)]
```

The first return is null. Later rows use consecutive populated public dates:

```text
r_t = (S_t/S_previous) / (F_previous(t)/S_previous) - 1
```

The previous date's positive forward and discount curves have an `(0,1)`
anchor and are interpolated log-linearly in NYSE business-session DTE. There is
no extrapolation. A typed-empty date is omitted, recorded in validation
evidence, and the next return uses the preceding populated public row and its
multi-session horizon. A half-day counts as one session.

## Deliberate exclusions

The exact recursive allowlist prevents pass-through. The Dataset has no
absolute spot, forward, strike, or price; no raw bid, ask, or midpoint; no
ticker, security, option, contract, or vendor identifier; no volume, open
interest, or liquidity diagnostic; no vendor implied volatility or Greek; and
no stored per-book fitted price or implied volatility.

It supports normalized SANOS/DLV pricing and dynamics, model ATM volatility,
daily-front normalized books, and spot-one spread experiments. It does not
claim exact vendor-book reconstruction, quote-containment replay, liquidity
models, absolute-dollar tick effects, expiry-fit-mode pricing, or contract
replay.

## Rights gate

Technical normalization does not establish redistribution rights. Changing a
candidate to `public` requires a new immutable materialization with a release
license, redistribution-rights basis, approver, approval date, durable approval
reference, and exact candidate/source equality evidence. This repository does
not infer or fabricate those facts. Until they are supplied and reviewed, keep
the candidate on non-public storage.

Once a human has those facts, create a strict release configuration from the
exact candidate and exact license bytes:

```console
sanos --root /path/to/quant-data dataset public approval-template \
  --candidate-name sanos-spx-2022-2025-public-candidate \
  --candidate-artifact-id <candidate-sha256> \
  --release-license '<license identifier>' \
  --redistribution-basis '<rights basis>' \
  --approver '<approving authority>' \
  --approval-date YYYY-MM-DD \
  --approval-reference '<durable reference>' \
  --license-file /path/to/LICENSE.txt \
  --output /path/to/release-config.json
```

Review that generated file and then publish the distinct release:

```console
sanos --root /path/to/quant-data --dry-run dataset public approve \
  --config /path/to/release-config.json
sanos --root /path/to/quant-data dataset public approve \
  --config /path/to/release-config.json \
  --license-file /path/to/LICENSE.txt
sanos --root /path/to/quant-data dataset public validate \
  --distribution public --name sanos-spx-2022-2025-public-candidate \
  --artifact-id <public-sha256>
```

Approval reprojects from the same internal Dataset, proves exact logical
equality with the approved candidate, validates the entire staged release, and
publishes under `sanos/public/`. It never aliases or mutates the candidate. The
release then remains shallowly reusable without candidate or source artifacts.
The repository intentionally contains no real approval document or license.
