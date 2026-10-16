# Public-candidate SANOS/DLV Dataset

The public projection is a distinct, deliberately lossy Dataset built from one
complete immutable internal Dataset. It is not an alias and never reads Raw or
Intermediate artifacts. The first materialization has distribution status
`public_candidate`: it has passed technical schema, leakage, reconstruction,
and capability checks, but it is **not approved for redistribution**.

## Build without WRDS

The checked reference configuration is a policy template. After building the
internal Dataset, resolve the graph and emit run-specific immutable configs:

```console
sanos --root /path/to/quant-data production finalize-configs \
  --dataset-config configs/widb-spx-reference.json \
  --public-template configs/sanos-public-candidate-reference.json \
  --experiment-template configs/dysanos-daily-front-reference.json \
  --output-directory /path/to/run-configs
sanos --root /path/to/quant-data --dry-run dataset public build \
  --config /path/to/run-configs/public-candidate.json
sanos --root /path/to/quant-data -v dataset public build \
  --config /path/to/run-configs/public-candidate.json
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
  --config /path/to/run-configs/public-candidate.json
sanos --root /path/to/quant-data dataset public validate \
  --name sanos-spx-2022-2025-public-candidate \
  --artifact-id <sha256>
```

Normal validation is shallow with respect to lineage. It checks the candidate's
own checksums, exact payload allowlist, deterministic documents, exact flat
Parquet schema, finite values, identity-owned coverage, model geometry, exact
initial full-surface support/density state, density normalization, full-support
calendar order, and off-grid SANOS/DLV pricing,
smooth-SANOS-derived ATM volatility, normalized books, exact NMS bounds,
capability declarations, and
the persisted ordered-return and complete logical-content digests. It runs with
`WRDS_OFF=1`. The internal source, Intermediate, and Raw roots may therefore be
absent.

Use `--deep-lineage` only when the exact internal source and its ancestors are
expected to remain available. Deep validation re-runs the exact projection from
the internal Dataset and compares every ordered scalar value and omission; this
is the mode which independently re-establishes the private-input CD-030 returns
and CD-029 driftless books. Shallow validation establishes artifact integrity and advertised
public-only capabilities, but cannot recreate private source inputs which are
deliberately absent. `--integrity-only` skips numerical capability execution.
`--version '*'` is an explicit warning-emitting reuse override and cannot build
a missing candidate.

## Physical and logical contract

The artifact contains:

- `data.parquet`: one tagged long-form scalar Polars frame;
- `README.md`: a portable human-readable summary;
- `config.json`: the exact projection identity;
- `schema.json`: flat fields, row kinds, units, return, and pricing conventions;
- `capabilities.json`: a closed list of supported and unsupported uses;
- `validation.json`: recomputable population evidence, typed-empty exclusions,
  return-horizon evidence, and exact return/content/coverage digests; and
- store-owned checksummed `manifest.json` and completion marker.

The `day`, `expiry`, `book`, `smooth_node`, and `dlv_node` rows collectively
contain only the trading date and half-day flag, the approved driftless return,
expiry geometry, normalized side-separated books, complete smooth SANOS state,
and complete initial full no-arbitrage common-surface support/density DLV state.
The DLV state is copied exactly from the strictly validated internal state
rather than regenerated during export; zero-mass nodes are not compressed and
finite squared-DLV coefficients are not invented.
The public state, interpreting geometry, and normalized
books remain in the same Polars frame; source fit diagnostics stay in the
internal Dataset and are not public fields. There are no separate per-date
quote, expiry, or surface files.

The approved units are:

```text
normalized strike  K/F
model call price   C/(DF*F), reconstructed in memory
half_spread        (driftless ask-driftless bid)/2 in C/(DF*F)
```

The first return is null. Later rows use consecutive populated public dates:

```text
r_t = DF_previous(t) * S_t / F_previous(t) - 1
```

The previous date's positive forward and discount curves have an `(0,1)` anchor
and are interpolated log-linearly in NYSE business-session DTE. There is no
extrapolation. A typed-empty date is omitted, recorded in validation evidence,
and the next return uses the preceding populated public row and its multi-session
horizon. A half-day counts as one session. Cumulative multiplication of `1+r_t`
reconstructs a normalized driftless spot path starting at one.

## Deliberate exclusions

The exact flat scalar allowlist prevents pass-through. The Dataset has no
absolute spot, forward, strike, or price; no raw bid, ask, or midpoint; no
ticker, security, option, contract, or vendor identifier; no volume, open
interest, or liquidity diagnostic; no vendor implied volatility or Greek; and
no stored per-book fitted price. Book rows do store six model-derived Black
IVs: SANOS and DLV fit IV plus synthetic fit-minus-half-spread and
fit-plus-half-spread IVs. A non-identifiable synthetic bound is null. Custom
configurations retaining a zero-time book store all six as null; the canonical
public and DySANOS 1.1 configurations exclude every 0DTE book. These are not
observed market or vendor IVs.

Day rows store SANOS and DLV market-quality triples computed privately against
the omitted source bid/ask: percentage of books outside, and conditional mean
and median excess in market half-spreads. No per-book market excess or source
quote is published.

It supports normalized SANOS/DLV pricing and dynamics, model ATM volatility,
daily-front normalized books, and driftless spread experiments. Smooth SANOS
and unsmoothed DLV are independent market fits; no cross-model price equality
or projection-error guarantee is claimed. It does not
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
