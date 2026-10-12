# Phase 5 final quantitative/data-logic review

**Verdict: PASS**

Reviewer: independent `quantagent`
Date: 2026-10-12
Pinned reference revision: `de15be96d2f283b6c31e60d3ea47325a0c093ff0`

I re-reviewed the settled Phase 5 tree after the CD-020 compatibility
remediation. I used offline fixtures only; I did not access WRDS or any network,
modify `quant_data`, or modify implementation files.

Phase 5 meets its acceptance criteria and may be committed. There are no open
quantitative or data-contract blockers in this phase.

## CD-020 disposition

The initial migration now preserves the pinned observable behavior. The pinned
`enrich_day` accidentally leaves its internal `is_valid` control column in an
all-valid day and then fails schema validation. `enrich_daily_frame` reproduces
that failure boundary explicitly with a targeted `DailyDataError` mentioning
CD-020, without leaking a temporary control column into the Polars schema.

This is the correct compatibility treatment under `AGENTS.md`: it does not
silently repair the suspected reference defect. A targeted regression constructs
an all-valid book and pins the failure. The optional repair remains documented
in CD-020 and still requires explicit user approval, a version change, and a
fresh-build population-impact report. That optional future decision does not
block the parity-preserving Phase 5 implementation.

## Quantitative findings

### Reference parity and numerical conventions

- The checked-in synthetic oracle freezes every ordered enriched Float32 vector
  from the pinned implementation. Float32-first spot return, Float32 TTE in
  fixed NMS, implied forwards, IVs, synthetic calls, pure prices, fixed/ATM NMS,
  and all Greeks remain pinned.
- The quote-cleaning and pairing sequence, strike conversion, pairing keys,
  duplicate behavior, zero-rate percent units, continuous ACT/365 discounting,
  vendor-forward matching, and analytical inverse-spread-weighted `DF*F`
  projection retain reference semantics.
- Put-derived calls, minimum-spread call blending, `K/F`, `C/(DF*F)`, liquidity
  aggregation, Black inversion, boundary-only IV rejection, Greeks, and NMS
  retain their reviewed units and invariants.
- Float32 persistence boundaries are deliberate and covered. The cdxcore Black
  adapters, tolerances, schemas, bad-data registry, and clock/NMS policy are
  included in the producer-version dependency closure.

### Clock and overnight variance fraction

- `overnight_variance_fraction` is canonical and versioned, defaults to `0.2`,
  and implements the reference AM subtraction `1 - fraction`; PM timing is
  unchanged.
- Tests cover `0.1`, `0.2`, `0.5`, and both endpoints, pin numerical effects,
  and establish distinct paired/enriched artifact identities. The fraction-zero
  next-session AM IV-boundary failure and fraction-one full-business-day result
  are explicit.
- Trading-session half-day status remains separate from expiry AM/PM settlement.
  Downstream propagation through no-arbitrage, DLV and experiment construction
  remains correctly assigned to later phases.

### Daily frame, diagnostics and artifact behavior

- Processing remains one Polars frame per trading day at each stage. The paired
  artifact closes over six Raw inputs and the enriched artifact over the paired
  artifact; no parallel quote/expiry/surface storage contract was introduced.
- Ordered filter counts, unmatched/repaired forwards, call/put/unpaired/paired
  counts, row and expiry rejections, output populations, and numerical ranges
  are structured diagnostics. Known-bad typed-empty artifacts record their
  reason and policy version; nonempty paired inputs cannot silently become empty.
- Output-first reuse, actual-target wildcard reuse, restart journals, atomic
  publication, safe runner labels, and vendor ticker identity do not change
  numerical content.

## Deferred work is not a Phase 5 blocker

No no-arbitrage repair, SANOS/DLV fit, public projection, public spread unit, or
proposed forward-adjusted public-return choice entered Phase 5. CD-004 and
CD-009 remain open for their assigned phases. The final 891-date/book-count
comparison requires the guarded fresh WRDS build and later pipeline stages.
Live WRDS work remains gated on immediate 2FA readiness and the Phase 4
account-wide connection limit.

## Independent verification

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  -p no:cacheprovider tests/test_widb_reference_parity.py \
  tests/test_widb_daily.py tests/test_widb_build.py
# 26 passed in 13.09s

WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
# 145 passed in 31.13s

ruff check src tests scripts
# All checks passed

python -m compileall -q src tests scripts
# pass

git --git-dir=.git-real diff --check
# pass
```

`mypy` remains unavailable in the active interpreter; no type-check pass is
claimed. This does not alter the quantitative verdict.

Signed: `quantagent` — independent quantitative/data reviewer
