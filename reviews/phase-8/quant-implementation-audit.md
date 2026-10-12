# Phase 8 quantitative/data implementation audit

**Reviewer:** `quantagent`
**Date:** 2026-10-12
**Scope:** current uncommitted public-candidate configuration, schema,
projection, numerical validation, publication, CLI, documentation, and tests;
offline only
**Verdict:** **FAIL / CHANGES REQUIRED**

The ordinary synthetic path implements the approved CD-004 equation and CD-009
units correctly, and the serialized smooth SANOS and corrected-DLV states are
substantively complete. Phase 8 is not ready to commit because shallow
validation does not bind non-first return values or typed-empty coverage
evidence, and the required independent selector/capability boundary matrix is
largely absent. I also found one configuration-domain inconsistency at the
zero overnight-fraction endpoint.

## Blocking findings

### 1. High — an arbitrary finite non-first CD-004 return passes re-signed semantic validation

`assemble_public_frame` correctly computes

```text
(S_now/S_previous) / q_previous(h) - 1
```

from consecutive retained rows, an NYSE-session horizon, positive log-linear
forward and discount curves, the `(0,1)` anchor, and no extrapolation
(`project.py:490-544`). The first row is correctly null.

The published return evidence, however, records only dates, session count, and
`discount_curve_validated=true`. `_validate_coverage_evidence`
(`validate.py:321-344`) never binds the actual non-first return value, and
`validate_public_frame` checks only null placement and finiteness. I changed a
valid second-row return to `42.0`, rewrote the Parquet file, and updated its
manifest/completion checksums. `validate_public_dataset` returned `passed`.

Required change: bind the ordered public return vector to immutable build
evidence (for example, exact per-date values or a versioned canonical digest)
and compare it during shallow semantic validation. Add a re-signed mutation
test for a finite non-first value. Keep private spot/carry/discount inputs out
of the public artifact. Separately retain independent formula tests for exact
nodes, interpolation, gaps, and no extrapolation, since public-only validation
cannot recreate private inputs.

### 2. High — fabricated typed-empty omissions pass shallow validation

`_validate_coverage_evidence` accepts any unique, ordered, non-published ISO
date with any nonempty reason (`validate.py:302-320`). It does not establish
that the date is an NYSE session in the source coverage, that the reason is the
versioned policy-owned typed-empty reason, or that `validation.json` agrees
with the manifest metadata's omission list.

Independent reproduction: on a valid one-row candidate, I added
`2023-12-29 / "fabricated"` to `validation.json`, changed the summary omission
count, and updated only the normal manifest/completion checksums. Shallow
semantic validation returned `VALIDATED 1` even though no such internal
partition had been omitted.

Required change: freeze enough coverage policy in the candidate identity or
self-contained evidence to validate exact planned/published/omitted coverage;
validate permitted reason/date semantics and reconcile the omission list and
counts across validation evidence and manifest metadata. Add re-signed tests
for an invented date, wrong reason, out-of-range/non-session date, metadata
mismatch, and removal of a real omission. Preserve the approved rule that only
policy-owned typed empties are omitted and every other empty/populated-with-no-
book day fails publication.

### 3. High — the daily-front selector and advertised capability matrix lack the required independent regressions

The implementation in `_book_payload` appears to apply the intended predicates
in the right order: `options_filter_eligible`, side activity, European style,
smooth availability and finite side fit, literal `0.2 +/- 1e-4`, inclusive
`0..70D`, and AF/DLV-derived inclusive NMS `[-3,2]`; calls precede puts and
each side survives independently. But no test changes any of those source
predicates or compares this second selector implementation to the checked-in
pinned oracle or the already-reviewed Phase 9 projection.

The current public tests also do not cover the approved acceptance matrix:

- exact/inside/outside `0.2` tolerance; zero, exact 70D, and over-70D;
- inclusive NMS `-3/+2` and one representable point outside each edge;
- inactive call/put independently, American style, unavailable and nonfinite
  side fits, and call-only/put-only survival;
- an expiry retained in the complete model state with no public book;
- a populated row after an omitted typed-empty date, proving the previous-
  retained-row multi-session CD-004 rule;
- non-flat multi-node forward and discount curves, exact carry nodes,
  weekends/holidays/half-days, inconsistent duplicate nodes, and both forward
  and discount no-extrapolation failures;
- independent CD-009 half-spread bridge assertions and zero/nonfinite bridge
  failures;
- public-only smooth and DLV calls versus source reconstructed calls at nodes
  and off-grid, puts/parity, ATM IV, and the complete CD-014 zero/one-minute/
  pre-first/between/post-final time matrix; and
- forbidden fields at every nested level and corresponding capability/non-
  capability execution after Raw, Intermediate, and internal Dataset roots are
  all unavailable.

`validate_public_frame` currently proves density mass/moment, corrected DLV
inverse residual, finite example prices, and a first-expiry ATM IV. Those are
valuable, but merely obtaining finite values is not the full advertised
reconstruction/parity gate.

Required change: add a hermetic, hash-pinned or independently fixed selector
oracle and the complete boundary/capability tests above. The Phase 9 checked-in
selector oracle can be reused as independent evidence, but expected values
must not be generated by `_book_payload` itself. Aggregate 891-date/fresh-WRDS
comparison remains correctly deferred to production; synthetic selector parity
does not require WRDS.

### 4. Medium — public configuration rejects a valid upstream clock endpoint

The source `ClockConfig` and program contract allow
`overnight_variance_fraction` in `[0,1]`, and the public build explicitly
requires equality with its source Dataset. `PublicProjectionConfig` instead
requires `(0,1]` (`config.py:114-115`). An independent construction with `0.0`
raises `ValueError`, so a valid internal Dataset using the configured zero
endpoint cannot receive a public projection even though nondefault values are
otherwise accepted and identity-bearing.

Required change: accept `[0,1]` and add endpoint/identity/source-clock tests, or
explicitly freeze this public profile to exactly `0.2`. The current mixed rule
of accepting arbitrary positive nondefaults but uniquely rejecting zero is not
the reviewed clock contract. The default must remain `0.2`.

### 5. Medium — shallow NMS validation is broader than the advertised inclusive population

Projection uses exact inclusive strike bounds, but `_validate_books` accepts
reconstructed NMS values up to `2e-6` outside either configured endpoint
(`validate.py:285-287`). No documented/versioned numerical argument or edge
fixture owns that widening. The approved contract requires the endpoints
included and one representable outside point excluded.

Required change: validate the exact producer decision in its persisted Float32
strike space, or document and identity-own a justified cast tolerance that is
proven not to admit the first representable producer-side outside point across
the expiry range. Add re-signed boundary mutations.

## Quantitative areas that pass

1. **Schema and units.** The one-row-per-day recursive Polars schema contains
   only date/half-day, forward-adjusted return, shared expiry geometry,
   side-separated normalized books, complete smooth SANOS state, and complete
   unsmoothed DLV state. Strikes are `K/F`, model calls are reconstructed in
   `C/(DF*F)`, half-spreads are `(ask-bid)/(2S)`, and the per-expiry bridge is
   `DF*(F/S)`.
2. **CD-009 production calculation.** Projection computes the bridge from the
   expiry discount factor and spot-normalized implied forward, computes the
   side-specific spot-one half-spread, and checks
   `(h/S)/[DF*(F/S)] == h/(DF*F)` against the internal pure books before
   serialization.
3. **State completeness.** Each smooth expiry contains state index, time
   increment, kernel input total variance, support, and density; each DLV
   expiry adds corrected annualized squared DLV with exact zero boundaries.
   Validation reconstructs both `SanosPricer` objects, checks support, mass and
   first moment, and performs the corrected DLV-to-density round trip at
   `1e-12`.
4. **Expiry clocks.** Public validation independently recomputes calendar DTE,
   business DTE, AM subtraction `1-overnight_fraction`, and `T=BDTE/252`.
   Trading-session half-day state is checked separately from expiry AM/PM.
5. **Lossiness.** The row schema excludes absolute levels, source/contract
   identifiers, quotes/mids, volumes/open interest, vendor Greeks/IV, and
   redundant fitted prices. The scale/identifier invariance fixture passes.
6. **Lifecycle.** The candidate has the internal Dataset as its sole direct
   upstream, builds only after internal semantic validation, and is reusable
   and publicly executable with the source absent under shallow validation.
   Wildcard remains reuse-only.
7. **Rights.** Every implementation and user-facing path says this artifact is
   `public_candidate`, not approved for redistribution. No redistribution
   right is inferred from normalization or leakage checks; a `public` release
   still requires a separate immutable materialization and explicit legal
   evidence.
8. **Reference semantics.** The fresh validated/promoted internal SANOS Dataset
   remains canonical. The 891-date artifact is only a migration oracle and is
   neither a public-build input nor co-equal authority.

## Commands and evidence

- Read all Phase 8 public config/schema/project/validate/publish/CLI code,
  checked configuration, public runbook, compatibility/data contracts, and
  public tests in the settled worktree.
- Focused offline suite:
  `PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider tests/test_public_dataset.py tests/test_cli.py -k public`
  -> `13 passed, 17 deselected` in `40.84s`.
- Independent adversarial return probe -> `VALIDATED 42.0 2`.
- Independent adversarial omission probe -> `VALIDATED 1` for fabricated
  `2023-12-29 / fabricated` evidence.
- Zero overnight-fraction construction ->
  `ValueError: overnight_variance_fraction must lie in (0, 1]`.
- `python -m compileall -q src/sanos/datasets/public tests/test_public_dataset.py`,
  `ruff check src/sanos/datasets/public tests/test_public_dataset.py`, and
  `git --git-dir=.git-real diff --check` passed.
- `gpuctl resources --json` used its local fallback because the daemon was not
  reachable; all reported RAM/disk checks were `OK`.

No network, WRDS, GPU, production artifact, or `quant_data` write was
performed. Temporary adversarial artifacts were isolated below
`/tmp/sanos/phase8-review/` and automatically removed.

Signed: `quantagent` — independent Phase 8 quantitative/data implementation reviewer

## Re-review — 2026-10-12

**Verdict: PASS.** The current remediated tree resolves all five blocking
findings above and satisfies the Phase 8 quantitative/data acceptance gate.
This verdict supersedes the initial `FAIL / CHANGES REQUIRED` verdict for the
settled implementation reviewed here.

### Finding closure and quantitative evidence

1. **CD-004 return binding and source fidelity — resolved.** The ordered return
   vector and complete logical frame now have canonical digests in
   `validation.json` and the closed manifest metadata. Ordinary file re-signing
   without updating that evidence is rejected shallowly. More importantly, an
   independent adversarial probe changed the second return to `42.0`, coherently
   recomputed the return/logical digests and all store checksums, and confirmed
   the intended trust boundary: shallow self-contained validation passed, while
   `deep_lineage=True` reprojected from the internal Dataset and rejected it with
   `Public content differs from an exact source re-projection`. Thus private
   spot/carry inputs remain absent from the public artifact, while source-deep
   validation independently re-establishes the approved formula.

2. **Coverage and typed-empty omissions — resolved.** Coverage start/end and the
   exact ordered `(date, reason)` omission tuple are now part of
   `PublicProjectionConfig`, hence the Dataset identity. Publication requires
   source coverage to equal every NYSE session in that range, requires source
   typed empties to equal the identity-owned omission tuple, and shallow
   validation requires published plus omitted dates to equal the same session
   set. A fabricated omission is rejected, and an independent identity probe
   confirmed that adding an omission changes the artifact ID. Populated empty
   projections and unapproved omissions remain fatal rather than implicit.

3. **Daily-front selector and exact NMS boundary — resolved.** The selector is
   side-specific and applies the required active, European, smooth-available,
   finite-fit, literal `0.2 +/- 1e-4`, inclusive `0..70D`, and inclusive NMS
   predicates. Calls precede puts and sides survive independently. Producer and
   validator both compare the persisted Float32 strike exactly after promotion
   to Float64; the previous `2e-6` widening is gone. The boundary regression
   covers exact/inside/outside DTE and first representable Float32 NMS points,
   plus inactive, American, unavailable and nonfinite-fit cases. My independent
   mutation to the first Float32 point below the lower bound was rejected as
   `Public book lies outside the inclusive configured NMS range`.

4. **Overnight fraction — resolved.** The public configuration accepts the
   closed interval `[0,1]`, retains `0.2` as its default, serializes the value in
   canonical identity, and publication requires exact equality with the source
   Dataset clock. The checked regression builds the `0.0` endpoint. My separate
   build exercised `1.0`, then changed only the public value to `0.2` and
   observed the required source/config mismatch rejection. A separate plan
   probe confirmed that `0.0`, `0.2`, and `1.0` produce three distinct artifact
   identities.

5. **Carry curve and state reconstruction — resolved.** CD-004 carry now uses
   every distinct positive source-day forward/discount node, not merely the
   DLV/SANOS state union; duplicate DTE nodes must agree exactly and `(0,1)` is
   the explicit anchor. The direct regression includes a non-model source node.
   The complete public smooth SANOS and corrected-DLV states reconstruct the
   source pricers. The checked source/public test covers intrinsic time zero,
   below one minute, exactly one minute, pre-first expiry, the first node, and
   the post-final branch. I additionally encoded and decoded an independent
   two-expiry smooth/unsmoothed state and obtained exact call parity through the
   between-expiry branch and after the final expiry at `atol=1e-14`. Corrected
   DLV density inversion, exact zero DLV boundaries, put-call parity, and ATM
   volatility derivation remain enforced.

6. **Allowlist, capabilities, and release transition — resolved.** The recursive
   schema contains only the approved day, expiry, normalized-book, SANOS and DLV
   fields; exact schema matching and finite-tree validation reject additions at
   any level. Capability flags advertise only executable normalized pricing,
   ATM-volatility, daily-front and spot-one-spread uses, with vendor-book,
   liquidity, absolute-level and replay capabilities explicitly false. A
   `public` release is a distinct immutable Dataset whose sole direct upstream
   remains the internal Dataset. Build verifies the approved candidate manifest
   and logical digest, reprojects afresh from the internal Dataset, and requires
   exact frame and deterministic Parquet-byte equality before publication. It
   also requires license bytes plus license, rights basis, approver, approval
   date and durable reference. This mechanism records supplied rights evidence;
   neither candidate status nor technical normalization is represented as
   granting redistribution rights. Completed releases retain output-first
   shallow reuse without candidate or internal artifacts present.

### Commands run on the settled tree

- `PYTHONPATH=src WRDS_OFF=1 python -m pytest -q -p no:cacheprovider tests/test_public_dataset.py tests/test_cli.py -k public`
  → **28 passed, 17 deselected** in 130.58s; only the two expected visible
  wildcard-reuse warnings were emitted.
- Independent coherent-return mutation → `SHALLOW_PASS 2` and
  `DEEP_REJECT Public content differs from an exact source re-projection`.
- Independent `overnight_variance_fraction=1.0` publication and mismatch probe
  → endpoint build passed; `0.2 != 1.0` source/config mismatch rejected.
- Independent clock-identity and omission-identity probes → three distinct
  clock identities and a changed identity for a changed omission policy.
- Independent two-expiry SANOS/DLV reconstruction probe across
  `0`, `ONE_MINUTE/2`, `ONE_MINUTE`, pre-first, node, between-node, final-node,
  and post-final times → exact parity within `1e-14`.
- Independent first-representable-Float32-outside NMS mutation → rejected.
- `python -m compileall -q src/sanos/datasets/public tests/test_public_dataset.py`,
  `ruff check src/sanos/datasets/public tests/test_public_dataset.py`, and
  `git --git-dir=.git-real diff --check` all passed.

No network, WRDS connection, GPU, production artifact, or `quant_data` write
was used. `gpuctl resources --json` used its local fallback because the daemon
was unreachable; every reported RAM/disk check was `OK`. Temporary probe
artifacts were confined to auto-removed directories below `/tmp/sanos/`.

### Residual production gate

The fresh 2022–2025 publication must still run the planned full-population
counts, exclusions, distributional comparisons and release-approval process.
That is production validation, not a remaining Phase 8 implementation defect.
The fresh validated/promoted SANOS Dataset remains canonical; the legacy
891-date artifact remains a migration/parity oracle only.

Signed: `quantagent` — independent Phase 8 quantitative/data implementation re-reviewer

## Final settled-tree spot-check — 2026-10-12

**Verdict: PASS.** The post-code-review remediation introduces no
quantitative or data-contract regression.

- Shallow validation of a completed `public` release now requires its logical
  content digest to equal the candidate digest frozen in `PublicApproval`. A
  coherently re-signed release return mutation is therefore rejected without
  opening the candidate or internal Dataset; output-first reuse is preserved.
  The candidate manifest digest remains build-time approval evidence, while
  the logical digest is the correct self-contained release-content binding.
- Fresh-process candidate and release producer-version resolution succeeds.
  The added dependency closure owns schema, coverage, documents, validation,
  projection and release evidence, while operational progress controls do not
  become content identity.
- Candidate, release and source-reprojection progress labels are distinct.
  Validation progress now carries elapsed time, completed/total/failed counts
  and estimated remaining time; these logging changes do not alter projected
  values or artifact schemas.
- Release staging failure leaves no final artifact, and two identical release
  publishers converge on one immutable artifact. The JSON CLI integration
  successfully executes approval-template -> approve -> validate with the
  release-specific event names and `WRDS_OFF=1`.
- The focused gate also re-ran exact NMS/clock boundaries and source/public
  SANOS/DLV reconstruction at the CD-014 time boundaries.

Evidence: the six selected high-risk tests passed in 41.82s with one expected
wildcard-reuse warning. Fresh-process version closure reported distinct,
nonempty candidate/release versions. `compileall`, Ruff and `git diff --check`
all passed. No network, WRDS, GPU or production data was accessed.

Signed: `quantagent` — final Phase 8 settled-tree quantitative/data spot-check

**Final environment-isolation confirmation (2026-10-12): PASS unchanged** — removing process-global `WRDS_OFF` mutation is quantitatively neutral and safer for concurrent validation; the two affected offline tests pass and validation preserves the caller's environment without adding any network path.
