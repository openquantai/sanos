# Phase 8 quantitative decision-freeze audit

**Reviewer:** `quantagent`
**Date:** 2026-10-12
**Scope:** CD-004/CD-009 decision freeze and related documentation only; no
implementation, WRDS, network, or `quant_data` access
**Verdict:** **PASS**

The approved technical contracts in `docs/compatibility-decisions.md` match the
dual-reviewed Phase 8 quantitative recommendations. The two documentation
contradictions from the initial review are resolved, and the repository does
not mistake technical approval for redistribution rights. I found no remaining
decision-freeze blocker to Phase 8 implementation.

## Remediation verification

### 1. README decision status — resolved

`README.md` now says the two technical unit contracts are approved and Phase 8
implementation is in progress. It separately says a technically validated
projection remains `public_candidate` until redistribution approval is
recorded. This is consistent with:

- `PROGRAM_PLAN.md:4-5`, which records technical approval on 2026-10-12;
- approved CD-004 and CD-009 in `docs/compatibility-decisions.md`; and
- `docs/data-contracts.md`, which now calls the version-1 allowlist and the two
  contracts approved.

The wording no longer describes either technical contract as blocked and does
not imply that implementation approval grants public-release rights.

### 2. First-date WRDS boundary — resolved

Approved CD-004 says the first public row is null even when an earlier Raw close
exists. Every non-null return uses the curve stored on the immediately preceding
**retained public row**. Therefore a forward and zero curve from the session
before the first public row are not required to evaluate the approved public
return.

`docs/wrds-runbook.md` and `PROGRAM_PLAN.md` now attribute the required
previous-session close to the internal legacy-compatible return, explicitly
state that CD-004 requires no pre-start forward or zero curve, and label any
such curves as optional conservative over-fetch rather than a semantic
requirement. CD-004's expected-impact text now says the same. The first public
row remains null; no unapproved boundary return was introduced.

## Contract checks that pass

1. **Authorization and scope.** CD-004 and CD-009 record the 2026-10-12
   technical approval and the user's authorization to proceed with the
   dual-reviewed recommendation. Legacy internal `spot_ret` remains frozen and
   is not relabelled.
2. **CD-004 exactness.** The frozen text preserves the reviewed populated-date
   policy, typed-empty omission and evidence, ascending dates, previous-retained
   row rule, NYSE-session horizon, `(0,1)` anchors, positive curves, consistent
   duplicate nodes, log-linear interpolation, exact-node behavior,
   no-extrapolation rule, Float64 formula, first-row null, multi-session gaps,
   holiday/weekend treatment, and one-session half days. Absolute spots,
   carry, and discount inputs remain non-public.
3. **CD-009 exactness.** The frozen text publishes only positive finite Float64
   `discounted_forward_over_spot = DF*(F/S)` and
   `half_spread_spot = half_spread/S`, retains reconstructed model prices in
   `C/(DF*F)` units, gives both conversion equations correctly, and forbids
   `DF`, `F/S`, spot, forward, absolute strike, absolute price, and separate
   carry fields.
4. **Distribution rights remain separate.** `PROGRAM_PLAN.md`,
   `docs/data-contracts.md`, CD-009, `AGENTS.md`, and the artifact contract all
   distinguish technical/schema/leakage approval from legal rights.
   `public_candidate` remains non-public. An immutable `public` artifact still
   requires explicit license and redistribution evidence and a separate
   materialization; no normalization or transformation is described as
   establishing rights.
5. **Lineage is unchanged.** The freeze does not alter the binding
   `public_candidate <- internal Dataset` and
   `public <- same internal Dataset` direct-lineage rule. Candidate equality is
   approval evidence rather than a candidate-to-public data lineage edge.

## Minor editorial note (not a blocker)

`PROGRAM_PLAN.md` now calls the calculation approved and says CD-004 fixes the
formula. One sentence still says positive curves “are proposed to interpolate”
log-linearly. In context this reads as the verb describing their operation,
not as a reopened decision: the same paragraph and the normative compatibility
register unambiguously mark the entire interpolation rule approved. Replacing
that isolated word with “interpolate” would be a useful editorial cleanup but
is not a contract ambiguity or implementation blocker.

The original `reviews/phase-8/quant-design-audit.md` correctly records the
pre-decision gate as it existed when signed. It need not be rewritten; this
decision-freeze audit and the compatibility register provide the subsequent
approval record.

No wording reviewed here grants or implies redistribution rights.

Signed: `quantagent` — independent Phase 8 quantitative/data decision reviewer
