# Phase 10 fitted-validator contract quantitative review

Date: 2026-10-14  
Reviewer: `quantagent`  
Baseline: `d01069ea559eef7e9adf96977b355058633d5a7e` plus the settled patch  
Network/WRDS access: none

## Verdict

**PASS after remediation.**

The initial review found that the implementation documented initial `fitsX`
calendar no-arbitrage as mandatory but had not yet passed the reconstructed
initial surface through the cross-expiry state validator. The settled
implementation derives independent initial and smooth tolerances from each
authenticated terminal `surface` call, validates both states, and records
their separately named residuals and applied tolerances.

An adversarial two-expiry state has valid nonnegative unit-mass and unit-mean
marginals at each expiry but decreasing convex order jointly; it passes each
single-expiry check and is rejected by the combined calendar check. This
closes the identified gap.

The copied quantitative semantics otherwise remain exact. `fits1` alone is
the quote-containment gate; initial `fitsX` is an exactly replayed global soft
fit with a diagnostic quote-band residual. Solver-conditional calendar
tolerances affect only the initial and smooth calendar checks. DLV retains its
existing direct solver grammar, default calendar boundary, inverse, and bound
contracts. Business DTE follows Float32 session count, Float64 AM subtraction,
clipping, and one final Float32 cast.

The reviewer independently observed 21 focused tests passing with 52
deselected and a clean Git diff check. No remaining quantitative or data
finding.

Signed: `quantagent` — independent quantitative/data reviewer
