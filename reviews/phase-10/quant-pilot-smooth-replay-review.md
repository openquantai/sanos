# Phase 10 smooth producer replay quantitative review

Date: 2026-10-14  
Reviewer: `quantagent`  
Baseline: `f8e546c56c175649820a2ea0b12803950eeccd32` plus the settled patch  
Network/WRDS access: none

## Verdict

**PASS after remediation.**

The validator now distinguishes the three numerical contracts actually
produced by the copied SANOS implementation: expiry-local `fits1`, solver-time
common-surface `fitsX`, and arbitrary-time listed-market fits. It preserves
the producer's Float32 support state and operation ordering instead of routing
audit observables through the canonical Float64 consumer.

All producer fields, row errors, and aggregate diagnostics are replayed
exactly. Dropped-expiry interpolation, retained-expiry selection, the
sub-minute intrinsic branch, and the one-minute boundary match the copied
implementation. The canonical listed-market bridge retains the `5e-8` gate;
the common-surface bridge remains diagnostic because its accumulated
Float32-to-Float64 transition is a different numerical contract.

The canonical SANOS/DLV representation and all fitted data remain unchanged.
Independent focused tests passed and the reviewed diff was clean. No remaining
quantitative or data-contract finding.

Signed: `quantagent` — independent quantitative/data reviewer
