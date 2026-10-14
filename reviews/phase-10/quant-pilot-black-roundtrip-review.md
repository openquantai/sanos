# Phase 10 initial Black round-trip quantitative review

Date: 2026-10-14  
Reviewer: `quantagent`  
Baseline: `43dcad12de43d6e6dcee969a3f3dab5b8b83666e` plus the settled patch  
Network/WRDS access: none

## Verdict

**PASS after boundary-test remediation.**

The copied producer contract is direct Float64 `fits1`/`fitsX` price followed
by implied volatility with `price_tol=1e-6` and kernel `eps=1e-8`, Black
price/Greek evaluation with `eps=1e-8`, then Float32 persistence. Making the
price tolerance explicit preserves that behavior and closes its version
identity over the previously inherited cdxcore default.

One full Float32 ULP is a conservative serialization allowance. The scalar
test now brackets the actual Float64-representable boundary: the greatest
representable source value whose recomputed error is within the exact allowance
passes, and its immediate successor fails. Production tolerance was not
widened.

The deterministic full-vector replay is numerically legitimate and prevents a
coherent low-vega alternate volatility/Greek vector from satisfying only a
price tolerance. The independent cell envelope and source-price audit remain
defense in depth. No SANOS density, support, mask, smooth fit, or DLV state is
changed.

Independent final focused result: 3 passed; diff check clean. No remaining
quantitative or data-contract finding.

Signed: `quantagent` — independent quantitative/data reviewer
