# Phase 10 initial Black round-trip code review

Date: 2026-10-14  
Reviewer: `code`  
Baseline: `43dcad12de43d6e6dcee969a3f3dab5b8b83666e` plus the settled patch  
Network/WRDS access: none

## Verdict

**PASS after remediation.**

The first review failed with a high-severity validation bypass: the scalar
source-price bound and independent Black cell envelope did not bind persisted
volatility/Greeks to the deterministic implied-volatility solution in a
low-vega region. A coherent alternative vector could therefore pass.

The settled `_black_source_vector_residual` reproduces the producer's exact
strike/time promotion, version-owned implied-volatility tolerance and kernel
epsilon, Black outputs, and Float32 casts. It compares raw Float32 bits for
volatility, price and every Greek for both `fits1` and `fitsX`. Shape, expiry,
dtype, null, finiteness, or field mismatches fail closed. The source-price and
independent cell-envelope checks remain complementary evidence.

The validator version and dependency closure include all replay inputs. The
coherent alternate-vector unit test demonstrates the old checks accept the
counterexample while exact replay rejects it; a high-level partition mutation
proves enforcement through `validate_fitted_partition`.

Independent final focused result: 3 passed. Ruff and diff checks passed. No
remaining code finding.

Signed: `code` — independent code/engineering reviewer
