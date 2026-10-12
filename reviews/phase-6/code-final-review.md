# Phase 6 final code and architecture review

**Verdict: PASS**

Date: 2026-10-12  
Reviewer: independent code agent (`/root/code`)  
Scope: current uncommitted Phase 6 implementation, tests, CLI and documentation  
Network/WRDS access: none

## Findings

No blocking code or architecture findings remain.

The review initially found incomplete version closure for the new Float32
boundary helper and missing injected timeout/unexpected-solver coverage. Both
were corrected before this verdict. A full rerun also caught and corrected the
stale test expectation after `SanosLPSolver` was intentionally bumped to 1.1.

## Acceptance evidence

- The implementation adds three immutable daily Intermediate stages:
  `no-arb-day`, `filtered-day`, and `fitted-day`. Each stage consumes and
  publishes one schema-validated Polars frame, records one direct upstream
  pointer, uses projected canonical configuration, and publishes through the
  existing atomic artifact store.
- Terminal resolution remains output-first. Tests prove that a complete fitted
  artifact is reusable after Raw is removed, and that a missing fitted artifact
  rebuilds from a retained filtered artifact without requiring earlier daily
  stages. Existing wildcard semantics remain reuse-only and retain the actual
  stored pointer, producer version and warning.
- Stage configuration identities are separated into
  `no_arb_configuration_id`, `options_filter_configuration_id`, and
  `smooth_sanos_configuration_id`. Artifact-identity tests prove that only the
  relevant projected setting changes each stage identity. The resolved fitted
  producer version now includes `_float32_column`, `_float_column`, schemas,
  configuration, numerical adapters, and the corrected `SanosPricer` closure.
- Initial no-arbitrage retains the legacy Float64 solver ingress and
  reference-compatible Float32 quote/Greek egress. Smooth SANOS retains the
  persisted Float32 expiry, strike, bid and ask ingress. Pinned subprocess
  comparisons use independent constants and stage-specific dtypes.
- CD-022 is implemented as the approved corrected-only unequal-grid DLV
  inverse. `SanosPricer` is bumped to 0.9, the forward encoding is unchanged,
  and actual fitted irregular supports round-trip density and off-grid calls.
  No legacy defective decoder mode is exposed.
- Filter parity now includes a pinned production-adapter oracle, rejection
  ordering, a binding stable cap, the vega compatibility no-op, a masked
  boundary, a dropped late expiry, and smooth index/mask alignment.
- Solver diagnostics record stage, solver, status, fallback flag and objective
  without changing numerical results. Tests inject infeasibility, primary
  solver failure, primary `user_limit`, terminal fallback `user_limit`, and a
  terminal unexpected exception with exact event streams. Failed daily work is
  journaled, publishes no terminal artifact, and restarts successfully.
- Spawned multi-day execution reuses a completed day after a later day fails.
  The CLI retains the previous paired/enriched result configuration exactly,
  adds the three Phase 6 terminal choices, preserves dry-run and structured
  output, and enforces the four-worker surface limit.
- Documentation now correctly treats the 891-date `quant_data` collection as
  a read-only migration oracle. The freshly downloaded, validated and
  published SANOS dataset becomes canonical; legacy artifacts are never build
  inputs.

## Commands run

```text
PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 python -m pytest -q -p no:cacheprovider
# 169 passed in 60.53s

ruff check src tests scripts
# All checks passed

PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests scripts
# passed

git --git-dir=.git-real --work-tree=. diff --check
# clean
```

The available project interpreter does not contain `mypy`, so I could not
independently rerun that command. Strict mypy remains a mandatory CI job and
the new orchestration, configuration and WIDB modules remain outside the five
documented legacy-module overrides.

## Residual risks and later gates

- No live WRDS access or fresh 2022--2025 build was performed. Population,
  exclusion and vendor-revision comparisons remain publication gates for the
  later fresh-build phase; the legacy 891-date counts are comparison evidence,
  not the canonical dataset.
- CD-020's all-valid enrichment failure and the other explicitly open semantic
  decisions remain documented compatibility boundaries. This review does not
  approve changing them.
- This code PASS is independent of, and does not replace, the required final
  quantitative/data review.

Signed: `code agent /root/code`
