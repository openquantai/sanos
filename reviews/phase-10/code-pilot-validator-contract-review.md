# Phase 10 fitted-validator contract code review

Date: 2026-10-14  
Reviewer: `code`  
Baseline: `d01069ea559eef7e9adf96977b355058633d5a7e` plus the settled patch  
Network/WRDS access: none

## Verdict

**PASS after remediation.**

The review first required solver grammar and tolerance selection to be one
typed flow rather than two independent parses. `_terminal_solver_events` now
returns the exact authenticated `terminal_solver_call`, which supports both
ordinary surfaces and the direct discrete-DLV surface. Only a successful
ordinary surface call can select a solver-derived calendar tolerance.

The quantitative rereview then exposed a missing full-state check for the
initial surface. The settled implementation independently derives initial and
smooth tolerances and passes both surfaces through the complete strike,
bounds, density, and cross-expiry state validator. The new adversarial test
accepts each valid marginal independently and rejects their decreasing convex
order jointly.

The final code review found one version-ownership issue: changing
`LEGACY_COMPARISON_SCHEMA` to v4 altered the output of
`_legacy_comparison_document` while its own version remained 1.1. The function
is now version 1.2 and a direct regression couples that version to the v4
schema. The evidence field was also renamed from the overly narrow
`terminal_surface_call` to `terminal_solver_call`.

Resolved dependency tests cover `SanosLPSolver`, the typed solver call,
business-calendar arithmetic, and validator helpers. Boundary tests cover all
three calendar tolerances and malformed or failed evidence. The reviewer
independently observed 26 focused tests passing with 121 deselected, plus
passing Ruff and Git diff checks, and reviewed the successful 16-date offline
pilot evidence. No remaining code finding.

Signed: `code` — independent code/engineering reviewer
