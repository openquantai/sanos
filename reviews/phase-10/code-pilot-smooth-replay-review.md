# Phase 10 smooth producer replay code review

Date: 2026-10-14  
Reviewer: `code`  
Baseline: `f8e546c56c175649820a2ea0b12803950eeccd32` plus the settled patch  
Network/WRDS access: none

## Verdict

**PASS after remediation.**

The initial review identified four issues: the sub-minute intrinsic branch was
missing, the existing `5e-8` canonical listed-market bridge gate had been
removed, the version dependency closure did not own the one-minute convention
and price-kernel epsilon, and time-region coverage was incomplete.

The settled implementation matches copied
`ArbFreePricer.price(mode="surface")` behavior below and exactly at the
one-minute cutoff, before, at, near, between, and after retained expiries,
including midpoint selection. It restores the canonical market bridge gate,
keeps the accumulated `fitsX` bridge diagnostic, and versions the cdxcore
price adapter, `1e-6` epsilon, and minute convention. Validation schema v5 is
appropriate because fitted artifact content and schema are unchanged.

The real 2024-12-23 artifact replays `fits1`, `fitsX`, listed-market fits, and
fit errors with exact zero maximum residual. Independent final results: 48
tests passed in `tests/test_widb_surface.py`; the focused remediation tests
passed; Ruff and diff checks passed. No remaining code finding. A mypy
executable was not available and is not a code finding.

Signed: `code` — independent code/engineering reviewer
