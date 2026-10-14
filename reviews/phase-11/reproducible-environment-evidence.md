# Phase 11 reproducible-environment evidence

Date: 2026-10-14
Baseline: `3069eeaf616c765ec9176e3fcda48d9281c77aa1` plus the reviewed worktree
Network/WRDS access: none (`WRDS_OFF=1`)

## Scope

The direct package versions used by the passing isolated CPython 3.12 Phase 11
environment are now recorded in
`constraints/phase11-tested-py312.txt`. The file covers every base, `dev`, and
`operator` requirement plus the directly relevant CLARABEL, JAXLIB, and IPython
transitive packages. It is explicitly a tested constraints file, not a
cross-platform lock or an artifact identity. The existing immutable runtime
fingerprint remains authoritative for its closed selected numerical package
identities, CVXPY solvers, Python/host identity, BLAS/LAPACK, CPU availability,
and thread settings.

README, operator, production-runbook, program-plan, and earlier runtime-evidence
wording now describe the same contract. The obsolete claims that the shared
interpreter and a sandbox-local daemon observation remained production blockers
are superseded by the separately reviewed isolated-environment evidence; the
owner classification gate is unchanged.

The source-distribution manifest and verifier require the constraints file so a
human following the runbook from an sdist does not receive instructions that
refer to an absent file. A regression reads `pyproject.toml` and requires every
base, `dev`, and `operator` dependency to have exactly one corresponding exact
constraint.

## Verification

- a constrained editable-install dry-run resolved with every direct dependency
  already satisfied at the recorded version;
- `pip check` reported no broken requirements;
- the focused `tests/test_production.py` suite passed all **8 tests**;
- Ruff passed for `src`, `tests`, `scripts`, and `setup.py`;
- the Markdown checker passed all **105 files**;
- wheel and sdist builds succeeded in `/tmp/sanos/`, the sdist included the
  constraints file, and the complete out-of-tree distribution verifier passed;
- `git diff --check` passed; and
- no WRDS connection or production-root creation occurred.

## Gate retained

This improves reproducibility only. It does not approve the exact 2025-01-02
and 2025-01-03 pilot differences, start Phase 11, designate a canonical
Dataset, or grant public redistribution authority.
