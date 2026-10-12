# Phase 1 independent code re-review

## Verdict

**PASS**

All eight findings from the initial code review are resolved. The
catalogue is now write-safe for the tested collision classes, the comparison
adapter applies one strict validator on both catalogue and load paths, the
runtime/configuration baseline is machine-readable, dtype parity is explicit,
the traceability descriptions are corrected, transformation dependencies are
declared, and CI includes `scripts`.

The final CD-009 wording correction makes the dimensional bridge explicitly
proposed and non-normative until approval. Phase 1 therefore meets its stated
code acceptance criteria. This PASS does **not** approve CD-004, CD-009,
CD-011, or any other open/provisional compatibility decision.

## Final narrow re-review

`docs/reference-baseline.md:198-203` now records the requested spot-normalized
half-spread, gives the dimensionally consistent `DF*(F/S)` derivation as
evidence, and explicitly says that retaining that public interpreting field is
only the CD-009 proposal and is non-normative until approved. This is consistent
with `docs/compatibility-decisions.md:119-131`, `PROGRAM_PLAN.md:428-432`, and
`docs/data-contracts.md:182-215`. The sole remaining blocker from the preceding
review is resolved.

## Prior-finding disposition

1. **Resolved — catalogue output collision/symlink safety.**
   `scripts/catalog_reference_market.py:31-47` resolves the reference root and
   output before scanning or creating directories and rejects the root or any
   descendant. Tests at `tests/test_reference_legacy_market.py:210-248` cover an
   input-file collision, an ordinary descendant, and an external symlink alias
   to an input file.
2. **Resolved — semantic contradictions.** The public allowlist now
   excludes redundant fitted price/IV arrays, the daily-front NMS interval is
   `[-3,2]`, and CD-004/CD-009 fields are consistently provisional in the plan,
   data contract, compatibility register, and reference baseline.
3. **Resolved — dtype behavior.** `PROGRAM_PLAN.md:374-378`,
   `docs/data-contracts.md:152-154`, and
   `docs/reference-baseline.md:177-180` preserve the exact legacy float32 daily
   boundaries before selected solver promotion; float64-through-fit remains
   blocked by CD-011.
4. **Resolved — numerical/runtime baseline.** The valid JSON file
   `reviews/phase-1/reference-runtime-baseline.json` records the actual evidence
   runtime, explicitly marks the historical artifact-generation runtime
   unknown, and freezes transform/schema versions, production configurations,
   solver choices/options/fallbacks, tolerances, daily schemas, and cast policy.
   `tests/test_reference_inventory.py:126-182` guards the critical fields.
5. **Resolved — strict shared payload validation.** `_validate_payload` at
   `src/sanos/reference/legacy_market.py:252-418` validates the minimum payload,
   schema, scalar fields, expiry geometry, ordered finite strikes, exact boolean
   masks, and aligned lengths. Both `_record` and `LegacyMarketReference.load_day`
   call it. Malformed encoding is consistently wrapped in
   `ReferenceFormatError`, with focused tests at
   `tests/test_reference_legacy_market.py:131-183`.
6. **Resolved — traceability accuracy.** The matrix now describes the CVXPY
   utility as a narrow shape patch, the WIDB generation module as a live/cache
   harness, and the historical SANOS utility test as coverage only for the
   omitted old ATM helper. Existing coverage is separated from tests to be
   written in later phases.
7. **Resolved — `@version` helper dependencies.** The decoder, validator,
   record, and catalogue dependency chains now include output-affecting helpers
   and schema constants. `tests/test_reference_legacy_market.py:251-254` guards
   the resolved dependency versions.
8. **Resolved — CI scope.** `.github/workflows/ci.yml:18-23` runs Ruff over
   `src tests scripts` and invokes mypy; `pyproject.toml:83-90` includes all
   three paths in mypy's configured files.

## Independent checks

No WRDS connection was opened and `quant_data` was not modified.

```text
env PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  TMPDIR=/tmp/sanos/phase1-code-rereview \
  python -m pytest -q -p no:cacheprovider
21 passed in 2.35s

ruff check --no-cache src tests scripts
All checks passed!

env PYTHONPYCACHEPREFIX=/tmp/sanos/phase1-code-rereview/pycache \
  python -m compileall -q src tests scripts
[exit 0]

git --git-dir=.git-real --work-tree=. diff --check
[exit 0]
```

An independent mypy rerun could not be performed because neither the default
interpreter nor `/home/hans/vqdata/bin/python` currently has mypy installed and
no `mypy` executable is on `PATH`. The corrected CI configuration installs the
`dev` extra and covers `scripts`; the Phase 1 evidence records an earlier clean
mypy run.

The checked-in catalogue SHA-256 independently recomputes as
`9b68a9638f273271ff4dbde6365c75ac9fd749993f9284b42428682339d516b4`,
matching the evidence and artifact inventory.

## Residual risks

- The historical artifact-generation package environment was never recorded.
  The new runtime baseline correctly records this as unknown and captures only
  the environment used to regenerate and inspect Phase 1 evidence.
- The 715 MB legacy collection and downstream experiment/NPZ remain external
  parity oracles. CI pins their catalogue and hashes but does not re-read all
  external files on every run.
- This PASS validates Phase 1 inventory and tooling only; it does not approve a
  semantic change, public redistribution, or any open or provisional
  compatibility decision.
