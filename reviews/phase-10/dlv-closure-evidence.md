# Phase 10 canonical-DLV closure evidence

Date: 2026-10-14  
Scope: canonical finite-DLV projection and 16-session offline rebuild after the
fresh Phase 10 Raw extraction  
Network access during this closure: none (`WRDS_OFF=1`)

## Outcome

The separate canonical finite-DLV projection now builds and validates for all
16 Phase 10 sessions. This projection is downstream of, and does not replace,
the copied/adapted reference smooth SANOS fitter in
`src/sanos/model/fitter.py`. The smooth fit retains its production smoothness
of `0.25`; the overnight-variance fraction remains a separate configuration
value whose backward-compatible default is `0.2`.

The accepted build and its immediate output-first reuse are recorded in:

```text
/home/hans/quantdata_data/sanos/phase-10-dlv/intermediate/.runs/widb/
7cc30929234eedb2d1fea987f4ba4ca9aa90ab3e62b2e76f079da054a31d18bc.jsonl
```

The journal SHA-256 is
`23d122270826efd19f406d75d4f7ebd06e848ce397fc28d4935fa28cbe36fc07`.
Run `e8fad38bd2e74a2db56b7cbd6a3d9be5` built all 16 terminal artifacts with
zero failures. Run `fc77d9440da543d5aa6da40180830970` immediately reused all
16 with zero builds and zero failures. Fifteen artifacts are populated;
2025-01-08 is the exact-schema typed-empty artifact with reason
`known_bad_day_all_tickers`.

The machine-readable aggregate is outside Git at:

```text
/home/hans/quantdata_data/sanos/phase-10-dlv/evidence/
successful-pilot-summary.json
```

## Numerical extrema

- canonical density inverse: `1.0570971181733668e-16`
- generic Float64 consumer density inverse: `4.863729709962138e-14`
- canonical call inverse: `2.220446049250313e-16`
- generic Float64 consumer call inverse: `6.661338147750939e-15`
- canonical DLV fixed point: `0.0`
- maximum normalized-call projection error: `9.000000000002062e-05`
- maximum conditioned annual-IV error: `0.004998949585661702`
- maximum conditioned annual-IV p99: `3.8515719869958754e-05`
- maximum ATM annual-IV error: `1.970645868709653e-15`
- minimum density: `9.999999969951017e-09`
- minimum/maximum squared DLV: `0.00010019762573292621` /
  `8.999940000111907`
- target-band maximum violation: `1.3877787807814457e-17`
- solver and DLV epsilon warning counts: zero

The mixed generic-decode/canonical-re-encode residual
`1.6144478844748846e-08` is retained as conditioning evidence and is not an
acceptance gate. Canonical and ordinary generic-consumer round trips are the
gates.

## Verification

```text
pytest tests/test_model_core.py tests/test_widb_surface.py -x
44 passed

pytest -q -p no:cacheprovider
344 passed, 2 expected wildcard-reuse warnings

ruff check <all changed Python files>
All checks passed

python -m compileall <source/test paths>
passed with PYTHONPYCACHEPREFIX outside the repository

scripts/check_markdown_links.py
77 Markdown files checked

scripts/verify_distribution.py <fresh sdist/wheel>
distribution verification passed outside the checkout

git diff --check
passed
```

Both independent reviewers approved this numerical closure. The maximum
conditioned annual-IV error is intentionally close to the frozen `0.005`
limit. The full-range build must remain fail-closed and must not relax that
tolerance.

This evidence approves the implementation and pilot numerical closure. The
formal `reference compare-daily --phase10-pilot-contract` report must still be
generated from a clean committed checkout before Phase 10 is designated
complete.
