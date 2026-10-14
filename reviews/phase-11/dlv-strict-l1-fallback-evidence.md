# Phase 11 discrete-DLV strict/L1 fallback evidence

Date: 2026-10-14

## Decision exercised

The canonical discrete-DLV projection keeps published DLV bounds `[0.01,3]`.
It first solves with the `9e-5` fitted-call cap and conditioned annual-IV price
bands. Only an authenticated HiGHS event reporting `infeasible` or
`infeasible_inaccurate` permits a
second solve with the same structural constraints and uniform L1 objective but
without those fit-quality bands. A timeout, solver error, unbounded result, or
structural validation failure does not permit relaxation.

No additional fixed strike anchors and no increase of `max_dlv` were used in
this replay.

## Offline replay

The replay used the fresh Phase 11 Raw cache with `WRDS_OFF=1`, source
generation `default-v1`, and two workers. The command completed two of two
fitted-day targets with zero failures and no network access. Its journal is:

```text
/home/hans/quantdata_data/sanos/phase-11/data/intermediate/.runs/widb/57b153e448ad5390e324def3e1898eee1e55d9d160953d9c540aaf1bf2b0b1ff.jsonl
sha256 d637e091ee0040a2340bfa7cb19f353b55ca5cc2fcd0a4adc2e1e6911cd29641
```

| Date | Artifact ID | Solver path | Maximum call error | Conditioned IV max | Conditioned IV p99 | Maximum DLV square |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2022-01-05 | `3d7c01c0e8469cb84ce3cbb44e8c8cea0156e5cdbb96721572769466639a9b9f` | strict infeasible; L1 optimal | 0.001283104773738608 | 0.018330167962417432 | 0.0012393975437826154 | 8.999940000100073 |
| 2022-01-06 | `63f6e0ab4da2c4917b49cee658e60787d38f22cfb55825436b7e2ef6e956f940` | strict infeasible; L1 optimal | 0.0007400801110121202 | 0.011299154853833426 | 0.00003278134817179923 | 8.999940000100048 |

The two `data.parquet` SHA-256 values are respectively
`9ab701935445825e9b254792a09736a320ba8ede59fe39ad81f97e8a12cf1f98`
and `ab57ab30d71e9d0e84fab478bd892eb6b3424bd11fdcd0c5b14f1933dbef2012`.
Their manifest SHA-256 values are
`96d515f5d025a7a54482d0e7e93fe2a5d5cc09ec9707e8ec19468add9672cefc`
and `81dcdc264d96ac117f016dcb46a4f937761f6f9266a8c711725be8ae80fe3059`.

Both artifacts record `fit_mode=uniform_l1_fallback`, explicitly list failed
preferred fit-quality thresholds, and pass the hard density, mass, normalized
first moment, calendar, DLV-bound, canonical/generic inverse, fixed-point, and
serialization checks.

## Regression coverage

`tests/test_widb_surface.py` forces a solver-reported strict infeasibility and proves
that the second fitter receives `max_abs_fit_error=None` and
`fit_price_bounds=None`. It also authenticates the exact two-attempt solver
grammar, including rejection of wrong solver, algorithm, expiry scope,
objective, and extra fields.

## Resume policy

The 2022-01-03 through 2022-01-06 fitted-day artifacts already completed
successfully under their recorded policy and are retained unchanged. They are
not numerically rerun as a consequence of documenting or hardening this policy.
Subsequent fitting resumes at the first missing date. During Dataset assembly,
the exact-four reviewed-source policy validates each complete fitted state and
all pinned fingerprints, then uses its original artifact identity directly.
It does not change diagnostics or stage provenance and does not publish the
bytes under a current-fit identity. The final Dataset permanently records the
accepted artifact, data, manifest, producer-version, stage-version,
configuration, namespace, and diagnostics-schema evidence. The generic
`version="*"` override remains reuse-only and is not used for this policy.
