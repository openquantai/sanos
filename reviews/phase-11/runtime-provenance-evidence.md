# Phase 11 runtime-provenance implementation evidence

**Date:** 2026-10-14

**Scope:** exact numerical-runtime binding for Phase 11 evidence and canonical
reference promotion

**Network/WRDS access:** none

## Implemented boundary

The production CLI now writes and reuses an immutable
`sanos-runtime-fingerprint-v1` document. Its canonical identity binds:

- Python implementation, version, ABI, compiler and byte order;
- operating-system, kernel, machine, libc, pointer size, logical CPU count and
  process CPU availability;
- installed and imported runtime versions for the closed required numerical
  package set, with JAX and JAXLIB represented explicitly when absent;
- the exact sorted CVXPY solver set, including mandatory HiGHS and CLARABEL;
- sanitized NumPy BLAS/LAPACK build identity, SIMD capabilities and a nonempty
  `threadpoolctl` runtime-library inventory with complete BLAS identities; and
- every supported numerical thread-count and dynamic-thread environment
  control, with a closed value grammar.

The document deliberately contains no hostname, username, credential or path.
All deterministic collectors, validators, readers and writers carry explicit
`cdxcore.version.version` ownership. Publication is create-if-absent and
atomic; an existing byte-identical document is reused, while conflicting
content fails closed. No lock leaf is created or followed.

The Phase 11 evidence index schema is now v3 and requires the runtime document.
Index creation reopens it, validates it, and requires exact equality with the
active runtime. The index records its digest, byte length and fingerprint ID.
Review-attestation creation, promotion-config creation, canonical promotion and
promotion validation all reopen and revalidate the sibling document. Promotion
copies it beside the retained evidence index and requires the closed resolved
version set to bind the same runtime ID.

This work changes no filtering rule, selected population, SANOS/DLV equation,
fit objective, tolerance, solver policy or artifact content identity. It adds a
production-evidence boundary only.

## Settled runtime evidence

The reviewed interpreter produced and then byte-identically reused:

```text
fingerprint_id=c48b4cf98bbe21ccc01e149ac2a75708f8330dd002c3822980451afee456207c
logical_cpu_count=24
affinity_cpu_count=24
available_cpu_count=24
runtime_blas_libraries=3 OpenBLAS identities
```

The production runbook now selects one absolute interpreter, exports the
reviewed checkout's `src` directory, asserts that `sanos.__file__` resolves to
that checkout, captures the runtime document before material work and requires
byte-identical reuse immediately before evidence indexing.

## Verification

```text
env WRDS_OFF=1 TMPDIR=/tmp/sanos/pytest-temp \
  PYTHONDONTWRITEBYTECODE=1 /home/hans/vqdata/bin/python \
  -m pytest -q -p no:cacheprovider
-> 420 passed, 2 expected wildcard-reuse warnings in 546.80s

ruff check src tests scripts setup.py
python -m compileall -q src tests scripts setup.py
python scripts/check_markdown_links.py
git --git-dir=.git-real --work-tree=. diff --check
-> passed (link check covered 96 Markdown files)
```

The code reviewer required removal of a lock-leaf symlink/unbounded-wait risk
and correction of `sys.version_info` field validation. Publication now uses the
repository's hard-link no-replace atomic JSON primitive with collision
revalidation; the type and release-level contract has adversarial re-signed
regressions. The final code re-review passed.

The quantitative reviewer required binding dynamic thread controls and CPU
availability, making `threadpoolctl` a direct mandatory dependency with a
nonempty BLAS inventory, closing the resolved-version/runtime-ID relation, and
making the Phase 10 owner gate unambiguous. All four findings were remediated;
the final quantitative re-review passed.

## Remaining production gates

This evidence does not authorize Phase 11. Production remains blocked until
the owner explicitly classifies the 2025-01-02 and 2025-01-03 pilot differences
and the accepted pilot has no unexplained discrepancy. The shared `vqdata`
interpreter incompatibility and sandbox-local resource-daemon observation above
were subsequently resolved without changing that owner gate: a dedicated
environment passes `pip check`, and authoritative daemon-backed resource checks
are all `OK`. See [environment-readiness-evidence.md](environment-readiness-evidence.md).
