# Phase 11 quantitative runtime-provenance review

**Reviewer:** `quantagent`

**Date:** 2026-10-14

**Scope:** completeness of numerical-runtime identity and its Phase 11 evidence
and promotion binding; offline only

**Initial verdict:** **CHANGES REQUIRED**

**Final re-review verdict:** **PASS**

## Initial findings

1. The initial fingerprint did not bind dynamic thread controls or CPU
   availability, so two materially different solver/runtime executions could
   share an identity.
2. `threadpoolctl` was treated as optional and the document did not require a
   nonempty runtime BLAS inventory.
3. The evidence/promotion contract did not close the relationship between the
   resolved runtime version identity and the runtime evidence record.
4. The production runbook did not state the unresolved 2025-01-02/03 owner gate
   strongly enough to prevent diagnostic approval being mistaken for authority
   to start Phase 11.

## Final re-review

All findings are resolved.

- The fingerprint binds logical, affinity and available CPU counts plus a
  closed set of numerical thread-count and dynamic-thread controls. The
  reviewed runtime reports 24/24/24 CPUs.
- `threadpoolctl>=3.5,<4` is a direct dependency. Collection and validation
  require its installed/imported version and a nonempty runtime-library
  inventory with complete BLAS identities; the reviewed runtime reports three
  OpenBLAS identities.
- Evidence schema v3, review creation, promotion configuration, promotion and
  validation all reopen the sibling document. The closed resolved-version set
  must contain the same runtime fingerprint ID recorded by the evidence item.
- The runbook explicitly blocks Phase 11 until the owner classifies both pilot
  differences and the accepted pilot contains no unexplained discrepancy.

The observed fingerprint ID is
`c48b4cf98bbe21ccc01e149ac2a75708f8330dd002c3822980451afee456207c`.
The reviewer independently passed 30 targeted tests and the diff check, and
confirmed that the change does not alter numerical fitting or data-selection
semantics. The coordinator subsequently passed all 420 offline tests.

No network, WRDS, GPU or `quant_data` write was performed.

Signed: `quantagent` — independent Phase 11 quantitative/data reviewer
