# Phase 10 owner-approval closure evidence

Date: 2026-10-14  
Reviewed source revision: `54a79a9c34266579269b84e150b75edb35a4a510`  
Network access: none (`WRDS_OFF=1`)  
Redistribution approval: none

## Immutable evidence

| Artifact | Identity |
|---|---|
| Blocked comparison report | report `bc5c5cf8aad4753e3f792f66f8b7a8e5751795771470615cd70fea57662bb434`; SHA-256 `1a42e3c4aaf82d35b76cb30e3f8b25dbfbd4946c1c2b237d5b5fa840a333e825` |
| Owner approval document | SHA-256 `4110efca403a6bc2df47b96a56ebe6dc8e3eacffd96bbddd81a3fec6708def57` |
| Passing comparison report | report `7e556e0ee1bb0368544e19aec7146d64c9242d22b4245a1cbbf7969712650566`; SHA-256 `3080b8818edaaf4f31e230e88c6b1fd4da917f8a08f7dfce8f6b6a7f7d0302c2` |
| Passing structured log | SHA-256 `55209b2b95a5d6fa747d470c1feb71d868dfe6d2fe24e1810fa0739f8bb43fe8` |

The owner approved only the exact 2025-01-02 and 2025-01-03 migration
discrepancies bound to the blocked report. The approval command reopened the
physical fresh Parquet artifacts and legacy HDF5 files, recomputed both selected
comparisons, and wrote a new no-replace approval document. It did not build an
artifact or access WRDS.

The approved comparison completed all 16 pilot dates with:

```text
match:                   12
approved_discrepancy:     2  (2025-01-02 and 2025-01-03)
oracle_gap:               1  (2024-12-24)
policy_empty:             1  (2025-01-08)
unexplained_discrepancy:  0
```

A recursive comparison with the blocked report found no change to daily fresh
or legacy evidence, comparison payloads, observations, special-case checks,
configuration, tolerances, request, catalog, or resolved transformation
versions. Differences are limited to the approval overlay, the two approved
classifications, aggregate outcome and identity, and the expected comparator
source-revision update.

The first attempted approved comparison stopped safely when invoked without
the historical pilot's separate Raw-root override. The accepted rerun supplied
that exact immutable Raw cache location and completed offline. No failed attempt
modified a fitted artifact or contacted WRDS.

The evidence directory passed the credential-label scan required by the pilot
runbook. Detailed reports remain internal evidence because they contain
proprietary comparison detail; only the counts, identities, hashes, and review
conclusions above belong in Git.

## Verification

The focused offline suite passed:

```text
tests/test_reference_approvals.py
tests/test_reference_compare.py
56 passed
```

Both required independent reviews pass. This closes the Phase 10 owner
classification gate and permits the guarded Phase 11 production protocol to
begin. It does not grant redistribution rights to the internal Dataset or the
technical public candidate.
