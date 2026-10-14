# Phase 10 pilot-comparison discrepancy code review

Date: 2026-10-14  
Reviewer: `code`  
Baseline: `e70a7de7f83e82fd7fbb281a60f247b1fef00ad7`  
Network/WRDS access: none

## Verdict

**PASS for the diagnosis; promotion remains blocked pending explicit owner
approval.**

The branch separation is compelling: all 12 populated dates whose copied
`MkArbFree` surface solve completed directly with HiGHS match, while exactly
the two HiGHS `invalid_solution` to CLARABEL `optimal` fallback dates differ.
Inputs, keys, dtypes, nulls, populations, masks, geometry, and pre-smooth data
agree. Recomputed objectives differ by only `4.698999983293706e-9` and
`2.579715580622227e-9`; reconstructed call surfaces remain inside the frozen
tolerance. The differing density coordinates are consistent with runtime
sensitivity among near-equal LP solutions.

The unknown historical solver runtime means that exact nonuniqueness and the
historical termination path cannot be proved. The untouched `MkArbFree 0.51`
replay timing out in the current runtime is supporting evidence, not proof.
No numerical or runtime code change is justified: do not force legacy values,
widen tolerances, change solver order/settings, or classify either date as a
match.

The formal report remains correctly blocked, with `network_accessed=false`
and `approval_document=null`. Any approval must be date scoped and bind the
exact sorted mismatch fields, fresh artifact and Parquet digests, legacy HDF5
digest, explanation, approver, date, and durable evidence reference. After an
approved rerun, the expected pilot result is 12 matches, two approved
discrepancies, one declared oracle gap, one policy-empty date, and no
unexplained discrepancy. Both reviewers must then review that result before
Phase 11.

The production provenance must also bind exact CVXPY, HiGHS, CLARABEL, NumPy,
and SciPy identities. This is a separate provenance requirement, not grounds
for changing the pilot fit.

At the original review, an independent current-runtime rebuild was still
pending. The completed result is reviewed below.

Signed: `code` — independent code/engineering reviewer

## Independent rebuild re-review

The completed clean rebuild has now been checked against its journal. The
journal is 24,310 bytes with SHA-256
`3e89875f8015a37a3605fd35fe1156c53c2ca1f65285164d96959ef675edfd45`.
For 2025-01-03, the fitted artifact ID and Parquet SHA-256 are unchanged and
the Parquet payload compares byte for byte with the first fresh build. This is
strong current-runtime determinism for that date.

For 2025-01-02, the clean rebuild recorded HiGHS `invalid_solution`, then
CLARABEL `user_limit` at the fixed 300-second boundary and objective
`0.055472622915076306`. It failed closed and published no fitted artifact.
This establishes termination instability, not corrupt accepted output. The
previously published exact artifact remains fully validated, but an approval
must explicitly acknowledge this operational risk and bind only that artifact
and Parquet hash. A different future output is not covered.

The refined diagnosis remains a pass. No solver, math, tolerance, or automatic
acceptance change is justified. Production must retain restart and completeness
gates; if controlled retries cannot obtain the approved artifact, the run must
stop for separate review.

Re-reviewed and signed: `code` — independent code/engineering reviewer,
2026-10-14
