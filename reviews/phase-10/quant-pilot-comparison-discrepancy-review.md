# Phase 10 pilot-comparison discrepancy quantitative review

Date: 2026-10-14  
Reviewer: `quantagent`  
Baseline: `e70a7de7f83e82fd7fbb281a60f247b1fef00ad7`  
Network/WRDS access: none

## Verdict

**PASS for quantitative validity and diagnosis; reference promotion remains
blocked pending explicit owner approval.**

The two fresh outputs pass the independent fitted-data validity contract. All
12 direct-HiGHS populated pilot dates match the migration oracle; only
2025-01-02 and 2025-01-03 took the copied `MkArbFree` HiGHS-to-CLARABEL
fallback and differ. Their inputs, populations, masks, and pre-smooth values
match. Their call surfaces remain within the named call tolerance and their
recomputed objectives differ from the historical state by less than
`5e-9`. This is strongly consistent with solver-path drift in an
ill-conditioned problem or on a nonunique optimal face.

Mathematical nonuniqueness is not proved because the historical runtime is
unknown. No global tolerance, model, filter, solver, or algorithm change is
quantitatively justified. The strict comparator should remain unchanged, and
the fresh state must neither be overwritten with the historical density nor
silently relabelled as an exact match.

An owner approval, if granted, must apply only to the exact two dates and
bound artifacts, hashes, and field lists in the report. Every future
discrepancy requires independent evidence and a separate decision. An
independent rebuild under the current locked environment is recommended to
establish current-runtime determinism, but cannot substitute for the owner
decision.

At the original review, that independent rebuild was still pending. The
completed result is reviewed below.

Signed: `quantagent` — independent quantitative/data reviewer

## Independent rebuild re-review

The completed clean rebuild has now been reviewed. Its journal SHA-256 is
`3e89875f8015a37a3605fd35fe1156c53c2ca1f65285164d96959ef675edfd45`.
With revision `e70a7de7f83e82fd7fbb281a60f247b1fef00ad7`, the same Raw inputs and
configuration, a new empty Intermediate root, one worker, and `WRDS_OFF=1`,
2025-01-03 reproduced the original fitted `data.parquet` byte for byte with
SHA-256 `0769438ff7f64e79575a7c63855b8c8de878fb4ff6a4768e493621b4170d08be`.
This establishes exact current-runtime reproducibility for that date.

The same rebuild failed closed for 2025-01-02 after HiGHS
`invalid_solution` and CLARABEL `user_limit` at the unchanged 300-second
boundary; no fitted artifact was published. This does not invalidate the
previously completed and fully validated 2025-01-02 artifact. It does establish
that termination is operationally unstable for this ill-conditioned solve, so
a future clean full-history build must retain the failure, resume, and
completeness gates and must not accept the partial `user_limit` iterate.

The diagnosis is therefore refined, not reversed: 2025-01-03 proves a stable
current-runtime solution distinct from the unknown historical solution, while
2025-01-02 additionally exhibits current-runtime termination sensitivity. No
solver, algorithm, or tolerance change follows from this evidence. The verdict
and owner gate are unchanged: both exact discrepancies remain blocked until an
owner approval record binds the report's dates, fields, artifacts, and digests.

Re-reviewed and signed: `quantagent` — independent quantitative/data reviewer,
2026-10-14
