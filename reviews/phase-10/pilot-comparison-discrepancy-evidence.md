# Phase 10 clean comparison discrepancy evidence

Date: 2026-10-14  
Repository revision: `e70a7de7f83e82fd7fbb281a60f247b1fef00ad7`  
Network access: none (`WRDS_OFF=1`)  
Report ID: `bc5c5cf8aad4753e3f792f66f8b7a8e5751795771470615cd70fea57662bb434`  
Report SHA-256: `1a42e3c4aaf82d35b76cb30e3f8b25dbfbd4946c1c2b237d5b5fa840a333e825`

## Result

The clean-tree, read-only comparison completed all 16 planned pilot sessions
without building an artifact or accessing the network. Its classifications
are:

```text
match:                       12
oracle_gap:                   1  (2024-12-24, declared before the run)
policy_empty:                 1  (2025-01-08)
unexplained_discrepancy:      2  (2025-01-02 and 2025-01-03)
```

The report is correctly blocked under the reviewed Phase 10 contract. Phase 11
must not start while either discrepancy remains unexplained or lacks an exact
owner approval record.

## Exact evidence bindings

| Date | Fresh fitted artifact | Fresh `data.parquet` SHA-256 | Legacy HDF5 SHA-256 |
|---|---|---|---|
| 2025-01-02 | `4de1d1094a51abeb7d46949049f41b807901c7454c1b2b93bffc76fbc9c76a4e` | `496317ea5adfc42e99647066d227e259e67e83742f6fcade6df19d53288aa136` | `9208f9133b00f9ef15efc2f9d53df9af6f15f7a123a198e56a2233c96cdb48d4` |
| 2025-01-03 | `e39309c855fd53ae1dbd4f989787582b126d7d2a6814dd99748c87937fe8542c` | `0769438ff7f64e79575a7c63855b8c8de878fb4ff6a4768e493621b4170d08be` | `9c8b3e7ef6904b621c7db6206ce146ae9f35bd52d603a64fee9c3bdbb4c8d5d6` |

The pinned legacy collection and per-file digests passed before comparison.
Both fresh partitions passed the complete fitted validator and canonical
source/order checks.

## Location of the difference

Both dates have exact populations, expiry populations, masks, keys, dtypes,
null patterns, and all comparable pre-smooth data. The mismatches begin at the
final smooth surface LP:

| Date | Market output fields | Smooth-state fields | Fit-diagnostic fields | Reconstruction fields |
|---|---:|---:|---:|---:|
| 2025-01-02 | 58 | 58 | 50 | 0 |
| 2025-01-03 | 56 | 56 | 50 | 1 |

The sole reconstruction failure is two 2025-01-03 ATM implied-volatility grid
values with maximum absolute difference `9.136592288294665e-8` against the
frozen `5e-8` parity tolerance. Reconstructed call surfaces differ by at most
`3.818725174964399e-8` on 2025-01-02 and
`2.5369903178251718e-8` on 2025-01-03, both within the named call tolerance.
The largest individual density differences are approximately `2.88e-4` and
`3.54e-3`, respectively; these remain visible and are not hidden by invariant
tolerances.

## Solver boundary

In the original successful 16-date build, these are exactly the only two
populated pilot dates for which the serial HiGHS smooth-surface solve could not
unpack a valid solution and the documented CLARABEL fallback was accepted:

```text
2025-01-02: HIGHS invalid_solution -> CLARABEL optimal
            accepted objective 0.003823286889714064
2025-01-03: HIGHS invalid_solution -> CLARABEL optimal
            accepted objective 0.0021661336336561295
```

Every populated pilot date whose smooth surface completed directly through
HiGHS matches the historical HDF5 under the strict field contract.

Recomputing the LP objective from each persisted solution on the same fresh
books and identical masks gives:

| Date | Fresh objective | Historical objective | Absolute difference |
|---|---:|---:|---:|
| 2025-01-02 | `0.0038227234366016447` | `0.003822728135601628` | `4.698999983293706e-9` |
| 2025-01-03 | `0.0021654474639510118` | `0.002165444884235431` | `2.579715580622227e-9` |

These objective values strongly support solver-environment drift among
numerically near-equivalent solutions. That behavior can arise from an
ill-conditioned problem or a nonunique optimal face, but the available
evidence does not prove mathematical nonuniqueness.

The original runtime that produced the historical HDF5 is explicitly unknown
in the frozen baseline. Replaying the untouched `quant_data` `MkArbFree 0.51`
on the exact 2025-01-02 filtered Float32 inputs in the captured current runtime
did not reproduce its historical HDF5: it reached the unchanged 300-second
timeout/iteration limit. Source comparison shows that the standalone port's
0.51-to-0.53 changes add solver diagnostics and expiry-decision evidence; they
do not change this LP's objective, constraints, input dtypes, solver order, or
solver settings.

The exact [replay and objective probes](probes/README.md) and their captured
results are retained with this evidence. They were SHA-256 identified at
execution time:

```text
diagnose_fallback_parity.py:
  670ec8750dbe5314aabddb24dea6ef724fb4b6b696baac63f54100e1dd39d1b1
diagnose_fallback_objective.py:
  3cac6183ca4ae097689686ae7f63ad55a40d7a8b16ef76da83bcb25007bded9d
```

The captured execution environment was
`/home/hans/vqdata/bin/python`, Python 3.12.3 on
Linux 6.18.33.2 WSL2/glibc 2.39, with:

```text
cvxpy==1.9.3
highspy==1.15.1
clarabel==0.11.1
numpy==2.5.4
scipy==1.18.1
```

These identities describe this diagnostic runtime only. They do not recover
or infer the unrecorded environment that produced the historical HDF5.

## Independent current-runtime rebuild

An independent rebuild then used revision
`e70a7de7f83e82fd7fbb281a60f247b1fef00ad7`, the same configuration and Raw
generation, a new empty Intermediate root, one local worker, and
`WRDS_OFF=1`. It requested only 2025-01-02 through 2025-01-03 and did not reuse
a fitted artifact. The run finished in `536.454627` seconds with one success
and one failure:

| Date | Result | Terminal copied `MkArbFree` surface calls |
|---|---|---|
| 2025-01-02 | failed closed; no fitted artifact published | HiGHS `invalid_solution`; CLARABEL `user_limit`, objective `0.055472622915076306` |
| 2025-01-03 | success; fitted payload byte-identical to the first fresh run | HiGHS `invalid_solution`; CLARABEL `optimal` |

The rebuilt 2025-01-03 artifact retained ID
`e39309c855fd53ae1dbd4f989787582b126d7d2a6814dd99748c87937fe8542c`.
Its `data.parquet` SHA-256 is again
`0769438ff7f64e79575a7c63855b8c8de878fb4ff6a4768e493621b4170d08be`;
the two Parquet files compare byte for byte. The manifests differ only in the
non-identity `created_at` timestamp.

The complete run journal is retained at
`phase-10-determinism-e70/data/intermediate/.runs/widb/`
`1a89ef6c30dedf277ab1efb42cf64550cd4930337205c4bf8201ec40494979ff.jsonl`
under the SANOS data root. It is 24,310 bytes with SHA-256
`3e89875f8015a37a3605fd35fe1156c53c2ca1f65285164d96959ef675edfd45`.

This establishes exact current-runtime reproducibility for 2025-01-03. For
2025-01-02 it establishes a solver termination instability: identical
code/configuration/input can either yield the already validated CLARABEL
solution or reach the fixed time/iteration boundary and publish nothing. The
failure is explicit and safe; it does not create a conflicting artifact with
the same identity.

## Interpretation and required gate

The evidence is strongly consistent with historical
solver-environment/termination drift between near-equal solutions, not a
Raw-data, population, filter, mask, model, or copied-algorithm change. The
unknown historical runtime prevents proving the exact cause. Both successfully
published fresh states satisfy their authenticated solver feasibility and the
complete independent SANOS validity contract.

This diagnosis does not itself authorize `approved_discrepancy`. The reviewed
comparison governance requires an explicit owner decision bound to the exact
dates, field lists, fresh artifact/data digests, legacy digests, explanation,
approver, approval date, and durable reference. The strict numerical
tolerances must not be widened, fresh values must not be overwritten from the
legacy HDF5, and the copied solver semantics must not be changed merely to force
oracle equality.
