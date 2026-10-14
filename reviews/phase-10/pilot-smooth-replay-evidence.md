# Phase 10 smooth producer replay evidence

Date: 2026-10-14  
Implementation baseline: `f8e546c56c175649820a2ea0b12803950eeccd32` plus the reviewed patch  
Network access: none (`WRDS_OFF=1`)

## Trigger

The first read-only comparison of the rebuilt 16-day pilot stopped on
2024-12-23 because the validator reconstructed all persisted smooth fitted
prices through the canonical Float64 `SanosPricer`. The copied producer has
three intentionally distinct, dtype-sensitive observables: expiry fits
(`fits1`), common-surface fits (`fitsX`), and the later listed-market fits
computed by `ArbFreePricer`. The validator had incorrectly treated them as one
calculation. This was a validator defect, not a fitting or data defect.

## Complete pilot measurement

The measurement covered all 15 populated fitted pilot dates. Replaying each
observable through its exact producer path gave:

```text
expiry fit maximum absolute residual:       0.0
surface fit maximum absolute residual:      0.0
listed-market fit maximum absolute residual: 0.0
fit-error maximum absolute residual:        0.0
canonical surface bridge maximum residual:  7.109712970437831e-8
canonical market bridge maximum residual:   3.155308114788369e-8
```

The previously reported 2024-12-23 discrepancy was
`5.798907092013916e-8`, consistent with the validator conflating the copied
Float32 producer path with the canonical Float64 consumer path.

## Settled contract

The validator now losslessly recovers the producer's Float32 support state and
replays these fields independently:

1. `smooth_expiry_fit_price` through the per-expiry `fits1` path;
2. `smooth_surface_fit_price` through the solver-time `fitsX` path; and
3. `smooth_call_mid`, row fit errors, and summary errors through the copied
   arbitrary-time `ArbFreePricer` path.

All persisted producer observables are required to match at `1e-12`. The
canonical listed-market bridge retains its existing `5e-8` acceptance bound.
The accumulated common-surface bridge is diagnostic because the producer's
Float32 state and algebraic operation order legitimately exceed that bound on
the pilot. The canonical `SanosPricer`, fitted values, SANOS state, and DLV
state are unchanged.

## Adversarial coverage

Tests independently mutate each of the three persisted fit fields, reject
non-lossless Float32 support serialization, and compare the replay against the
copied `ArbFreePricer` at the intrinsic cutoff, its exact boundary, pre-first,
retained, near-retained, interpolation, midpoint-tie, and post-final times.
The version closure explicitly owns the Black price kernel, its `1e-6`
epsilon, and the one-minute convention.

## Result

Both independent reviewers passed the settled implementation. The complete
surface module has 48 passing tests, including exact zero replay on the real
2024-12-23 artifact. The correction changes validation schema identity only;
it does not require rebuilding fitted pilot artifacts.
