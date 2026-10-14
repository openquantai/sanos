# Phase 10 fitted-validator contract evidence

Date: 2026-10-14  
Implementation baseline: `d01069ea559eef7e9adf96977b355058633d5a7e` plus the reviewed patch  
Network access: none (`WRDS_OFF=1`)

## Trigger

After exact producer replay was established, the clean 16-day pilot exposed
three independent validator assumptions that did not match the copied daily
producer. None changed a fitted artifact, population, SANOS state, or DLV
state.

## Quote-band contract

The copied `MkArbFree` implementation gives `fits1` and `fitsX` different
contracts. An accepted per-expiry `fits1` is contained in the retained quote
band up to the producer's `1e-6` numerical tolerance. The final common-surface
`fitsX` is a global soft closest fit and is not promised to stay inside every
quote.

Across all 15 populated pilot dates:

```text
maximum accepted per-expiry quote-band violation: 9.922384824334607e-7
maximum global surface quote-band violation:       1.1905097704500989e-4
```

The validator therefore enforces only the first contract and records the
second value as `initial_surface_fit_quote_band_violation` in absolute
normalized pure-call units `C/(DF*F)` over `mask_arb_free`. Exact producer
replay and all state/no-arbitrage checks remain mandatory for both fits.

## Solver-conditional calendar contract

The 2025-01-03 primary HiGHS surface solve recorded `invalid_solution`; its
adjacent CLARABEL fallback recorded `optimal`. The reconstructed smooth
calendar residual is `2.0046636678427987e-7`, within that accepted solver's
configured `1e-6` feasibility tolerance but above the validator's former
universal `1e-7` check.

Each initial and smooth terminal surface call is now returned as typed evidence
by the exact solver-event grammar check. Only those already-authenticated calls
select their independent tolerances directly from `SanosLPSolver` settings:

```text
HiGHS terminal success:                    1e-7
CLARABEL fallback, optimal:                1e-6
CLARABEL fallback, optimal_inaccurate:     5e-5
```

All other populated pilot smooth residuals are at most
`8.067573398218286e-8`; initial-state residuals are at most
`8.7691252648181e-8`; canonical DLV residuals are at most
`4.440892098500626e-16`. Strike-shape, bounds, and DLV acceptance contracts
are unchanged.

## Business-clock replay

The copied daily producer casts the integer business-session count to
Float32, performs the configurable AM subtraction in Float64, clips, and then
casts to Float32 once. The previous validator cast the subtraction too early.
On 2025-01-16 that affected 145 front-AM-expiry rows by one Float32 step,
`1.4901161193847656e-8`. Producer-order replay is bitwise exact for both
`business_dte` and the subsequent `time_to_expiry_years` calculation.

## Complete pilot and adversarial checks

The read-only offline validator completed all 16 pilot dates: 15 populated
partitions and the expected 2025-01-08 policy-empty partition. Tests cover all
three calendar tolerances at their inclusive boundary and one representable
step above it; direct and fallback terminal statuses; rejected non-surface or
failed evidence; malformed, orphaned, and mismatched solver events; both
initial and smooth surface policy wiring; a valid-density cross-expiry calendar
inversion; unchanged direct DLV evidence; masked quote residuals; the `1e-6`
quote boundary and one step above; invalid residuals; AM/PM, same-day clipping,
multiple dates, and overnight fractions `0`, `0.2`, `0.5`, and `1`; and
resolved version dependencies.

The complete offline repository suite passes 398 tests with two intentional
`version="*"` reuse warnings. Ruff and Git diff checks pass. The correction
changes validator and comparison-report identity only; existing fitted pilot
artifacts remain valid and are not rebuilt.
