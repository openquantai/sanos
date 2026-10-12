# Phase 6 final quantitative/data review

**Verdict: CHANGES_REQUIRED**

Date: 2026-10-12  
Reviewer: independent quantitative/data reviewer (`/root/quantagent`)  
Scope: current uncommitted Phase 6 implementation and offline tests  
Network/WRDS access: none

## Blocking findings

### 1. Smooth SANOS is not using the reference Float32 ingress

The pinned producer reads filtered `tte_y`, `pure_strike`, `pure_bid_price`,
and `pure_ask_price` with plain `to_numpy()`
(`reference/quant_data/quantdata/dysanos/market.py:148-153`) and therefore sends
the persisted Float32 arrays to `MkArbFree`. The standalone implementation
instead constructs Float64 expiries and uses `_float_column` for strikes, bids,
and asks (`src/sanos/widb/surface.py:793-800`). This conflicts with
`PROGRAM_PLAN.md` and the Phase 6 quant design audit's explicit mixed-dtype
contract under CD-011.

This is materially observable on the checked two-expiry fixture: running the
same copied `MkArbFree(smoothness=0.25)` with the actual Float32 inputs versus
the widened Float64 inputs changed surface density masses by up to
`1.2209305578e-3`, fitted calls by up to `2.5240828552e-8`, and variances by
`5.8887194686e-10` (masks happened to agree on this benign fixture). The
current pinned subprocess cannot detect the defect because it unconditionally
casts every input to Float64 at
`tests/reference_surface_probe.py:22-25`.

Required change: preserve Float64 ingress for the *initial* WIDB no-arbitrage
wrapper, but preserve the filtered frame's Float32 expiry/strike/bid/ask arrays
for the production smooth fit. Make the reference probe exercise the matching
dtype separately for each stage, and compare the smooth state against the
actual Float32 pinned path.

### 2. The persisted DLV vectors fail the required DLV-to-density reconstruction

Phase 6 writes `initial_surface_dlv_squares` from the complete unequal surface
support (`src/sanos/widb/surface.py:196-210`), but the checked fitted-day state
cannot be inverted by the repository's public `SanosPricer.densities_from_dlvsqs`.
An offline reconstruction from the persisted full vectors failed on expiry 0:

```text
computed q_t must have expectation, but found xstrikes_t@q_t=1.009062
instead of 1 [tolerance 'dlv_dens_int1' 0.0001]
```

The existing Phase 6 test only reconstructs prices from the separately stored
density (`compatibility_dlv_pricer`, `tests/test_widb_surface.py:281-293`); it
does not reconstruct density or calls from the stored DLV parameters. The
failure exposes the unequal-grid coefficient orientation in the frozen inverse:
`densities_from_dlvsqs` passes the quantity with the upper spacing denominator
as `w_minus` and the lower-spacing quantity as `w_plus`
(`src/sanos/model/tools.py:582-601`), unlike the forward consistency solve.
The existing generic round-trip fixture does not reveal this on its grids.

Required change: register and approve this reference defect before altering
the numerical core, then provide a version-owned corrected inverse (or an
explicitly approved encoding/decoder) and a Phase 6 state-level round trip on
the actual irregular support. It must recover masses and off-grid calls within
the declared tolerance. Do not silently change the pinned reference behavior.

### 3. The pinned parity fixture does not cover the production population or failure contracts

The sole surface parity fixture has 26/26 filter-eligible rows, zero initial
masked strikes, zero dropped expiries, zero smooth-masked rows, and zero
inactive selected books. The reference subprocess calls only the numerical
`MkArbFree` core; it does not call the pinned `mk_arb_free_day`,
`OptionsFilter`, normalization, or `smooth_interpolated_market` adapters.
Consequently it did not catch finding 1 and cannot validate filter ordering,
Float32 egress/ingress, dropped-expiry alignment, pre-mask versus smooth-mask
semantics, or active-book populations.

The Phase 6 design gate explicitly requires a pinned fixture containing a
boundary trim or expiry drop, an initial false strike mask, a filter rejection,
and retained/missing final masks. It also requires a forced solver fallback,
terminal solver failure, diagnostic-sink numerical neutrality, and interrupted
multi-day restart. Existing tests cover a successful diagnostic stream, an
isolated infeasible LP, generic runner interruption, and a self-derived filter
cap case, but not those production-path combinations.

Required change: add a stage-by-stage pinned production-adapter oracle with the
required nontrivial populations and exact schema/dtype/order/mask assertions;
add forced fallback/failure and sink-on/off parity tests. A loose numerical
tolerance must not excuse a population or mask mismatch.

## Confirmed behavior

- The implemented order is initial `MkArbFree(surface, smoothness=None)`, then
  the legacy `OptionsFilter` sequence, then smooth
  `MkArbFree(surface, smoothness=0.25, surface_expiry_linear=True)`.
- The filter does not apply the initial strike mask. `smooth_model_mask` is the
  selected/pre-mask/smooth-mask intersection, while `call_active` and
  `put_active` intentionally use the selected pre-mask and valid real side
  book, not the final smooth mask.
- Spot normalization and direct real-book conversion retain Float32 arithmetic;
  direct books are in `C/(DF*F)`. Synthetic pure call books remain the smooth
  fitting input. Put mids use exact normalized parity.
- The initial fit uses Float64 solver ingress and Float32 quote-aligned output.
  The in-review remediation now correctly uses the pinned `1e-8` implied-vol
  and Greek epsilons through version-owned Black adapters, and the oracle now
  derives those fields from the pinned wrapper constants.
- The pre-filter linear surface is retained as the compatibility DLV universe;
  no unapproved post-filter DLV optimization was introduced. Smooth and
  unsmoothed density/support/variance/index state is present in the one daily
  Polars frame.
- Solver statuses, fallback flags, objectives, stages, and the fixed 300-second
  timeout are recorded; hard fit failures propagate instead of becoming silent
  exclusions. The missing forced-path tests are covered by finding 3.
- `overnight_variance_fraction` reaches enriched timing, filter eligibility,
  both model expiry vectors, variances, and DLV time increments through the
  upstream artifact identity. Offline tests cover `0.1`, `0.2`, and `0.5`, with
  the Phase 5 endpoint failures still explicit. No AM node is silently merged
  or jittered.
- The four storage-class and output-first artifact semantics are unchanged;
  the fitted stage can rebuild from a retained filtered artifact and can be
  reused after upstream deletion.

## Population gates and residual evidence

The historical oracles remain correctly distinguished and protected by the
offline inventory tests:

- smooth cache: 891 populated dates, 26,732 expiries, 2,414,341 strike nodes,
  and 4,651,416 active side books (including 8,090 at the 0.2-day node);
- daily-front DySANOS adapter: 3,566,258 books; and
- separate Daily25 SDLV artifact: 2,192,458 books, not a Phase 6 target.

These are catalog gates, not evidence that the new implementation reproduces
the full history. The fresh 2022-2025 run must still check per-stage counts and
the explicit 2025-01-08 empty marker before Phase 6/7 population acceptance;
any discrepancy must be traced from raw fingerprints to the first divergent
stage.

## Offline commands and probes

```text
WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_widb_surface.py tests/test_widb_build.py tests/test_model_core.py \
  tests/test_reference_inventory.py tests/test_model_reference_parity.py
# 38 passed

WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_reference_inventory.py tests/test_model_reference_parity.py tests/test_runner.py
# 18 passed

git --git-dir=.git-real --work-tree=. diff --check
# clean
```

Additional offline probes compared Float32 and Float64 smooth ingress on the
checked Phase 6 fixture and attempted density reconstruction from the persisted
full-support DLV vectors; their quantitative results are reported in findings
1 and 2.

Signed: `quantagent` — independent quantitative/data reviewer
