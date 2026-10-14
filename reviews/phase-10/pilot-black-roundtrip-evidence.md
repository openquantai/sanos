# Phase 10 initial Black round-trip evidence

Date: 2026-10-14  
Implementation baseline: `43dcad12de43d6e6dcee969a3f3dab5b8b83666e` plus the reviewed patch  
Network access: none (`WRDS_OFF=1`)

## Trigger

The first read-only comparison of the 16-day fresh pilot stopped on
2024-12-23 because the validator required reconstructed initial fitted prices
to differ from persisted `pure_fitX_price` by no more than a bare `1e-6`.
This was a validator failure, not a SANOS/no-arbitrage solve failure.

The copied producer maps direct Float64 `fits1`/`fitsX` prices through
`cdxcore.bs.implied`, reprices price and Greeks, and stores the resulting
vector as Float32. It supplied the separate Black-kernel `eps=1e-8` but
previously inherited cdxcore's `price_tol=1e-6` default.

## Complete pilot measurement

The measurement covered all 16 fitted pilot partitions, of which 15 are
populated, and all 136,737 rows belonging to retained initial expiries.

```text
surface values above bare 1e-6:       34
per-expiry values above bare 1e-6:    36
maximum source-to-stored residual:    1.023709774061743e-6
Float32 ULP at that stored value:     5.960464477539063e-8
values above 1e-6 + one Float32 ULP:  0
```

A deterministic replay from serialized state through the version-owned
`_black_implied` and `_black_values` adapters followed by Float32 casting gave:

```text
fit1 volatility/price/delta/gamma/vega/theta/dk mismatches: 0
fitX volatility/price/delta/gamma/vega/theta/dk mismatches: 0
fit1 source-price contract excess: 0
fitX source-price contract excess: 0
```

## Adversarial review and remediation

The first implementation checked the source-price envelope plus an independent
Black rounding-cell envelope. Code review found that a coherent alternate
deep-in-the-money volatility/Greek vector could satisfy both checks because
materially different volatility values can reprice within `1e-6` when vega is
small. The reviewer supplied a concrete counterexample at normalized strike
`0.7527551` and time `0.01190476`.

The settled validator therefore uses three layers:

1. deterministic producer replay and exact Float32 bit equality for implied
   volatility, price, delta, gamma, vega, theta, and `dk`;
2. the measured `1e-6 + one Float32 ULP` source-price envelope as an audit
   metric; and
3. the independent Black rounding-cell envelope as defense in depth.

Tests reproduce the coherent-vector counterexample, prove it passed the two
weaker checks, and prove both the direct replay and high-level partition
validator reject it. A separate regression brackets the actual representable
inclusive price boundary and its immediately rejected Float64 successor.

## Result

The correction changes validator and producer-version identity because the
previously hidden cdxcore price tolerance is now explicit. It does not change
any fitted value, state, mask, population, SANOS parameter, or DLV parameter.
The pilot fitted intermediates must be rebuilt under the new identity before
the read-only legacy comparison resumes.
