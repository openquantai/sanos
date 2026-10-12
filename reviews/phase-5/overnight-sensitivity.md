# Phase 5 overnight-fraction sensitivity

Date: 2026-10-12

This offline sensitivity check uses a synthetic next-session AM expiry, a
same-date PM comparison, normalized strike `1.02`, annual volatility `0.4`, and
252 business days per year. It contains no WRDS observations. The default
`0.2` is the compatibility case.

| Fraction | AM business DTE | AM years / first DLV `dt` | NMS5 | ATM-vol NMS5 | `0.4² × T` |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 0.0003968254 | 1.4749082 | 1.2809552 | 0.0000634921 |
| 0.2 | 0.2 | 0.0007936508 | 1.4606576 | 1.1409388 | 0.0001269841 |
| 0.5 | 0.5 | 0.0019841270 | 1.4202629 | 0.8964080 | 0.0003174603 |

For all three cases, the corresponding next-session PM expiry remains exactly
`1.0 / 252` year. AM remains ordered before PM, and the synthetic book remains
immediately eligible under the same quote/trade thresholds. Changing the
fraction changes the canonical configuration ID and both daily artifact
identities; it does not change PM clock values.

For a constant `0.4` variance curve, the forward total variance from AM to PM
is `0.4² × (1-fraction) / 252`: respectively `0.0005714286`, `0.0005079365`,
and `0.0003174603`. Phase 5 supplies these time inputs but does not yet execute
calendar no-arbitrage fitting, the full DLV transition, or experiment horizon
masks. Those population effects must be measured against the authoritative
891-date dataset in Phases 6 and 9; this report deliberately does not claim
results from stages that are not implemented.

Offline tests pin the exact AM shift, PM invariance, NMS response, fraction
endpoints, configuration identity, and the historical Saturday-AM zero-DTE
exception.
