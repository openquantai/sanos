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
and `0.0003174603`.

The completed offline pipeline now also runs the `0.1`, `0.2`, and `0.5` cases
through initial no-arbitrage fitting, compatibility DLV, smooth SANOS,
daily-front Experiment projection, HDF5 write, and semantic HDF5 validation.
The first AM expiry reaches the Experiment as `10.1`, `10.2`, and `10.5`
business days respectively, while the following PM expiry remains exactly
`31.0`; the compatibility-DLV vectors are not identical across the three clock
inputs. With an intentionally short `10.25`-business-day Experiment horizon,
the `0.1` and `0.2` cases retain the AM expiry and the `0.5` case is rejected as
out of horizon. This is a synthetic sensitivity check, not a claim about how
many dates or books change in the fresh production Dataset.

Offline tests pin the exact AM shift, PM invariance, NMS response, DLV response,
Experiment horizon response, HDF5 round trip, fraction endpoints,
configuration identity, and the historical Saturday-AM zero-DTE exception.
