# Compatibility decision register

No issue listed here may be silently fixed or preserved during migration.
Each resolution needs evidence, code-agent review, quantagent review, and user
approval before it changes behavior.

## Open decisions

1. The legacy `widb_sanos_day.fit_sanos_day` path appears to group an empty
   expiry list and references stale fields. Determine whether it produced the
   reference dataset and specify the intended replacement.
2. Verify whether the WIDB option-column set check incorrectly accepts missing
   required columns.
3. Decide the intended validation behavior for empty enriched days where legacy
   schema checking appears unreachable.
4. Define public `forward_adjusted_spot_return`; do not relabel the internal
   simple close-to-close `spot_return`.
5. Reconcile fit-percentage documentation and logs.
6. Decide whether solver timeout participates in transformation identity.
7. Recover the reference dataset's exact option-population cap and filters.
8. Reconcile WIDB AM timing with the FLEX 4.8-business-day convention.
9. Define the public unit bridge between spot-normalized half-spreads and SANOS
   prices in discounted-forward units.

