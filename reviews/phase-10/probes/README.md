# Phase 10 discrepancy probes

These are the exact one-off read-only probes used for the Phase 10 discrepancy
diagnosis. They deliberately bind the local pilot paths and are retained as
evidence, not supported CLI tools. Run them only with `WRDS_OFF=1`.

`diagnose_fallback_objective.py` recomputes the copied `MkArbFree` surface
objective for the successful fresh and historical persisted states. Its
captured output was:

```json
{"date": "2025-01-02", "fresh_objective_recomputed": 0.0038227234366016447, "legacy_objective_recomputed": 0.003822728135601628, "mask_equal": true}
{"date": "2025-01-03", "fresh_objective_recomputed": 0.0021654474639510118, "legacy_objective_recomputed": 0.002165444884235431, "mask_equal": true}
```

`diagnose_fallback_parity.py` invokes untouched `quant_data` `MkArbFree 0.51`
on the exact fresh filtered Float32 inputs before comparing states. In the
captured runtime the first 2025-01-02 invocation reached the unchanged
300-second timeout/iteration limit, so the script correctly terminated before
printing a comparison record. The later clean standalone rebuild captured the
same boundary more completely in the checksummed run journal cited by the
parent evidence document.
