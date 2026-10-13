# SANOS quantitative/data offline completion final re-review

**Reviewer:** `quantagent`  
**Date:** 2026-10-13  
**Baseline:** `ffecaed34af00adc1364cf5e334daeff290d6352` plus the settled,
uncommitted Phase 11/12 worktree  
**Scope:** final offline review against `AGENTS.md`, `PROGRAM_PLAN.md`, and
`reviews/program/quant-offline-completion-audit.md`; no WRDS or network access
and no modification of `quant_data`

## Verdict

**PASS — the quantitative and data-contract prerequisites for Phases 11 and 12
are complete offline.**

The three Priority-1 gaps and both Priority-2 gaps from the prior completion
audit are closed. I found no remaining offline numerical, population, lineage,
comparison, promotion, public-boundary, or operator-contract blocker. This
approval authorizes neither WRDS access nor public redistribution. The guarded
live pilot and fresh production observations remain separate evidence gates.

## Quantitative and data findings

### 1. Exact population and Raw closure — passed

The frozen production profile now proves, rather than merely documents, the
complete population identity:

- **918** ordered NYSE sessions comprise **911** full sessions and the exact
  **7** half days;
- the legacy catalog has **892** files: **891** populated common dates and the
  exact typed-empty marker for 2025-01-08;
- the **26** declared catalog gaps are required to be populated fresh dates,
  including all seven half days; consequently the fresh Dataset has **917**
  populated dates and one typed-empty date;
- terminal classifications must be 891 `match` or evidence-bound
  `approved_discrepancy`, 26 `oracle_gap`, one `policy_empty`, and zero
  unexplained discrepancies; and
- the catalog aggregates remain pinned, including 26,732 expiries, 2,414,341
  strike nodes and 4,651,416 active smooth-cache books.

The Raw distinction requested by the owner is exact. Acquisition contains
**3,676** immutable Raw artifacts:

```text
security 1 + options 918 + spot 919 + forwards 919 + zero_curve 919
```

The fitted daily chains consume **3,674** unique Raw pointers. The prior-session
2021-12-31 spot is consumed; the prior-session forward and zero-curve
partitions are exactly the two acquisition-only artifacts. The contract now
compares the actual pointer sets and rejects a count-preserving swap, so the
3,676/3,674 distinction cannot pass through numerically equal counts alone.

### 2. Daily comparison and Dataset binding — passed

The Phase 11 daily profile freezes SPX, 2022-01-03 through 2025-08-29,
`default-v1`, half-day inclusion, the default daily/surface configurations, the
0.2 overnight fraction, the hash-pinned oracle catalog, the exact 26-date gap
declaration, and the named numerical tolerances. Gap dates cannot be converted
to approvals, the policy-empty reason is exact, approvals bind the mismatch
field set and both fresh and legacy bytes, and an unused or broadened approval
fails closed.

Promotion evidence reconciles the comparator's fitted Intermediate artifacts
with the published internal Dataset. All 918 ordered Dataset partition records
must identify the same fitted artifact and the same fitted-source payload hash
as the corresponding daily report record. This preserves the one-frame-per-day
WIDB/no-arbitrage/DLV/smooth-SANOS result and prevents a comparison of one daily
population from authorizing a different Dataset.

The current checked profile still resolves to the frozen internal Dataset ID
`2ea20498735adf1e625b84f99c812bae78c90e9aeb81b36c9e0a330a21bd4585`.
The identity is a planned contract; only fresh semantic validation and the
subsequent evidence promotion make its materialization authoritative.

### 3. Actual legacy Experiment comparison — passed offline

The new comparator targets the real 891-date DySANOS daily-front migration
oracle, not a dead or synthetic HDF endpoint. It independently reads the fresh
Experiment, invokes the hash-pinned legacy selector/consumer in an isolated
CPU-only process, and compares:

- ordered date, expiry, AM flag, quote, price-location, strike and side
  geometry;
- bid, ask, fitted price, liquidity, open-interest and absolute spot-reference
  fields used by the internal Experiment;
- ATM/model volatility, NMS bounds, audit counts, quote identities and the
  rebased training fingerprint;
- recovered day tensors and padded batch tensors, including masks, shapes and
  dtypes; and
- the exact legacy population: **3,566,258** books, 1,783,129 calls and puts,
  `Qmax=7,898`, `Pmax=3,949`, `Emax=36`, 729/162 diagnostic date split and 861
  weekly pairs.

Float32 consumer values are exact; named Float64 comparisons use the frozen
`1e-12` absolute tolerance and zero relative tolerance. The fresh Experiment
must contain 917 dates, the ordered 891 common dates, and exactly the 26
fresh-only dates. Its checked plan still resolves to
`b621170c92b039aefb8987e9e20fc49927020834b28ff68665b51c4b99f3f792`
and derives from the exact internal Dataset above.

The worker now pins and verifies all **27** local `quant_data` source files in
its observed executable import closure, rejects an unpinned local module, and
records Python, NumPy and Torch runtime identities. Independently hashing the
read-only checkout found 27 matches and zero mismatches, and loading the pinned
consumer observed all 27 files. Parent-side interruption or progress failure
terminates, bounded-waits for, kills if needed, and reaps the owned worker.

No fresh Experiment exists yet, so the 891-day value comparison itself remains
live production evidence. The implementation is deliberately fail-closed: if
an approved vendor revision changes an Experiment tensor, promotion still
stops until that observed consequence receives an explicit reviewed treatment;
there is no silent tensor waiver.

### 4. Overnight-variance propagation — passed

The default remains exactly `overnight_variance_fraction=0.2` and remains part
of canonical configuration and artifact identity. Existing endpoint tests cover
0 and 1. The completed synthetic sensitivity runs 0.1, 0.2 and 0.5 through AM
clock construction, NMS/filtering, initial no-arbitrage fit, unsmoothed
compatibility DLV, smooth SANOS, daily-front projection, HDF5 write and semantic
validation. It proves:

- exact first-AM business times of 10.1, 10.2 and 10.5 days;
- an unchanged following PM time of 31.0 days;
- changed compatibility-DLV vectors; and
- the expected 10.25-day horizon boundary: 0.1 and 0.2 retain the AM expiry,
  while 0.5 rejects it.

Together with the existing calendar, variance, state-reconstruction and public
projection tests, this closes the required clock-to-DLV/SANOS-to-Experiment
chain without changing production semantics.

### 5. Immutable evidence, authority and distribution roles — passed

The Phase 11 index now binds the exact Dataset manifest/configuration/source
revision, 918-partition daily reconciliation, daily report and approval digest,
public-candidate pointer and manifest digest, Experiment pointer/manifest and
comparison report, and every input evidence-file digest. Index construction
reopens the public candidate and rejects a stale or replaced on-disk manifest.
Real build-miss tests exercise concurrent promotion convergence, missing review
evidence, stale index evidence and copied-evidence tampering.

Both independent review attestations bind the Dataset, daily report,
Experiment report and evidence index identities and byte hashes. Promotion
creates a separate immutable **internal Dataset-class designation**; it neither
rewrites the curated Dataset nor grants distribution rights. Shallow validation
is self-contained, while deep validation deliberately reopens the designated
Dataset and disposable Experiment.

The authority model is correct:

- the freshly built, validated and dual-reviewed internal SANOS **Dataset**
  becomes canonical only after successful promotion;
- the old 891-date `quant_data` collection remains only the migration oracle;
- the HDF5 **Experiment** remains disposable and never becomes canonical; and
- the lossy **public_candidate** remains a technical validation artifact. A
  separate `public` release still requires license, rights basis, approver,
  approval date and durable governing reference. Normalization, smoothing or
  leakage validation does not imply redistribution rights.

### 6. Human operator and release contract — passed offline

The production runbook freezes a new root, source generation, run ID, commands,
3,676-artifact empty-root preflight, cache-only proof, at-most-two-connection
live command, restart behavior, 918-partition build, both comparisons, final
config binding, public-candidate and Experiment validation, evidence indexing,
dual review and designation. The operator extra includes WRDS, HDF5 and the
Torch runtime required by the pinned comparator. Wheel/sdist resources include
the oracle catalog, checked configs, gap declaration and exact source revision,
and the installed-worker preflight is exercised outside the checkout.

## Independent evidence

All commands below were offline with `WRDS_OFF=1`; no credential or WRDS access
occurred.

```text
Focused Phase 11/12 quantitative gate:
  13 passed in 5.96s

Final public-manifest defensive regression:
  1 passed in 1.32s

ruff check src tests scripts setup.py:
  passed

python -m compileall -q src tests scripts setup.py:
  passed

git --git-dir=.git-real diff --check:
  passed

Current planned identities:
  Dataset    2ea20498735adf1e625b84f99c812bae78c90e9aeb81b36c9e0a330a21bd4585  match
  Experiment b621170c92b039aefb8987e9e20fc49927020834b28ff68665b51c4b99f3f792  match

Pinned legacy consumer closure:
  27 files, 0 hash mismatches; runtime loader observed 27 files
```

The settled-tree implementation evidence additionally reports **305 passed**
for the complete offline suite, two intentional wildcard warnings, clean Ruff,
compile, Markdown-link (73 links), diff checks, and a passing outside-checkout
wheel/sdist verifier. My earlier full-suite run crossed an actively changing
worktree and is not used as acceptance evidence; the settled high-risk tests
above were rerun independently.

`gpuctl resources --json` could not reach the daemon and fell back to local
measurement. Local RAM/disk checks were `OK`, but host policy treats the missing
daemon as a warning and the production runbook correctly treats any fallback as
a blocker for material/live work.

## Gates outside this offline PASS

1. **Phase 10 live pilot:** do not access WRDS until the owner is present and
   gives a new immediate 2FA-readiness confirmation, confirms no other account
   client is active, and daemon-backed `gpuctl resources --json` reports `OK`
   for every check. Use at most two connections and one authentication attempt.
2. **Full fresh build:** only a successfully reviewed pilot can authorize the
   3,676-artifact download. Actual vendor schemas/rows, all 918 daily outcomes,
   solver completion, 891-date numerical parity, 917-date Experiment and
   public-candidate populations require fresh evidence and both reviews.
3. **Discrepancies:** any unknown empty day, failed invariant, common-period
   mismatch or downstream tensor difference stops promotion. Vendor revisions
   may be approved only with exact evidence; fresh values must never be changed
   to force legacy equality.
4. **Public release:** no `public` artifact may be approved without actual
   redistribution authority and the complete legal metadata. This PASS covers
   only the internal Dataset and technical public candidate.
5. **Repository release:** project/paper/imported-code licensing and the
   `.git-real` metadata handoff remain owner/host decisions, not quantitative
   approval.

Signed: `quantagent` — independent quantitative/data-contract reviewer
