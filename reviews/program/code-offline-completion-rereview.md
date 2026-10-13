# SANOS offline-completion code re-review

Date: 2026-10-13  
Reviewer: `code` (independent code/engineering reviewer)  
Baseline: `ffecaed34af00adc1364cf5e334daeff290d6352` plus the current uncommitted Phase 11/12 worktree  
Network/WRDS access: none

## Verdict

**CHANGES REQUIRED.** The new full-range daily contract, production runbook,
configuration finalizer, evidence-index structure, packaged resources, distribution
verifier, operator documentation, and CI jobs resolve most of the prior audit's
offline gaps. The worktree is nevertheless not ready to commit or to authorize the
live pilot because three acceptance-critical defects remain in the new Experiment
oracle path, and the current static gate is red. Two evidence-publication issues
should be closed in the same remediation.

This verdict does not supersede any earlier phase approval and does not authorize
WRDS access. The live 2FA/resource gates and the separate license and `.git-real`
handoffs are listed below.

## Blocking findings

### 1. High — interruption can orphan the long-running Experiment oracle worker

`_run_oracle_worker` creates a subprocess at
`src/sanos/reference/experiment_compare.py:161-177`, writes its request and then
iterates its stdout at lines 180-214. There is no `try/finally` that terminates,
waits for, and if necessary kills/reaps the owned process. A `KeyboardInterrupt`,
an exception from the progress callback at lines 198-207, or another parent-side
exception therefore escapes while the 891-day Torch worker can continue running.
This violates the repository's established spawned-worker ownership/restart
contract and the host rule that project-owned material work must be stopped under
resource pressure.

Required fix: put setup, request writing, output consumption, and waiting under
one ownership boundary. On every exceptional exit, close pipes, request graceful
termination, wait for a bounded interval, then kill and unconditionally reap.
Add subprocess-double tests for interrupt and progress-callback failure which prove
that no worker remains alive after the API returns or raises.

### 2. High — the advertised pinned consumer does not pin its executable source closure

`PINNED_SOURCE_SHA256` contains six files
(`src/sanos/reference/_experiment_oracle_worker.py:30-49`), and
`_load_pinned_consumer` validates only that set before importing the reference
consumer (`:159-195`). But the hashed `flex/training.py` imports and executes
additional local modules at import time, including `quantdata.dysanos.flex.metrics`,
`quantdata.dysanos.flex.model`, and `quantdata.torch.ffw`; the latter is imported by
`flex/config.py`. Changes in those unpinned files can alter import behavior or the
materialized consumer while the report continues to claim the same
`pinned_source_sha256` evidence.

Required fix: define and verify the complete local transitive code closure needed
to import and execute `materialize_day`/`materialize_batch` and record that complete
map in the report, or vendor a deliberately minimal exact oracle implementation and
prove it against the pinned consumer. Add a regression that changes one transitive
file in a copied code tree and requires rejection. Relevant runtime identities
(at least Python, NumPy, and Torch) should also be recorded with the comparison
evidence.

### 3. High — the documented aggregate operator installation cannot run the oracle

The worker directly imports the reference `training.py`, which imports `torch`.
However the `operator` extra at `pyproject.toml:48-52` contains only HDF5, pandas,
and WRDS dependencies, while `docs/operator-guide.md:10-23` describes
`pip install '.[operator]'` as the complete operator installation. A clean operator
environment therefore cannot perform the Phase 11 Experiment comparison required
by the production runbook.

Required fix: add a supported, bounded Torch dependency to an explicitly documented
operator/reference-comparison extra and make the aggregate operator profile include
it, or remove the runtime Torch dependency from the oracle. The distribution
verifier must exercise the real import/worker preflight from the built wheel rather
than a mocked comparator.

### 4. Blocking gate — Ruff currently fails

The required command fails with `E501` at
`tests/test_dataset_promotion.py:487` (104 characters). The worktree cannot be
approved while its own CI lint command is red.

Required fix: format that comprehension and rerun all static and offline test gates
on a stable worktree.

## High-priority evidence/publication findings

### 5. High — the Phase 11 index does not bind the public-candidate manifest bytes

Dataset and Experiment evidence record exact manifest SHA-256 values
(`src/sanos/production.py:468-472,641-655,680-690`). The public-candidate branch
accepts only a pointer, canonical configuration, and the bytes of a validation log
(`:570-593,686,705-707`). The CLI validation record itself emits no manifest digest
(`src/sanos/cli/main.py:2410-2428`). Thus the indexed evidence cannot prove which
manifest bytes were validated or detect a stale/replaced candidate independently of
the operator log.

Required fix: make public-candidate validation emit its validated manifest checksum,
require and store it in the Phase 11 index, and reopen the exact candidate during
index construction or artifact binding. Add stale-manifest and coherently re-signed
validation-log adversarial tests.

### 6. High — create-new review/promotion evidence has check-then-replace races

The review and promotion template commands first test `output.exists()` and then
call the replacement-capable `atomic_write_json`
(`src/sanos/cli/main.py:1697-1701` and `:1795-1799`). Two publishers can both pass
the check and the later one can replace the earlier immutable evidence. Parent
creation also lacks the no-follow/root-containment treatment used by the artifact
store. The same pattern appears in the public approval template at
`src/sanos/cli/main.py:2179-2183`.

Required fix: route all immutable evidence/config creation through one create-new,
locked, no-follow publication primitive with directory fsync and identical-content
reuse only where explicitly intended. Cover concurrent different-content writers,
symlinked output/lock/parent paths, setup failure, commit failure, and lock release.

## What is otherwise accepted in this re-review

- The daily Phase 11 comparator freezes the 918-session range, exact 26-date
  migration-oracle gap set, typed-empty policy day, source generation, and expected
  population contract; free-form gap expansion is no longer an approval path.
- Raw preflight requires a clean 3,676-artifact generation and the cache-only
  validation requires exact table/date multiplicities, including the three prior-day
  boundary partitions.
- Downstream public-candidate and Experiment configurations are deterministically
  finalized from the published internal Dataset rather than edited by hand.
- Promotion is a distinct internal Dataset-class designation, does not mutate the
  source Dataset, grants no redistribution rights, and supports self-contained
  shallow/wildcard reuse. The current test patch now exercises build-miss concurrent
  convergence, stale evidence, missing review evidence, and copied-evidence tamper,
  although the complete suite must be rerun after remediation.
- Wheel resource lookup and embedded source revision replace checkout-relative
  defaults. The sdist inventory and outside-checkout distribution verifier are a
  substantial improvement over the previous release shape.
- CI now sets `WRDS_OFF=1`, runs lint/type/compile/offline tests, checks Markdown,
  builds and verifies both distributions outside the checkout, and builds the paper
  out of tree.
- The operator and production runbooks correctly keep canonical authority with the
  fresh validated/promoted SANOS Dataset. The legacy 891-date artifact remains a
  migration/parity oracle; the HDF5 Experiment remains disposable.

## Commands and evidence

All commands were offline; no WRDS connection or network operation was attempted,
and `~/python/quant_data` was read only.

```text
git --git-dir=.git-real --work-tree=. diff --check
# passed

env WRDS_OFF=1 ruff check src tests scripts setup.py
# failed: E501 at tests/test_dataset_promotion.py:487

# Targeted source/test inspection covered:
# src/sanos/{production.py,datasets/promotion.py,reference/compare.py,
#            reference/experiment_compare.py,
#            reference/_experiment_oracle_worker.py,resources/__init__.py}
# src/sanos/cli/main.py, setup.py, MANIFEST.in, pyproject.toml,
# scripts/{verify_distribution.py,check_markdown_links.py}, CI, operator/runbooks,
# and the new Phase 11/12 tests.
```

A full suite was not rerun after finding the deterministic lint failure and observing
that the promotion test file changed during the review. The final gate must be run
from a settled tree after the blockers above are fixed.

## Gates deliberately outside this offline approval

Even after this review passes, the following remain separate prerequisites:

1. **Live WRDS pilot:** immediate owner-present 2FA readiness, confirmation that no
   other client uses the account, a daemon-backed `gpuctl resources --json` result
   with every check `OK`, bounded connections, and dual review of pilot evidence.
2. **Full production:** a successful reviewed pilot, then a new 2FA confirmation and
   the fresh full-range download/build/comparison. Offline fixtures cannot promote a
   Dataset.
3. **Public release:** actual rights/license evidence, approver, approval date, and a
   durable governing reference. Normalization or fitting does not imply rights.
4. **Repository release:** the owner must decide project/paper/imported-code licensing
   and move `.git-real` to ordinary `.git` only after the protected empty mount is
   removed. Application code must not attempt that handoff.

Signed: `code` — independent code/engineering reviewer
