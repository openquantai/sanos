# Phase 2 independent code review

**Reviewer:** code agent (`/root/code`)
**Date:** 2026-10-12
**Verdict:** **CHANGES_REQUESTED**

## Scope

I reviewed the Phase 2 implementation and documentation against `AGENTS.md`,
`PROGRAM_PLAN.md` Phase 2, and `reviews/phase-2/evidence.md`. The review covered
the complete uncommitted Phase 2 change set, including the artifact store,
configuration/location resolution, process runner and journal, CLI, tests, and
documentation. I did not access WRDS or modify `quant_data`.

The ordinary offline checks pass, but the acceptance gate is not met because
the adversarial checks below expose path-containment, lock-lifetime, restart,
and lifecycle-contract failures.

## Findings

### 1. High — Artifact and lock paths can escape configured storage roots through symlinks

Evidence:

- `src/sanos/artifacts/store.py:297-313` resolves the destination before an
  atomic write. If the destination is a symlink, this follows the link and
  replaces its target outside the intended tree.
- `src/sanos/artifacts/store.py:396-437` validates only the lexical relative
  path. A symlinked parent inside the staging tree lets `write_bytes` and
  `copy_file` write outside staging before commit later notices the symlink.
- `src/sanos/artifacts/store.py:542-551` joins a namespace to a configured
  storage-class root without checking canonical containment or rejecting
  symlinked components. A namespace symlink redirects staging, lock, and final
  artifacts outside the configured root.
- Existence checks use `Path.exists()`, so a dangling final symlink is not
  treated as a collision and fails later with an implementation exception.
  The `.locks` directory and individual lock paths are not protected against
  symlink substitution either.

I reproduced all three write escapes under `/tmp/sanos/phase2-code-review`:
an atomic write changed a symlink target, a nested staging symlink created an
external file before commit failed, and a namespace symlink caused a published
artifact's resolved path to lie outside the configured dataset root.

Required change: establish one no-follow path-containment policy for roots,
namespaces, staging paths, destinations, and lock files. Reject every symlink
or non-directory path component before writing; use `lstat`-style collision
checks for final paths; and re-check containment at the operation boundary to
limit time-of-check/time-of-use exposure. Add adversarial tests for destination,
parent, namespace, dangling, and lock symlinks.

### 2. High — Lock release is not unconditional on setup or cleanup failure

Evidence:

- `src/sanos/artifacts/store.py:370-394` acquires the artifact lock before
  stale-staging cleanup and directory creation, but errors from those operations
  bypass `_release_lock()`.
- `src/sanos/artifacts/store.py:496-505` performs staging cleanup before lock
  release in `__exit__`; a cleanup error also leaves the lock held.

Fault-injection checks that made `shutil.rmtree` fail confirmed that the lock
remains held after both `__enter__` and `__exit__` failures. This can deadlock a
long-lived CLI or test process and violates interrupted-run recovery.

Required change: use nested `try/finally` blocks so every post-acquisition path
releases the lock, while preserving the primary exception and reporting any
cleanup failure. Add fault-injection tests for stale cleanup, staging creation,
commit, and exit cleanup failures.

### 3. High — The append-only journal is not recoverable after realistic emit failures

Evidence:

- `src/sanos/pipeline/runner.py:161-205` advances `_sequence` before payload
  canonicalization and durable append. A canonicalization error consumes a
  sequence number: the next successful record starts at sequence 2 and a new
  `RunJournal` rejects the file because sequence 1 is absent.
- A short/partial `os.write` followed by an exception leaves a truncated JSON
  tail. Journal construction then rejects the entire file, preventing restart.
- Initial sequence discovery happens outside the append lock, so two journal
  instances can begin from the same sequence.
- `src/sanos/pipeline/runner.py:301-309` runs reuse probes before the interrupt
  handling region. A probe raising `KeyboardInterrupt` leaves only
  `run_started`, with no `run_interrupted` event.
- The `ProcessPoolExecutor` context-manager shutdown path is not tested under a
  real interrupt and may wait for running workers instead of promptly recording
  and returning control for restart.

Required change: assign/advance sequence numbers while holding the lock and
only after a complete durable append; define recovery for an incomplete final
record (without accepting corruption in earlier records); encompass probes and
execution in the interruption/failure lifecycle; and define explicit executor
cancel/shutdown behavior. Add tests for canonicalization failure, injected
partial writes, concurrent journal instances, interrupt during probe, and
interrupt with active workers followed by restart.

### 4. High — Generic dataset publication can violate the mandatory distribution contract

`StorageClass.DATASET` artifacts can currently be finalized and validated with
arbitrary or empty metadata. Nothing in `src/sanos/artifacts/store.py` requires
the immutable `internal`, `public_candidate`, or `public` status mandated by
`AGENTS.md`; nor are the approval fields for `public` enforced. Metadata is not
part of the artifact identity, so semantically different distribution states
can also collide under one identity.

Required change: either prohibit generic `DATASET` finalization until the typed
dataset publisher exists, or introduce and validate a typed distribution
contract now, including identity semantics and mandatory public-approval
fields. Test rejection of missing/invalid status and public artifacts lacking
license, rights basis, approver, approval date, or durable approval reference.

### 5. High — Storage-class lineage rules are labels rather than enforced lifecycle invariants

The store accepts arbitrary upstream edges for every class. It can therefore
publish a `RAW` artifact derived from an `EXPERIMENT`, or an `EXPERIMENT`
without a dataset source, despite the binding Raw/Intermediate/Dataset/
Experiment lifecycle. Deep validation will regard such graphs as valid.

Required change: define and enforce allowed lineage edges, or prevent
finalization of lifecycle classes whose typed publisher is intentionally
deferred. Add positive and negative graph tests, including public projection
from an internal dataset and experiment derivation from a dataset.

### 6. Medium — Runner journals do not identify the plan being resumed, and worker results are unbounded

`run_started` records do not contain a deterministic identity for the stage,
ordered units, payloads, configuration/schema/producer version, and expected
artifact pointers. Two materially different runs can therefore produce
indistinguishable restart provenance. In addition,
`src/sanos/pipeline/runner.py:245-253` accepts any string from a worker, with no
syntax or size bound, so the claimed “small artifact identity” IPC boundary is
not enforced. `DailyPlan.__post_init__` also accepts non-integral numeric worker
counts such as `1.5`, deferring failure to the executor.

Required change: persist and verify a canonical plan identity on restart;
record the artifact pointer expected and/or returned for each unit; validate a
bounded pointer/identity result type; and require a real positive integer for
`max_workers`. Cover changed-plan restart, oversized/invalid worker output, and
non-integral worker counts.

### 7. Medium — Structured CLI output does not cover Click usage and parsing errors

The custom configuration and not-implemented errors honor
`--log-format json`, but Click failures do not. For example,
`python -m sanos --log-format json --unknown` and an unknown option after
`config show` both emit prose `Usage`/`Error` output. Automation therefore must
still scrape human text for common failures, contrary to the stable structured
output requirement and the evidence claim.

Required change: route root and subcommand usage/parse errors through a stable
JSON error envelope when JSON mode is requested, while retaining Click's exit
code 2. Test unknown options, missing commands/arguments, and malformed values
at both root and nested-command levels.

The placeholder guidance is also stale: `raw download` points to Phase 3,
`intermediate build` to Phase 3, `dataset build/validate` to Phase 4, and
`experiment build` to Phase 5, while `PROGRAM_PLAN.md` assigns these to Phases
4, 5/6, 7, and 9 respectively. `tests/test_cli.py` currently asserts one of the
incorrect messages. Correct the guidance and tests.

### 8. Medium — Version dependencies omit output- and validation-relevant helpers

The `@version` dependency list for `ArtifactBuilder` omits `_copy_stream`, even
though that helper controls copied artifact bytes. `ArtifactStore` omits direct
validation dependencies including `_read_json_object`, `_safe_relative_path`,
and `_sha256_file`. Existing tests check only a version prefix, so they do not
verify that helper changes propagate into the resolved producer version as
required by `AGENTS.md`.

Required change: include every content-, identity-, and validation-relevant
helper in the decorated dependency graph (or refactor so dependencies are
transitive and explicit), and test that changing a dependency changes the
resolved version.

### 9. Medium — Directory durability is weaker than the documented atomic-publication claim

`atomic_write_bytes` fsyncs a leaf file and its immediate parent, and commit
fsyncs the staging root before rename. For nested partition paths, created
intermediate directories are not all fsynced bottom-up. A crash can therefore
lose nested directory entries even though the manifest has been published.

Required change: fsync all created directories bottom-up before the final
rename, then fsync the destination parent, and state the supported filesystem
assumptions. Add a focused call-order/fault-injection test for nested artifact
paths.

## Commands run

```text
env PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 TMPDIR=/tmp/sanos/phase2-code-review \
  python -m pytest -q -p no:cacheprovider
# 54 passed in 4.80s

ruff check --no-cache src tests scripts
# All checks passed!

env PYTHONPYCACHEPREFIX=/tmp/sanos/phase2-code-review/pycache \
  python -m compileall -q src tests scripts
# passed

env PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
# Success: no issues found in 28 source files

git --git-dir=.git-real diff --check
# passed
```

I also ran isolated offline Python probes under
`/tmp/sanos/phase2-code-review` for symlink containment, setup/cleanup failure,
journal canonicalization failure, partial journal writes, probe interruption,
and JSON CLI parse failures. No probe contacted WRDS.

## Residual risks after required changes

- Filesystem crash guarantees vary by filesystem and mount options; document
  the guarantee that has actually been tested.
- File locking is host-local. Multi-host storage must remain unsupported unless
  a distributed ownership protocol is introduced and tested.
- `version="*"` behavior is well covered on the ordinary path, but the path,
  journal, and lifecycle issues above must be fixed before its shallow reuse
  semantics can be considered safe at published boundaries.

---

## Re-review — 2026-10-12

**Reviewer:** code agent (`/root/code`)
**Final re-review verdict:** **CHANGES_REQUESTED**

I reviewed the complete remediated working tree and the remediation record in
`evidence.md`. The baseline suite and all newly added targeted tests pass. The
first-round fixes are substantial, but the active-worker restart boundary and
publication-time lineage contract remain incomplete, so Phase 2 cannot yet be
approved.

### First-round finding status

1. **Path containment and symlinks — RESOLVED for the reported cases.** Atomic
   destinations, nested payload parents, namespace descendants, final paths,
   dangling finals, and static lock symlinks now fail before an external write.
   Configured roots are deliberately canonicalized once; SANOS-controlled
   descendant components are checked without following a leaf symlink. The
   focused tests and independent probes pass.
2. **Unconditional lock release — RESOLVED.** Setup, implicit-commit, body, and
   cleanup fault paths release the artifact lock through nested `finally`
   handling, while cleanup errors are attached to the primary exception.
3. **Journal/restart safety — PARTIALLY RESOLVED.** Sequence assignment under
   the lock, canonicalization failure, concurrent instances, incomplete-tail
   recovery, corruption rejection, probe interruption, and journaling of pool
   interruption are fixed. Running process workers are not actually stopped;
   see remaining finding A.
4. **Dataset distribution contract — RESOLVED.** Distribution is typed,
   mandatory for Dataset, included in exact and compatibility identity, and
   `public` requires all approval fields with an ISO date. This remains
   infrastructure only and does not approve any public release.
5. **Lifecycle invariants — PARTIALLY RESOLVED.** Backward class edges and
   source-free Intermediate/Dataset/Experiment specs are rejected. Publication
   still accepts fictitious upstream pointers and cannot distinguish an
   internal Dataset from a public/public-candidate Dataset upstream; see
   remaining finding B.
6. **Plan identity and worker-result bounds — RESOLVED for successful results.**
   Plan identity covers the ordered work, ticker, stage, payload, schema, and
   concrete producer version; changed-plan journal reuse is rejected; worker
   counts are integral; and successful result/probe IPC is restricted to the
   size-bounded `ArtifactPointer` or `None`.
7. **CLI structured errors and phase guidance — RESOLVED.** Root and nested
   Click usage errors produce a single JSON error object with exit code 2 when
   requested, and the phase messages now agree with `PROGRAM_PLAN.md`.
8. **`@version` dependency graph — RESOLVED.** The resolved Builder/Store graph
   includes copy, checksum, JSON validation, path-safety, and directory-fsync
   helpers. The explicit version assertions pass.
9. **Nested directory durability — RESOLVED for artifact publication.** Payload
   and staging directories are fsynced bottom-up before rename, followed by the
   final parent fsync. The documented local-POSIX filesystem assumptions are
   appropriately scoped.

### Remaining finding A — High: an interrupted runner returns while active workers continue side effects

At `src/sanos/pipeline/runner.py:520-541`, `Future.cancel()` cannot cancel work
that has already started, and `ProcessPoolExecutor.shutdown(wait=False,
cancel_futures=True)` explicitly returns without waiting for it. The test named
`test_active_pool_interrupt_is_journaled_and_plan_can_restart` does not assert
that an active worker has stopped or that it cannot publish after the parent
returns.

An independent spawn-based probe made a worker write a marker after a delay,
waited until the worker had definitely started, then raised `KeyboardInterrupt`
from the progress callback. `run_daily_plan` returned after 0.327 seconds with
no marker; the same worker created the marker afterward. This creates a real
restart race: an immediate completion probe can report a miss and submit the
same unit while the prior worker is still publishing. Per-identity artifact
locks prevent simultaneous rename, but they do not make the second worker a
successful reuse; it can wait and then fail with `ArtifactExistsError`.

Required change: define and implement a bounded stop protocol for submitted
workers (cooperative cancellation plus bounded wait, or process termination
with cleanup), and do not return the interruption until no old worker can
continue material side effects. Alternatively, a reviewed protocol must prove
that an immediate restart joins/reuses the in-flight publication successfully.
Test with a worker-start handshake and a post-interrupt side-effect marker;
then restart immediately without a probe that merely fabricates a pointer.

### Remaining finding B — High: publication records labels, not validated upstream provenance

`_validate_lineage` at `src/sanos/artifacts/store.py:174-204` checks only the
storage class embedded in each pointer. Neither builder entry nor commit
(`src/sanos/artifacts/store.py:623-778`) requires an upstream artifact to exist
and validate. The ordinary artifact helper currently publishes Datasets over a
synthetic all-zero Raw pointer, demonstrating that a complete artifact can
claim nonexistent provenance.

The same representation cannot enforce the required internal-to-public
projection. I published and deep-validated this graph successfully:

```text
Raw -> internal Dataset -> public Dataset -> public_candidate Dataset
```

The final edge violates the binding rule that the public projection is made
from the completed internal Dataset. `ArtifactPointer` carries only class,
namespace, and ID, so `ArtifactSpec.__post_init__` cannot know the upstream
Dataset's distribution status.

Required change: validate every declared upstream artifact at the new-artifact
publication boundary while it is required to exist. For `public_candidate` and
`public`, require the designated projection source to be a validated
`distribution=internal` Dataset. Preserve current shallow validation for an
already published self-contained Dataset after upstream removal. Add tests for
missing/fictitious upstreams, public-on-public and candidate-on-public edges,
and a valid internal-to-public projection.

### Remaining finding C — Medium: creation of a journal is not directory-durable

`RunJournal.emit` fsyncs the journal descriptor
(`src/sanos/pipeline/runner.py:309-325`), but neither initial parent creation at
line 200 nor first creation of the journal file fsyncs the containing
directory. On a POSIX filesystem, file fsync alone does not make a new
directory entry crash-durable. The implementation therefore falls short of its
“durable append-only JSONL journal” contract precisely on the first-run recovery
path.

Required change: fsync each newly created journal parent and the journal's
immediate parent after first file creation, under the same documented
filesystem assumptions as artifact publication. Add a helper-call/fault test
for a new nested journal path.

### Re-review verification

```text
env PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  TMPDIR=/tmp/sanos/phase2-code-rereview \
  python -m pytest -q -p no:cacheprovider
# 70 passed in 4.95s

pytest [16 targeted remediation nodes]
# 16 passed in 0.92s

ruff check --no-cache src tests scripts
# All checks passed!

env PYTHONPYCACHEPREFIX=/tmp/sanos/phase2-code-rereview/pycache \
  python -m compileall -q src tests scripts
# passed

env PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
# Success: no issues found in 28 source files

git --git-dir=.git-real diff --check
# passed before this Markdown-only append
```

Additional offline probes under `/tmp/sanos/phase2-code-rereview` covered the
active-worker post-interrupt write, deep validation of public-on-public
lineage, expanded root/nested JSON usage failures, and resolved `@version`
graphs. No WRDS connection or `quant_data` mutation occurred.

The no-Redun decision remains coherent: these are correctness gaps in the
small local runner and artifact contract, not evidence that a workflow engine
would simplify the current design.

---

## Second re-review — 2026-10-12

**Reviewer:** code agent (`/root/code`)
**Final verdict:** **PASS**

I independently reviewed the second remediation, including the final
exact-single-upstream correction applied during this review. All three
remaining findings are resolved, the complete offline gate passes, and I found
no new Phase 2 blocker. This approval is limited to the configuration,
artifact, CLI-shell, and local-runner foundation; it does not approve WRDS
access, numerical behavior, a Dataset release, or redistribution.

### A. Active-worker interruption — RESOLVED

`_stop_process_pool()` now:

1. cancels futures that have not started;
2. gives submitted work a one-second grace period;
3. terminates only the direct child processes captured from this executor;
4. joins those children against one shared one-second deadline;
5. kills any survivors and joins against a second shared deadline; and
6. waits for executor shutdown before the original interruption propagates.

The repository test uses a worker-start handshake, interrupts while that worker
is active, proves its delayed side-effect is absent both immediately and after
the worker's original deadline, and immediately reruns the same plan without a
fabricated completion pointer.

I additionally ran a worker that installed a `SIGTERM` ignore handler and
would have written after ten seconds. The kill fallback returned in 2.336
seconds; the completion marker remained absent, and
`multiprocessing.active_children()` was empty. A simultaneous unrelated child
process remained alive, confirming that cleanup ownership is limited to the
executor's own process map.

The implementation uses CPython's private `ProcessPoolExecutor._processes`
because Python 3.12 has no public terminate-workers API. This is acceptable for
the tested CPython 3.12 baseline, but it is a recorded compatibility risk:
future Python/interpreter support must re-run this handshake/ownership test or
replace the private access with an available public termination API.

### B. Publication-time provenance and public projection lineage — RESOLVED

Immediately before final publication, every declared upstream is now resolved
and shallow-validated. Missing, incomplete, corrupt, or falsely labelled
upstream artifacts therefore prevent publication. Existing artifacts retain
the required output-first behavior: ordinary shallow validation does not need
upstreams to remain, while explicit deep validation traverses them.

For `public_candidate` and `public`, all three enforcement points now require
the complete direct-upstream tuple to contain exactly one item, that item to be
a Dataset, and its validated manifest to declare `distribution=internal`:

- `ArtifactSpec.__post_init__()` rejects a mixed or multiple direct source;
- `_validate_publication_upstreams()` checks the real source immediately before
  publication; and
- `ArtifactStore.validate(..., deep=True)` enforces the same invariant during
  lineage audit.

The previous independent counterexample—one internal Dataset plus an extra Raw
pointer—now fails at spec construction for both non-internal statuses. Tests
also cover extra Intermediate input, fictitious upstreams, public-on-public,
candidate-on-public, and the valid internal-to-public path.

### C. First-journal durability — RESOLVED

New nested journal directories are created one level at a time and each new
directory entry is followed by a parent-directory fsync. On the first append,
the journal bytes are fully written and the file descriptor is fsynced before
the containing directory is fsynced. Later appends preserve the existing
append/fsync sequence and partial-tail recovery behavior. The focused helper
probe observes the journal's immediate parent in the fsync calls.

### Verification

```text
env PYTHONDONTWRITEBYTECODE=1 WRDS_OFF=1 \
  TMPDIR=/tmp/sanos/phase2-code-second-rereview-final \
  python -m pytest -q -p no:cacheprovider
# 72 passed in 9.72s

ruff check --no-cache src tests scripts
# All checks passed!

env PYTHONPYCACHEPREFIX=/tmp/sanos/phase2-code-second-rereview-final/pycache \
  python -m compileall -q src tests scripts
# passed

env PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
# Success: no issues found in 28 source files

git --git-dir=.git-real diff --check
# passed before this Markdown-only append
```

Focused offline checks also covered the active-pool restart test, nested
journal fsync test, real-upstream/internal-projection test, mixed Raw and
Intermediate rejection, a `SIGTERM`-resistant worker, and unrelated-process
ownership. No check contacted WRDS or modified `quant_data`.

The no-Redun decision remains appropriate. The explicit store and bounded
local runner now provide the Phase 2 behavior without exposing workflow-engine
state as a user-facing contract.
