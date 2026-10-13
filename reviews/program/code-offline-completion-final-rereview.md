# SANOS offline-completion final code re-review

Date: 2026-10-13  
Reviewer: `code` (independent code/engineering reviewer)  
Baseline: `ffecaed34af00adc1364cf5e334daeff290d6352` plus the settled Phase 11/12 worktree  
Network/WRDS access: none

## Verdict

**PASS — the offline Phase 11/12 engineering work is approved.** All six
blocking/high findings in `code-offline-completion-rereview.md` are resolved. The
full-range comparison/evidence/promotion path, packaged operator resources,
installed-distribution verifier, operator documentation, and CI contract are
coherent with `AGENTS.md` and `PROGRAM_PLAN.md`.

This is an offline code approval only. It does not authorize a WRDS connection,
the Phase 10 pilot, a full production run, public redistribution, or the Git
metadata handoff. Those independent gates are listed below.

## Resolution of prior findings

### 1. Oracle subprocess ownership — resolved

`src/sanos/reference/experiment_compare.py:158-253` now owns the worker for its
entire lifetime. Every parent-side `BaseException` invokes a bounded
terminate/wait, escalates to kill after timeout, reaps the child, and closes both
pipes. The normal path also closes pipes. Parameterized tests inject a progress
callback failure and `KeyboardInterrupt`, including the forced-kill path, and prove
termination, reaping, and stream closure.

### 2. Pinned consumer source closure — resolved

`PINNED_SOURCE_SHA256` now contains the 27 local files needed by the pinned
consumer, including `flex.metrics`, `flex.model`, `flex.sanos`, and
`quantdata.torch.ffw`. `_load_pinned_consumer` hashes every declared source before
import and then examines loaded `quantdata.*` modules, rejecting any local imported
source outside the declared closure. A changed transitive-source regression is
present. The Phase 11 contract requires the worker's complete hash map, while the
report records Python, NumPy, and Torch versions.

An independent import against the read-only `quant_data` tree succeeded with
exactly 27 declared and 27 observed hashes and no unaccounted declared path.

### 3. Operator runtime and built-wheel preflight — resolved

`pyproject.toml` now includes bounded Torch dependencies in both `reference` and
the aggregate `operator` extra, and the human operator guide describes that
runtime. `scripts/verify_distribution.py` installs the built wheel outside the
checkout and launches the actual isolated worker preflight, requiring HDF5, Torch,
the protocol identity, and representative transitive pinned sources. This is no
longer merely a mocked CLI path.

The independent source-tree preflight returned the expected protocol with Python
3.12.3, NumPy 2.5.4, Torch 2.14.1, HDF5 3.16.0, and 27 pinned source files.

### 4. Static gate — resolved

The prior overlong promotion-test expression is formatted. The exact repository
Ruff command passes, as do compileall and `git diff --check`.

### 5. Public-candidate manifest binding — resolved

Public Dataset validation now emits the checksum of the validated manifest.
`build_phase11_evidence_index` requires and records that checksum. During evidence
publication, the CLI reopens the exact candidate through `ArtifactStore`, hashes
the manifest currently on disk, validates it against the index, and includes the
result in the final artifact binding. `attach_phase11_artifact_binding` requires
the pointer and digest to agree. Tests cover successful binding and stale-manifest
rejection; the final defensive type check also rejects a non-string supplied
checksum before applying the hexadecimal matcher.

### 6. Immutable one-file evidence publication — resolved

`atomic_create_bytes`/`atomic_create_json` provide canonical durable create-new
publication using an exclusive temporary file, fsync, an atomic no-replace hard
link, parent-directory fsync, no-follow checks, and cleanup. Review attestations,
promotion configurations, and public approval configurations now use this helper.
Tests cover concurrent different-content publishers, symlinked leaf and parent
paths, invalid setup, injected commit failure, and temporary-file cleanup.

## Broader Phase 11/12 assessment

- The production daily contract fixes the exact 918-session range, 917 populated
  sessions, 26 migration-oracle gaps, the typed-empty policy day, source generation,
  and the expected Raw table/date closure. A free-form extra gap cannot silently
  become acceptance evidence.
- Public-candidate and Experiment configurations are finalized deterministically
  from the actual internal Dataset pointer. The evidence index binds Raw preflight
  and validation, daily comparison, Dataset, public candidate, Experiment, exact
  report bytes, partition lineage, and both independent review attestations.
- Canonical promotion is a new immutable internal Dataset-class designation. It
  neither mutates the source Dataset nor grants redistribution rights. Output-first
  shallow and wildcard reuse remain self-contained, while a build miss performs
  deep source checks. Concurrent publishers converge on one artifact.
- The fresh validated/promoted SANOS Dataset is the prospective authority. The old
  891-date `quant_data` artifact is only the migration/parity oracle, and the HDF5
  Experiment remains disposable.
- Wheel resources use `importlib.resources` and carry the exact build revision.
  The sdist includes the documented source/provenance material. The outside-checkout
  verifier covers both CLI entry points, resources, provenance, configurations,
  rebuilt wheel, and worker runtime preflight.
- CI keeps `WRDS_OFF=1`, runs Ruff, mypy, compileall and the complete offline suite,
  checks local Markdown links, verifies wheel/sdist outside the checkout, and builds
  the paper out of tree.
- The human operator guide and guarded pilot/production runbooks provide ordinary
  CLI commands, stable JSON output, exit-code and restart guidance, retention rules,
  explicit 2FA/resource stop points, and a review handoff without requiring an
  agent.

## Independent commands and evidence

All commands were offline. `~/python/quant_data` was read only.

```text
env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/test_experiment_reference_compare.py tests/test_production.py \
  tests/test_dataset_promotion.py tests/test_artifact_store.py
# 41 passed in 3.90s

env WRDS_OFF=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider tests/test_production.py
# 6 passed in 1.72s (repeated after the final checksum type-check edit)

env WRDS_OFF=1 ruff check src tests scripts setup.py
# All checks passed

env WRDS_OFF=1 python -m compileall -q src tests scripts setup.py
# passed

python scripts/check_markdown_links.py
# checked local links in 73 Markdown files

git --git-dir=.git-real --work-tree=. diff --check
# passed

# Fresh worker/source-closure probe against read-only quant_data:
# 27 declared hashes, 27 observed hashes, no missing declaration

# Real isolated worker preflight:
# sanos-experiment-oracle-worker-preflight-v1; Python/NumPy/Torch/HDF5 present;
# 27 pinned source files

# Fresh-process cdxcore version resolution:
# worker, comparison, atomic-create, and Phase 11 evidence-index closures resolved
# without recursion or missing dependency
```

The implementation owner additionally reports the settled full offline suite as
305 passed with two intentional wildcard warnings, plus passing distribution
build/verifier and the same static/documentation gates. I did not rerun the
five-minute full suite or distribution build because the focused tests exercise
the remediated risk areas directly. `mypy` was not installed in this review shell;
the checked CI workflow retains it as a mandatory gate.

## Remaining gates outside this approval

1. **Live Phase 10 pilot:** obtain immediate owner-present WRDS 2FA readiness,
   confirm no other account client is active, and require daemon-backed
   `gpuctl resources --json` with every check `OK`. Keep connections bounded and
   dual-review the resulting external evidence.
2. **Full production:** only after the pilot passes, obtain a new immediate 2FA
   confirmation and execute the fresh full-range download/build/comparison. The
   real output identities and numerical results cannot be approved offline.
3. **Public redistribution:** a `public_candidate` is not permission to release.
   A public Dataset still requires actual rights/license evidence, approver, date,
   and durable governing reference.
4. **Project release:** the owner must decide the licensing treatment of project
   code, imported code, and paper/assets. The ignored `.git-real` database must be
   handed off to ordinary `.git` only after the protected empty mount is removed.

Signed: `code` — independent code/engineering reviewer
