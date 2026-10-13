# SANOS offline completion-gap audit

Date: 2026-10-13  
Reviewer: `code` (independent code/engineering reviewer)  
Baseline: clean commit `ffecaed34af00adc1364cf5e334daeff290d6352`  
Scope: binding `AGENTS.md`, the complete `PROGRAM_PLAN.md`, package/CLI/docs,
checked configurations, tests/CI, and the signed phase reviews  
Network/WRDS access: none

## Verdict

**OFFLINE COMPLETION WORK REMAINS.** Phases 0--9 and the Phase 10 offline
prerequisites have reviewed implementations. No additional numerical-core,
daily-pipeline, artifact-store, public-projection, or HDF5 redesign is indicated
before the pilot. However, the repository is not yet at the program's offline
completion boundary: the full-range production operation is not frozen as a
human-executable evidence contract, built distributions are not self-contained
for documented reference/provenance operations, and Phase 12 release/operator
checks are not implemented.

These findings do **not** authorize live WRDS access. The Phase 10 pilot remains
separately blocked on immediate owner-present 2FA readiness, confirmation that
the account has no other active client, a daemon-backed all-`OK` resource probe,
the bounded pilot itself, and dual review of its external evidence.

## Offline priorities

### P0 — Freeze and test the Phase 11 production contract before the pilot can promote to full range

The repository has an excellent, exact Phase 10 pilot runbook, but no equivalent
Phase 11 runbook or machine-checked full-range acceptance profile. The generic
commands can perform the individual operations; what is missing is the binding,
restartable sequence and evidence contract needed to prove that the production
run is the requested *fresh* build rather than a merely valid build.

Implement and dual-review, entirely offline:

1. a checked Phase 11 operator runbook with immutable run/source-generation
   identities, clean-root preflight, dry-run expectations, resource/2FA stop
   points, retry limits, restart commands, redaction, checksums, and review
   handoff;
2. a full-range comparison contract binding the exact 2022-01-03 through
   2025-08-29 calendar, the pinned legacy catalog, the exact expected oracle-gap
   set, the known typed-empty policy day, and the configured half-day policy.
   An operator must not be able to turn a new missing oracle date into a passing
   Phase 11 result merely by adding another free-form `--expected-oracle-gap`;
3. preflight evidence that every required Raw artifact in the new production
   generation is initially missing and that all boundary dependencies are in
   the plan. Resume may reuse only artifacts created in that same immutable
   generation;
4. an evidence index binding Raw validation, fitted comparison, internal
   Dataset identity/validation, public-candidate identity/validation, and
   Experiment identity/validation. Each step must stop on unexplained
   discrepancy and be repeatable output-first;
5. a deterministic way to finalize the public-candidate and Experiment
   configurations with the newly published internal Dataset pointer. The final
   canonical configurations and their digests must be emitted as evidence; the
   operator should not have to perform undocumented JSON surgery.

Acceptance can be tested with synthetic/completed fixture artifacts and a fake
full-range evidence bundle. Actual identities, counts, and comparison outcomes
remain live-data evidence.

### P0 — Make built distributions honor the documented CLI contract

An isolated build exposed a real checkout-versus-wheel split:

- the wheel contains only Python modules and distribution metadata; it contains
  neither the pinned default legacy catalog nor checked reference configs;
- `reference compare-daily` computes its default catalog as
  `Path(__file__).parents[3]/reviews/phase-1/reference-market-catalog.json`.
  From the built wheel that path does not exist, and Click exits 2 before the
  comparison starts;
- publication and comparison accept `SANOS_SOURCE_REVISION` outside a Git
  checkout, but the built artifact does not embed a source revision. A wheel
  user must know and supply an undocumented release-build detail despite the
  documentation's claim of packaged provenance;
- the sdist contains the package and a partial `tests/` tree, but no `configs/`,
  `docs/`, `reviews/`, `tex/`, test fixtures, reference probes, or imported
  reference snapshot. It is therefore neither the documented project source
  distribution nor a self-testing minimal runtime distribution.

Choose and test one explicit release model. The recommended minimal model is to
package immutable runtime resources under `sanos` and load them with
`importlib.resources`: the pinned catalog, portable reference configs, and a
build-generated exact source revision. Keep large provenance/history and paper
assets in the sdist/repository, not the wheel. Alternatively make every such
path required and remove misleading defaults, but then the installed-package
documentation must supply the assets and exact arguments. Add wheel and sdist
tests executed from outside the checkout, including both CLI entry points,
default resource lookup, source provenance, config parsing, and package-data
inventory.

The release metadata also needs an explicit `readme` and, after the owner makes
the licensing decision below, license metadata/files. CI's current successful
`python -m build` only proves archive creation, not usability of either archive.

### P1 — Complete Phase 12's non-AI operator and release verification offline

The current README and focused runbooks document most individual commands, but
Phase 12 remains unimplemented as an end-to-end release gate. Add one ordered
operator path from installation through location configuration, WRDS setup and
explicit 2FA gates, Raw plan/download/validation, Intermediate build, internal
publication and shallow/deep validation, public-candidate build, rights-gated
public release, Experiment build, safe cleanup, and recovery. It should include:

- exact text and JSON examples plus stable exit-code meanings;
- a troubleshooting table for missing roots/extras, `WRDS_OFF`, authentication,
  lease contention, corrupt artifacts, solver failure, wildcard miss, restart
  workspaces, and lineage removed after shallow publication;
- a storage-retention/cleanup policy identifying Raw and Dataset as durable,
  Intermediate and Experiment as reproducible/disposable, with no blanket
  deletion commands;
- installation profiles. `.[wrds]` alone is insufficient for the complete
  documented workflow because comparison and Experiment commands need the
  separately optional HDF5 dependency; offer an aggregate operator extra or
  document the exact combined extras;
- updated `sanos status` output. It currently reports
  `phase=internal-experiment-materialization`, which predates the implemented
  public Dataset and Phase 10 comparator and is misleading operational state;
- refreshed traceability/status prose so superseded design-review blockers are
  clearly archival and the authoritative final review is easy to identify.

### P1 — Turn the release acceptance sentence into executable CI

CI currently runs Ruff, mypy, the source-tree suite, and `python -m build`.
Before release it should additionally:

1. set `WRDS_OFF=1` for the entire ordinary test job (and retain explicit
   no-network tests) so a future test cannot accidentally cross the network;
2. install the wheel in a clean environment and exercise `sanos`,
   `python -m sanos`, resource lookup, and a representative offline command
   from outside the checkout;
3. inspect/install the sdist according to the selected distribution contract;
4. validate Markdown links or otherwise define what "build docs" means for the
   Markdown-only documentation;
5. build the paper out of tree with documented TeX dependencies and verify that
   no generated files enter Git; and
6. preserve the full offline test, mypy, Ruff, compile, and diff-clean gates.

This is all offline work. The current full-suite result recorded by Phase 10 is
256 passing tests; this audit did not rerun it because no implementation changed
and the host resource daemon was unavailable.

## Human/operational decisions not gated by WRDS data

1. **Project licensing:** `PORTING.md` correctly records that the paper has no
   explicit license and copied code must not be relicensed implicitly. The
   owner must choose the treatment of project code, imported MIT-declared code,
   and paper/assets before a public package/repository release. This is a
   human/legal gate, not a WRDS-data gate and not permission to infer public
   Dataset redistribution rights.
2. **Git metadata handoff:** this workspace still uses ignored `.git-real`
   because `.git` is a protected empty mount. Moving the real database to
   ordinary `.git` remains an operational release prerequisite once the mount
   is removed. It cannot be solved safely by application code.
3. **Open compatibility decisions:** CD-002/003/006/010/012/013/015/020/021 and
   the documented provisional behaviors remain intentional unavailable or
   compatibility-preserved surfaces. They do not block the reviewed production
   path. Do not silently fix them; address them only as separately approved,
   versioned future work.
4. **Public redistribution:** the technical `public_candidate` workflow is
   complete. A real `public` release still requires actual license and rights
   evidence, approver, date, and durable reference. This is deliberately
   separate from both code completeness and the fresh internal Dataset build.

## Work that genuinely requires live data or external state

- Phase 10's sole guarded pilot download, cache/restart proof, end-to-end fitted
  build, legacy comparison, evidence redaction, and dual evidence review;
- resolution or explicit approval of any vendor revision or pilot discrepancy;
- a fresh Phase 11 full-range WRDS extraction after a new immediate 2FA
  confirmation, followed by actual daily builds and artifact publication;
- comparison of real fresh populations, masks, fits, diagnostics and HDF5
  tensors against the migration oracle, and dual production sign-off;
- promotion of the resulting immutable SANOS Dataset identity as canonical;
- an actual public redistribution release, but only if separate rights/license
  evidence is obtained.

No offline fixture or passing unit suite can substitute for those observations.

## Evidence gathered

All probes were offline and used baseline `ffecaed`:

```text
git --git-dir=.git-real --work-tree=. status --short
# clean at audit start

python -m build --no-isolation --skip-dependency-check \
  --outdir /tmp/sanos/program-audit/dist
# sdist and wheel built

# wheel inventory
catalog resources: []
configs: []

# sdist inventory
configs: []; reviews: []; tests/fixtures: []; reference probes: [];
docs: 0; tex: 0

# wheel-backed CLI, outside checkout
reference compare-daily ...
# exit 2: default reference-market-catalog.json does not exist

PYTHONPATH=src python -m sanos --help
PYTHONPATH=src python -m sanos status
# both pass; status reports the stale internal-experiment phase label

gpuctl resources --json
# local measurements all OK, but daemon unreachable and source=local
```

The last result is a host-level WARN under `AGENTS.md` and remains a hard live
pilot blocker. No SANOS long-running job was launched, no GPU was allocated,
and no WRDS credential, connection, query, or proprietary artifact was touched.

## Recommended sequence

1. Implement and dual-review the Phase 11 offline contract/evidence tooling.
2. Repair wheel/sdist resource and provenance behavior; add installed-artifact
   tests.
3. Complete the Phase 12 operator documentation and CI release gate.
4. Obtain the separate project-license decision and perform the `.git-real`
   handoff when the host permits it.
5. Only then (and only after the existing immediate safety gates) run and review
   the Phase 10 live pilot. Phase 11 live execution remains blocked until that
   review passes.

Signed: `code` — independent program completion reviewer
