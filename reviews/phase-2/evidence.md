# Phase 2 evidence — configuration, artifacts, CLI, and local runner

Review state: complete. After two remediation rounds, the code and
quantitative/data-contract reviewers both recorded `PASS`. No WRDS connection
or network-capable data command was invoked.

## Scope

- Added deterministic canonical JSON and SHA-256 configuration identities.
- Added read-only four-root location resolution with CLI, canonical
  environment, repository TOML, user TOML, general-root, and legacy fallback
  semantics.
- Added an immutable artifact store with per-identity locks, private staging,
  atomic writes, exact manifests, completion markers, checksums, shallow and
  deep validation, and output-first resolution.
- Added exact `version="*"` behavior: existing complete output only, all
  identities except producer version still checked, visible warning and log
  record, fail on miss, no mutation or build authority.
- Added a deterministic per-day planner and bounded spawn-based process runner
  with an append-only, fsynced JSONL parent journal and progress/ETA records.
- Extended the CLI with location options, `--dry-run`, structured output and
  errors, stable Phase/configuration exit codes, `config show`, enhanced
  `doctor`, and enhanced `status`.
- Added `filelock` as a core dependency and documented configuration, artifact,
  and runner contracts.

The change does not implement WRDS access, WIDB transformations, SANOS
numerics, dataset publication, public projection, or experiment generation.
Those commands continue to fail explicitly with exit code 7.

## Artifact schema

The artifact ID hashes storage class, namespace, canonical configuration ID,
schema version, concrete producer version, and sorted upstream pointers. The
separate compatibility ID omits only producer version.

Each final directory contains payloads plus:

- `manifest.json`: exact schema/status/identity fields, canonical configuration,
  upstream pointers, timezone-aware creation timestamp, checksummed file list,
  and canonical metadata;
- `COMPLETE.json`: exact schema/status/artifact ID and manifest checksum.

The marker is written last inside staging; the fully fsynced staging directory
is atomically renamed to its immutable identity. Validation rejects changed
field sets, identity drift, incomplete or malformed markers, unexpected files,
size/checksum drift, invalid timestamps, unsafe paths, and symlinks.

## Verification

Run on 2026-10-12 with Python 3.12.3:

```text
ruff check src tests
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests
Success: no issues found in 27 source files

python -m compileall -q src tests
(exit 0)

pytest -q
54 passed in 4.01s

git --git-dir=.git-real --work-tree=. diff --check
(exit 0)
```

Mypy 1.20.2 and its dependencies were installed under the project-specific
`/tmp/sanos/phase-2-tools` directory because the host interpreter did not have
the declared development extra installed. This did not modify the repository or
user environment.

Coverage includes canonicalization rejection cases, all location tiers and
legacy mappings, conflict warnings, no-directory-creation diagnostics, stable
CLI exits and JSON errors, deterministic identity, interrupted publication,
stale staging, immutable collision handling, corruption and symlink detection,
shallow validation after upstream deletion, deep validation failure, exact and
wildcard reuse, wildcard ambiguity/fail-on-miss, bounded multiprocess success
and failure, completion-probe reuse, small worker results, journal sequence
coordination, malformed journals, progress, and version metadata.

## Representative CLI output

```text
$ python -m sanos --root /tmp/sanos/phase-2-demo --dry-run \
    --log-format json config show --require-complete
{"dry_run":true,"event":"config","filesystem_modified":false,
 "locations":{"datasets":{"exists":false,
 "path":"/tmp/sanos/phase-2-demo/datasets","source":"cli:root"},
 "experiments":{"exists":false,
 "path":"/tmp/sanos/phase-2-demo/experiments","source":"cli:root"},
 "intermediate":{"exists":false,
 "path":"/tmp/sanos/phase-2-demo/intermediate","source":"cli:root"},
 "raw":{"exists":false,"path":"/tmp/sanos/phase-2-demo/raw",
 "source":"cli:root"}},"missing_locations":[],
 "repository_config":null,"user_config":null,"verbosity":0,"warnings":[]}

$ python -m sanos --log-format json \
    --repository-config /tmp/sanos/does-not-exist.toml config show
{"error_type":"ConfigurationCliError","event":"error","exit_code":3,
 "message":"Configuration file does not exist: \
 /tmp/sanos/does-not-exist.toml"}
(exit 3)
```

The first command created no directory. Both commands were offline.

## Known limits and review questions

- The local runner is deliberately small and does not introduce Redun. Review
  whether its artifact-reference and journal boundary is sufficient for the
  planned 891-day workload.
- Published Dataset distribution metadata and public leakage checks belong to
  the later dataset phase; this generic store does not claim those artifacts
  are publishable yet.
- The global `--dry-run`, logging, and location options are present now;
  material-command-specific plans will be added with those commands.
- File locks coordinate SANOS writers on filesystems with normal advisory-lock
  behavior. The later WRDS account semaphore is a separate host-wide mechanism
  and is not implemented here.
- No compatibility decision or quantitative semantic change is proposed by
  this phase. `overnight_variance_fraction` remains a later WIDB clock input
  with its required parity default of `0.2`.

## First-review remediation

The signed first-round reviews are `code-review.md` and `quant-review.md`, both
with verdict `CHANGES_REQUESTED`. Their findings were addressed as follows:

- Dataset distribution is now a typed identity input. Dataset creation rejects
  missing distribution; `public` requires license, redistribution basis,
  approver, approval date, and durable approval reference. Status/approval are
  stored in the manifest and affect full and wildcard compatibility identity.
- Storage lifecycle edges are enforced. Tests cover valid Raw → Intermediate →
  internal Dataset → Experiment, internal Dataset → public candidate Dataset,
  and rejected backward/source-free edges.
- Raw now requires `source_generation`; identical acquisition retries retain
  identity and an explicit refreshed generation changes identity.
- Atomic destinations, namespace/staging/payload parents, final paths, and lock
  paths reject symlinks and root escapes. Dangling destinations are collisions.
  Lock release is unconditional across setup, commit, body, and cleanup
  failures. Nested directories are fsynced bottom-up.
- Daily plans now hash ordered dates, ticker, stage, canonical payload, schema,
  and concrete producer version. Journals reject a changed plan, record plan
  identity and immutable returned/reused pointers, recover only an incomplete
  final record, do not consume sequence numbers on canonicalization failure,
  and journal probe/active-pool interrupts before bounded cancellation.
- Worker IPC is restricted to a bounded `ArtifactPointer` or `None`; worker
  counts must be real positive integers.
- Root and nested Click parsing/usage errors honor JSON mode with exit code 2.
  Placeholder phase guidance now matches `PROGRAM_PLAN.md` (Raw 4,
  Intermediate 5–6, Dataset 7, Experiment 9).
- The resolved `@version` graph explicitly contains copy, checksum, JSON
  validation, path-safety, and directory-durability helpers.

Remediation verification on 2026-10-12:

```text
ruff check src tests
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests
Success: no issues found in 27 source files

python -m compileall -q src tests
(exit 0)

pytest -q
70 passed in 5.19s
```

The added adversarial tests cover distribution/public approval, lifecycle
inversion, Raw refresh identity, atomic destination/parent/namespace/staging/
lock symlinks, dangling finals, setup/commit/cleanup lock release, nested fsync,
plan drift, bounded pointer IPC, canonicalization failure, injected partial
journal append and recovery, corruption before the final record, probe and
active-worker interrupts followed by restart, non-integral worker counts, and
root/nested JSON usage errors.

## Second-review remediation

The code re-review retained three findings after the first remediation. They
are addressed as follows:

- Before final publication, the builder now resolves and shallow-validates
  every declared upstream pointer. A public candidate or public Dataset has
  exactly one direct upstream in total: one Dataset projection source whose
  validated manifest declares `internal`. Deep lineage validation enforces the
  same projection boundary. Tests cover a fictitious upstream, valid
  internal-to-public publication, rejected public-on-public and
  candidate-on-public publication, and rejected Raw/Intermediate mixed-source
  lineages for both non-internal statuses.
- An interrupted daily runner now cancels queued futures, allows one second for
  active workers to finish, then terminates and, if required, kills only its
  own process-pool children. It waits for shutdown before propagating the
  interruption. A start-handshake test proves that a worker cannot create its
  delayed side-effect marker after the runner returns, then restarts the same
  plan immediately without a fabricated completion probe.
- Journal parent creation durably links each new directory, and the first
  successful journal append fsyncs the journal's containing directory. A
  nested-path helper-call test covers the first-create path.

Second-remediation verification on 2026-10-12:

```text
ruff check --no-cache src tests
All checks passed!

PYTHONPATH=/tmp/sanos/phase-2-tools python -m mypy src tests scripts
Success: no issues found in 28 source files

python -m compileall -q src tests scripts
(exit 0)

pytest -q
72 passed in 8.78s

git --git-dir=.git-real --work-tree=. diff --check
(exit 0)
```

All verification remained offline with `WRDS_OFF=1`; no WRDS connection or
reference-repository mutation occurred.
