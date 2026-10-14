# Human operator guide

This is the non-AI route through SANOS. Every operation uses the installed
`sanos` command (equivalently `python -m sanos`), prints actionable errors, and
supports `--log-format json`. The focused runbooks linked below remain binding
for the guarded reference build.

## Install and inspect

Python 3.12 is required. The aggregate operator extra includes WRDS access,
HDF5 Experiment support, and the Torch and scikit-learn runtimes used by the
pinned legacy Experiment consumer comparison:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[operator]'
sanos --help
sanos config resources
```

Use `.[dev,operator]` in a source checkout that will run tests. The narrower
`.[wrds]`, `.[experiment]`, and `.[reference]` extras remain available when the
complete workflow is not needed. The `reference` extra includes HDF5, Torch,
and scikit-learn because the exact 891-day migration comparison executes the
pinned legacy consumer in an isolated CPU-only worker.

Configure all four lifecycle roots with one general root:

```bash
export QUANT_DATA_ROOT=/path/to/writable/sanos/local
sanos --log-format json config show --require-complete
sanos doctor --strict-locations
```

The path is an operator choice, not a required system mount. Choose durable
storage with enough capacity, keep all SANOS-owned files beneath a dedicated
`sanos/` directory, and verify that the parent is writable before a live Raw
download. The guarded reference runbooks use a separate shell-only
`SANOS_RUNS_ROOT` variable so their data and evidence directories share that
explicitly selected parent without baking a machine path into configuration.

Class-specific `QUANT_DATA_RAW`, `QUANT_DATA_INTERMEDIATE`,
`QUANT_DATA_DATASETS`, and `QUANT_DATA_EXPERIMENTS` values override that root.
See [configuration.md](configuration.md) for local TOML and legacy aliases.
Credentials stay in the standard WRDS credential mechanism and never in SANOS
configuration.

## Ordered workflow

1. Run `sanos --dry-run --log-format json raw download ...`. Review the exact
   vendor tables, sessions, cache misses, and boundary dependencies.
2. Check `gpuctl resources --json`, then run the same Raw command. No manual
   confirmation flag is required. The default is two connections and the hard
   maximum is four. Connection opens are serialized by an account-scoped
   monitor; a fourth failed or interrupted attempt within 15 minutes is blocked
   before network access. Authentication and 2FA failures are never retried
   within a command, and transient queries have at most three total attempts.
3. Disable network access with `WRDS_OFF=1` and rerun Raw with `--cache-only`.
4. Build daily Intermediate frames with `sanos intermediate build`. Processing
   remains one Polars frame per day through `fitted-day`.
5. Publish with `sanos dataset build`, then run `sanos dataset validate
   --deep-lineage` before considering any Intermediate cleanup.
6. Build a separate public candidate with `sanos dataset public build` and
   validate it. Do not use the public approval commands without documented
   redistribution authority.
7. Build and validate the disposable HDF5 materialization with `sanos
   experiment build` and `sanos experiment validate`.
8. For the canonical 2022--2025 reference, follow the exact
   [Phase 10 pilot](phase-10-pilot.md), then the
   [Phase 11 production runbook](production-runbook.md). Those workflows add
   parity comparison, deterministic downstream config finalization, indexed
   evidence, dual review, and immutable canonical designation.

Every material command accepts `-v`; long work reports elapsed time,
completed/total units, failures, and ETA. JSON output is one object per line so
shells and monitoring systems need not scrape prose. An interrupted command is
restarted with the same root, source generation, config, and arguments; output
validation skips completed immutable work.

## Exit status contract

| Exit | Meaning | Operator action |
|---:|---|---|
| 0 | Completed or valid output-first reuse | Record the structured result and identity. |
| 2 | CLI syntax, missing option, or invalid Click value | Correct the command; no material operation began. |
| 3 | Location/configuration resolution failure | Run `config show --require-complete`; correct paths or JSON. |
| 4 | Safe material failure or cache miss | Read the final JSON `error`; repair the named prerequisite, then rerun exactly. |
| 7 | Deliberately unavailable phase surface | Consult status/docs; do not improvise around the gate. |
| 130 | Interrupt from the shell | Confirm child work stopped, then rerun the identical command. |

Unexpected Python/process failures are defects: preserve the complete log and
working directory rather than deleting either.

A successful structured command emits records such as:

```json
{"event":"dataset_validation_result","status":"passed","semantic":true,"deep_lineage":true,"network_accessed":false,"filesystem_modified":false,"identity":"<64-hex-dataset-id>"}
```

A safe material failure ends with one machine-readable record and exit 4:

```json
{"event":"error","error_type":"MaterialCliError","exit_code":4,"message":"Required Raw artifact is absent; run sanos raw download ..."}
```

Progress events may appear before the terminal record; consumers should read
JSON Lines until EOF and branch on `event` and the process exit status.

## Troubleshooting

| Symptom | Diagnosis and recovery |
|---|---|
| A root is unresolved | Set `QUANT_DATA_ROOT`, all four class variables, or a documented local TOML; confirm with `config show`. |
| Import for WRDS or HDF5 is missing | Install `.[operator]`, or the exact `.[wrds,experiment]` combination. |
| `WRDS_OFF` blocks a download | This is intentional. Keep it set for all non-Raw work; unset it only for a deliberate live Raw command. |
| Authentication or 2FA fails | SANOS fails closed without prompting and records the failed connection attempt. Do not loop the command or enter credentials into a logged command. Close competing clients and update `.pgpass` from a private terminal if needed. The account-scoped monitor reports the remaining cooldown and blocks a fourth failed or interrupted attempt within 15 minutes. |
| WRDS attempt monitor is busy | Another SANOS process is opening a connection. Let that attempt finish; do not start parallel login attempts. |
| WRDS lease is unavailable | Inspect the reported account-scoped lease and running clients. Never delete a live lease or raise the aggregate limit above four. |
| Artifact checksum or manifest fails | Preserve the artifact and log. Immutable outputs are not repaired in place; rebuild only under a new valid identity after diagnosis. |
| A no-arbitrage/SANOS solver day fails | Preserve its daily journal and inputs. It is a blocking data/numerical discrepancy, not permission to drop the day. |
| `--version '*'` misses | Supply the concrete current version and rebuild through the ordinary command. Wildcard mode is reuse-only. |
| A restart workspace exists | Rerun the identical command. SANOS validates completed units and owns cleanup only for its identity-specific staging area. |
| Shallow Dataset/Experiment validation passes after lineage was removed | Expected: published boundaries are self-contained. Use `--deep-lineage` only while the upstream lineage intentionally remains. |
| `gpuctl resources` is unavailable or uses local fallback | Treat it as WARN and do not begin material production work until daemon-backed checks are all `OK`. |

## Retention and cleanup

Raw is durable vendor provenance. Dataset is curated, validated research input.
Keep both. Intermediate is reproducible per-stage cache and may be removed only
after deep Dataset validation and publication evidence are complete. Experiment
is reproducible project materialization and may be removed after its validation
and any required comparison/promotion evidence are complete.

SANOS does not provide a blanket recursive-delete command. First record exact
artifact paths from structured build/validation results, confirm no process has
them open, and move only those specific Intermediate or Experiment artifacts to
a SANOS-owned quarantine below the same lifecycle root. Preserve the run
journal and evidence index. Never remove a class root, use a wildcard target,
delete Raw as routine cache cleanup, or delete any Dataset in place. A later
space-reclamation operation should enumerate and review quarantined identities
explicitly.

Public release is independent of cleanup and technical validation. A
`public_candidate` becomes `public` only through the rights-gated approval
contract with a license, redistribution basis, approver, date, and durable
reference.
